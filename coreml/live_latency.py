#!/usr/bin/env python3
"""Offline rerun of the live loop (coreml/livecore.py) on dev-clean + dev-other with the Core ML model
(default: the released package, V1CoreML("release", "fp16", "CPU_AND_NE")), modes A and C; word
latency, revisions, WER, mode C corrections.

  ~/.venv-coreml/bin/python coreml/live_latency.py [--limit N] [--shard I --nshards N]   # simulate
  ~/.venv-coreml/bin/python coreml/live_latency.py --calibrate [--quick]                 # idle machine
  ~/.venv-coreml/bin/python coreml/live_latency.py --report                             # tables
  --variant faithful|fp16safe|release (default release): outputs live/sim_<variant>/,
  live/calibration_<variant>.json, live/summary_<variant>.json, LIVE_LATENCY.md (release) or
  LIVE_LATENCY_<variant>.md.

Simulation. Every utterance of the FULL list (common.load_list('full')) is fed in 0.4 s chunks to a
mode A and a mode C LiveEngine; both see the same audio, so the segmentation is the same and one Core
ML encode per step serves both (the engines' own logic decides what to encode and decode). After the
last full chunk the remaining audio is one more step (engine.flush(), at the next step boundary g);
if the segment is still open it is closed (engine.finish(), reusing that encode: the stream is assumed
to continue in silence, so a live engine would close it at the first step boundary >= max(g, end of
the last speech frame + 0.6 s), which is the time given to that step). Per step the simulation keeps
the audio time, what was computed (window, samples, decoder, frames) and the display after the step
(committed words + the current segment's words, the in-progress word included).

Latency of a word of the final transcript = (time its final text is displayed) - (the word's end):
  display time of step k: d_k = max(t_k, d_(k-1)) + c_k, with t_k the audio time of step k (a multiple
    of 0.4 s) and c_k its compute (calibrated, below). This includes the chunk wait (up to 400 ms),
    the model's 160 ms lookahead (the word's last output frame needs 4 more frames of audio), compute
    and any backlog when compute > 400 ms.
  final display of word i: the first step from which the displayed prefix up to and including word i
    never changes again (decode/stream_study.py's rule).
  word end: exclusive end frame of the word's last token in a Viterbi CTC alignment
    (decode/streaming.ctc_align_ends) of the final text, tokenized with V1's SentencePiece model, on
    the utterance's saved full-utterance V1 log-probs (decode/logprobs/<split>.npy), x 40 ms. This is
    the end of the 40 ms slot of the last frame CTC assigns to the word: a model-side end, the same for
    both modes and independent of segmentation. CTC emits a token's spike near the end of its sound,
    with blanks after it, so it is close to the acoustic end [INFERENCE: not checked against forced
    alignments]; the 160 ms lookahead comes on top of it.
  c_k from --calibrate (idle machine): per step VAD time + for each encode mel(samples) + model(window)
    (not for an encode reused within the engine) + decoder(frames) per decode, interpolated in the
    segment length. The simulation's own wall times run under load and are not used.
Revisions (stream_study 'all'): changed = shown words that later change (a word past the common
prefix of consecutive displays); revised = changed minus the in-progress word growing ('tw' ->
'twelve'). Mode C corrections: per segment end, word edit distance live text -> beam+LM text.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from common import (LOGPROBS, OUT, SR, V1_WER, WINDOWS, check_v1_files, load_audio, load_list,
                    norm_text, sp)
from livecore import FRAME, EnergyVAD, LiveEngine, build_decoders, encode
from transcribe import Transcriber
from streaming import ctc_align_ends
from wer import corpus_wer

LIVE = OUT / "live"
PRECISION, UNITS = "fp16", "CPU_AND_NE"


def sim_dir(variant):
    return LIVE / f"sim_{variant}"


def cal_path(variant, quick=False):
    return LIVE / f"calibration_{variant}{'_quick' if quick else ''}.json"


def report_path(variant):
    return OUT / ("LIVE_LATENCY.md" if variant == "release" else f"LIVE_LATENCY_{variant}.md")


SPLITS = ("dev-clean", "dev-other")
MODES = ("A", "C")
STEP_S, PAUSE_S = 0.4, 0.6
FRAME_S = 0.04
V1_HYPS = {s: Path.home() / f"gemma_stt/decode/results/{p}_{s}_cbd_sp6_beam32_units-piece/hyps.jsonl"
           for s, p in (("dev-clean", "sweep"), ("dev-other", "apply"))}


# ---------------------------------------------------------------- stream_study rules (copied: that
# module imports pyctcdecode, which this venv does not have)
def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def word_ends(ids, tok_ends, pieces):
    """stream_study.word_ends: words grouped at pieces starting with U+2581, end frame of each."""
    groups = []
    for i, e in zip(ids, tok_ends):
        if not groups or pieces[i].startswith("\u2581"):
            groups.append([[i], e])
        else:
            groups[-1][0].append(i)
            groups[-1][1] = e
    words, ends = [], []
    for g, e in groups:
        w = sp().decode(g).lower().strip()
        if w:
            words.append(w)
            ends.append(e)
    return (words, ends) if words == sp().decode(ids).lower().split() else None


def revisions(displays, final):
    """(shown, changed, revised) as stream_study.policy_stats."""
    seq = displays + [final]
    shown = sum(len(seq[k]) - lcp(seq[k - 1] if k else [], seq[k]) for k in range(len(seq)))
    changed = revised = 0
    for k in range(len(displays)):
        a, b = seq[k], seq[k + 1]
        n = lcp(a, b)
        changed += len(a) - n
        grew = n == len(a) - 1 and n < len(b) and b[n] != a[n] and b[n].startswith(a[n])
        revised += 0 if grew else len(a) - n
    return shown, changed, revised


def settle_steps(displays, final):
    """Per final word: first step index from which the display's prefix through that word never
    changes again (len(displays) - 1 at the latest: the last display is the final transcript)."""
    last_bad = [-1] * len(final)
    for k, d in enumerate(displays):
        for i in range(lcp(d, final), len(final)):
            last_bad[i] = k
    return [b + 1 for b in last_bad]


# ---------------------------------------------------------------- simulation
class SharedEncoder:
    """encode() memoized per (segment start, end) so the A and C engines share one Core ML call."""

    def __init__(self, cml):
        self.cml, self.memo = cml, {}

    def clear(self):
        self.memo.clear()

    def __call__(self, wav, key):
        if key not in self.memo:
            if len(self.memo) > 8:
                self.memo.clear()
            self.memo[key] = encode(self.cml, wav)
        return self.memo[key]


def full_logprobs(split):
    idx = json.loads((LOGPROBS / f"{split}.json").read_text())
    arr = np.load(LOGPROBS / f"{split}.npy", mmap_mode="r")
    return {u["id"]: (u["offset"], u["frames"]) for u in idx["utts"]}, arr


def simulate_utt(u, engines, enc, lp_full, pieces, whole):
    wav = load_audio(u["audio"])
    enc.clear()
    step = engines["A"].step
    out = {}
    for m, e in engines.items():
        e.reset()
        out[m] = {"steps": [], "disp": [], "corr": [], "segs": []}

    def record(m, t_samples, events):
        e, r = engines[m], out[m]
        items, meas = [], 0.0
        for ev in events:
            d = ev["detail"]
            meas += ev["compute_ms"]
            if "decoder" in d:
                items.append([d["window"], d["samples"], d["decoder"], d["frames"], int(d["reused"])])
            if ev["kind"] == "segment_end":
                r["corr"].append([ev["changed_words"], len(ev["replaced"]), len(ev["final"])])
                r["segs"].append([round(d["start_s"], 3), round(d["end_s"], 3)])
                if ev["changed_words"]:
                    r.setdefault("examples", []).append([" ".join(ev["replaced"]), " ".join(ev["final"])])
        r["steps"].append([round(t_samples / SR, 4), items, round(meas, 2)])
        r["disp"].append(e.committed + e.shown)

    n_full = len(wav) // step
    for j in range(n_full):
        chunk = wav[j * step:(j + 1) * step]
        for m, e in engines.items():
            record(m, e.done + step, e.feed(chunk))
    g = -(-len(wav) // step) * step                       # next step boundary at or after the end
    rest = wav[n_full * step:]
    for m, e in engines.items():
        if len(rest):
            assert not e.feed(rest)                       # < one step: no step yet
            record(m, g, e.flush())
        if e.active:
            ls = e.last_speech if e.last_speech is not None else e.done
            t_close = max(g, -(-(ls + e.pause) // step) * step)
            record(m, t_close, e.finish())

    finals = {m: list(engines[m].committed) for m in engines}
    res = {"id": u["id"], "split": u["split"], "dur": u["dur"], "ref": u["ref"], "word_end": None,
           "whole": whole.transcribe(wav)[0]}
    fa = finals["A"]
    if fa:
        ids = sp().encode(" ".join(fa))
        ends = ctc_align_ends(lp_full, ids, 256)
        we = None if ends is None else word_ends(ids, ends, pieces)
        if we is not None and we[0] == fa:
            res["word_end"] = [round(e * FRAME_S, 3) for e in we[1]]
    else:
        res["word_end"] = []
    for m in engines:
        r, final = out[m], finals[m]
        sh, ch, rv = revisions(r["disp"], final)
        res[m] = {"hyp": " ".join(final), "steps": r["steps"], "settle": settle_steps(r["disp"], final),
                  "shown": sh, "changed": ch, "revised": rv, "corr": r["corr"], "segs": r["segs"],
                  "examples": r.get("examples", [])[:5]}
    return res


def run_sim(args):
    from cmlrun import V1CoreML
    check_v1_files()
    sim = sim_dir(args.variant)
    sim.mkdir(parents=True, exist_ok=True)
    utts = [u for u in load_list("full") if u["split"] in args.splits]
    utts = [u for j, u in enumerate(utts) if j % args.nshards == args.shard]
    if args.limit:
        utts = utts[:args.limit]
    path = sim / (f"sim.{args.shard}of{args.nshards}" + (f"_n{args.limit}" if args.limit else "") + ".jsonl")
    done = {r["id"] for r in read_jsonl(path)} if path.exists() else set()
    if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("a") as f:                         # a crashed run cut its last line
            f.write("\n")
    todo = [u for u in utts if u["id"] not in done]
    print(f"{path}: {len(done)} done, {len(todo)} to go", flush=True)
    t0 = time.time()
    cml = V1CoreML(args.variant, PRECISION, UNITS)
    print(f"model load {time.time() - t0:.1f} s", flush=True)
    enc = SharedEncoder(cml)
    whole = Transcriber(cml=cml)                          # whole-utterance reference, same model
    dec = build_decoders("C")
    engines = {"A": LiveEngine("A", decoders=dec, encoder=enc), "C": LiveEngine("C", decoders=dec, encoder=enc)}
    pieces = [sp().id_to_piece(i) for i in range(sp().get_piece_size())]
    lps = {s: full_logprobs(s) for s in args.splits}
    t0, audio = time.time(), 0.0
    with path.open("a") as f:
        for i, u in enumerate(todo, 1):
            index, arr = lps[u["split"]]
            off, n = index[u["id"]]
            res = simulate_utt(u, engines, enc, np.asarray(arr[off:off + n], np.float32), pieces, whole)
            f.write(json.dumps(res) + "\n")
            f.flush()
            audio += u["dur"]
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"{i}/{len(todo)} utts, {audio / 3600:.2f} h audio, {el / 60:.1f} min, "
                      f"x{audio / max(el, 1e-9):.2f} real time", flush=True)


# ---------------------------------------------------------------- calibration (idle machine only)
def timed(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def calibrate(args):
    from cmlrun import V1CoreML
    check_v1_files()
    LIVE.mkdir(parents=True, exist_ok=True)
    q = args.quick
    reps = 3 if q else 30
    fast = load_list("fast")
    wav = np.concatenate([load_audio(u["audio"]) for u in fast[:40]])[:36 * SR]
    t0 = time.time()
    cml = V1CoreML(args.variant, PRECISION, UNITS)
    cal = {"machine_note": "run on an idle machine (SLOT)", "quick": bool(q), "load_s": time.time() - t0,
           "compute_units": UNITS, "variant": f"{args.variant}/{PRECISION}", "date": time.strftime("%Y-%m-%d %H:%M")}
    from common import mel
    m_all, nv = mel(wav[:35 * SR])
    cal["model_ms"] = {}
    for w, tw in WINDOWS.items():
        m = m_all[:tw]
        for _ in range(2 if q else 5):
            cml.predict(m, tw, w)
        ts = timed(lambda: cml.predict(m, tw, w), reps)
        cal["model_ms"][w] = {"median": float(np.median(ts)), "p90": float(np.percentile(ts, 90)),
                              "mean": float(np.mean(ts)), "n": reps}
        print(w, cal["model_ms"][w], flush=True)
    secs = [0.4, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30, 35]
    mel(wav[:SR])
    cal["mel_ms"] = {"samples": [int(s * SR) for s in secs],
                     "median": [float(np.median(timed(lambda: mel(wav[:int(s * SR)]), 3 if q else 7))) for s in secs]}
    print("mel", cal["mel_ms"], flush=True)
    idx = json.loads((LOGPROBS / "dev-clean.json").read_text())
    arr = np.load(LOGPROBS / "dev-clean.npy", mmap_mode="r")
    lp_all = np.asarray(arr[:20000], np.float32)
    dec = build_decoders("C")
    frames = [10, 25, 50, 100, 150, 200, 300, 400, 500, 625, 750, 875]
    offs = np.linspace(0, len(lp_all) - 900, 3 if q else 9).astype(int)
    cal["decode_ms"] = {"frames": frames}
    for name, d in dec.items():
        d.decode(lp_all[:100])
        meds = []
        for F in frames:
            ts = [timed(lambda: d.decode(lp_all[o:o + F]), 1)[0] for o in offs]
            meds.append(float(np.median(ts)))
        cal["decode_ms"][name] = meds
        print(name, meds, flush=True)
    vad = EnergyVAD()
    stepw = wav[:int(STEP_S * SR) * 100]
    ts = []
    for k in range(100):
        ch = stepw[k * int(STEP_S * SR):(k + 1) * int(STEP_S * SR)]
        t0 = time.perf_counter()
        for a in range(0, len(ch), FRAME):
            vad.step(ch[a:a + FRAME])
        ts.append((time.perf_counter() - t0) * 1e3)
    cal["vad_ms_per_step"] = float(np.median(ts))
    # end-to-end check: the real engine on a few utterances, measured vs predicted per step
    cost = Cost(cal)
    ratios, meas_all, pred_all = [], [], []
    for mode in MODES:
        eng = LiveEngine(mode, cml=cml, decoders=dec)
        for u in fast[:3 if q else 15]:
            w = load_audio(u["audio"])
            eng.reset()
            evs = []
            for j in range(0, len(w), eng.step):
                evs += [eng.feed(w[j:j + eng.step])]
            evs += [eng.finish()]
            for step_evs in evs:
                if not step_evs:
                    continue
                items = [[e["detail"].get("window"), e["detail"].get("samples"), e["detail"]["decoder"],
                          e["detail"]["frames"], int(e["detail"]["reused"])] for e in step_evs if "decoder" in e["detail"]]
                if not items:
                    continue
                mm = sum(e["compute_ms"] for e in step_evs)
                pp = cost(items)
                meas_all.append(mm)
                pred_all.append(pp)
                ratios.append(mm / pp)
    cal["check"] = {"steps": len(ratios), "measured_ms_median": float(np.median(meas_all)),
                    "predicted_ms_median": float(np.median(pred_all)),
                    "ratio_median": float(np.median(ratios)), "ratio_p10": float(np.percentile(ratios, 10)),
                    "ratio_p90": float(np.percentile(ratios, 90))}
    print("check", cal["check"], flush=True)
    path = cal_path(args.variant, q)
    path.write_text(json.dumps(cal, indent=1))
    print(f"wrote {path}", flush=True)


class Cost:
    """Calibrated compute (ms) of one step from its work items."""

    def __init__(self, cal):
        self.model = {w: v["median"] for w, v in cal["model_ms"].items()}
        self.mel_x, self.mel_y = cal["mel_ms"]["samples"], cal["mel_ms"]["median"]
        self.dec_x = cal["decode_ms"]["frames"]
        self.dec = {k: v for k, v in cal["decode_ms"].items() if k != "frames"}
        self.vad = cal["vad_ms_per_step"]

    def __call__(self, items):
        c = self.vad
        for window, samples, decoder, frames, reused in items:
            if window is not None and not reused:
                c += self.model[window] + float(np.interp(samples, self.mel_x, self.mel_y))
            if frames:
                c += float(np.interp(frames, self.dec_x, self.dec[decoder]))
        return c


# ---------------------------------------------------------------- report
def latencies(rec, mode, cost):
    """Per final word latency (s) under a step-cost function (ms)."""
    r = rec[mode]
    if rec["word_end"] is None or len(rec["word_end"]) != len(r["settle"]):
        return None
    d, prev = [], 0.0
    for t, items, _ in r["steps"]:
        prev = max(t, prev) + cost(items) / 1e3
        d.append(prev)
    return [d[k] - e for k, e in zip(r["settle"], rec["word_end"])]


def vad_last_speech(recs):
    """{id: end (s) of the last speech frame of the whole utterance} by livecore's EnergyVAD (the same
    frame rule as the engine): an energy-based acoustic end of the last word. Cached in live/."""
    path = LIVE / "vad_last_speech.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    paths = {u["id"]: u["audio"] for u in load_list("full")}
    todo = [i for i in recs if i not in cache]
    for i in todo:
        w, vad, last = load_audio(paths[i]), EnergyVAD(), None
        for a in range(0, len(w) - FRAME + 1, FRAME):
            if vad.step(w[a:a + FRAME])[0]:
                last = (a + FRAME) / SR
        cache[i] = last
    if todo:
        path.write_text(json.dumps(cache))
    return cache


def pct(x):
    x = np.asarray(x) * 1e3
    return {"n": int(len(x)), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90))} if len(x) else None


def stats(x):
    x = np.asarray(x) * 1e3
    return {"n": int(len(x)), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "mean": float(x.mean()), "negative": int((x < 0).sum()),
            "share_below_160ms": float((x < 160).mean())} if len(x) else None


def read_jsonl(path):
    """Records of a simulation file; a line cut by a crashed run is skipped (its utterance is redone
    by the next run of the same shard)."""
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def report(args):
    recs = {}
    for p in sorted(sim_dir(args.variant).glob("sim.*.jsonl")):
        if "_n" in p.name and not args.include_limited:
            continue
        for r in read_jsonl(p):
            recs[r["id"]] = r
    cpath = Path(args.calibration) if args.calibration else cal_path(args.variant)
    cal = json.loads(cpath.read_text()) if cpath.exists() else None
    cost = Cost(cal) if cal else None
    zero = lambda items: 0.0  # noqa: E731
    v1 = {s: {r["id"]: r["hyp"] for r in map(json.loads, V1_HYPS[s].read_text().splitlines())} for s in SPLITS}
    full = {s: [u["id"] for u in load_list("full") if u["split"] == s] for s in SPLITS}
    vad = vad_last_speech(recs)
    summary = {"variant": args.variant, "precision": PRECISION, "compute_units": UNITS, "calibration": cal,
               "calibration_file": str(cpath), "splits": {}}
    for s in SPLITS:
        rs = [recs[i] for i in full[s] if i in recs]
        if not rs:
            continue
        S = {"utts": len(rs), "of": len(full[s]), "audio_h": sum(r["dur"] for r in rs) / 3600,
             "no_alignment": sum(r["word_end"] is None for r in rs)}
        refs = [r["ref"] for r in rs]
        v1h = [v1[s][r["id"]] for r in rs]
        S["v1_wer_subset"] = corpus_wer(refs, v1h)[0]
        S["v1_wer_split"] = V1_WER[s]
        cmh = [r["whole"] for r in rs]                    # Core ML whole utterance (transcribe.Transcriber)
        S["coreml_whole_wer_subset"] = corpus_wer(refs, cmh)[0]
        nseg = [len(r["A"]["segs"]) for r in rs]
        S["segments"] = int(sum(nseg))
        S["multi_segment_utts"] = int(sum(n > 1 for n in nseg))
        S["empty_segments"] = int(sum(c[2] == 0 and c[1] == 0 for r in rs for c in r["A"]["corr"]))
        multi = [i for i, n in enumerate(nseg) if n > 1]
        S["coreml_whole_wer_multi_segment"] = corpus_wer([refs[i] for i in multi], [cmh[i] for i in multi])[0] if multi else None
        last = [r for r in rs if r["word_end"] and vad.get(r["id"]) is not None]
        S["align_end_minus_vad_end_last_word"] = pct([r["word_end"][-1] - vad[r["id"]] for r in last])
        for m in MODES:
            hyps = [r[m]["hyp"] for r in rs]
            w, e, nw = corpus_wer(refs, hyps)
            M = {"wer": w, "edits": e, "words": nw, "identical_to_v1": sum(h == v for h, v in zip(hyps, v1h)),
                 "identical_to_coreml_whole": sum(h == v for h, v in zip(hyps, cmh)),
                 "wer_multi_segment": corpus_wer([refs[i] for i in multi], [hyps[i] for i in multi])[0] if multi else None,
                 "v1_wer_multi_segment": corpus_wer([refs[i] for i in multi], [v1h[i] for i in multi])[0] if multi else None,
                 "identical_to_coreml_whole_single_segment": sum(hyps[i] == cmh[i] for i in range(len(rs)) if nseg[i] == 1),
                 "single_segment_utts": sum(n == 1 for n in nseg)}
            sh = sum(r[m]["shown"] for r in rs)
            M.update(shown=sh, changed=sum(r[m]["changed"] for r in rs), revised=sum(r[m]["revised"] for r in rs))
            M["changed_share"], M["revised_share"] = M["changed"] / sh, M["revised"] / sh
            for name, fn in (("latency_zero_compute", zero), ("latency", cost)):
                if fn is None:
                    M[name] = None
                    continue
                lat = [x for r in rs for x in (latencies(r, m, fn) or [])]
                M[name] = stats(lat)
            if cost is not None:
                comp = [cost(it) for r in rs for _, it, _ in r[m]["steps"]]
                busy = [c for c in comp if c > cost.vad]
                M["compute_ms_per_step"] = {"median_active": float(np.median(busy)) if busy else 0.0,
                                            "p90_active": float(np.percentile(busy, 90)) if busy else 0.0,
                                            "max": float(max(comp)),
                                            "share_over_step": float(np.mean(np.asarray(comp) > STEP_S * 1e3))}
            corr = [c for r in rs for c in r[m]["corr"] if c[1] or c[2]]
            ch = np.asarray([c[0] for c in corr])
            M["corrections"] = {"segments": len(corr), "corrected": int((ch > 0).sum()),
                                "share_corrected": float((ch > 0).mean()) if len(ch) else 0.0,
                                "words_changed_mean": float(ch.mean()) if len(ch) else 0.0,
                                "words_changed_mean_when_corrected": float(ch[ch > 0].mean()) if (ch > 0).any() else 0.0,
                                "distribution": {str(k): int((ch == k).sum()) for k in range(5)} | {"5+": int((ch >= 5).sum())},
                                "replaced_words": int(sum(c[1] for c in corr)), "final_words": int(sum(c[2] for c in corr))}
            M["agree_A"] = sum(r[m]["hyp"] == r["A"]["hyp"] for r in rs)
            if cost is not None:
                lw = [(latencies(r, m, cost), r) for r in last]
                lw = [(x[-1], x[-1] + r["word_end"][-1] - vad[r["id"]]) for x, r in lw if x]
                M["last_word_latency"] = pct([a for a, _ in lw])
                M["last_word_latency_vs_vad_end"] = pct([b for _, b in lw])
            S[m] = M
            if m == "C":
                at_corr = tot = 0
                for r in rs:
                    for k in r["C"]["settle"]:
                        tot += 1
                        at_corr += any(it[2] == "beam" for it in r["C"]["steps"][k][1])
                M["words_settled_by_pause_correction"] = at_corr / max(tot, 1)
        S["examples_C"] = [x for r in rs for x in r["C"]["examples"]][:12]
        summary["splits"][s] = S
    LIVE.mkdir(parents=True, exist_ok=True)
    (LIVE / f"summary_{args.variant}.json").write_text(json.dumps(summary, indent=1))
    write_md(summary)
    print(report_path(args.variant).read_text())


def fmt(x, nd=0):
    return "pending" if x is None else f"{x:.{nd}f}"


def write_md(summary):
    cal = summary["calibration"]
    L = ["# Live transcription latency (Core ML V1, modes A and C)", "",
         f"Generated by `coreml/live_latency.py --report` on {time.strftime('%Y-%m-%d %H:%M')}. Engine: "
         f"`coreml/livecore.py`; model: V1CoreML(\"{summary['variant']}\", \"{summary['precision']}\", "
         f"\"{summary['compute_units']}\"); step 0.4 s, pause 0.6 s, "
         "max segment 35 s. Mode A = V1 beam+LM on the current segment every step; mode C = greedy + "
         "LM fusion (beam 1, alpha 0.5, beta 1.5) live, replaced by V1 beam+LM at each pause.", "",
         "## Latency definition", "",
         "Per word of the final transcript: latency = (time the word's final text is displayed) - (word end).",
         "",
         "- Step k runs at audio time t_k = 0.4 s x k (the stream's step boundaries) and re-encodes the current "
         "segment from its start to t_k. Its display time is d_k = max(t_k, d_(k-1)) + c_k, with c_k the step's "
         "compute; the chunk wait (up to 400 ms), compute and any backlog are included.",
         "- Lookahead: an output frame has its full 160 ms of right context only 4 frames (160 ms) after its own "
         "slot. Every step runs the model on the segment up to t_k, and the model still emits the last 4 frames, "
         "with the missing future masked (as at the end of any utterance). So a word can be displayed, and become "
         "final, before its 160 ms of future audio exist, when that edge prediction is already right. When it is "
         "wrong, the next re-encode (with the future) changes it, which counts as a revision and delays the "
         "final display. The lookahead therefore costs at most 160 ms and often nothing; the column "
         "'< 160 ms' gives the share of words displayed final sooner than 160 ms after their end.",
         "- c_k is calibrated on an idle machine (`--calibrate`): VAD per step + for each Core ML encode "
         "mel(samples) + model(window, median) + per decode decoder(frames) (beam = V1 ctc-beam-decoder, fusion "
         "= dec_subword beam 1), linearly interpolated in the segment length. The simulation's own wall times "
         "(machine under load) are not used.",
         "- Final display of word i = the first step from which the displayed prefix up to and including word i "
         "never changes again (decode/stream_study.py). Displayed = committed words of earlier segments + "
         "every word of the current segment, the in-progress word included.",
         "- Word end = exclusive end frame of the word's last token in a Viterbi CTC alignment "
         "(decode/streaming.py `ctc_align_ends`) of the final text (V1 SentencePiece tokens) on the "
         "utterance's saved full-utterance V1 log-probs, x 40 ms: the end of the 40 ms slot of the last frame "
         "CTC assigns to the word. It is a model-side end, identical for both modes and independent of the "
         "segmentation; CTC spikes sit near the end of a token's sound, so it approximates the acoustic end "
         "[INFERENCE: not checked against forced alignments].",
         "- End of an utterance: the leftover audio (< 0.4 s) is one more step at the next boundary; an open "
         "segment is then closed as if silence followed: at the first boundary >= the last speech frame + 0.6 s "
         "(the model input stays the real audio).",
         "- Zero-compute latency (c_k = 0) is shown as the algorithmic floor: chunk wait + edge effects + pause rule.",
         "- Revisions as stream_study 'all': 'changed' = shown words that are later changed (past the common "
         "prefix of consecutive displays); 'real revisions' = changed minus the in-progress last word only "
         "growing ('tw' -> 'twelve'). Mode C pause corrections count as revisions.",
         "", "## Segmentation rule (livecore.py)", "",
         "Energy VAD on the waveform, 10 ms frames: level = 10 log10(mean x^2) dBFS; adaptive noise floor (starts at "
         "the first frame, clamped at -90 dBFS; falls fast, floor += 0.2 (level - floor), rises 3 dB/s); speech "
         "frame = level > max(floor + 12 dB, -70 dBFS). A segment opens when 5 of the last 20 frames are speech, "
         "starting 0.5 s before the first of them (rounded down to the 40 ms output grid, never before the previous "
         "segment's end). It ends at the first 0.4 s step boundary with >= 0.6 s since the last speech frame (the "
         "silence stays in the segment), or is cut at the quietest 100 ms point of its last 5 s when it would "
         "exceed 35 s. Nothing is encoded while no segment is open.",
         "", "## Results", ""]
    hdr = ("| split | mode | utts | latency median ms | latency p90 ms | < 160 ms | zero-compute median / p90 ms | "
           "shown words later changed | real revisions | final WER % | Core ML whole-utterance WER % | "
           "V1 WER % (same utts) | identical to Core ML whole / V1 |")
    L += [hdr, "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, S in summary["splits"].items():
        for m in MODES:
            M = S[m]
            lat, z = M["latency"], M["latency_zero_compute"]
            L.append(f"| {s} | {m} | {S['utts']}/{S['of']} | {fmt(lat and lat['median'])} | {fmt(lat and lat['p90'])} | "
                     f"{'pending' if lat is None else f'{100 * lat['share_below_160ms']:.1f}%'} | "
                     f"{fmt(z['median'])} / {fmt(z['p90'])} | {100 * M['changed_share']:.2f}% ({M['changed']}/{M['shown']}) | "
                     f"{100 * M['revised_share']:.2f}% ({M['revised']}) | {100 * M['wer']:.3f} | "
                     f"{100 * S['coreml_whole_wer_subset']:.3f} | {100 * S['v1_wer_subset']:.3f} | "
                     f"{M['identical_to_coreml_whole']} / {M['identical_to_v1']} |")
    L += ["", "Latency words: " + "; ".join(
        f"{s} {S['A']['latency_zero_compute']['n']} final-transcript words timed per mode "
        f"({S['no_alignment']} utterances without a word alignment; words displayed final before their alignment "
        f"end: A {S['A']['latency_zero_compute']['negative']}, C {S['C']['latency_zero_compute']['negative']} "
        f"at zero compute)" for s, S in summary["splits"].items()) + ". "
        "Mode C words whose final display is the pause correction: " + "; ".join(
        f"{s} {100 * S['C']['words_settled_by_pause_correction']:.1f}%" for s, S in summary["splits"].items()) + ".",
        "", "V1 reference WER (whole utterances, PyTorch bf16, beam+LM): dev-clean 3.726 %, dev-other 10.469 %. "
        "Core ML whole-utterance = the same Core ML model through coreml/transcribe.py Transcriber (one encode "
        "per utterance, windows up to 35 s, longer audio cut at pauses), computed in the same simulation run.", "",
        "## Segmentation and WER", "",
        "| split | segments | utterances with > 1 segment | empty segments | live WER % on those | Core ML whole WER % "
        "on those | V1 WER % on those | single-segment utts identical to Core ML whole | mode A = mode C final |",
        "|---|---|---|---|---|---|---|---|---|"]
    for s, S in summary["splits"].items():
        A = S["A"]
        pc = lambda x: fmt(None if x is None else 100 * x, 3)  # noqa: E731
        L.append(f"| {s} | {S['segments']} | {S['multi_segment_utts']} | {S['empty_segments']} | "
                 f"{pc(A['wer_multi_segment'])} | {pc(S['coreml_whole_wer_multi_segment'])} | "
                 f"{pc(A['v1_wer_multi_segment'])} | {A['identical_to_coreml_whole_single_segment']}/"
                 f"{A['single_segment_utts']} | {S['C']['agree_A']}/{S['utts']} |")
    L += [""]
    for s, S in summary["splits"].items():
        A = S["A"]
        L.append(f"- {s}: live (segment-wise beam+LM) {100 * A['wer']:.3f} % vs Core ML whole-utterance "
                 f"{100 * S['coreml_whole_wer_subset']:.3f} % ({100 * (A['wer'] - S['coreml_whole_wer_subset']):+.3f} "
                 f"points: segmentation at pauses and leading-silence trimming) and V1 "
                 f"{100 * S['v1_wer_subset']:.3f} % ({100 * (A['wer'] - S['v1_wer_subset']):+.3f} points in total).")
    L += ["", "## Word end: CTC alignment vs energy end", "",
          "Check of the word-end choice on the last word of each utterance, whose acoustic end can be estimated "
          "from the energy: the end of the utterance's last VAD speech frame (livecore EnergyVAD over the whole "
          "file). Offset = alignment end - energy end (positive: the CTC end lies after the energy end, so the "
          "latency above is measured from a later point than the sound's end). Last-word latency is shown both "
          "ways (calibrated compute).", "",
          "| split | utts | alignment end - energy end, median (p10 / p90) ms | mode | last-word latency vs alignment "
          "end, median / p90 ms | vs energy end, median / p90 ms |", "|---|---|---|---|---|---|"]
    for s, S in summary["splits"].items():
        o = S["align_end_minus_vad_end_last_word"]
        for m in MODES:
            a, b = S[m].get("last_word_latency"), S[m].get("last_word_latency_vs_vad_end")
            L.append(f"| {s} | {o['n']} | {o['median']:.0f} ({o['p10']:.0f} / {o['p90']:.0f}) | {m} | "
                     f"{fmt(a and a['median'])} / {fmt(a and a['p90'])} | {fmt(b and b['median'])} / {fmt(b and b['p90'])} |")
    L += ["", "## Mode C pause corrections", "",
          "| split | segments | corrected | share | words changed per correction, mean (all / corrected only) | "
          "distribution 0/1/2/3/4/5+ |", "|---|---|---|---|---|---|"]
    for s, S in summary["splits"].items():
        c = S["C"]["corrections"]
        dist = "/".join(str(c["distribution"][k]) for k in ("0", "1", "2", "3", "4", "5+"))
        L.append(f"| {s} | {c['segments']} | {c['corrected']} | {100 * c['share_corrected']:.1f}% | "
                 f"{c['words_changed_mean']:.3f} / {c['words_changed_mean_when_corrected']:.2f} | {dist} |")
    L += ["", "Examples (live fusion text -> beam+LM text):", ""]
    for s, S in summary["splits"].items():
        for a, b in S["examples_C"][:6]:
            L.append(f"- {s}: `{a}` -> `{b}`")
    L += ["", "## Compute (calibrated)", ""]
    if cal is None:
        L.append("Pending: `live_latency.py --calibrate` needs an idle-machine slot (SLOT LiveEngine).")
    else:
        L.append(f"Calibration {cal['date']}{' (QUICK test run)' if cal.get('quick') else ''}: model ms per window "
                 + ", ".join(f"{w} {v['median']:.1f} (p90 {v['p90']:.1f})" for w, v in cal["model_ms"].items())
                 + f"; VAD {cal['vad_ms_per_step']:.2f} ms/step; check on the real engine: measured/predicted step "
                 f"compute median {cal['check']['ratio_median']:.2f} (p10 {cal['check']['ratio_p10']:.2f}, "
                 f"p90 {cal['check']['ratio_p90']:.2f}).")
        L += ["", "| split | mode | active step compute median / p90 ms | max ms | steps over 400 ms |",
              "|---|---|---|---|---|"]
        for s, S in summary["splits"].items():
            for m in MODES:
                c = S[m]["compute_ms_per_step"]
                L.append(f"| {s} | {m} | {c['median_active']:.0f} / {c['p90_active']:.0f} | {c['max']:.0f} | "
                         f"{100 * c['share_over_step']:.2f}% |")
    report_path(summary["variant"]).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--quick", action="store_true", help="with --calibrate: few repetitions (test only)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--include-limited", action="store_true", help="--report: also read --limit runs")
    ap.add_argument("--calibration", default="", help="--report: calibration JSON (default live/calibration_<variant>.json)")
    ap.add_argument("--variant", default="release", help="Core ML variant (default %(default)s)")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    if args.calibrate:
        calibrate(args)
    elif args.report:
        report(args)
    else:
        run_sim(args)


if __name__ == "__main__":
    sys.exit(main())
