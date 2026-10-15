#!/usr/bin/env python3
"""End-to-end speech-to-text latency of streaming V1 in mode A (beam+LM, every word shown).

  ~/.venv-coreml/bin/python coreml/stream_latency.py [--split dev-clean] [--sample N --seed 0] [--limit N]
      [--model step|standin] [--pause-s 0] [--timing]
  ~/.venv-coreml/bin/python coreml/stream_latency.py --combine-full FULL.utts.jsonl ... \
      --combine-timed TIMED.utts.jsonl ...

Each utterance is one stream: coreml/stream_runner.StreamingRecognizer (IncrementalMel -> step model
-> V1 ctc-beam-decoder re-decoding the open segment after every step) is fed the audio and returns
one event per 160 ms step plus the end-of-stream event. Timeline of event k:
  arrival a_k = t_audio: when the step's last input sample exists (step j: sample 2560 j + 2560,
               i.e. (2560 j + 2561) / 16000 s); the end-of-stream event arrives at N / 16000 s
  done_k      = max(a_k, done_{k-1}) + compute_k   (compute = measured mel + model + decode time of
               that event; the max carries any backlog from slow steps)
  display D_k = committed + live + pending words after event k, shown at done_k.
Per final word i: settle step k_i = the first event from which the display's prefix up to and
including word i never changes again; latency_i = done_{k_i} - end_i, where end_i is the word's end
in the audio: the exclusive end frame (x 40 ms) of its last token in a Viterbi CTC alignment
(decode/streaming.ctc_align_ends) of the final token ids on the OFFLINE log-probs
(~/gemma_stt/decode/logprobs/<split>.npy, V1's PyTorch log-probs). Step j returns frames [4j-4, 4j),
so the model's 160 ms lookahead is inside done_{k_i}; 'algorithmic' latency uses a_{k_i} (compute 0).
Revisions (stream_study.policy_stats definitions, display 'all'): changed = shown words later changed;
revised = changed minus changes where only the last shown word grew ('tw' -> 'twelve').

Compute is measured inside the run (time.perf_counter). Only a run with --timing on an idle machine
gives reportable numbers (it warms the model up first and is labelled calibrated); without --timing
the same numbers are written labelled 'uncalibrated (shared machine)'. Arrival times, settle events
and word ends do not depend on compute, so full splits run anywhere and --combine-* re-simulates
their timelines with the compute distribution of short --timing runs (see combine()).
Outputs: ~/gemma_stt/coreml/stream/latency_<split>_<model>_<seg>_<timing>[_s<N>seed<S>][_n<N>]
.{json,md,utts.jsonl} (utts.jsonl holds each utterance's timeline); latency_combined_<full>.json.
"""
import argparse
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LOGPROBS, OUT, SR, audio_path, check_v1_files, load_audio, norm_text, sp  # noqa: E402
from stream_runner import BLANK, FRAME_S, StreamingRecognizer, load_step_model  # noqa: E402
from streaming import ctc_align_ends  # noqa: E402
from wer import corpus_wer  # noqa: E402

V1_HYPS = {"dev-clean": Path.home() / "gemma_stt/decode/results/sweep_dev-clean_cbd_sp6_beam32_units-piece/hyps.jsonl",
           "dev-other": Path.home() / "gemma_stt/decode/results/apply_dev-other_cbd_sp6_beam32_units-piece/hyps.jsonl"}


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def word_ends(ids, tok_ends, pieces):
    """decode/stream_study.word_ends: words grouped at pieces starting with U+2581 and the end frame
    of each; None if the grouping disagrees with sp.decode."""
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


def timeline(events):
    """(arrival s, done s, display words) per event."""
    arr, done, disp, committed, t = [], [], [], [], 0.0
    for ev in events:
        t = max(ev["t_audio"], t) + ev["compute_ms"] / 1e3
        if ev["kind"] == "segment_end":
            committed = committed + ev["final"]
            d = list(committed)
        else:
            committed = ev["committed"]
            d = ev["committed"] + ev["live"] + ([ev["pending"]] if ev["pending"] is not None else [])
        arr.append(ev["t_audio"])
        done.append(t)
        disp.append(d)
    return arr, done, disp


def display_stats(disp, arr, done, ends):
    """(shown, changed, revised, latencies s, algorithmic latencies s, per word: settled only by the
    end-of-stream event)."""
    final = disp[-1]
    shown = sum(len(disp[k]) - lcp(disp[k - 1] if k else [], disp[k]) for k in range(len(disp)))
    changed = revised = 0
    for k in range(len(disp) - 1):
        a, b = disp[k], disp[k + 1]
        n = lcp(a, b)
        changed += len(a) - n
        grew = n == len(a) - 1 and n < len(b) and b[n] != a[n] and b[n].startswith(a[n])
        revised += 0 if grew else len(a) - n
    last_bad = [-1] * len(final)
    for k in range(len(disp)):
        n = lcp(disp[k], final)
        for i in range(n, len(final)):
            last_bad[i] = k
    lat, alg, at_end, ks = [], [], [], []
    if ends is not None:
        for i, e in enumerate(ends):
            k = last_bad[i] + 1
            ks.append(k)
            at_end.append(k == len(disp) - 1)
            lat.append(done[k] - e * FRAME_S)
            alg.append(arr[k] - e * FRAME_S)
    return shown, changed, revised, lat, alg, at_end, ks


def final_word_ends(events, lp_off, pieces):
    """End frame of every final word, from each closed segment's ids aligned on the offline log-probs;
    None if any segment does not align or group."""
    ends = []
    for ev in events:
        if ev["kind"] != "segment_end":
            continue
        d = ev["detail"]
        a, b, ids = d["start_frame"], d["end_frame"], d["ids"]
        if not ids:
            continue
        te = ctc_align_ends(lp_off[a:b], ids, BLANK)
        we = None if te is None else word_ends(ids, [a + e for e in te], pieces)
        if we is None or we[0] != ev["final"]:
            return None
        ends += we[1]
    return ends


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def machine_info():
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        info["chip"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                      text=True).stdout.strip()
        info["load_avg"] = list(np.round(np.array(__import__("os").getloadavg()), 2))
    except OSError:
        pass
    return info


def run(args):
    check_v1_files()
    idx = json.loads((LOGPROBS / f"{args.split}.json").read_text())
    arr = np.load(LOGPROBS / f"{args.split}.npy", mmap_mode="r")
    utts = idx["utts"]
    if args.sample:                          # seeded subset, the same for every run with this seed
        utts = sorted(random.Random(args.seed).sample(utts, min(args.sample, len(utts))), key=lambda u: u["id"])
    if args.limit:
        utts = utts[:args.limit]
    pieces = [sp().id_to_piece(i) for i in range(sp().get_piece_size())]
    v1 = {}
    if V1_HYPS.get(args.split, Path("/nonexistent")).exists():
        v1 = {r["id"]: r["hyp"] for r in map(json.loads, V1_HYPS[args.split].read_text().splitlines())}
    seg = (None, None) if not args.pause_s else (args.pause_s, args.max_segment_s)
    t_load = time.perf_counter()
    model = load_step_model(args.model, args.variant, args.precision, args.compute_units, mech=args.mech)
    load_s = time.perf_counter() - t_load
    rec = StreamingRecognizer(model, pause_s=seg[0], max_segment_s=seg[1])
    chunk = int(round(args.feed_ms * SR / 1000))
    if args.timing:                          # warm-up stream (device specialization, caches), discarded
        w = load_audio(audio_path(args.split, utts[0]["id"]))
        rec.reset()
        for a in range(0, len(w), chunk):
            rec.feed(w[a:a + chunk])
        rec.finish()

    tag = f"{args.split}_{args.model}_{'seg%g' % args.pause_s if args.pause_s else 'noseg'}_" \
          f"{'timed' if args.timing else 'untimed'}" + (f"_s{args.sample}seed{args.seed}" if args.sample else "") \
          + (f"_n{args.limit}" if args.limit else "")
    out = OUT / "stream"
    out.mkdir(parents=True, exist_ok=True)
    f_utts = open(out / f"latency_{tag}.utts.jsonl", "w")
    acc = {"shown": 0, "changed": 0, "revised": 0, "lat": [], "alg": [], "at_end": [], "no_align": 0,
           "frame_mismatch": 0, "same_v1": 0, "segments": 0}
    comp = {"model_ms": [], "decode_ms": [], "mel_ms": [], "step_ms": [], "end_ms": [], "backlog_ms": []}
    refs, hyps = [], []
    t0 = time.time()
    for n_done, u in enumerate(utts):
        w = load_audio(audio_path(args.split, u["id"]))
        rec.reset()
        events = []
        for a in range(0, len(w), chunk):
            events += rec.feed(w[a:a + chunk])
        events += rec.finish()
        lp_off = np.asarray(arr[u["offset"]:u["offset"] + u["frames"]], np.float32)
        if rec.nf != len(lp_off):
            acc["frame_mismatch"] += 1
        a_, d_, disp = timeline(events)
        ends = final_word_ends(events, lp_off, pieces)
        acc["no_align"] += ends is None
        sh, ch, rv, lat, alg, at_end, ks = display_stats(disp, a_, d_, ends)
        text = " ".join(disp[-1])
        acc["shown"] += sh
        acc["changed"] += ch
        acc["revised"] += rv
        acc["lat"] += lat
        acc["alg"] += alg
        acc["at_end"] += at_end
        acc["same_v1"] += text == v1.get(u["id"])
        acc["segments"] += rec.segment
        refs.append(norm_text(u["ref"]))
        hyps.append(text)
        for ev, a, dn in zip(events, a_, d_):
            dt = ev["detail"]
            comp["backlog_ms"].append((dn - a) * 1e3 - ev["compute_ms"])
            if ev["kind"] == "update":
                comp["model_ms"].append(dt["model_ms"])
                comp["decode_ms"].append(dt["decode_ms"])
                comp["mel_ms"].append(dt["mel_ms"])
                comp["step_ms"].append(ev["compute_ms"])
            if ev is events[-1]:
                comp["end_ms"].append(ev["compute_ms"])
        f_utts.write(json.dumps({"id": u["id"], "dur": u["dur"], "hyp": text, "v1": v1.get(u["id"]),
                                 "events": len(events), "segments": rec.segment, "shown": sh, "changed": ch,
                                 "revised": rv, "latency_ms": [round(x * 1e3, 1) for x in lat],
                                 # timeline for `combine`: per event, then per final word (settle event, end)
                                 "a": a_, "kind": [e["kind"] for e in events],
                                 "calls": [e["detail"]["calls"] for e in events],
                                 "frames": [e["detail"].get("frames", 0) for e in events],
                                 "model_ms": [round(e["detail"]["model_ms"], 3) for e in events],
                                 "decode_ms": [round(e["detail"].get("decode_ms", 0.0), 3) for e in events],
                                 "mel_ms": [round(e["detail"]["mel_ms"], 3) for e in events],
                                 "k": ks, "end": [] if ends is None else ends}) + "\n")
        if (n_done + 1) % 100 == 0:
            print(f"{n_done + 1}/{len(utts)} utts, {time.time() - t0:.0f} s, latency median so far "
                  f"{1e3 * np.median(acc['lat']):.0f} ms", flush=True)
    f_utts.close()
    wer, edits, words = corpus_wer(refs, hyps)
    lat = np.asarray(acc["lat"]) * 1e3
    alg = np.asarray(acc["alg"]) * 1e3
    res = {"split": args.split, "utts": len(utts), "model": args.model,
           "variant": getattr(model, "variant", None) or getattr(getattr(model, "cml", None), "variant", None),
           "precision": args.precision, "compute_units": args.compute_units, "mech": getattr(model, "mech", None),
           "segmentation": {"pause_s": seg[0], "max_segment_s": seg[1]},
           "timing": "calibrated (--timing, idle machine required)" if args.timing
           else "uncalibrated (shared machine; compute numbers not for reporting)",
           "load_s": load_s, "wer": wer, "edits": edits, "words": words,
           "identical_to_v1_offline_beam": acc["same_v1"], "v1_hyps_available": len(v1) > 0,
           "segments": acc["segments"], "frame_count_mismatch": acc["frame_mismatch"],
           "utts_without_alignment": acc["no_align"],
           "shown": acc["shown"], "changed": acc["changed"], "changed_share": acc["changed"] / max(acc["shown"], 1),
           "revised": acc["revised"], "revised_share": acc["revised"] / max(acc["shown"], 1),
           "latency_words": len(lat), "latency_ms_median": pct(lat, 50), "latency_ms_p90": pct(lat, 90),
           "latency_ms_mean": float(lat.mean()) if len(lat) else None,
           "algorithmic_latency_ms_median": pct(alg, 50), "algorithmic_latency_ms_p90": pct(alg, 90),
           "settled_only_at_end": int(sum(acc["at_end"])), "negative_latency_words": int((lat < 0).sum()),
           "latency_ms_median_settled_before_end": pct(lat[~np.asarray(acc["at_end"], bool)], 50),
           "latency_ms_p90_settled_before_end": pct(lat[~np.asarray(acc["at_end"], bool)], 90),
           "latency_ms_median_settled_at_end": pct(lat[np.asarray(acc["at_end"], bool)], 50),
           "compute": {k: {"n": len(v), "p50": pct(v, 50), "p90": pct(v, 90), "p99": pct(v, 99),
                           "max": float(np.max(v)) if v else None} for k, v in comp.items()},
           "steps_over_160ms": int((np.asarray(comp["step_ms"]) > 160).sum()),
           "seconds": round(time.time() - t0, 1), "machine": machine_info(),
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / f"latency_{tag}.json").write_text(json.dumps(res, indent=1))
    c = res["compute"]
    md = [f"# Streaming V1 mode A latency, {args.split} ({len(utts)} utts), model {args.model}, "
          f"{'segmented at %g s pauses' % seg[0] if seg[0] else 'one segment per utterance'}", "",
          f"Timing: {res['timing']}. {res['created']}, {res['machine'].get('chip', '')}.", "",
          "| WER % | identical to V1 offline beam | latency median ms | p90 ms | algorithmic median / p90 ms | "
          "shown words later changed | real revisions | settled only at end | step compute p50 / p90 / p99 ms | "
          "model p50 / p99 ms | decode p50 / p99 ms |",
          "|---|---|---|---|---|---|---|---|---|---|---|",
          f"| {100 * wer:.3f} | {acc['same_v1']}/{len(utts)} | {res['latency_ms_median']:.0f} | "
          f"{res['latency_ms_p90']:.0f} | {res['algorithmic_latency_ms_median']:.0f} / "
          f"{res['algorithmic_latency_ms_p90']:.0f} | {100 * res['changed_share']:.2f}% ({acc['changed']}/"
          f"{acc['shown']}) | {100 * res['revised_share']:.2f}% ({acc['revised']}) | "
          f"{res['settled_only_at_end']}/{len(lat)} | "
          f"{c['step_ms']['p50']:.1f} / {c['step_ms']['p90']:.1f} / {c['step_ms']['p99']:.1f} | "
          f"{c['model_ms']['p50']:.1f} / {c['model_ms']['p99']:.1f} | "
          f"{c['decode_ms']['p50']:.2f} / {c['decode_ms']['p99']:.2f} |"]
    (out / f"latency_{tag}.md").write_text("\n".join(md) + "\n")
    print(json.dumps(res))
    print("\n".join(md))
    return res


def _read_utts(paths):
    rows = []
    for p in paths:
        rows += [json.loads(line) for line in Path(p).read_text().splitlines()]
    return rows


def _simulate(rows, comp_fn):
    """Latencies (ms) of every aligned final word with event compute from comp_fn(row, event index)."""
    lat = []
    for r in rows:
        done, t = [], 0.0
        for i, a in enumerate(r["a"]):
            t = max(a, t) + comp_fn(r, i) / 1e3
            done.append(t)
        lat += [1e3 * (done[k] - e * FRAME_S) for k, e in zip(r["k"], r["end"])]
    return np.asarray(lat)


def combine(full_paths, timed_paths, seed=0):
    """Full-split timelines (any machine: arrival times, settle events and word ends do not depend on
    compute) + the compute distribution of --timing runs (idle machine): per event, model time = one
    draw per model call from the timed per-call times, mel = one draw from the timed per-step mel
    times, decode = timed least-squares fit a + b x frames plus a drawn timed residual (only for events
    that decoded). Draws are seeded. The same model applied to the timed runs' own timelines is
    compared with their measured latency ('model_check')."""
    full, timed = _read_utts(full_paths), _read_utts(timed_paths)
    model_pool = np.asarray([m for r in timed for m, c, kd in zip(r["model_ms"], r["calls"], r["kind"])
                             if c == 1 and kd == "update"])
    mel_pool = np.asarray([m for r in timed for m, kd in zip(r["mel_ms"], r["kind"]) if kd == "update"])
    dec = np.asarray([(f, d) for r in timed for f, d in zip(r["frames"], r["decode_ms"]) if d > 0])
    b, a = np.polyfit(dec[:, 0], dec[:, 1], 1)
    resid = dec[:, 1] - (a + b * dec[:, 0])
    rng = np.random.default_rng(seed)

    def comp(r, i):
        c = float(rng.choice(model_pool, r["calls"][i]).sum()) if r["calls"][i] else 0.0
        c += float(rng.choice(mel_pool))
        if r["decode_ms"][i] > 0:
            c += max(0.0, a + b * r["frames"][i] + float(rng.choice(resid)))
        return c

    lat = _simulate(full, comp)
    alg = _simulate(full, lambda r, i: 0.0)
    chk = _simulate(timed, comp)
    measured = np.asarray([x for r in timed for x in r["latency_ms"]])
    res = {"full": [str(p) for p in full_paths], "timed": [str(p) for p in timed_paths], "seed": seed,
           "utts": len(full), "latency_words": len(lat),
           "latency_ms_median": pct(lat, 50), "latency_ms_p90": pct(lat, 90), "latency_ms_mean": float(lat.mean()),
           "algorithmic_latency_ms_median": pct(alg, 50), "algorithmic_latency_ms_p90": pct(alg, 90),
           "compute_model": {"model_ms_per_call_p50": pct(model_pool, 50), "model_ms_per_call_p99": pct(model_pool, 99),
                             "mel_ms_p50": pct(mel_pool, 50), "decode_ms": f"{a:.4f} + {b:.5f} x frames",
                             "timed_steps": len(model_pool)},
           "model_check": {"timed_words": len(measured), "measured_median": pct(measured, 50),
                           "measured_p90": pct(measured, 90), "modelled_median": pct(chk, 50),
                           "modelled_p90": pct(chk, 90)},
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = OUT / "stream" / f"latency_combined_{Path(full_paths[0]).name.split('.')[0]}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev-clean", choices=("dev-clean", "dev-other"))
    ap.add_argument("--limit", type=int, default=0, help="first N utterances (of the seeded sample, if any)")
    ap.add_argument("--sample", type=int, default=0, help="seeded random subset of N utterances")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--combine-full", nargs="+", help="combine mode: untimed full-split *.utts.jsonl")
    ap.add_argument("--combine-timed", nargs="+", help="combine mode: --timing *.utts.jsonl")
    ap.add_argument("--model", choices=("step", "standin"), default="step")
    ap.add_argument("--mech", default=None, help="StepModel mechanism: state | io (default: stream_step's)")
    ap.add_argument("--variant", default=None, help="default: faithful (step), release (standin windows)")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--compute-units", default="CPU_AND_NE")
    ap.add_argument("--pause-s", type=float, default=0.0,
                    help="decoder segmentation at blank pauses (0 = one segment per utterance, final = V1)")
    ap.add_argument("--max-segment-s", type=float, default=35.0)
    ap.add_argument("--feed-ms", type=float, default=10.0, help="audio per feed() call (any value; "
                    "the timeline uses each step's own arrival time)")
    ap.add_argument("--timing", action="store_true",
                    help="reportable timing run: warm-up first; run ONLY on an idle machine (SLOT)")
    args = ap.parse_args()
    if args.combine_full or args.combine_timed:
        if not (args.combine_full and args.combine_timed):
            raise SystemExit("combine needs both --combine-full and --combine-timed")
        combine(args.combine_full, args.combine_timed, args.seed)
        return
    run(args)


if __name__ == "__main__":
    main()
