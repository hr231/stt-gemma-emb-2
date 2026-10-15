"""stream: the V1 streaming step model (coreml/stream_step.py, 16 mel frames = 160 ms per call) against
the offline model, on FAST and on a 60-minute continuous stream.

Model under test: StepModel(V1_STREAM_VARIANT (faithful), V1_STREAM_PRECISION (fp16), V1_COMPUTE_UNITS,
V1_STREAM_MECH (default: the shipped mechanism, stream_step.DEFAULT_MECH)). References: the FAST fp32
reference (reference/fast, HF tower + V1 head), the offline Core ML window build under test in conftest
(V1_VARIANT, default the release package; `fast_out`), the PyTorch fp32 StepTorch / V1Window. Every
test records its metrics (suite "stream") in results.jsonl.
"""
import collections
import os
import subprocess

import numpy as np
import pytest
import torch

from cmlrun import cos_rows
from common import beam_decoder, greedy_text, load_audio, load_list, norm_text
from conftest import UNITS, VARIANT
from stream_step import (DEFAULT_MECH, StepModel, StepTorch, build_core, cos_flat, load_v1, long_mel,
                         offline_torch, package_path, run_stream, tile_offline)
from wer import corpus_wer

pytestmark = pytest.mark.stream
MECH = os.environ.get("V1_STREAM_MECH", DEFAULT_MECH)
S_VARIANT = os.environ.get("V1_STREAM_VARIANT", "faithful")
S_PRECISION = os.environ.get("V1_STREAM_PRECISION", "fp16")
COS_MIN = 0.999                 # per-utterance (per-minute on the long stream) cosine bar
T2_IDENTICAL_MIN = 0.94         # test_fast.T2_IDENTICAL_MIN
FP32_COS_MIN = 0.999999         # PyTorch fp32 step vs offline fp32: same math, other summation order
LONG_S = 3600
SPAN = 15000                    # 10 minutes of L11 frames (25 / s)
RSS_EVERY_MEL = 30000           # 5 minutes of mel frames
RSS_GROWTH_MAX_MB = 50


def _rec(record, m, passed):
    record({"suite": "stream", "mech": MECH, "stream_variant": S_VARIANT, "stream_precision": S_PRECISION,
            "offline_variant": VARIANT, **m}, passed)


@pytest.fixture(scope="module")
def step():
    return StepModel(S_VARIANT, S_PRECISION, UNITS, MECH)


@pytest.fixture(scope="module")
def stream_fast(step, fast_out):
    rows = []
    for r in fast_out:
        lp, l11, calls = run_stream(step, r["mel"], r["u"]["mel_valid"])
        rows.append({**r, "s_lp": lp, "s_l11": l11, "calls": calls})
    return rows


@pytest.fixture(scope="module")
def v1():
    return load_v1()


@pytest.fixture(scope="module")
def torch_step(v1):
    torch.set_num_threads(4)
    return StepTorch(build_core(S_VARIANT, v1=v1))


# ---------------------------------------------------------------- (a) stepwise vs offline on FAST
def test_stream_numerics(stream_fast, record):
    """Per-utterance cosine of L11 and log-probs >= 0.999 against the fp32 reference (the offline model);
    against the offline Core ML window build the same bar, except on utterances where that build is itself
    further from the reference than the stream is (two fp16 graphs with different rounding: their mutual
    distance is bounded by both errors, the offline build's own worst FAST L11 cosine is ~0.998); output
    counts = ceil(V/4); no NaN/Inf."""
    finite = all(np.isfinite(r["s_lp"]).all() and np.isfinite(r["s_l11"]).all() for r in stream_fast)
    counts = all(len(r["s_lp"]) == len(r["s_l11"]) == len(r["ref_lp"]) for r in stream_fast)
    m = {"utts": len(stream_fast), "finite": bool(finite), "counts_ok": bool(counts),
         "calls": int(sum(r["calls"] for r in stream_fast))}
    cos = {}
    for name, key in (("ref", "ref_"), ("offline_coreml", "")):
        l = np.array([cos_flat(r["s_l11"], r[key + "l11"]) for r in stream_fast])
        p = np.array([cos_flat(r["s_lp"], r[key + "lp"]) for r in stream_fast])
        fr = np.concatenate([cos_rows(r["s_l11"], r[key + "l11"]) for r in stream_fast])
        cos[name] = (l, p)
        m[name] = {"l11_utt_cos_min": float(l.min()), "l11_utt_cos_mean": float(l.mean()),
                   "lp_utt_cos_min": float(p.min()), "lp_utt_cos_mean": float(p.mean()),
                   "l11_frame_cos_min": float(fr.min()), "l11_frame_cos_mean": float(fr.mean()),
                   "l11_frames_below_0.99": int((fr < 0.99).sum()), "frames": len(fr)}
    # the offline window build's own distance to the reference
    ol = np.array([cos_flat(r["l11"], r["ref_l11"]) for r in stream_fast])
    op = np.array([cos_flat(r["lp"], r["ref_lp"]) for r in stream_fast])
    m["offline_coreml_vs_ref"] = {"l11_utt_cos_min": float(ol.min()), "l11_utt_cos_mean": float(ol.mean()),
                                  "lp_utt_cos_min": float(op.min())}
    ref_ok = bool(min(cos["ref"][0].min(), cos["ref"][1].min()) >= COS_MIN)
    below = []
    for i, r in enumerate(stream_fast):
        (l, p), (rl, rp) = cos["offline_coreml"], cos["ref"]
        if min(l[i], p[i]) < COS_MIN:
            below.append({"id": r["u"]["id"], "stream_vs_offline_l11": float(l[i]), "stream_vs_offline_lp": float(p[i]),
                          "stream_vs_ref_l11": float(rl[i]), "stream_vs_ref_lp": float(rp[i]),
                          "offline_vs_ref_l11": float(ol[i]), "offline_vs_ref_lp": float(op[i]),
                          "offline_build_error": bool(ol[i] < rl[i] and op[i] <= max(rp[i], COS_MIN))})
    m["offline_coreml"]["utts_below_bar"] = below
    passed = finite and counts and ref_ok and all(b["offline_build_error"] for b in below)
    _rec(record, m, passed)
    assert passed, m


def _transcripts(rows, kind, key):
    if kind == "greedy":
        return [greedy_text(r[key]) for r in rows]
    dec = beam_decoder()
    return [norm_text(dec.decode(r[key])) for r in rows]


@pytest.mark.parametrize("kind", ["greedy", "beam"])
def test_stream_transcripts(stream_fast, kind, record):
    """greedy: identical to the reference transcripts on >= 94 % of FAST and WER within 0.1 points;
    beam (V1 decoder, beam + LM): WER within 0.1 points of the reference."""
    refs = [r["u"]["ref"] for r in stream_fast]
    hyps = _transcripts(stream_fast, kind, "s_lp")
    off = _transcripts(stream_fast, kind, "lp")
    ref_hyps = [r["u"][kind] for r in stream_fast]
    w, e, n = corpus_wer(refs, hyps)
    wr, er, _ = corpus_wer(refs, ref_hyps)
    wo, eo, _ = corpus_wer(refs, off)
    same = sum(a == b for a, b in zip(hyps, ref_hyps))
    m = {"kind": kind, "wer": w, "edits": e, "ref_wer": wr, "ref_edits": er, "offline_coreml_wer": wo,
         "offline_coreml_edits": eo, "words": n, "identical": same, "identical_share": same / len(hyps),
         "identical_to_offline_coreml": sum(a == b for a, b in zip(hyps, off)),
         "offline_coreml_identical_to_ref": sum(a == b for a, b in zip(off, ref_hyps)),
         "wer_diff_points": 100 * (w - wr),
         "differences": [{"id": r["u"]["id"], "stream": a, "reference": b}
                         for r, a, b in zip(stream_fast, hyps, ref_hyps) if a != b]}
    passed = abs(m["wer_diff_points"]) <= 0.1 and (kind == "beam" or m["identical_share"] >= T2_IDENTICAL_MIN)
    _rec(record, m, passed)
    assert passed, {k: v for k, v in m.items() if k != "differences"}


def test_stream_torch_fp32(torch_step, v1, fast_ref, record):
    """PyTorch fp32 StepTorch (the module that is converted) vs the fp32 reference on 10 FAST utterances
    and vs V1Window fp32 on 3: cosine >= 0.999999, greedy identical."""
    from cmlrun import ref_slices
    utts, arrays = fast_ref
    rows = []
    for i, u in enumerate(utts[:10]):
        mel, rl, rp = ref_slices(u, arrays)
        lp, l11, _ = run_stream(torch_step, mel, u["mel_valid"])
        r = {"id": u["id"], "l11_cos": cos_flat(l11, rl), "lp_cos": cos_flat(lp, rp),
             "lp_max_abs_diff": float(np.abs(lp - rp).max()), "greedy_same": greedy_text(lp) == u["greedy"]}
        if i < 3:
            wlp, wl11 = offline_torch(v1, mel, u["mel_valid"], S_VARIANT)
            r.update({"win_l11_cos": cos_flat(l11, wl11), "win_lp_cos": cos_flat(lp, wlp),
                      "win_l11_max_abs_diff": float(np.abs(l11 - wl11).max())})
        rows.append(r)
    cos = [r[k] for r in rows for k in ("l11_cos", "lp_cos", "win_l11_cos", "win_lp_cos") if k in r]
    m = {"utts": len(rows), "cos_min": min(cos), "greedy_identical": sum(r["greedy_same"] for r in rows), "rows": rows}
    passed = m["cos_min"] >= FP32_COS_MIN and m["greedy_identical"] == len(rows)
    _rec(record, m, passed)
    assert passed, m


def test_stream_edges(torch_step, v1, fast_ref, record):
    """Stream lengths around the chunk (16 mel), the head context (68 L11 frames) and the receptive
    field (452 L11 frames), V mod 4 = 0..3: output count ceil(V/4), exactly the expected calls (one flush
    only when needed), fp32 outputs equal to V1Window on the same crop."""
    from cmlrun import ref_slices
    utts, arrays = fast_ref
    u = next(u for u in utts if u["mel_valid"] >= 1900)
    mel = ref_slices(u, arrays)[0]
    rows = []
    for V in (5, 13, 16, 17, 30, 61, 270, 271, 272, 273, 1805, 1806, 1807, 1808):
        lp, l11, calls = run_stream(torch_step, mel[:V], V)
        L, J = -(-V // 4), -(-V // 16)
        want_calls = J + (4 * J - 4 < L)
        wlp, wl11 = offline_torch(v1, mel[:V], V, S_VARIANT)
        rows.append({"V": V, "frames": len(lp), "L": L, "calls": calls, "want_calls": want_calls,
                     "l11_cos": cos_flat(l11, wl11), "lp_cos": cos_flat(lp, wlp),
                     "lp_max_abs_diff": float(np.abs(lp - wlp).max())})
    passed = all(r["frames"] == r["L"] and r["calls"] == r["want_calls"] and min(r["l11_cos"], r["lp_cos"]) >= FP32_COS_MIN
                 for r in rows)
    _rec(record, {"id": u["id"], "rows": rows}, passed)
    assert passed, rows


# ---------------------------------------------------------------- (c) reset, (d) determinism
def test_stream_reset(step, stream_fast, record):
    """A, reset, B on one StepModel == B on a fresh StepModel, bitwise; the state is zero after reset()."""
    fresh = StepModel(S_VARIANT, S_PRECISION, UNITS, MECH)
    rows = []
    for i in range(0, 10, 2):
        a, b = stream_fast[i], stream_fast[i + 1]
        run_stream(step, a["mel"], a["u"]["mel_valid"])
        lp1, l111, _ = run_stream(step, b["mel"], b["u"]["mel_valid"])
        lp2, l112, _ = run_stream(fresh, b["mel"], b["u"]["mel_valid"])
        rows.append({"A": a["u"]["id"], "B": b["u"]["id"], "lp_equal": bool(np.array_equal(lp1, lp2)),
                     "l11_equal": bool(np.array_equal(l111, l112))})
    step.reset()
    zero = all(not np.any(v) for v in step.read_states().values())
    passed = zero and all(r["lp_equal"] and r["l11_equal"] for r in rows)
    _rec(record, {"pairs": rows, "state_zero_after_reset": zero}, passed)
    assert passed, rows


def test_stream_determinism(step, stream_fast, record):
    """The same stream twice gives bitwise identical outputs (20 FAST utterances)."""
    same_lp = same_l11 = 0
    for r in stream_fast[:20]:
        lp, l11, _ = run_stream(step, r["mel"], r["u"]["mel_valid"])
        same_lp += np.array_equal(lp, r["s_lp"])
        same_l11 += np.array_equal(l11, r["s_l11"])
    m = {"utts": 20, "identical_logprobs": int(same_lp), "identical_l11": int(same_l11)}
    passed = same_lp == same_l11 == 20
    _rec(record, m, passed)
    assert passed, m


# ---------------------------------------------------------------- placement
def test_stream_placement(record):
    """Neural Engine share of the estimated cost >= 99 % for the shipped package; every op that is not on
    the Neural Engine is listed. The StateType build's result is recorded alongside."""
    from placement import plan_for, summarize
    s = summarize(plan_for(package_path(S_VARIANT, S_PRECISION, MECH), "CPU_AND_NE"))
    m = {"ane_cost_share": s["ane_cost_share"], "ops": s["ops"], "by_device": s["by_device"],
         "not_on_ane": sorted({f"{r['type']}@{r['device']}" for r in s["not_on_ane"]})}
    for other in ("state", "io"):
        if other == MECH or not package_path(S_VARIANT, S_PRECISION, other).exists():
            continue
        try:
            m[f"{other}_ane_cost_share"] = summarize(plan_for(package_path(S_VARIANT, S_PRECISION, other)))["ane_cost_share"]
            StepModel(S_VARIANT, S_PRECISION, UNITS, other).step(np.zeros((16, 128), np.float32))
            m[f"{other}_predict"] = "ok"
        except RuntimeError as e:
            m[f"{other}_error"] = " ".join(str(e).split())[:600]
    passed = m["ane_cost_share"] >= 0.99
    _rec(record, m, passed)
    assert passed, m


# ---------------------------------------------------------------- (b) 60-minute stream
def _rss_mb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    return int(out.strip()) / 1024.0


def _footprint_mb():
    """phys_footprint of this process (task_info TASK_VM_INFO): resident + compressed dirty memory, what
    Activity Monitor shows. RSS alone drops and regrows as the system compresses / pages a process."""
    import ctypes
    lib = ctypes.CDLL("/usr/lib/libSystem.dylib")
    lib.mach_task_self.restype = ctypes.c_uint32
    buf = (ctypes.c_uint32 * 128)()
    count = ctypes.c_uint32(128)
    kr = lib.task_info(lib.mach_task_self(), 22, buf, ctypes.byref(count))        # TASK_VM_INFO = 22
    assert kr == 0, kr
    return ctypes.c_uint64.from_buffer(buf, 144).value / 2 ** 20                  # task_vm_info.phys_footprint


def _long_audio(seconds=LONG_S):
    """dev-clean utterances, speakers interleaved round-robin (sorted ids), cut to exactly `seconds`."""
    by_spk = collections.defaultdict(list)
    for u in sorted((u for u in load_list("full") if u["split"] == "dev-clean"), key=lambda u: u["id"]):
        by_spk[u["id"].split("-")[0]].append(u)
    queues = [collections.deque(by_spk[s]) for s in sorted(by_spk)]
    picked, total = [], 0.0
    while total < seconds + 1:
        for q in queues:
            if q and total < seconds + 1:
                u = q.popleft()
                picked.append(u)
                total += u["dur"]
    wav = np.concatenate([load_audio(u["audio"]) for u in picked])[:16000 * seconds]
    return wav, picked


def _put(dst, g0, f0, rows):
    """Write rows for frames [f0, f0+len) into dst covering frames [g0, g0+len(dst))."""
    a, b = max(f0, g0), min(f0 + len(rows), g0 + len(dst))
    if a < b:
        dst[a - g0:b - g0] = rows[a - f0:b - f0]


def test_stream_long(step, cml, v1, record):
    """60 minutes of dev-clean (several speakers) fed step by step without reset. No drift: stepwise
    log-probs / L11 of the last 10 minutes match an offline recomputation of that span (receptive-field
    tiling with >= 452 frames of left context; offline Core ML 35 s windows, and PyTorch fp32 V1Window)
    at cosine >= 0.999 per minute, as do the first 10 minutes; process memory (phys_footprint, sampled
    every 5 minutes of audio; RSS recorded too) grows < 50 MB after the first 5 minutes."""
    from wrapper import V1Window
    wav, picked = _long_audio()
    mel = long_mel(wav)
    V = len(mel)
    L = -(-V // 4)
    spans = {"first": 0, "last": L - SPAN}
    out = {k: (np.zeros((SPAN, 257), np.float32), np.zeros((SPAN, 1024), np.float32)) for k in spans}
    for lp_a, l11_a in out.values():                         # commit the pages before measuring memory
        lp_a.fill(1.0), l11_a.fill(1.0)
    step.reset()
    rss, fp = [], []
    s = 0
    while 4 * s < V:
        n = min(16, V - 4 * s)
        lp = step.step(mel[4 * s:4 * s + n], n)
        for k, g0 in spans.items():
            _put(out[k][0], g0, s - 4, lp)
            _put(out[k][1], g0, s, step.last_l11)
        s += 4
        if (4 * s) % RSS_EVERY_MEL == 0:
            rss.append(round(_rss_mb(), 1))
            fp.append(round(_footprint_mb(), 1))
    if s - 4 < L:
        lp = step.flush()
        for k, g0 in spans.items():
            _put(out[k][0], g0, s - 4, lp)
    calls = step.calls

    def cml_predict(x, nv):
        return cml.predict(x, nv, "35s")

    base, head, ck = v1
    win = V1Window(base.audio_tower, head, ck["mean"], ck["std"], 3500).eval()

    def torch_predict(x, nv):
        buf = torch.zeros(1, 3500, 128)
        buf[0, :len(x)] = torch.from_numpy(np.array(x, np.float32))
        with torch.no_grad():
            p, h = win(buf, torch.tensor([nv], dtype=torch.int32))
        k = -(-nv // 4)
        return p[0, :k].numpy(), h[0, :k].numpy()

    per_min = lambda a, b: [cos_flat(a[i:i + 1500], b[i:i + 1500]) for i in range(0, SPAN, 1500)]
    m = {"audio_s": len(wav) / 16000, "utts": len(picked), "speakers": len({u["id"].split("-")[0] for u in picked}),
         "mel_frames": V, "l11_frames": L, "calls": calls, "footprint_mb_every_5min": fp, "rss_mb_every_5min": rss}
    passed = True
    for k, g0 in spans.items():
        refs = {"offline_coreml": tile_offline(cml_predict, mel, V, g0, g0 + SPAN),
                "offline_fp32": tile_offline(torch_predict, mel, V, g0, g0 + SPAN)}
        for name, (rlp, rl11, nwin) in refs.items():
            lp_c, l11_c = per_min(out[k][0], rlp), per_min(out[k][1], rl11)
            fr = cos_rows(out[k][1], rl11)
            m[f"{k}_{name}"] = {"windows": nwin, "lp_cos_per_min": lp_c, "l11_cos_per_min": l11_c,
                                "l11_frame_cos_min": float(fr.min()), "l11_frame_cos_mean": float(fr.mean()),
                                "greedy_same": greedy_text(out[k][0]) == greedy_text(rlp)}
            passed &= min(lp_c + l11_c) >= COS_MIN
    m["footprint_growth_after_warmup_mb"] = round(max(fp) - fp[0], 1)
    m["footprint_last_minus_first_mb"] = round(fp[-1] - fp[0], 1)
    passed &= m["footprint_growth_after_warmup_mb"] < RSS_GROWTH_MAX_MB and calls == -(-V // 16) + (4 * -(-V // 16) - 4 < L)
    _rec(record, m, bool(passed))
    assert passed, m
