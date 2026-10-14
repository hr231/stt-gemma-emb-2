"""fast: T1-T8 on FAST (200 seeded dev-clean utterances) against the PyTorch fp32 reference.

Palettized builds (precision name contains "pal", Phase 4) are accepted on WER (T2's WER part, T3,
T9), not on numerics or identical-transcript rates: for them T1's cosine thresholds and T2's
identical share are recorded but not gated (NaN/Inf still is).
"""
import json

import numpy as np
import pytest

from cmlrun import cos_rows
from common import OUT, WINDOWS, beam_decoder, greedy_text, load_audio, norm_text
from conftest import PRECISION, VARIANT
from wer import corpus_wer

pytestmark = pytest.mark.fast
PALETTIZED = "pal" in PRECISION
T2_IDENTICAL_MIN = 0.94      # >= V1's own bf16-vs-fp32 noise floor on FAST (recorded by T2)
V1_GREEDY_HYPS = OUT.parent / "decode/results/dev-clean_greedy_none_a0_b0_beam16/hyps.jsonl"


def cos_flat(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


@pytest.mark.parametrize("window", list(WINDOWS))
def test_t1_numerics(window, fast_out, record):
    """L11 cosine vs the reference (per utterance: mean >= 0.999, min >= 0.99; per frame: mean >= 0.999,
    min reported), log-prob max |d| reported, no NaN/Inf."""
    rows = [r for r in fast_out if r["window"] == window]
    assert rows, f"no FAST utterance falls in the {window} window"
    finite = [bool(np.isfinite(r["lp"]).all() and np.isfinite(r["l11"]).all()) for r in rows]
    ok = [r for r, f in zip(rows, finite) if f]
    utt = np.array([cos_flat(r["l11"], r["ref_l11"]) for r in ok])
    frame = np.concatenate([cos_rows(r["l11"], r["ref_l11"]) for r in ok])
    m = {"window": window, "utts": len(rows), "nonfinite_utts": finite.count(False),
         "utt_cos_mean": float(utt.mean()), "utt_cos_min": float(utt.min()),
         "frame_cos_mean": float(frame.mean()), "frame_cos_min": float(frame.min()),
         "frames": len(frame), "frames_below_0.99": int((frame < 0.99).sum()),
         "l11_max_abs_diff": max(float(np.abs(r["l11"] - r["ref_l11"]).max()) for r in ok),
         "logprob_max_abs_diff": max(float(np.abs(r["lp"] - r["ref_lp"]).max()) for r in ok)}
    numerics_ok = m["utt_cos_mean"] >= 0.999 and m["utt_cos_min"] >= 0.99 and m["frame_cos_mean"] >= 0.999
    m["gated_on_numerics"] = not PALETTIZED
    passed = m["nonfinite_utts"] == 0 and (numerics_ok or PALETTIZED)
    record({**m, "numerics_ok": numerics_ok}, passed)
    assert passed, m


def _parity(fast_out, kind):
    refs = [r["u"]["ref"] for r in fast_out]
    if kind == "greedy":
        hyps = [greedy_text(r["lp"]) for r in fast_out]
    else:
        dec = beam_decoder()
        hyps = [norm_text(dec.decode(r["lp"])) for r in fast_out]
    ref_hyps = [r["u"][kind] for r in fast_out]
    w, e, n = corpus_wer(refs, hyps)
    wr, er, _ = corpus_wer(refs, ref_hyps)
    same = sum(a == b for a, b in zip(hyps, ref_hyps))
    diffs = [{"id": r["u"]["id"], "coreml": a, "reference": b} for r, a, b in zip(fast_out, hyps, ref_hyps) if a != b]
    return {"wer": w, "edits": e, "ref_wer": wr, "ref_edits": er, "words": n, "identical": same,
            "identical_share": same / len(fast_out), "wer_diff_points": 100 * (w - wr), "differences": diffs}


def test_t2_greedy_parity(fast_out, record):
    """Greedy transcripts identical on >= 94 % of FAST and WER within 0.1 points of the reference.
    Also records the noise floor: V1's own greedy transcripts (bf16 features on MPS, the cached V1
    pipeline) vs the same fp32 reference on the same utterances."""
    m = _parity(fast_out, "greedy")
    v1 = {r["id"]: r["hyp"] for r in map(json.loads, V1_GREEDY_HYPS.read_text().splitlines())}
    floor = sum(v1[r["u"]["id"]] == r["u"]["greedy"] for r in fast_out) / len(fast_out)
    m.update({"identical_min": T2_IDENTICAL_MIN, "noise_floor_v1_bf16_identical_share": floor,
              "threshold_at_or_below_noise_floor": T2_IDENTICAL_MIN <= floor,
              "gated_on_identical": not PALETTIZED})
    wer_ok = abs(m["wer_diff_points"]) <= 0.1
    passed = wer_ok and (m["identical_share"] >= T2_IDENTICAL_MIN or PALETTIZED)
    record(m, passed)
    assert passed, {k: v for k, v in m.items() if k != "differences"}


def test_t3_beam_parity(fast_out, record):
    """Beam + LM (V1 decoder) WER within 0.1 points of the reference on FAST."""
    m = _parity(fast_out, "beam")
    passed = abs(m["wer_diff_points"]) <= 0.1
    record(m, passed)
    assert passed, {k: v for k, v in m.items() if k != "differences"}


def test_t4_causality(cml, fast_out, record):
    """Perturbing mel frames from t on leaves L11 frames before t/4 and head outputs before t/4 - 4
    exactly unchanged (encoder: no lookahead; head: 4 frames = 160 ms), and changes later frames."""
    rng = np.random.default_rng(0)
    cases = [r for r in fast_out if 600 <= r["u"]["mel_valid"] <= 1000][:3]
    assert len(cases) == 3
    out = []
    for r in cases:
        nv = r["u"]["mel_valid"]
        base_lp, base_l11 = cml.predict(r["mel"], nv, "10s")
        for t in (200, 400):
            mel = r["mel"].copy()
            mel[t:nv] += rng.normal(0, 1, size=mel[t:nv].shape).astype(np.float32)
            lp, l11 = cml.predict(mel, nv, "10s")
            k = t // 4
            out.append({"id": r["u"]["id"], "t_mel": t, "l11_before_max_abs_diff": float(np.abs(l11[:k] - base_l11[:k]).max()),
                        "head_before_max_abs_diff": float(np.abs(lp[:k - 4] - base_lp[:k - 4]).max()),
                        "l11_after_max_abs_diff": float(np.abs(l11[k + 1:] - base_l11[k + 1:]).max()),
                        "head_window_max_abs_diff": float(np.abs(lp[k - 4:k] - base_lp[k - 4:k]).max())})
    m = {"cases": out, "max_l11_before": max(c["l11_before_max_abs_diff"] for c in out),
         "max_head_before": max(c["head_before_max_abs_diff"] for c in out),
         "min_l11_after": min(c["l11_after_max_abs_diff"] for c in out)}
    passed = m["max_l11_before"] == 0.0 and m["max_head_before"] == 0.0 and m["min_l11_after"] > 0.0
    record(m, passed)
    assert passed, m


def test_t5_padding_invariance(cml, fast_out, record):
    """Clips up to 10 s give the same valid outputs in the 10 s and 20 s windows (cosine >= 0.9999)."""
    rows = [r for r in fast_out if r["u"]["mel_frames"] <= WINDOWS["10s"]]
    l11_cos, lp_cos, lp_d, same = [], [], 0.0, 0
    for r in rows:
        a_lp, a_l11 = cml.predict(r["mel"], r["u"]["mel_valid"], "10s")
        b_lp, b_l11 = cml.predict(r["mel"], r["u"]["mel_valid"], "20s")
        l11_cos.append(cos_flat(a_l11, b_l11))
        lp_cos.append(cos_flat(a_lp, b_lp))
        lp_d = max(lp_d, float(np.abs(a_lp - b_lp).max()))
        same += greedy_text(a_lp) == greedy_text(b_lp)
    m = {"utts": len(rows), "l11_cos_min": min(l11_cos), "l11_cos_mean": float(np.mean(l11_cos)),
         "logprob_cos_min": min(lp_cos), "logprob_max_abs_diff": lp_d, "greedy_identical": same}
    passed = m["l11_cos_min"] >= 0.9999 and m["logprob_cos_min"] >= 0.9999
    record(m, passed)
    assert passed, m


@pytest.mark.parametrize("window", list(WINDOWS))
def test_t6_placement(window, record):
    """Share of the estimated cost on the Neural Engine (CPU_AND_NE) >= the share recorded at the first
    passing run of this variant / precision / window (regression guard)."""
    from cmlrun import package
    from placement import plan_for, summarize
    s = summarize(plan_for(package(VARIANT, PRECISION, window), "CPU_AND_NE"))
    path = OUT / "placement_thresholds.json"
    thresholds = json.loads(path.read_text()) if path.exists() else {}
    key = f"{VARIANT}/{PRECISION}/{window}"
    first = key not in thresholds
    if first:
        thresholds[key] = s["ane_cost_share"]
        path.write_text(json.dumps(thresholds, indent=1))
    m = {"window": window, "ane_cost_share": s["ane_cost_share"], "threshold": thresholds[key], "recorded_now": first,
         "ops_by_device": s["by_device"], "not_on_ane": sorted({f"{r['type']}@{r['device']}" for r in s["not_on_ane"]})}
    passed = s["ane_cost_share"] >= thresholds[key] - 1e-9
    record(m, passed)
    assert passed, m


def test_t7_determinism(cml, fast_out, record):
    """Two runs on the same inputs give bitwise identical outputs."""
    rows = fast_out[:20]
    same_lp = same_l11 = 0
    for r in rows:
        lp, l11 = cml.predict(r["mel"], r["u"]["mel_valid"], r["window"])
        same_lp += np.array_equal(lp, r["lp"])
        same_l11 += np.array_equal(l11, r["l11"])
    m = {"utts": len(rows), "identical_logprobs": same_lp, "identical_l11": same_l11}
    passed = same_lp == same_l11 == len(rows)
    record(m, passed)
    assert passed, m


def test_t8_long_audio(cml, fast_ref, record):
    """3 dev-clean utterances concatenated (> 35 s, so cut at pauses): WER within 0.2 points of decoding
    them separately."""
    from transcribe import Transcriber
    utts, _ = fast_ref
    three = [u for u in utts if 13 <= u["dur"] <= 17][:3]
    wavs = [load_audio(u["audio"]) for u in three]
    cat = np.concatenate(wavs)
    tr = Transcriber(cml=cml)
    text, info = tr.transcribe(cat)
    sep = " ".join(tr.transcribe(w)[0] for w in wavs)
    ref = " ".join(u["ref"] for u in three)
    wc, ec, n = corpus_wer([ref], [text])
    ws, es, _ = corpus_wer([ref], [sep])
    m = {"ids": [u["id"] for u in three], "audio_s": len(cat) / 16000, "segments": info["segments"],
         "wer_concatenated": wc, "wer_separate": ws, "edits_concatenated": ec, "edits_separate": es, "words": n,
         "diff_points": 100 * (wc - ws)}
    passed = m["audio_s"] > 35 and len(info["segments"]) > 1 and abs(m["diff_points"]) <= 0.2
    record(m, passed)
    assert passed, m
