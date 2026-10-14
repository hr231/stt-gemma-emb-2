"""full: T9 (dev-clean and dev-other beam+LM WER within 0.1 points of V1) and T10 (speed, idle Mac)."""
import json
import math

import pytest

from common import OUT, V1_WER, load_audio, load_list, norm_text
from conftest import PRECISION, VARIANT
from wer import corpus_wer

pytestmark = pytest.mark.full
PALETTIZED = "pal" in PRECISION


def test_t9_wer_vs_v1(cml, record):
    """Core ML end to end (mel -> model -> V1 decoder) on all of dev-clean and dev-other: beam+LM WER
    within 0.1 points of V1 (V1.md: 3.726 % / 10.469 %). The binding accuracy gate."""
    from transcribe import Transcriber
    tr = Transcriber(cml=cml)
    utts = load_list("full")
    m, passed = {}, True
    d = OUT / "eval"
    d.mkdir(parents=True, exist_ok=True)
    for split in ("dev-clean", "dev-other"):
        us = [u for u in utts if u["split"] == split]
        hyps = [tr.transcribe(load_audio(u["audio"]))[0] for u in us]
        w, e, n = corpus_wer([norm_text(u["ref"]) for u in us], hyps)
        diff = 100 * (w - V1_WER[split])
        m[split] = {"utts": len(us), "wer": w, "edits": e, "words": n, "v1_wer": V1_WER[split], "diff_points": diff}
        passed &= abs(diff) <= 0.1
        (d / f"t9_{VARIANT}_{PRECISION}_{split}_hyps.jsonl").write_text(
            "".join(json.dumps({"id": u["id"], "hyp": h}) + "\n" for u, h in zip(us, hyps)))
    record(m, passed)
    assert passed, m


def test_t10_speed(record):
    """RTFx (audio duration / processing time, model load excluded, after warm-up) over full dev-clean:
    this build's encoder+head under CPU_AND_NE, CPU_AND_GPU and CPU_ONLY with the decoder separately and
    end to end, plus the exact fp32 build under CPU_AND_GPU (same method as the Phonon-2 harness);
    WER, peak RSS, cold / warm load. Needs an otherwise idle Mac."""
    import bench
    configs = [f"{PRECISION}:CPU_AND_NE", f"{PRECISION}:CPU_AND_GPU", f"{PRECISION}:CPU_ONLY"]
    if not PALETTIZED:
        configs.append("fp32:CPU_AND_GPU")
    res = bench.run_all(configs, VARIANT)
    ok = [r for r in res["runs"] if all(math.isfinite(r[k]) and r[k] > 0 for k in ("rtfx_e2e", "rtfx_model", "rtfx_decoder"))]
    passed = len(ok) == len(configs)
    record(res, passed)
    assert passed, res
