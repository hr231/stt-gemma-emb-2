"""release: T11, test-clean and test-other once, next to V1 PyTorch and Phonon-2 (same scorer and
normalizer). Run only for a release: nothing is tuned or selected on these sets."""
import json
import re
import time
from pathlib import Path

import pytest

from common import OUT, V1_WER, load_audio, load_list, norm_text
from conftest import PRECISION, RUN_ID, UNITS, VARIANT
from wer import corpus_wer

pytestmark = pytest.mark.release
PHONON2 = Path.home() / "gemma_stt/phonon2/results"
PUNCT = re.compile(r"[^a-z0-9' ]+")


def norm_punct(t):
    """norm_text after mapping punctuation (except apostrophes) to spaces: the shared normalizer for all
    three systems (Phonon-2 writes punctuation and casing; for V1 / Core ML outputs it is a no-op)."""
    return " ".join(PUNCT.sub(" ", norm_text(t)).split())


def test_t11_release(cml, record):
    from transcribe import Transcriber
    tr = Transcriber(cml=cml)
    utts = load_list("release")
    m = {"splits": {}}
    for split in ("test-clean", "test-other"):
        us = [u for u in utts if u["split"] == split]
        hyps = [tr.transcribe(load_audio(u["audio"]))[0] for u in us]
        refs = [norm_punct(u["ref"]) for u in us]
        w, e, n = corpus_wer(refs, [norm_punct(h) for h in hyps])
        p2 = json.loads((PHONON2 / f"score_harness_{split}.json").read_text())
        m["splits"][split] = {"utts": len(us), "words": n, "coreml_wer": w, "coreml_edits": e,
                              "punct_step_changed_hyps": sum(norm_punct(h) != norm_text(h) for h in hyps),
                              "v1_pytorch_wer": V1_WER[split], "phonon2_wer": p2["punct"]["wer"],
                              "phonon2_edits": p2["punct"]["edits"], "phonon2_rtfx_our_method": p2["rtfx"]}
        (OUT / "eval" / f"t11_{VARIANT}_{PRECISION}_{split}_hyps.jsonl").write_text(
            "".join(json.dumps({"id": u["id"], "hyp": h}) + "\n" for u, h in zip(us, hyps)))
    lines = [f"# Release check (T11), {time.strftime('%F %T')}, run {RUN_ID}", "",
             f"Core ML build: {VARIANT}/{PRECISION} under {UNITS}. Scorer: corpus WER (decode/wer.py) against our "
             "references; normalizer: probe.norm_text + punctuation (except apostrophes) to spaces, for all three "
             "systems (a no-op on V1 and Core ML outputs).", "",
             "| split | words | Core ML (this build) | V1 PyTorch | Phonon-2 (FluidAudio) |", "|---|---|---|---|---|"]
    for split, s in m["splits"].items():
        lines.append(f"| {split} | {s['words']:,} | {100 * s['coreml_wer']:.3f} % ({s['coreml_edits']}) | "
                     f"{100 * s['v1_pytorch_wer']:.3f} % | {100 * s['phonon2_wer']:.3f} % ({s['phonon2_edits']}) |")
    (OUT / f"release_{RUN_ID}.md").write_text("\n".join(lines) + "\n")
    m["table"] = "\n".join(lines)
    passed = all(s["punct_step_changed_hyps"] == 0 for s in m["splits"].values())
    record(m, passed)
    assert passed, m
