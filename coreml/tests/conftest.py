"""Fixtures for the V1 Core ML test suite (markers fast / full / release, see pytest.ini).

Model under test: V1_VARIANT (default release: the one multifunction package; faithful / fp16safe are the
internal per-window builds), V1_PRECISION (fp16), V1_COMPUTE_UNITS (CPU_AND_NE).
Every test calls `record(metrics, passed)` before asserting: one JSON line per test in
~/gemma_stt/coreml/results.jsonl with the run id, versions, model weight hashes and git commit.
"""
import importlib.metadata as md
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from cmlrun import V1CoreML, load_reference, package, ref_slices  # noqa: E402
from common import OUT, WINDOWS, sha256  # noqa: E402

VARIANT = os.environ.get("V1_VARIANT", "release")
PRECISION = os.environ.get("V1_PRECISION", "fp16")
UNITS = os.environ.get("V1_COMPUTE_UNITS", "CPU_AND_NE")
RUN_ID = os.environ.get("V1_RUN_ID", time.strftime("%Y%m%d-%H%M%S"))
RESULTS = OUT / "results.jsonl"


def _git():
    repo = HERE.parents[1]
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


@pytest.fixture(scope="session")
def meta():
    pkgs = {w: package(VARIANT, PRECISION, w) for w in WINDOWS}       # the release: one package for all windows
    sums = {p: sha256(p / "Data/com.apple.CoreML/weights/weight.bin") for p in set(pkgs.values())}
    weights = {w: sums[p] for w, p in pkgs.items()}
    return {"run": RUN_ID, "variant": VARIANT, "precision": PRECISION, "compute_units": UNITS, "git": _git(),
            "versions": {p: md.version(p) for p in ("coremltools", "torch", "transformers", "numpy", "ctc-beam-decoder")},
            "model_weight_sha256": weights}


@pytest.fixture
def record(request, meta):
    def rec(metrics, passed):
        marker = next((m.name for m in request.node.iter_markers() if m.name in ("fast", "full", "release", "stream")), None)
        row = {"time": time.strftime("%F %T"), "test": request.node.name, "marker": marker, "pass": bool(passed),
               "metrics": metrics, **meta}
        with open(RESULTS, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")
    return rec


@pytest.fixture(scope="session")
def cml():
    return V1CoreML(VARIANT, PRECISION, UNITS)


@pytest.fixture(scope="session")
def fast_ref():
    return load_reference("fast")


@pytest.fixture(scope="session")
def fast_out(cml, fast_ref):
    """Core ML outputs for every FAST utterance in its smallest fitting window, with the reference."""
    utts, arrays = fast_ref
    rows = []
    for u in utts:
        mel, rl, rp = ref_slices(u, arrays)
        w = cml.window_for(len(mel))
        lp, l11 = cml.predict(mel, u["mel_valid"], w)
        rows.append({"u": u, "mel": mel, "window": w, "lp": lp, "l11": l11, "ref_l11": rl, "ref_lp": rp})
    return rows
