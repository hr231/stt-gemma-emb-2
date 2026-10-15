"""Running the exported V1 Core ML windows: load packages, pick the smallest window that fits,
pad, predict, slice to the valid frames. Shared by the evaluation scripts, the runner and the tests.

Packages are compiled once to persistent .mlmodelc directories (OUT/compiled/...) and loaded with
CompiledMLModel: MLModel(<.mlpackage>) recompiles into a new temporary path on every load, so the
Neural Engine compile (~45 s for the four windows) never hits Core ML's cache.
"""
import fcntl
import hashlib
import json
import time
import threading

import numpy as np

from common import OUT, WINDOWS

COMPUTE_UNITS = ("CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY", "ALL")
COMPILED = OUT / "compiled"
RELEASE = "release"     # the released build: ONE multifunction package, one function per window


def function_name(window):
    """Function of the release package for a window: win_5s, win_10s, ... (no '.': the ANE compiler rejects it)."""
    return f"win_{window}"


def package(variant, precision, window=None):
    """Package file: the release multifunction package, or an internal per-window package."""
    if variant == RELEASE:
        return OUT / "models" / RELEASE / precision / "v1.mlpackage"
    return OUT / "models" / variant / precision / f"v1_{window}.mlpackage"


def _package_key(src):
    """Fingerprint of a package (relative path, size, mtime of every file)."""
    files = sorted(f for f in src.rglob("*") if f.is_file())
    return hashlib.sha256("\n".join(f"{f.relative_to(src)}:{f.stat().st_size}:{f.stat().st_mtime_ns}"
                                    for f in files).encode()).hexdigest()


def compiled_path(variant, precision, window=None):
    """Persistent compiled model for a package; recompiled (old copy moved aside) when the package changes."""
    import coremltools as ct
    src = package(variant, precision, window)
    name = src.stem
    dst = COMPILED / variant / precision / f"{name}.mlmodelc"
    stamp = dst.parent / f"{name}.source"
    dst.parent.mkdir(parents=True, exist_ok=True)
    key = _package_key(src)
    with open(dst.parent / f".{name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if dst.exists() and stamp.exists() and stamp.read_text() == key:
            return dst
        if dst.exists():
            dst.rename(dst.with_name(f"{dst.name}.old-{int(time.time())}"))
        tmp = dst.with_name(f"{name}.partial.mlmodelc")                  # compile_model wants .mlmodelc
        if tmp.exists():
            tmp.rename(dst.with_name(f"{dst.name}.stale-{int(time.time())}"))
        ct.utils.compile_model(str(src), str(tmp))
        tmp.rename(dst)
        stamp.write_text(key)
    return dst


class V1CoreML:
    """All four windows of one variant / precision, loaded with the given compute units."""

    def __init__(self, variant="faithful", precision="fp16", compute_units="CPU_AND_NE", windows=None):
        import coremltools as ct
        self.variant, self.precision, self.compute_units = variant, precision, compute_units
        self.models, self.load_s = {}, {}
        # Core ML releases a prediction's input feature values later, on its own queue; if Python has
        # dropped the arrays by then the process segfaults (MLE5ExecutionStream resetAfterLingering).
        # So every window keeps its input buffers for the model's lifetime and reuses them. Keep them
        # float32 / int32: Core ML re-reads those after in-place changes, but caches fp16 inputs from first use.
        self._inputs, self._lock = {}, threading.Lock()
        for w in windows or WINDOWS:
            t0 = time.time()
            fn = {"function_name": function_name(w)} if variant == RELEASE else {}
            self.models[w] = ct.models.CompiledMLModel(str(compiled_path(variant, precision, w)),
                                                       compute_units=getattr(ct.ComputeUnit, compute_units), **fn)
            self.load_s[w] = time.time() - t0
            self._inputs[w] = (np.zeros((1, WINDOWS[w], 128), np.float32), np.zeros(1, np.int32))

    @staticmethod
    def window_for(n_mel, windows=WINDOWS):
        fits = [w for w, t in sorted(windows.items(), key=lambda kv: kv[1]) if t >= n_mel]
        if not fits:
            raise ValueError(f"{n_mel} mel frames exceed the largest window ({max(windows.values())})")
        return fits[0]

    def predict(self, mel, n_valid, window=None):
        """(log_probs [ceil(n_valid/4), 257], l11 [ceil(n_valid/4), 1024]) as float32 copies."""
        w = window or self.window_for(len(mel))
        n = -(-n_valid // 4)
        with self._lock:
            x, v = self._inputs[w]
            x[0, :len(mel)] = mel
            x[0, len(mel):] = 0.0
            v[0] = n_valid
            out = self.models[w].predict({"mel": x, "valid_len": v})
            return (np.array(out["log_probs"], np.float32)[0, :n], np.array(out["l11"], np.float32)[0, :n])


def load_reference(name="fast"):
    d = OUT / "reference" / name
    idx = json.loads((d / "index.json").read_text())["utts"]
    arrays = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("mel", "l11", "logprobs")}
    return idx, arrays


def ref_slices(u, arrays):
    return (np.asarray(arrays["mel"][u["mel_off"]:u["mel_off"] + u["mel_frames"]]),
            np.asarray(arrays["l11"][u["off"]:u["off"] + u["frames"]]),
            np.asarray(arrays["logprobs"][u["off"]:u["off"] + u["frames"]]))


def cos_rows(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)
