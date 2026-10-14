#!/usr/bin/env python3
"""Which part of the V1 window fails the Neural Engine compile? Converts parts alone (fp16,
macOS 14) and loads each with CPU_AND_NE in a child process, capturing Core ML's stderr.

  ~/.venv-coreml/bin/python coreml/bisect_ane.py [--parts layer sscp head int8] [--variant faithful]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

from common import load_head_cpu, load_tower_fp32
from export import VARIANTS
from wrapper import V1Window

T_MEL = 500


class Layer(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.l, self.w = w.layers[0], w

    def forward(self, h):
        return self.l(h, self.w.attn_place, self.w.attn_mask)


class Sscp(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, mel, valid_len):
        w = self.w
        x = w._sscp_layer(w.sscp.layer0, mel[:, None], (w.t0 < valid_len).to(mel.dtype))
        x = w._sscp_layer(w.sscp.layer1, x, (w.t1 < valid_len).to(mel.dtype))
        return w.sscp.input_proj_linear(x.permute(0, 2, 3, 1).reshape(1, w.T, -1))


class Head(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, z, valid_len):
        return self.w._head(z, (self.w.t2 < valid_len)[None, :, None])


class Int8(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, l11):
        scale = l11.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127.0
        return (torch.round(l11 / scale).clamp(-127, 127) * scale - self.w.mean) * self.w.inv_std


LOAD = """
import sys, numpy as np, coremltools as ct
m = ct.models.MLModel(sys.argv[1], compute_units=ct.ComputeUnit.CPU_AND_NE)
feeds = {n: np.zeros(s, dtype=d) for n, s, d in eval(sys.argv[2])}
m.predict(feeds)
print("predict ok")
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["layer", "sscp", "head", "int8"])
    ap.add_argument("--variant", default="faithful")
    args = ap.parse_args()
    base, (head, ck) = load_tower_fp32(), load_head_cpu()
    w = V1Window(base.audio_tower, head, ck["mean"], ck["std"], T_MEL, **VARIANTS[args.variant]).eval()
    T = w.T
    parts = {
        "layer": (Layer(w), [("h", (1, T, 1024), np.float32)]),
        "sscp": (Sscp(w), [("mel", (1, T_MEL, 128), np.float32), ("valid_len", (1,), np.int32)]),
        "head": (Head(w), [("z", (1, T, 1024), np.float32), ("valid_len", (1,), np.int32)]),
        "int8": (Int8(w), [("l11", (1, T, 1024), np.float32)]),
    }
    tmp = Path(tempfile.mkdtemp(prefix="v1bisect_"))
    for name in args.parts:
        mod, ins = parts[name]
        ex = tuple(torch.from_numpy(np.full(s, 3 if d == np.int32 else 0.5, dtype=d)) for _, s, d in ins)
        with torch.no_grad():
            tr = torch.jit.trace(mod.eval(), ex, check_trace=False)
        ml = ct.convert(tr, inputs=[ct.TensorType(name=n, shape=s, dtype=d) for n, s, d in ins],
                        convert_to="mlprogram", minimum_deployment_target=ct.target.macOS14,
                        compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_ONLY)
        path = tmp / f"{name}.mlpackage"
        ml.save(str(path))
        spec = repr([(n, s, np.dtype(d).name) for n, s, d in ins]).replace("'float32'", "np.float32").replace("'int32'", "np.int32")
        p = subprocess.run([sys.executable, "-c", LOAD, str(path), spec], capture_output=True, text=True)
        err = [l for l in (p.stderr + p.stdout).splitlines() if "ANE" in l or "E5RT" in l or "Error" in l]
        status = "ANE COMPILE FAILED" if any("ANECCompile" in l or "failed to compile ANE" in l for l in err) else "ok"
        print(f"[bisect] {name:6s}: {status} | predict: {'ok' if 'predict ok' in p.stdout else 'FAILED'} | "
              + (err[0][:300] if err else ""), flush=True)


if __name__ == "__main__":
    main()
