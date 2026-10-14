#!/usr/bin/env python3
"""Phase 1 step 6: where does fp16 diverge? Per-layer comparison of the tower in Core ML fp16
(CPU_AND_NE) against the same module in PyTorch fp32, on the FAST utterances that fit the 5 s window.

  ~/.venv-coreml/bin/python coreml/fp16_layers.py --variant faithful [--n 12]

Builds a debug package whose outputs are the hidden states after each of the 12 tower layers,
runs the n utterances with the lowest L11 cosine of the given eval (eval_fast.py output), and prints
per layer: min / mean frame cosine vs fp32, and the largest |x| entering an RMSNorm in that layer
(fp32; x^2 overflows fp16 for |x| >= 256).
"""
import argparse
import json
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

from cmlrun import cos_rows, load_reference, ref_slices
from common import OUT, load_head_cpu, load_tower_fp32
from export import VARIANTS
from wrapper import V1Window

T_MEL = 500


class Layers(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, mel, valid_len):
        w, V = self.w, valid_len
        x = w._sscp_layer(w.sscp.layer0, mel[:, None], (w.t0 < V).to(mel.dtype))
        x = w._sscp_layer(w.sscp.layer1, x, (w.t1 < V).to(mel.dtype))
        h = w.sscp.input_proj_linear(x.permute(0, 2, 3, 1).reshape(1, w.T, -1))
        if w.pad_fill is not None:
            h = torch.where((w.t2 < V)[None, :, None], h, torch.full_like(h, w.pad_fill))
        outs = []
        for layer in w.layers:
            h = layer(h, w.attn_place, w.attn_mask)
            outs.append(h)
        return tuple(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="faithful")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    base, (head, ck) = load_tower_fp32(), load_head_cpu()
    w = V1Window(base.audio_tower, head, ck["mean"], ck["std"], T_MEL, **VARIANTS[args.variant]).eval()
    mod = Layers(w).eval()
    names = [f"layer{i}" for i in range(12)]
    with torch.no_grad():
        tr = torch.jit.trace(mod, (torch.zeros(1, T_MEL, 128), torch.tensor([T_MEL], dtype=torch.int32)), check_trace=False)
    ml = ct.convert(tr, inputs=[ct.TensorType(name="mel", shape=(1, T_MEL, 128), dtype=np.float32),
                                ct.TensorType(name="valid_len", shape=(1,), dtype=np.int32)],
                    outputs=[ct.TensorType(name=n) for n in names], convert_to="mlprogram",
                    minimum_deployment_target=ct.target.macOS14, compute_precision=ct.precision.FLOAT16,
                    compute_units=ct.ComputeUnit.CPU_AND_NE)
    path = Path(tempfile.mkdtemp(prefix="v1layers_")) / "layers.mlpackage"
    ml.save(str(path))
    ml = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)

    utts, arrays = load_reference("fast")
    ev = OUT / "eval" / f"{args.variant}_fp16_CPU_AND_NE_hyps.jsonl"
    fit = [u for u in utts if u["mel_frames"] <= T_MEL]
    # lowest-cosine utterances first (from the full-model eval's per-utterance L11 if present)
    scores = []
    for u in fit:
        mel, rl, _ = ref_slices(u, arrays)
        scores.append((u, mel))
    norm_max = {}
    hooks = []
    for i, layer in enumerate(w.layers):
        for n, m in layer.named_modules():
            if type(m).__name__ == "RMSNormExport":
                hooks.append(m.register_forward_pre_hook(
                    lambda mod_, inp, i=i: norm_max.__setitem__(i, max(norm_max.get(i, 0.0), float(inp[0].abs().max())))))
    per = {i: [] for i in range(12)}
    worst = []
    for u, mel in scores:
        x = np.zeros((1, T_MEL, 128), np.float32)
        x[0, :len(mel)] = mel
        v = np.array([u["mel_valid"]], np.int32)
        out = ml.predict({"mel": x, "valid_len": v})
        with torch.no_grad():
            ref = mod(torch.from_numpy(x), torch.from_numpy(v))
        n = -(-u["mel_valid"] // 4)
        cs = []
        for i in range(12):
            c = cos_rows(np.asarray(out[names[i]], np.float32)[0, :n], ref[i][0, :n].numpy())
            per[i].append(c)
            cs.append(float(c.min()))
        worst.append((cs[-1], u["id"], cs))
    for h in hooks:
        h.remove()
    worst.sort()
    print(f"variant {args.variant}: {len(scores)} utterances in the 5 s window")
    for i in range(12):
        c = np.concatenate(per[i])
        print(f"  layer {i:2d}: frame cosine vs fp32 min {c.min():.6f} mean {c.mean():.6f} | frames < 0.99: "
              f"{int((c < 0.99).sum())} | max |x| into an RMSNorm (fp32) {norm_max.get(i, 0):.1f}")
    print("  worst utterances (L11 min cosine, per-layer min):")
    for c, uid, cs in worst[: args.n]:
        first = next((i for i, v in enumerate(cs) if v < 0.99), None)
        print(f"    {uid}: L11 min {c:.4f} | first layer < 0.99: {first} | " + " ".join(f"{v:.3f}" for v in cs))


if __name__ == "__main__":
    main()
