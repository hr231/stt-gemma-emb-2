#!/usr/bin/env python3
"""Phase 1 export: V1Window per window -> torch.jit.trace -> Core ML mlprogram (macOS 14).

  ~/.venv-coreml/bin/python coreml/export.py --windows 5s 10s 20s 35s --variant faithful --precision fp16

Writes OUT/models/<variant>/<precision>/v1_<window>.mlpackage (inputs: mel [1, T_mel, 128] float32,
valid_len [1] int32 = valid mel frames; outputs: log_probs [1, T_mel/4, 257], l11 [1, T_mel/4, 1024])
and appends one JSON line per package to OUT/models/export_log.jsonl. Variants: faithful (HF ops),
fp16safe (power-of-two-scaled RMSNorm + constant padding fill, identical in fp32), noint8 (no int8
round trip; V1's head needs it, see the reference). fp16 = compute precision FLOAT16 (op_selector
may keep named ops in fp32: --fp32-ops FILE with one MIL op name per line).
"""
import argparse
import json
import time

import coremltools as ct
import numpy as np
import torch

from common import OUT, WINDOWS, load_head_cpu, load_tower_fp32, sha256
from wrapper import V1Window

VARIANTS = {"faithful": {}, "fp16safe": {"rms_scale": 32.0, "pad_fill": 1.0}, "noint8": {"int8": False},
            "fp16safe_noint8": {"rms_scale": 32.0, "pad_fill": 1.0, "int8": False}}


def package_path(variant, precision, window):
    return OUT / "models" / variant / precision / f"v1_{window}.mlpackage"


def convert(model, t_mel, precision, fp32_ops=None):
    mel = torch.zeros(1, t_mel, 128)
    valid = torch.tensor([t_mel], dtype=torch.int32)
    with torch.no_grad():
        traced = torch.jit.trace(model, (mel, valid), check_trace=False)
    if precision == "fp32":
        cp = ct.precision.FLOAT32
    elif fp32_ops:
        keep = set(fp32_ops)
        cp = ct.transform.FP16ComputePrecision(op_selector=lambda op: op.name not in keep)
    else:
        cp = ct.precision.FLOAT16
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=(1, t_mel, 128), dtype=np.float32),
                ct.TensorType(name="valid_len", shape=(1,), dtype=np.int32)],
        outputs=[ct.TensorType(name="log_probs"), ct.TensorType(name="l11")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=cp,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", nargs="+", default=list(WINDOWS))
    ap.add_argument("--variant", default="faithful", choices=VARIANTS)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--fp32-ops", default=None, help="file with MIL op names to keep in fp32 (fp16 only)")
    ap.add_argument("--tag", default="", help="suffix for the precision directory, e.g. fp16_opsel")
    args = ap.parse_args()
    fp32_ops = [l.strip() for l in open(args.fp32_ops) if l.strip()] if args.fp32_ops else None
    base, (head, ck) = load_tower_fp32(), load_head_cpu()
    for w in args.windows:
        t_mel = WINDOWS[w]
        t0 = time.time()
        model = V1Window(base.audio_tower, head, ck["mean"], ck["std"], t_mel, **VARIANTS[args.variant]).eval()
        ml = convert(model, t_mel, args.precision, fp32_ops)
        prec = args.precision + args.tag
        path = package_path(args.variant, prec, w)
        path.parent.mkdir(parents=True, exist_ok=True)
        ml.short_description = f"V1 (EmbeddingGemma 2 L11 + CTC head), {w} window, {args.variant}, {prec}"
        ml.save(str(path))
        weights = path / "Data/com.apple.CoreML/weights/weight.bin"
        rec = {"time": time.strftime("%F %T"), "window": w, "t_mel": t_mel, "variant": args.variant, "precision": prec,
               "fp32_ops": len(fp32_ops or []), "path": str(path), "seconds": round(time.time() - t0, 1),
               "weight_bytes": weights.stat().st_size if weights.exists() else None,
               "weight_sha256": sha256(weights) if weights.exists() else None,
               "coremltools": ct.__version__, "torch": torch.__version__}
        with open(OUT / "models" / "export_log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[export] {json.dumps(rec)}", flush=True)
        del model, ml


if __name__ == "__main__":
    main()
