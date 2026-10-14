#!/usr/bin/env python3
"""Check the export wrapper (PyTorch fp32, CPU) against the fp32 reference before converting.

  ~/.venv-coreml/bin/python coreml/check_wrapper.py [--variant faithful|fp16safe] [--limit N]

Every FAST utterance runs in its smallest fitting window (and, when it fits, also the next larger
one, for padding invariance); valid frames are compared with the reference: L11 cosine per frame,
max abs differences of L11 and log-probs, greedy transcript identity.
"""
import argparse
import json
import time

import numpy as np
import torch

from common import OUT, WINDOWS, greedy_text
from wrapper import V1Window, build  # noqa: F401

VARIANTS = {"faithful": {}, "fp16safe": {"rms_scale": 32.0, "pad_fill": 1.0}}


def run(model, mel, n_valid):
    x = torch.zeros(1, model.t_mel, 128)
    x[0, :len(mel)] = torch.from_numpy(mel)
    with torch.no_grad():
        lp, l11 = model(x, torch.tensor([n_valid], dtype=torch.int32))
    n = -(-n_valid // 4)
    return lp[0, :n].numpy(), l11[0, :n].numpy()


def cos_rows(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="faithful", choices=VARIANTS)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(8)
    ref = OUT / "reference/fast"
    utts = json.loads((ref / "index.json").read_text())["utts"]
    utts = utts[: args.limit] if args.limit else utts
    mel, l11r, lpr = (np.load(ref / f, mmap_mode="r") for f in ("mel.npy", "l11.npy", "logprobs.npy"))
    sizes = sorted(WINDOWS.values())
    from common import load_head_cpu, load_tower_fp32
    base, (head, ck) = load_tower_fp32(), load_head_cpu()
    stats = {}
    t0 = time.time()
    for w in sizes:
        model = V1Window(base.audio_tower, head, ck["mean"], ck["std"], w, **VARIANTS[args.variant]).eval()
        bigger = [s for s in sizes if s < w]
        for u in utts:
            fits = u["mel_frames"] <= w
            smallest = fits and all(u["mel_frames"] > s for s in bigger)
            second = fits and not smallest and len([s for s in bigger if u["mel_frames"] <= s]) == 1
            if not (smallest or second):
                continue
            m = np.asarray(mel[u["mel_off"]:u["mel_off"] + u["mel_frames"]])
            lp, l11 = run(model, m, u["mel_valid"])
            rl, rp = np.asarray(l11r[u["off"]:u["off"] + u["frames"]]), np.asarray(lpr[u["off"]:u["off"] + u["frames"]])
            assert lp.shape[0] == u["frames"], (u["id"], lp.shape, u["frames"])
            key = (w, "smallest" if smallest else "padded")
            s = stats.setdefault(key, {"utts": 0, "cos": [], "l11_maxabs": 0.0, "lp_maxabs": 0.0, "same_greedy": 0, "nonfinite": 0})
            c = cos_rows(l11, rl)
            s["utts"] += 1
            s["cos"].append(c)
            s["l11_maxabs"] = max(s["l11_maxabs"], float(np.abs(l11 - rl).max()))
            s["lp_maxabs"] = max(s["lp_maxabs"], float(np.abs(lp - rp).max()))
            s["same_greedy"] += greedy_text(lp) == u["greedy"]
            s["nonfinite"] += int(not (np.isfinite(lp).all() and np.isfinite(l11).all()))
        del model
        print(f"[check] window {w} done, {time.time() - t0:.0f} s", flush=True)
    print(f"variant {args.variant}: {VARIANTS[args.variant]}")
    for (w, kind), s in sorted(stats.items()):
        c = np.concatenate(s["cos"])
        print(f"  window {w:4d} ({kind:8s}) utts {s['utts']:3d} | L11 cos mean {c.mean():.8f} min {c.min():.8f} | "
              f"L11 max|d| {s['l11_maxabs']:.2e} | logprob max|d| {s['lp_maxabs']:.2e} | greedy identical "
              f"{s['same_greedy']}/{s['utts']} | non-finite {s['nonfinite']}")


if __name__ == "__main__":
    main()
