#!/usr/bin/env python3
"""
Diagnostics for a saved linear probe head and for the audio tower's lookahead.

  1. greedy WER on a random train subset (compare with dev: under- vs overfitting)
  2. dev examples (reference / hypothesis) and substitution / deletion / insertion counts
  3. lookahead: replace mel frames >= --noise-from with Gaussian noise, rerun the tower, and
     report the first output frame whose relative change exceeds --threshold

  python diag.py --feats ~/gemma_stt/feats/probe
"""
import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn

from probe import decode, evaluate, load_split, pick_device

MEL_PER_FRAME = 4   # 100 mel frames/s -> 25 output frames/s
MS_PER_FRAME = 40


def sdi_counts(ref, hyp):
    """(substitutions, deletions, insertions) of one minimum-edit alignment."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    s = dl = ins = 0
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            dl += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, dl, ins


def load_head(path, device):
    ck = torch.load(path)
    cfg = ck["cfg"]
    taps = tuple(cfg.get("taps", (cfg["tap"],)))
    w = ck["head"]["weight"]
    head = nn.Linear(w.shape[1], w.shape[0])
    head.load_state_dict(ck["head"])
    return head.to(device).eval(), cfg, taps, ck


def probe_diagnostics(args, device):
    import sentencepiece as spm
    root = Path(args.feats).expanduser()
    head, cfg, taps, ck = load_head(Path(args.head).expanduser() if args.head else
                                    root / "probe_results" / "head_L11.pt", device)
    sp = spm.SentencePieceProcessor(model_file=ck["spm"] + ".model")
    blank = ck["vocab"]
    stats = (ck["mean"], ck["std"])
    predict = lambda x, L: head(x)
    print(f"[head] {cfg['name']}  taps {taps}  vocab {blank}+blank", flush=True)

    tr = load_split(root / "train", taps, cfg["int8"], sp, device, stats=stats)
    sub = sorted(random.Random(args.seed).sample(tr["all"], min(args.train_subset, len(tr["all"]))))
    sub_h = sum(tr["lens"][i] for i in sub) / 25 / 3600
    wer_tr, blank_tr = evaluate(predict, tr, sub, sp, blank, cfg["stack"], device, args.max_frames)
    print(f"[A1] train subset: {len(sub)} utts ({sub_h:.2f} h) | greedy WER {100 * wer_tr:.2f}% | "
          f"blank {100 * blank_tr:.1f}%", flush=True)
    del tr

    dv = load_split(root / "dev", taps, cfg["int8"], sp, device, stats=stats)
    hyps = {i: h for i, h, _ in decode(predict, dv, dv["all"], sp, blank, cfg["stack"], device,
                                        args.max_frames)}
    S = D = I = N = 0
    for i, h in hyps.items():
        ref = dv["refs"][i].split()
        s, d, ins = sdi_counts(ref, h.split())
        S, D, I, N = S + s, D + d, I + ins, N + len(ref)
    print(f"[A1] dev: {len(hyps)} utts | greedy WER {100 * (S + D + I) / N:.2f}%", flush=True)
    print(f"[A2] dev reference words {N} | substitutions {S} ({100 * S / N:.2f}%) | "
          f"deletions {D} ({100 * D / N:.2f}%) | insertions {I} ({100 * I / N:.2f}%)", flush=True)
    for k, i in enumerate(sorted(random.Random(args.seed).sample(dv["all"], args.examples)), 1):
        ref = dv["refs"][i].split()
        s, d, ins = sdi_counts(ref, hyps[i].split())
        print(f"\n[A2 {k:2d}] {dv['lens'][i] / 25:.1f}s  S{s} D{d} I{ins} / {len(ref)} words")
        print(f"  REF: {dv['refs'][i]}")
        print(f"  HYP: {hyps[i]}", flush=True)
    del dv


@torch.no_grad()
def lookahead(args, device):
    import soundfile as sf
    from cache_feats import SR, _get_fe, load_model, run_taps

    utts = json.loads((Path(args.feats).expanduser() / "dev" / "index.json").read_text())["utts"]
    u = min((u for u in utts if u["dur"] >= args.clip_min_s), key=lambda u: u["dur"])   # shortest such clip
    w, sr = sf.read(u["path"], dtype="float32")
    f = _get_fe()([w], sampling_rate=SR, padding="longest", truncation=False, return_tensors="pt")
    x, m = f["input_features"], f["input_features_mask"]
    t0 = args.noise_from
    g = torch.Generator().manual_seed(args.seed)
    xn = x.clone()
    xn[:, t0:] = torch.randn(xn[:, t0:].shape, generator=g) * x.std() + x.mean()

    model = load_model(torch.bfloat16, device)
    run = lambda inp: run_taps(model, inp.to(device, torch.bfloat16), m.to(device), [args.tap])
    a, La = run(x)
    a2, _ = run(x)
    b, Lb = run(xn)
    a, a2, b = (t[args.tap][0].float().cpu() for t in (a, a2, b))
    norm = a.norm(dim=-1).clamp_min(1e-12)
    rel = (b - a).norm(dim=-1) / norm
    rel_rep = (a2 - a).norm(dim=-1) / norm
    out0 = t0 // MEL_PER_FRAME
    over = (rel > args.threshold).nonzero().flatten()
    first = int(over[0]) if len(over) else None
    print(f"\n[A3] clip {u['id']} {u['dur']:.2f}s | mel {tuple(x.shape)} -> {args.tap} {tuple(a.shape)} "
          f"(valid frames {int(La[0])}) | noise from mel frame {t0} = output frame {out0}", flush=True)
    print(f"[A3] repeat run, unchanged input: max relative change {rel_rep.max():.2e}", flush=True)
    lo = max(0, out0 - 8)
    for t in range(lo, min(out0 + 3, len(rel))):
        print(f"  frame {t:4d}: relative change {rel[t]:.3e}")
    if first is None:
        print(f"[A3] no frame exceeds {args.threshold:g}")
    else:
        print(f"[A3] first frame with relative change > {args.threshold:g}: {first} | "
              f"lookahead {(out0 - first) * MS_PER_FRAME} ms ({out0 - first} frames)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True, help="dir with train/ and dev/ caches")
    ap.add_argument("--head", default=None, help="default <feats>/probe_results/head_L11.pt")
    ap.add_argument("--train-subset", type=int, default=2700)
    ap.add_argument("--examples", type=int, default=15)
    ap.add_argument("--max-frames", type=int, default=12000)
    ap.add_argument("--tap", default="L11", help="tap for the lookahead test")
    ap.add_argument("--clip-min-s", type=float, default=8.0)
    ap.add_argument("--noise-from", type=int, default=400, help="first mel frame replaced by noise")
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    probe_diagnostics(args, device)
    if device.type == "mps":
        torch.mps.empty_cache()
    lookahead(args, device)


if __name__ == "__main__":
    main()
