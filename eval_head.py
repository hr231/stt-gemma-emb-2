#!/usr/bin/env python3
"""
Score a saved head.py head (head_<run>.pt) on cached splits, each separately: greedy CTC decoding
and corpus-level WER (all word edits / all reference words of the split).

  python eval_head.py --head ~/gemma_stt/runs/full_160ms/head_F0-0-0-4c.pt --feats ~/gemma_stt/feats/full
  python eval_head.py --head HEAD --feats DIR --splits dev-clean test-other --device cpu

Uses the head's own training mean/std, tap and tokenizer; int8 caches are dequantized on load.
Writes eval_<head>_<time>.json next to the head.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sentencepiece as spm
import torch

from head import CausalHead
from probe import evaluate, load_split, pick_device

SPLITS = ("dev-clean", "dev-other", "test-clean", "test-other")


def load_head(path, device):
    ck = torch.load(path, map_location="cpu")
    a = ck["arch"]
    future = tuple(a["future"]) if isinstance(a["future"], (list, tuple)) else a["future"]
    model = CausalHead(a["d_in"], a["n_out"], future=future, dropout=a["dropout"],
                       causal_conv=a.get("causal_conv", False))
    model.load_state_dict(ck["head"])
    return model.to(device).eval(), ck


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", required=True, help="head_<run>.pt written by head.py")
    ap.add_argument("--feats", required=True, help="dir holding one cache folder per split")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--max-frames", type=int, default=12000, help="eval batches hold 2x this many frames")
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = ap.parse_args()

    head_path = Path(args.head).expanduser()
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    model, ck = load_head(head_path, device)
    sp = spm.SentencePieceProcessor(model_file=ck["spm"] + ".model")
    blank = ck["vocab"]
    print(f"[head] {head_path} | lookahead {model.lookahead} frames = {40 * model.lookahead} ms | "
          f"tap {ck['tap']} | {sp.get_piece_size()} pieces + blank | device {device}", flush=True)

    rows = []
    for split in args.splits:
        t0 = time.time()
        data = load_split(Path(args.feats).expanduser() / split, (ck["tap"],), False, sp, device,
                          stats=(ck["mean"], ck["std"]))
        words = sum(len(r.split()) for r in data["refs"])
        wer, blank_frac = evaluate(model, data, data["all"], sp, blank, 0, device, args.max_frames)
        rows.append({"split": split, "utts": len(data["all"]), "words": words, "wer": wer, "blank": blank_frac})
        print(f"  {split:<11} {len(data['all']):5d} utts {words:7d} words | WER {100 * wer:6.2f}% | "
              f"blank {100 * blank_frac:5.1f}% | {time.time() - t0:5.1f}s", flush=True)
        del data
        if device.type == "mps":
            torch.mps.empty_cache()

    out = head_path.parent / f"eval_{head_path.stem}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"head": str(head_path), "lookahead_frames": model.lookahead,
                               "feats": str(Path(args.feats).expanduser()), "results": rows}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
