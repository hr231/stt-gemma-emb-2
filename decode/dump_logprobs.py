#!/usr/bin/env python3
"""Dump a trained head's CTC log-probabilities per split for the decoding study.

  python decode/dump_logprobs.py                                   # full_160ms head, all 4 splits, auto device
  python decode/dump_logprobs.py --head HEAD --feats DIR --splits dev-clean --device cpu --threads 2

Per split writes <out>/<split>.npy (float16 [sum_T, 257] natural-log log_softmax, blank = 256,
utterances in cache order) and <out>/<split>.json (index: vocab, tokenizer, per-utterance id, ref,
offset, frames, dur). The log_softmax is computed in fp32 from the same forward pass that
probe.decode uses; greedy_wer is probe.evaluate's WER from those logits (fp32 argmax),
greedy_wer_fp16 the same greedy decoding of the saved float16 array. Runs in the training venv.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import sentencepiece as spm  # noqa: E402
import torch  # noqa: E402

from eval_head import SPLITS, load_head  # noqa: E402
from featstore import read_meta  # noqa: E402
from probe import decode, edit_distance, greedy_collapse, load_split, pick_device  # noqa: E402

HOME = Path.home()


def dump_split(model, ck, sp, feats_dir, split, out_dir, device, max_frames):
    blank = ck["vocab"]
    path = feats_dir / split
    data = load_split(path, (ck["tap"],), False, sp, device, stats=(ck["mean"], ck["std"]))
    meta = read_meta(path)
    lens, offs, refs = data["lens"], data["offs"], data["refs"]
    assert len(meta) == len(lens)
    C = blank + 1

    tmp = out_dir / f"{split}.tmp.npy"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(offs[-1], C))

    cur = {"lp": None, "k": 0}

    def predict(x, L):                   # probe.decode takes argmax of these logits (fp32 path)
        logits = model(x, L)
        cur["lp"] = logits.float().log_softmax(-1).cpu().numpy()
        cur["k"] = 0
        return logits

    edits = words = blanks = frames = mismatch = 0
    for i, hyp, p in decode(predict, data, data["all"], sp, blank, 0, device, max_frames):
        k = cur["k"]
        cur["k"] += 1
        lp = cur["lp"][k, :lens[i]]
        assert lp.shape == (lens[i], C), (lp.shape, lens[i], C)
        mismatch += int((lp.argmax(-1) != np.asarray(p)).sum())
        arr[offs[i]:offs[i + 1]] = lp.astype(np.float16)
        # probe.evaluate, line for line
        blanks += sum(1 for t in p if t == blank)
        frames += len(p)
        ref = refs[i].split()
        edits += edit_distance(ref, hyp.split())
        words += len(ref)
    arr.flush()
    greedy_wer = edits / max(words, 1)

    edits16 = 0                          # greedy decoding of what was saved
    for i in data["all"]:
        ids = greedy_collapse(arr[offs[i]:offs[i + 1]].argmax(-1).tolist(), blank)
        edits16 += edit_distance(refs[i].split(), sp.decode(ids).split())
    greedy_wer16 = edits16 / max(words, 1)
    del arr
    os.replace(tmp, out_dir / f"{split}.npy")

    index = {"split": split, "head": str(ck["_path"]), "blank": blank,
             "vocab": [sp.id_to_piece(j) for j in range(blank)],
             "spm_model": ck["spm"] + ".model", "frame_ms": 40,
             "utts": [{"id": meta[i]["id"], "ref": refs[i], "offset": offs[i], "frames": lens[i],
                       "dur": meta[i]["dur"]} for i in data["all"]],
             "greedy_wer": greedy_wer, "greedy_wer_fp16": greedy_wer16,
             "words": words, "greedy_edits": edits, "greedy_edits_fp16": edits16,
             "blank_frac": blanks / max(frames, 1), "argmax_logits_vs_logsoftmax_mismatch_frames": mismatch,
             "tap": ck["tap"], "feats": str(path), "lookahead_frames": model.lookahead,
             "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmpj = out_dir / f"{split}.json.tmp"
    tmpj.write_text(json.dumps(index))
    os.replace(tmpj, out_dir / f"{split}.json")
    del data
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", default=str(HOME / "gemma_stt/runs/full_160ms/head_F0-0-0-4c.pt"))
    ap.add_argument("--feats", default=str(HOME / "gemma_stt/feats/full"), help="dir with one cache per split")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--out", default=str(HOME / "gemma_stt/decode/logprobs"))
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    ap.add_argument("--max-frames", type=int, default=12000, help="batches hold 2x this many frames (as eval)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    head_path = Path(args.head).expanduser()
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    model, ck = load_head(head_path, device)
    ck["_path"] = head_path
    sp = spm.SentencePieceProcessor(model_file=ck["spm"] + ".model")
    assert sp.get_piece_size() == ck["vocab"] and sp.id_to_piece(0) == "<unk>"
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[head] {head_path} | tap {ck['tap']} | {ck['vocab']} pieces + blank | device {device} | "
          f"tokenizer {ck['spm']}.model", flush=True)
    for split in args.splits:
        t0 = time.time()
        with torch.no_grad():
            ix = dump_split(model, ck, sp, Path(args.feats).expanduser(), split, out_dir, device, args.max_frames)
        frames = sum(u["frames"] for u in ix["utts"])
        print(f"  {split:<11} {len(ix['utts']):5d} utts {frames:8d} frames | greedy WER fp32 "
              f"{100 * ix['greedy_wer']:.4f}% | fp16 {100 * ix['greedy_wer_fp16']:.4f}% | "
              f"argmax mismatch {ix['argmax_logits_vs_logsoftmax_mismatch_frames']} frames | "
              f"{time.time() - t0:.1f}s -> {out_dir / split}.npy", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
