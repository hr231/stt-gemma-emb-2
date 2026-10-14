#!/usr/bin/env python3
"""
Train the SentencePiece unigram tokenizer on every transcript of a cached training split, with the
settings probe.py / head.py use (text lowercased and whitespace-normalized; unigram, character
coverage 1.0, <unk> only). Writes <feats>/spm_unigram<vocab>.model (head.py then loads it instead of
training its own) and refuses to overwrite an existing model.

  python train_tokenizer.py --feats ~/gemma_stt/feats/full --compare ~/gemma_stt/feats/probe/spm_unigram256.model
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import sentencepiece as spm

from featstore import read_meta
from probe import norm_text, train_spm


def report(label, sp, texts):
    n = [len(sp.encode(t)) for t in texts]
    unk = sum(sp.encode(t).count(sp.unk_id()) for t in texts)
    chars = sum(len(t) for t in texts)
    print(f"  {label:<44} {sp.get_piece_size():4d} pieces | {statistics.mean(n):6.1f} tokens/utt "
          f"(median {statistics.median(n):.0f}) | {chars / sum(n):.2f} chars/token | <unk> {unk}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True, help="dir with train/ written by cache_feats.py")
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--out", default=None, help="model prefix (default <feats>/spm_unigram<vocab>)")
    ap.add_argument("--compare", nargs="*", default=[], help="other .model files to score on the same text")
    args = ap.parse_args()

    feats = Path(args.feats).expanduser()
    prefix = args.out or str(feats / f"spm_unigram{args.vocab}")
    if Path(prefix + ".model").exists():
        sys.exit(f"{prefix}.model exists; not overwriting")
    texts = [norm_text(r["text"]) for r in read_meta(feats / "train")]
    print(f"[text] {len(texts)} transcripts from {feats / 'train'}", flush=True)
    t0 = time.time()
    sp = train_spm(texts, prefix, args.vocab)
    print(f"[spm] trained in {time.time() - t0:.0f}s -> {prefix}.model", flush=True)
    print(f"[tokens] on these {len(texts)} transcripts:")
    report(f"new  {Path(prefix).name}", sp, texts)
    for m in args.compare:
        report(f"compare  {m}", spm.SentencePieceProcessor(model_file=str(Path(m).expanduser())), texts)


if __name__ == "__main__":
    main()
