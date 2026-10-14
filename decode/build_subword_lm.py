"""Build a KenLM n-gram LM over SentencePiece-256 piece ids (words "t0".."t255").

Pipeline (each step restartable; finished outputs are reused unless --force):
  1. librispeech-lm-norm.txt.gz -> normalize like probe.norm_text (lowercase, collapse whitespace)
     -> sp.encode -> one line per sentence of "t{id}" tokens            (<work>/tokens.txt)
  2. lmplz -o ORDER --prune ...                                          (<work>/lm.arpa)
  3. build_binary                                                        (--out)
The ARPA is deleted after a successful binary build unless --keep-arpa.

Run with ~/.venv-gemma-stt/bin/python (needs sentencepiece only).
"""
import argparse
import gzip
import itertools
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import sentencepiece as spm

HOME = Path.home()
KENLM_BIN = HOME / "gemma_stt/tools/kenlm/build/bin"


def norm_text(t):  # identical to probe.norm_text
    return " ".join(t.lower().split())


def child_peak_rss_mb():
    # macOS: ru_maxrss in bytes; max over all waited-for children
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20


def tokenize(corpus, out, sp, max_lines, threads, batch=20000):
    t0 = time.time()
    n_lines = n_tok = n_unk = 0
    tmp = out.with_suffix(".partial")
    names = [f"t{i}" for i in range(sp.get_piece_size())]
    with gzip.open(corpus, "rt", encoding="utf-8") if corpus.suffix == ".gz" else open(corpus, encoding="utf-8") as f, \
            open(tmp, "w", encoding="utf-8") as w:
        src = f if max_lines is None else itertools.islice(f, max_lines)
        while True:
            lines = [t for t in (norm_text(x) for x in itertools.islice(src, batch)) if t]
            if not lines:
                break
            for ids in sp.encode(lines, num_threads=threads):
                w.write(" ".join(names[i] for i in ids))
                w.write("\n")
                n_tok += len(ids)
                n_unk += ids.count(0)
            n_lines += len(lines)
            if n_lines % 1_000_000 < batch:
                print(f"[tok] {n_lines:,} lines {n_tok:,} tokens {time.time() - t0:.0f}s", flush=True)
    tmp.rename(out)
    dt = time.time() - t0
    print(f"[tok] done: {n_lines:,} lines, {n_tok:,} tokens ({n_tok / max(n_lines, 1):.1f}/line), "
          f"<unk> {n_unk}, {out.stat().st_size / 2**20:.1f} MiB, {dt:.1f}s wall", flush=True)


def run(cmd, log_label):
    print(f"[{log_label}] $ {' '.join(map(str, cmd))}", flush=True)
    t0 = time.time()
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    subprocess.run([str(c) for c in cmd], check=True)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(f"[{log_label}] {time.time() - t0:.1f}s wall, {r1.ru_utime - r0.ru_utime + r1.ru_stime - r0.ru_stime:.1f}s CPU, "
          f"peak child RSS so far {child_peak_rss_mb():.0f} MiB", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=HOME / "gemma_stt/decode/lm/src/librispeech-lm-norm.txt.gz")
    ap.add_argument("--spm", type=Path, default=HOME / "gemma_stt/feats/full/spm_unigram256.model")
    ap.add_argument("--out", type=Path, default=HOME / "gemma_stt/decode/lm/sp256_6gram.bin")
    ap.add_argument("--work", type=Path, default=HOME / "gemma_stt/decode/lm/sp256_work")
    ap.add_argument("--max-lines", type=int, default=None, help="only the first N corpus lines (testing)")
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--prune", default="0 0 1 1 2 2",
                    help="lmplz --prune count thresholds per order (n-grams with count <= t are dropped)")
    ap.add_argument("--mem", default="4G", help="lmplz -S memory budget")
    ap.add_argument("--threads", type=int, default=2, help="sentencepiece encode threads")
    ap.add_argument("--binary-type", default="trie", choices=["trie", "probing"])
    ap.add_argument("--quant", default="-q 8 -b 8", help="build_binary quantization flags for trie ('' = none)")
    ap.add_argument("--discount-fallback", action="store_true", help="needed for tiny test corpora")
    ap.add_argument("--keep-arpa", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo steps even if outputs exist")
    a = ap.parse_args()

    a.work.mkdir(parents=True, exist_ok=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tokens, arpa = a.work / "tokens.txt", a.work / "lm.arpa"
    t_all = time.time()

    if a.force or not tokens.exists():
        sp = spm.SentencePieceProcessor(model_file=str(a.spm))
        tokenize(a.corpus, tokens, sp, a.max_lines, a.threads)
    else:
        print(f"[tok] reusing {tokens}", flush=True)

    if a.force or not arpa.exists():
        cmd = [KENLM_BIN / "lmplz", "-o", a.order, "-S", a.mem, "-T", a.work, "--text", tokens,
               "--arpa", str(arpa) + ".partial", "--prune", *a.prune.split()]
        if a.discount_fallback:
            cmd.append("--discount_fallback")
        run(cmd, "lmplz")
        os.rename(str(arpa) + ".partial", arpa)
        print(f"[lmplz] ARPA {arpa.stat().st_size / 2**20:.1f} MiB", flush=True)
    else:
        print(f"[lmplz] reusing {arpa}", flush=True)

    cmd = [KENLM_BIN / "build_binary", "-T", a.work, "-S", a.mem]
    if a.binary_type == "trie":
        cmd += a.quant.split()
    cmd += [a.binary_type, arpa, str(a.out) + ".partial"]
    run(cmd, "build_binary")
    os.rename(str(a.out) + ".partial", a.out)
    print(f"[build_binary] {a.out} {a.out.stat().st_size / 2**20:.1f} MiB", flush=True)
    if not a.keep_arpa:
        arpa.unlink()
    print(f"[done] {time.time() - t_all:.1f}s total, peak child RSS {child_peak_rss_mb():.0f} MiB, "
          f"self RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
