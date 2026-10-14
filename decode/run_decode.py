#!/usr/bin/env python3
"""Decode one dumped split with one decoder, single-threaded, and record WER and cost.

  python decode/run_decode.py --decoder greedy --split dev-clean
  python decode/run_decode.py --decoder pyctc --split dev-clean --lm 4g --alpha 0.5 --beta 1.0 --beam 16
  python decode/run_decode.py --decoder subword --split dev-clean --lm sp6 --alpha 0.4 --beam 8 \
      --opts '{"merge": "sum"}' --limit 100

--lm takes a label (none, 3g, 4g, sp6) or a path. Decoders: greedy, subword (this folder), cbd
(dec_cbd.py: ctc-beam-decoder), pyctc (dec_pyctc.py: pyctcdecode); only the chosen one is imported,
torch never is, so it runs under either venv.

Writes <results-dir>/<tag>/hyps.jsonl and metrics.json and prints one summary line.
Metrics: WER (corpus: all word edits / all reference words), decode CPU seconds (process time
spent inside decode() only), x real time per core = audio seconds / decode CPU seconds, peak RSS
(ru_maxrss), and the decoder construction time, which includes loading the LM (lm_load_s).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
    os.environ[_v] = "false" if _v == "TOKENIZERS_PARALLELISM" else "1"

import argparse
import importlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wer import utt_errors  # noqa: E402

HOME = Path.home()
LOGPROBS = HOME / "gemma_stt/decode/logprobs"
RESULTS = HOME / "gemma_stt/decode/results"
LM_DIR = HOME / "gemma_stt/decode/lm"
LMS = {"none": None,
       "3g": LM_DIR / "3-gram.pruned.1e-7.lower.bin",
       "4g": LM_DIR / "4-gram.lower.bin",
       "sp6": LM_DIR / "sp256_6gram.bin"}
DECODERS = ("greedy", "cbd", "pyctc", "subword")


def resolve_lm(lm):
    """(label, path or None) for a label in LMS or a file path."""
    if lm is None or lm in LMS:
        label = lm or "none"
        return label, LMS[label]
    p = Path(lm).expanduser()
    return p.stem, p


class LogprobStore:
    """Read-only view of <split>.npy. Each utterance is memory-mapped on its own and unmapped
    after copying, so the store's pages do not accumulate in the decoder's resident set."""

    def __init__(self, path):
        self.path = Path(path)
        m = np.load(self.path, mmap_mode="r")
        self.dtype, self.shape, self.header = m.dtype, m.shape, m.offset
        del m

    def utt(self, offset, frames):
        """float32 [frames, C] copy of rows offset .. offset + frames."""
        if frames == 0:
            return np.zeros((0, self.shape[1]), np.float32)
        mm = np.memmap(self.path, dtype=self.dtype, mode="r", shape=(frames, self.shape[1]),
                       offset=self.header + offset * self.shape[1] * self.dtype.itemsize)
        x = np.array(mm, dtype=np.float32)
        del mm
        return x


def load_store(logprobs_dir, split):
    """(index dict, LogprobStore of <split>.npy)."""
    d = Path(logprobs_dir).expanduser()
    index = json.loads((d / f"{split}.json").read_text())
    return index, LogprobStore(d / f"{split}.npy")


# pyctcdecode's word-break handling: dec_pyctc fixes it per beam by default (fix_break True; False =
# stock pyctcdecode), ctc-beam-decoder reproduces the stock behaviour. None = not applicable.
FIX_BREAK_DEFAULT = {"pyctc": True, "cbd": False}


def fix_break(decoder, opts):
    return opts.get("fix_break", FIX_BREAK_DEFAULT.get(decoder))


def fmt_num(x):
    return f"{x:g}"


def default_tag(args, lm_label):
    tag = f"{args.split}_{args.decoder}_{lm_label}_a{fmt_num(args.alpha)}_b{fmt_num(args.beta)}_beam{args.beam}"
    if args.opts:
        tag += "_" + "_".join(f"{k}-{v}" for k, v in sorted(json.loads(args.opts).items())
                              if not isinstance(v, (list, dict)) and "/" not in str(v))
    if args.limit:
        tag += f"_n{args.limit}"
    return tag


def peak_rss_bytes():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def package_versions(names):
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            pass
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoder", choices=DECODERS, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--logprobs-dir", default=str(LOGPROBS))
    ap.add_argument("--lm", default=None, help=f"label ({', '.join(LMS)}) or path")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N utterances only (0 = all)")
    ap.add_argument("--opts", default="", help="JSON dict of extra decoder options")
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--tag", default=None, help="results subfolder (default: built from the settings)")
    args = ap.parse_args(argv)

    opts = json.loads(args.opts) if args.opts else {}
    lm_label, lm_path = resolve_lm(args.lm)
    if lm_path is not None and not lm_path.exists():
        raise SystemExit(f"LM not found: {lm_path}")
    index, store = load_store(args.logprobs_dir, args.split)
    utts = index["utts"][:args.limit] if args.limit else index["utts"]

    mod = importlib.import_module(f"dec_{args.decoder}")
    t0 = time.perf_counter()
    dec = mod.make_decoder(index["vocab"], index["blank"], index["spm_model"],
                           lm=None if lm_path is None else str(lm_path),
                           alpha=args.alpha, beta=args.beta, beam=args.beam, **opts)
    lm_load_s = time.perf_counter() - t0
    rss_load = peak_rss_bytes()

    tag = args.tag or default_tag(args, lm_label)
    out_dir = Path(args.results_dir).expanduser() / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    edits = words = 0
    audio_s = cpu_s = wall_s = 0.0
    with open(out_dir / "hyps.jsonl.tmp", "w") as f:
        for u in utts:
            x = store.utt(u["offset"], u["frames"])
            c0, w0 = time.process_time(), time.perf_counter()
            hyp = dec.decode(x)
            cpu_s += time.process_time() - c0
            wall_s += time.perf_counter() - w0
            e, n = utt_errors(u["ref"], hyp)
            edits += e
            words += n
            audio_s += u["dur"]
            f.write(json.dumps({"id": u["id"], "ref": u["ref"], "hyp": hyp, "edits": e, "words": n}) + "\n")
    os.replace(out_dir / "hyps.jsonl.tmp", out_dir / "hyps.jsonl")

    wer = edits / max(words, 1)
    m = {"tag": tag, "decoder": args.decoder, "split": args.split, "lm": lm_label,
         "lm_path": None if lm_path is None else str(lm_path),
         "alpha": args.alpha, "beta": args.beta, "beam": args.beam, "opts": opts,
         "limit": args.limit, "n_utts": len(utts), "n_utts_split": len(index["utts"]),
         "words": words, "edits": edits, "wer": wer,
         "audio_s": audio_s, "decode_cpu_s": cpu_s, "decode_wall_s": wall_s,
         "xrt_per_core": audio_s / cpu_s if cpu_s > 0 else float("inf"),
         "peak_rss_bytes": peak_rss_bytes(), "peak_rss_after_load_bytes": rss_load, "lm_load_s": lm_load_s,
         "fix_break": fix_break(args.decoder, opts),
         "greedy_wer": index.get("greedy_wer"), "greedy_wer_fp16": index.get("greedy_wer_fp16"),
         "head": index.get("head"), "logprobs": str(Path(args.logprobs_dir).expanduser() / f"{args.split}.npy"),
         "python": platform.python_version(), "executable": sys.executable,
         "packages": package_versions(["numpy", "sentencepiece", "kenlm", "pyctcdecode", "ctc-beam-decoder",
                                       "ctc_beam_decoder"]),
         "machine": platform.machine(), "platform": platform.platform(),
         "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=1))
    print(f"{tag} | {len(utts)} utts {words} words | WER {100 * wer:.2f}% | "
          f"{m['xrt_per_core']:.1f}x RT/core ({cpu_s:.1f} cpu s for {audio_s:.0f} s audio) | "
          f"RSS {m['peak_rss_bytes'] / 2**20:.0f} MB | load {lm_load_s:.2f}s", flush=True)
    return m


if __name__ == "__main__":
    main()
