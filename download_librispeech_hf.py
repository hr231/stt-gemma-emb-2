#!/usr/bin/env python3
"""
LibriSpeech from the Hugging Face Hub (openslr/librispeech_asr, parquet), written out in the
OpenSLR layout that cache_feats.py reads:
    <data_root>/LibriSpeech/<split>/<speaker>/<chapter>/<id>.flac  (+ <speaker>-<chapter>.trans.txt)

The parquet files hold the original FLAC bytes and transcripts (byte-identical to the OpenSLR
archives, checked on dev-clean), and the Hub serves them far faster than openslr.org. Uses the
same .done_<split> markers as download_librispeech.sh, so either script can fill a data root.

  python download_librispeech_hf.py ~/gemma_stt/data train-other-500 dev-other test-clean test-other

Splits: train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other.
Needs pyarrow. Interrupted runs resume: finished parquet files are not downloaded again.
"""
import argparse
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = "openslr/librispeech_asr"
REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"   # pinned Hub commit
HF_SPLIT = {
    "train-clean-100": "all/train.clean.100", "train-clean-360": "all/train.clean.360",
    "train-other-500": "all/train.other.500", "dev-clean": "all/validation.clean",
    "dev-other": "all/validation.other", "test-clean": "all/test.clean", "test-other": "all/test.other",
}


def write_split(parquet_files, dst):
    """Parquet rows -> <dst>/<speaker>/<chapter>/<id>.flac and one trans.txt per chapter."""
    import pyarrow.parquet as pq
    chapters = defaultdict(list)
    for f in parquet_files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["id", "text", "audio"]):
            for uid, text, audio in zip(batch.column("id").to_pylist(), batch.column("text").to_pylist(),
                                        batch.column("audio").to_pylist()):
                spk, ch, _ = uid.split("-")
                d = dst / spk / ch
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{uid}.flac").write_bytes(audio["bytes"])
                chapters[(spk, ch)].append((uid, text))
    for (spk, ch), rows in chapters.items():
        (dst / spk / ch / f"{spk}-{ch}.trans.txt").write_text("".join(f"{u} {t}\n" for u, t in sorted(rows)))
    return sum(len(r) for r in chapters.values()), len(chapters)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root")
    ap.add_argument("splits", nargs="+", choices=sorted(HF_SPLIT))
    ap.add_argument("--workers", type=int, default=8, help="parallel file downloads")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")   # public dataset; a stale token would 401
    from huggingface_hub import snapshot_download

    root = Path(args.data_root).expanduser()
    stage = root / ".hf_parquet"
    for split in args.splits:
        if (root / f".done_{split}").exists():
            print(f"[skip] {split}", flush=True)
            continue
        t0 = time.time()
        print(f"[get] {split} from huggingface.co/datasets/{REPO} ({HF_SPLIT[split]})", flush=True)
        snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=stage,
                          allow_patterns=[f"{HF_SPLIT[split]}/*.parquet"], max_workers=args.workers)
        files = sorted((stage / HF_SPLIT[split]).glob("*.parquet"))
        gb = sum(f.stat().st_size for f in files) / 1e9
        t1 = time.time()
        print(f"[got] {len(files)} parquet files, {gb:.2f} GB in {t1 - t0:.0f}s ({gb * 1e3 / max(t1 - t0, 1e-9):.0f} MB/s)",
              flush=True)
        dst = root / "LibriSpeech" / split
        if dst.exists():                                   # partial earlier attempt (no .done marker)
            shutil.rmtree(dst)
        n, n_ch = write_split(files, dst)
        print(f"[write] {split}: {n} utterances, {n_ch} chapters in {time.time() - t1:.0f}s -> {dst}", flush=True)
        shutil.rmtree(stage / HF_SPLIT[split])
        (root / f".done_{split}").touch()
    print(f"done -> {root / 'LibriSpeech'}")


if __name__ == "__main__":
    sys.exit(main())
