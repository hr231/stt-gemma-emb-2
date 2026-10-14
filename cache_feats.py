#!/usr/bin/env python3
"""
Cache frozen EmbeddingGemma 2 audio-tower features on Apple silicon (MPS).

Reads LibriSpeech in the OpenSLR layout (see download_librispeech.sh), decodes audio
and computes mel features in parallel worker processes, runs the audio tower on the
GPU, and writes resumable shards (format: featstore.py). Each shard holds a random sample of
the chosen utterances (shuffled with --seed), sorted by duration inside the shard, so every
shard spans the full length range and batches inside a shard need little padding.

Examples
  # 20 h probe set, three taps, with a one-batch MPS-vs-CPU-fp32 check first
  caffeinate -dims python cache_feats.py --splits train-clean-100 --hours 20 \
      --taps L6 L9 L11 --out ~/gemma_stt/feats/probe/train --verify
  caffeinate -dims python cache_feats.py --splits dev-clean \
      --taps L6 L9 L11 --out ~/gemma_stt/feats/probe/dev

  # full 960 h, best layer only, int8 (~89 GB)
  caffeinate -dims python cache_feats.py \
      --splits train-clean-100 train-clean-360 train-other-500 \
      --taps L11 --int8 --out ~/gemma_stt/feats/full/train

  # the plan only (shard durations, padding overhead): no model, nothing written
  python cache_feats.py --splits train-clean-100 --plan-only

Interrupted runs resume: rerun the same command and finished shards are skipped.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F

from featstore import quantize_rows

MODEL_ID = "google/embeddinggemma-2"
SR = 16000
FPS = 25
VALID_TAPS = {f"L{i}" for i in range(12)}   # L11 = the 12th (last) conformer layer


# ---------------------------------------------------------------- data plan

LAYOUT = "random-shards"   # recorded in index.json; older caches used one global duration sort


def read_utts(data_root, splits):
    """Every utterance of the splits: id, transcript, flac path, split, duration in seconds."""
    import soundfile as sf
    rows = []
    for split in splits:
        root = Path(data_root).expanduser() / "LibriSpeech" / split
        if not root.is_dir():
            sys.exit(f"missing {root}  ->  bash download_librispeech.sh {data_root} {split}")
        n0 = len(rows)
        for t in sorted(root.glob("*/*/*.trans.txt")):
            for line in t.read_text().splitlines():
                if not line.strip():
                    continue
                uid, text = line.split(" ", 1)
                rows.append({"id": uid, "text": text.strip(),
                             "path": str(t.parent / f"{uid}.flac"), "split": split})
        print(f"[index] {split}: {len(rows) - n0} utterances", flush=True)

    print(f"[index] reading {len(rows)} audio headers ...", flush=True)
    with ThreadPoolExecutor(16) as ex:
        durs = list(ex.map(lambda r: sf.info(r["path"]).duration, rows, chunksize=256))
    for r, d in zip(rows, durs):
        r["dur"] = float(d)
    return rows


def choose(rows, max_hours, seed):
    """A random subset of max_hours (representative, not the shortest clips); all if None."""
    if max_hours is None:
        return list(rows)
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    keep, total = [], 0.0
    for r in rows:
        if total >= max_hours * 3600:
            break
        keep.append(r)
        total += r["dur"]
    return keep


def shard_layout(rows, shard_utts, seed):
    """Shuffle with seed, cut into shards of shard_utts, sort each shard by duration.
    Returns (utterances in shard order, shard sizes)."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    utts, sizes = [], []
    for i in range(0, len(rows), shard_utts):
        shard = sorted(rows[i:i + shard_utts], key=lambda r: r["dur"])
        utts += shard
        sizes.append(len(shard))
    return utts, sizes


def make_plan(data_root, splits, max_hours, seed, shard_utts):
    return shard_layout(choose(read_utts(data_root, splits), max_hours, seed), shard_utts, seed)


def build_index(data_root, splits, max_hours, seed, shard_utts, out):
    """The plan from <out>/index.json when resuming, else a new one (saved there)."""
    key = {"layout": LAYOUT, "splits": splits, "max_hours": max_hours, "seed": seed,
           "shard_utts": shard_utts}
    idx_path = out / "index.json"
    if idx_path.exists():
        plan = json.loads(idx_path.read_text())
        found = {k: plan.get(k) for k in key}
        if found != key:
            sys.exit(f"{idx_path} was built with {found}, this run asks for {key}; "
                     "use a different --out")
        return plan["utts"], plan["shard_sizes"]
    utts, sizes = make_plan(data_root, splits, max_hours, seed, shard_utts)
    idx_path.write_text(json.dumps({**key, "shard_sizes": sizes, "utts": utts}))
    return utts, sizes


def plan_batches(utts, batch_seconds, max_batch):
    batches, cur, cur_max = [], [], 0.0
    for i, u in enumerate(utts):
        d = u["dur"]
        if cur and (len(cur) + 1 > max_batch or max(cur_max, d) * (len(cur) + 1) > batch_seconds):
            batches.append(cur)
            cur, cur_max = [], 0.0
        cur.append(i)
        cur_max = max(cur_max, d)
    if cur:
        batches.append(cur)
    return batches


def plan_shards(utts, sizes, batch_seconds, max_batch):
    """Batches planned inside each shard, never across shards.
    Returns (batches as utterance indices, shards as lists of batch indices)."""
    batches, shards, start = [], [], 0
    for n in sizes:
        first = len(batches)
        batches += [[start + j for j in b]
                    for b in plan_batches(utts[start:start + n], batch_seconds, max_batch)]
        shards.append(list(range(first, len(batches))))
        start += n
    return batches, shards


def padded_seconds(utts, batches):
    return sum(max(utts[j]["dur"] for j in b) * len(b) for b in batches)


def plan_report(utts, sizes, batches, batch_seconds, max_batch, n_shards=5):
    real = sum(u["dur"] for u in utts)
    flat = sorted(utts, key=lambda u: u["dur"])               # previous layout: one global sort
    old_pad = padded_seconds(flat, plan_batches(flat, batch_seconds, max_batch)) / real
    print(f"[plan] {len(utts)} utts, {real / 3600:.2f} h, {len(sizes)} shards, {len(batches)} batches")
    durs = [u["dur"] for u in flat]
    print(f"[plan] all utterances: min {durs[0]:.2f}s  median {statistics.median(durs):.2f}s  "
          f"max {durs[-1]:.2f}s")
    start = 0
    for k, n in enumerate(sizes[:n_shards]):
        d = [u["dur"] for u in utts[start:start + n]]           # sorted inside the shard
        print(f"[plan] shard {k}: {n} utts, min {d[0]:.2f}s  median {statistics.median(d):.2f}s  "
              f"max {d[-1]:.2f}s")
        start += n
    print(f"[plan] padding overhead (padded s / real s): {padded_seconds(utts, batches) / real:.4f} "
          f"this layout | {old_pad:.4f} previous layout (one global duration sort)")


# ------------------------------------------------- worker side (spawn-safe)

_FE = None


def _get_fe():
    global _FE
    if _FE is None:
        from transformers import AutoFeatureExtractor
        try:
            _FE = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        except Exception:
            from transformers import AutoProcessor
            _FE = AutoProcessor.from_pretrained(MODEL_ID).feature_extractor
    return _FE


def _worker_init(_):
    torch.set_num_threads(1)   # one process per core, no oversubscription
    _get_fe()


class BatchSet(torch.utils.data.Dataset):
    """Item k = one fully prepared batch: (batch_id, mel, mask, num_samples)."""

    def __init__(self, utts, batches, batch_ids):
        self.paths = [u["path"] for u in utts]
        self.batches = batches
        self.batch_ids = batch_ids

    def __len__(self):
        return len(self.batch_ids)

    def __getitem__(self, k):
        import soundfile as sf
        bi = self.batch_ids[k]
        wavs = []
        for j in self.batches[bi]:
            w, sr = sf.read(self.paths[j], dtype="float32", always_2d=False)
            if sr != SR or w.ndim != 1:
                raise ValueError(f"{self.paths[j]}: expected 16 kHz mono, got {sr} Hz {w.shape}")
            wavs.append(w)
        # truncation=False: otherwise the extractor cuts audio at 30 s while the
        # transcript stays complete -> silent CTC label/audio mismatch.
        f = _get_fe()(wavs, sampling_rate=SR, padding="longest",
                      truncation=False, return_tensors="pt")
        return bi, f["input_features"], f["input_features_mask"], [len(w) for w in wavs]


# ---------------------------------------------------------------- model side

def load_model(dtype, device):
    from transformers import AutoModel
    base = AutoModel.from_pretrained(MODEL_ID, dtype=dtype)
    for name in ("vision_tower", "language_model", "embed_vision"):
        if hasattr(base, name):
            delattr(base, name)
    return base.to(device).eval()


@torch.no_grad()
def run_taps(model, x, m, taps):
    store, hooks = {}, []
    for t in taps:
        def fn(mod, inp, out, name=t):
            store[name] = out[0] if isinstance(out, tuple) else out
        hooks.append(model.audio_tower.layers[int(t[1:])].register_forward_hook(fn))
    try:
        out = model.audio_tower(x, attention_mask=m)   # mask: True = valid (verified)
    finally:
        for h in hooks:
            h.remove()
    return store, out.attention_mask.sum(-1)


def verify(base, dtype, device, item, taps):
    """Compare MPS features against a CPU fp32 reference on one batch."""
    _, x, m, _ = item
    print("[verify] loading CPU fp32 reference ...", flush=True)
    ref = load_model(torch.float32, "cpu")
    a, la = run_taps(base, x.to(device, dtype), m.to(device), taps)
    b, lb = run_taps(ref, x.float(), m, taps)
    ok = True
    for t in taps:
        cos = F.cosine_similarity(a[t].float().cpu(), b[t].float(), dim=-1)   # [B, T]
        valid = torch.arange(cos.shape[1])[None] < lb.cpu()[:, None]
        c = cos[valid]
        print(f"[verify] {t}: cosine mean {c.mean():.5f}  min {c.min():.5f}", flush=True)
        ok &= bool(c.mean() > 0.99)
    if not torch.equal(la.cpu(), lb.cpu()):
        print("[verify] WARNING: frame counts differ between MPS and CPU", flush=True)
        ok = False
    del ref
    if not ok:
        sys.exit("[verify] MPS features disagree with the CPU reference; try --dtype fp32")
    print("[verify] OK", flush=True)


def write_shard(out, k, buf, lens, meta, taps, int8):
    lens_t = torch.tensor(lens, dtype=torch.long)
    for t in taps:
        x = torch.cat(buf[t])
        obj = {"lens": lens_t}
        if int8:
            obj["q"], obj["scale"] = quantize_rows(x)
        else:
            obj["feats"] = x
        p = out / t / f"shard_{k:05d}.pt"
        tmp = p.with_suffix(".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, p)
    mp = out / f"meta_{k:05d}.jsonl"          # written last = shard complete
    tmp = mp.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in meta))
    os.replace(tmp, mp)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="~/gemma_stt/data")
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--hours", type=float, default=None, help="random subset of this many hours")
    ap.add_argument("--taps", nargs="+", default=["L11"])
    ap.add_argument("--out", help="cache directory (required unless --plan-only)")
    ap.add_argument("--int8", action="store_true", help="store per-frame int8 (~half the size)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-seconds", type=float, default=240.0, help="padded audio per batch")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--shard-utts", type=int, default=1000)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--verify", action="store_true", help="check MPS vs CPU fp32 on one batch first")
    ap.add_argument("--seed", type=int, default=0, help="subset choice and shard assignment")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the shard plan (durations, padding overhead) and exit; no model, no files")
    args = ap.parse_args()

    bad = [t for t in args.taps if t not in VALID_TAPS]
    if bad:
        sys.exit(f"unknown taps {bad}; use L0..L11 (L11 = last conformer layer)")
    if args.plan_only:
        utts, sizes = make_plan(args.data_root, args.splits, args.hours, args.seed, args.shard_utts)
        batches, _ = plan_shards(utts, sizes, args.batch_seconds, args.max_batch)
        plan_report(utts, sizes, batches, args.batch_seconds, args.max_batch)
        return
    if not args.out:
        ap.error("--out is required")
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for t in args.taps:
        (out / t).mkdir(exist_ok=True)

    utts, sizes = build_index(args.data_root, args.splits, args.hours, args.seed, args.shard_utts, out)
    batches, shards = plan_shards(utts, sizes, args.batch_seconds, args.max_batch)
    done = {int(p.stem.split("_")[1]) for p in out.glob("meta_*.jsonl")}
    pending = [k for k in range(len(shards)) if k not in done]

    total_h = sum(u["dur"] for u in utts) / 3600
    pending_batches = [bi for k in pending for bi in shards[k]]
    todo_s = sum(utts[j]["dur"] for bi in pending_batches for j in batches[bi])
    bytes_per_frame = (1024 + 4) if args.int8 else 2048
    gb = total_h * 3600 * FPS * bytes_per_frame * len(args.taps) / 1e9
    print(f"[plan] {len(utts)} utts, {total_h:.2f} h, {len(batches)} batches, {len(shards)} shards "
          f"({len(done)} done) | ~{gb:.1f} GB for {args.taps}{' int8' if args.int8 else ''}", flush=True)
    if not pending:
        print("[plan] nothing to do")
        return

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        print("WARNING: MPS not available, running on CPU (slow)")
        device = torch.device("cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"[model] loading {MODEL_ID} ({args.dtype}) on {device} ...", flush=True)
    base = load_model(dtype, device)

    ds = BatchSet(utts, batches, pending_batches)
    if args.verify:
        verify(base, dtype, device, ds[0], args.taps)

    last_of_shard = {shards[k][-1]: k for k in pending}
    loader = torch.utils.data.DataLoader(
        ds, batch_size=None, shuffle=False, num_workers=args.workers,
        worker_init_fn=_worker_init if args.workers > 0 else None,
        prefetch_factor=4 if args.workers > 0 else None)

    buf = {t: [] for t in args.taps}
    buf_lens, buf_meta = [], []
    t0 = last = time.time()
    audio_done = 0.0
    n_warn = 0
    for bi, x, m, nsamp in loader:
        store, L = run_taps(base, x.to(device, dtype), m.to(device), args.taps)
        L = L.cpu().tolist()
        for t in args.taps:
            y = store[t].float().cpu()
            if not torch.isfinite(y).all():
                print(f"WARNING: non-finite values in {t} (batch {bi})", flush=True)
                y = torch.nan_to_num(y)
            for k, T in enumerate(L):
                buf[t].append(y[k, :T].to(torch.bfloat16))
        for k, (j, T) in enumerate(zip(batches[bi], L)):
            u = utts[j]
            exp = nsamp[k] / SR * FPS
            if abs(T - exp) > 3 and n_warn < 20:
                n_warn += 1
                print(f"WARNING: {u['id']} {u['dur']:.1f}s -> {T} frames (expected ~{exp:.0f})", flush=True)
            buf_lens.append(int(T))
            buf_meta.append({"id": u["id"], "text": u["text"], "frames": int(T),
                             "dur": u["dur"], "split": u["split"]})
            audio_done += nsamp[k] / SR
        if bi in last_of_shard:
            write_shard(out, last_of_shard[bi], buf, buf_lens, buf_meta, args.taps, args.int8)
            buf = {t: [] for t in args.taps}
            buf_lens, buf_meta = [], []
        now = time.time()
        if now - last > 30:
            last = now
            rt = audio_done / (now - t0)
            eta = (todo_s - audio_done) / max(rt, 1e-9)
            print(f"  {audio_done / 3600:7.2f} / {todo_s / 3600:.2f} h | {rt:4.0f}x real time | "
                  f"ETA {eta / 60:5.0f} min", flush=True)

    el = time.time() - t0
    print(f"[done] {audio_done / 3600:.2f} h in {el / 60:.1f} min = {audio_done / el:.0f}x real time "
          f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
