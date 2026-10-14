#!/usr/bin/env python3
"""
Small nonlinear causal CTC head on cached EmbeddingGemma 2 audio-tower features.

Standardized tap features -> Linear D->512 -> 4 pre-LayerNorm residual blocks of
    local multi-head self-attention (8 heads; each frame sees 64 past frames and R future frames)
    depthwise Conv1d (kernel 5; right padding min(R, 2): causal for R=0, centered for R>=2)
    FFN 512->2048->512 (GELU, dropout 0.1)
-> LayerNorm -> Linear 512 -> SentencePiece pieces + blank. CTC loss in fp32, greedy dev WER per epoch.
R applies per block, so the head's total lookahead is 4 * (R + min(R, 2)) frames of 40 ms.
--block-future F1 F2 F3 F4 sets the attention future frames per block instead (one run), and
--causal-conv makes every depthwise conv strictly causal; the total lookahead is printed at
start-up, measured from the actual attention masks and conv padding (--expect-lookahead N stops
before loading any data unless it is exactly N frames).
Training only: 2 time masks (<= 10 frames) and 2 feature masks (<= 64 dims) on standardized features.

Reuses probe.py's tokenizer, data loading, batching and WER code.
  python head.py --feats ~/gemma_stt/feats/probe --lookahead 0 4
  python head.py --feats ~/gemma_stt/feats/probe --lookahead 0 --epochs 20 --int8 --out DIR
  python head.py --feats ~/gemma_stt/feats/probe --block-future 0 0 0 4 --causal-conv --epochs 30
  python head.py --feats ~/gemma_stt/feats/full --stream     # training cache larger than RAM

Expects <feats>/train and <feats>/dev written by cache_feats.py; writes --out (default
<feats>/head_results): head_<run>.pt (best dev WER), ckpt_<run>_ep<N>.pt after every epoch
(model, optimizer, scheduler, all RNG states, best WER, history, feature stats) and a results
JSON. --resume continues each run from its latest checkpoint with every state the next epoch uses.
--plateau-patience N prints a [plateau] line after each epoch once dev WER has not improved for N
epochs; --stop-patience M ends the run after the epoch that makes M such epochs (a resumed run that
already stopped stays stopped). The best head is saved whenever dev WER improves.
"""
import argparse
import itertools
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

from featstore import fake_quant_int8, load_shard, read_meta, read_shard_meta, shard_keys
from probe import (ctc_device_for, evaluate, feat_stats, gather, load_split, make_batches, norm_text,
                   pick_device, targets, train_spm)


# ---------------------------------------------------------------- model

def local_mask(L, T, past, future):
    """[B, 1, T, T] bool, True = may attend: keys within [t - past, t + future] and not padding.
    The diagonal is always allowed so padded query rows are never fully masked (no NaN)."""
    t = torch.arange(T, device=L.device)
    rel = t[None, :] - t[:, None]                                    # key - query
    band = (rel >= -past) & (rel <= future)
    valid = t[None, :] < L[:, None]                                  # [B, T]
    return ((band[None] & valid[:, None, :]) | torch.eye(T, dtype=torch.bool, device=L.device))[:, None]


class Block(nn.Module):
    def __init__(self, d, heads, ff, kernel, conv_future, dropout):
        super().__init__()
        self.heads = heads
        self.ln_att = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln_conv = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, kernel, groups=d)
        self.conv_pad = (kernel - 1 - conv_future, conv_future)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(ff, d), nn.Dropout(dropout))

    def forward(self, x, mask, valid):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln_att(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # [B, H, T, D/H]
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        h = (self.ln_conv(x) * valid).transpose(1, 2)                 # zero padding: it must not leak in
        x = x + self.conv(F.pad(h, self.conv_pad)).transpose(1, 2)
        return x + self.ff(self.ln_ff(x))


def mask_reach(past, future):
    """Largest key - query offset that local_mask lets a frame attend to."""
    T = past + future + 2
    q, k = local_mask(torch.tensor([T]), T, past, future)[0, 0].nonzero(as_tuple=True)
    return int((k - q).max())


class CausalHead(nn.Module):
    def __init__(self, d_in, n_out, future, d=512, blocks=4, heads=8, ff=2048, kernel=5, past=64,
                 dropout=0.1, causal_conv=False):
        super().__init__()
        futures = list(future) if isinstance(future, (list, tuple)) else [future] * blocks
        if len(futures) != blocks:
            raise ValueError(f"need one future-frame count per block ({blocks}), got {futures}")
        self.past, self.futures = past, futures
        self.inp = nn.Linear(d_in, d)
        self.blocks = nn.ModuleList(
            Block(d, heads, ff, kernel, 0 if causal_conv else min(f, (kernel - 1) // 2), dropout)
            for f in futures)
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_out)
        self.reach = [(mask_reach(past, f), blk.conv_pad[1]) for f, blk in zip(futures, self.blocks)]
        self.lookahead = sum(a + c for a, c in self.reach)            # frames, from masks + padding

    def forward(self, x, L):
        T = x.shape[1]
        L = L.to(x.device)
        valid = (torch.arange(T, device=x.device)[None] < L[:, None])[..., None].to(x.dtype)
        masks = {f: local_mask(L, T, self.past, f) for f in set(self.futures)}
        h = self.inp(x)
        for blk, f in zip(self.blocks, self.futures):
            h = blk(h, masks[f], valid)
        return self.out(self.ln(h))


def spec_augment(x, L, gen, n_time=2, max_time=10, n_feat=2, max_feat=64):
    """Zero (= the mean, after standardizing) random time spans and feature bands, per utterance."""
    B, T, D = x.shape
    t, f = torch.arange(T), torch.arange(D)
    drop_t = torch.zeros(B, T, dtype=torch.bool)
    for _ in range(n_time):
        w = torch.randint(0, max_time + 1, (B,), generator=gen)
        s = (torch.rand(B, generator=gen) * (L - w + 1).clamp_min(1)).long()
        drop_t |= (t[None] >= s[:, None]) & (t[None] < (s + w)[:, None])
    drop_f = torch.zeros(B, D, dtype=torch.bool)
    for _ in range(n_feat):
        w = torch.randint(0, max_feat + 1, (B,), generator=gen)
        s = (torch.rand(B, generator=gen) * (D - w + 1)).long()
        drop_f |= (f[None] >= s[:, None]) & (f[None] < (s + w)[:, None])
    keep = ~(drop_t[:, :, None] | drop_f[:, None, :])
    return x * keep.to(x.device, x.dtype)


# ---------------------------------------------------------------- training data

class InMemory:
    """The whole training split on the device (default)."""

    def __init__(self, tr, max_frames):
        self.tr, self.max_frames = tr, max_frames
        self.dim, self.mean, self.std = tr["feats"].shape[1], tr["mean"], tr["std"]
        self.n_utts, self.n_ok = len(tr["all"]), len(tr["ok"])
        self.steps = len(make_batches(tr["lens"], tr["ok"], max_frames))

    def epoch(self, rng):
        for b in make_batches(self.tr["lens"], self.tr["ok"], self.max_frames, rng):
            yield self.tr, b


class ShardStream:
    """Training split streamed from disk, for caches larger than RAM: one shard on the device at a
    time while a background thread reads the next. Each epoch shuffles the shard order and the
    batches within each shard. cache_feats.py shards are random samples of utterances, so each
    shard spans the full length range (caches written before that layout are duration-banded:
    re-cache them before streaming)."""

    def __init__(self, d, tap, int8, sp, max_frames, device, stats=None):
        self.d, self.tap, self.int8 = Path(d).expanduser(), tap, int8
        self.max_frames, self.device = max_frames, device
        self.shards = []
        for k in shard_keys(self.d):
            meta = read_shard_meta(self.d, k)
            lens = [r["frames"] for r in meta]
            _, toks, ok = targets(sp, meta, lens)
            self.shards.append({"key": k, "lens": lens, "offs": [0, *itertools.accumulate(lens)],
                                "toks": toks, "ok": ok})
        if not self.shards:
            raise FileNotFoundError(f"no complete shards in {self.d}")
        self.n_utts = sum(len(s["lens"]) for s in self.shards)
        self.n_ok = sum(len(s["ok"]) for s in self.shards)
        self.steps = sum(len(make_batches(s["lens"], s["ok"], max_frames)) for s in self.shards)
        if stats is None:                                   # one pass, one shard at a time
            stats = feat_stats((self._read(s) for s in self.shards), chunk=1 << 15)   # ~0.3 GB float64 chunks
        mean, std = stats
        self.dim, self.mean, self.std = len(mean), mean.to(device), std.to(device)

    def _read(self, s):
        feats, lens = load_shard(self.d, self.tap, s["key"])
        assert lens == s["lens"], f"shard {s['key']}: lengths differ from its meta file"
        return fake_quant_int8(feats) if self.int8 else feats

    def epoch(self, rng):
        order = list(range(len(self.shards)))
        rng.shuffle(order)
        with ThreadPoolExecutor(1) as pool:
            nxt = pool.submit(self._read, self.shards[order[0]])
            for j, k in enumerate(order):
                feats = nxt.result().to(self.device)
                if j + 1 < len(order):
                    nxt = pool.submit(self._read, self.shards[order[j + 1]])
                s = self.shards[k]
                data = {"feats": feats, "offs": s["offs"], "lens": s["lens"], "toks": s["toks"],
                        "mean": self.mean, "std": self.std}
                for b in make_batches(s["lens"], s["ok"], self.max_frames, rng):
                    yield data, b
                del data, feats
                if self.device.type == "mps":
                    torch.mps.empty_cache()                         # shards differ in size: don't cache them


def run_name(R, args):
    name = f"F{'-'.join(map(str, R))}" if isinstance(R, tuple) else f"R{R}"
    return name + ("c" if args.causal_conv else "") + ("-int8" if args.int8 else "")


CONFIG_KEYS = ("tap", "stream", "vocab", "epochs", "lr", "wd", "warmup", "dropout", "max_frames", "seed")


def train_config(args):
    """Settings a resumed run must share with its checkpoint (the run name covers the architecture)."""
    return {k: getattr(args, k) for k in CONFIG_KEYS}


def checkpoints(out_dir, name):
    """End-of-epoch checkpoints of a run, oldest first."""
    return sorted(out_dir.glob(f"ckpt_{name}_ep*.pt"), key=lambda p: int(p.stem.rsplit("_ep", 1)[1]))


def save_atomic(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------- train

def train_head(R, train, dv, sp, V, args, device, ctc_dev, out_dir):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    blank = V
    model = CausalHead(train.dim, V + 1, future=R, dropout=args.dropout, causal_conv=args.causal_conv).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    steps_per_epoch = train.steps
    total = steps_per_epoch * args.epochs
    warm = args.warmup
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    name = run_name(R, args)
    print(f"\n=== {name}  ({n_params:,} params, lookahead {model.lookahead} frames = "
          f"{40 * model.lookahead} ms, {steps_per_epoch} steps/epoch, {total} steps)", flush=True)
    print(f"[lookahead] per block (attention future frames, conv right padding): {model.reach} "
          f"-> total {model.lookahead} frames = {40 * model.lookahead} ms", flush=True)
    best = (float("inf"), -1)
    history = []
    start = 1
    found = checkpoints(out_dir, name) if args.resume else []
    if found:
        ck = torch.load(found[-1], map_location="cpu")
        changed = {k: (ck["config"].get(k), v) for k, v in train_config(args).items() if ck["config"].get(k) != v}
        if changed:
            raise SystemExit(f"{found[-1]} was trained with other settings (saved, now): {changed}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        rng.setstate(ck["rng"])
        gen.set_state(ck["gen"])
        torch.set_rng_state(ck["torch_rng"])
        if device.type == "mps" and ck["mps_rng"] is not None:
            torch.mps.set_rng_state(ck["mps_rng"])
        best, history, start = tuple(ck["best"]), ck["history"], ck["epoch"] + 1
        print(f"[resume] {found[-1].name}: epochs 1-{ck['epoch']} done, best dev WER {100 * best[0]:.2f}% "
              f"(epoch {best[1]}); continuing at epoch {start}", flush=True)
    elif args.resume:
        print(f"[resume] no checkpoint for {name} in {out_dir}: starting at epoch 1", flush=True)
    stopped = False
    if args.stop_patience and history and (start - 1) - best[1] >= args.stop_patience:
        print(f"[stop] {name}: stopped earlier ({(start - 1) - best[1]} epochs without a dev WER improvement "
              f"after epoch {best[1]}); not training further", flush=True)
        start, stopped = args.epochs + 1, True
    for ep in range(start, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for d, b in train.epoch(rng):
            x, L = gather(d["feats"], d["offs"], d["lens"], b, d["mean"], d["std"], 0, device)
            x = spec_augment(x, L, gen)
            lp = model(x, L).float().log_softmax(-1).transpose(0, 1)  # [T, B, C], fp32
            tgt = torch.cat([d["toks"][i] for i in b])
            tl = torch.tensor([len(d["toks"][i]) for i in b])
            loss = F.ctc_loss(lp.to(ctc_dev), tgt.to(ctc_dev), L, tl, blank=blank, zero_infinity=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
        t_train = time.time() - t0
        model.eval()
        wer, blank_frac = evaluate(model, dv, dv["all"], sp, blank, 0, device, args.max_frames)
        t_eval = time.time() - t0 - t_train
        history.append({"epoch": ep, "loss": tot / n, "wer": wer, "blank": blank_frac,
                        "train_s": t_train, "eval_s": t_eval})
        flag = ""
        if wer < best[0]:
            best = (wer, ep)
            flag = "  *"
            torch.save({"head": model.state_dict(), "mean": train.mean.cpu(), "std": train.std.cpu(),
                        "arch": {"d_in": train.dim, "n_out": V + 1, "future": R,
                                 "causal_conv": args.causal_conv, "dropout": args.dropout},
                        "tap": args.tap, "vocab": V, "spm": args.spm_path},
                       out_dir / f"head_{name}.pt")
        print(f"  ep {ep:2d} | loss {tot / n:7.3f} | dev WER {100 * wer:6.2f}% | "
              f"blank {100 * blank_frac:5.1f}% | train {t_train:5.1f}s eval {t_eval:4.1f}s{flag}", flush=True)
        save_atomic({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "rng": rng.getstate(), "gen": gen.get_state(),
                     "torch_rng": torch.get_rng_state(),
                     "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
                     "best": best, "history": history, "mean": train.mean.cpu(), "std": train.std.cpu(),
                     "config": train_config(args)}, out_dir / f"ckpt_{name}_ep{ep}.pt")
        since = ep - best[1]
        if args.plateau_patience and since >= args.plateau_patience:
            print(f"[plateau] {name}: dev WER not improved for {since} epochs (best {100 * best[0]:.2f}% "
                  f"at epoch {best[1]})", flush=True)
        if args.stop_patience and since >= args.stop_patience:
            print(f"[stop] {name}: {since} epochs without a dev WER improvement, stopping after epoch {ep}; "
                  f"best head is head_{name}.pt (epoch {best[1]})", flush=True)
            stopped = True
            break
    return {"run": name, "params": n_params, "lookahead_frames": model.lookahead,
            "best_wer": best[0], "best_epoch": best[1], "stopped_early": stopped, "history": history}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True, help="dir with train/ and dev/ caches")
    ap.add_argument("--tap", default="L11")
    ap.add_argument("--out", default=None, help="heads + results JSON (default <feats>/head_results)")
    ap.add_argument("--lookahead", type=int, nargs="+", default=[0, 4], help="R values, one run each")
    ap.add_argument("--block-future", type=int, nargs="+", default=None,
                    help="attention future frames per block, one run (replaces --lookahead)")
    ap.add_argument("--causal-conv", action="store_true", help="every depthwise conv strictly causal")
    ap.add_argument("--int8", action="store_true", help="int8 round-trip on train and dev features")
    ap.add_argument("--stream", action="store_true", help="stream training shards from disk (caches > RAM)")
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-frames", type=int, default=12000, help="padded frames per batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect-lookahead", type=int, default=None,
                    help="stop before loading data unless every run's measured lookahead is this many frames")
    ap.add_argument("--resume", action="store_true",
                    help="continue each run from its latest end-of-epoch checkpoint in --out (if any)")
    ap.add_argument("--plateau-patience", type=int, default=None,
                    help="print a [plateau] line after each epoch once dev WER has not improved for N epochs")
    ap.add_argument("--stop-patience", type=int, default=None,
                    help="stop after the epoch that makes N epochs without a dev WER improvement")
    args = ap.parse_args()

    root = Path(args.feats).expanduser()
    out_dir = Path(args.out).expanduser() if args.out else root / "head_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [tuple(args.block_future)] if args.block_future else args.lookahead
    if args.expect_lookahead is not None:
        for R in runs:
            la = CausalHead(1, 2, future=R, causal_conv=args.causal_conv).lookahead
            if la != args.expect_lookahead:
                raise SystemExit(f"{run_name(R, args)}: lookahead {la} frames, expected {args.expect_lookahead}")
        print(f"[lookahead] {args.expect_lookahead} frames = {40 * args.expect_lookahead} ms for "
              f"{[run_name(R, args) for R in runs]}, as expected", flush=True)
    stats = None                      # resuming: the feature stats from the checkpoint (no stats pass)
    if args.resume:
        latest = [c for R in runs for c in checkpoints(out_dir, run_name(R, args))[-1:]]
        if latest:
            ck = torch.load(latest[0], map_location="cpu")
            stats = (ck["mean"], ck["std"])
    device = pick_device()
    ctc_dev = ctc_device_for(device)
    print(f"[setup] device {device}, CTC loss on {ctc_dev}", flush=True)

    texts = [norm_text(r["text"]) for r in read_meta(root / "train")]
    args.spm_path = str(root / f"spm_unigram{args.vocab}")
    sp = train_spm(texts, args.spm_path, args.vocab)
    V = sp.get_piece_size()

    print(f"[data] {'streaming' if args.stream else 'loading'} {args.tap}"
          f"{' (int8 round-trip)' if args.int8 else ''} ...", flush=True)
    if args.stream:
        train = ShardStream(root / "train", args.tap, args.int8, sp, args.max_frames, device, stats=stats)
    else:
        train = InMemory(load_split(root / "train", (args.tap,), args.int8, sp, device, stats=stats),
                         args.max_frames)
    dv = load_split(root / "dev", (args.tap,), args.int8, sp, device,
                    stats=(train.mean.cpu(), train.std.cpu()))
    print(f"[data] train {train.n_utts} utts ({train.n_utts - train.n_ok} skipped), "
          f"dev {len(dv['all'])} utts", flush=True)

    results = [train_head(R, train, dv, sp, V, args, device, ctc_dev, out_dir) for R in runs]

    print("\n=== summary (greedy, dev) ===")
    for r in sorted(results, key=lambda r: r["best_wer"]):
        print(f"  {r['run']:<6} {100 * r['best_wer']:6.2f}%  (epoch {r['best_epoch']}, {r['params']:,} params)")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out_dir / f"results_{stamp}.json").write_text(json.dumps(results, indent=1))
    print(f"\nresults + best heads -> {out_dir}")


if __name__ == "__main__":
    main()
