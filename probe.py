#!/usr/bin/env python3
"""
Linear CTC probes on cached EmbeddingGemma 2 audio-tower features.

Trains a SentencePiece unigram-256 tokenizer on the training transcripts, then one
linear CTC head per run spec, and reports greedy WER on the dev set after each epoch.

Run specs:  TAP[:option...]   options: int8 (simulate int8 storage), stackK (K future frames),
                              cat (L9 and L11 joined per frame; write it as L11:cat)
  python probe.py --feats ~/gemma_stt/feats/probe --runs L6 L9 L11 L11:int8
  python probe.py --feats ~/gemma_stt/feats/probe --runs L11:stack2 L11:stack4 L11:stack6
  python probe.py --feats ~/gemma_stt/feats/probe --runs L11:cat

Expects <feats>/train and <feats>/dev written by cache_feats.py.
"""
import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

from featstore import fake_quant_int8, load_tap, read_meta


# ---------------------------------------------------------------- helpers

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def norm_text(t):
    return " ".join(t.lower().split())


CAT_TAPS = ("L9", "L11")   # run option "cat": these taps joined per frame


def parse_spec(spec):
    tap, *opts = spec.split(":")
    cfg = {"tap": tap, "int8": False, "stack": 0, "name": spec}
    cat = False
    for o in opts:
        if o == "int8":
            cfg["int8"] = True
        elif o == "cat":
            cat = True
        elif o.startswith("stack") and o[5:].isdigit():
            cfg["stack"] = int(o[5:])
        else:
            raise SystemExit(f"unknown option '{o}' in run spec '{spec}'")
    if cat and tap != CAT_TAPS[-1]:
        raise SystemExit(f"option 'cat' joins {' and '.join(CAT_TAPS)}; write it as '{CAT_TAPS[-1]}:cat'")
    cfg["taps"] = CAT_TAPS if cat else (tap,)
    return cfg


def train_spm(texts, prefix, vocab):
    import sentencepiece as spm
    if not Path(prefix + ".model").exists():
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(texts), model_prefix=prefix, vocab_size=vocab,
            model_type="unigram", character_coverage=1.0,
            unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1, num_threads=8)
    return spm.SentencePieceProcessor(model_file=prefix + ".model")


def ctc_min_frames(ids):
    """Frames CTC needs: one per token plus a blank between repeated tokens."""
    return len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b)


def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def greedy_collapse(frame_ids, blank):
    out, prev = [], None
    for t in frame_ids:
        if t != blank and t != prev:
            out.append(t)
        prev = t
    return out


def make_batches(lens, idxs, max_frames, rng=None):
    order = sorted(idxs, key=lambda i: lens[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = lens[i]
        if cur and max(cur_max, L) * (len(cur) + 1) > max_frames:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    if rng is not None:
        rng.shuffle(batches)
    return batches


def feat_stats(xs, chunk=1 << 18):
    """Per-dimension mean/std in float64 on CPU, over one [N, D] tensor or an iterable of them
    (e.g. one shard at a time)."""
    s = ss = None
    n = 0
    for x in ([xs] if torch.is_tensor(xs) else xs):
        if s is None:
            s = torch.zeros(x.shape[1], dtype=torch.float64)
            ss = torch.zeros_like(s)
        for i in range(0, x.shape[0], chunk):
            c = x[i:i + chunk].double()
            s += c.sum(0)
            ss += (c * c).sum(0)
        n += x.shape[0]
    mean = s / n
    std = (ss / n - mean * mean).clamp_min(1e-10).sqrt()
    return mean.float(), std.float()


def gather(feats, offs, lens, b, mean, std, stack, device):
    xs = [feats[offs[i]:offs[i + 1]] for i in b]
    x = nn.utils.rnn.pad_sequence(xs, batch_first=True).float()
    x = (x - mean) / std
    L = torch.tensor([lens[i] for i in b])
    valid = (torch.arange(x.shape[1])[None] < L[:, None]).to(device)
    x = x * valid[..., None]                       # padding = 0 = the mean after standardizing
    if stack:                                      # lookahead: concat the next `stack` frames
        x = torch.cat([x] + [F.pad(x[:, j:], (0, 0, 0, j)) for j in range(1, stack + 1)], -1)
    return x, L


def ctc_device_for(device):
    if device.type != "mps":
        return device
    try:
        lp = torch.randn(10, 2, 5, device=device).log_softmax(-1).requires_grad_()
        loss = F.ctc_loss(lp, torch.tensor([1, 2, 1, 2, 3], device=device),
                          torch.tensor([10, 10]), torch.tensor([3, 2]), blank=4)
        loss.backward()
        return device
    except Exception as e:
        print(f"[ctc] not available on MPS ({type(e).__name__}); computing the loss on CPU")
        return torch.device("cpu")


# ---------------------------------------------------------------- train / eval

@torch.no_grad()
def decode(predict, data, idxs, sp, blank, stack, device, max_frames):
    """Greedy CTC decoding. predict(x [B, T, D], L [B]) -> logits [B, T, C].
    Yields (utterance index, hypothesis text, per-frame argmax ids)."""
    feats, offs, lens = data["feats"], data["offs"], data["lens"]
    for b in make_batches(lens, idxs, max_frames * 2):
        x, L = gather(feats, offs, lens, b, data["mean"], data["std"], stack, device)
        pred = predict(x, L).argmax(-1).cpu()
        for k, i in enumerate(b):
            p = pred[k, :lens[i]].tolist()
            yield i, sp.decode(greedy_collapse(p, blank)), p


def evaluate(predict, data, idxs, sp, blank, stack, device, max_frames):
    edits = words = blanks = frames = 0
    for i, hyp, p in decode(predict, data, idxs, sp, blank, stack, device, max_frames):
        blanks += sum(1 for t in p if t == blank)
        frames += len(p)
        ref = data["refs"][i].split()
        edits += edit_distance(ref, hyp.split())
        words += len(ref)
    return edits / max(words, 1), blanks / max(frames, 1)


def run_probe(cfg, tr, dv, sp, V, args, device, ctc_dev, out_dir):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    blank = V
    D = tr["feats"].shape[1] * (cfg["stack"] + 1)
    head = nn.Linear(D, V + 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)

    steps_per_epoch = len(make_batches(tr["lens"], tr["ok"], args.max_frames))
    total = steps_per_epoch * args.epochs
    warm = min(args.warmup, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    print(f"\n=== {cfg['name']}  (input {D}, vocab {V}+blank, {steps_per_epoch} steps/epoch)", flush=True)
    best = (float("inf"), -1)
    history = []
    for ep in range(1, args.epochs + 1):
        head.train()
        t0, tot, n = time.time(), 0.0, 0
        for b in make_batches(tr["lens"], tr["ok"], args.max_frames, rng):
            x, L = gather(tr["feats"], tr["offs"], tr["lens"], b, tr["mean"], tr["std"],
                          cfg["stack"], device)
            lp = head(x).float().log_softmax(-1).transpose(0, 1)        # [T, B, C], fp32
            tgt = torch.cat([tr["toks"][i] for i in b])
            tl = torch.tensor([len(tr["toks"][i]) for i in b])
            loss = F.ctc_loss(lp.to(ctc_dev), tgt.to(ctc_dev), L, tl,
                              blank=blank, zero_infinity=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
        head.eval()
        wer, blank_frac = evaluate(lambda x, L: head(x), dv, dv["all"], sp, blank, cfg["stack"],
                                   device, args.max_frames)
        history.append({"epoch": ep, "loss": tot / n, "wer": wer, "blank": blank_frac})
        flag = ""
        if wer < best[0]:
            best = (wer, ep)
            flag = "  *"
            torch.save({"head": head.state_dict(), "mean": tr["mean"].cpu(), "std": tr["std"].cpu(),
                        "cfg": cfg, "vocab": V, "spm": args.spm_path},
                       out_dir / f"head_{cfg['name'].replace(':', '_')}.pt")
        print(f"  ep {ep:2d} | loss {tot / n:7.3f} | dev WER {100 * wer:6.2f}% | "
              f"blank {100 * blank_frac:5.1f}% | {time.time() - t0:5.1f}s{flag}", flush=True)
    return {"run": cfg["name"], "best_wer": best[0], "best_epoch": best[1], "history": history}


def targets(sp, meta, lens):
    """Normalized references, token tensors, and the utterances CTC can train on
    (non-empty transcript that fits in the frames)."""
    refs = [norm_text(r["text"]) for r in meta]
    toks = [torch.tensor(sp.encode(r), dtype=torch.long) for r in refs]
    ok = [i for i in range(len(lens)) if lens[i] >= ctc_min_frames(toks[i].tolist()) and len(toks[i])]
    return refs, toks, ok


def load_split(path, taps, int8, sp, device, stats=None):
    parts = []
    for tap in taps:                 # several taps (run option "cat") -> joined per frame
        feats, lens, offs, meta = load_tap(path, tap)
        parts.append(fake_quant_int8(feats) if int8 else feats)
    feats = parts[0] if len(parts) == 1 else torch.cat(parts, 1)
    del parts
    mean, std = stats if stats is not None else feat_stats(feats)
    refs, toks, ok = targets(sp, meta, lens)
    return {"feats": feats.to(device), "lens": lens, "offs": offs, "refs": refs, "toks": toks,
            "ok": ok, "all": list(range(len(lens))),
            "mean": mean.to(device), "std": std.to(device)}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True, help="dir with train/ and dev/ caches")
    ap.add_argument("--train", default=None, help="override train cache dir")
    ap.add_argument("--dev", default=None, help="override dev cache dir")
    ap.add_argument("--runs", nargs="+", default=["L6", "L9", "L11", "L11:int8"])
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--max-frames", type=int, default=12000, help="padded frames per batch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.feats).expanduser()
    tr_dir = Path(args.train).expanduser() if args.train else root / "train"
    dv_dir = Path(args.dev).expanduser() if args.dev else root / "dev"
    out_dir = root / "probe_results"
    out_dir.mkdir(exist_ok=True)

    device = pick_device()
    ctc_dev = ctc_device_for(device)
    print(f"[setup] device {device}, CTC loss on {ctc_dev}", flush=True)

    texts = [norm_text(r["text"]) for r in read_meta(tr_dir)]
    args.spm_path = str(root / f"spm_unigram{args.vocab}")
    sp = train_spm(texts, args.spm_path, args.vocab)
    V = sp.get_piece_size()
    toks_per_utt = sum(len(sp.encode(t)) for t in texts) / max(len(texts), 1)
    print(f"[spm] {V} pieces, {toks_per_utt:.1f} tokens per training utterance", flush=True)

    by_tap = defaultdict(list)
    for spec in args.runs:
        cfg = parse_spec(spec)
        by_tap[(cfg["taps"], cfg["int8"])].append(cfg)

    results = []
    for (taps, int8), cfgs in by_tap.items():
        print(f"\n[data] loading {'+'.join(taps)}{' (int8 round-trip)' if int8 else ''} ...", flush=True)
        tr = load_split(tr_dir, taps, int8, sp, device)
        dv = load_split(dv_dir, taps, int8, sp, device, stats=(tr["mean"].cpu(), tr["std"].cpu()))
        skipped = len(tr["all"]) - len(tr["ok"])
        print(f"[data] train {len(tr['all'])} utts ({skipped} too short for their transcript, skipped), "
              f"dev {len(dv['all'])} utts", flush=True)
        for cfg in cfgs:
            results.append(run_probe(cfg, tr, dv, sp, V, args, device, ctc_dev, out_dir))
        del tr, dv
        if device.type == "mps":
            torch.mps.empty_cache()

    print("\n=== summary (greedy, dev) ===")
    for r in sorted(results, key=lambda r: r["best_wer"]):
        print(f"  {r['run']:<14} {100 * r['best_wer']:6.2f}%  (epoch {r['best_epoch']})")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out_dir / f"results_{stamp}.json").write_text(json.dumps(results, indent=1))
    print(f"\nresults + best heads -> {out_dir}")


if __name__ == "__main__":
    main()
