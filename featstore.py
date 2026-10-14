"""Shared read/write helpers for the cached feature store.

Layout of a cache directory:
    index.json              utterance plan (sorted by duration), reused on resume
    meta_00000.jsonl ...    one line per utterance: id, text, frames, dur, split
    <tap>/shard_00000.pt    bf16: {"feats": [sum_T, D], "lens": [n]}
                            int8: {"q": [sum_T, D] int8, "scale": [sum_T] fp32, "lens": [n]}
A shard is complete once its meta file exists.
"""
import json
from pathlib import Path

import torch


def meta_files(d):
    return sorted(Path(d).expanduser().glob("meta_*.jsonl"))


def read_meta(d):
    rows = []
    for mp in meta_files(d):
        rows += [json.loads(line) for line in mp.read_text().splitlines() if line]
    return rows


def quantize_rows(x):
    """Per-frame absmax int8. x: [N, D] float -> (q int8 [N, D], scale fp32 [N])."""
    x = x.float()
    scale = x.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    q = torch.round(x / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale


def dequantize_rows(q, scale, dtype=torch.bfloat16):
    return (q.float() * scale.float()[:, None]).to(dtype)


def fake_quant_int8(x, chunk=1 << 18, dtype=torch.bfloat16):
    """Quantize -> dequantize, to measure what int8 storage would cost."""
    out = torch.empty(x.shape, dtype=dtype)
    for i in range(0, x.shape[0], chunk):
        q, s = quantize_rows(x[i:i + chunk])
        out[i:i + chunk] = dequantize_rows(q, s, dtype)
    return out


def shard_keys(d):
    """Keys ("00000", ...) of the complete shards in d, in order."""
    return [mp.stem.split("_", 1)[1] for mp in meta_files(d)]


def read_shard_meta(d, k):
    mp = Path(d).expanduser() / f"meta_{k}.jsonl"
    return [json.loads(line) for line in mp.read_text().splitlines() if line]


def load_shard(d, tap, k, dtype=torch.bfloat16):
    """One shard of one tap: (feats [sum_T, D] on CPU, lens list[int])."""
    obj = torch.load(Path(d).expanduser() / tap / f"shard_{k}.pt")
    if "q" in obj:
        feats = dequantize_rows(obj["q"], obj["scale"], dtype)
    else:
        feats = obj["feats"].to(dtype)
    return feats, obj["lens"].tolist()


def load_tap(d, tap, dtype=torch.bfloat16):
    """Returns (feats [sum_T, D] on CPU, lens list[int], offsets list[int], meta list)."""
    d = Path(d).expanduser()
    feats, lens, meta = [], [], []
    for k in shard_keys(d):
        f, n = load_shard(d, tap, k, dtype)
        feats.append(f)
        lens += n
        meta += read_shard_meta(d, k)
    if not feats:
        raise FileNotFoundError(f"no complete shards in {d}")
    assert len(meta) == len(lens), f"meta/shard mismatch in {d}"
    offsets = [0]
    for n in lens:
        offsets.append(offsets[-1] + n)
    return torch.cat(feats), lens, offsets, meta
