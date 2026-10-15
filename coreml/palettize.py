#!/usr/bin/env python3
"""Phase 4 (internal experiment): 6-bit post-training palettization of V1's encoder (audio tower)
weights with grouped LUTs; the CTC head stays fp16.

  ~/.venv-coreml/bin/python coreml/palettize.py --groups 8 [16] [--windows 5s 10s 20s 35s] [--device mps]

Writes OUT/models/faithful/fp16_pal6_g<G>/v1_<window>.mlpackage: per_grouped_channel palettization, one
64-entry LUT per G output channels, minimum deployment target macOS 15 (grouped LUTs are the iOS18
constexpr_lut_to_dense; coremltools refuses per_grouped_channel below iOS18). See PALETTIZE.md.

Method: every tower weight that coremltools would compress by default (rank >= 2, > 2048 elements: sscp conv,
input projection, the 12 layers' linears and depthwise convs) is rounded to fp16 (the package's dtype) and
split into groups of G output channels (axis 0, as coremltools does for linear / conv). A group with
<= 64 distinct fp16 values is kept exactly; the others are clustered by batched 1-D k-means (Lloyd,
quantile init, empty clusters re-seeded at the worst-fit value) on torch MPS, the centroids rounded to
fp16 and the weights snapped to the nearest one. The snapped weights go into V1Window (wrapper.py), which
is traced and converted as in export.py (same inputs / outputs, FLOAT16), then
palettize_weights(mode="unique", granularity="per_grouped_channel", group_size=G) turns exactly those
consts into LUTs. Checked per tensor: palettized, <= 6 bits, the values used by every k-means group equal to
its fp16 palette, and the decompressed weight equal to the snapped fp16 weight. Head consts and the
attention constants are not touched.

macOS 15: the iOS18 frontend lowers F.scaled_dot_product_attention without a scale to the fused iOS18
scaled_dot_product_attention op, which makes the ANE compile of the whole program fail. The head's SDPA is
traced with an explicit scale = head_dim^-0.5 (mathematically the default), which coremltools decomposes
into matmul / softmax as for macOS 14.

Also writes <precision>/palettize.json (per-tensor statistics, k-means wall time / device) and appends one
JSON line per package to OUT/models/palettize_log.jsonl.
"""
import argparse
import contextlib
import json
import time

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F

from common import OUT, WINDOWS, check_v1_files, load_head_cpu, load_tower_fp32, sha256
from export import VARIANTS
from wrapper import V1Window

K = 64                       # 6-bit palettes
WEIGHT_THRESHOLD = 2048      # coremltools' default: smaller tensors are not compressed


def precision_name(G):
    return f"fp16_pal6_g{G}"


def mil_name(param_name):
    return param_name.replace(".", "_") + "_to_fp16"


def encoder_weights(model):
    """(name, parameter) of the tower weights coremltools compresses by default; never the head."""
    return [(f"{prefix}.{n}", p) for prefix, mod in (("sscp", model.sscp), ("layers", model.layers))
            for n, p in mod.named_parameters() if p.dim() >= 2 and p.numel() > WEIGHT_THRESHOLD]


# ---------------------------------------------------------------- k-means
def distinct_rows(S):
    """Sorted rows S [n_groups, n] -> (U [n_groups, m] distinct values ascending, padded with the row
    maximum; Wt their counts, 0 for padding; distinct count per row)."""
    new = torch.ones_like(S, dtype=torch.bool)
    new[:, 1:] = S[:, 1:] != S[:, :-1]
    uid = new.long().cumsum(1) - 1
    nuniq = uid[:, -1] + 1
    m = int(nuniq.max())
    U = S[:, -1:].expand(-1, m).contiguous().scatter_(1, uid, S)
    Wt = torch.zeros(S.shape[0], m, device=S.device).scatter_add_(1, uid, torch.ones_like(S))
    return U, Wt, nuniq


def kmeans_1d(U, Wt, k=K, iters=2000):
    """Batched weighted 1-D Lloyd over distinct values U [n_groups, m] (sorted rows) with counts Wt ->
    (centroids [n_groups, k] fp32, iterations). Init: k weighted quantiles; an empty cluster is re-seeded
    at the worst-fit value of its group; stops when no assignment changes."""
    rows = torch.arange(U.shape[0], device=U.device)
    cw = Wt.cumsum(1)
    q = (torch.arange(k, device=U.device) + 0.5) / k * cw[:, -1:]
    C = U.gather(1, torch.searchsorted(cw, q).clamp(max=U.shape[1] - 1))
    prev = None
    for it in range(iters):
        C, _ = C.sort(1)
        idx = torch.searchsorted((C[:, 1:] + C[:, :-1]) * 0.5, U)
        cnt = torch.zeros_like(C).scatter_add_(1, idx, Wt)
        new = torch.where(cnt > 0, torch.zeros_like(C).scatter_add_(1, idx, Wt * U) / cnt.clamp(min=1e-30), C)
        empty = cnt == 0
        has = empty.any(1)
        err = Wt * (U - C.gather(1, idx)) ** 2
        worst = U[rows, err.argmax(1)]
        first = empty.float().argmax(1)
        new[rows, first] = torch.where(has, worst, new[rows, first])
        C = new
        if it % 10 == 0:
            if prev is not None and not bool(has.any()) and torch.equal(idx, prev):
                break
            prev = idx
    return C.sort(1)[0], it + 1


def snap(W, G, device):
    """fp16-rounded W with every group of G output channels on <= K values.
    Returns (snapped fp32 tensor holding fp16 values, per group the sorted fp16 palette or None if the
    group was kept exactly, stats)."""
    X = W.detach().reshape(W.shape[0], -1).half().float()
    X = X.reshape(X.shape[0] // G, -1).to(device)
    ref = X.clone()
    U, Wt, nuniq = distinct_rows(X.sort(1)[0])
    lossy = nuniq > K
    iters = 0
    pal = [None] * X.shape[0]
    if lossy.any():
        C, iters = kmeans_1d(U[lossy], Wt[lossy])
        C = C.half().float().sort(1)[0]                          # LUT dtype: fp16
        X[lossy] = C.gather(1, torch.searchsorted((C[:, 1:] + C[:, :-1]) * 0.5, X[lossy]))
        for i, c in zip(torch.nonzero(lossy)[:, 0].tolist(), C.cpu().numpy().astype(np.float16)):
            pal[i] = np.unique(c)
    rel = lambda a, b: float(((a - b).norm() / b.norm()).item())
    stats = {"shape": list(W.shape), "groups": int(X.shape[0]), "max_distinct_fp16": int(nuniq.max()),
             "lossy_groups": int(lossy.sum()), "kmeans_iters": iters,
             "rel_err_lossy_groups": rel(X[lossy], ref[lossy]) if lossy.any() else 0.0,
             "rel_err_tensor": rel(X, ref)}
    return X.reshape(W.shape).cpu(), pal, stats


def snap_encoder(model, G, device):
    t0 = time.time()
    snapped, palettes, stats = {}, {}, {}
    for name, p in encoder_weights(model):
        snapped[name], palettes[name], stats[name] = snap(p, G, device)
    if device == "mps":
        torch.mps.synchronize()
    return snapped, palettes, stats, time.time() - t0


# ---------------------------------------------------------------- conversion
@contextlib.contextmanager
def sdpa_explicit_scale():
    """F.scaled_dot_product_attention with scale = head_dim^-0.5 spelled out (= PyTorch's default): the
    iOS18 frontend then decomposes it (matmul / softmax) instead of emitting the fused iOS18 op."""
    orig = F.scaled_dot_product_attention

    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                    scale=q.shape[-1] ** -0.5 if scale is None else scale, **kw)

    F.scaled_dot_product_attention = sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = orig


def convert(model, t_mel, target):
    mel = torch.zeros(1, t_mel, 128)
    valid = torch.tensor([t_mel], dtype=torch.int32)
    with torch.no_grad(), sdpa_explicit_scale():
        traced = torch.jit.trace(model, (mel, valid), check_trace=False)
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=(1, t_mel, 128), dtype=np.float32),
                ct.TensorType(name="valid_len", shape=(1,), dtype=np.int32)],
        outputs=[ct.TensorType(name="log_probs"), ct.TensorType(name="l11")],
        convert_to="mlprogram",
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )


def palettize(ml, names, G):
    cfg = ct.optimize.coreml.OpPalettizerConfig(mode="unique", granularity="per_grouped_channel", group_size=G,
                                                weight_threshold=WEIGHT_THRESHOLD)
    return ct.optimize.coreml.palettize_weights(ml, ct.optimize.coreml.OptimizationConfig(
        op_name_configs={mil_name(n): cfg for n in names}))


def verify(ml, snapped, palettes):
    """Every snapped tensor is a LUT op of <= 6 bits whose decompressed value equals the snapped fp16 weight,
    and the values used by each k-means group are its fp16 palette. Returns ({bits: tensors}, groups whose
    used values equal the whole palette, k-means groups, LUT ops not made here)."""
    luts = {op.name: op for op in ml._mil_program.functions["main"].operations
            if op.op_type == "constexpr_lut_to_dense"}
    bits, full, n_km = {}, 0, 0
    for name, w in snapped.items():
        op = luts.pop(mil_name(name) + "_palettized", None)
        if op is None:
            raise SystemExit(f"{name}: not palettized")
        lut = op.lut.val                                      # [n_groups, 1.., 2^bits, 1]
        nb = int(np.log2(lut.shape[-2]))
        bits[nb] = bits.get(nb, 0) + 1
        if nb > 6:
            raise SystemExit(f"{name}: {nb}-bit LUT")
        dense = np.asarray(op.materialized_val_inference(), np.float16)
        if not np.array_equal(dense.reshape(w.shape), w.numpy().astype(np.float16)):
            raise SystemExit(f"{name}: decompressed weight != snapped weight")
        groups = dense.reshape(len(palettes[name]), -1)
        for gi, pal in enumerate(palettes[name]):
            if pal is None:
                continue
            used = np.unique(groups[gi])
            if not np.isin(used, pal).all():
                raise SystemExit(f"{name} group {gi}: values outside the k-means palette")
            n_km += 1
            full += len(used) == len(pal)
    return bits, full, n_km, len(luts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", type=int, default=[8], help="output channels per LUT")
    ap.add_argument("--windows", nargs="+", default=list(WINDOWS))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    check_v1_files()
    base, (head, ck) = load_tower_fp32(), load_head_cpu()
    variant = "faithful"
    target = ct.target.macOS15
    for G in args.groups:
        precision = precision_name(G)
        outdir = OUT / "models" / variant / precision
        outdir.mkdir(parents=True, exist_ok=True)
        snapped, palettes, stats, km_s = None, None, None, None
        rep_path = outdir / "palettize.json"
        old = json.loads(rep_path.read_text()).get("windows", {}) if rep_path.exists() else {}
        report = {"precision": precision, "group_size": G, "target": target.name, "device": args.device,
                  "windows": old}
        for w in args.windows:
            t_mel, t0 = WINDOWS[w], time.time()
            model = V1Window(base.audio_tower, head, ck["mean"], ck["std"], t_mel, **VARIANTS[variant]).eval()
            if snapped is None:                               # the same weights for every window
                snapped, palettes, stats, km_s = snap_encoder(model, G, args.device)
                report.update(kmeans_seconds=round(km_s, 1), tensors=stats,
                              encoder_weights=sum(v.numel() for v in snapped.values()))
                lossy = sum(s["lossy_groups"] for s in stats.values())
                print(f"[palettize] {precision}: k-means on {args.device} {km_s:.1f}s, {len(stats)} tensors, "
                      f"{lossy}/{sum(s['groups'] for s in stats.values())} groups > {K} values", flush=True)
            with torch.no_grad():
                params = dict(model.named_parameters())
                for n, v in snapped.items():
                    params[n].copy_(v)
            t1 = time.time()
            ml = convert(model, t_mel, target)
            t_conv = time.time() - t1
            rec = {"time": time.strftime("%F %T"), "window": w, "t_mel": t_mel, "variant": variant,
                   "precision": precision, "group_size": G, "target": target.name, "convert_s": round(t_conv, 1)}
            t1 = time.time()
            ml = palettize(ml, snapped, G)
            rec["palettize_s"] = round(time.time() - t1, 1)
            bits, full, n_km, other_luts = verify(ml, snapped, palettes)
            rec.update(lut_bits=bits, kmeans_groups=n_km, kmeans_groups_full_palette=full, other_lut_ops=other_luts)
            path = outdir / f"v1_{w}.mlpackage"
            ml.short_description = f"V1 (EmbeddingGemma 2 L11 + CTC head), {w} window, {variant}, {precision}"
            ml.save(str(path))
            weights = path / "Data/com.apple.CoreML/weights/weight.bin"
            rec.update(path=str(path), seconds=round(time.time() - t0, 1), weight_bytes=weights.stat().st_size,
                       weight_sha256=sha256(weights), coremltools=ct.__version__, torch=torch.__version__)
            report["windows"][w] = rec
            with open(OUT / "models" / "palettize_log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[palettize] {json.dumps(rec)}", flush=True)
            rep_path.write_text(json.dumps(report, indent=1))
            del model, ml


if __name__ == "__main__":
    main()
