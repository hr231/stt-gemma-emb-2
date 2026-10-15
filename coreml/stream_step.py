#!/usr/bin/env python3
"""V1 streaming step model: 16 new mel frames (160 ms) + fixed-size caches -> 4 L11 frames and the
head log-probs of the 4 frames before them (the head's 4-frame lookahead), in PyTorch and Core ML.

Plan: ~/gemma_stt/coreml/STREAMING_PLAN.md (sections 2, 3, 5). Call j (0-based, C = 4 L11 frames per
call) consumes mel frames [16j, 16j+16) and returns
  log_probs [C, 257]  for L11 frames [4j-4, 4j)   (frames < 0 at j = 0 are meaningless: discard)
  l11       [C, 1024] for L11 frames [4j, 4j+4)   (raw layer-11 output, before the int8 round trip)
The caches (sizes from the encoder config, `state_spec`) are fixed-size FIFOs of recent activations
that slide by a dense shift (concat + static slice); all masks come from float validity flags in the
state, so no integer op is on the hot path. A zero state reproduces the offline start exactly
(zero mel context -> layer-0 frame -1 = 0, zero conv buffers = F.pad zeros, zero flags = keys before
frame 0 do not exist). StepCore reuses V1Window's submodules and weights (wrapper.py); the math is
V1Window's restricted to the new rows.

Core ML: two wrappers around the same StepCore
  state: caches are torch buffers -> ct.StateType (fp16 states, minimum target macOS 15 / iOS 18)
  io:    caches are explicit fp16 inputs <name> and outputs <name>_out (macOS 14 is enough)
Packages: ~/gemma_stt/coreml/models/stream/<variant>_<precision>/v1_step_<mech>.mlpackage (+ a
compiled v1_step_<mech>.mlmodelc next to it, loaded with ct.models.CompiledMLModel).

  ~/.venv-coreml/bin/python coreml/stream_step.py spec                    # state table from the config
  ~/.venv-coreml/bin/python coreml/stream_step.py export --mech state io  # convert + compile
  ~/.venv-coreml/bin/python coreml/stream_step.py check-torch [--n 200]   # fp32 StepTorch vs offline
  ~/.venv-coreml/bin/python coreml/stream_step.py placement --mech state io
  ~/.venv-coreml/bin/python coreml/stream_step.py bench --mech state      # idle machine only (SLOT)
"""
import argparse
import collections
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import OUT
from wrapper import FP16_MAX, NEG, V1Window

C_DEFAULT = 4                          # L11 frames per call (160 ms); mel frames per call = 4 C
MECHS = ("state", "io")
DEFAULT_MECH = "io"                     # ct.StateType does not run on the ANE (STREAMING.md, probe-state)
STREAM_DIR = OUT / "models" / "stream"
# Rows of mel context kept in sscp_mel per mechanism: 3 are needed; a [3, 128] *state* makes the ANE
# compile fail (ANECCompile -14, probe-state: tower1_sscp), [4, 128] compiles (tower1_sscp_pad4).
SSCP_ROWS = {"state": 4, "io": 3}


# ---------------------------------------------------------------- state sizing
def geometry(win, cfg):
    """Cache lengths derived from the encoder config `cfg` and the head (V1Window `win`)."""
    assert cfg.attention_context_right == 0, "the tower must be causal"
    window = cfg.attention_context_left - 1                     # HF max_past_horizon: 0 <= q - k < 12
    assert window == cfg.attention_chunk_size == 12
    attn = win.layers[0].attn
    k0 = win.sscp.layer0.conv.kernel_size[0]
    k1 = win.sscp.layer1.conv.kernel_size[0]
    s0 = win.sscp.layer0.conv.stride[0]
    hd = win.head
    futures = list(hd.futures)
    assert all(f == 0 for f in futures[:-1]), f"only the last head block may look ahead: {futures}"
    for blk in hd.blocks:
        assert blk.conv_pad[1] == 0, "head convs must be causal"
    return {
        "layers": len(win.layers), "heads": attn.H, "head_dim": attn.D, "hidden": cfg.hidden_size,
        "tower_window": window,                                 # frames a query sees, itself included
        "tower_kv": window - 1,                                 # cached past frames: 11
        "tower_conv": cfg.conv_kernel_size - 1,                 # causal light conv kernel 5 -> 4
        "mel_ctx": (k1 // 2) * s0 + k0 // 2,                    # mel rows of left context: 3
        "n_rel": attn.rel.shape[-1],
        "head_blocks": len(hd.blocks), "head_heads": hd.blocks[0].heads, "head_d": hd.inp.out_features,
        "head_past": hd.past, "head_future": futures[-1],       # 64, 4
        "head_conv": hd.blocks[0].conv.kernel_size[0] - 1,      # 4
        "n_out": hd.out.out_features,
    }


def state_spec(g, sscp_rows=None):
    """Ordered {name: shape} of the step state (fp16 in Core ML). sscp_rows: rows of the mel context
    kept (default the 3 needed; the StateType build keeps 4, see SSCP_ROWS)."""
    L, H, D, Th, Tc = g["layers"], g["heads"], g["head_dim"], g["tower_kv"], g["tower_conv"]
    nb, hh, d = g["head_blocks"], g["head_heads"], g["head_d"]
    dh, P, Fu = d // hh, g["head_past"], g["head_future"]
    return collections.OrderedDict([
        ("sscp_mel", (sscp_rows or g["mel_ctx"], 128)),  # last 3 (or 4) masked mel rows
        ("tower_k", (L, H, D, Th)),                     # K * k_scale, [H, D, T]
        ("tower_v", (L, H, Th, D)),
        ("tower_conv", (L, Tc, g["hidden"])),           # GLU outputs of the last 4 frames, time-major
        ("head_k", (nb - 1, hh, dh, P)),                # blocks 1-3, [H, Dh, T]
        ("head_v", (nb - 1, hh, P, dh)),
        ("head_k4", (hh, dh, P + Fu)),                  # block 4 keys over [s-68, s)
        ("head_v4", (hh, P + Fu, dh)),
        ("head_h4", (Fu, d)),                           # block-4 input of the 4 pending query frames
        ("head_conv", (nb, g["head_conv"], d)),         # ln_conv(h) * valid of the last 4 frames per block
        ("flags", (max(P + Fu, Th),)),                  # validity of L11 frames [s-68, s)
    ])


def spec_table(spec):
    rows = [(n, s, int(np.prod(s))) for n, s in spec.items()]
    total = sum(r[2] for r in rows)
    return rows, total


# ---------------------------------------------------------------- PyTorch step
class StepCore(nn.Module):
    """One call: (mel [1, 4C, 128], mel_valid [1, 4C], *states) -> (log_probs [1, C, 257],
    l11 [1, C, 1024], *new states). Pure function of its inputs; the wrappers own the state."""

    def __init__(self, win, cfg, C=C_DEFAULT, sscp_rows=None):
        super().__init__()
        self.g = g = geometry(win, cfg)
        self.spec = state_spec(g, sscp_rows)
        self.sscp_rows = self.spec["sscp_mel"][0]
        self.names = list(self.spec)
        self.C = C
        self.sscp, self.layers, self.head = win.sscp, win.layers, win.head
        self.int8 = win.int8
        self.register_buffer("mean", win.mean)
        self.register_buffer("inv_std", win.inv_std)
        Th, nr, P, Fu = g["tower_kv"], g["n_rel"], g["head_past"], g["head_future"]
        i = torch.arange(C)[:, None]
        # tower: keys = cat(cache Th, new C); query i sits at key slot Th + i
        j = torch.arange(Th + C)[None, :]
        dist = (Th + i) - j                                       # query - key
        place = torch.stack([dist == (nr - 1 - r) for r in range(nr)], 1).float()   # [C, 13, Th+C]
        self.register_buffer("t_place", place)
        self.register_buffer("t_band", torch.where((dist >= 0) & (dist < g["tower_window"]), 0.0, NEG))
        # head blocks 1-3: keys = frames [s-P, s+C), query i = frame s+i at slot P+i; band rel in [-P, 0]
        j = torch.arange(P + C)[None, :]
        rel = j - (P + i)
        self.register_buffer("h_band", ((rel >= -P) & (rel <= 0)).float())
        self.register_buffer("h_diag", (rel == 0).float())
        # head block 4: keys = frames [s-P-Fu, s+C), query i = frame s-Fu+i at slot P+i; rel in [-P, Fu]
        j = torch.arange(P + Fu + C)[None, :]
        rel = j - (P + i)
        self.register_buffer("h4_band", ((rel >= -P) & (rel <= Fu)).float())
        self.register_buffer("h4_diag", (rel == 0).float())

    # -- subsampling: x = cat(sscp_mel, mel * valid); convs without time padding
    @staticmethod
    def _ln_act(l, y):
        """ReLU(LayerNorm over channels) of an NCHW tensor. The LayerNorm has no bias; it is given an
        explicit zero bias because coremltools 9.0 + Core ML compute transpose -> bias-free layer_norm ->
        transpose wrongly in fp32 on the CPU (max |d| 1.5 on random data, also at the window shape; the
        fp16 ANE result happened to be right; repro in STREAMING.md), while the same with a bias is exact."""
        n = l.norm
        y = F.layer_norm(y.permute(0, 2, 3, 1), n.normalized_shape, n.weight, torch.zeros_like(n.weight), n.eps)
        return l.act(y.permute(0, 3, 1, 2))

    def _sscp(self, mel, m, sscp_mel):
        C, ctx = self.C, self.g["mel_ctx"]
        x = torch.cat([sscp_mel[-ctx:], mel[0] * m[:, None]], 0)                  # [ctx + 4C, 128]
        l0, l1 = self.sscp.layer0, self.sscp.layer1
        y = F.conv2d(x[None, None], l0.conv.weight, stride=l0.conv.stride, padding=(0, 1))
        y = self._ln_act(l0, y)                                                    # [1, 128, 2C+1, 64]
        m1 = torch.cat([torch.ones_like(m[:1]), m[0::2]], 0)                       # cached frame 2s-1 is valid
        y = y * m1[None, None, :, None]
        y = F.conv2d(y, l1.conv.weight, stride=l1.conv.stride, padding=(0, 1))
        y = self._ln_act(l1, y)                                                    # [1, 32, C, 32]
        y = y.permute(0, 2, 3, 1).reshape(1, C, -1)
        return self.sscp.input_proj_linear(y), x[-self.sscp_rows:]

    def _tower_layer(self, layer, h, k_c, v_c, conv_c, mask):
        C, Th, Tc = self.C, self.g["tower_kv"], self.g["tower_conv"]
        a = layer.attn
        H, D = a.H, a.D
        h = layer.ff1(h)
        res = h
        x = layer.norm_pre_attn(h.clamp(-FP16_MAX, FP16_MAX))
        q = (a.q_proj(x).view(C, H, D) * (a.q_scale * F.softplus(a.per_dim_scale))).permute(1, 0, 2)
        k = (a.k_proj(x).view(C, H, D) * a.k_scale).permute(1, 2, 0)              # [H, D, C]
        v = a.v_proj(x).view(C, H, D).permute(1, 0, 2)                             # [H, C, D]
        K = torch.cat([k_c, k], -1)                                                # [H, D, Th+C]
        V = torch.cat([v_c, v], 1)                                                 # [H, Th+C, D]
        bd = ((q @ a.rel).permute(1, 0, 2) @ self.t_place).permute(1, 0, 2)        # [H, C, Th+C]
        w = torch.tanh((q @ K + bd) / a.softcap) * a.softcap + mask
        o = torch.softmax(w, dim=-1) @ V
        h = a.post(o.permute(1, 0, 2).reshape(1, C, H * D))
        h = layer.norm_post_attn(h.clamp(-FP16_MAX, FP16_MAX)) + res
        lc = layer.lconv
        res = h
        x = F.glu(lc.linear_start(lc.pre_layer_norm(h)), dim=-1)                   # [1, C, 1024]
        G = torch.cat([conv_c[None], x], 1)                                         # [1, Tc+C, 1024]
        y = F.conv1d(G.transpose(1, 2), lc.depthwise_conv1d.weight, groups=self.g["hidden"]).transpose(1, 2)
        gc = min(float(lc.gradient_clipping), FP16_MAX)
        y = lc.linear_end(lc.act_fn(lc.conv_norm(y.clamp(-gc, gc))))
        h = layer.ff2(y + res)
        h = layer.norm_out(h.clamp(-FP16_MAX, FP16_MAX))
        return h, K[..., -Th:], V[:, -Th:], G[0, -Tc:]

    @staticmethod
    def _attend(blk, q, K, V, allowed):
        mask = (allowed - 1.0) * (-NEG)                                            # 0 / -1e4
        s = (q @ K) * (1.0 / math.sqrt(blk.qkv.out_features // 3 // blk.heads)) + mask
        return torch.softmax(s, dim=-1) @ V                                        # [H, Tq, Dh]

    def _head(self, z, vn, head_k, head_v, head_k4, head_v4, head_h4, head_conv, flags):
        hd, C, g = self.head, self.C, self.g
        P, Fu, Tc, d = g["head_past"], g["head_future"], g["head_conv"], g["head_d"]
        hh = g["head_heads"]
        dh = d // hh
        kv_all = torch.cat([flags, vn], 0)                                         # frames [s-68, s+C)
        h = hd.inp(z)                                                              # [1, C, d]
        new_k, new_v, new_conv = [], [], []
        allowed = torch.maximum(self.h_band * kv_all[-(P + C):][None, :], self.h_diag)
        for b, blk in enumerate(hd.blocks[:-1]):
            qkv = blk.qkv(blk.ln_att(h)).view(C, 3, hh, dh).permute(1, 2, 0, 3)   # [3, H, C, Dh]
            K = torch.cat([head_k[b], qkv[1].transpose(1, 2)], -1)                 # [H, Dh, P+C]
            V = torch.cat([head_v[b], qkv[2]], 1)                                  # [H, P+C, Dh]
            a = self._attend(blk, qkv[0], K, V, allowed)
            h = h + blk.proj(a.permute(1, 0, 2).reshape(1, C, d))
            G = torch.cat([head_conv[b][None], blk.ln_conv(h) * vn[None, :, None]], 1)
            h = h + F.conv1d(G.transpose(1, 2), blk.conv.weight, blk.conv.bias, groups=d).transpose(1, 2)
            h = h + blk.ff(blk.ln_ff(h))
            new_k.append(K[..., -P:])
            new_v.append(V[:, -P:])
            new_conv.append(G[0, -Tc:])
        blk = hd.blocks[-1]                                                        # delay line
        Hq = torch.cat([head_h4[None], h], 1)                                      # frames [s-Fu, s+C)
        qkv = blk.qkv(blk.ln_att(Hq)).view(Fu + C, 3, hh, dh).permute(1, 2, 0, 3)
        K = torch.cat([head_k4, qkv[1][:, Fu:].transpose(1, 2)], -1)               # [H, Dh, P+Fu+C]
        V = torch.cat([head_v4, qkv[2][:, Fu:]], 1)
        allowed4 = torch.maximum(self.h4_band * kv_all[None, :], self.h4_diag)
        a = self._attend(blk, qkv[0][:, :C], K, V, allowed4)
        x = Hq[:, :C] + blk.proj(a.permute(1, 0, 2).reshape(1, C, d))             # frames [s-Fu, s-Fu+C)
        qv = kv_all[-(Fu + C):][:C]
        G = torch.cat([head_conv[-1][None], blk.ln_conv(x) * qv[None, :, None]], 1)
        x = x + F.conv1d(G.transpose(1, 2), blk.conv.weight, blk.conv.bias, groups=d).transpose(1, 2)
        x = x + blk.ff(blk.ln_ff(x))
        logits = hd.out(hd.ln(x))
        s = logits - logits.amax(-1, keepdim=True)                                 # stable log-softmax
        lp = s - torch.log(torch.exp(s).sum(-1, keepdim=True))
        new_conv.append(G[0, -Tc:])
        return (lp, torch.stack(new_k), torch.stack(new_v), K[..., -(P + Fu):], V[:, -(P + Fu):],
                Hq[0, -Fu:], torch.stack(new_conv), kv_all[-self.spec["flags"][0]:])

    def forward(self, mel, mel_valid, sscp_mel, tower_k, tower_v, tower_conv, head_k, head_v,
                head_k4, head_v4, head_h4, head_conv, flags):
        Th = self.g["tower_kv"]
        m = mel_valid[0]                                                           # [4C] 0/1
        vn = m[0::4]                                                               # validity of the new L11 frames
        h, new_mel = self._sscp(mel, m, sscp_mel)
        kmask = (torch.cat([flags[-Th:], torch.ones_like(vn)], 0) - 1.0) * (-NEG)  # cached frames < 0: masked
        mask = self.t_band + kmask[None, :]
        ks, vs, cs = [], [], []
        for l, layer in enumerate(self.layers):
            h, k, v, c = self._tower_layer(layer, h, tower_k[l], tower_v[l], tower_conv[l], mask)
            ks.append(k)
            vs.append(v)
            cs.append(c)
        l11 = h
        if self.int8:
            scale = l11.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127.0
            x = torch.round(l11 / scale).clamp(-127, 127) * scale
        else:
            x = l11
        z = (x - self.mean) * self.inv_std * vn[None, :, None]
        lp, hk, hv, hk4, hv4, hh4, hc, fl = self._head(z, vn, head_k, head_v, head_k4, head_v4, head_h4,
                                                       head_conv, flags)
        return (lp, l11, new_mel, torch.stack(ks), torch.stack(vs), torch.stack(cs),
                hk, hv, hk4, hv4, hh4, hc, fl)


class StatefulStep(nn.Module):
    """ct.StateType wrapper: the caches are buffers, overwritten in place (read_state -> slice_update ->
    coreml_update_state)."""

    def __init__(self, core):
        super().__init__()
        self.core = core
        self.names = core.names
        for n, s in core.spec.items():
            self.register_buffer(n, torch.zeros(s))

    def forward(self, mel, mel_valid):
        out = self.core(mel, mel_valid, *[getattr(self, n) for n in self.names])
        for n, t in zip(self.names, out[2:]):
            getattr(self, n)[:] = t
        return out[0], out[1]


def load_v1():
    """(HF base model fp32, V1 head, head checkpoint) from the V1 files (sha256-checked)."""
    from common import check_v1_files, load_head_cpu, load_tower_fp32
    check_v1_files()
    base = load_tower_fp32()
    head, ck = load_head_cpu()
    return base, head, ck


def build_core(variant="faithful", C=C_DEFAULT, v1=None, sscp_rows=None):
    """StepCore around a V1Window (its submodules and weights) of the given export variant."""
    from export import VARIANTS
    base, head, ck = v1 or load_v1()
    opts = {k: v for k, v in VARIANTS[variant].items() if k != "pad_fill"}        # no padded window rows here
    win = V1Window(base.audio_tower, head, ck["mean"], ck["std"], 4 * C, **opts).eval()
    return StepCore(win, base.audio_tower.config, C, sscp_rows).eval()


class StepTorch:
    """PyTorch reference with the StepModel API (fp32, CPU)."""

    def __init__(self, core=None, variant="faithful"):
        self.core = core if core is not None else build_core(variant)
        self.C = self.core.C
        self.reset()

    def reset(self):
        self.states = [torch.zeros(s) for s in self.core.spec.values()]
        self.calls = 0

    def step(self, mel, n_valid=None):
        C4 = 4 * self.C
        n = C4 if n_valid is None else int(n_valid)
        x = torch.zeros(1, C4, 128)
        x[0, :n] = torch.from_numpy(np.array(mel[:n], np.float32))
        mv = torch.zeros(1, C4)
        mv[0, :n] = 1.0
        with torch.no_grad():
            out = self.core(x, mv, *self.states)
        self.states = list(out[2:])
        self.calls += 1
        self.last_l11 = out[1][0].numpy()
        return out[0][0].numpy()

    def flush(self):
        return self.step(np.zeros((4 * self.C, 128), np.float32), 0)

    def read_states(self):
        return {n: t.numpy() for n, t in zip(self.core.names, self.states)}


# ---------------------------------------------------------------- Core ML
def package_path(variant="faithful", precision="fp16", mech=DEFAULT_MECH):
    return STREAM_DIR / f"{variant}_{precision}" / f"v1_step_{mech}.mlpackage"


def compiled_path(variant="faithful", precision="fp16", mech=DEFAULT_MECH):
    return package_path(variant, precision, mech).with_suffix(".mlmodelc")


def convert(core, mech, precision="fp16"):
    import coremltools as ct
    C4 = 4 * core.C
    cp = ct.precision.FLOAT32 if precision == "fp32" else ct.precision.FLOAT16
    # Explicit state I/O is fp32 at the model boundary (cast to fp16 inside, values are fp16-exact): the
    # Python bridge does not see in-place updates of a reused fp16 input array (Core ML keeps the content
    # from the array's first use; measured: flags stayed stale with persistent or double-buffered fp16
    # arrays, correct with fresh ones), while reused fp32 inputs are read every call.
    sdt = np.float32
    inputs = [ct.TensorType(name="mel", shape=(1, C4, 128), dtype=np.float32),
              ct.TensorType(name="mel_valid", shape=(1, C4), dtype=np.float32)]
    mel, mv = torch.zeros(1, C4, 128), torch.ones(1, C4)
    if mech == "state":
        if precision == "fp32":
            raise ValueError("Core ML states are fp16 only")
        mod = StatefulStep(core).eval()
        with torch.no_grad():
            traced = torch.jit.trace(mod, (mel, mv), check_trace=False)
        return ct.convert(
            traced, inputs=inputs,
            outputs=[ct.TensorType(name="log_probs", dtype=np.float32), ct.TensorType(name="l11", dtype=np.float32)],
            states=[ct.StateType(wrapped_type=ct.TensorType(shape=s, dtype=np.float16), name=n)
                    for n, s in core.spec.items()],
            convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
            compute_precision=cp, compute_units=ct.ComputeUnit.CPU_AND_NE)
    states = [torch.zeros(s) for s in core.spec.values()]
    with torch.no_grad():
        traced = torch.jit.trace(core, (mel, mv, *states), check_trace=False)
    return ct.convert(
        traced,
        inputs=inputs + [ct.TensorType(name=n, shape=s, dtype=sdt) for n, s in core.spec.items()],
        outputs=[ct.TensorType(name="log_probs", dtype=np.float32), ct.TensorType(name="l11", dtype=np.float32)]
        + [ct.TensorType(name=f"{n}_out", dtype=sdt) for n in core.spec],
        convert_to="mlprogram", minimum_deployment_target=ct.target.macOS14,
        compute_precision=cp, compute_units=ct.ComputeUnit.CPU_AND_NE)


def compile_package(pkg, dst):
    import shutil
    import coremltools as ct
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)                       # our own compiled artifact, rebuilt from the package
    return Path(ct.models.utils.compile_model(str(pkg), str(dst)))


def export(variant="faithful", precision="fp16", mechs=MECHS, v1=None):
    import coremltools as ct
    from common import sha256
    v1 = v1 or load_v1()
    recs = []
    for mech in mechs:
        t0 = time.time()
        core = build_core(variant, v1=v1, sscp_rows=SSCP_ROWS[mech])
        ml = convert(core, mech, precision)
        path = package_path(variant, precision, mech)
        path.parent.mkdir(parents=True, exist_ok=True)
        ml.short_description = (f"V1 streaming step ({core.C} L11 frames = {40 * core.C} ms per call), "
                                f"{variant}, {precision}, {mech}")
        ml.save(str(path))
        t_conv = time.time() - t0
        t1 = time.time()
        comp = compile_package(path, compiled_path(variant, precision, mech))
        weights = path / "Data/com.apple.CoreML/weights/weight.bin"
        rec = {"time": time.strftime("%F %T"), "kind": "stream_step", "mech": mech, "C": core.C, "variant": variant,
               "precision": precision, "path": str(path), "compiled": str(comp),
               "convert_s": round(t_conv, 1), "compile_s": round(time.time() - t1, 1),
               "min_target": "macOS15" if mech == "state" else "macOS14",
               "weight_bytes": weights.stat().st_size if weights.exists() else None,
               "weight_sha256": sha256(weights) if weights.exists() else None,
               "coremltools": ct.__version__, "torch": torch.__version__}
        with open(STREAM_DIR / "export_log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[export] {json.dumps(rec)}", flush=True)
        recs.append(rec)
        del ml
    return recs


class StepModel:
    """Core ML streaming step. reset(); step(mel16 [16, 128] float32, n_valid=16) -> log_probs [4, 257]
    float32 for L11 frames [4j-4, 4j) at call j (frames < 0 and >= ceil(V/4) are the caller's to
    discard); flush() = step on zeros with n_valid 0 (frames [4J-4, 4J) after the last data call J-1).
    last_l11 [4, 1024]: L11 frames [4j, 4j+4) of the last call."""

    def __init__(self, variant="faithful", precision="fp16", compute_units="CPU_AND_NE", mech=DEFAULT_MECH,
                 path=None):
        import coremltools as ct
        self.variant, self.precision, self.compute_units, self.mech = variant, precision, compute_units, mech
        self.path = Path(path) if path else compiled_path(variant, precision, mech)
        if not self.path.exists():
            compile_package(package_path(variant, precision, mech), self.path)
        t0 = time.time()
        self.model = ct.models.CompiledMLModel(str(self.path), compute_units=getattr(ct.ComputeUnit, compute_units))
        self.load_s = time.time() - t0
        self.spec = state_spec(DEFAULT_GEOMETRY, SSCP_ROWS[mech])
        # host-side state arrays are float32 for both mechanisms: explicit I/O, see convert(); StateType:
        # MLState.write_state rejects float16 numpy ("value type not convertible"), takes float32 for an
        # fp16 state, and read_state returns float32
        self.dtype = np.float32
        self.C4 = 4 * C_DEFAULT
        # Core ML releases input feature values later on its own queue: every array passed to predict()
        # (and the dict holding them) stays alive for the model's lifetime and is refilled in place.
        self._mel = np.zeros((1, self.C4, 128), np.float32)
        self._mv = np.zeros((1, self.C4), np.float32)
        self._data = {"mel": self._mel, "mel_valid": self._mv}
        self._zero = {n: np.zeros(s, self.dtype) for n, s in self.spec.items()}    # write_state sources
        if mech == "io":
            self._states = {n: np.zeros(s, self.dtype) for n, s in self.spec.items()}
            self._data.update(self._states)
        self._retired = []                                                         # previous MLState objects
        self.state = None
        self.reset()

    def reset(self):
        """New stream: fresh state with every buffer explicitly zeroed (make_state() contents are not
        documented)."""
        self.calls = 0
        self.last_l11 = None
        if self.mech == "state":
            if self.state is not None:
                self._retired = (self._retired + [self.state])[-2:]
            self.state = self.model.make_state()
            for n, z in self._zero.items():
                self.state.write_state(n, z)
        else:
            for v in self._states.values():
                v[...] = 0

    def read_states(self):
        if self.mech == "state":
            return {n: np.array(self.state.read_state(n)) for n in self.spec}
        return {n: np.array(v) for n, v in self._states.items()}

    def step(self, mel, n_valid=None):
        n = self.C4 if n_valid is None else int(n_valid)
        self._mel[:] = 0.0
        self._mel[0, :n] = np.asarray(mel, np.float32)[:n]
        self._mv[:] = 0.0
        self._mv[0, :n] = 1.0
        if self.mech == "state":
            out = self.model.predict(self._data, state=self.state)
        else:
            out = self.model.predict(self._data)
            for k, v in self._states.items():
                np.copyto(v, out[f"{k}_out"])
        self.calls += 1
        self.last_l11 = np.array(out["l11"], np.float32)[0]
        return np.array(out["log_probs"], np.float32)[0]

    def flush(self):
        return self.step(np.zeros((self.C4, 128), np.float32), 0)


# Geometry of V1 (asserted against the live config by `spec` / the tests via geometry(win)); kept as a
# constant so StepModel does not load the PyTorch weights.
DEFAULT_GEOMETRY = {"layers": 12, "heads": 8, "head_dim": 128, "hidden": 1024, "tower_window": 12, "tower_kv": 11,
                    "tower_conv": 4, "mel_ctx": 3, "n_rel": 13, "head_blocks": 4, "head_heads": 8, "head_d": 512,
                    "head_past": 64, "head_future": 4, "head_conv": 4, "n_out": 257}


# ---------------------------------------------------------------- host loop helpers
def run_stream(model, mel, n_valid):
    """Feed one utterance (mel [T, 128], V valid frames) step by step from a reset model; returns
    (log_probs [ceil(V/4), 257], l11 [ceil(V/4), 1024], calls)."""
    C = C_DEFAULT
    C4 = 4 * C
    L = -(-n_valid // 4)
    model.reset()
    lps, l11s = [], []
    s = 0
    while s * 4 < n_valid:                                                         # data calls
        a = 4 * s
        n = min(C4, n_valid - a)
        lps.append(model.step(mel[a:a + n], n))
        l11s.append(model.last_l11)
        s += C
    if s - 4 < L:
        lps.append(model.flush())
    lp = np.concatenate(lps)[4:4 + L]                                              # first rows: frames -4..-1
    l11 = np.concatenate(l11s)[:L]
    return lp, l11, model.calls


def long_mel(wav, chunk=6000):
    """Valid mel frames [V, 128] of a long waveform, computed by common.mel() on overlapping pieces of
    `chunk` frames (frame i only uses samples [160i-160, 160i+160) and the validity sample 160i+160)."""
    from common import mel as mel_fn
    N = len(wav)
    V = (N - 161) // 160 + 1 if N >= 161 else 0
    out = np.zeros((V, 128), np.float32)
    for a in range(0, V, chunk):
        b = min(V, a + chunk)
        o = 0 if a == 0 else 160 * (a - 1)
        m, nv = mel_fn(wav[o:min(N, 160 * b + 161)])
        first = 0 if a == 0 else 1
        assert nv >= first + (b - a), (a, b, nv)
        out[a:b] = m[first:first + (b - a)]
    return out


def tile_offline(predict, mel, n_valid, g0, g1, t_mel=3500, keep_from=453):
    """Offline recomputation of L11 / head frames [g0, g1) of a long stream by receptive-field tiling:
    windows of t_mel mel frames (W = t_mel/4 L11 frames); in each window keep the frames >= keep_from
    (past receptive field 452 L11 frames + 1 for the subsampling edge) and < W - 4 (head lookahead),
    except at the stream start (keep from 0) and end (keep to L). predict(mel_slice, nv) -> (lp, l11).
    Returns (lp [g1-g0, 257], l11 [g1-g0, 1024], windows)."""
    W, L = t_mel // 4, -(-n_valid // 4)
    lp, l11 = np.zeros((g1 - g0, 257), np.float32), np.zeros((g1 - g0, 1024), np.float32)
    g, n = g0, 0
    while g < g1:
        w0 = max(0, g - keep_from)
        nv = min(t_mel, n_valid - 4 * w0)
        a, b = g - w0, (W - 4 if 4 * w0 + t_mel < n_valid else L - w0)
        b = min(b, g1 - w0)
        p, x = predict(mel[4 * w0:4 * w0 + nv], nv)
        lp[g - g0:g - g0 + b - a], l11[g - g0:g - g0 + b - a] = p[a:b], x[a:b]
        g, n = w0 + b, n + 1
    return lp, l11, n


def cmd_diag(args):
    """Core ML step vs StepTorch call by call on one FAST utterance: max |d| of outputs and of every state."""
    from cmlrun import load_reference, ref_slices
    utts, arrays = load_reference("fast")
    mel, _, _ = ref_slices(utts[0], arrays)
    st = StepTorch(build_core(args.variant, sscp_rows=SSCP_ROWS[args.mech[0]]))
    cm = StepModel(args.variant, args.precision, args.compute_units, args.mech[0])
    for j in range(args.n):
        x = mel[16 * j:16 * j + 16]
        a, b = st.step(x, len(x)), cm.step(x, len(x))
        sa, sb = st.read_states(), cm.read_states()
        d = {n: float(np.abs(sa[n] - sb[n].astype(np.float32)).max()) for n in sa}
        print(f"[diag] call {j}: lp {np.abs(a - b).max():.4g} l11 {np.abs(st.last_l11 - cm.last_l11).max():.4g} "
              f"| states {json.dumps({k: round(v, 4) for k, v in d.items()})}", flush=True)


# ---------------------------------------------------------------- CLI
def cmd_spec(args):
    v1 = load_v1()
    cfg = v1[0].audio_tower.config
    keys = ("attention_chunk_size", "attention_context_left", "attention_context_right", "conv_kernel_size",
            "num_attention_heads", "hidden_size", "num_hidden_layers", "subsampling_conv_channels")
    print("[config] " + json.dumps({k: getattr(cfg, k) for k in keys}))
    core = build_core(args.variant, v1=v1)
    assert core.g == DEFAULT_GEOMETRY, (core.g, DEFAULT_GEOMETRY)
    rows, total = spec_table(core.spec)
    print("[geometry] " + json.dumps(core.g))
    for n, s, v in rows:
        print(f"[state] {n:12s} {str(list(s)):22s} {v:9,d} values {2 * v:11,d} bytes fp16")
    print(f"[state] total {total:,d} values = {2 * total:,d} bytes = {2 * total / 2 ** 20:.3f} MiB (fp16)")


def offline_torch(v1, mel, n_valid, variant="faithful"):
    """V1Window fp32 on exactly len(mel) mel frames: (log_probs, l11) of the ceil(V/4) valid frames."""
    from export import VARIANTS
    base, head, ck = v1
    win = V1Window(base.audio_tower, head, ck["mean"], ck["std"], len(mel), **VARIANTS[variant]).eval()
    with torch.no_grad():
        lp, l11 = win(torch.from_numpy(np.ascontiguousarray(mel, np.float32))[None],
                      torch.tensor([n_valid], dtype=torch.int32))
    L = -(-n_valid // 4)
    return lp[0, :L].numpy(), l11[0, :L].numpy()


def cos_flat(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def cmd_check_torch(args):
    """S2: fp32 StepTorch vs the FAST reference (HF fp32) on --n utterances and vs V1Window fp32 on the
    first --window-check of them."""
    from cmlrun import cos_rows, load_reference, ref_slices
    from common import greedy_text
    torch.set_num_threads(args.threads)
    v1 = load_v1()
    st = StepTorch(build_core(args.variant, v1=v1))
    utts, arrays = load_reference("fast")
    utts = utts[:args.n]
    rows = []
    t0 = time.time()
    for i, u in enumerate(utts):
        mel, rl, rp = ref_slices(u, arrays)
        lp, l11, calls = run_stream(st, mel, u["mel_valid"])
        r = {"id": u["id"], "frames": len(lp), "calls": calls, "l11_cos": cos_flat(l11, rl), "lp_cos": cos_flat(lp, rp),
             "l11_frame_cos_min": float(cos_rows(l11, rl).min()), "lp_max_abs_diff": float(np.abs(lp - rp).max()),
             "greedy_same": greedy_text(lp) == u["greedy"]}
        if i < args.window_check:
            wlp, wl11 = offline_torch(v1, mel, u["mel_valid"], args.variant)
            r.update({"win_l11_cos": cos_flat(l11, wl11), "win_lp_cos": cos_flat(lp, wlp),
                      "win_l11_max_abs_diff": float(np.abs(l11 - wl11).max()),
                      "win_lp_max_abs_diff": float(np.abs(lp - wlp).max())})
        rows.append(r)
        if i % 10 == 0 or i < args.window_check:
            print(f"[check-torch] {i + 1}/{len(utts)} {json.dumps(r)} ({time.time() - t0:.0f} s)", flush=True)
    summ = {"utts": len(rows), "l11_cos_min": min(r["l11_cos"] for r in rows),
            "lp_cos_min": min(r["lp_cos"] for r in rows),
            "l11_frame_cos_min": min(r["l11_frame_cos_min"] for r in rows),
            "lp_max_abs_diff": max(r["lp_max_abs_diff"] for r in rows),
            "greedy_identical": sum(r["greedy_same"] for r in rows), "seconds": round(time.time() - t0, 1)}
    w = [r for r in rows if "win_lp_cos" in r]
    if w:
        summ.update({"window_utts": len(w), "window_l11_cos_min": min(r["win_l11_cos"] for r in w),
                     "window_lp_cos_min": min(r["win_lp_cos"] for r in w),
                     "window_l11_max_abs_diff": max(r["win_l11_max_abs_diff"] for r in w),
                     "window_lp_max_abs_diff": max(r["win_lp_max_abs_diff"] for r in w)})
    print("[check-torch] " + json.dumps(summ), flush=True)
    d = OUT / "stream"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"check_torch_{args.variant}.json").write_text(json.dumps({"summary": summ, "utts": rows}, indent=0))


def cmd_export(args):
    export(args.variant, args.precision, args.mech)


def cmd_placement(args):
    from placement import plan_for, summarize
    out = {}
    for mech in args.mech:
        try:
            s = summarize(plan_for(package_path(args.variant, args.precision, mech), args.compute_units))
        except RuntimeError as e:                                                  # e.g. ANECCompile failure (-14)
            out[mech] = {"error": str(e)}
            print(f"[placement] {mech}: no compute plan: {' '.join(str(e).split())}", flush=True)
            continue
        out[mech] = s
        types = collections.Counter((r["type"], r["device"]) for r in s["not_on_ane"])
        print(f"[placement] {mech}: {s['ops']} ops, ANE share {100 * s['ane_cost_share']:.3f}% | "
              + ", ".join(f"{d} {v['ops']} ops {100 * v['cost_share']:.3f}%" for d, v in s["by_device"].items())
              + f" | not on ANE: {dict(types)}", flush=True)
    d = OUT / "stream"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"placement_{args.variant}_{args.precision}.json").write_text(json.dumps(out, indent=1))


class _PartStep(nn.Module):
    """Pieces of StepCore with their states as buffers, to locate a StateType ANE compile failure."""

    PARTS = {"shift": ["flags"], "sscp": ["sscp_mel"], "tower1": ["sscp_mel", "tower_k", "tower_v", "tower_conv"],
             "tower1_none": [], "tower1_k": ["tower_k"], "tower1_v": ["tower_v"], "tower1_conv": ["tower_conv"],
             "tower1_sscp": ["sscp_mel"], "tower1_kv": ["tower_k", "tower_v"],
             "tower1_kvc": ["tower_k", "tower_v", "tower_conv"],
             "tower1_sscp_pad4": ["sscp_mel"], "tower1_sscp_t": ["sscp_mel"],
             "head": ["head_k", "head_v", "head_k4", "head_v4", "head_h4", "head_conv", "flags"]}
    # sscp_mel layouts tried: [3, 128] (as shipped), [4, 128] (last 4 rows kept, 3 used), [128, 3]
    SSCP_SHAPE = {"tower1_sscp_pad4": (4, 128), "tower1_sscp_t": (128, 3)}

    def __init__(self, core, part):
        super().__init__()
        self.core, self.part, self.names = core, part, self.PARTS[part]
        for n in self.names:
            s = core.spec[n]
            if n == "sscp_mel":
                s = self.SSCP_SHAPE.get(part, s)
            self.register_buffer(n, torch.zeros((1,) + s[1:] if n.startswith("tower_") else s))

    def forward(self, x, mel_valid):
        c, b = self.core, {n: getattr(self, n) for n in self.names}
        for n in ("sscp_mel", "tower_k", "tower_v", "tower_conv"):                 # pieces not in state: zeros
            if n not in b:
                b[n] = torch.zeros((1,) + c.spec[n][1:] if n.startswith("tower_") else c.spec[n])
        vn = mel_valid[0][0::4]
        if self.part == "shift":
            new = {"flags": torch.cat([b["flags"], vn], 0)[-c.spec["flags"][0]:]}
            y = b["flags"][None] * 2.0
        elif self.part == "head":
            out = c._head(x[:, :c.C], vn, *[b[n] for n in self.names])
            y, new = out[0], dict(zip(self.names, out[1:]))
        else:
            sm = b["sscp_mel"]
            sm = sm.transpose(0, 1) if self.part == "tower1_sscp_t" else sm[-3:]
            y, mel_ctx = c._sscp(x, mel_valid[0], sm)
            if self.part == "tower1_sscp_pad4":
                mel_ctx = torch.cat([x[0, -4:-3] * mel_valid[0][-4:-3, None], mel_ctx], 0)   # masked mel rows -4..-1
            elif self.part == "tower1_sscp_t":
                mel_ctx = mel_ctx.transpose(0, 1)
            new = {"sscp_mel": mel_ctx}
            if self.part.startswith("tower1"):
                Th = c.g["tower_kv"]
                kmask = (torch.cat([torch.ones(Th), torch.ones_like(vn)], 0) - 1.0) * (-NEG)
                y, k, v, cc = c._tower_layer(c.layers[0], y, b["tower_k"][0], b["tower_v"][0], b["tower_conv"][0],
                                             c.t_band + kmask[None, :])
                new.update({"tower_k": k[None], "tower_v": v[None], "tower_conv": cc[None]})
        for n in self.names:
            getattr(self, n)[:] = new[n]
        return y


def cmd_probe_state(args):
    """StateType vs explicit I/O on pieces of the step: does the Neural Engine compile succeed?"""
    import subprocess
    import sys
    import tempfile
    import coremltools as ct
    from probe_ane import CHILD
    core = build_core(args.variant)
    C4 = 4 * core.C
    res = {}
    for part in args.parts:
        mod = _PartStep(core, part).eval()
        xin = torch.zeros(1, C4, 1024 if part == "head" else 128)
        with torch.no_grad():
            tr = torch.jit.trace(mod, (xin, torch.ones(1, C4)), check_trace=False)
        ml = ct.convert(tr, inputs=[ct.TensorType(name="x", shape=tuple(xin.shape), dtype=np.float32),
                                    ct.TensorType(name="mel_valid", shape=(1, C4), dtype=np.float32)],
                        states=[ct.StateType(wrapped_type=ct.TensorType(shape=tuple(getattr(mod, n).shape),
                                                                        dtype=np.float16), name=n) for n in mod.names],
                        convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                        compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_ONLY)
        path = Path(tempfile.mkdtemp(prefix="v1sprobe_")) / f"{part}.mlpackage"
        ml.save(str(path))
        p = subprocess.run([sys.executable, "-c", CHILD, str(path)], capture_output=True, text=True)
        text = p.stdout + p.stderr
        marker = [l for l in p.stdout.splitlines() if l.startswith("PLACEMENT_JSON:")]
        rows = json.loads(marker[-1][len("PLACEMENT_JSON:"):]) if marker else []
        res[part] = {"states": mod.names, "ane_compile_failed": "ANECCompile" in text or "error code: -14" in text,
                     "plan_ok": bool(marker), "ops": {f"{t}@{d}": v for t, d, v in rows}}
        print(f"[probe-state] {part}: {json.dumps(res[part])}", flush=True)
    d = OUT / "stream"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "probe_state.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    p.write_text(json.dumps({**old, **res}, indent=1))


_LOAD_CHILD = """
import json, sys, time
t0 = time.perf_counter()
import coremltools as ct, numpy as np
sys.path.insert(0, sys.argv[3])
from stream_step import StepModel
t1 = time.perf_counter()
m = StepModel(mech=sys.argv[2], path=sys.argv[1])
t2 = time.perf_counter()
m.step(np.zeros((16, 128), np.float32))
t3 = time.perf_counter()
m.step(np.zeros((16, 128), np.float32))
t4 = time.perf_counter()
print("LOAD_JSON:" + json.dumps({"import_s": t1 - t0, "load_s": t2 - t1, "first_step_s": t3 - t2, "second_step_s": t4 - t3}))
"""


def _child_load(comp, mech):
    import subprocess
    import sys
    p = subprocess.run([sys.executable, "-c", _LOAD_CHILD, str(comp), mech, str(Path(__file__).resolve().parent)],
                       capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("LOAD_JSON:")]
    if not line:
        raise RuntimeError(p.stdout[-2000:] + p.stderr[-2000:])
    return json.loads(line[-1][len("LOAD_JSON:"):])


def cmd_bench(args):
    """Idle machine only (after 'SLOT StreamStep'). Compile cold (mlpackage -> new .mlmodelc), load cold
    (fresh process, freshly compiled path) and warm (fresh process, the persistent path, twice), then
    per-step time on CPU_AND_NE over --steps calls on real FAST mel after --warmup calls."""
    import shutil
    import tempfile
    import coremltools as ct
    from cmlrun import load_reference, ref_slices
    out = {"time": time.strftime("%F %T"), "mech": args.mech[0], "variant": args.variant, "precision": args.precision,
           "compute_units": args.compute_units, "C": C_DEFAULT}
    mech = args.mech[0]
    pkg, comp = package_path(args.variant, args.precision, mech), compiled_path(args.variant, args.precision, mech)
    tmp = Path(tempfile.mkdtemp(prefix="v1sbench_"))
    t0 = time.perf_counter()
    fresh = ct.models.utils.compile_model(str(pkg), str(tmp / "fresh.mlmodelc"))
    out["compile_s"] = time.perf_counter() - t0
    out["load_cold"] = _child_load(fresh, mech)                                    # never-loaded compiled path
    out["load_warm"] = [_child_load(comp, mech) for _ in range(2)]                 # persistent path
    shutil.rmtree(tmp)
    print("[bench] load " + json.dumps({k: out[k] for k in ("compile_s", "load_cold", "load_warm")}), flush=True)
    m = StepModel(args.variant, args.precision, args.compute_units, mech)
    utts, arrays = load_reference("fast")
    mel = np.concatenate([ref_slices(u, arrays)[0][:u["mel_valid"]] for u in utts])
    need = 16 * (args.steps + args.warmup)
    mel = np.tile(mel, (-(-need // len(mel)), 1))[:need]
    times, pred = [], []
    for j in range(args.steps + args.warmup):
        t0 = time.perf_counter()
        m.step(mel[16 * j:16 * j + 16])
        dt = time.perf_counter() - t0
        if j >= args.warmup:
            times.append(dt)
    t = np.array(times) * 1e3
    out["step_ms"] = {"n": len(t), "warmup": args.warmup, "median": float(np.median(t)), "p90": float(np.percentile(t, 90)),
                      "p99": float(np.percentile(t, 99)), "mean": float(t.mean()), "min": float(t.min()), "max": float(t.max())}
    out["rtf_median"] = out["step_ms"]["median"] / (40.0 * C_DEFAULT)
    import subprocess
    out["load_avg"] = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout.strip()
    print("[bench] " + json.dumps(out), flush=True)
    d = OUT / "stream"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"bench_{args.variant}_{args.precision}_{mech}_{args.compute_units}.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"spec": cmd_spec, "export": cmd_export, "check-torch": cmd_check_torch, "placement": cmd_placement,
            "probe-state": cmd_probe_state, "bench": cmd_bench, "diag": cmd_diag}
    for name in cmds:
        p = sub.add_parser(name)
        p.add_argument("--variant", default="faithful")
        p.add_argument("--precision", default="fp16")
        p.add_argument("--mech", nargs="+", default=list(MECHS), choices=MECHS)
        p.add_argument("--compute-units", default="CPU_AND_NE")
        p.add_argument("--n", type=int, default=200)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--window-check", type=int, default=0)
        p.add_argument("--parts", nargs="+", default=list(_PartStep.PARTS))
        p.add_argument("--steps", type=int, default=3000)
        p.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args()
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
