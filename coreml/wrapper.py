"""V1 as one fixed-window PyTorch module for Core ML: mel [1, T_mel, 128] + valid mel frames [1]
-> log-probs [1, T_mel/4, 257] and the L11 frames [1, T_mel/4, 1024].

Gemma4AudioModel.forward up to layers[11] is re-implemented with static shapes, reusing the HF
submodules and weights (transformers 5.19.0, models/gemma4/modeling_gemma4.py):
  - subsampling: HF's two conv layers; the input masks (mel * m, then m[::2]) are rebuilt in the
    graph from the valid mel count V, so frame j of layer 1 is valid iff 2j < V, of L11 iff 4j < V;
  - attention: Gemma4AudioAttention in its blocked form (12-frame blocks, 24-key context = previous
    + current block, Transformer-XL rel shift, tanh soft cap 50), with `unfold` replaced by
    pad -> reshape into blocks -> concat(block b-1, block b), and the mask by a static additive
    mask (each frame sees itself and the 11 frames before it; HF sliding_window_mask_function
    ((12, 0))). HF also masks padded keys, which only changes padded query rows: attention and the
    depthwise conv are causal, so valid frames never see padding;
  - FFN, light conv, RMSNorm, clipped linears: HF modules; the +-1e10 "gradient clipping" clamps
    are set to the fp16 maximum (HF does the same for fp16 weights).
Then the int8 round trip (featstore.quantize_rows / dequantize_rows), standardization with the V1
mean / std, zeroing beyond the valid L11 length (probe.py: padding = 0 after standardizing), and the
V1 head (head.CausalHead) with its boolean masks (band within [-64, +future] and valid keys, or the
diagonal) as additive masks built from the valid length.

fp16 guards that do not change fp32 results:
  - 1/std is capped at 6e4: 8 L11 dims have std 1e-5 (1/std = 1e5 overflows fp16); their values are
    exactly 0 after the int8 round trip (mean 0), so 0 * 6e4 = 0 as in V1;
  - masks: where(valid, z, 0) (NaN-safe), additive masks use -1e4 (logits are soft-capped to +-50).
Options: int8 (default True), rms_scale (None = HF RMSNorm; s = power-of-two pre-scale x / s before
squaring, mathematically identical, keeps x^2 within fp16), pad_fill (fill padded tower input
rows with a constant instead of 0 so no RMSNorm sees an all-zero row; padded rows never reach
valid frames).
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4
FP16_MAX = 65504.0
INV_STD_MAX = 6e4


class RMSNormExport(nn.Module):
    """Gemma4RMSNorm: x * (mean(x^2) + eps)^-0.5 * w; with scale s: (x/s) * (mean((x/s)^2) + eps/s^2)^-0.5 * w."""

    def __init__(self, norm, scale=None):
        super().__init__()
        self.eps = norm.eps
        self.weight = norm.weight if norm.with_scale else None
        self.scale = scale

    def forward(self, x):
        if self.scale is not None:
            x = x * (1.0 / self.scale)
            y = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.eps / self.scale ** 2, -0.5)
        else:
            y = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        return y * self.weight if self.weight is not None else y


class BlockedAttention(nn.Module):
    def __init__(self, attn, pos_emb, T):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.post = attn.q_proj, attn.k_proj, attn.v_proj, attn.post
        self.H, self.D, self.C = attn.num_heads, attn.head_dim, attn.chunk_size
        assert attn.max_past_horizon == self.C and attn.max_future_horizon == 0 and attn.context_size == 2 * self.C
        self.T, self.nb = T, -(-T // self.C)
        self.q_scale, self.k_scale = float(attn.q_scale), float(attn.k_scale)
        self.per_dim_scale = attn.per_dim_scale
        self.softcap = float(attn.softcap)
        with torch.no_grad():
            rel = attn.relative_k_proj(pos_emb).view(-1, self.H, self.D)          # [13, H, D]
        self.register_buffer("rel", rel.permute(1, 2, 0).contiguous())           # [H, D, 13]
        i = torch.arange(self.C)[:, None]
        j = torch.arange(2 * self.C)[None, :]
        b = torch.arange(self.nb)[:, None, None]
        dist = self.C + i - j                                                    # query - key
        ok = (dist >= 0) & (dist < self.C) & (b * self.C - self.C + j >= 0)      # [nb, 12, 24]
        self.register_buffer("mask", torch.where(ok, 0.0, NEG)[None])           # [1, nb, 12, 24]

    def _ctx(self, x):                                                           # [T, H, D] -> [nb, 24, H, D]
        xp = F.pad(x, (0, 0, 0, 0, self.C, self.nb * self.C - self.T)).view(self.nb + 1, self.C, self.H, self.D)
        return torch.cat([xp[:-1], xp[1:]], dim=1)

    def forward(self, h):                                                        # h [1, T, 1024]
        T, H, D, C, nb = self.T, self.H, self.D, self.C, self.nb
        q = self.q_proj(h).view(T, H, D) * (self.q_scale * F.softplus(self.per_dim_scale))
        k = self.k_proj(h).view(T, H, D) * self.k_scale
        v = self.v_proj(h).view(T, H, D)
        qb = F.pad(q, (0, 0, 0, 0, 0, nb * C - T)).view(nb, C, H, D).permute(2, 0, 1, 3)   # [H, nb, 12, D]
        kc = self._ctx(k).permute(2, 0, 3, 1)                                    # [H, nb, D, 24]
        vc = self._ctx(v).permute(2, 0, 1, 3)                                    # [H, nb, 24, D]
        ac = qb @ kc                                                             # [H, nb, 12, 24]
        bd = (qb.reshape(H, nb * C, D) @ self.rel).view(H, nb, C, -1)           # [H, nb, 12, 13]
        bd = F.pad(bd, (0, 2 * C + 1 - bd.shape[-1])).view(H, nb, C * (2 * C + 1))[..., :C * 2 * C]
        w = ac + bd.view(H, nb, C, 2 * C)
        w = torch.tanh(w / self.softcap) * self.softcap + self.mask
        out = torch.softmax(w, dim=-1) @ vc                                      # [H, nb, 12, D]
        out = out.permute(1, 2, 0, 3).reshape(nb * C, H * D)[:T]
        return self.post(out[None])


class TowerLayer(nn.Module):
    def __init__(self, layer, attn):
        super().__init__()
        self.ff1, self.ff2, self.lconv = layer.feed_forward1, layer.feed_forward2, layer.lconv1d
        self.norm_pre_attn, self.norm_post_attn, self.norm_out = layer.norm_pre_attn, layer.norm_post_attn, layer.norm_out
        self.attn = attn

    def forward(self, h):
        h = self.ff1(h)
        res = h
        h = self.attn(self.norm_pre_attn(h.clamp(-FP16_MAX, FP16_MAX)))
        h = self.norm_post_attn(h.clamp(-FP16_MAX, FP16_MAX)) + res
        h = self.ff2(self.lconv(h))
        return self.norm_out(h.clamp(-FP16_MAX, FP16_MAX))


class V1Window(nn.Module):
    def __init__(self, tower, head, mean, std, t_mel, int8=True, rms_scale=None, pad_fill=None):
        super().__init__()
        tower = copy.deepcopy(tower).float().eval()
        for name, m in list(tower.named_modules()):
            for cname, c in list(m.named_children()):
                if type(c).__name__ == "Gemma4RMSNorm":
                    setattr(m, cname, RMSNormExport(c, rms_scale))
            if hasattr(m, "gradient_clipping"):
                m.gradient_clipping = FP16_MAX
        self.t_mel, self.T = t_mel, -(-t_mel // 4)
        assert self.T == -(-(-(-t_mel // 2)) // 2)
        self.sscp = tower.subsample_conv_projection
        with torch.no_grad():
            pos = tower.rel_pos_enc(torch.zeros(1, 1, tower.config.hidden_size))
        self.layers = nn.ModuleList(TowerLayer(l, BlockedAttention(l.self_attn, pos, self.T)) for l in tower.layers[:12])
        self.head = copy.deepcopy(head).float().eval()
        self.int8, self.pad_fill = int8, pad_fill
        self.register_buffer("mean", mean.float())
        self.register_buffer("inv_std", (1.0 / std.float()).clamp(max=INV_STD_MAX))
        self.register_buffer("t0", torch.arange(t_mel, dtype=torch.int32))
        self.register_buffer("t1", torch.arange(-(-t_mel // 2), dtype=torch.int32) * 2)
        self.register_buffer("t2", torch.arange(self.T, dtype=torch.int32) * 4)
        t = torch.arange(self.T)
        rel = t[None, :] - t[:, None]                                            # key - query
        for f in sorted(set(self.head.futures)):
            band = (rel >= -self.head.past) & (rel <= f)
            self.register_buffer(f"band_{f}", band[None, None])
        self.register_buffer("eye", torch.eye(self.T, dtype=torch.bool)[None, None])

    def _sscp_layer(self, l, x, m):
        x = x * m[None, None, :, None]
        return l.act(l.norm(l.conv(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

    def forward(self, mel, valid_len):                                           # mel [1, T_mel, 128], valid_len [1] int32
        V = valid_len
        x = self._sscp_layer(self.sscp.layer0, mel[:, None], (self.t0 < V).to(mel.dtype))
        x = self._sscp_layer(self.sscp.layer1, x, (self.t1 < V).to(mel.dtype))
        x = x.permute(0, 2, 3, 1).reshape(1, self.T, -1)
        h = self.sscp.input_proj_linear(x)
        valid = (self.t2 < V)[None, :, None]                                     # [1, T, 1] bool
        if self.pad_fill is not None:
            h = torch.where(valid, h, torch.full_like(h, self.pad_fill))
        for layer in self.layers:
            h = layer(h)
        l11 = h
        if self.int8:
            scale = l11.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127.0
            x = torch.round(l11 / scale).clamp(-127, 127) * scale
        else:
            x = l11
        z = torch.where(valid, (x - self.mean) * self.inv_std, torch.zeros_like(x))
        return self._head(z, valid), l11

    def _head(self, z, valid):
        hd = self.head
        keyv = valid[:, :, 0][:, None, None, :]                                   # [1, 1, 1, T]
        masks = {f: torch.where((getattr(self, f"band_{f}") & keyv) | self.eye, 0.0, NEG) for f in set(hd.futures)}
        vf = valid.to(z.dtype)
        h = hd.inp(z)
        for blk, f in zip(hd.blocks, hd.futures):
            B, T, D = h.shape
            q, k, v = blk.qkv(blk.ln_att(h)).view(B, T, 3, blk.heads, D // blk.heads).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=masks[f])
            h = h + blk.proj(a.transpose(1, 2).reshape(B, T, D))
            c = (blk.ln_conv(h) * vf).transpose(1, 2)
            h = h + blk.conv(F.pad(c, blk.conv_pad)).transpose(1, 2)
            h = h + blk.ff(blk.ln_ff(h))
        return hd.out(hd.ln(h)).log_softmax(-1)


def build(t_mel, **opts):
    """V1Window for a window of t_mel mel frames, from the V1 files (fp32, CPU, eval)."""
    from common import check_v1_files, load_head_cpu, load_tower_fp32
    check_v1_files()
    base = load_tower_fp32()
    head, ck = load_head_cpu()
    return V1Window(base.audio_tower, head, ck["mean"], ck["std"], t_mel, **opts).eval()
