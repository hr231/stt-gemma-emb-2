"""V1 as one fixed-window PyTorch module for Core ML: mel [1, T_mel, 128] + valid mel frames [1]
-> log-probs [1, T_mel/4, 257] and the L11 frames [1, T_mel/4, 1024].

Gemma4AudioModel.forward up to layers[11] is re-implemented with static shapes, reusing the HF
submodules and weights (transformers 5.19.0, models/gemma4/modeling_gemma4.py):
  - subsampling: HF's two conv layers; the input masks (mel * m, then m[::2]) are rebuilt in the
    graph from the valid mel count V, so frame j of layer 1 is valid iff 2j < V, of L11 iff 4j < V;
  - attention: Gemma4AudioAttention evaluated densely over the window: scores Q K^T for every frame
    pair plus the relative-position term, tanh soft cap 50, a static additive mask (each frame sees
    itself and the 11 frames before it = HF sliding_window_mask_function((12, 0))), softmax, V.
    HF computes the same per 12-frame block with a 24-key context (unfold) and a Transformer-XL
    rel shift (pad / merge / slice / reshape); those ops fall back to the CPU or make the Neural
    Engine compile fail, the dense form is matmuls only (fp32: identical). The relative term
    Q . rel[12 - (q - k)] is Q @ rel^T ([T, 13]) placed on its diagonals by one batched matmul with
    a constant 0/1 tensor G[q, r, k] = (k == q - 12 + r), shared by all layers. HF also masks padded
    keys, which only changes padded query rows: attention and the depthwise conv are causal, so
    valid frames never see padding;
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


class DenseAttention(nn.Module):
    def __init__(self, attn, pos_emb, T):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.post = attn.q_proj, attn.k_proj, attn.v_proj, attn.post
        self.H, self.D = attn.num_heads, attn.head_dim
        assert attn.max_past_horizon == attn.chunk_size == 12 and attn.max_future_horizon == 0
        self.T = T
        self.q_scale, self.k_scale = float(attn.q_scale), float(attn.k_scale)
        self.per_dim_scale = attn.per_dim_scale
        self.softcap = float(attn.softcap)
        with torch.no_grad():
            rel = attn.relative_k_proj(pos_emb).view(-1, self.H, self.D)          # [13, H, D]; row r: distance 12 - r
        self.register_buffer("rel", rel.permute(1, 2, 0).contiguous())           # [H, D, 13]

    def forward(self, h, place, mask):                                           # h [1, T, 1024]
        T, H, D = self.T, self.H, self.D
        q = (self.q_proj(h).view(T, H, D) * (self.q_scale * F.softplus(self.per_dim_scale))).permute(1, 0, 2)
        k = (self.k_proj(h).view(T, H, D) * self.k_scale).permute(1, 2, 0)
        v = self.v_proj(h).view(T, H, D).permute(1, 0, 2)
        bd = ((q @ self.rel).permute(1, 0, 2) @ place).permute(1, 0, 2)          # [H, T, T]
        w = torch.tanh((q @ k + bd) / self.softcap) * self.softcap + mask
        out = torch.softmax(w, dim=-1) @ v                                       # [H, T, D]
        return self.post(out.permute(1, 0, 2).reshape(1, T, H * D))


def attention_constants(T, n_rel=13, window=12):
    """place [T, 13, T]: place[q, r, k] = 1 where k = q - 12 + r; mask [T, T]: 0 where 0 <= q - k < 12."""
    t = torch.arange(T)
    dist = t[:, None] - t[None, :]                                               # query - key
    place = torch.stack([dist == (n_rel - 1 - r) for r in range(n_rel)], 1).float()
    mask = torch.where((dist >= 0) & (dist < window), 0.0, NEG)
    return place, mask


class TowerLayer(nn.Module):
    def __init__(self, layer, attn):
        super().__init__()
        self.ff1, self.ff2, self.lconv = layer.feed_forward1, layer.feed_forward2, layer.lconv1d
        self.norm_pre_attn, self.norm_post_attn, self.norm_out = layer.norm_pre_attn, layer.norm_post_attn, layer.norm_out
        self.attn = attn

    def forward(self, h, place, mask):
        h = self.ff1(h)
        res = h
        h = self.attn(self.norm_pre_attn(h.clamp(-FP16_MAX, FP16_MAX)), place, mask)
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
        self.layers = nn.ModuleList(TowerLayer(l, DenseAttention(l.self_attn, pos, self.T)) for l in tower.layers[:12])
        place, amask = attention_constants(self.T)
        self.register_buffer("attn_place", place)                                # shared by all 12 layers
        self.register_buffer("attn_mask", amask)
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
            h = layer(h, self.attn_place, self.attn_mask)
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
        masks = {f: torch.where(torch.logical_or(torch.logical_and(getattr(self, f"band_{f}"), keyv), self.eye), 0.0, NEG)
                 for f in set(hd.futures)}
        vf = valid.to(z.dtype)
        h = hd.inp(z)
        B, T, D = 1, self.T, hd.inp.out_features
        for blk, f in zip(hd.blocks, hd.futures):
            q, k, v = blk.qkv(blk.ln_att(h)).view(B, T, 3, blk.heads, D // blk.heads).permute(2, 0, 3, 1, 4).unbind(0)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=masks[f])
            h = h + blk.proj(a.transpose(1, 2).reshape(B, T, D))
            c = (blk.ln_conv(h) * vf).transpose(1, 2)
            h = h + blk.conv(F.pad(c, blk.conv_pad)).transpose(1, 2)
            h = h + blk.ff(blk.ln_ff(h))
        logits = hd.out(hd.ln(h))
        # coremltools lowers log_softmax to log(softmax(x)), which underflows in fp16 (log(0) for every
        # non-dominant token); (x - max) - log(sum(exp(x - max))) never takes the log of a tiny number.
        s = logits - logits.amax(-1, keepdim=True)
        return s - torch.log(torch.exp(s).sum(-1, keepdim=True))


def build(t_mel, **opts):
    """V1Window for a window of t_mel mel frames, from the V1 files (fp32, CPU, eval)."""
    from common import check_v1_files, load_head_cpu, load_tower_fp32
    check_v1_files()
    base = load_tower_fp32()
    head, ck = load_head_cpu()
    return V1Window(base.audio_tower, head, ck["mean"], ck["std"], t_mel, **opts).eval()
