"""Normalized causal linear attention with ELU+1 features and cumulative sums."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from studio_modules.ops import linear, merge_heads, split_heads
from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Issue, Layer


@dataclass
class LinearConfig:
    heads: int = field(default=4, metadata=dict(min=1, max=64))
    bias: bool = True
    eps: float = field(default=1e-6, metadata=dict(min=1e-12, max=1e-2))


def prefix_sums(x):
    """Running sums over the time axis (dim 2) of [B, H, T, ...]. A single token has nothing to add up, so it is returned as is:
    on Apple MPS, torch.cumsum over a length-1 axis whose stride is odd (k[..., :, None] * v[..., None, :] produces one) writes
    outside its output and corrupts neighbouring tensors, which showed up as NaN or garbage logits while generating."""
    return x if x.shape[2] == 1 else torch.cumsum(x, dim=2)


class LinearAttention(Layer):
    cacheable = True

    def __init__(self, cfg, dims):
        super().__init__(dims)
        d = dims.d_model
        self.heads, self.eps = cfg.heads, cfg.eps
        self.add("linear.qkv.w", (d, 3 * d))
        self.add("linear.proj.w", (d, d), "residual_out")
        if cfg.bias:
            self.add("linear.qkv.b", (3 * d,), "bias")
            self.add("linear.proj.b", (d,), "bias")

    def forward(self, x, ctx):
        raw_q, raw_k, v = [split_heads(part, self.heads) for part in linear(self, x, "linear.qkv").chunk(3, dim=-1)]
        q = ctx.emit("linear.q", F.elu(raw_q) + 1)
        k = ctx.emit("linear.k", F.elu(raw_k) + 1)
        v = ctx.emit("linear.v", v)
        # Causal prefix sums of k_i v_j and of k; each position reads only earlier positions. Memory: T × h² per head.
        kv = prefix_sums(k.unsqueeze(-1) * v.unsqueeze(-2))
        ks = prefix_sums(k)
        if ctx.cache is not None:
            # Generation: the sums so far are the whole state ([B, H, h, h] and [B, H, h]), so a new token costs the same at any length.
            if ctx.block in ctx.cache:
                kv, ks = kv + ctx.cache[ctx.block][0].unsqueeze(2), ks + ctx.cache[ctx.block][1].unsqueeze(2)
            ctx.cache[ctx.block] = (kv[:, :, -1].clone(), ks[:, :, -1].clone())  # a copy, so the [T, h, h] sums are freed
        numerator = torch.einsum("bhti,bhtij->bhtj", q, kv)
        denominator = (q * ks).sum(dim=-1, keepdim=True) + self.eps
        stream = ctx.emit("linear.out", numerator / denominator)
        return linear(self, merge_heads(stream), "linear.proj")


@register
class Linear(ModuleSpec):
    id = "mixer/linear"
    slot = "mixer"
    name = "Causal linear attention"
    blurb = "ELU+1 feature maps with recurrent numerator and denominator states."
    ref = "Katharopoulos et al. 2020"
    config = LinearConfig
    provides = frozenset({"ordered"})
    hooks = (("linear.q", "B H T h"), ("linear.k", "B H T h"), ("linear.v", "B H T h"),
             ("linear.out", "B H T h"))
    glyph3d = {"kind": "ribbon", "hooks": ["linear.out"]}

    def symbols(self, cfg, dims):
        return {"H": cfg.heads, "h": dims.d_model // cfg.heads}

    def estimate(self, cfg, dims):
        d = dims.d_model
        params = 4 * d * d + (4 * d if cfg.bias else 0)
        return Estimate(params=params, flops_per_token=2 * params + 4 * d * d // cfg.heads,
                        activation_bytes=(5 * d + 2 * d * d // cfg.heads) * dims.bytes,
                        kv_bytes=(d + d * d // cfg.heads) * dims.bytes)

    def validate(self, cfg, dims):
        if dims.d_model % cfg.heads:
            return [Issue("ERROR", f"d_model {dims.d_model} must be divisible by {cfg.heads} linear-attention heads.")]
        return []

    def build(self, cfg, dims, rng):
        return LinearAttention(cfg, dims)
