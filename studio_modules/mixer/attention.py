"""Causal multi-head softmax attention."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from studio_modules.ops import linear, linear_back, merge_heads, softmax, split_heads
from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Issue, Layer


@dataclass
class AttentionConfig:
    heads: int = field(default=4, metadata=dict(min=1, max=64, help="Parallel attention heads. d_model must divide evenly; each head works in d_model ÷ heads dimensions."))
    bias: bool = field(default=True, metadata=dict(help="Bias terms on the QKV and output projections (GPT-2 has them; many later models drop them)."))


class Attention(Layer):
    def __init__(self, cfg, dims):
        super().__init__(dims)
        d = dims.d_model
        self.heads = cfg.heads
        self.add("qkv.w", (d, 3 * d))
        self.add("proj.w", (d, d), "residual_out")
        if cfg.bias:
            self.add("qkv.b", (3 * d,), "bias")
            self.add("proj.b", (d,), "bias")

    def forward(self, x, ctx):
        xp, h = self.xp, self.heads
        length, depth = x.shape[1], x.shape[2] // h
        position = ctx.position
        q, k, v = [split_heads(part, h) for part in xp.split(linear(self, x, "qkv"), 3, axis=-1)]
        q, k = position.rotate(q, k)
        q = ctx.emit("attn.q", q)
        k = ctx.emit("attn.k", k)
        v = ctx.emit("attn.v", v)
        if hasattr(ctx, "attention_summary"):
            from studio_summary import chunked_attention
            output, mass = chunked_attention(q, k, v, position, ctx.attention_summary, xp=xp)
            ctx.summaries[f"block{ctx.block}.attn.pattern"] = mass
            self.cache = None
            return linear(self, merge_heads(output), "proj")
        raw = ctx.emit("attn.scores", q @ k.swapaxes(-1, -2))
        scores = raw / math.sqrt(depth)
        bias = position.score_bias(length, h)
        if bias is not None:
            scores = scores + bias
        scores = xp.where(ctx.mask(length), -1e9, scores)
        probabilities = ctx.emit("attn.pattern", softmax(scores, xp))
        attention, mask = ctx.drop(probabilities)
        context = merge_heads(ctx.emit("attn.head_out", attention @ v))
        self.cache = (x, q, k, v, probabilities, attention, mask, context, position)
        return linear(self, context, "proj")

    def backward(self, dy):
        x, q, k, v, probabilities, attention, mask, context, position = self.cache
        depth = x.shape[2] // self.heads
        dcontext = split_heads(linear_back(self, dy, context, "proj"), self.heads)
        dv = attention.swapaxes(-1, -2) @ dcontext
        dattention = dcontext @ v.swapaxes(-1, -2)
        if mask is not None:
            dattention = dattention * mask
        # softmax backward, one row (query) at a time; masked keys have probability 0 so they get no gradient
        dscores = probabilities * (dattention - (dattention * probabilities).sum(axis=-1, keepdims=True)) / math.sqrt(depth)
        dq, dk = position.rotate_back(dscores @ k, dscores.swapaxes(-1, -2) @ q)
        dqkv = self.xp.concatenate([merge_heads(value) for value in (dq, dk, dv)], axis=-1)
        return linear_back(self, dqkv, x, "qkv")


@register
class AttentionSpec(ModuleSpec):
    id = "mixer/attention"
    slot = "mixer"
    name = "Softmax attention"
    blurb = "Every position reads earlier positions, weighted by softmax(q·k / √d_head). Cost grows with T²."
    ref = "Vaswani et al. 2017"
    config = AttentionConfig
    provides = frozenset({"qk", "scores", "attention"})
    glyph3d = {"kind": "beams", "hooks": ["attn.pattern"]}
    hooks = (("attn.q", "B H T h"), ("attn.k", "B H T h"), ("attn.v", "B H T h"), ("attn.scores", "B H T T"),
             ("attn.pattern", "B H T T"), ("attn.head_out", "B H T h"))

    def symbols(self, cfg, dims):
        return {"H": cfg.heads, "h": dims.d_model // cfg.heads}

    def estimate(self, cfg, dims):
        d = dims.d_model
        params = 4 * d * d + (4 * d if cfg.bias else 0)
        return Estimate(params=params, flops_per_token=2 * params + 2 * dims.n_ctx * d,
                        activation_bytes=(5 * d + 2 * cfg.heads * dims.n_ctx) * dims.bytes, kv_bytes=2 * d * dims.bytes)

    def validate(self, cfg, dims):
        if dims.d_model % cfg.heads:
            return [Issue("ERROR", f"d_model {dims.d_model} must be divisible by its {cfg.heads} attention heads.")]
        return []

    def build(self, cfg, dims, rng):
        return Attention(cfg, dims)
