"""Grouped-query causal attention: multiple query heads share each KV head."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from studio_modules.ops import linear, linear_back, merge_heads, softmax, split_heads
from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Issue, Layer


@dataclass
class GQAConfig:
    heads: int = field(default=4, metadata=dict(min=1, max=64))
    kv_heads: int = field(default=2, metadata=dict(min=1, max=64, help="One key/value head per group; 1 is MQA."))
    bias: bool = True


class GQALayer(Layer):
    def __init__(self, cfg, dims):
        super().__init__(dims)
        d = dims.d_model
        self.heads, self.kv_heads = cfg.heads, cfg.kv_heads
        self.depth = d // cfg.heads
        self.group = cfg.heads // cfg.kv_heads
        self.score_scale = 1.0  # μP: set by init/mup if width_mult != 1
        kv_width = cfg.kv_heads * self.depth
        self.add("gqa.q.w", (d, d))
        self.add("gqa.k.w", (d, kv_width))
        self.add("gqa.v.w", (d, kv_width))
        self.add("gqa.proj.w", (d, d), "residual_out")
        if cfg.bias:
            for name, width in (("gqa.q", d), ("gqa.k", kv_width), ("gqa.v", kv_width), ("gqa.proj", d)):
                self.add(name + ".b", (width,), "bias")

    def forward(self, x, ctx):
        xp, h, groups = self.xp, self.heads, self.group
        length = x.shape[1]
        position = ctx.position
        q = split_heads(linear(self, x, "gqa.q"), h)
        k = split_heads(linear(self, x, "gqa.k"), self.kv_heads)
        v = split_heads(linear(self, x, "gqa.v"), self.kv_heads)
        q, k = position.rotate(q, k)
        q = ctx.emit("attn.q", q)
        k = ctx.emit("attn.k", k)
        v = ctx.emit("attn.v", v)
        expanded_k = xp.repeat(k, groups, axis=1)
        expanded_v = xp.repeat(v, groups, axis=1)
        if hasattr(ctx, "attention_summary"):
            from studio_summary import chunked_attention
            output, mass = chunked_attention(q, expanded_k, expanded_v, position, ctx.attention_summary, xp=xp)
            ctx.summaries[f"block{ctx.block}.attn.pattern"] = mass
            self.cache = None
            return linear(self, merge_heads(output), "gqa.proj")
        raw = ctx.emit("attn.scores", q @ expanded_k.swapaxes(-1, -2))
        scores = raw / math.sqrt(self.depth) * self.score_scale
        bias = position.score_bias(length, h)
        if bias is not None:
            scores = scores + bias
        scores = xp.where(ctx.mask(length), -1e9, scores)
        probabilities = ctx.emit("attn.pattern", softmax(scores, xp))
        attention, mask = ctx.drop(probabilities)
        context = merge_heads(ctx.emit("attn.head_out", attention @ expanded_v))
        self.cache = (x, q, expanded_k, expanded_v, probabilities, attention, mask, context, position)
        return linear(self, context, "gqa.proj")

    def backward(self, dy):
        x, q, k, v, probabilities, attention, mask, context, position = self.cache
        dcontext = split_heads(linear_back(self, dy, context, "gqa.proj"), self.heads)
        dv = attention.swapaxes(-1, -2) @ dcontext
        dattention = dcontext @ v.swapaxes(-1, -2)
        if mask is not None:
            dattention *= mask
        dscores = probabilities * (dattention - (dattention * probabilities).sum(axis=-1, keepdims=True)) / math.sqrt(self.depth) * self.score_scale
        dq = dscores @ k
        dk = dscores.swapaxes(-1, -2) @ q
        shape = (x.shape[0], self.kv_heads, self.group, x.shape[1], self.depth)
        dk = dk.reshape(shape).sum(axis=2)
        dv = dv.reshape(shape).sum(axis=2)
        dq, dk = position.rotate_back(dq, dk)
        return (linear_back(self, merge_heads(dq), x, "gqa.q")
                + linear_back(self, merge_heads(dk), x, "gqa.k")
                + linear_back(self, merge_heads(dv), x, "gqa.v"))


@register
class GQA(ModuleSpec):
    id = "mixer/gqa"
    slot = "mixer"
    name = "Grouped-query attention"
    blurb = "Query heads share fewer key/value heads; one KV head is multi-query attention."
    ref = "Ainslie et al. 2023"
    config = GQAConfig
    provides = frozenset({"qk", "scores", "attention"})
    hooks = (("attn.q", "B H T h"), ("attn.k", "B G T h"), ("attn.v", "B G T h"),
             ("attn.scores", "B H T T"), ("attn.pattern", "B H T T"), ("attn.head_out", "B H T h"))
    glyph3d = {"kind": "beams", "hooks": ["attn.pattern"]}
    # A constant key bias shifts every softmax row equally, so its gradient is zero.
    zero_grad_params = frozenset({"gqa.k.b"})

    def symbols(self, cfg, dims):
        return {"H": cfg.heads, "G": cfg.kv_heads, "h": dims.d_model // cfg.heads}

    def estimate(self, cfg, dims):
        d, kv = dims.d_model, cfg.kv_heads * (dims.d_model // cfg.heads)
        params = 2 * d * d + 2 * d * kv + (2 * d + 2 * kv if cfg.bias else 0)
        return Estimate(params=params, flops_per_token=2 * params + 2 * dims.n_ctx * d,
                        activation_bytes=(3 * d + 2 * kv + 2 * cfg.heads * dims.n_ctx) * dims.bytes,
                        kv_bytes=2 * kv * dims.bytes)

    def validate(self, cfg, dims):
        if dims.d_model % cfg.heads or cfg.heads % cfg.kv_heads:
            return [Issue("ERROR", f"d_model {dims.d_model} must divide into {cfg.heads} query heads, and query heads must divide into {cfg.kv_heads} KV heads.")]
        return []

    def build(self, cfg, dims, rng):
        return GQALayer(cfg, dims)
