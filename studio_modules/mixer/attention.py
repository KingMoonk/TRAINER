"""Causal multi-head softmax attention."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from studio_modules.ops import linear, merge_heads, split_heads
from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Issue, Layer


@dataclass
class AttentionConfig:
    heads: int = field(default=4, metadata=dict(min=1, max=64, help="Parallel attention heads. d_model must divide evenly; each head works in d_model ÷ heads dimensions."))
    bias: bool = field(default=True, metadata=dict(help="Bias terms on the QKV and output projections (GPT-2 has them; many later models drop them)."))


def causal_softmax_attention(ctx, q, k, v, scale, position, heads):
    """Scores → scale → bias → emit logits → mask → softmax → dropout → weighted values. Shared by attention and GQA."""
    length = q.shape[-2]
    raw = ctx.emit("attn.scores", q @ k.transpose(-1, -2))
    scores = raw * scale
    bias = position.score_bias(length, heads)
    if bias is not None:
        scores = scores + bias
    scores = ctx.emit("attn.logits", scores)
    scores = scores.masked_fill(ctx.mask(length), -1e9)
    probabilities = ctx.emit("attn.pattern", torch.softmax(scores, dim=-1))
    attention, _ = ctx.drop(probabilities)
    return merge_heads(ctx.emit("attn.head_out", attention @ v))


class Attention(Layer):
    def __init__(self, cfg, dims):
        super().__init__(dims)
        d = dims.d_model
        self.heads = cfg.heads
        self.score_scale = 1.0  # μP: set by init/mup if width_mult != 1
        self.add("qkv.w", (d, 3 * d))
        self.add("proj.w", (d, d), "residual_out")
        if cfg.bias:
            self.add("qkv.b", (3 * d,), "bias")
            self.add("proj.b", (d,), "bias")

    def forward(self, x, ctx):
        h = self.heads
        depth = x.shape[2] // h
        position = ctx.position
        q, k, v = [split_heads(part, h) for part in linear(self, x, "qkv").chunk(3, dim=-1)]
        q, k = position.rotate(q, k)
        q = ctx.emit("attn.q", q)
        k = ctx.emit("attn.k", k)
        v = ctx.emit("attn.v", v)
        if hasattr(ctx, "attention_summary"):
            from studio_summary import chunked_attention
            output, mass = chunked_attention(q, k, v, position, ctx.attention_summary)
            ctx.summaries[f"block{ctx.block}.attn.pattern"] = mass
            return linear(self, merge_heads(output), "proj")
        context = causal_softmax_attention(ctx, q, k, v, self.score_scale / math.sqrt(depth), position, h)
        return linear(self, context, "proj")


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
             ("attn.logits", "B H T T"), ("attn.pattern", "B H T T"), ("attn.head_out", "B H T h"))

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
