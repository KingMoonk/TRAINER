"""Causal multi-head softmax attention."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import torch

from studio_modules.ops import linear, merge_heads, split_heads
from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Issue, Layer


@dataclass
class AttentionConfig:
    heads: int = field(default=4, metadata=dict(min=1, max=64, help="Parallel attention heads. d_model must divide evenly; each head works in d_model ÷ heads dimensions."))
    bias: bool = field(default=True, metadata=dict(help="Bias terms on the QKV and output projections (GPT-2 has them; many later models drop them)."))


HALF = (torch.bfloat16, torch.float16)
INSPECTED = ("attn.scores", "attn.logits", "attn.pattern")  # the intermediates only the hand-written path can show


def fused_attention(ctx, q, k, v, scale, position, heads):
    """PyTorch's fused kernel, or None when this call must take the exact path (see causal_softmax_attention)."""
    length, keys = q.shape[-2], k.shape[-2]
    decode = keys != length   # with a KV cache: the queries start after the cached keys, so a one-token query may read every key and a chunk is masked from its own offset
    if any(ctx.wants(name) for name in INSPECTED):
        return None   # an inspection page is listening for the scores, logits or pattern
    p = ctx.dropout if ctx.training else 0.
    if p and ctx.device not in ("cpu", "cuda"):
        return None   # the fused kernel has no dropout on Apple GPUs
    if decode:
        bias = position.score_bias_rows(ctx.offset, ctx.offset + length, keys, heads) if hasattr(position, "score_bias_rows") else position.score_bias(keys, heads)
    else:
        bias = position.score_bias(length, heads)
    if bias is not None and (q.dtype in HALF or torch.is_autocast_enabled(q.device.type)):
        return None   # a bias such as ALiBi's reaches hundreds and loses its meaning in half precision; the exact path adds it in float32
    mask = None
    if decode and length > 1:   # a prefill chunk after the first: row i reads the keys up to its own absolute index
        future = torch.triu(torch.ones((length, keys), dtype=torch.bool, device=q.device), ctx.offset + 1)
        hidden = torch.zeros((length, keys), dtype=q.dtype, device=q.device).masked_fill(future, -torch.inf)
        mask = hidden if bias is None else (bias + hidden)[None]   # [L, K] without a bias: PyTorch's CPU flash kernel declines a [1, L, K] mask
    elif bias is not None:
        mask = (bias if decode else bias + torch.zeros_like(bias).masked_fill(ctx.mask(length), -torch.inf))[None]
    seed = ctx.attention_seed(q.device) if p else 0
    cuda = [q.device.index if q.device.index is not None else torch.cuda.current_device()] if q.device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda, enabled=bool(p), device_type=q.device.type):
        if p:
            torch.manual_seed(seed)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p, is_causal=mask is None and not decode, scale=scale)
    return out


def causal_softmax_attention(ctx, q, k, v, scale, position, heads):
    """Scores → scale → bias → emit logits → mask → softmax → dropout → weighted values. Shared by attention and GQA.

    Takes PyTorch's fused kernel whenever nothing needs the intermediates (fused_attention); otherwise every step is explicit,
    so inspection pages can read attn.scores, attn.logits and attn.pattern. Both paths give the same outputs and gradients.
    """
    fused = fused_attention(ctx, q, k, v, scale, position, heads)
    if fused is not None:
        return merge_heads(ctx.emit("attn.head_out", fused))
    length, offset = q.shape[-2], ctx.offset
    raw = ctx.emit("attn.scores", q @ k.transpose(-1, -2))
    scores = (raw.float() if raw.dtype in HALF else raw) * scale  # scores, mask and softmax stay in float32 under mixed precision
    bias = position.score_bias(length, heads) if not offset or not hasattr(position, "score_bias_rows") else position.score_bias_rows(offset, offset + length, k.shape[-2], heads)
    if bias is not None:
        scores = scores + bias
    scores = ctx.emit("attn.logits", scores)
    mask = ctx.mask(length) if not offset else torch.triu(torch.ones((length, k.shape[-2]), dtype=torch.bool, device=ctx.device), offset + 1)
    scores = scores.masked_fill(mask, -1e9)
    probabilities = ctx.emit("attn.pattern", torch.softmax(scores, dim=-1))
    attention, _ = ctx.drop(probabilities)
    return merge_heads(ctx.emit("attn.head_out", attention @ v))


@lru_cache(maxsize=None)
def reduced_dtype(device):
    """The half-precision dtype a KV cache is stored in on `device`: bfloat16 where the device really holds and converts it, float16 otherwise.

    A cache only stores and converts; it never multiplies in this dtype, so every CPU qualifies for bfloat16. A CUDA GPU
    needs bfloat16 support (Ampere or newer), and an Apple GPU needs a PyTorch/macOS pair that runs bfloat16 tensors (probed).
    bfloat16 keeps float32's range at about 3 significant digits; float16 has more digits but overflows beyond 65504.
    """
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    try:
        probe = torch.ones((2, 2), dtype=torch.bfloat16, device=device)
        if device == "cpu" or float(torch.cat((probe, probe), dim=-2).float().sum()) == 8.:
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


class KVCache(dict):
    """The `cache` of CompiledModel.forward: {"n": tokens cached, block index: (keys, values)}, a plain dict plus the dtype new entries are stored in.

    dtype=None keeps keys and values as the forward pass computes them (float32 without autocast), which is what a plain
    `{"n": 0}` does. A half dtype stores them in that dtype, halving the bytes per cached token; attention upcasts them to the
    compute dtype when it reads them. An entry keeps the dtype it was first stored in, so a plain dict copy of this cache still works.
    """

    def __init__(self, dtype=None):
        super().__init__(n=0)
        self.dtype = dtype


def new_cache(device, reduced=True):
    """A fresh cache for generating on `device`: reduced=True stores keys and values in reduced_dtype(device), False exactly as computed."""
    return KVCache(reduced_dtype(device) if reduced else None)


def cached(ctx, k, v):
    """With a KV cache: append this pass's keys/values to the block's stored ones and return them all, in the dtype they were computed in
    (the cache may hold them in half precision; the maths of attending stays that of the compute dtype)."""
    if ctx.cache is not None:
        if ctx.block in ctx.cache:
            old_k, old_v = ctx.cache[ctx.block]
            stored = torch.cat((old_k, k.to(old_k.dtype)), dim=-2), torch.cat((old_v, v.to(old_v.dtype)), dim=-2)
            ctx.cache[ctx.block] = stored
            return stored[0].to(k.dtype), stored[1].to(v.dtype)
        dtype = k.dtype if k.dtype in HALF else getattr(ctx.cache, "dtype", None) or k.dtype  # a half-precision pass is already stored as it is
        ctx.cache[ctx.block] = (k.to(dtype, copy=True), v.to(dtype, copy=True))  # k and v are views of the q/k/v projection; keep only their own bytes
    return k, v


class Attention(Layer):
    cacheable = True

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
        q, k = position.rotate(q, k, ctx.offset)
        q = ctx.emit("attn.q", q)
        k = ctx.emit("attn.k", k)
        v = ctx.emit("attn.v", v)
        k, v = cached(ctx, k, v)
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
