"""Normalized causal linear attention with ELU+1 features and recurrent sums."""
from __future__ import annotations

from dataclasses import dataclass, field

from studio_modules.ops import linear, linear_back, merge_heads, split_heads
from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Issue, Layer


@dataclass
class LinearConfig:
    heads: int = field(default=4, metadata=dict(min=1, max=64))
    bias: bool = True
    eps: float = field(default=1e-6, metadata=dict(min=1e-12, max=1e-2))


class LinearAttention(Layer):
    def __init__(self, cfg, dims):
        super().__init__(dims)
        d = dims.d_model
        self.heads, self.eps = cfg.heads, cfg.eps
        self.add("linear.qkv.w", (d, 3 * d))
        self.add("linear.proj.w", (d, d), "residual_out")
        if cfg.bias:
            self.add("linear.qkv.b", (3 * d,), "bias")
            self.add("linear.proj.b", (d,), "bias")

    def _feature(self, x):
        return self.xp.where(x > 0, x + 1, self.xp.exp(x))

    def forward(self, x, ctx):
        xp = self.xp
        raw_q, raw_k, v = [split_heads(part, self.heads) for part in xp.split(linear(self, x, "linear.qkv"), 3, axis=-1)]
        q = ctx.emit("linear.q", self._feature(raw_q))
        k = ctx.emit("linear.k", self._feature(raw_k))
        v = ctx.emit("linear.v", v)
        batch, heads, length, depth = q.shape
        matrix = xp.zeros((batch, heads, depth, depth), dtype=x.dtype)
        vector = xp.zeros((batch, heads, depth), dtype=x.dtype)
        outputs, saved = [], []
        for t in range(length):
            qt, kt, vt = q[:, :, t], k[:, :, t], v[:, :, t]
            matrix = matrix + kt[..., :, None] * vt[..., None, :]
            vector = vector + kt
            numerator = xp.einsum("bhi,bhij->bhj", qt, matrix)
            denominator = xp.sum(qt * vector, axis=-1, keepdims=True) + self.eps
            outputs.append(numerator / denominator)
            saved.append((matrix, vector, numerator, denominator))
        stream = ctx.emit("linear.out", xp.stack(outputs, axis=2))
        merged = merge_heads(stream)
        self.cache = (x, raw_q, raw_k, q, k, v, saved, merged)
        return linear(self, merged, "linear.proj")

    def backward(self, dy):
        xp = self.xp
        x, raw_q, raw_k, q, k, v, saved, merged = self.cache
        dstream = split_heads(linear_back(self, dy, merged, "linear.proj"), self.heads)
        dq, dk, dv = xp.zeros_like(q), xp.zeros_like(k), xp.zeros_like(v)
        batch, heads, length, depth = q.shape
        dmatrix = xp.zeros((batch, heads, depth, depth), dtype=x.dtype)
        dvector = xp.zeros((batch, heads, depth), dtype=x.dtype)
        for t in reversed(range(length)):
            matrix, vector, numerator, denominator = saved[t]
            qt, kt, vt = q[:, :, t], k[:, :, t], v[:, :, t]
            dn = dstream[:, :, t] / denominator
            dd = -xp.sum(dstream[:, :, t] * numerator, axis=-1, keepdims=True) / (denominator * denominator)
            dq[:, :, t] = xp.einsum("bhj,bhij->bhi", dn, matrix) + dd * vector
            dmatrix = dmatrix + qt[..., :, None] * dn[..., None, :]
            dvector = dvector + dd * qt
            dk[:, :, t] = xp.einsum("bhij,bhj->bhi", dmatrix, vt) + dvector
            dv[:, :, t] = xp.einsum("bhi,bhij->bhj", kt, dmatrix)
        dq *= xp.where(raw_q > 0, 1, xp.exp(raw_q))
        dk *= xp.where(raw_k > 0, 1, xp.exp(raw_k))
        dqkv = xp.concatenate([merge_heads(value) for value in (dq, dk, dv)], axis=-1)
        return linear_back(self, dqkv, x, "linear.qkv")


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
