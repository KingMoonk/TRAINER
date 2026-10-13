"""MLP: W₂ · GELU(W₁ · x), run on each position separately."""
from __future__ import annotations

from dataclasses import dataclass, field

from studio_modules.ops import activate, linear
from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


@dataclass
class MLPConfig:
    d_ff: int = field(default=0, metadata=dict(min=0, max=8192, help="Hidden width. 0 means 4 × d_model."))
    bias: bool = True
    layers: int = field(default=1, metadata=dict(min=1, max=4, help="Hidden layers. A standard transformer MLP has 1; more is kept for older designs."))
    activation: str = field(default="gelu", metadata=dict(choices=["gelu", "relu"]))


def width(cfg, dims):
    return cfg.d_ff or 4 * dims.d_model


class MLP(Layer):
    def __init__(self, cfg, dims):
        super().__init__(dims)
        d, hidden = dims.d_model, width(cfg, dims)
        self.layers, self.kind = cfg.layers, cfg.activation
        self.names = ["f1"] + [f"fh{j}" for j in range(1, cfg.layers)]
        self.add("f1.w", (d, hidden))
        for name in self.names[1:]:
            self.add(name + ".w", (hidden, hidden))
        self.add("f2.w", (hidden, d), "residual_out")
        if cfg.bias:
            for name in self.names:
                self.add(name + ".b", (hidden,), "bias")
            self.add("f2.b", (d,), "bias")

    def forward(self, x, ctx):
        act = x
        for i, name in enumerate(self.names):
            pre = linear(self, act, name)
            if i == len(self.names) - 1:
                pre = ctx.emit("mlp.pre", pre)  # only the final hidden layer exposes hooks, also for legacy multi-layer MLPs
            act = activate(pre, self.kind)
        act = ctx.emit("mlp.act", act)
        return linear(self, act, "f2")


@register
class MLPSpec(ModuleSpec):
    id = "channel/mlp_gelu"
    slot = "channel"
    name = "MLP · GELU"
    blurb = "Two linear layers with a GELU between them, hidden width 4 × d."
    ref = "GPT-2"
    config = MLPConfig
    glyph3d = {"kind": "tower", "hooks": ["mlp.act"]}
    hooks = (("mlp.pre", "B T F"), ("mlp.act", "B T F"))

    def symbols(self, cfg, dims):
        return {"F": width(cfg, dims)}

    def estimate(self, cfg, dims):
        d, f = dims.d_model, width(cfg, dims)
        params = 2 * d * f + (cfg.layers - 1) * f * f + ((cfg.layers * f + d) if cfg.bias else 0)
        return Estimate(params=params, flops_per_token=2 * params, activation_bytes=(2 * cfg.layers * f + d) * dims.bytes)

    def build(self, cfg, dims, rng):
        return MLP(cfg, dims)
