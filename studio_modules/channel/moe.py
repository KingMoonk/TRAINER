"""Top-k mixture of biased GELU MLP experts with differentiable router weights."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from studio_modules.ops import activate
from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


@dataclass
class MoEConfig:
    experts: int = field(default=8, metadata=dict(min=2, max=32))
    top_k: int = field(default=2, metadata=dict(min=1, max=8))
    d_ff: int = field(default=0, metadata=dict(min=0, max=8192))
    aux_weight: float = field(default=.01, metadata=dict(min=0, max=1))


def width(cfg, dims):
    return cfg.d_ff or 4 * dims.d_model


class MoELayer(Layer):
    def __init__(self, cfg, dims):
        super().__init__(dims)
        self.experts, self.top_k, self.aux_weight = cfg.experts, cfg.top_k, cfg.aux_weight
        d, f = dims.d_model, width(cfg, dims)
        self.add("moe.router.w", (d, cfg.experts))
        for expert in range(cfg.experts):
            prefix = f"moe.expert{expert}"
            self.add(prefix + ".f1.w", (d, f))
            self.add(prefix + ".f1.b", (f,), "bias")
            self.add(prefix + ".f2.w", (f, d), "residual_out")
            self.add(prefix + ".f2.b", (d,), "bias")

    def forward(self, x, ctx):
        p = self.params
        batch, length, d = x.shape
        flat = x.reshape(-1, d)
        logits = ctx.emit("moe.router.logits", x @ p["moe.router.w"])
        probabilities = ctx.emit("moe.router.prob", torch.softmax(logits, dim=-1))
        top = ctx.emit("moe.router.top", torch.topk(probabilities, self.top_k, dim=-1).indices)
        selected = torch.gather(probabilities, -1, top)
        weights = selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        chosen = top.reshape(-1, self.top_k)
        weights = weights.reshape(-1, self.top_k)
        output = torch.zeros_like(flat)
        for expert in range(self.experts):
            hit = chosen == expert
            locations = hit.any(dim=-1).nonzero().squeeze(1)
            if locations.numel() == 0:
                continue
            prefix = f"moe.expert{expert}"
            inputs = flat[locations]
            act = activate(inputs @ p[prefix + ".f1.w"] + p[prefix + ".f1.b"], "gelu")
            values = act @ p[prefix + ".f2.w"] + p[prefix + ".f2.b"]
            weight = (weights[locations] * hit[locations]).sum(dim=-1)
            output = output.index_add(0, locations, weight[:, None] * values)
        load = torch.bincount(chosen.reshape(-1), minlength=self.experts).to(x.dtype) / (chosen.shape[0] * self.top_k)
        if self.aux_weight:
            ctx.aux = ctx.aux + self.aux_weight * self.experts * (probabilities.mean(dim=(0, 1)) * load).sum()
        return output.reshape(batch, length, d)


@register
class MoE(ModuleSpec):
    id = "channel/moe"
    slot = "channel"
    name = "Top-k mixture of experts"
    blurb = "A router selects k GELU MLP experts per token and adds a load-balancing loss."
    ref = "Shazeer et al. 2017; Fedus et al. 2022"
    config = MoEConfig
    hooks = (("moe.router.logits", "B T X"), ("moe.router.prob", "B T X"),
             ("moe.router.top", "B T K"))
    glyph3d = {"kind": "fan", "hooks": ["moe.router.prob"]}

    def symbols(self, cfg, dims):
        return {"X": cfg.experts, "K": cfg.top_k}

    def estimate(self, cfg, dims):
        d, f = dims.d_model, width(cfg, dims)
        expert = 2 * d * f + f + d
        router = d * cfg.experts
        return Estimate(params=router + cfg.experts * expert,
                        active_params=router + cfg.top_k * expert,
                        flops_per_token=2 * (router + cfg.top_k * expert),
                        activation_bytes=(cfg.top_k * 2 * f + cfg.experts + d) * dims.bytes)

    def validate(self, cfg, dims):
        from studio_slots import Issue
        if cfg.top_k > cfg.experts:
            return [Issue("ERROR", "MoE top_k cannot exceed the number of experts.")]
        return []

    def build(self, cfg, dims, rng):
        return MoELayer(cfg, dims)
