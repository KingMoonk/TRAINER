"""Lion: an interpolated momentum sign update with decoupled weight decay."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from studio_registry import ModuleSpec, register
from studio_slots import Optimizer


@dataclass
class LionConfig:
    lr: float = field(default=1e-4, metadata=dict(min=1e-7, max=.01, help="Peak learning rate; Lion wants about 3-10× less than AdamW."))
    beta1: float = field(default=.9, metadata=dict(min=0, max=.9999, help="Momentum interpolation used to choose the sign update."))
    beta2: float = field(default=.99, metadata=dict(min=0, max=.99999, help="Momentum decay used for the stored state."))
    weight_decay: float = field(default=.01, metadata=dict(min=0, max=1, help="Decoupled shrinkage applied to weights, excluding gains and biases."))


class Lion(Optimizer):
    states = ("m",)

    def apply(self, key, parameter, gradient, lr):
        cfg = self.cfg
        momentum = self.m[key]
        direction = cfg.beta1 * momentum + (1 - cfg.beta1) * gradient
        if self.decays(key) and cfg.weight_decay:
            parameter.mul_(1 - lr * cfg.weight_decay)
        parameter.sub_(lr * torch.sign(direction))
        momentum.mul_(cfg.beta2).add_(gradient, alpha=1 - cfg.beta2)


@register
class LionSpec(ModuleSpec):
    id = "optimizer/lion"
    slot = "optimizer"
    name = "Lion"
    blurb = "A momentum interpolation selects a sign update; one momentum array is retained."
    ref = "Chen et al. 2023 · Symbolic Discovery of Optimization Algorithms"
    config = LionConfig
    backward = "Update rule checked against a hand-computed reference"

    def build(self, cfg, dims, rng):
        return Lion(cfg)
