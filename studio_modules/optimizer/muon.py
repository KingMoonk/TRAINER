"""Muon matrix updates with AdamW for embeddings, heads, filters and vectors."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from studio_registry import ModuleSpec, register
from studio_slots import Optimizer


@dataclass
class MuonConfig:
    lr: float = field(default=.02, metadata=dict(min=1e-7, max=.1, help="Peak rate for the orthogonalized matrix updates."))
    adam_lr: float = field(default=1e-3, metadata=dict(min=1e-7, max=.1, help="Peak rate for the AdamW parameters (embeddings, head, norms, filters). The schedule scales both."))
    momentum: float = field(default=.95, metadata=dict(min=0, max=.9999))
    ns_steps: int = field(default=5, metadata=dict(min=1, max=20))
    weight_decay: float = field(default=0., metadata=dict(min=0, max=1))
    adam_beta1: float = field(default=.9, metadata=dict(min=0, max=.9999))
    adam_beta2: float = field(default=.95, metadata=dict(min=0, max=.99999))
    adam_eps: float = field(default=1e-10, metadata=dict(min=1e-12, max=1e-3))


def zeropower_newton_schulz(matrix, steps, xp=None):
    """The published quintic Newton-Schulz transform in the parameter's dtype."""
    transposed = matrix.shape[0] > matrix.shape[1]
    x = matrix.t().clone() if transposed else matrix.clone()
    x = x / (torch.sqrt((x * x).sum()) + 1e-7)
    for _ in range(steps):
        a = x @ x.t()
        b = -4.7750 * a + 2.0315 * (a @ a)
        x = 3.4445 * x + b @ x
    return x.t() if transposed else x


class Muon(Optimizer):
    states = ("m", "v")

    def apply(self, key, parameter, gradient, lr):
        cfg = self.cfg
        is_hidden = (parameter.ndim == 2 and self.owner.get(key, (None,))[0] in ("mixer", "channel")
                     and self.roles.get(key) in ("weight", "residual_out") and key not in self.elementwise)
        if is_hidden:
            self.m[key].mul_(cfg.momentum).add_(gradient, alpha=1 - cfg.momentum)
            candidate = (1 - cfg.momentum) * gradient + cfg.momentum * self.m[key]
            update = zeropower_newton_schulz(candidate, cfg.ns_steps)
            update = update * max(1, parameter.shape[0] / parameter.shape[1]) ** .5
        else:
            lr = lr * cfg.adam_lr / cfg.lr  # the schedule scales the Muon rate; Adam parameters get their own peak
            self.m[key].mul_(cfg.adam_beta1).add_(gradient, alpha=1 - cfg.adam_beta1)
            self.v[key].mul_(cfg.adam_beta2).addcmul_(gradient, gradient, value=1 - cfg.adam_beta2)
            first = self.m[key] / (1 - cfg.adam_beta1 ** self.steps)
            second = self.v[key] / (1 - cfg.adam_beta2 ** self.steps)
            update = first / (torch.sqrt(second) + cfg.adam_eps)
        if self.decays(key):
            parameter.mul_(1 - lr * cfg.weight_decay)
        parameter.sub_(lr * update)


@register
class MuonSpec(ModuleSpec):
    id = "optimizer/muon"
    slot = "optimizer"
    name = "Muon + AdamW"
    blurb = "Orthogonalized momentum updates for hidden matrices; AdamW for other parameters."
    ref = "Jordan et al. 2024"
    config = MuonConfig
    backward = "Newton-Schulz matrix update and AdamW fallback; hand-reference tests"

    def build(self, cfg, dims, rng):
        return Muon(cfg)
