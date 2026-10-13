"""Cross-entropy with a squared log-partition regularizer."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from studio_modules.objective.ce import as_objective
from studio_registry import ModuleSpec, register


@dataclass
class ZLossConfig:
    coefficient: float = field(default=1e-4, metadata=dict(min=0, max=1, help="Strength of the squared log-sum-exp penalty."))


def cross_entropy_z_loss(coefficient=1e-4):
    """Mean over valid targets of (logsumexp − logit[target] + coefficient · logsumexp²)."""
    def loss(flat, labels):
        valid = labels != -100
        rows, picked = flat[valid], labels[valid]
        partition = torch.logsumexp(rows, dim=-1)
        return (partition - rows.gather(1, picked[:, None])[:, 0] + coefficient * partition * partition).mean()
    return as_objective(loss)


@register
class ZLoss(ModuleSpec):
    id = "objective/ce_z"
    slot = "objective"
    name = "Cross-entropy + z-loss"
    blurb = "Cross-entropy plus a penalty on the squared log-sum-exp of valid logits."
    ref = "Paulus et al. 2020"
    config = ZLossConfig
    backward = "PyTorch autograd · loss gradient checked by finite differences"

    def build(self, cfg, dims, rng):
        return cross_entropy_z_loss(cfg.coefficient)
