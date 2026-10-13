"""Cross-entropy with a squared log-partition regularizer."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from studio_registry import ModuleSpec, register


@dataclass
class ZLossConfig:
    coefficient: float = field(default=1e-4, metadata=dict(min=0, max=1, help="Strength of the squared log-sum-exp penalty."))


def cross_entropy_z_loss(logits, targets, gradient=False, xp=np, coefficient=1e-4):
    """Mean CE + coefficient * logsumexp(logits)^2 over valid targets; -100 rows are ignored."""
    flat = logits.reshape(-1, logits.shape[-1])
    labels = xp.asarray(targets).reshape(-1)
    valid = labels != -100
    count = int(valid.sum())
    if count == 0:
        raise ValueError("This batch has no target tokens.")
    shifted = flat - flat.max(axis=-1, keepdims=True)
    exponential = xp.exp(shifted)
    denominator = exponential.sum(axis=-1, keepdims=True)
    log_partition = flat.max(axis=-1) + xp.log(denominator[:, 0])
    rows = xp.flatnonzero(valid)
    probs = exponential / denominator
    losses = log_partition[rows] - flat[rows, labels[rows]]
    loss = (losses + coefficient * log_partition[rows] ** 2).mean()
    if not gradient:
        return float(loss)
    derivative = probs.copy()
    derivative[rows, labels[rows]] -= 1
    derivative[rows] += (2 * coefficient * log_partition[rows, None]) * probs[rows]
    derivative[~valid] = 0
    derivative /= count
    return float(loss), derivative.reshape(logits.shape)


@register
class ZLoss(ModuleSpec):
    id = "objective/ce_z"
    slot = "objective"
    name = "Cross-entropy + z-loss"
    blurb = "Cross-entropy plus a penalty on the squared log-sum-exp of valid logits."
    ref = "Paulus et al. 2020"
    config = ZLossConfig
    backward = "No model backward pass · loss gradient checked by finite differences"

    def build(self, cfg, dims, rng):
        return lambda logits, targets, gradient=False, xp=np: cross_entropy_z_loss(
            logits, targets, gradient, xp, cfg.coefficient)
