"""Cross-entropy with a uniformly smoothed target distribution."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from studio_registry import ModuleSpec, register


@dataclass
class LabelSmoothingConfig:
    epsilon: float = field(default=.1, metadata=dict(min=0, max=1, help="Fraction of target probability spread uniformly across the vocabulary."))


def cross_entropy_label_smoothing(logits, targets, gradient=False, xp=np, epsilon=.1):
    """Mean soft-target cross-entropy; targets equal -100 are excluded from loss and gradient."""
    flat = logits.reshape(-1, logits.shape[-1])
    labels = xp.asarray(targets).reshape(-1)
    valid = labels != -100
    count = int(valid.sum())
    if count == 0:
        raise ValueError("This batch has no target tokens.")
    shifted = flat - flat.max(axis=-1, keepdims=True)
    exponential = xp.exp(shifted)
    denominator = exponential.sum(axis=-1, keepdims=True)
    rows = xp.flatnonzero(valid)
    log_probs = shifted[rows] - xp.log(denominator[rows])
    loss = -(1 - epsilon) * log_probs[xp.arange(count), labels[rows]].mean() - epsilon * log_probs.mean()
    if not gradient:
        return float(loss)
    derivative = exponential / denominator
    derivative[rows] -= epsilon / flat.shape[-1]
    derivative[rows, labels[rows]] -= 1 - epsilon
    derivative[~valid] = 0
    derivative /= count
    return float(loss), derivative.reshape(logits.shape)


@register
class LabelSmoothing(ModuleSpec):
    id = "objective/ce_ls"
    slot = "objective"
    name = "Label-smoothed cross-entropy"
    blurb = "Mix the correct-token target with a uniform distribution over the vocabulary."
    ref = "Szegedy et al. 2016"
    config = LabelSmoothingConfig
    backward = "No model backward pass · loss gradient checked by finite differences"

    def build(self, cfg, dims, rng):
        return lambda logits, targets, gradient=False, xp=np: cross_entropy_label_smoothing(
            logits, targets, gradient, xp, cfg.epsilon)
