"""Next-token cross-entropy: the mean of −ln p(correct next token) over every target position."""
from __future__ import annotations

import numpy as np

from studio_registry import ModuleSpec, register


def cross_entropy(logits, targets, gradient=False, xp=np):
    """Loss, or (loss, d loss / d logits) with gradient=True. Targets of -100 are ignored."""
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
    loss = float((xp.log(denominator[rows, 0]) - shifted[rows, labels[rows]]).mean())
    if not gradient:
        return loss
    derivative = exponential / denominator  # softmax
    derivative[rows, labels[rows]] -= 1     # minus the one-hot target
    derivative[~valid] = 0
    derivative /= count
    return loss, derivative.reshape(logits.shape)


@register
class CrossEntropy(ModuleSpec):
    id = "objective/ce"
    slot = "objective"
    name = "Next-token cross-entropy"
    blurb = "Mean of −ln p(correct next token) over every position."
    ref = "standard"

    def build(self, cfg, dims, rng):
        """Returns f(logits, targets, gradient=False, xp=np) -> loss, or (loss, dlogits)."""
        return cross_entropy
