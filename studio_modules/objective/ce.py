"""Next-token cross-entropy: the mean of −ln p(correct next token) over every target position."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from studio_registry import ModuleSpec, register
from studio_slots import as_tensor


def as_objective(fn):
    """Wrap fn(flat_logits, labels) -> scalar tensor into the objective contract.

    objective(logits, targets, gradient=False, xp=None) returns the loss as a tensor (attached to the
    computation graph when `logits` is), or (float loss, d loss / d logits) with gradient=True.
    Targets of -100 are ignored; a batch without targets raises ValueError.
    """
    def objective(logits, targets, gradient=False, xp=None):
        logits = as_tensor(logits)
        labels = as_tensor(targets, dtype=torch.long, device=logits.device).reshape(-1)
        if not bool((labels != -100).any()):
            raise ValueError("This batch has no target tokens.")
        if not gradient:
            return fn(logits.reshape(-1, logits.shape[-1]), labels)
        leaf = logits.detach().requires_grad_(True)
        loss = fn(leaf.reshape(-1, logits.shape[-1]), labels)
        (derivative,) = torch.autograd.grad(loss, leaf)
        return float(loss.detach()), derivative
    return objective


def _cross_entropy(flat, labels):
    return F.cross_entropy(flat, labels, ignore_index=-100)


cross_entropy = as_objective(_cross_entropy)


@register
class CrossEntropy(ModuleSpec):
    id = "objective/ce"
    slot = "objective"
    name = "Next-token cross-entropy"
    blurb = "Mean of −ln p(correct next token) over every position."
    ref = "standard"
    backward = "PyTorch autograd"

    def build(self, cfg, dims, rng):
        """Returns f(logits, targets, gradient=False) -> loss tensor, or (float loss, dlogits)."""
        return cross_entropy
