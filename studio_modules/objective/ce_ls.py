"""Cross-entropy with a uniformly smoothed target distribution."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn.functional as F

from studio_modules.objective.ce import as_objective
from studio_registry import ModuleSpec, register


@dataclass
class LabelSmoothingConfig:
    epsilon: float = field(default=.1, metadata=dict(min=0, max=1, help="Fraction of target probability spread uniformly across the vocabulary."))


def cross_entropy_label_smoothing(epsilon=.1):
    """Mean soft-target cross-entropy: −(1−ε)·ln p(target) − ε·mean over the vocabulary of ln p."""
    return as_objective(lambda flat, labels: F.cross_entropy(flat, labels, ignore_index=-100, label_smoothing=epsilon))


@register
class LabelSmoothing(ModuleSpec):
    id = "objective/ce_ls"
    slot = "objective"
    name = "Label-smoothed cross-entropy"
    blurb = "Mix the correct-token target with a uniform distribution over the vocabulary."
    ref = "Szegedy et al. 2016"
    config = LabelSmoothingConfig
    backward = "PyTorch autograd · loss gradient checked by finite differences"

    def build(self, cfg, dims, rng):
        return cross_entropy_label_smoothing(cfg.epsilon)
