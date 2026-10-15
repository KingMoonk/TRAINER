"""float16 mixed precision with dynamic loss scaling: float32 master weights, optimizer state, losses and softmax."""
from __future__ import annotations

import torch

from studio_registry import ModuleSpec, register


@register
class FP16(ModuleSpec):
    id = "precision/fp16"
    slot = "precision"
    name = "float16 mixed"
    blurb = "Matrix multiplies run in float16 under a loss scale that grows and shrinks to avoid underflow; weights, optimizer and loss stay float32. A step whose gradients overflow is skipped."
    ref = "Micikevicius et al. 2018"
    backward = "No backward pass"
    compute = torch.float16
    scaled = True

    def build(self, cfg, dims, rng):
        return torch.float32
