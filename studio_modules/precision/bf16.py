"""bfloat16 mixed precision: matrix multiplies in bfloat16, float32 master weights, optimizer state, losses and softmax."""
from __future__ import annotations

import torch

from studio_registry import ModuleSpec, register


@register
class BF16(ModuleSpec):
    id = "precision/bf16"
    slot = "precision"
    name = "bfloat16 mixed"
    blurb = "Matrix multiplies run in bfloat16 for speed and memory; weights, optimizer and loss stay float32. Needs no loss scaling."
    ref = "Kalamkar et al. 2019"
    backward = "No backward pass"
    compute = torch.bfloat16

    def build(self, cfg, dims, rng):
        return torch.float32
