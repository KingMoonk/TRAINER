"""float64 weights and activations."""
from __future__ import annotations

import numpy as np

from studio_registry import ModuleSpec, register


@register
class FP64(ModuleSpec):
    id = "precision/fp64"
    slot = "precision"
    name = "float64"
    blurb = "Slower, but exact enough for finite-difference gradient checks."
    ref = "—"
    backward = "No backward pass"

    def build(self, cfg, dims, rng):
        return np.float64
