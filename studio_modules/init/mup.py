"""Width-aware μP initialization and Adam-like hidden-matrix LR multipliers.

The reference width is explicit. Tied readout scaling remains the model's existing
definition, so this initializer makes no claim of complete μP transfer behavior.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from studio_registry import ModuleSpec, register


@dataclass
class MuPConfig:
    base_width: int = field(default=256, metadata=dict(min=1, max=4096))
    std: float = field(default=.02, metadata=dict(min=1e-4, max=1))
    scale_residual: bool = True


class MuPInit:
    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, layer, rng, dims):
        xp = layer.xp
        width_mult = dims.d_model / self.cfg.base_width
        for name, role in layer.roles.items():
            parameter = layer.params[name]
            hidden_matrix = (parameter.ndim == 2 and role in ("weight", "residual_out")
                             and all(size >= dims.d_model and size % dims.d_model == 0 for size in parameter.shape)
                             and "head" not in type(layer).__name__.lower()
                             and "embedding" not in type(layer).__name__.lower())
            if role in ("weight", "embedding", "residual_out"):
                std = self.cfg.std / math.sqrt(width_mult) if hidden_matrix else self.cfg.std
                if role == "residual_out" and self.cfg.scale_residual:
                    std /= math.sqrt(2 * dims.n_layer)
                parameter[...] = xp.asarray(rng.normal(0, std, parameter.shape).astype(np.dtype(layer.dtype)))
                if hidden_matrix:
                    layer.lr_scale[name] = 1 / width_mult
            elif role == "gain":
                parameter.fill(1)
            elif role == "bias":
                parameter.fill(0)


@register
class MuP(ModuleSpec):
    id = "init/mup"
    slot = "init"
    name = "Width-scaled μP init"
    blurb = "Scales hidden matrix initialization and Adam-like learning rates from an explicit base width."
    ref = "Yang et al. 2022"
    config = MuPConfig
    backward = "Initialization and LR multipliers; no backward pass"

    def build(self, cfg, dims, rng):
        return MuPInit(cfg)
