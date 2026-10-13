"""Maximal Update Parametrization (μP): width-aware initialization and learning rate scaling.

Follows Yang et al. 2022 (Tensor Programs V) for Adam-like and SGD-like optimizers, relative
to an explicit base width. Attention and the tied head apply the score/logit multipliers set here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from studio_modules.init.gpt2 import fill_normal
from studio_registry import ModuleSpec, register


@dataclass
class MuPConfig:
    base_width: int = field(default=256, metadata=dict(min=1, max=4096, help="The reference model width for scaling rules."))
    std: float = field(default=.02, metadata=dict(min=1e-4, max=1, help="Standard deviation for input-like parameters."))
    scale_residual: bool = field(default=True, metadata=dict(help="Scale residual outputs by 1/√(2·layers)."))
    optimizer_family: str = field(default="adam", metadata=dict(choices=["adam", "sgd"], help="Adam-like (adaptive) or SGD-like (first-order) optimizer family."))


class MuPInit:
    def __init__(self, cfg):
        self.cfg = cfg

    @torch.no_grad()
    def apply(self, layer, rng, dims):
        m = dims.d_model / self.cfg.base_width
        sgd = self.cfg.optimizer_family == "sgd"
        kind = type(layer).__name__.lower()
        unit = getattr(layer, "depth", dims.d_model)  # grouped-query k/v projections are kv_heads * depth wide and still scale with width
        for name, role in layer.roles.items():
            parameter = layer.params[name]
            readout = parameter.ndim == 2 and role == "weight" and "untied" in kind
            hidden = (parameter.ndim == 2 and role in ("weight", "residual_out")
                      and all(size >= unit and size % unit == 0 for size in parameter.shape)
                      and "head" not in kind and "embedding" not in kind)
            if role in ("weight", "embedding", "residual_out"):
                std = self.cfg.std / (m if readout else math.sqrt(m) if hidden else 1)
                if role == "residual_out" and self.cfg.scale_residual:
                    std /= math.sqrt(2 * dims.n_layer)
                fill_normal(parameter, rng, std)
            elif role == "gain":
                parameter.fill_(1)
            elif role == "bias":
                parameter.fill_(0)
            # LR multipliers (Tensor Programs V, table 8): readout 1/m; hidden 1/m for Adam, 1 for SGD; input-like 1 for Adam, m for SGD
            if readout or (hidden and not sgd):
                layer.lr_scale[name] = 1 / m
            elif sgd and not hidden and role in ("weight", "embedding", "gain", "bias"):
                layer.lr_scale[name] = m
        if hasattr(layer, "score_scale"):  # heads stay fixed across widths, so the head width multiplier is m
            layer.score_scale = 1 / math.sqrt(m)
        if hasattr(layer, "logit_scale"):
            layer.logit_scale = 1 / m


@register
class MuP(ModuleSpec):
    id = "init/mup"
    slot = "init"
    name = "Width-scaled μP initialization"
    blurb = "Maximal Update Parametrization: width-scaled initialization, learning rates, attention scores and tied-readout logits from an explicit base width."
    ref = "Yang et al. 2022 (Tensor Programs V)"
    config = MuPConfig
    backward = "Initialization and LR multipliers; score/logit scaling in the layers' forward passes"

    def build(self, cfg, dims, rng):
        return MuPInit(cfg)
