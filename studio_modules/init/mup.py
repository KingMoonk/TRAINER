"""Maximal Update Parametrization (μP): width-aware initialization and learning rate scaling.

Implements complete μP (Yang et al. 2022, Tensor Programs V) with adaptive (Adam-like) and
first-order (SGD-like) optimizer families. Transfer learning works across any model widths
at the same hyperparameters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

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

    def apply(self, layer, rng, dims):
        xp = layer.xp
        width_mult = dims.d_model / self.cfg.base_width
        base_d_head = self.cfg.base_width // (self._infer_heads(layer, dims) or 1)  # for attention scaling

        for name, role in layer.roles.items():
            parameter = layer.params[name]

            # Classify the parameter
            is_hidden_matrix = (parameter.ndim == 2 and role in ("weight", "residual_out")
                                and all(size >= dims.d_model and size % dims.d_model == 0 for size in parameter.shape)
                                and "head" not in type(layer).__name__.lower()
                                and "embedding" not in type(layer).__name__.lower())
            is_readout = (parameter.ndim == 2 and role == "weight"
                          and "untied" in type(layer).__name__.lower())

            if role in ("weight", "embedding", "residual_out"):
                # Initialization: std varies by parameter type
                if is_readout:
                    # Untied readout: std / m
                    std = self.cfg.std / width_mult
                elif is_hidden_matrix:
                    # Hidden matrices: std / sqrt(m)
                    std = self.cfg.std / math.sqrt(width_mult)
                else:
                    # Input-like (embeddings, position): std
                    std = self.cfg.std

                if role == "residual_out" and self.cfg.scale_residual:
                    std /= math.sqrt(2 * dims.n_layer)

                parameter[...] = xp.asarray(rng.normal(0, std, parameter.shape).astype(np.dtype(layer.dtype)))

                # Learning rate scaling: depends on parameter type and optimizer family
                if is_readout:
                    # Untied readout: always 1/m for both families
                    layer.lr_scale[name] = 1 / width_mult
                elif is_hidden_matrix:
                    if self.cfg.optimizer_family == "adam":
                        # Hidden matrices with Adam: 1/m
                        layer.lr_scale[name] = 1 / width_mult
                    else:
                        # Hidden matrices with SGD: no scaling (default 1)
                        pass
                elif role == "embedding":
                    if self.cfg.optimizer_family == "sgd":
                        # Input-like with SGD: m
                        layer.lr_scale[name] = width_mult
                    # Adam: no scaling needed (default 1)

            elif role == "gain":
                parameter.fill(1)
            elif role == "bias":
                parameter.fill(0)

        # Set multiplicative scale factors on the layer for forward/backward passes
        # These are applied by the specific layer types (Attention, TiedHead)
        if width_mult != 1:
            # For attention: scale pre-softmax scores by 1/sqrt(m_head)
            if hasattr(layer, 'heads'):
                # Attention-like layer with heads
                d_head = dims.d_model // layer.heads
                m_head = d_head / base_d_head
                if m_head > 0:
                    layer.score_scale = 1.0 / math.sqrt(m_head)

            # For tied readout: scale logits by 1/m
            if "tied" in type(layer).__name__.lower():
                layer.logit_scale = 1.0 / width_mult

    @staticmethod
    def _infer_heads(layer, dims):
        """Infer the number of attention heads from the layer."""
        if hasattr(layer, 'heads'):
            return layer.heads
        # For GQA, use the query head count
        if hasattr(layer, 'heads') and hasattr(layer, 'kv_heads'):
            return layer.heads
        return None


@register
class MuP(ModuleSpec):
    id = "init/mup"
    slot = "init"
    name = "Width-scaled μP initialization"
    blurb = "Complete Maximal Update Parametrization: transfers hyperparameters across model widths for both adaptive and first-order optimizers."
    ref = "Yang et al. 2022 (Tensor Programs V)"
    config = MuPConfig
    backward = "Initialization and LR multipliers; score/logit scaling in layer-specific forward/backward passes"

    def build(self, cfg, dims, rng):
        return MuPInit(cfg)
