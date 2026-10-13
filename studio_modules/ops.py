"""Small forward building blocks shared by modules. `layer` is the Layer that owns the weights."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def linear(layer, x, name):
    """x @ W (+ b when the layer registered a bias for `name`)."""
    output = x @ layer.params[name + ".w"]
    if name + ".b" in layer.params:
        output = output + layer.params[name + ".b"]
    return output


def softmax(x, xp=None):
    """Softmax over the last axis of a tensor or a NumPy array."""
    if isinstance(x, torch.Tensor):
        return torch.softmax(x, dim=-1)
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def activate(pre, kind="gelu"):
    if kind == "relu":
        return torch.relu(pre)
    if kind == "silu":
        return F.silu(pre)
    return F.gelu(pre, approximate="tanh")


def split_heads(x, heads):
    """[B, T, heads * h] -> [B, heads, T, h]"""
    batch, length, width = x.shape
    return x.reshape(batch, length, heads, width // heads).transpose(1, 2)


def merge_heads(x):
    """[B, heads, T, h] -> [B, T, heads * h]"""
    batch, heads, length, depth = x.shape
    return x.transpose(1, 2).reshape(batch, length, heads * depth)


def causal_conv(x, weight, bias):
    """Depthwise causal convolution: out[t] = bias + Σ_lag x[t - lag] · weight[lag]. x [B, T, C], weight [K, C]."""
    length = x.shape[1]
    out = bias.expand_as(x) if bias is not None else torch.zeros_like(x)
    for lag in range(min(weight.shape[0], length)):
        shifted = x if lag == 0 else F.pad(x, (0, 0, lag, 0))[:, :length]
        out = out + shifted * weight[lag]
    return out
