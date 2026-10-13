"""Small forward/backward building blocks shared by modules. `layer` is the Layer that owns the weights."""
from __future__ import annotations

import math


def linear(layer, x, name):
    """x @ W (+ b when the layer registered a bias for `name`)."""
    output = x @ layer.params[name + ".w"]
    if name + ".b" in layer.params:
        output = output + layer.params[name + ".b"]
    return output


def linear_back(layer, dy, x, name):
    """Adds the gradients of linear(layer, x, name) into layer.grads and returns d(loss)/dx."""
    layer.grads[name + ".w"] += x.reshape(-1, x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
    if name + ".b" in layer.grads:
        layer.grads[name + ".b"] += dy.sum(axis=tuple(range(dy.ndim - 1)))
    return dy @ layer.params[name + ".w"].T


def softmax(x, xp):
    shifted = x - xp.max(x, axis=-1, keepdims=True)
    exp = xp.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def activate(xp, pre, kind="gelu"):
    """Activation plus the value its derivative reuses (tanh for GELU, sigmoid for SiLU)."""
    if kind == "relu":
        return xp.maximum(pre, 0), None
    if kind == "silu":
        sigmoid = 1 / (1 + xp.exp(-pre))
        return pre * sigmoid, sigmoid
    tanh = xp.tanh(math.sqrt(2 / math.pi) * (pre + .044715 * pre ** 3))
    return .5 * pre * (1 + tanh), tanh


def activate_derivative(pre, saved, kind="gelu"):
    if kind == "relu":
        return (pre > 0).astype(pre.dtype)
    if kind == "silu":
        return saved * (1 + pre * (1 - saved))
    return .5 * (1 + saved) + .5 * pre * (1 - saved * saved) * math.sqrt(2 / math.pi) * (1 + 3 * .044715 * pre ** 2)


def split_heads(x, heads):
    """[B, T, heads * h] -> [B, heads, T, h]"""
    batch, length, width = x.shape
    return x.reshape(batch, length, heads, width // heads).transpose(0, 2, 1, 3)


def merge_heads(x):
    """[B, heads, T, h] -> [B, T, heads * h]"""
    batch, heads, length, depth = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, length, heads * depth)
