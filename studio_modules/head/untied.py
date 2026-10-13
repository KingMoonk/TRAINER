"""Independent vocabulary projection from hidden states to logits."""
from __future__ import annotations

from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


class UntiedHead(Layer):
    def __init__(self, dims):
        super().__init__(dims)
        self.add("output.w", (dims.d_model, dims.vocab), "weight")
        self.add("output.b", (dims.vocab,), "bias")

    def forward(self, x, ctx):
        return x @ self.params["output.w"] + self.params["output.b"]


@register
class Untied(ModuleSpec):
    id = "head/untied"
    slot = "head"
    name = "Untied projection"
    blurb = "A separate learned matrix and bias project the stream to vocabulary logits."
    ref = "standard"

    def estimate(self, cfg, dims):
        return Estimate(params=dims.d_model * dims.vocab + dims.vocab,
                        flops_per_token=2 * dims.d_model * dims.vocab,
                        activation_bytes=dims.d_model * dims.bytes)

    def build(self, cfg, dims, rng):
        return UntiedHead(dims)
