"""Character tokenizer: one token per distinct character seen in the training data."""
from __future__ import annotations

from studio_registry import ModuleSpec, register
from studio_modules.tokenizer.byte import Tokenizer


@register
class Characters(ModuleSpec):
    id = "tokenizer/character"
    slot = "tokenizer"
    name = "Characters"
    blurb = "One token per distinct character in your data. Shortest sequences for plain text; unseen characters become unknown."
    ref = "char-rnn, Karpathy 2015"
    backward = "No backward pass"

    def vocab(self, cfg):
        return None

    def fit(self, cfg, examples):
        return Tokenizer.fit("character", examples)

    def restore(self, data):
        return Tokenizer.restore(data)
