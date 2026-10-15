"""Byte tokenizer: one token per UTF-8 byte. Also home of the Tokenizer class the character tokenizer shares."""
from __future__ import annotations

import numpy as np

from studio_ui_guard import slow
from studio_core import messages_of
from studio_registry import ModuleSpec, register
from studio_slots import OFFSET, UNK


class Tokenizer:
    def __init__(self, kind="byte", characters=None):
        self.kind = kind
        self.characters = list(characters or [])
        self.lookup = {char: i + OFFSET for i, char in enumerate(self.characters)}
        self.vocab_size = (256 if kind == "byte" else len(self.characters)) + OFFSET

    @classmethod
    @slow("train a tokenizer")
    def fit(cls, kind, examples):
        if kind == "byte":
            return cls("byte")
        if kind != "character":
            from studio_registry import get
            module = get("tokenizer", kind)
            return module.fit(module.parse(), examples)
        characters = set()
        for example in examples:
            texts = [example["text"]] if example.get("text") else [m["content"] for m in messages_of(example)]
            for text in texts:
                characters.update(text)
            if len(characters) > 16384:
                raise ValueError("Use the byte tokenizer for a vocabulary larger than 16,384 characters.")
        return cls("character", sorted(characters))

    def encode(self, text):
        if self.kind == "byte":
            return self.encode_array(text).tolist()
        return [self.lookup.get(char, UNK) for char in text]

    def encode_array(self, text):
        """encode() as an int64 array; bytes never become one Python int per token on the way."""
        if self.kind == "byte":
            return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64) + OFFSET
        return np.array(self.encode(text), dtype=np.int64)

    def decode(self, tokens):
        if self.kind == "byte":
            return bytes(int(t) - OFFSET for t in tokens if OFFSET <= t < self.vocab_size).decode("utf-8", errors="replace")
        return "".join(self.characters[int(t) - OFFSET] for t in tokens if OFFSET <= t < self.vocab_size)

    def serialize(self):
        return {"kind": self.kind, "characters": self.characters}

    @classmethod
    def restore(cls, data):
        kind = data.get("kind")
        if kind in {"byte", "character"}:
            return cls(kind, data.get("characters", []))
        from studio_registry import get
        try:
            return get("tokenizer", kind).restore(data)
        except ValueError:
            raise ValueError("Invalid tokenizer in checkpoint.") from None


@register
class Bytes(ModuleSpec):
    id = "tokenizer/byte"
    slot = "tokenizer"
    name = "Bytes"
    blurb = "One token per UTF-8 byte. Tiny vocabulary, longer sequences, no unknown characters."
    ref = "ByT5, Xue et al. 2022"
    backward = "No backward pass"

    def vocab(self, cfg):
        """Vocabulary size including the 7 shared special tokens, or None when it depends on the data."""
        return 256 + OFFSET

    def fit(self, cfg, examples):
        return Tokenizer("byte")

    def restore(self, data):
        return Tokenizer.restore(data)
