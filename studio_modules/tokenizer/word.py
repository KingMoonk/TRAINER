"""Whole-word tokenizer: the most frequent words and punctuation of the selected data, one token each."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from studio_ui_guard import slow
from studio_modules.tokenizer.bpe_custom import texts_of
from studio_registry import ModuleSpec, register
from studio_slots import OFFSET, UNK

# A word or a punctuation run keeps its one leading space (as in GPT-2), any other whitespace character stands alone;
# the matches of any text join back into that exact text, so decoding a known sentence reproduces it.
PIECE = re.compile(r" ?\w+| ?[^\w\s]+|\s")


@dataclass
class WordConfig:
    vocab_size: int = field(default=8192, metadata=dict(min=263, max=16384,
                                                         help="Maximum vocabulary including the seven shared special tokens."))


class WordTokenizer:
    kind = "word"

    def __init__(self, words):
        self.words = list(words)
        self.lookup = {word: i + OFFSET for i, word in enumerate(self.words)}
        self.pieces = [word.encode("utf-8") for word in self.words]   # lets studio_model match rows across tokenizers
        self.vocab_size = len(self.words) + OFFSET

    @classmethod
    @slow("train a tokenizer")
    def fit(cls, examples, vocab_size):
        """Keep the vocab_size - OFFSET most frequent pieces; ties break alphabetically so the fit is deterministic.

        A word outside the list becomes the shared unknown token, the same as an unseen character does in the
        character tokenizer. A byte fallback would keep every word but mix bytes into the list and make the
        sequences, and what the model learns, depend on how rare a word is.
        """
        counts = Counter()
        for text in texts_of(examples):
            counts.update(PIECE.findall(text))
        ranked = sorted(counts, key=lambda piece: (-counts[piece], piece))
        return cls(ranked[:max(0, vocab_size - OFFSET)])

    def encode(self, text):
        return [self.lookup.get(piece, UNK) for piece in PIECE.findall(text)]

    def decode(self, tokens):
        return "".join(self.words[int(token) - OFFSET] for token in tokens if OFFSET <= int(token) < self.vocab_size)

    def serialize(self):
        return {"kind": self.kind, "words": self.words}

    @classmethod
    def restore(cls, data):
        words = data.get("words") if data.get("kind") == cls.kind else None
        if not isinstance(words, list) or not all(isinstance(word, str) and word for word in words) or len(set(words)) != len(words):
            raise ValueError("Invalid whole-word checkpoint.")
        return cls(words)


@register
class Words(ModuleSpec):
    id = "tokenizer/word"
    slot = "tokenizer"
    name = "Whole words"
    blurb = "One token per frequent word or punctuation mark in your data. Short sequences; rare and unseen words become unknown."
    ref = "Bengio et al. 2003"
    config = WordConfig
    backward = "No backward pass"

    def vocab(self, cfg):
        return None

    def fit(self, cfg, examples):
        return WordTokenizer.fit(examples, cfg.vocab_size)

    def restore(self, data):
        return WordTokenizer.restore(data)
