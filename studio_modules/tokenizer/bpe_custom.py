"""Deterministic byte-pair tokenizer trained from the selected dataset."""
from __future__ import annotations

from dataclasses import dataclass, field

from studio_core import messages_of
from studio_registry import ModuleSpec, register
from studio_slots import OFFSET


@dataclass
class CustomBPEConfig:
    vocab_size: int = field(default=512, metadata=dict(min=263, max=16384,
                                                        help="Maximum vocabulary including the seven shared special tokens."))


def texts_of(examples):
    for example in examples:
        if example.get("text") is not None:
            yield example["text"]
        else:
            for message in messages_of(example):
                yield message["content"]


class CustomBPETokenizer:
    kind = "bpe_custom"

    def __init__(self, merges):
        self.merges = [(tuple(left), tuple(right)) for left, right in merges]
        self.pieces = [bytes([byte]) for byte in range(256)]
        for left, right in self.merges:
            self.pieces.append(self.pieces[left[0]] + self.pieces[right[0]] if len(left) == len(right) == 1
                               else b"".join(self.pieces[part] for part in left + right))
        self.vocab_size = len(self.pieces) + OFFSET

    @classmethod
    def fit(cls, examples, vocab_size):
        sequences = [list(text.encode("utf-8")) for text in texts_of(examples)]
        merges = []
        next_id = 256
        while next_id + OFFSET < vocab_size:
            counts = {}
            for sequence in sequences:
                for pair in zip(sequence, sequence[1:]):
                    counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            pair = min(counts, key=lambda item: (-counts[item], item))
            if counts[pair] < 2:
                break
            merges.append(((pair[0],), (pair[1],)))
            for sequence in sequences:
                merged = []
                i = 0
                while i < len(sequence):
                    if i + 1 < len(sequence) and tuple(sequence[i:i + 2]) == pair:
                        merged.append(next_id)
                        i += 2
                    else:
                        merged.append(sequence[i])
                        i += 1
                sequence[:] = merged
            next_id += 1
        return cls(merges)

    def encode(self, text):
        sequence = list(text.encode("utf-8"))
        for index, (left, right) in enumerate(self.merges):
            pair = (left[0], right[0])
            merged = []
            i = 0
            while i < len(sequence):
                if i + 1 < len(sequence) and tuple(sequence[i:i + 2]) == pair:
                    merged.append(256 + index)
                    i += 2
                else:
                    merged.append(sequence[i])
                    i += 1
            sequence = merged
        return [value + OFFSET for value in sequence]

    def decode(self, tokens):
        return b"".join(self.pieces[int(token) - OFFSET] for token in tokens if int(token) >= OFFSET).decode("utf-8", errors="replace")

    def serialize(self):
        return {"kind": self.kind, "merges": [[left[0], right[0]] for left, right in self.merges]}

    @classmethod
    def restore(cls, data):
        if data.get("kind") != cls.kind or "merges" not in data:
            raise ValueError("Invalid custom BPE checkpoint.")
        return cls([((int(left),), (int(right),)) for left, right in data["merges"]])


@register
class CustomBPE(ModuleSpec):
    id = "tokenizer/bpe_custom"
    slot = "tokenizer"
    name = "Custom BPE"
    blurb = "Trains deterministic byte-pair merges from the selected data."
    ref = "Sennrich et al. 2016"
    config = CustomBPEConfig
    backward = "No backward pass"

    def vocab(self, cfg):
        return None

    def fit(self, cfg, examples):
        return CustomBPETokenizer.fit(examples, cfg.vocab_size)

    def restore(self, data):
        return CustomBPETokenizer.restore(data)
