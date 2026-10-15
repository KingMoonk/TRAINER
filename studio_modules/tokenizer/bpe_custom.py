"""Deterministic byte-pair tokenizer trained from the selected dataset."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from studio_ui_guard import slow
from studio_core import messages_of
from studio_registry import ModuleSpec, register
from studio_slots import OFFSET


@dataclass
class CustomBPEConfig:
    vocab_size: int = field(default=512, metadata=dict(min=263, max=16384,
                                                        help="Maximum vocabulary including the seven shared special tokens."))


def texts_of(examples):
    for example in examples:
        if example.get("text"):
            yield example["text"]
        else:
            for message in messages_of(example):
                yield message["content"]


def _merge(sequence, pair, new_id):
    merged, i, n = [], 0, len(sequence)
    while i < n:
        if i + 1 < n and sequence[i] == pair[0] and sequence[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(sequence[i])
            i += 1
    return merged


class CustomBPETokenizer:
    kind = "bpe_custom"

    def __init__(self, merges):
        self.merges = [(tuple(left), tuple(right)) for left, right in merges]
        self.pieces = [bytes([byte]) for byte in range(256)]
        for left, right in self.merges:
            self.pieces.append(self.pieces[left[0]] + self.pieces[right[0]] if len(left) == len(right) == 1
                               else b"".join(self.pieces[part] for part in left + right))
        self.ranks = {(left[0], right[0]): index for index, (left, right) in enumerate(self.merges)}
        self.vocab_size = len(self.pieces) + OFFSET
        self._cache = {}

    @classmethod
    @slow("train a tokenizer")
    def fit(cls, examples, vocab_size):
        """Learn merges on the table of distinct whitespace-delimited words with their counts.

        Pair counts are kept incrementally: a merge only revisits the words that contain the pair, so fitting
        is linear in the text plus the number of merges, not quadratic.
        """
        words = Counter()
        for text in texts_of(examples):
            for word in text.encode("utf-8").split(b" "):
                words[word + b" "] += 1   # the space travels with the word, as in GPT-2's pretokenizer
        table = {index: (list(word), count) for index, (word, count) in enumerate(words.items())}
        pairs, where = Counter(), {}
        for index, (sequence, count) in table.items():
            for pair in zip(sequence, sequence[1:]):
                pairs[pair] += count
                where.setdefault(pair, set()).add(index)
        merges = []
        next_id = 256
        while next_id + OFFSET < vocab_size and pairs:
            pair = min(pairs, key=lambda item: (-pairs[item], item))
            if pairs[pair] < 2:
                break
            merges.append(((pair[0],), (pair[1],)))
            for index in list(where.get(pair, ())):
                sequence, count = table[index]
                for old in zip(sequence, sequence[1:]):
                    pairs[old] -= count
                    if pairs[old] <= 0:
                        del pairs[old]
                    where[old].discard(index)
                sequence = _merge(sequence, pair, next_id)
                table[index] = (sequence, count)
                for new in zip(sequence, sequence[1:]):
                    pairs[new] += count
                    where.setdefault(new, set()).add(index)
            next_id += 1
        return cls(merges)

    def _encode_word(self, word):
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        sequence = list(word)
        while len(sequence) > 1:
            best = min(zip(sequence, sequence[1:]), key=lambda pair: self.ranks.get(pair, len(self.ranks)))
            if best not in self.ranks:
                break
            sequence = _merge(sequence, best, 256 + self.ranks[best])
        if len(self._cache) < 200_000:
            self._cache[word] = sequence
        return sequence

    def encode(self, text):
        data = text.encode("utf-8")
        words = data.split(b" ")
        output = []
        for i, word in enumerate(words):
            piece = word + b" " if i < len(words) - 1 else word
            if piece:
                output.extend(self._encode_word(piece))
        return [value + OFFSET for value in output]

    def decode(self, tokens):
        return b"".join(self.pieces[int(token) - OFFSET] for token in tokens if OFFSET <= int(token) < self.vocab_size).decode("utf-8", errors="replace")

    def serialize(self):
        return {"kind": self.kind, "merges": [[left[0], right[0]] for left, right in self.merges]}

    @classmethod
    def restore(cls, data):
        if data.get("kind") != cls.kind or not isinstance(data.get("merges"), list):
            raise ValueError("Invalid custom BPE checkpoint.")
        merges = []
        for index, item in enumerate(data["merges"]):
            try:
                left, right = int(item[0]), int(item[1])
            except (TypeError, ValueError, IndexError):
                raise ValueError("Invalid custom BPE checkpoint.") from None
            if not (0 <= left < 256 + index and 0 <= right < 256 + index):
                raise ValueError("Invalid custom BPE checkpoint: merge refers to an unknown piece.")
            merges.append(((left,), (right,)))
        return cls(merges)


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
