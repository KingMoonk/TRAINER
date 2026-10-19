"""Keep test data out of training: group ids that tie a rewrite to its source, and GPT-3's 13-gram overlap check.

Window-free and standard-library only. Text is normalised (lower case, punctuation stripped, whitespace collapsed) and cut
into word 13-grams, each hashed to 64 bits. An evaluation item is flagged when any of its 13-grams also occurs in the
training text (the rule GPT-3 used). An item shorter than 13 words has one gram, the whole item, so only a verbatim
copy of it flags.

Memory is bounded by `max_grams` hashes (8 bytes of payload each, about 70 with the set around them). Training text is
streamed one document at a time. When the set outgrows the bound, the index keeps only grams whose hash falls in
1 of `stride` buckets, and halves again each time it overflows; an item is checked against the same bucket. A long
overlap still flags (the chance of missing one of L shared grams is (1 - 1/stride)^L), a short one may not, so a
sampled index says so (`Index.note`).

studio_data_kinds crc32-hashes 3-word shingles to 32 bits for near-duplicate detection (Jaccard over min-hash sketches).
That is the wrong shape for "does a 13-word run appear anywhere", so only the word-splitting idea is shared.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

N = 13                    # GPT-3's n-gram length
MAX_GRAMS = 4_000_000     # hashes kept before the index starts sampling
_MASK = (1 << 64) - 1
_BASE = 0x100000001B3 | 1  # odd multiplier of the rolling hash (mod 2**64)
_BUCKET_SHIFT = 20        # the bucket test reads high bits, which the polynomial hash mixes best


def words(text):
    """Normalised words: lower case, punctuation dropped (a word is a run of letters, digits and underscores)."""
    return re.findall(r"\w+", str(text or "").lower())


def normalise(text):
    return " ".join(words(text))


def group_id(text, prefix="question"):
    """A group id from the normalised text, so every row built from the same question lands on one side of a split."""
    return f"{prefix}:{hashlib.sha256(normalise(text).encode()).hexdigest()[:32]}"


@lru_cache(maxsize=1 << 20)
def _word_hash(word):
    return int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big")


def grams(text, n=N):
    """The 64-bit hashes of every word n-gram of text (one hash of the whole text when it has fewer than n words)."""
    tokens = [_word_hash(w) for w in words(text)]
    if not tokens:
        return
    if len(tokens) < n:
        value = len(tokens)
        for token in tokens:
            value = (value * _BASE + token) & _MASK
        yield value
        return
    drop, value = pow(_BASE, n, 1 << 64), 0
    for i, token in enumerate(tokens):
        value = (value * _BASE + token) & _MASK
        if i >= n:
            value = (value - tokens[i - n] * drop) & _MASK
        if i >= n - 1:
            yield value


class Index:
    """The 13-grams of the training text. add() streams documents in; overlaps() asks about one evaluation item."""

    def __init__(self, n=N, max_grams=MAX_GRAMS):
        self.n, self.max_grams, self.stride, self.documents = n, max(1, int(max_grams)), 1, 0
        self._grams = set()

    def _kept(self, gram):
        return (gram >> _BUCKET_SHIFT) % self.stride == 0

    def add(self, text):
        self.documents += 1
        self._grams.update(g for g in grams(text, self.n) if self._kept(g))
        while len(self._grams) > self.max_grams:
            self.stride *= 2
            self._grams = {g for g in self._grams if self._kept(g)}

    def add_all(self, texts):
        for text in texts:
            self.add(text)
        return self

    @property
    def sampled(self):
        return self.stride > 1

    @property
    def note(self):
        return (f"Sampled: kept 1 in {self.stride} of the training 13-grams to stay within memory, so a short overlap may be missed."
                if self.sampled else "")

    def overlaps(self, text):
        return any(g in self._grams for g in grams(text, self.n) if self._kept(g))


@dataclass
class Report:
    checked: int = 0
    left_out: int = 0
    note: str = ""

    def as_dict(self):
        return {"checked": self.checked, "left_out": self.left_out, "note": self.note}


def exclude_overlap(items, index, text=lambda item: item):
    """(clean items, Report): the items none of whose 13-grams appear in `index`."""
    items = list(items)
    clean = [item for item in items if not index.overlaps(text(item))]
    return clean, Report(len(items), len(items) - len(clean), index.note)


def row_text(row):
    """One training row as text: a text row, a messages list, or a user and assistant pair (what dataset.jsonl holds)."""
    if row.get("text"):
        return str(row["text"])
    if row.get("messages"):
        return "\n".join(str(m.get("content", "")) for m in row["messages"])
    return "\n".join(str(s) for s in (row.get("user"), row.get("assistant")) if s)


def jsonl_texts(path):
    """The text of each row of a jsonl file, streamed; nothing for a file that is not there. A line that is not JSON is skipped."""
    try:
        stream = open(path, encoding="utf-8")
    except OSError:
        return
    with stream:
        for line in stream:
            try:
                row = json.loads(line) if line.strip() else None
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row_text(row)


def index_of_runs(run_dirs, max_grams=MAX_GRAMS):
    """The Index of the training text of runs: each run folder's dataset.jsonl is the exact text that run trained on."""
    index = Index(max_grams=max_grams)
    for run in run_dirs:
        for text in jsonl_texts(Path(run) / "dataset.jsonl"):
            index.add(text)
    return index


def overlap_mask(items, index, text=lambda item: item):
    """One bool per item: whether it shares a 13-gram with the indexed training text. Use this when a position must stay
    aligned with the item list (exclude_overlap drops them instead)."""
    return [index.overlaps(text(item)) for item in items]
