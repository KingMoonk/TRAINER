"""Fast byte-level subword tokenizer learned from a stream of the whole corpus (Hugging Face `tokenizers` does the counting and encoding).

Learns 1k to 64k pieces from a sample drawn evenly across the corpus in bounded memory, and encodes in native Rust code, so a tokenizer is not
the bottleneck of the sharded path. `tokenizers` ships wheels for Windows, macOS and Linux (abi3), so CI installs it from requirements.

The vocabulary is the 256 bytes plus one piece per merge, so every byte of any input has a token (no unknown token): text that
is valid UTF-8 splits by the modern GPT-4 / Llama 3 (`cl100k_base`) regex pattern (contractions, letter runs, 1-3 digit numbers, punctuation
and newlines) before byte-level mapping, and bytes that are not valid UTF-8 become single byte pieces (`encode_bytes`).

What a checkpoint stores is the merge list (`serialize`), a few hundred kilobytes, with a SHA-256 of it that `restore` checks.
The same list is a shareable artifact (`save_artifact` / `load_artifact`): DiLoCo peers receive it from the coordinator and
verify the hash, so every peer cuts text identically.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from studio_core import messages_of
from studio_registry import ModuleSpec, register
from studio_slots import OFFSET

FORMAT = "groundwork-tokenizer/1"
MIN_VOCAB, MAX_VOCAB = 1024, 65536
REPLACEMENT = chr(0xFFFD)
# Modern GPT-4 (cl100k_base) / Llama 3 regex split: case-insensitive English contractions, words with optional leading non-letter/digit,
# 1-3 digit numbers, punctuation runs with optional trailing newlines, newline runs, and whitespace runs.
SPLIT_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def _pre_tokenizer(tk):
    return tk.pre_tokenizers.Sequence([
        tk.pre_tokenizers.Split(pattern=tk.Regex(SPLIT_PATTERN), behavior="isolated"),
        tk.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])


@dataclass
class FastBPEConfig:
    vocab_size: int = field(default=16384, metadata=dict(min=MIN_VOCAB, max=MAX_VOCAB,
                                                         help="Pieces in the vocabulary including the 7 shared special tokens (256 are single bytes). Larger cuts text into fewer pieces but makes the word table bigger."))
    fit_chars: int = field(default=200_000_000, metadata=dict(min=100_000, max=4_000_000_000,
                                                              help="Characters of text the vocabulary is learned from. A larger corpus is sampled evenly (every k-th document), which bounds memory."))
    min_frequency: int = field(default=2, metadata=dict(min=1, max=1000, help="A pair must occur this often to become a piece."))


def suggest_vocab(d_model, n_layer, ceiling=32768):
    """A vocabulary size for a model of this size: the word table (vocab x width) at most a third of the parameters in the blocks,
    rounded to a multiple of 1,024 and kept between 1,024 and `ceiling`. A small model gets a small table; a 100M one gets 32k."""
    blocks = n_layer * 12 * d_model * d_model
    wanted = blocks / 3 / max(1, d_model)
    return int(min(ceiling, max(MIN_VOCAB, round(wanted / 1024) * 1024)))


def _byte_chars():
    """GPT-2's byte -> printable character map: the alphabet the trainer sees, one character per byte."""
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars, extra = {}, 0
    for byte in range(256):
        if byte in keep:
            chars[byte] = chr(byte)
        else:
            chars[byte] = chr(256 + extra)
            extra += 1
    return chars


BYTE_CHAR = _byte_chars()


def texts_of(examples):
    for example in examples:
        if example.get("text"):
            yield example["text"]
        else:
            for message in messages_of(example):
                yield message["content"]


def sample_texts(examples, fit_chars, progress=None):
    """The texts to learn from: all of them when they fit `fit_chars`, else every k-th text of the stream so the sample
    spans the whole corpus (not just its start). A sized corpus is indexed at the stride, so only the sampled rows are read, and when it knows its
    character total (`characters()`, as spool-backed Rows do) there is no counting pass either; any other list is summed first.
    An iterator is read once, from its start. `progress(done, total)` hears the characters taken against the budget."""
    taken = 0
    if hasattr(examples, "__len__"):
        known = getattr(examples, "characters", None)
        total = known() if callable(known) else sum(len(text) for text in texts_of(examples))
        stride = max(1, math.ceil(total / fit_chars))
        goal = min(total, fit_chars)
        if hasattr(examples, "__getitem__"):
            source = (text for index in range(0, len(examples), stride) for text in texts_of([examples[index]]))
        else:
            source = (text for index, text in enumerate(texts_of(examples)) if index % stride == 0)
        for text in source:
            taken += len(text)
            yield text
            if progress:
                progress(taken, goal)
            if taken >= fit_chars:
                return
        return
    for text in texts_of(examples):
        taken += len(text)
        yield text
        if progress:
            progress(taken, fit_chars)
        if taken >= fit_chars:
            return


def _tokenizers():
    try:
        import tokenizers
    except ImportError as error:
        raise ImportError("The fast subword tokenizer needs the `tokenizers` package: pip install -r requirements.txt") from error
    return tokenizers


def merge_digest(merges):
    return hashlib.sha256(json.dumps([[int(a), int(b)] for a, b in merges], separators=(",", ":")).encode()).hexdigest()


def _surrogate(char):
    return 0xD800 <= ord(char) <= 0xDFFF


class FastBPETokenizer:
    kind = "bpe_fast"

    def __init__(self, merges):
        self.merges = [(int(a), int(b)) for a, b in merges]
        symbols = [BYTE_CHAR[byte] for byte in range(256)]
        self.pieces = [bytes([byte]) for byte in range(256)]
        for index, (left, right) in enumerate(self.merges):
            if not (0 <= left < 256 + index and 0 <= right < 256 + index):
                raise ValueError("Invalid fast BPE merges: a merge refers to an unknown piece.")
            symbols.append(symbols[left] + symbols[right])
            self.pieces.append(self.pieces[left] + self.pieces[right])
        self.symbols = symbols
        self.vocab_size = len(self.pieces) + OFFSET
        self.digest = merge_digest(self.merges)
        self._tok = None

    @property
    def native(self):
        """The native tokenizer, built on first use from the merges."""
        if self._tok is None:
            tk = _tokenizers()
            vocab = {symbol: index for index, symbol in enumerate(self.symbols)}
            merges = [(self.symbols[a], self.symbols[b]) for a, b in self.merges]
            tok = tk.Tokenizer(tk.models.BPE(vocab=vocab, merges=merges))
            tok.pre_tokenizer = _pre_tokenizer(tk)
            tok.decoder = tk.decoders.ByteLevel()
            self._tok = tok
        return self._tok

    # -- fitting --------------------------------------------------------------------------------------------------
    @classmethod
    def fit(cls, examples, vocab_size=16384, fit_chars=200_000_000, min_frequency=2, progress=None):
        """Learn `vocab_size - 7 - 256` merges from a sample of `examples` (any iterable of examples). Deterministic for the same
        sample, vocabulary size and library version; memory follows the number of distinct words in the sample, not the corpus."""
        tk = _tokenizers()
        tok = tk.Tokenizer(tk.models.BPE())
        tok.pre_tokenizer = _pre_tokenizer(tk)
        trainer = tk.trainers.BpeTrainer(vocab_size=max(256, vocab_size - OFFSET), min_frequency=min_frequency, show_progress=False,
                                         initial_alphabet=tk.pre_tokenizers.ByteLevel.alphabet(), special_tokens=[])
        tok.train_from_iterator(sample_texts(examples, fit_chars, progress), trainer)
        ids = {BYTE_CHAR[byte]: byte for byte in range(256)}
        merges = []
        for entry in json.loads(tok.to_str())["model"]["merges"]:
            left, right = entry if isinstance(entry, list) else entry.split(" ")
            merges.append((ids[left], ids[right]))
            ids[left + right] = 255 + len(merges)
        return cls(merges)

    # -- encoding -------------------------------------------------------------------------------------------------
    def _ids(self, text):
        return [i + OFFSET for i in self.native.encode(text).ids] if text else []

    def encode(self, text):
        """Token ids of a string. A lone surrogate that carries a raw byte (U+DC80..DCFF, as `surrogateescape` writes it) becomes
        that byte's piece; any other lone surrogate reads as U+FFFD."""
        if text.isascii() or not any(_surrogate(char) for char in text):
            return self._ids(text)
        out, start = [], 0
        for index, char in enumerate(text):
            if _surrogate(char):
                out.extend(self._ids(text[start:index]))
                out.append(ord(char) - 0xDC00 + OFFSET) if 0xDC80 <= ord(char) <= 0xDCFF else out.extend(self._ids(REPLACEMENT))
                start = index + 1
        out.extend(self._ids(text[start:]))
        return out

    def encode_batch(self, texts):
        """encode() for many texts at once (the native code uses every core)."""
        texts = list(texts)
        if any(not t.isascii() and any(_surrogate(c) for c in t) for t in texts):
            return [self.encode(t) for t in texts]
        return [[i + OFFSET for i in e.ids] for e in self.native.encode_batch(texts)]

    def encode_array(self, text):
        import numpy as np
        return np.asarray(self.encode(text), dtype=np.int64)

    def encode_bytes(self, data):
        """Token ids of raw bytes: valid UTF-8 as text, each byte that is not part of valid UTF-8 as its own piece."""
        return self.encode(bytes(data).decode("utf-8", errors="surrogateescape"))

    # -- decoding -------------------------------------------------------------------------------------------------
    def decode_bytes(self, tokens):
        return b"".join(self.pieces[int(t) - OFFSET] for t in tokens if OFFSET <= int(t) < self.vocab_size)

    def decode(self, tokens):
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")

    def characters_per_token(self, texts):
        """Characters of text per piece on `texts` (the reciprocal of the pieces per character the plan shows)."""
        texts = list(texts)
        pieces = sum(len(ids) for ids in self.encode_batch(texts))
        return sum(len(t) for t in texts) / max(1, pieces)

    # -- the checkpoint form and the shareable artifact -----------------------------------------------------------
    def serialize(self):
        return {"kind": self.kind, "merges": [[a, b] for a, b in self.merges], "sha256": self.digest}

    @classmethod
    def restore(cls, data):
        if data.get("kind") != cls.kind or not isinstance(data.get("merges"), list):
            raise ValueError("Invalid fast BPE checkpoint.")
        try:
            tokenizer = cls([(item[0], item[1]) for item in data["merges"]])
        except (TypeError, ValueError, IndexError):
            raise ValueError("Invalid fast BPE checkpoint.") from None
        if data.get("sha256") not in (None, tokenizer.digest):
            raise ValueError("The tokenizer's merges do not match their hash: it was changed or damaged.")
        return tokenizer

    def save_artifact(self, path):
        """Write the tokenizer as a self-describing JSON file and return its SHA-256 (the id peers compare)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"format": FORMAT, **self.serialize()}, separators=(",", ":")), encoding="utf-8")
        return self.digest

    @classmethod
    def load_artifact(cls, path, sha256=None):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError("Not a tokenizer artifact.")
        tokenizer = cls.restore(data)
        if sha256 and sha256 != tokenizer.digest:
            raise ValueError("This tokenizer is not the one expected: its hash differs.")
        return tokenizer


@register
class FastBPE(ModuleSpec):
    id = "tokenizer/bpe_fast"
    slot = "tokenizer"
    name = "Learned word pieces (fast)"
    blurb = "Byte-level subword pieces learned from a sample of all your text, encoded in native code. About 4 times more text per context window than bytes."
    ref = "Sennrich et al. 2016; Radford et al. 2019; OpenAI cl100k / Llama 3 byte-level BPE"
    config = FastBPEConfig
    backward = "No backward pass"

    def vocab(self, cfg):
        return None

    def fit(self, cfg, examples, progress=None):
        return FastBPETokenizer.fit(examples, cfg.vocab_size, cfg.fit_chars, cfg.min_frequency, progress)

    def restore(self, data):
        return FastBPETokenizer.restore(data)
