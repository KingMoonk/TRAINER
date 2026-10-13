"""GPT-2 byte-level BPE using OpenAI's published encoder and merge assets."""
from __future__ import annotations

import json
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from studio_registry import ModuleSpec, register
from studio_slots import OFFSET

BASE_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/"
ASSETS = ("encoder.json", "vocab.bpe")
CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def bytes_to_unicode():
    visible = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    numbers = visible[:]
    next_code = 256
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            numbers.append(next_code)
            next_code += 1
    return dict(zip(visible, map(chr, numbers)))


def category(char):
    # Match regex \s: Zs, Zl, Zp categories + ASCII whitespace and NEL (U+0085).
    cat = unicodedata.category(char)
    if cat in ("Zs", "Zl", "Zp") or char in (" ", "\t", "\n", "\r", "\f", "\v", "\x85"):
        return "space"
    group = cat[0]
    return "letter" if group == "L" else "number" if group == "N" else "other"


def pretokenize(text):
    """Stdlib Unicode-category scan of GPT-2's ordered regex alternatives."""
    pieces = []
    i = 0
    while i < len(text):
        contraction = next((part for part in CONTRACTIONS if text.startswith(part, i)), None)
        if contraction:
            pieces.append(contraction)
            i += len(contraction)
            continue
        start = i
        if text[i] == " " and i + 1 < len(text) and category(text[i + 1]) != "space":
            i += 1
        kind = category(text[i])
        if kind == "space":
            while i < len(text) and category(text[i]) == "space":
                i += 1
            if i < len(text) and i - start > 1:
                i -= 1
        else:
            while i < len(text) and category(text[i]) == kind:
                i += 1
        pieces.append(text[start:i])
    return pieces


def merge_word(word, ranks):
    symbols = list(word)
    while len(symbols) > 1:
        choices = [(ranks.get((symbols[i], symbols[i + 1]), float("inf")), i) for i in range(len(symbols) - 1)]
        rank, _ = min(choices)
        if rank == float("inf"):
            break
        merged = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and ranks.get((symbols[i], symbols[i + 1])) == rank:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
    return symbols


class GPT2Tokenizer:
    kind = "bpe_gpt2"

    def __init__(self, encoder, merges):
        self.encoder = {key: int(value) for key, value in encoder.items()}
        self.decoder = {value: key for key, value in self.encoder.items()}
        self.merges = [tuple(pair) for pair in merges]
        self.ranks = dict(zip(self.merges, range(len(self.merges))))
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        self.vocab_size = max(self.decoder) + 1 + OFFSET

    def encode(self, text):
        output = []
        for piece in pretokenize(text):
            mapped = "".join(self.byte_encoder[byte] for byte in piece.encode("utf-8"))
            for symbol in merge_word(mapped, self.ranks):
                if symbol not in self.encoder:
                    raise ValueError("GPT-2 vocabulary asset is incomplete or inconsistent with its merges.")
                output.append(self.encoder[symbol] + OFFSET)
        return output

    def decode(self, tokens):
        symbols = "".join(self.decoder[int(token) - OFFSET] for token in tokens if int(token) >= OFFSET)
        return bytes(self.byte_decoder[char] for char in symbols).decode("utf-8", errors="replace")

    def serialize(self):
        return {"kind": self.kind, "encoder": self.encoder, "merges": self.merges}

    @classmethod
    def restore(cls, data):
        if data.get("kind") != cls.kind or "encoder" not in data or "merges" not in data:
            raise ValueError("Invalid GPT-2 tokenizer checkpoint.")
        return cls(data["encoder"], data["merges"])


@dataclass
class GPT2Config:
    asset_dir: str = field(default="data/gpt2", metadata=dict(help="Directory for encoder.json and vocab.bpe."))


def asset_paths(cfg):
    root = Path(cfg.asset_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    return [root / name for name in ASSETS]


def load_assets(cfg):
    paths = asset_paths(cfg)
    for path in paths:
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(BASE_URL + path.name, timeout=30) as response:
                data = response.read()
            path.write_bytes(data)
        except (OSError, urllib.error.URLError) as error:
            raise ValueError(f"GPT-2 tokenizer asset {path.name} is unavailable; connect to the internet or place it in {path.parent}.") from error
    encoder = json.loads(paths[0].read_text(encoding="utf-8"))
    lines = paths[1].read_text(encoding="utf-8").splitlines()[1:]
    merges = [tuple(line.split()) for line in lines if line and not line.startswith("#")]
    return GPT2Tokenizer(encoder, merges)


@register
class GPT2BPE(ModuleSpec):
    id = "tokenizer/bpe_gpt2"
    slot = "tokenizer"
    name = "GPT-2 BPE"
    blurb = "OpenAI's 50,257-token byte-level BPE, with assets downloaded on first use."
    ref = "Radford et al. 2019"
    config = GPT2Config
    backward = "No backward pass"

    def vocab(self, cfg):
        return 50257 + OFFSET

    def fit(self, cfg, examples):
        return load_assets(cfg)

    def restore(self, data):
        return GPT2Tokenizer.restore(data)
