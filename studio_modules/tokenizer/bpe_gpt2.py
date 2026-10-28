"""GPT-2 byte-level BPE using OpenAI's published encoder and merge assets, executed in native Rust (`tokenizers`)."""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from studio_core import replace_file
from studio_modules.tokenizer.bpe_fast import _tokenizers
from studio_registry import ModuleSpec, register
from studio_slots import OFFSET

# Published hashes of OpenAI's GPT-2 assets (fill in from a trusted download to pin them; empty = not checked).
ASSET_SHA256 = {"encoder.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783", "vocab.bpe": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"}
BASE_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/"
ASSETS = ("encoder.json", "vocab.bpe")


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
        self._tok = None

    @property
    def native(self):
        """The native Rust tokenizer, built on first use from OpenAI's encoder and merge table."""
        if self._tok is None:
            if not set(self.byte_encoder.values()).issubset(self.encoder):
                raise ValueError("GPT-2 vocabulary asset is incomplete or inconsistent with its merges.")
            tk = _tokenizers()
            try:
                tok = tk.Tokenizer(tk.models.BPE(vocab=self.encoder, merges=self.merges))
            except Exception as error:
                raise ValueError("GPT-2 vocabulary asset is incomplete or inconsistent with its merges.") from error
            tok.pre_tokenizer = tk.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
            tok.decoder = tk.decoders.ByteLevel()
            self._tok = tok
        return self._tok

    def encode(self, text):
        return [i + OFFSET for i in self.native.encode(text).ids] if text else []

    def encode_batch(self, texts):
        texts = list(texts)
        return [[i + OFFSET for i in e.ids] for e in self.native.encode_batch(texts)]

    def encode_array(self, text):
        import numpy as np
        return np.asarray(self.encode(text), dtype=np.int64)

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
        from studio_core import ROOT  # the writable data folder, also inside a frozen build
        root = ROOT / root
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
            digest = hashlib.sha256(data).hexdigest()
            if ASSET_SHA256.get(path.name) and digest != ASSET_SHA256[path.name]:
                raise OSError(f"{path.name} did not match its known checksum (got {digest[:12]}…).")
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_bytes(data)
            replace_file(temporary, path)
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
