"""Training windows on disk: data/shards/<key>/, read back with np.memmap so the token buffers never have to fit in memory.

A folder holds the two splits of one tokenized training set, `train` and `val`, each as three flat files:
  <split>.tokens.bin   every window's tokens, back to back (uint16; uint32 when the vocabulary needs it)
  <split>.flags.bin    one byte per target position, 1 = trained (left out when every position is a target)
  <split>.offsets.bin  int64: window i is tokens[offsets[i]:offsets[i + 1]]; it is also the document boundary index
and `index.json` (the format, the dtype and each split's counts), written last: a folder without it is unfinished and is
rebuilt. The key names what the files hold: the data split, the tokenizer and the windowing (see `key_for`), so changing any
of them makes a new folder and the old one stays until Resources > Storage deletes it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np

from studio_slots import BOS

FORMAT = 1
SPLITS = ("train", "val")
CHUNK = 1 << 16  # offsets are appended to the file this many at a time


def key_for(*parts):
    """Short name of a cache folder: a hash of everything that decides what the windows contain."""
    return hashlib.sha256(json.dumps([FORMAT, *parts], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def tokenizer_folder(root, split_hash, spec):
    """root/tok-<key>: everything cached for this data split under this tokenizer spec (id and config) - the fitted tokenizer and,
    in folders of their own, its windows. Another spec on the same data is another folder, so each format chosen keeps its own."""
    return Path(root) / f"tok-{key_for('tokenizer', split_hash, spec)}"


def load_tokenizer(folder):
    """The serialized tokenizer fitted earlier into `folder`, or None when there is none or the file cannot be read."""
    try:
        data = json.loads((Path(folder) / "tokenizer.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def save_tokenizer(folder, data):
    """Write the serialized tokenizer to folder/tokenizer.json through a temporary file, so it is whole or absent."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    scratch = folder / f"tokenizer.json.{uuid.uuid4().hex[:8]}.part"
    try:
        scratch.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(scratch, folder / "tokenizer.json")
    finally:
        scratch.unlink(missing_ok=True)


def dtype_for(vocab_size):
    return np.uint16 if vocab_size <= 65535 else np.uint32


class ShardWriter:
    """Takes windows as `_WindowBuilder` does (add(segment, flags)) but appends them to files instead of memory."""

    def __init__(self, directory, name, vocab_size):
        self.dtype, self.paths = dtype_for(vocab_size), {kind: Path(directory) / f"{name}.{kind}.bin" for kind in ("tokens", "flags", "offsets")}
        self.files = {kind: path.open("wb") for kind, path in self.paths.items()}
        self.tokens, self.windows, self.targets, self.from_start, self.all_targets = 0, 0, 0, 0, True
        self.pending = [0]  # offsets not yet written; the first window starts at 0

    def add(self, segment, flags):
        segment = np.ascontiguousarray(segment, dtype=self.dtype)
        flags = np.ascontiguousarray(flags, dtype=np.uint8)
        self.files["tokens"].write(segment.tobytes())
        self.files["flags"].write(flags.tobytes())
        self.tokens += len(segment)
        self.windows += 1
        self.targets += int(flags.sum())
        self.from_start += int(segment[0] == BOS)
        self.all_targets = self.all_targets and int(flags.sum()) == len(flags)
        self.pending.append(self.tokens)
        if len(self.pending) >= CHUNK:
            self._flush()

    def _flush(self):
        self.files["offsets"].write(np.asarray(self.pending, dtype=np.int64).tobytes())
        self.pending = []

    def close(self):
        """Finish the files; returns the split's entry in index.json."""
        if self.pending:
            self._flush()
        for stream in self.files.values():
            stream.close()
        if self.all_targets:
            self.paths["flags"].unlink()  # nothing to mask: the reader makes the ones itself
        return {"windows": self.windows, "tokens": self.tokens, "targets": self.targets, "from_start": self.from_start, "all_targets": self.all_targets}


class Ones:
    """The flags of windows that train on every position: nothing stored, ones on demand (what WindowStore slices and sums)."""

    nbytes = 0

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return np.ones(len(range(*index.indices(self.size))), dtype=np.uint8)

    def sum(self):
        return self.size


def _map(path, dtype, size):
    return np.memmap(path, dtype=dtype, mode="r", shape=(size,)) if size else np.empty(0, dtype=dtype)  # mmap refuses an empty file


def open_split(directory, name, vocab_size, entry):
    """(tokens, flags, offsets) of one split as read-only memory maps; ValueError when the files are not what the index says."""
    dtype, directory = dtype_for(vocab_size), Path(directory)
    flag_count = entry["tokens"] - entry["windows"]
    expected = {"tokens": entry["tokens"] * np.dtype(dtype).itemsize, "offsets": (entry["windows"] + 1) * 8}
    if not entry["all_targets"]:
        expected["flags"] = flag_count
    for kind, size in expected.items():
        if (directory / f"{name}.{kind}.bin").stat().st_size != size:
            raise ValueError(f"{name}.{kind}.bin is not the size its index says.")
    flags = Ones(flag_count) if entry["all_targets"] else _map(directory / f"{name}.flags.bin", np.uint8, flag_count)
    return (_map(directory / f"{name}.tokens.bin", dtype, entry["tokens"]), flags,
            _map(directory / f"{name}.offsets.bin", np.int64, entry["windows"] + 1))


def load(root, key, vocab_size):
    """{split: (tokens, flags, offsets, entry)} of a finished folder, or None when there is none or it cannot be read."""
    directory = Path(root) / key
    try:
        index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        if index["format"] != FORMAT or index["vocab_size"] != vocab_size:
            return None
        found = {name: (*open_split(directory, name, vocab_size, index[name]), index[name]) for name in SPLITS}
        os.utime(directory / "index.json")  # Storage ages folders by their newest file: a folder in use is not old
        return found
    except (OSError, ValueError, KeyError, TypeError):
        return None


def build(root, key, vocab_size, fill):
    """Run fill({split: ShardWriter}) into a scratch folder and move it to root/key. Returns load()'s result."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    scratch = root / f"{key}.{uuid.uuid4().hex[:8]}.part"
    scratch.mkdir()
    try:
        writers = {name: ShardWriter(scratch, name, vocab_size) for name in SPLITS}
        try:
            fill(writers)
        finally:
            entries = {name: writer.close() for name, writer in writers.items()}
        (scratch / "index.json").write_text(json.dumps({"format": FORMAT, "vocab_size": vocab_size, **entries}), encoding="utf-8")
        final = root / key
        try:
            scratch.rename(final)
        except OSError:  # a folder is already there: another run just finished the same one, or it is unfinished or unreadable
            if load(root, key, vocab_size) is None:
                shutil.rmtree(final, ignore_errors=True)
                scratch.rename(final)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    found = load(root, key, vocab_size)
    if found is None:
        raise OSError("The token shards could not be read back after writing.")
    return found
