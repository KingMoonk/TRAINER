"""Training windows on disk: data/shards/<key>/, read back with np.memmap so the token buffers never have to fit in memory.

A folder holds the two splits of one tokenized training set, `train` and `val`, each as three flat files:
  <split>.tokens.bin   every window's tokens, back to back (uint16; uint32 when the vocabulary needs it)
  <split>.flags.bin    one byte per target position, 1 = trained (left out when every position is a target)
  <split>.offsets.bin  int64: window i is tokens[offsets[i]:offsets[i + 1]]; it is also the document boundary index
and `index.json` (the format, the dtype and each split's counts), written last: a folder without it is unfinished and is
rebuilt. The key names what the files hold: the data split, the tokenizer and the windowing (see `key_for`), so changing any
of them makes a new folder and the old one stays until Resources > Storage deletes it. Beside them, `TokenCache` keeps the encoded examples themselves (tokens-<key>/), keyed by tokenizer and example text, so a new window folder is cut from it without tokenizing text an earlier run already did.
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


def release_maps(*arrays):
    """Close the memory maps among `arrays` now, so a Windows file is not left locked until the garbage collector gets to them."""
    for array in arrays:
        mapped = getattr(array, "_mmap", None)
        if mapped is not None:
            try:
                mapped.close()
            except (BufferError, ValueError, OSError):
                pass


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
    opened = []
    try:
        for kind, kind_dtype, size in (("tokens", dtype, entry["tokens"]), ("flags", np.uint8, flag_count), ("offsets", np.int64, entry["windows"] + 1)):
            opened.append(Ones(flag_count) if kind == "flags" and entry["all_targets"] else _map(directory / f"{name}.{kind}.bin", kind_dtype, size))
    except BaseException:
        release_maps(*opened)  # a half-opened split must not keep its files locked
        raise
    return tuple(opened)


def load(root, key, vocab_size):
    """{split: (tokens, flags, offsets, entry)} of a finished folder, or None when there is none or it cannot be read."""
    directory = Path(root) / key
    try:
        index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        if index["format"] != FORMAT or index["vocab_size"] != vocab_size:
            return None
        found = {}
        try:
            for name in SPLITS:
                found[name] = (*open_split(directory, name, vocab_size, index[name]), index[name])
            os.utime(directory / "index.json")  # Storage ages folders by their newest file: a folder in use is not old
        except BaseException:
            for opened in found.values():
                release_maps(*opened[:3])  # the splits opened before the failure must not keep the folder locked
            raise
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


class TokenCache:
    """The encoded examples of one tokenizer, kept across runs: root/tokens-<key>/seg-<id>/, each segment a few thousand examples.

    A segment holds, back to back and memory-mapped on read: `tokens.bin` (the ids, as in a shard), `mask.bin` (one byte per token,
    1 = a training target), `aux.bin` (int32 per example: how many role positions and assistant spans, then the positions, the role
    ids and the spans' bounds - what `studio_model.cut_windows` needs besides the tokens), `offsets.bin` / `aux_offsets.bin` (int64:
    example i is tokens[offsets[i]:offsets[i + 1]]), `fps.bin` (16 bytes of each example's fingerprint) and `index.json`, written
    last: a segment without it is unfinished and ignored.

    The key is the tokenizer and whether only assistant text is a target; examples are found by fingerprint, so the cache does not care
    which split an example is in, in what order, or what else the data holds. A run encodes only the examples no segment has."""

    SEGMENT_TOKENS = 1 << 24  # a segment closes once its examples hold this many tokens (32 MB of ids)
    SEGMENT_EXAMPLES = 1 << 15

    def __init__(self, root, tokenizer_hash, vocab_size, assistant_only):
        self.vocab_size, self.dtype = vocab_size, dtype_for(vocab_size)
        self.folder = Path(root) / f"tokens-{key_for('tokencache', tokenizer_hash, vocab_size, bool(assistant_only))}"
        self.maps = {}  # segment number -> its opened files
        self.names, self.where = [], {}
        self._scan()

    @staticmethod
    def digest(fingerprint):
        """The 16 bytes a fingerprint (a hex SHA-256) is stored as."""
        return bytes.fromhex(fingerprint[:32])

    def _scan(self):
        """Index every finished segment: where[digest] = (segment number, example number)."""
        self.names, self.where = [], {}
        for path in sorted(self.folder.glob("seg-*")):
            try:
                entry = json.loads((path / "index.json").read_text(encoding="utf-8"))
                if entry["format"] != FORMAT or entry["dtype"] != np.dtype(self.dtype).name:
                    continue
                count = entry["examples"]
                raw = (path / "fps.bin").read_bytes()
                sizes = {"tokens": entry["tokens"] * np.dtype(self.dtype).itemsize, "mask": entry["tokens"], "aux": entry["aux"] * 4,
                         "offsets": (count + 1) * 8, "aux_offsets": (count + 1) * 8}
                if len(raw) != 16 * count or any((path / f"{kind}.bin").stat().st_size != size for kind, size in sizes.items()):
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                continue
            number = len(self.names)
            self.names.append(path.name)
            for i in range(count):
                self.where.setdefault(raw[16 * i:16 * i + 16], (number, i))

    def __contains__(self, fingerprint):
        return self.digest(fingerprint) in self.where

    def touch(self):
        """Mark the cache as in use, so Storage ages it by its last run, not by when its oldest segment was written."""
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / "used").write_text("", encoding="utf-8")
            os.utime(self.folder / "used")
        except OSError:
            pass

    def add(self, items, encode, progress=None):
        """Encode `items` - (fingerprint, example, weight) triples - that the cache lacks into new segments. `encode(example)` returns
        what studio_model._encode_arrays does; progress(weight) is told after each example. Returns how many it encoded."""
        seen, writer = set(), None
        try:
            for fingerprint, example, weight in items:
                key = self.digest(fingerprint)
                if key in self.where or key in seen:
                    continue
                seen.add(key)
                if writer is None:
                    writer = _SegmentWriter(self.folder, self.dtype)
                writer.add(key, *encode(example))
                if progress:
                    progress(weight)
                if writer.tokens >= self.SEGMENT_TOKENS or writer.count >= self.SEGMENT_EXAMPLES:
                    writer.close()
                    writer = None
            if writer is not None:
                writer.close()
                writer = None
        finally:
            if writer is not None:
                writer.abandon()
        if seen:
            self._scan()
        return len(seen)

    def _files(self, number):
        if number not in self.maps:
            path = self.folder / self.names[number]
            entry = json.loads((path / "index.json").read_text(encoding="utf-8"))
            self.maps[number] = (_map(path / "tokens.bin", self.dtype, entry["tokens"]), _map(path / "mask.bin", np.uint8, entry["tokens"]),
                                 _map(path / "aux.bin", np.int32, entry["aux"]), _map(path / "offsets.bin", np.int64, entry["examples"] + 1),
                                 _map(path / "aux_offsets.bin", np.int64, entry["examples"] + 1))
        return self.maps[number]

    def get(self, fingerprint):
        """(tokens int32, target mask bool, role positions, role ids, spans) of a cached example, as _encode_arrays returns them."""
        number, i = self.where[self.digest(fingerprint)]
        tokens, mask, aux, offsets, aux_offsets = self._files(number)
        low, high = int(offsets[i]), int(offsets[i + 1])
        extra = np.asarray(aux[int(aux_offsets[i]):int(aux_offsets[i + 1])])
        positions_count, spans_count = int(extra[0]), int(extra[1])
        body = extra[2:]
        positions = body[:positions_count].tolist()
        roles = body[positions_count:2 * positions_count].tolist()
        bounds = body[2 * positions_count:2 * positions_count + 2 * spans_count].tolist()
        return (np.asarray(tokens[low:high], dtype=np.int32), np.asarray(mask[low:high], dtype=bool), positions, roles,
                list(zip(bounds[::2], bounds[1::2])))

    def close(self):
        """Release the memory maps (a Windows file stays locked while one is open)."""
        for files in self.maps.values():
            release_maps(*files)
        self.maps.clear()


class _SegmentWriter:
    """One TokenCache segment being written into a .part folder that close() renames to its name."""

    def __init__(self, folder, dtype):
        folder.mkdir(parents=True, exist_ok=True)
        self.folder, self.dtype = folder, dtype
        self.name = f"seg-{uuid.uuid4().hex[:12]}"
        self.scratch = folder / f"{self.name}.part"
        self.scratch.mkdir()
        self.files = {kind: (self.scratch / f"{kind}.bin").open("wb") for kind in ("tokens", "mask", "aux", "offsets", "aux_offsets", "fps")}
        self.count = self.tokens = self.aux = 0
        self.offsets, self.aux_offsets = [0], [0]

    def add(self, key, tokens, mask, positions, roles, spans):
        extra = np.array([len(positions), len(spans), *positions, *roles, *[bound for span in spans for bound in span]], dtype=np.int32)
        self.files["tokens"].write(np.ascontiguousarray(tokens, dtype=self.dtype).tobytes())
        self.files["mask"].write(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
        self.files["aux"].write(extra.tobytes())
        self.files["fps"].write(key)
        self.count += 1
        self.tokens += len(tokens)
        self.aux += len(extra)
        self.offsets.append(self.tokens)
        self.aux_offsets.append(self.aux)

    def close(self):
        self.files["offsets"].write(np.asarray(self.offsets, dtype=np.int64).tobytes())
        self.files["aux_offsets"].write(np.asarray(self.aux_offsets, dtype=np.int64).tobytes())
        for stream in self.files.values():
            stream.close()
        (self.scratch / "index.json").write_text(json.dumps({"format": FORMAT, "dtype": np.dtype(self.dtype).name, "examples": self.count,
                                                              "tokens": self.tokens, "aux": self.aux}), encoding="utf-8")
        try:
            self.scratch.rename(self.folder / self.name)
        finally:
            shutil.rmtree(self.scratch, ignore_errors=True)

    def abandon(self):
        for stream in self.files.values():
            stream.close()
        shutil.rmtree(self.scratch, ignore_errors=True)
