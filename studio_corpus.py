"""A training text read as a stream, not loaded: the rows go to a spool file on disk and only a few bytes per row stay in memory.

`studio_model.train` used to take a list: every row, then a dict of fingerprints, then the split's lists, all at once. A pretraining
run over a few billion characters sat at ~18 GB before its first step (#732). Given an iterator instead, train() calls collect():

- rows are read one at a time, fingerprinted and de-duplicated (the same rule as the list path, duplicates' lineage merged into the
  kept row), and written to the spool;
- reading stops once the unique text is enough for the run's token budget (budget_chars), so a budget below one pass reads less;
- what stays in memory per kept row is its 32-byte fingerprint, its character count, its file offset and its lineage ids, plus the
  set that finds duplicates. Not the text.

Text under `stream_chars` is returned as a plain list, so small data runs exactly the code it always did."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import tempfile
import weakref
from array import array
from pathlib import Path

from studio_core import fingerprint, messages_of
from studio_planning import merge_duplicate_lineage

CHARS_PER_TOKEN = 6  # the most text one token can stand for in practice (a byte token is one character, a big BPE vocabulary about four): reading this many characters per budgeted token covers the budget with room to spare


def budget_chars(token_budget):
    """Characters of unique text that cover a run's token budget (None when the run has no budget)."""
    return None if token_budget is None else int(token_budget) * CHARS_PER_TOKEN


def characters_of(example):
    return len(example["text"]) if example.get("text") else sum(len(m["content"]) for m in messages_of(example))


def _identities(example):
    from studio_model import row_identities
    return row_identities(example)


def _release(files, path):
    """Close the spool's files and delete it (also when the corpus is dropped by a run that failed)."""
    for name, handle in list(files.items()):
        if handle is not None:
            handle.close()
        files[name] = None
    try:
        os.unlink(path)
    except OSError:
        pass


class RowStream:
    """An iterator of rows that says how many characters it expects to hold (None when unknown), so the preparation can show a percent."""

    def __init__(self, rows, expected_characters):
        self.rows, self.expected_characters = rows, expected_characters

    def __iter__(self):
        return iter(self.rows)


class Rows:
    """Some rows of a Corpus in a fixed order, read from the spool one at a time. Quacks like what train() and shard_windows need
    of a list: len(), iteration, rows[i], and meta() (each row's fingerprint and character count, from memory)."""

    def __init__(self, corpus, order):
        self.corpus, self.order = corpus, order

    def __len__(self):
        return len(self.order)

    def __getitem__(self, position):
        return self.corpus.row(self.order[position])

    def __iter__(self):
        for number in self.order:
            yield self.corpus.row(number)

    def meta(self):
        keys, weights = self.corpus.keys, self.corpus.weights
        return ((keys[number].hex(), weights[number]) for number in self.order)

    def characters(self):
        weights = self.corpus.weights
        return sum(weights[number] for number in self.order)


class Corpus:
    """The unique rows of a text, on disk. add() each row in reading order, finish() once, then use the views and snapshot_rows()."""

    def __init__(self, folder):
        Path(folder).mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix="corpus-", suffix=".spool", dir=folder)
        self.path = Path(name)
        self._files = {"out": os.fdopen(handle, "wb", buffering=1 << 20), "in": None}
        self._release = weakref.finalize(self, _release, self._files, name)  # a run that fails leaves no spool behind
        self.keys = []                 # each kept row's 32-byte fingerprint, in reading order
        self.weights = array("q")      # its characters
        self.offsets = array("q")      # where its line starts in the spool
        self.identities = []           # its row_identities (None for most rows)
        self._index = {}               # fingerprint -> row number: what finds a duplicate
        self._merges = {}              # row number -> the duplicates whose lineage joins it (applied when the row is read for the snapshot)
        self._position = 0
        self.digest = hashlib.sha256()  # of the fingerprints in reading order: the dataset digest a resume checks
        self.read = 0                  # rows read, duplicates included
        self.characters = 0            # characters of the kept rows

    def __len__(self):
        return len(self.keys)

    def add(self, example):
        """Keep the row unless one with the same fingerprint was kept; a duplicate's lineage is remembered for the kept row."""
        self.read += 1
        key = bytes.fromhex(fingerprint(example))
        number = self._index.get(key)
        if number is not None:
            lineage = {name: example[name] for name in ("id", "groupId", "provenance") if name in example}
            if lineage:
                self._merges.setdefault(number, []).append(lineage)
            return False
        self._index[key] = len(self.keys)
        line = json.dumps(example, ensure_ascii=False).encode("utf-8") + b"\n"
        self.keys.append(key)
        self.offsets.append(self._position)
        self.weights.append(characters_of(example))
        identities = _identities(example)
        self.identities.append(tuple(identities) if identities else None)
        self.characters += self.weights[-1]
        self.digest.update(key.hex().encode())
        self._files["out"].write(line)
        self._position += len(line)
        return True

    def finish(self):
        """No more rows: close the writer, free the duplicate finder, and settle the lineage of the rows that had duplicates."""
        self._files["out"].close()
        self._index = {}
        self._files["in"] = open(self.path, "rb")
        for number in self._merges:
            self.identities[number] = tuple(_identities(self.row(number, merged=True))) or None

    def row(self, number, merged=False):
        """The kept row `number`. With `merged` the lineage of its duplicates is in it (as the list path merges it)."""
        reader = self._files["in"]
        reader.seek(self.offsets[number])
        example = json.loads(reader.readline())
        if merged and number in self._merges:
            example = dict(example)
            for duplicate in self._merges[number]:
                merge_duplicate_lineage(example, duplicate)
        return example

    def snapshot_rows(self):
        """Every kept row in reading order, lineage merged: what the run's dataset.jsonl holds (a resume rebuilds the split from it)."""
        for number in range(len(self.keys)):
            yield self.row(number, merged=True)

    def everything(self):
        return Rows(self, range(len(self.keys)))

    def partition(self, fraction, sampler, grouped):
        """(training Rows, validation Rows, report, split_hash): studio_model.partition_examples over the corpus, with the split hash the
        list path computes (the fingerprints of the training rows then the validation rows)."""
        from studio_model import partition_indices
        training, validation, report = partition_indices(len(self.keys), lambda number: self.identities[number] or (), fraction, sampler, grouped)
        digest = hashlib.sha256()
        for number in training:
            digest.update(self.keys[number].hex().encode())
        digest.update(b"|")
        for number in validation:
            digest.update(self.keys[number].hex().encode())
        return Rows(self, training), Rows(self, validation), report, digest.hexdigest()

    def discard(self):
        """Close and delete the spool (the snapshot has been written, or the run is over)."""
        self._release()


def collect(examples, folder, stream_chars, wanted_chars=None, stop=None, say=None):
    """Read an iterator of examples. Returns (rows, None) when they are fewer than stream_chars characters (or wanted_chars, the most
    the run needs): rows is the plain list, nothing de-duplicated yet, for the list path. Otherwise (None, corpus): the rows have
    gone to a Corpus on disk, de-duplicated, and reading stopped once its text reached wanted_chars.

    say(read_characters, wanted_or_None) is told as rows are read; stop is a threading.Event checked on every row."""
    stop = stop
    say = say or (lambda done, of: None)
    stream = iter(examples)
    limit = stream_chars if wanted_chars is None else min(stream_chars, wanted_chars)
    buffered, characters = [], 0
    for example in stream:
        if stop is not None and stop.is_set():
            raise InterruptedError("Training preparation cancelled.")
        buffered.append(example)
        characters += characters_of(example)
        if characters >= limit:
            break
    else:
        return buffered, None
    if wanted_chars is not None and characters >= wanted_chars and wanted_chars <= stream_chars:
        return buffered, None  # the budget needs no more than this: small enough for memory, nothing more is read
    corpus = Corpus(folder)
    try:
        expected = getattr(examples, "expected_characters", None)
        of = wanted_chars if expected is None else min(expected, wanted_chars or expected)
        for example in itertools.chain(buffered, stream):
            if stop is not None and stop.is_set():
                raise InterruptedError("Training preparation cancelled.")
            corpus.add(example)
            if corpus.read % 500 == 0 or corpus.characters >= (wanted_chars or float("inf")):
                say(corpus.characters, of)
            if wanted_chars is not None and corpus.characters >= wanted_chars:
                break
        buffered.clear()
        corpus.finish()
        say(corpus.characters, corpus.characters)
    except BaseException:
        corpus.discard()
        raise
    return None, corpus
