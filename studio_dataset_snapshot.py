"""Dataset snapshots: the training data of a run, written once and shared by every attempt that trains on it.

A run folder keeps a `dataset.jsonl` (resume, checkpoint-now and `train.py --resume` rebuild the same examples from it), and
a long run has several attempts: a retry, a restart (`queue_<id>-before-restart`), a resume. Writing 2 GB for each of them
costs minutes before the first step and fills the disk. Here the file is stored once under the hash of its bytes, in
`<runs folder>/.snapshots/<sha256>.jsonl`, and every run folder holds a hard link to it at `dataset.jsonl` (a copy when the
filesystem cannot link). The hash is the identity a peer can be asked for the same data by, the way token shards are
(studio_shards.py). A snapshot is never changed after it is stored, so a link and its file can never disagree.

The lines are `json.dumps(example, ensure_ascii=False)` joined and ended by "\\n", as they always were. Split on "\\n" only:
a document may hold U+2028, which `str.splitlines()` would cut in two (#403).
"""
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

FOLDER = ".snapshots"   # beside the run folders, so a hard link never crosses a filesystem
SUFFIX = ".jsonl"
NAME = "dataset.jsonl"  # what a run folder calls its link
_BUFFER = 1 << 20


def folder_for(output_dir):
    """Where the snapshots of the runs next to `output_dir` live: `.snapshots` in the `runs` folder holding it. None for a run
    folder anywhere else (a test, a script): nothing is shared there, and no folder is made in a place nobody cleans."""
    parent = Path(output_dir).parent
    return parent / FOLDER if parent.name == "runs" else None


def store(examples, directory, name=None):
    """Write `examples` (any iterable, one at a time: nothing is held) into `directory` and return (path, created).

    The bytes go to a temporary file while they are hashed; when a snapshot of the same hash is already stored the temporary
    file is dropped and that one is returned with created False. With `name` the file is called that, not by its hash."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
    digest = hashlib.sha256()
    try:
        with open(temporary, "wb", buffering=_BUFFER) as out:
            for example in examples:
                line = json.dumps(example, ensure_ascii=False).encode("utf-8") + b"\n"
                digest.update(line)
                out.write(line)
        path = directory / (name or digest.hexdigest() + SUFFIX)
        if path.exists() and not name:
            return path, False
        try:
            os.replace(temporary, path)
        except OSError:
            if name or not path.exists():  # a rival stored the same bytes first and holds it open (Windows): that one will do
                raise
            return path, False
        return path, True
    finally:
        temporary.unlink(missing_ok=True)


def link(snapshot, destination):
    """Make `destination` the same file as `snapshot`: a hard link, or a copy when the filesystem cannot link. Returns True for a link."""
    destination = Path(destination)
    try:
        os.link(snapshot, destination)
        return True
    except OSError:
        temporary = destination.with_name(destination.name + f".tmp-{uuid.uuid4().hex}")
        try:
            shutil.copyfile(snapshot, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return False


def adopt(examples, directory, destination):
    """store() the examples, then link the snapshot at `destination`. Returns (snapshot path, created, linked). Without a
    `directory` the examples are written at `destination` itself."""
    destination = Path(destination)
    if directory is None:
        path, created = store(examples, destination.parent, destination.name)
        return path, created, False
    path, created = store(examples, directory)
    return path, created, link(path, destination)


def read(path):
    """The examples of a snapshot, one at a time (a 2 GB file is never held whole)."""
    with open(path, "rb") as source:
        for line in source:  # binary: splits on b"\n" only
            if line.strip():
                yield json.loads(line)


def hash_of(path):
    """sha256 of the file's bytes: its name in the store, and what a peer checks a snapshot it received against."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while block := source.read(_BUFFER):
            digest.update(block)
    return digest.hexdigest()


def listing(directory):
    """[(path, bytes, links)] of the snapshots stored in `directory`. links is 1 when no run folder holds a link to it."""
    out = []
    directory = Path(directory)
    if directory.is_dir():
        for path in sorted(directory.glob("*" + SUFFIX)):
            stat = path.stat()
            out.append((path, stat.st_size, stat.st_nlink))
    return out
