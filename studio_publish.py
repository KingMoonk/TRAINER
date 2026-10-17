"""Share trained versions between machines: a shared folder or an S3-compatible bucket (Cloudflare R2, Backblaze B2).

Upload bandwidth is the bottleneck between nodes, so a published version is not the checkpoint but a weights-only copy of it:

* the optimizer state (`adam_*` arrays) is dropped: a resume on another node restarts the optimizer anyway;
* float32 weights (`weight::`) become float16, half the size. LoRA adapters (`lora::`) are kept as they are: they are a few
  percent of the model, and their values are small corrections a half-precision grid would round hardest.

float16, not bfloat16. float16 has 11 bits of precision against bfloat16's 8, and the only thing bfloat16 buys is a wider
exponent range (float16 holds 6e-5 to 65504 at full precision, then fades into subnormals down to 6e-8). Trained weights sit
nowhere near either end: on a trained tiny model the largest weight was 1.4, none of its 118,656 weights overflowed or flushed
to zero, and the 0.14% under 6e-5 were off by under 3e-8 each (tests/test_publish.py checks the same on every run).
Layer-norm gains are near 1 and biases near 0.1. A weight above 65504 is a diverged model, so `publish` refuses it in plain
words rather than store an infinity. If a future architecture needs the range, store the bfloat16 bits as uint16 here and
flag it in `published.stored_dtype`; nothing else changes.

`pull` casts the weights back to float32 and writes an ordinary checkpoint, because `load_checkpoint` and the rest of the
engine read float32 and a pulled version must be a file any of them accepts (`load_checkpoint(path, weights_only=True)`). The
copy in the store stays half precision; only the local file is widened.

A published artifact is two objects under one key, `<origin>/v<number>-<sha256[:12]>`:

* `<key>.npz`   the weights-only checkpoint. Written with fixed zip timestamps, so the same checkpoint always gives the same
                bytes, the same hash and the same key: publishing twice uploads once, and a pull never races a re-publish.
* `<key>.json`  what a node needs to list it without downloading it: version number, name, spec, scorecard, origin, the
                device that trained it (when the checkpoint records one) and the one that scored it (when known), publish time, size and sha256 of the .npz.

The .json goes up last, so a version is listed only once its weights are complete, and an interrupted upload is simply redone.
`pull` checks the sha256 before it widens anything.

A node can also hand a version straight to a peer, without the store (`shared_copy`): the same two things, the same bytes, the
same key, cached under `data/shared/` so a peer's metadata call and its weights call build the copy once.

A bucket store never takes an upload that would pass 9.5 GB, half a GB under the provider's 10 GB free tier: see `S3Store._admit`
and `DEFAULT_CAP_BYTES`.
`python -m studio_publish --check` reads the TRAINER_STORE_* settings and says what the bucket holds, and which job checkpoints no job
of this queue is waiting on (docs/cluster.md).

Everything here is slow (disk and network): call it from a worker, never the Tk thread.
"""
from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import logging
import math
import os
import re
import socket
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from studio_core import replace_file
from studio_ui_guard import slow
from studio_versions import NoCheckpoint

WEIGHT = "weight::"
OPTIMIZER = "adam_"  # every optimizer state array is adam_<state>::<parameter>, whatever the optimizer (see studio_model.load_checkpoint)
HALF_MAX = float(np.finfo(np.float16).max)
FORMAT = 1
_EPOCH = (1980, 1, 1, 0, 0, 0)  # the earliest time a zip entry can carry; a constant keeps the file's bytes a function of its arrays
_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$")
NO_STORE = ("No version store is set up, so versions cannot be shared between machines. Set TRAINER_STORE_FOLDER to a shared folder, "
            "or the TRAINER_STORE_* bucket variables (docs/cluster.md), or \"studioConfig\": {\"store\": {\"folder\": ...}} in the workspace.")
ENV = {"endpoint": "TRAINER_STORE_ENDPOINT", "bucket": "TRAINER_STORE_BUCKET", "prefix": "TRAINER_STORE_PREFIX",
       "key_id": "TRAINER_STORE_KEY_ID", "secret": "TRAINER_STORE_SECRET", "region": "TRAINER_STORE_REGION"}
REQUIRED = ("endpoint", "bucket", "key_id", "secret")
FREE_TIER_BYTES = 10 * 10**9  # Cloudflare R2's free storage. The most a bucket store ever lets the whole bucket hold.
FREE_TIER_GB = 10
DEFAULT_CAP_GB = 9.5  # what the bucket may hold unless TRAINER_STORE_LIMIT_GB says otherwise: headroom for what the check cannot see
DEFAULT_CAP_BYTES = int(DEFAULT_CAP_GB * 10**9)
LIMIT_ENV = "TRAINER_STORE_LIMIT_GB"  # sets the limit anywhere up to FREE_TIER_GB; a value above that is an error, never a way to raise it


@runtime_checkable
class RemoteStore(Protocol):
    """Where nodes exchange versions. A key is what `publish` returns and `list` reports; everything else is derived from it."""

    def publish(self, version: dict, checkpoint_path) -> str:
        """Upload a weights-only half-precision copy of `checkpoint_path` for `version` (a VersionStore row); returns its key."""

    def pull(self, key: str, dest_dir) -> Path:
        """Download `key` into `dest_dir` as a float32 checkpoint `load_checkpoint` accepts; returns its path."""

    def list(self) -> list[dict]:
        """The metadata of every complete published version, oldest first, each with its "key"."""

    def exists(self, key: str) -> bool:
        """Whether `key` is published and complete."""


# ---- the artifact -------------------------------------------------------------------------------------------------

def _origin_name(origin):
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", (origin or socket.gethostname()).lower()).strip(".-")
    return name or "node"


def own_origin(store=None):
    """The name this machine's versions are published under in `store`: the store's own origin, else the hostname."""
    return _origin_name(getattr(store, "origin", None))


class StoreError(RuntimeError):
    """A failure of a version store, already written for people: what went wrong and what to do. `plain` tells the API's jobs to show
    the message as it is, without the exception's name in front."""
    plain = True


class StoreLimitError(StoreError):
    """An upload the bucket store refused so the bucket stays inside its free tier: it would pass the limit, the bucket's size could not
    be read, or the limit setting is not valid. Nothing was uploaded."""


class StoreUsageError(StoreLimitError):
    """The bucket's size could not be read, or what it said did not add up, so the upload could not be checked and was refused. Unlike
    its parent this says nothing about the limit being reached: it is worth trying again."""


def checkpoint_version(job_id, step):
    """The version row a runner publishes a job's checkpoint under: the name and change are how a checkpoint is told from a version, and
    which job it belongs to (`checkpoint_job`)."""
    return {"number": 0, "name": f"job:{job_id}", "change": f"checkpoint at step {step}"}


def checkpoint_job(meta):
    """The id of the job whose checkpoint a published version is (see `checkpoint_version`); None for anything else."""
    name, change = (meta or {}).get("name"), (meta or {}).get("change")
    if isinstance(name, str) and name.startswith("job:") and len(name) > 4 and isinstance(change, str) and change.startswith("checkpoint at"):
        return name[4:]
    return None


def size_text(count):
    """`count` bytes as people read them: decimal units, the ones a provider's free tier is counted in."""
    for unit, size in (("GB", 10**9), ("MB", 10**6), ("KB", 10**3)):
        if count >= size:
            return f"{count / size:.2f} {unit}"
    return f"{count} B"


def store_limit(setting=None):
    """The most bytes the bucket may hold: `DEFAULT_CAP_BYTES` (9.5 GB), or `setting` (TRAINER_STORE_LIMIT_GB, in GB) when one is given.
    The setting may be anything up to the free tier's 10 GB, so it can lower the cap or use the last half GB on purpose. A setting above
    the free tier, or one that is not a positive number, is a StoreLimitError: a mistyped limit must not raise the limit."""
    text = str(setting if setting is not None else "").strip()
    if not text:
        return DEFAULT_CAP_BYTES
    try:
        gigabytes = float(text)
    except ValueError:
        gigabytes = math.nan
    if not math.isfinite(gigabytes) or gigabytes <= 0:
        raise StoreLimitError(f"{LIMIT_ENV} is {text!r}, which is not a number of gigabytes above 0. Nothing was uploaded. "
                              f"Set it to a number up to {FREE_TIER_GB}, or remove it.")
    if gigabytes > FREE_TIER_GB:
        raise StoreLimitError(f"{LIMIT_ENV} is {text}, above the {FREE_TIER_GB} GB free tier. It can only lower the limit, so nothing was "
                              f"uploaded. Set it to {FREE_TIER_GB} or less, or remove it.")
    return int(gigabytes * 10**9)


def limit_source(setting=None):
    """Where the limit comes from, for a sentence: the default cap with its headroom, or the setting."""
    if str(setting if setting is not None else "").strip():
        return LIMIT_ENV
    return f"the default cap, {FREE_TIER_GB - DEFAULT_CAP_GB:g} GB under the R2 free tier"


_DENIED = ("InvalidAccessKeyId", "SignatureDoesNotMatch", "AccessDenied", "403", "InvalidToken", "ExpiredToken", "AllAccessDisabled")
_log = logging.getLogger(__name__)


def store_error(error, store=None):
    """`error`, raised while using `store`, as a StoreError whose sentence says what to do about it. The log keeps the original."""
    if isinstance(error, StoreError):
        return error
    folder = isinstance(store, FolderStore)
    code = str(getattr(error, "response", {}).get("Error", {}).get("Code", ""))
    if isinstance(error, (NoCheckpoint, ValueError)) or (isinstance(error, FileNotFoundError) and "checkpoint" in str(error)):
        text = str(error)  # these are written for people already (no checkpoint, a weight half precision cannot hold, a failed hash)
    elif isinstance(error, KeyError) and error.args:
        name = str(error.args[0])
        text = name if name.startswith("No published version") else (
            f"{name} is missing from the version store. Publish the version again from the machine that has it, then try again.")
    elif isinstance(error, FileNotFoundError) and folder:
        text = str(error)
    elif isinstance(error, OSError) and getattr(error, "errno", None) == errno.ENOSPC:
        text = "The disk is full. Free space in Resources › Storage, then try again."
    elif code in _DENIED:
        text = ("The bucket refused the key. Check TRAINER_STORE_KEY_ID and TRAINER_STORE_SECRET, and that the key can read and write "
                "this bucket, then try again.")
    elif code == "NoSuchBucket":
        text = f"The bucket {getattr(store, 'bucket', '')} was not found. Check TRAINER_STORE_BUCKET and TRAINER_STORE_ENDPOINT, then try again."
    elif folder and isinstance(error, OSError):
        text = (f"The shared folder {store.root} could not be read or written ({error.strerror or error}). "
                "Check that it is connected and that you can write to it, then try again.")
    elif isinstance(error, (OSError, TimeoutError)) or type(error).__name__ in (
            "EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError", "ConnectionClosedError", "SSLError"):
        where = f" at {getattr(store, 'bucket', None)}" if getattr(store, "bucket", None) else ""
        text = f"Could not reach the version store{where}. Check the connection and TRAINER_STORE_ENDPOINT, then try again."
    else:
        _log.error("Unexpected %s from the version store", type(error).__name__, exc_info=error)
        text = (f"The version store failed ({code or type(error).__name__}). Check its settings (docs/cluster.md) and that the bucket "
                "or folder is reachable, then try again. The log has the details.")
    return StoreError(text)


def _check_key(key):
    if not isinstance(key, str) or not _KEY.match(key):
        raise ValueError(f"{key!r} is not a published version's key.")
    return key


def _half(name, array):
    if array.dtype.kind != "f" or array.dtype.itemsize <= 2:
        return array
    peak = float(np.max(np.abs(array), initial=0.))
    if peak > HALF_MAX:
        raise ValueError(f"{name.removeprefix(WEIGHT)} reaches {peak:.3g}, more than half precision can hold ({HALF_MAX:g}), "
                         "so this version cannot be published.")
    return array.astype(np.float16)


def _write_npz(path, items):
    """An .npz (what np.savez_compressed writes) from (name, array) pairs, with the same entry times every time."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name, array in items:
            entry = zipfile.ZipInfo(name + ".npy", date_time=_EPOCH)
            entry.compress_type = zipfile.ZIP_DEFLATED
            with archive.open(entry, "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asanyarray(array), allow_pickle=False)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_artifact(version, checkpoint_path, scratch, origin=None):
    """Write the weights-only half-precision copy of `checkpoint_path` into `scratch`; returns (path, metadata)."""
    if not checkpoint_path:
        raise NoCheckpoint(version.get("number"))
    source = Path(checkpoint_path)
    if not source.is_file():
        raise FileNotFoundError(f"Version {version.get('number')}'s checkpoint is not at {source}.")
    origin = _origin_name(origin)
    artifact = Path(scratch) / "artifact.npz"
    with np.load(source, allow_pickle=False) as archive:
        inner = json.loads(str(archive["metadata"].item()))
        if inner.get("architecture") != "groundwork_numpy_decoder" or inner.get("format_version") != 1:
            raise ValueError("Choose a Groundwork NumPy checkpoint (.npz).")
        inner["published"] = {"weights_only": True, "stored_dtype": "float16"}

        def items():
            yield "metadata", np.array(json.dumps(inner))
            for name in archive.files:
                if name == "metadata" or name.startswith(OPTIMIZER):
                    continue
                array = archive[name]
                yield name, _half(name, array) if name.startswith(WEIGHT) else array

        _write_npz(artifact, items())
    digest = sha256_of(artifact)
    key = f"{origin}/v{version['number']}-{digest[:12]}"
    meta = {"format": FORMAT, "key": key, "number": version["number"], "name": version.get("name"), "change": version.get("change"),
            "parent": version.get("parent"), "spec": version.get("spec"), "scorecard": version.get("scorecard") or {},
            "origin": origin, "device": inner.get("device"), "train_device": inner.get("device"),
            "score_device": version.get("score_device"), "step": inner.get("step"),
            "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "weights_dtype": "float16",
            "size": artifact.stat().st_size, "sha256": digest}
    return artifact, meta


def restore_checkpoint(artifact, target):
    """A published artifact back into an ordinary checkpoint: weights widened to float32, nothing else changed."""
    target = Path(target)
    temporary = target.with_name(target.name + ".tmp")
    def widened(archive, name):
        array = archive[name]
        return array.astype(np.float32) if name.startswith(WEIGHT) and array.dtype == np.float16 else array

    with np.load(artifact, allow_pickle=False) as archive:
        _write_npz(temporary, ((name, widened(archive, name)) for name in archive.files))
    replace_file(temporary, target)
    return target


def finish_pull(artifact, meta, dest_dir):
    """Check the downloaded `artifact` against `meta`'s sha256, then widen it into `dest_dir` as an ordinary checkpoint with `meta`
    beside it; returns the checkpoint's path. Whatever the artifact came from (a store, a peer), a bad download stops here."""
    if sha256_of(artifact) != meta["sha256"]:
        raise ValueError(f"The download of {meta['key']} does not match what was published. Try the pull again.")
    target = restore_checkpoint(artifact, Path(dest_dir) / (meta["key"].replace("/", "-") + ".npz"))
    target.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return target


_ROW = ("number", "name", "change", "parent", "spec", "scorecard", "score_device")  # the parts of a version row that end up in the metadata
_shared = threading.Lock()


@slow("copy a version for a peer")
def shared_copy(version, checkpoint_path, folder, origin=None):
    """The weights-only copy of a version a peer fetches, as (path, metadata): what `publish` would upload, kept as `v<n>.npz` and
    `v<n>.json` in `folder`. It is built once and reused until the checkpoint or the version's row changes."""
    if not checkpoint_path:
        raise NoCheckpoint(version.get("number"))
    source, folder = Path(checkpoint_path), Path(folder)
    if not source.is_file():
        raise FileNotFoundError(f"Version {version.get('number')}'s checkpoint is not at {source}.")
    stat = source.stat()
    wanted = hashlib.sha256(json.dumps([str(source), stat.st_size, stat.st_mtime_ns, _origin_name(origin), [version.get(k) for k in _ROW]],
                                       sort_keys=True, default=str).encode()).hexdigest()
    artifact, index = folder / f"v{version['number']}.npz", folder / f"v{version['number']}.json"
    with _shared:
        try:
            held = json.loads(index.read_text(encoding="utf-8"))
            if held["stamp"] == wanted and artifact.stat().st_size == held["meta"]["size"]:
                return artifact, held["meta"]
        except (OSError, ValueError, KeyError):
            pass  # nothing cached, or not readable: build it
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder, prefix=".build-") as scratch:
            built, meta = build_artifact(version, source, scratch, origin)
            replace_file(built, artifact)
            marked = Path(scratch) / "index.json"
            marked.write_text(json.dumps({"stamp": wanted, "meta": meta}, indent=1), encoding="utf-8")
            replace_file(marked, index)  # last: the artifact it describes is complete
        return artifact, meta


# ---- the stores ---------------------------------------------------------------------------------------------------

class _Store:
    """publish / pull / list / exists over six primitives a backend supplies. Names are "<key>.npz" and "<key>.json".
    A missing object is a KeyError from _get_*; _has says False instead."""
    origin = None

    @slow("publish a version")
    def publish(self, version, checkpoint_path):
        return self.publish_reporting(version, checkpoint_path)[0]

    @slow("publish a version")
    def publish_reporting(self, version, checkpoint_path):
        """`publish`, and whether this call put the objects up: (key, created). False when the version was already there, in which case
        the objects are not this caller's to delete."""
        with tempfile.TemporaryDirectory(prefix="trainer-publish-") as scratch:
            artifact, meta = build_artifact(version, checkpoint_path, scratch, self.origin)
            key = meta["key"]
            described = json.dumps(meta, indent=1).encode()
            with self._admit(artifact.stat().st_size + len(described), key) as room:  # once for both objects, before the first byte
                if room:
                    self._put_file(key + ".npz", artifact)
                    self._put_bytes(key + ".json", described)  # last: it is what makes the version visible
            return key, bool(room)

    @contextlib.contextmanager
    def _admit(self, size, key=None):
        """A context to hold the upload of `size` bytes in, under `key` when it has one. It yields True when the upload should go ahead and
        False when `key` is published already (nothing to upload). A store with a limit raises StoreLimitError here, before anything is put;
        the look-up of `key` happens inside whatever lock the check holds, so two uploads of one key cannot both count."""
        yield key is None or not self.exists(key)

    @slow("delete a version")
    def delete(self, key):
        """Remove a published version, its listing first so nobody is offered weights that are going. A missing object is not an error."""
        key = _check_key(key)
        self._delete(key + ".json")
        self._delete(key + ".npz")

    @slow("pull a version")
    def pull(self, key, dest_dir):
        key = _check_key(key)
        try:
            meta = json.loads(self._get_bytes(key + ".json"))
        except KeyError:
            raise KeyError(f"No published version {key}.") from None
        destination = Path(dest_dir)
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination, prefix=".pull-") as scratch:
            artifact = Path(scratch) / "artifact.npz"
            self._get_file(key + ".npz", artifact)
            return finish_pull(artifact, {**meta, "key": key}, destination)

    @slow("list published versions")
    def list(self):
        entries = []
        for name in self._json_names():
            key = name[:-len(".json")]
            try:
                _check_key(key)
                meta = json.loads(self._get_bytes(name))
            except (KeyError, ValueError):
                continue  # not ours, or not readable: not a version to offer
            entries.append({**meta, "key": key})
        return sorted(entries, key=lambda entry: (entry.get("published_at") or "", entry["key"]))

    @slow("look up a published version")
    def exists(self, key):
        return self._has(_check_key(key) + ".json")

    @slow("look up a published version")
    def metadata(self, key):
        """The metadata of the published `key`: None when it is not there, {} when it is there and cannot be read."""
        try:
            meta = json.loads(self._get_bytes(_check_key(key) + ".json"))
        except KeyError:
            return None
        except ValueError:
            return {}
        return meta if isinstance(meta, dict) else {}


class FolderStore(_Store):
    """A shared folder: a LAN share, or a synced drive. Objects are plain files at <path>/<key>.npz and .json."""

    def __init__(self, path, origin=None):
        self.root, self.origin = Path(path), origin

    def _path(self, name):
        return self.root / name

    def _prepare(self, name):
        if not self.root.is_dir():
            raise FileNotFoundError(f"The shared folder {self.root} is not there. Check that the drive or share is connected.")
        target = self._path(name)
        target.parent.mkdir(exist_ok=True)
        return target

    def _put_file(self, name, path):
        target = self._prepare(name)
        temporary = target.with_name(target.name + ".tmp")
        with open(path, "rb") as source, open(temporary, "wb") as stream:
            while block := source.read(1 << 20):
                stream.write(block)
        replace_file(temporary, target)

    def _put_bytes(self, name, data):
        target = self._prepare(name)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_bytes(data)
        replace_file(temporary, target)

    def _get_file(self, name, path):
        try:
            data = self._path(name).open("rb")
        except FileNotFoundError:
            raise KeyError(name) from None
        with data, open(path, "wb") as stream:
            while block := data.read(1 << 20):
                stream.write(block)

    def _get_bytes(self, name):
        try:
            return self._path(name).read_bytes()
        except FileNotFoundError:
            raise KeyError(name) from None

    def _has(self, name):
        return self._path(name).is_file()

    def _delete(self, name):
        self._path(name).unlink(missing_ok=True)

    def _json_names(self):
        return sorted(path.relative_to(self.root).as_posix() for path in self.root.glob("*/*.json")) if self.root.is_dir() else []


def _missing(error):
    code = str(getattr(error, "response", {}).get("Error", {}).get("Code", ""))
    return code in ("404", "NoSuchKey", "NotFound")


def _boto_client(config):
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise RuntimeError("The bucket store needs boto3. Install it with: pip install -r requirements.txt") from None
    return boto3.client(
        "s3", endpoint_url=config["endpoint"], aws_access_key_id=config["key_id"], aws_secret_access_key=config["secret"],
        region_name=config.get("region") or "auto",  # R2 wants "auto"; B2 wants the region in its endpoint, e.g. us-west-004
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}, s3={"addressing_style": "path"},
                      # boto3 1.36+ adds CRC checksums to every request by default; B2 and older R2 gateways reject them
                      request_checksum_calculation="when_required", response_checksum_validation="when_required"))


_admission = threading.Lock()  # one check-then-upload at a time in this process, so two uploads cannot each see room the other takes
_UNSURE = "Nothing was uploaded: the bucket's usage could not be added up, because "


def _size(entry, what, name):
    """The size an S3 listing gives for `entry`; a listing without one is not trusted (StoreUsageError), never counted as empty."""
    size = entry.get("Size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise StoreUsageError(_UNSURE + f"the listing gave no usable size for {what} ({name}).")
    return size


class S3Store(_Store):
    """An S3-compatible bucket through boto3. `config` is {"endpoint", "bucket", "key_id", "secret"} plus optional "prefix"
    (a folder in the bucket) and "region"; `from_env()` reads the same from TRAINER_STORE_* variables. `client` is for tests.

    The bucket is never filled past `DEFAULT_CAP_BYTES` (9.5 GB, under R2's 10 GB free storage): before an upload `_admit` adds up every
    object in the bucket, not just the prefix, and the parts of uploads started and not finished, and refuses when the upload would
    pass the limit or when the sum cannot be read or does not add up. `limit` is the TRAINER_STORE_LIMIT_GB text: any limit up to 10 GB.
    Another machine uploading at the same moment is not seen: the check is against what the bucket holds when it is made."""

    def __init__(self, config, client=None, origin=None, limit=None):
        missing = [name for name in REQUIRED if not str(config.get(name) or "").strip()]
        if missing:
            raise ValueError("The bucket store is missing " + ", ".join(f"{ENV[name]}" for name in missing) + ".")
        self.bucket = config["bucket"].strip()
        prefix = (config.get("prefix") or "").strip().strip("/")
        self.prefix = prefix + "/" if prefix else ""
        self.origin, self.limit = origin, limit
        self.client = client or _boto_client(config)

    @classmethod
    def from_env(cls, environ=None):
        """The store the TRAINER_STORE_* variables describe; None when none of them is set (a cluster of one)."""
        environ = os.environ if environ is None else environ
        config = {name: environ.get(variable, "").strip() for name, variable in ENV.items()}
        return cls(config, limit=environ.get(LIMIT_ENV)) if any(config.values()) else None

    def __repr__(self):
        return f"S3Store(bucket={self.bucket!r}, prefix={self.prefix!r})"

    def _put_file(self, name, path):
        self.client.upload_file(str(path), self.bucket, self.prefix + name)  # multipart and retried per part: Starlink uploads drop

    def _put_bytes(self, name, data):
        self.client.put_object(Bucket=self.bucket, Key=self.prefix + name, Body=data, ContentType="application/json")

    def _get_file(self, name, path):
        try:
            self.client.download_file(self.bucket, self.prefix + name, str(path))
        except Exception as error:
            if _missing(error):
                raise KeyError(name) from error
            raise

    def _get_bytes(self, name):
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self.prefix + name)["Body"].read()
        except Exception as error:
            if _missing(error):
                raise KeyError(name) from error
            raise

    def _has(self, name):
        try:
            self.client.head_object(Bucket=self.bucket, Key=self.prefix + name)
        except Exception as error:
            if _missing(error):
                return False
            raise
        return True

    def _delete(self, name):
        self.client.delete_object(Bucket=self.bucket, Key=self.prefix + name)  # S3 says nothing when the key is not there

    def usage(self):
        """(objects, bytes) the whole bucket holds, whatever the prefix: the free tier is counted over everything in it. The bytes include
        the parts of uploads that were started and never finished (they are stored, and billed, until aborted). Any page or answer that
        does not add up is a StoreUsageError: a sum that might be short is not a sum."""
        objects, stored, _, pending = self.usage_detail()
        return objects, stored + pending

    def usage_detail(self):
        """(objects, bytes in objects, unfinished uploads, bytes in their parts) of the whole bucket."""
        uploads, pending = self._unfinished()  # first: an upload that finishes meanwhile is then counted as an object
        objects, stored = self._objects()
        return objects, stored, uploads, pending

    def _objects(self):
        count, total, token, seen = 0, 0, None, set()
        while True:
            page = self.client.list_objects_v2(Bucket=self.bucket, **({"ContinuationToken": token} if token else {}))
            for item in page.get("Contents", ()):
                count, total = count + 1, total + _size(item, "an object", item.get("Key"))
            following = page.get("NextContinuationToken")
            if bool(page.get("IsTruncated")) != bool(following):
                raise StoreUsageError(_UNSURE + "its list of objects was cut off without saying where it goes on, or went on after it said "
                                      "it had ended.")
            if not following:
                return count, total
            if following in seen:
                raise StoreUsageError(_UNSURE + "its list of objects went round in a circle.")
            seen.add(following)
            token = following

    def _unfinished(self):
        """(uploads, bytes) in the multipart uploads started in the bucket and not completed or aborted. R2 keeps their parts, and bills
        them, until the upload is aborted (docs/cluster.md has the lifecycle rule that aborts them after a day)."""
        uploads, total, marker, seen = 0, 0, {}, set()
        while True:
            page = self.client.list_multipart_uploads(Bucket=self.bucket, **marker)
            for upload in page.get("Uploads", ()):
                uploads += 1
                total += self._parts(upload.get("Key"), upload.get("UploadId"))
            following = {"KeyMarker": page.get("NextKeyMarker"), "UploadIdMarker": page.get("NextUploadIdMarker")}
            if not page.get("IsTruncated"):  # (S3 may still send markers on the last page; they are not followed)
                return uploads, total
            if not (following["KeyMarker"] or following["UploadIdMarker"]):
                raise StoreUsageError(_UNSURE + "its list of unfinished uploads was cut off without saying where it goes on.")
            point = (following["KeyMarker"], following["UploadIdMarker"])
            if point in seen:
                raise StoreUsageError(_UNSURE + "its list of unfinished uploads went round in a circle.")
            seen.add(point)
            marker = {name: value for name, value in following.items() if value}

    def _parts(self, key, upload_id):
        total, marker, seen = 0, None, set()
        if not key or not upload_id:
            raise StoreUsageError(_UNSURE + "an unfinished upload came without its name.")
        while True:
            try:
                page = self.client.list_parts(Bucket=self.bucket, Key=key, UploadId=upload_id, **({"PartNumberMarker": marker} if marker else {}))
            except Exception as error:  # noqa: BLE001
                if str(getattr(error, "response", {}).get("Error", {}).get("Code", "")) == "NoSuchUpload":
                    return total  # it finished or was aborted since it was listed: if it finished, the objects list has it
                raise
            for part in page.get("Parts", ()):
                total += _size(part, "an unfinished upload's part", key)
            if not page.get("IsTruncated"):  # (S3 sends the last part's number even then; it is not followed)
                return total
            following = page.get("NextPartNumberMarker")
            if not following:
                raise StoreUsageError(_UNSURE + "the parts of an unfinished upload were cut off without saying where they go on.")
            if following in seen:
                raise StoreUsageError(_UNSURE + "the parts of an unfinished upload went round in a circle.")
            seen.add(following)
            marker = following

    @contextlib.contextmanager
    def _admit(self, size, key=None):
        limit = store_limit(self.limit)
        with _admission:
            if key is not None and self.exists(key):  # inside the lock: a second publish of this key finds it there, and counts nothing
                yield False
                return
            try:
                used = self.usage()[1]
            except StoreUsageError as error:
                _log.warning("Could not add up bucket %s: %s", self.bucket, error)
                raise
            except Exception as error:  # noqa: BLE001 - an upload that cannot be checked is not made
                _log.warning("Could not read the size of bucket %s: %s", self.bucket, type(error).__name__)
                raise StoreUsageError("Nothing was uploaded: the bucket's usage could not be read, so the upload could not be checked "
                                      f"against the {size_text(limit)} limit. Try again; if it keeps happening, run python -m studio_publish "
                                      f"--check. {store_error(error, self)}") from error
            if used + size > limit:
                raise StoreLimitError(
                    f"Not uploaded: this upload needs {size_text(size)} and the bucket already holds {size_text(used)}, which would make "
                    f"{size_text(used + size)}, over the {size_text(limit)} limit ({limit_source(self.limit)}). "
                    "Delete versions you no longer need from the bucket (the Cloudflare dashboard, R2, the bucket's Objects), then publish again.")
            yield True

    def _json_names(self):
        names, token = [], None
        while True:
            page = self.client.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix, **({"ContinuationToken": token} if token else {}))
            names += [item["Key"][len(self.prefix):] for item in page.get("Contents", ()) if item["Key"].endswith(".json")]
            token = page.get("NextContinuationToken") if page.get("IsTruncated") else None
            if not token:
                return sorted(names)


def configured_store(studio_config=None, environ=None):
    """The version store a workspace uses: the shared folder `studioConfig.store.folder` names, else the one the TRAINER_STORE_* variables
    describe (a shared folder, else a bucket), else None (a cluster of one). The routes and the queue's dispatch both use this."""
    environ = os.environ if environ is None else environ
    store = (studio_config or {}).get("store")
    folder = str(store.get("folder") or "").strip() if isinstance(store, dict) else ""
    folder = folder or (environ.get("TRAINER_STORE_FOLDER") or "").strip()
    if folder:
        return FolderStore(folder)
    return S3Store.from_env(environ)


# ---- the connection check -----------------------------------------------------------------------------------------

NOT_CONFIGURED = ("Not configured: no bucket is set, so there is nothing to check. Set TRAINER_STORE_ENDPOINT, TRAINER_STORE_BUCKET, "
                  "TRAINER_STORE_KEY_ID and TRAINER_STORE_SECRET (a template is in docs/store.env.example, and docs/cluster.md has the steps), "
                  "then run this again.")


def _round_trip(store):
    """Write a few bytes into the bucket, read them back and delete them: what a publish needs, with nothing left behind."""
    name, data = f".check-{uuid.uuid4().hex[:12]}", b"trainer store check"
    with store._admit(len(data)):  # the same limit as any upload
        store.client.put_object(Bucket=store.bucket, Key=store.prefix + name, Body=data)
    try:
        if store._get_bytes(name) != data:
            raise StoreError("The bucket gave back something other than what was written. Check the endpoint and the bucket, then try again.")
    finally:
        store.client.delete_object(Bucket=store.bucket, Key=store.prefix + name)


QUEUE_FILE = Path("data") / "queue.json"  # where a workspace keeps its queue, relative to the workspace folder
_UNFINISHED = ("waiting", "held", "running")


def queue_jobs(path):
    """(statuses, results, guests) for the queue file at `path`, or None when it cannot be read: {job id: status}, the set of result keys
    of its jobs, and the set of ids of jobs this machine runs (or ran) for another node. Such a job is listed under the owner's id (its own
    id is that and "-e<epoch>"), the one its checkpoints are published under."""
    try:
        jobs = json.loads(Path(path).read_text(encoding="utf-8"))["jobs"]
        known, results, guests = {}, set(), set()
        for job in jobs:
            job_id = str(job["id"])
            if job.get("owner"):
                job_id = job_id.rsplit("-e", 1)[0]
                guests.add(job_id)
            known[job_id] = job["status"]
            result = ((job.get("params") or {}).get("remote_result") or {}).get("key")
            if isinstance(result, str):
                results.add(result)
        return known, results, guests
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def report_checkpoints(store, queue, out):
    """Say which of the job checkpoints in `store` no unfinished job of the queue at `queue` can resume from: the job is not in the queue
    at all, or it ended. The key a done job ends with (its result can be the bytes of a checkpoint) is not one. Only lists them; deleting
    is by hand. A job this machine ran for another node is marked: its owner may still resume it. Checkpoints of a job another machine owns
    and this one never ran show here too: check there."""
    entries = [(entry, checkpoint_job(entry)) for entry in store.list()]
    entries = [(entry, job_id) for entry, job_id in entries if job_id]
    noun = f"{len(entries)} job checkpoint object{'s' if len(entries) != 1 else ''}"
    queue_state = queue_jobs(queue)
    if queue_state is None:
        out(f"Checkpoints: {noun}. The queue at {queue} could not be read, so which of them are left over is not known. "
            "Run this from the TRAINER folder, or pass --queue.")
        return
    known, results, guests = queue_state
    free = [(entry, job_id) for entry, job_id in entries if known.get(job_id) not in _UNFINISHED and entry["key"] not in results]
    left = [(entry, job_id) for entry, job_id in free if job_id not in guests]
    foreign = [(entry, job_id) for entry, job_id in free if job_id in guests]

    def line(entry, job_id, why):
        return (f"  {store.prefix}{entry['key']}  {why}, job {job_id}, {entry.get('change') or ''}, {size_text(int(entry.get('size') or 0))}, "
                f"published {entry.get('published_at') or 'at an unknown time'}")

    if not left:
        out(f"Checkpoints: {noun}, none left over.")
    else:
        out(f"Checkpoints left over: {len(left)} of {len(entries)}, {size_text(sum(int(e.get('size') or 0) for e, _ in left))}. No unfinished job "
            f"in {queue} can resume from them. Delete each one's .npz and .json from the bucket by hand if no other machine's job needs it:")
        for entry, job_id in left:
            out(line(entry, job_id, f"job ended here ({known[job_id]})" if job_id in known else "job not in this queue"))
    if foreign:
        out(f"Owned by another machine, do not delete: {len(foreign)}. This machine only ran these jobs; their owner may still resume them:")
        for entry, job_id in foreign:
            out(line(entry, job_id, "owned by another machine, do not delete"))


def check(environ=None, write=False, out=print, queue=QUEUE_FILE):
    """Report on the version store the TRAINER_STORE_* variables describe: what the bucket holds against its limit, the job checkpoints no
    unfinished job of the queue file `queue` can use, and with `write` a write, read and delete round trip. Prints plain lines (never the
    key or the secret); returns the exit code, 0 when everything works."""
    environ = os.environ if environ is None else environ
    folder = (environ.get("TRAINER_STORE_FOLDER") or "").strip()
    if folder:
        out(f"Shared folder: {folder} ({'there' if Path(folder).is_dir() else 'NOT THERE, check that the drive or share is connected'}). "
            "While it is set the app uses it instead of a bucket.")
    try:
        store = S3Store.from_env(environ)
    except (ValueError, RuntimeError) as error:
        out(f"Not ready: {error}")
        return 1
    if store is None:
        out("No bucket is set. Nothing else to check." if folder else NOT_CONFIGURED)
        return 0 if folder and Path(folder).is_dir() else 1
    out(f"Bucket: {store.bucket}, prefix {store.prefix or '(none)'}")
    try:
        limit = store_limit(store.limit)
        objects, stored, uploads, pending = store.usage_detail()
    except StoreError as error:
        out(f"Failed: {error}")
        return 1
    except Exception as error:  # noqa: BLE001 - whatever the provider throws ends as a sentence, the log has the rest
        out(f"Failed: {store_error(error, store)}")
        return 1
    used = stored + pending
    out(f"Read: ok. The bucket holds {objects} object{'s' if objects != 1 else ''}, {size_text(stored)}.")
    out(f"Unfinished uploads: {uploads}, holding {size_text(pending)}" + (" (counted; abort them, see docs/cluster.md)." if uploads else "."))
    out(f"Limit: {size_text(limit)} ({limit_source(store.limit)})")
    out(f"Room left: {size_text(max(limit - used, 0))}" + ("  OVER THE LIMIT: uploads are refused until versions are deleted." if used > limit else ""))
    try:
        report_checkpoints(store, queue, out)
    except Exception as error:  # noqa: BLE001 - the listing is extra: the rest of the check still runs
        out(f"Checkpoints: could not be listed. {store_error(error, store)}")
    if write:
        try:
            _round_trip(store)
        except StoreError as error:
            out(f"Write: failed. {error}")
            return 1
        except Exception as error:  # noqa: BLE001
            out(f"Write: failed. {store_error(error, store)}")
            return 1
        out("Write: ok. A test object was written, read back and deleted.")
    else:
        out("Write: not tried. Add --write to write, read back and delete a test object.")
    return 0


def main(argv=None, environ=None, out=print):
    parser = argparse.ArgumentParser(prog="python -m studio_publish", description="Check the version store (TRAINER_STORE_* variables).")
    parser.add_argument("--check", action="store_true", help="read the settings, list the bucket, print objects, bytes used, limit and room left, and the job checkpoints left over")
    parser.add_argument("--write", action="store_true", help="with --check: also write, read back and delete a small test object")
    parser.add_argument("--queue", default=str(QUEUE_FILE), help="with --check: this machine's queue file, to tell which job checkpoints "
                        f"are left over (default {QUEUE_FILE}, run from the TRAINER folder)")
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("say what to do: --check")
    return check(environ, write=args.write, out=out, queue=args.queue)


if __name__ == "__main__":
    sys.exit(main())
