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
                device that trained it (when the checkpoint records one), publish time, size and sha256 of the .npz.

The .json goes up last, so a version is listed only once its weights are complete, and an interrupted upload is simply redone.
`pull` checks the sha256 before it widens anything.

A node can also hand a version straight to a peer, without the store (`shared_copy`): the same two things, the same bytes, the
same key, cached under `data/shared/` so a peer's metadata call and its weights call build the copy once.

Everything here is slow (disk and network): call it from a worker, never the Tk thread.
"""
from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import re
import socket
import tempfile
import threading
import time
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
            "origin": origin, "device": inner.get("device"), "step": inner.get("step"),
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


_ROW = ("number", "name", "change", "parent", "spec", "scorecard")  # the parts of a version row that end up in the metadata
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
        with tempfile.TemporaryDirectory(prefix="trainer-publish-") as scratch:
            artifact, meta = build_artifact(version, checkpoint_path, scratch, self.origin)
            key = meta["key"]
            if not self.exists(key):
                self._put_file(key + ".npz", artifact)
                self._put_bytes(key + ".json", json.dumps(meta, indent=1).encode())  # last: it is what makes the version visible
            return key

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


class S3Store(_Store):
    """An S3-compatible bucket through boto3. `config` is {"endpoint", "bucket", "key_id", "secret"} plus optional "prefix"
    (a folder in the bucket) and "region"; `from_env()` reads the same from TRAINER_STORE_* variables. `client` is for tests."""

    def __init__(self, config, client=None, origin=None):
        missing = [name for name in REQUIRED if not str(config.get(name) or "").strip()]
        if missing:
            raise ValueError("The bucket store is missing " + ", ".join(f"{ENV[name]}" for name in missing) + ".")
        self.bucket = config["bucket"].strip()
        prefix = (config.get("prefix") or "").strip().strip("/")
        self.prefix = prefix + "/" if prefix else ""
        self.origin = origin
        self.client = client or _boto_client(config)

    @classmethod
    def from_env(cls, environ=None):
        """The store the TRAINER_STORE_* variables describe; None when none of them is set (a cluster of one)."""
        environ = os.environ if environ is None else environ
        config = {name: environ.get(variable, "").strip() for name, variable in ENV.items()}
        return cls(config) if any(config.values()) else None

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
