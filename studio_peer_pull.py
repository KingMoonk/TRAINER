"""Pull a published version straight from the node that holds it (docs/cluster.md), so a copy between two machines on the tailnet
does not go up to a bucket and down again.

The owner serves two routes (studio_api/routers/versions.py): `GET /versions/{n}/weights/meta`, the metadata `publish` would
upload (key, name, design, size, sha256), and `GET /versions/{n}/weights`, the weights-only half-precision file, streamed. Nothing
else crosses: no optimizer state, no run files. The puller checks the sha256 and widens the weights exactly as a pull from the
store does (`studio_publish.finish_pull`), so the file it writes is the same either way.

A peer is found by name: a version's origin is the hostname of the machine that published it, and `GET /node` says each listed
node's hostname. Looking is bounded by `PROBE` seconds, in parallel, so an owner that is off costs one wait, not one per node.

Slow (network, disk): call it from a worker, never the Tk thread.
"""
from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import studio_cluster
from studio_peer import PeerError, base_url
from studio_publish import _check_key, _origin_name, finish_pull

PROBE = studio_cluster.POLL_TIMEOUT  # seconds a node has to say it is there before the store is used instead
BUILD = 60.  # seconds the owner may take to make its copy the first time (it half-precisions the weights and hashes them)
READ = 30.  # seconds of silence in the middle of a download before it is given up on
MISSING = {"sha256", "key", "number", "origin", "spec"}


def find_owner(addresses, origin, client_for):
    """(client, "") for the listed node whose hostname makes `origin`; (None, "off") when a listed node of that name did not answer;
    (None, "") when no listed node is known to be it. `client_for(address, **options)` makes the PeerClient."""
    if not addresses:
        return None, ""

    def ask(address):
        try:
            client = client_for(address, retries=0, timeout=PROBE)
            return address, client, _origin_name(str(client.get_json("node").get("hostname") or ""))
        except Exception:  # noqa: BLE001 - unreachable, slow or nonsense all mean "did not answer"
            return address, None, ""

    with ThreadPoolExecutor(max_workers=min(8, len(addresses))) as pool:
        answers = list(pool.map(ask, addresses))
    for _, client, name in answers:
        if client is not None and name == origin:
            return client, ""
    # a node that did not answer cannot say its hostname: its address, which is usually the machine's name, is all there is to go on
    named = [a for a, client, _ in answers if client is None and _origin_name(urlsplit(base_url(a)).hostname or "").split(".")[0] == origin.split(".")[0]]
    return None, "off" if named else ""


def fetch(client, number, dest_dir, held):
    """Copy version `number` from the node `client` reaches into `dest_dir`. Returns (metadata, path); path is None, and nothing
    was downloaded, when `held(metadata)` says this workspace already has it. Raises PeerError when the node cannot send it and
    ValueError when what it sent does not check out; either way the caller uses the store."""
    meta = client.get_json(f"versions/{number}/weights/meta", timeout=BUILD)
    if not isinstance(meta, dict) or MISSING - meta.keys() or meta["number"] != number:
        raise PeerError(f"{client.address} sent something that is not version {number}'s description.")
    _check_key(meta["key"])
    if held(meta):
        return meta, None
    destination = Path(dest_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination, prefix=".pull-") as scratch:
        artifact = Path(scratch) / "artifact.npz"
        client.get_file(f"versions/{number}/weights", artifact, timeout=READ)
        return meta, finish_pull(artifact, meta, destination)
