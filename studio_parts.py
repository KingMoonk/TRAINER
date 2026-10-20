"""The serving side of parallel transfers (studio_peer.PeerClient; docs/distributed.md, "Parallel transfers"). Pure Python: no server.

`byte_range(header, size)` reads a `Range: bytes=a-b` request header against a body of `size` bytes, so a route that holds a body in
memory (the round's weights, the replica state) can answer a part of it with 206.

`Assembler` takes a body that arrives as parts posted at the same time and gives it back whole once every part is in. A part is
checked against its own hash when it comes; the whole is checked against the sender's hash before it is handed on, and a body that
fails either check is dropped, so the sender starts it again.
"""
import hashlib
import threading
import time

MAX_TOTAL = 4 << 30  # bytes a body in parts may total
MAX_PARTS = 4096
KEEP = 600.  # seconds an unfinished body is kept after its latest part


class BadPart(ValueError):
    """A part or a whole body that is not what the sender said it is."""


def byte_range(header, size):
    """(first, last) of the bytes `header` asks of a body of `size` bytes, both inclusive; None when the request has no single range
    (no header, several ranges or another unit: the whole body is the answer); raises BadPart when the range cannot be satisfied (416)."""
    if not header or not header.strip().lower().startswith("bytes="):
        return None
    spec = header.strip()[6:]
    if "," in spec or "-" not in spec:
        return None
    first, _, last = (part.strip() for part in spec.partition("-"))
    try:
        if not first:  # the last N bytes
            count = int(last)
            first, last = max(0, size - count), size - 1
            if count <= 0:
                first = size
        else:
            first, last = int(first), min(int(last), size - 1) if last else size - 1
    except ValueError:
        return None
    if first < 0 or first >= size or last < first:
        raise BadPart(f"Range {header!r} cannot be satisfied for a body of {size} bytes.")
    return first, last


class Assembler:
    """Bodies in parts, by key (who sent which upload). `add` returns (body, seconds) once the last part is in, else (None, 0.):
    seconds are from the first part's start (`began`, on this clock, when its request began to be read) to the last part's end, so
    size / seconds is the speed of the upload as the receiver saw it."""

    def __init__(self, clock=time.monotonic):
        self.clock, self.lock, self.open = clock, threading.Lock(), {}

    def add(self, key, index, count, total, digest, data, part_digest, began=None):
        if not 0 < count <= MAX_PARTS or not 0 <= index < count:
            raise BadPart(f"Part {index} of {count} is out of range.")
        if not 0 < total <= MAX_TOTAL:
            raise BadPart(f"A body of {total} bytes is not accepted.")
        if hashlib.sha256(data).hexdigest() != part_digest:
            raise BadPart(f"Part {index} does not match its hash; send it again.")
        now = self.clock()
        with self.lock:
            for old in [k for k, held in self.open.items() if now - held["latest"] > KEEP]:
                del self.open[old]
            held = self.open.get(key)
            if held is not None and (held["count"], held["total"], held["digest"]) != (count, total, digest):
                del self.open[key]  # a different upload under the same key: the earlier one was given up
                held = None
            if held is None:
                held = self.open[key] = {"count": count, "total": total, "digest": digest, "parts": {}, "began": now if began is None else began, "latest": now}
            held["parts"][index], held["latest"] = data, now
            if began is not None:
                held["began"] = min(held["began"], began)
            if len(held["parts"]) < count:
                return None, 0.
            del self.open[key]
        body = b"".join(held["parts"][i] for i in range(count))
        if len(body) != total:
            raise BadPart(f"The parts add up to {len(body)} bytes, not the {total} the sender said.")
        if hashlib.sha256(body).hexdigest() != digest:
            raise BadPart("The assembled body does not match its hash; send it again.")
        return body, now - held["began"]
