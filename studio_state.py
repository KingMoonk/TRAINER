"""The observation store every node keeps (docs/distributed.md, "Cluster state: what every node knows"). Pure logic plus one thin
I/O edge: no threads, no files except the incarnation counter, and every clock is injected, so each rule is a window-free test
with a fake clock and fake peers (the same shape as studio_scheduler).

* An entry is `(origin, kind, key) -> value`, versioned `(incarnation, seq)`. Only the origin writes its own entries (`put`);
  everything else arrives through `merge`.
* `merge` keeps the higher version of each entry, and a newer incarnation of an origin replaces all of its older entries.
  It is commutative, associative and idempotent, so replicas that have seen the same writes hold the same state.
* `vector()` is `{origin: (incarnation, seq)}`, the newest version held per origin. `delta(vector)` is every entry newer than
  that, from every origin: the answer to a sync. `handle_sync` is the peer's side of `POST /state/sync` and `sync_peer` the
  caller's (pull, then push back what the peer lacks).
* Ages, not timestamps. An entry's age travels with it; the receiver stores when it was born on its own monotonic clock.
* An origin whose newest entry is older than `prune_after` is pruned, leaving a tombstone `(origin, incarnation)` that refuses
  that incarnation (and older ones) for `tombstone_for` seconds. A node that returns has a newer incarnation and is accepted.
* `next_incarnation` is the only thing persisted: a counter in the node's data folder, incremented at every start.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

PRUNE_AFTER = 3600.  # seconds of silence from an origin before its entries are pruned
TOMBSTONE_FOR = 86400.  # seconds a pruned origin's tombstone refuses its old incarnation
INCARNATION_FILE = "incarnation.json"


def next_incarnation(path, now=time.time):
    """Read the counter at `path`, add one, save it and return it. A missing or unreadable file starts from the wall clock's
    seconds, which is above any counter this folder has held (a restart a second would be needed to catch up), so a node whose
    data folder was wiped still outranks the entries its old incarnations left in other replicas. A folder that cannot be written
    still returns the number: the node runs, and the next start reads the clock again."""
    path = Path(path)
    try:
        last = int(json.loads(path.read_text(encoding="utf-8"))["incarnation"])
    except (OSError, ValueError, KeyError, TypeError):
        last = int(now())
    number = last + 1
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps({"incarnation": number}), encoding="utf-8")
        os.replace(temp, path)
    except OSError:
        pass
    return number


class Replica:
    """One node's copy of the store. `origin` is this node's name in it and `incarnation` the counter from `next_incarnation`."""

    def __init__(self, origin, incarnation, clock=time.monotonic, prune_after=PRUNE_AFTER, tombstone_for=TOMBSTONE_FOR):
        self.origin, self.incarnation, self.clock = origin, int(incarnation), clock
        self.prune_after, self.tombstone_for = prune_after, tombstone_for
        self._entries = {}  # (origin, kind, key) -> (incarnation, seq, value, born on this clock)
        self._newest = {}  # origin -> (incarnation, highest seq held): the version vector
        self._tombstones = {}  # origin -> (incarnation, expires on this clock)
        self._seq = 0
        self._lock = threading.RLock()
        self._watchers = []

    def watch(self, watcher):
        """Call `watcher(changes)` after a write or merge that changed a value, `changes` being the set of (origin, kind, key) whose
        value is new. A version bump that leaves the value as it was is not a change. Returns the call that stops it."""
        self._watchers.append(watcher)
        return lambda: watcher in self._watchers and self._watchers.remove(watcher)

    def _tell(self, changes):
        if changes:
            for watcher in list(self._watchers):
                watcher(changes)

    # ---- writing ----------------------------------------------------------------------------------------------------
    def put(self, kind, key, value):
        """Write one of this node's own entries; returns its version. `value` must be JSON-serialisable (TypeError otherwise)."""
        value = json.loads(json.dumps(value))
        with self._lock:
            self._seq += 1
            held = self._entries.get((self.origin, kind, key))
            self._entries[(self.origin, kind, key)] = (self.incarnation, self._seq, value, self.clock())
            self._newest[self.origin] = (self.incarnation, self._seq)
            version = self.incarnation, self._seq
        self._tell(set() if held is not None and held[2] == value else {(self.origin, kind, key)})
        return version

    # ---- reading ----------------------------------------------------------------------------------------------------
    def get(self, origin, kind, key):
        """The entry as a dict (origin, kind, key, incarnation, seq, value, age), or None."""
        with self._lock:
            self.prune()
            held = self._entries.get((origin, kind, key))
            return None if held is None else self._row(origin, kind, key, held, self.clock())

    def entries(self, origin=None, kind=None):
        """Every entry held (optionally of one origin and/or kind) as dicts with the age each has now, sorted for a stable display."""
        with self._lock:
            self.prune()
            now = self.clock()
            return [self._row(o, k, key, held, now) for (o, k, key), held in sorted(self._entries.items())
                    if (origin is None or o == origin) and (kind is None or k == kind)]

    def contents(self):
        """What two replicas must agree on once they have seen the same writes: every entry with its version and value, no ages."""
        with self._lock:
            self.prune()
            return sorted((o, k, key, inc, seq, json.dumps(value, sort_keys=True))
                          for (o, k, key), (inc, seq, value, _) in self._entries.items())

    def vector(self):
        with self._lock:
            self.prune()
            return {origin: list(version) for origin, version in self._newest.items()}

    @staticmethod
    def _row(origin, kind, key, held, now):
        inc, seq, value, born = held
        return {"origin": origin, "kind": kind, "key": key, "incarnation": inc, "seq": seq, "value": value, "age": max(0., now - born)}

    # ---- syncing ----------------------------------------------------------------------------------------------------
    def delta(self, vector):
        """Every entry, from every origin, whose version is newer than the origin's in `vector` (all of them for an empty one), as the
        wire form: the rows of `entries()`."""
        with self._lock:
            self.prune()
            now = self.clock()
            return [self._row(o, k, key, held, now) for (o, k, key), held in sorted(self._entries.items())
                    if (held[0], held[1]) > _version(vector, o)]

    def merge(self, wire):
        """Take in entries (rows as `delta` gives them). Returns how many changed this replica. Rows that are malformed, for this node's
        own origin (only it writes those), tombstoned or of an older incarnation than held are skipped, as are older versions."""
        changed, changes = 0, set()
        with self._lock:
            self.prune()
            now = self.clock()
            for row in wire or ():
                try:
                    origin, kind, key = str(row["origin"]), str(row["kind"]), str(row["key"])
                    inc, seq, age = int(row["incarnation"]), int(row["seq"]), float(row["age"])
                    value = row["value"]
                except (KeyError, TypeError, ValueError):
                    continue
                if origin == self.origin or not math.isfinite(age) or seq < 1 or inc < 0:
                    continue
                held = self._entries.get((origin, kind, key))
                if self._accept(origin, kind, key, inc, seq, value, now - max(0., age)):
                    changed += 1
                    if held is None or held[2] != value:
                        changes.add((origin, kind, key))
            self.prune()
        self._tell(changes)
        return changed

    def _accept(self, origin, kind, key, inc, seq, value, born):
        tomb = self._tombstones.get(origin)
        if tomb and inc <= tomb[0]:
            return False
        held_inc, held_seq = self._newest.get(origin, (-1, 0))
        if inc < held_inc:
            return False
        if inc > held_inc:  # a restarted origin: everything older it left is replaced
            for stale in [k for k in self._entries if k[0] == origin]:
                del self._entries[stale]
            held_seq = 0
        current = self._entries.get((origin, kind, key))
        if current and (current[0], current[1]) >= (inc, seq):
            if (current[0], current[1]) == (inc, seq) and born > current[3]:
                self._entries[(origin, kind, key)] = (*current[:3], born)  # the same write seen again, by a shorter way
            return False
        self._entries[(origin, kind, key)] = (inc, seq, value, born)
        self._newest[origin] = (inc, max(held_seq, seq))
        return True

    def prune(self):
        """Drop every other origin whose newest entry is `prune_after` old, leaving a tombstone, and tombstones past their day."""
        with self._lock:
            now = self.clock()
            for origin, (_, expires) in list(self._tombstones.items()):
                if now >= expires:
                    del self._tombstones[origin]
            newest = {}
            for (origin, _, _), (_, _, _, born) in self._entries.items():
                newest[origin] = max(born, newest.get(origin, born))
            for origin, born in newest.items():
                if origin != self.origin and now - born >= self.prune_after:
                    self._tombstones[origin] = (self._newest[origin][0], now + self.tombstone_for)
                    del self._newest[origin]
                    for stale in [k for k in self._entries if k[0] == origin]:
                        del self._entries[stale]

    def handle_sync(self, body):
        """The peer's side of `POST /state/sync`: take in the pushed entries, answer with everything newer than the caller's vector
        and this node's own vector."""
        self.merge(body.get("entries"))
        return {"entries": self.delta(body.get("vector") or {}), "vector": self.vector()}


def _version(vector, origin):
    try:
        inc, seq = vector.get(origin, (-1, 0))
        return int(inc), int(seq)
    except (AttributeError, TypeError, ValueError):
        return -1, 0


def sync_peer(replica, client):
    """One push-pull exchange with a peer through `client` (a studio_peer.PeerClient): send our vector and take what is newer, then
    push back what the peer's vector shows it lacks. Raises the client's error when the peer cannot be reached."""
    reply = client.post_json("state/sync", {"vector": replica.vector(), "entries": []})
    replica.merge(reply.get("entries"))
    lacking = replica.delta(reply.get("vector") or {})
    if lacking:
        client.post_json("state/sync", {"vector": replica.vector(), "entries": lacking})
