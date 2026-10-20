"""What the collector writes to disk besides records: the state (survives a restart), the journal, and the daily ledger of caps.

Everything is under data/collect/. Each file is replaced atomically, so a crash leaves the last complete copy.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from studio_core import atomic_json, read_json
from studio_collect.config import collect_dir

JOURNAL_LIMIT = 5 * 1024 * 1024
DROPS = ("licence", "quality", "pii", "duplicate", "near_duplicate", "decontam", "invalid")


class LimitReached(Exception):
    """A cap (disk, bandwidth, requests per host) is used up. The work stops and resumes when the cap frees up."""


class Stopped(Exception):
    """The collector was told to stop or pause, or the machine stopped being idle. Work saved so far is kept."""


def blank_stats():
    return {"seen": 0, "kept": 0, "dropped": dict.fromkeys(DROPS, 0), "redactions": 0, "bytes_kept": 0}


class State:
    """data/collect/state.json: status, per-source cursors (what makes a run resumable) and the counters the page shows."""

    def __init__(self, root):
        self.path = collect_dir(root) / "state.json"
        self.lock = threading.RLock()
        self.data = self._read()

    def _read(self):
        try:
            data = read_json(self.path)
            if isinstance(data, dict):
                return {**self._blank(), **data}
        except (OSError, ValueError):
            pass
        return self._blank()

    @staticmethod
    def _blank():
        return {"status": "off", "reason": "", "pid": None, "heartbeat": None, "plan": [], "plans": {}, "sources": {},
                "stats": blank_stats(), "groups": {}, "registered": {}}

    def save(self, **fields):
        with self.lock:
            self.data.update(fields)
            atomic_json(self.path, self.data)

    def source(self, key):
        return self.data["sources"].setdefault(key, {})

    def update_source(self, key, **fields):
        with self.lock:
            self.source(key).update(fields)

    def reload(self):
        with self.lock:
            self.data = self._read()
        return self.data

    def add_stats(self, delta, group=None):
        """Add a stats dict to the totals, or (with `group`, the dataset the records land in) to that group's counters."""
        with self.lock:
            target = self.data["groups"].setdefault(group, blank_stats()) if group else self.data["stats"]
            for key, value in delta.items():
                if key == "dropped":
                    for reason, n in value.items():
                        target["dropped"][reason] = target["dropped"].get(reason, 0) + n
                else:
                    target[key] = target.get(key, 0) + value


class Journal:
    """data/collect/journal.jsonl: one line per event, oldest first. A file past JOURNAL_LIMIT is kept as .1 and started afresh."""

    def __init__(self, root):
        self.path = collect_dir(root) / "journal.jsonl"
        self.lock = threading.Lock()

    def append(self, type_, **fields):
        line = json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "type": type_, **fields}, ensure_ascii=False)
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size > JOURNAL_LIMIT:
                os.replace(self.path, self.path.with_suffix(".jsonl.1"))
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

    def tail(self, count=50):
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()[-count:]
        except OSError:
            return []
        rows = []
        for line in lines:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        return rows


class Ledger:
    """data/collect/ledger.json: today's bandwidth and requests per host, against the caps. A new UTC day starts at zero."""

    def __init__(self, root, clock=time.time):
        self.path = collect_dir(root) / "ledger.json"
        self.clock, self.lock = clock, threading.Lock()
        try:
            self.data = read_json(self.path)
        except (OSError, ValueError):
            self.data = {}
        self._roll()
        self._dirty = 0.0

    def _day(self):
        return datetime.fromtimestamp(self.clock(), timezone.utc).strftime("%Y-%m-%d")

    def _roll(self):
        if self.data.get("day") != self._day():
            self.data = {"day": self._day(), "bytes": 0, "requests": {}, "total_bytes": self.data.get("total_bytes", 0)}

    def check(self, host, limits):
        with self.lock:
            self._roll()
            if self.data["bytes"] >= limits["bandwidth_mb_per_day"] * 1_000_000:
                raise LimitReached(f"Today's bandwidth cap of {limits['bandwidth_mb_per_day']:g} MB is used up.")
            if self.data["requests"].get(host, 0) >= limits["requests_per_host_per_day"]:
                raise LimitReached(f"Today's request cap for {host} ({limits['requests_per_host_per_day']}) is used up.")

    def request(self, host):
        with self.lock:
            self.data["requests"][host] = self.data["requests"].get(host, 0) + 1
            self._save()

    def received(self, count, limits):
        with self.lock:
            self.data["bytes"] += count
            self.data["total_bytes"] = self.data.get("total_bytes", 0) + count
            self._save()
            if self.data["bytes"] >= limits["bandwidth_mb_per_day"] * 1_000_000:
                raise LimitReached(f"Today's bandwidth cap of {limits['bandwidth_mb_per_day']:g} MB is used up.")

    def _save(self, force=False):
        if force or self.clock() - self._dirty > 2:   # twice a second would be a lot of small writes
            self._dirty = self.clock()
            atomic_json(self.path, self.data)

    def flush(self):
        with self.lock:
            self._save(True)

    def today(self):
        with self.lock:
            self._roll()
            return {"day": self.data["day"], "bytes": self.data["bytes"], "requests": dict(self.data["requests"]), "total_bytes": self.data.get("total_bytes", 0)}


def disk_used(root):
    """Bytes under data/collect/ (hand-off shards, downloads in progress, the dedup index, state and journal)."""
    total = 0
    for base, _, files in os.walk(collect_dir(root)):
        for name in files:
            try:
                total += (Path(base) / name).stat().st_size
            except OSError:
                continue
    return total
