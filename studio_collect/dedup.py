"""Duplicates: what the workspace already holds, and what the collector has already kept.

One SQLite file (data/collect/seen.sqlite) with two parts:
- exact: the first 16 bytes of the sha256 of a record's content, both studio_core.fingerprint (the key records.sqlite uses, so a
  record the workspace already stores is found) and the same hash of the whitespace/case-normalised text;
- near: min-hash signatures from studio_data_kinds (16 hashes in 8 bands of 2, plus a bottom-k sketch). A band lookup returns the
  few earlier documents sharing a bucket and the Jaccard of the sketches decides, as match_signatures does for a batch, so the cost
  per document stays constant however much is kept.

The index is committed after the shard that holds the records is on disk (`commit`), never before: a crash can re-offer a record
(the shard drops it as a duplicate on registration) but cannot lose one.
"""
from __future__ import annotations

import hashlib
import sqlite3

import studio_core
import studio_data_kinds as dk
from studio_collect.config import collect_dir

MAX_NEAR_DOCUMENTS = 3_000_000   # past this the near-duplicate index stops growing (exact duplicates are still found)
SCHEMA = """
CREATE TABLE IF NOT EXISTS exact (fp BLOB PRIMARY KEY) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS sketch (id INTEGER PRIMARY KEY, packed BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS band (b INTEGER NOT NULL, h1 INTEGER NOT NULL, h2 INTEGER NOT NULL, id INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS band_lookup ON band(b, h1, h2);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
"""


def _key(hex_digest):
    return bytes.fromhex(hex_digest[:32])


def normalised_fingerprint(example):
    return hashlib.sha256(dk.normalized_text(example).encode()).hexdigest()


class Seen:
    def __init__(self, root):
        path = collect_dir(root) / "seen.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self._new_exact, self._new_signatures, self._new_bands = set(), [], {}   # accepted since the last commit()

    def close(self):
        self.db.close()

    def _meta(self, key, default=0):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def _set_meta(self, key, value):
        self.db.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    # ---- the workspace's own records -----------------------------------------------------------------------
    def absorb_workspace(self, workspace, stop=lambda: None):
        """Remember the fingerprint of every record records.sqlite holds (only those added since the last call), so a record the
        workspace already has is a duplicate here. Returns how many were added."""
        watermark, added = self._meta("workspace_rowid"), 0
        if not workspace.db_path.exists():
            return 0
        source = sqlite3.connect(f"file:{workspace.db_path}?mode=ro", uri=True, timeout=30)
        try:
            cursor = source.execute("SELECT rowid, fingerprint FROM records WHERE rowid > ? ORDER BY rowid", (watermark,))
            while batch := cursor.fetchmany(20000):
                stop()
                self.db.executemany("INSERT OR IGNORE INTO exact(fp) VALUES (?)", [(_key(fp),) for _, fp in batch if len(fp) >= 32])
                watermark, added = batch[-1][0], added + len(batch)
                self._set_meta("workspace_rowid", watermark)
                self.db.commit()
        except sqlite3.OperationalError:   # no records table yet
            return 0
        finally:
            source.close()
        return added

    # ---- checks ----------------------------------------------------------------------------------------------
    def exact(self, example):
        """(keys, is_duplicate): the two exact keys of a record, and whether either is already known (kept, or in the workspace)."""
        keys = (_key(studio_core.fingerprint(example)), _key(normalised_fingerprint(example)))
        known = any(k in self._new_exact or self.db.execute("SELECT 1 FROM exact WHERE fp=?", (k,)).fetchone() for k in keys)
        return keys, known

    def near(self, example):
        """(is a near-copy, signature or None). The signature is kept for add() when the record is accepted."""
        found = dk.signature(dk.normalized_text(example))
        if found is None:
            return False, None
        sig, sketch = found
        for b in range(8):
            band = (b, sig[2 * b], sig[2 * b + 1])
            if any(self._jaccard(sketch, other) >= dk.NEAR_DUPLICATE for other in self._new_bands.get(band, ())):
                return True, None
            rows = self.db.execute("SELECT packed FROM sketch WHERE id IN (SELECT id FROM band WHERE b=? AND h1=? AND h2=? LIMIT ?)",
                                   (*band, dk.BUCKET_CAP)).fetchall()
            if any(self._jaccard(sketch, dk.unpack_signature(packed)[1]) >= dk.NEAR_DUPLICATE for (packed,) in rows):
                return True, None
        return False, (sig, sketch)

    @staticmethod
    def _jaccard(a, b):
        return len(a & b) / max(1, len(a | b))

    def add(self, keys, signature=None):
        """Remember an accepted record (written at the next commit; found by the checks at once)."""
        self._new_exact.update(keys)
        if signature is not None:
            sig, sketch = signature
            self._new_signatures.append(signature)
            for b in range(8):
                bucket = self._new_bands.setdefault((b, sig[2 * b], sig[2 * b + 1]), [])
                if len(bucket) < dk.BUCKET_CAP:
                    bucket.append(sketch)

    def commit(self):
        """Write what add() collected. Call after the shard holding those records is on disk."""
        db, documents = self.db, self._meta("near_documents")
        db.executemany("INSERT OR IGNORE INTO exact(fp) VALUES (?)", [(k,) for k in self._new_exact])
        for sig, sketch in self._new_signatures:
            if documents >= MAX_NEAR_DOCUMENTS:
                break
            cursor = db.execute("INSERT INTO sketch(packed) VALUES (?)", (dk.pack_signature(sig, sketch),))
            db.executemany("INSERT INTO band(b,h1,h2,id) VALUES (?,?,?,?)", [(b, sig[2 * b], sig[2 * b + 1], cursor.lastrowid) for b in range(8)])
            documents += 1
        self._set_meta("near_documents", documents)
        db.commit()
        self._new_exact, self._new_signatures, self._new_bands = set(), [], {}

    def counts(self):
        return {"exact": self.db.execute("SELECT COUNT(*) FROM exact").fetchone()[0], "near": self._meta("near_documents")}
