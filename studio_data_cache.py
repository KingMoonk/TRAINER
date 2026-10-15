"""The Resources summaries on disk: data/summaries.sqlite.

Three tables. `datasets` holds one small JSON entry per dataset (counts and the dataset's `rev`), `rows` one row per sample
example (what the table shows, and the near-duplicate signature) and `meta` the format, the stamp of the files it describes
and the view: the near-duplicate counts and the first page of rows of each kind. A launch or a page read loads the entries
and the view only; the rows are read when somebody searches, and only as far as the first page of matches. Changing a
dataset rewrites that dataset's rows, not the whole file. One transaction writes it all, so a reader never sees half of it.
"""
import json
import sqlite3
from contextlib import closing

import studio_data_kinds as dk

PAGE = 200  # sample rows shown at once

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, entry TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rows (
    dataset_id TEXT NOT NULL, kind TEXT NOT NULL, pos INTEGER NOT NULL, id TEXT NOT NULL, a TEXT NOT NULL, b TEXT NOT NULL,
    status TEXT NOT NULL, can INTEGER NOT NULL, dup INTEGER NOT NULL DEFAULT 0, sig BLOB,
    PRIMARY KEY (dataset_id, kind, pos)) WITHOUT ROWID;
"""


class SummaryCache:
    def __init__(self, path, format):
        self.path, self.format = path, str(format)

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            connection.executescript(SCHEMA)
        except BaseException:
            connection.close()
            raise
        return connection

    def load(self):
        """({dataset id: entry}, stamp, view) of a saved summary of this format; ({}, None, None) when there is none, it is of
        another format or it cannot be read."""
        if not self.path.exists():
            return {}, None, None
        try:
            with closing(self._connect()) as connection:
                meta = dict(connection.execute("SELECT key, value FROM meta"))
                if meta.get("format") != self.format:
                    return {}, None, None
                entries = {key: json.loads(text) for key, text in connection.execute("SELECT id, entry FROM datasets")}
                return entries, json.loads(meta["stamp"]), json.loads(meta["view"])
        except (sqlite3.Error, ValueError, KeyError, TypeError):
            return {}, None, None

    def save(self, datasets, entries, stamp, scan_limit=20000):
        """Keep `entries` (the ones just read still carry their `rows`) and make the view. Rows past an entry's turn in the
        budget are dropped, so the file never holds more than `scan_limit` per kind. Returns the view."""
        try:
            connection = self._connect()
        except sqlite3.DatabaseError:  # not a database at all: start again
            self.path.unlink(missing_ok=True)
            connection = self._connect()
        with closing(connection), connection:
            if connection.execute("SELECT value FROM meta WHERE key='format'").fetchone() != (self.format,):
                for table in ("meta", "datasets", "rows"):
                    connection.execute(f"DELETE FROM {table}")
            ids = [d["id"] for d in datasets]
            for table, column in (("datasets", "id"), ("rows", "dataset_id")):
                connection.execute(f"DELETE FROM {table} WHERE {column} NOT IN ({','.join('?' * len(ids))})", ids)
            plan = list(dk.budgets(datasets, entries, scan_limit))
            for dataset, entry, take in plan:
                fresh = entry.pop("rows", None)
                if fresh is not None:
                    connection.execute("DELETE FROM rows WHERE dataset_id=?", (dataset["id"],))
                    for kind in dk.KINDS:
                        connection.executemany("INSERT INTO rows (dataset_id, kind, pos, id, a, b, status, can, sig) VALUES (?,?,?,?,?,?,?,?,?)",
                                               ((dataset["id"], kind, pos, *row) for pos, row in enumerate(fresh[kind][:take[kind]])))
                        entry["kept"][kind] = min(len(fresh[kind]), take[kind])
                else:
                    for kind in dk.KINDS:
                        if entry["kept"][kind] > take[kind]:
                            connection.execute("DELETE FROM rows WHERE dataset_id=? AND kind=? AND pos>=?", (dataset["id"], kind, take[kind]))
                            entry["kept"][kind] = take[kind]
                connection.execute("INSERT OR REPLACE INTO datasets (id, entry) VALUES (?,?)", (dataset["id"], json.dumps(entry, ensure_ascii=False)))
            view = self._view(connection, plan)
            connection.executemany("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                                   [("format", self.format), ("stamp", json.dumps(stamp)), ("view", json.dumps(view, ensure_ascii=False))])
        return view

    @staticmethod
    def _view(connection, plan):
        """{"duplicates": {kind: n}, "rows": {kind: first page of rows}, "takes": {kind: [[dataset id, rows taken]]}}; the
        near-duplicate flag of every row is brought up to date on the way, from the signatures of the whole budget in order."""
        names = {dataset["id"]: dataset["name"] for dataset, _, _ in plan}
        view = {"duplicates": {}, "rows": {}, "takes": {}}
        for kind in dk.KINDS:
            takes = [[dataset["id"], take[kind]] for dataset, _, take in plan if take[kind]]
            ordered = [(did, pos, *rest) for did, take in takes for pos, *rest in connection.execute(
                "SELECT pos, id, dup, sig FROM rows WHERE dataset_id=? AND kind=? AND pos<? ORDER BY pos", (did, kind, take))]
            dups = dk.match_signatures((id, *dk.unpack_signature(sig)) for _, _, id, _, sig in ordered if sig)
            connection.executemany("UPDATE rows SET dup=? WHERE dataset_id=? AND kind=? AND pos=?",
                                   [(int(id in dups), did, kind, pos) for did, pos, id, dup, _ in ordered if dup != (id in dups)])
            rows = []
            for did, take in takes:
                if len(rows) >= PAGE:
                    break
                rows += [_row(found, names[did]) for found in connection.execute(
                    "SELECT id, a, b, status, can, dup FROM rows WHERE dataset_id=? AND kind=? AND pos<? ORDER BY pos LIMIT ?",
                    (did, kind, take, PAGE - len(rows)))]
            view["duplicates"][kind], view["rows"][kind], view["takes"][kind] = len(dups), rows, takes
        return view

    def search(self, kind, query, takes, names, limit=PAGE):
        """The first `limit` rows of the kind whose two columns contain `query`: what dk.matching() finds in every row, read
        one dataset at a time until the page is full. `takes` and `names` come from the view and the workspace."""
        needle = query.casefold().strip()
        found = []
        with closing(self._connect()) as connection:
            for did, take in takes:
                for row in connection.execute("SELECT id, a, b, status, can, dup FROM rows WHERE dataset_id=? AND kind=? AND pos<? ORDER BY pos",
                                              (did, kind, take)):
                    if not needle or needle in f"{row[1]} {row[2]}".casefold():
                        found.append(_row(row, names[did]))
                        if len(found) >= limit:
                            return found
        return found


def _row(found, dataset_name):
    id, a, b, status, can, dup = found
    return {"id": id, "a": a, "b": b, "status": "Near-duplicate" if can and dup else status, "dataset": dataset_name}
