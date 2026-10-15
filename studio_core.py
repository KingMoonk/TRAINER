"""Local workspace: datasets, examples and runs stored on disk."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import re
import sqlite3
import sys
import tempfile
import time
import threading
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import orjson
import pandas as pd
from studio_ui_guard import slow

SCHEMA_VERSION = 3   # PRAGMA user_version of records.sqlite; 0 = made before the version was recorded (no kind/holdout columns), 2 = before the kind_stats index
MIGRATE_BATCH = 5000
INSERT_BATCH = 5000
_COLUMNS = "id,dataset_id,fingerprint,payload,user,assistant,intent,quality,characters,kind,holdout"
_READ = "payload,user,assistant"  # what _row_to_example needs
_CREATE_RECORDS = """CREATE TABLE IF NOT EXISTS records (
    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
    payload TEXT NOT NULL, user TEXT NOT NULL, assistant TEXT NOT NULL,
    intent TEXT NOT NULL, quality TEXT NOT NULL, characters INTEGER NOT NULL,
    kind TEXT, holdout INTEGER,
    UNIQUE(dataset_id, fingerprint))"""
_PICK_INDEX = "CREATE INDEX IF NOT EXISTS records_pick ON records(kind, holdout, quality, dataset_id)"
_STATS_INDEX = "CREATE INDEX IF NOT EXISTS records_stats ON records(dataset_id, kind, quality, characters, holdout)"  # kind_stats and review_count read this alone


def _root():
    """The folder holding data/ and runs/. Next to the code when run from source; a frozen bundle is read-only and replaced on upgrade, so it goes in the user's data folder."""
    if not getattr(sys, "frozen", False):
        return Path(__file__).resolve().parent
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    (base / "Groundwork").mkdir(parents=True, exist_ok=True)
    return base / "Groundwork"


def module_command(module, *args):
    """Command line that runs `module` as a script in a new process. A frozen app has no python: its executable dispatches on --run-module (see app.py)."""
    return [sys.executable, *(["--run-module"] if getattr(sys, "frozen", False) else ["-m"]), module, *args]


ROOT = _root()
SUPPORTED = {".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".txt"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def uid(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(orjson.dumps(value, option=_JSON))
            stream.flush()
            os.fsync(stream.fileno())  # a renamed file must not come back empty after a power loss
        replace_file(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


_JSON = orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY  # stdlib's indent=2, int keys and numpy scalars as before


def read_json(path):
    return orjson.loads(Path(path).read_bytes())


# Files a run is still writing (metrics.jsonl, samples.jsonl) are polled every refresh. Each is parsed once: a later read
# stats it, and if it grew parses only the new bytes. Keyed on size and mtime, reset when the file shrank or was replaced.
_TAILS = OrderedDict()
_TAILS_KEPT = 256
_TAILS_LOCK = threading.Lock()  # the shell's workers read the same runs at once
_CHECK = 64  # bytes before the saved offset that must still match, or the file was replaced by a longer one


def _jsonl_row(line):
    try:
        row = orjson.loads(line)
    except orjson.JSONDecodeError:
        try:
            row = json.loads(line)  # a diverged run logs NaN, which json reads and orjson refuses
        except ValueError:
            return None
    return row if isinstance(row, dict) else None


def _remember(key, entry):  # call with _TAILS_LOCK held
    _TAILS[key] = entry
    _TAILS.move_to_end(key)
    while len(_TAILS) > _TAILS_KEPT:
        _TAILS.popitem(last=False)


def _jsonl_rows(data):
    return [row for row in map(_jsonl_row, filter(bytes.strip, data.split(b"\n"))) if row is not None]


def jsonl_rows(path):
    """The rows of a JSONL file a run appends to. Lines that do not parse (a crash or a write in progress) are skipped."""
    key = str(path)
    try:
        stat = os.stat(path)
    except OSError:
        with _TAILS_LOCK:
            _TAILS.pop(key, None)
        return []
    with _TAILS_LOCK:
        seen = _TAILS.get(key)
        if seen and (seen["size"], seen["mtime"]) == (stat.st_size, stat.st_mtime_ns):
            _TAILS.move_to_end(key)
            return seen["rows"] + seen["last"]
        try:
            with open(path, "rb") as stream:
                if seen and stat.st_size >= seen["offset"]:
                    stream.seek(max(0, seen["offset"] - _CHECK))
                    if stream.read(min(_CHECK, seen["offset"])) != seen["check"]:
                        seen = None
                else:
                    seen = None
                if seen is None:
                    stream.seek(0)
                data = stream.read()
        except OSError:
            _TAILS.pop(key, None)
            return []
        base = seen or {"offset": 0, "rows": [], "check": b""}
        whole = data[:data.rfind(b"\n") + 1]  # the unfinished last line waits for its newline
        rows = base["rows"] + _jsonl_rows(whole)
        offset = base["offset"] + len(whole)
        check = (base["check"] + whole)[-_CHECK:]
        # a last line that parses is a row even without its newline; it is kept apart so the offset stays on a line start
        last = _jsonl_rows(data[len(whole):])
        _remember(key, {"size": stat.st_size, "mtime": stat.st_mtime_ns, "offset": offset, "rows": rows, "check": check, "last": last})
        return rows + last


def jsonl_tail(path, count):
    """The last `count` rows of a JSONL file whose lines can be long (samples.jsonl), reading only its end."""
    key = (str(path), count)
    try:
        stat = os.stat(path)
    except OSError:
        return []
    with _TAILS_LOCK:
        seen = _TAILS.get(key)
        if seen and (seen["size"], seen["mtime"]) == (stat.st_size, stat.st_mtime_ns):
            _TAILS.move_to_end(key)
            return list(seen["rows"])
        try:
            with open(path, "rb") as stream:
                size, chunk = stream.seek(0, 2), 1 << 16
                while True:
                    start = max(0, size - chunk)
                    stream.seek(start)
                    data = stream.read(size - start)
                    if start:
                        data = data[data.find(b"\n") + 1:]  # the first line is probably cut
                    if start and data.count(b"\n") <= count:
                        chunk *= 2  # not enough whole lines yet; do not parse until there are
                        continue
                    rows = _jsonl_rows(data)
                    if len(rows) >= count or not start:
                        break
                    chunk *= 2
        except OSError:
            return []
        _remember(key, {"size": stat.st_size, "mtime": stat.st_mtime_ns, "rows": rows[-count:]})
        return rows[-count:]


def replace_file(source, target, attempts=40):
    """Atomic rename that tolerates Windows' transient sharing locks.

    A reader (the runs list, antivirus, the search indexer) holding the target
    open makes os.replace fail with Access denied for a few milliseconds.
    """
    for attempt in range(attempts):
        try:
            return Path(source).replace(target)
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(.5, .01 * (attempt + 1)))


def truthy(value):
    """bool() that reads the strings "False", "0", "no" and "" as false (CSV and JSON imports carry flags as text)."""
    return value.strip().lower() not in {"", "false", "0", "no", "none", "null"} if isinstance(value, str) else bool(value)


def messages_of(example):
    if example.get("messages"):
        return example["messages"]
    return [{"role": "user", "content": example.get("user", "")},
            {"role": "assistant", "content": example.get("assistant", "")}]


def fingerprint(example):
    value = {"text": example["text"]} if example.get("text") else {"messages": messages_of(example)}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def normalize(row, dataset_id="", source="import"):
    if isinstance(row, str):
        row = {"text": row}
    if not isinstance(row, dict):
        return None
    raw_messages = row.get("messages", row.get("conversation", row.get("conversations")))
    if isinstance(raw_messages, str) and raw_messages.strip().startswith("["):
        try:
            raw_messages = json.loads(raw_messages)
        except ValueError:
            pass
    messages = []
    if isinstance(raw_messages, list):
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", item.get("from", ""))).lower()
            role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role)
            content = str(item.get("content", item.get("value", "")) or "").strip()
            if role in {"system", "user", "assistant"} and content:
                messages.append({"role": role, "content": content})
    if not messages:
        prompt = row.get("user", row.get("prompt", row.get("question", row.get("instruction", ""))))
        if row.get("instruction") and row.get("input"):
            prompt = str(prompt) + "\n" + str(row["input"])
        answer = row.get("assistant", row.get("completion", row.get("answer", row.get("output", row.get("response", "")))))
        prompt = "" if prompt is None else str(prompt).strip()
        answer = "" if answer is None else str(answer).strip()
        if prompt and answer:
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    text = str(row.get("text", "") or "").strip() if not messages else ""
    if not text and not ({m["role"] for m in messages} >= {"user", "assistant"}):
        return None
    result = {
        "id": uid("ex"), "datasetId": dataset_id, "messages": messages, "text": text,
        "user": text if text else "\n".join(m["content"] for m in messages if m["role"] == "user"),
        "assistant": "\n".join(m["content"] for m in messages if m["role"] == "assistant"),
        "intent": str(row.get("intent", row.get("category", "general"))), "quality": row.get("quality") or "Review",
        "synthetic": truthy(row.get("synthetic", source == "synthetic")), "source": row.get("source", source), "updatedAt": now(),
    }
    # Keep audit and attribution information when a draft is accepted or edited.
    for key in ("provenance", "validation", "qualityScore", "quality_score", "flags", "recipe", "recipeVersion",
                "seed", "difficulty", "tags", "sourceId", "source_url", "license", "split", "metadata",
                "generation", "reviewNotes", "groupId", "computedVerified", "qualityReport"):
        if key in row:
            result[key] = row[key]
    return result


def parse_file(path):
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file: {path.name}")
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("Import files in chunks below 100 MB.")
    text = path.read_text(encoding="utf-8-sig")
    ext = path.suffix.lower()
    if ext in {".jsonl", ".ndjson"}:
        rows = []
        for line_number, line in enumerate(text.split("\n"), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}, line {line_number}: {exc.msg}") from exc
        return rows
    if ext == ".json":
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("examples", "data", "records"):
                if isinstance(parsed.get(key), list):
                    return parsed[key]
            return [parsed]
        raise ValueError("JSON must contain records or a record array.")
    if ext in {".csv", ".tsv"}:
        frame = pd.read_csv(io.StringIO(text), sep="," if ext == ".csv" else "\t", dtype=str, keep_default_na=False)
        frame.columns = [str(k).strip().lower().replace(" ", "_") for k in frame.columns]
        return frame.to_dict(orient="records")
    # Tagged TXT preserves turn order. Plain TXT is a language-model corpus.
    blocks = re.split(r"\n\s*\n", text.strip())
    records = []
    for block in blocks:
        messages = []
        for line in block.splitlines():
            match = re.match(r"^\s*(system|user|human|assistant|bot)\s*:\s*(.*)$", line, re.I)
            if match:
                role = {"human": "user", "bot": "assistant"}.get(match[1].lower(), match[1].lower())
                messages.append({"role": role, "content": match[2]})
            elif messages:
                messages[-1]["content"] += "\n" + line
        if messages:
            records.append({"messages": messages})
        elif block.strip():
            records.append({"text": block})
    return records


class Workspace:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.path = self.root / "data" / "workspace.json"
        self.run_dir = self.root / "runs"
        self.db_path = self.root / "data" / "records.sqlite"
        self.lock = threading.RLock()
        self._migrate_lock = threading.Lock()
        self._rev_lock = threading.Lock()
        self.progress = None  # optional callback(text, fraction): told how far a schema migration has got
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(self.state, dict) or any(not isinstance(self.state.get(k), list) for k in ("datasets", "examples")):
                raise ValueError(f"Invalid workspace file: {self.path}")
        else:
            self.state = {"datasets": [], "examples": [], "generated": [], "runs": []}
        self.state.setdefault("generated", [])
        self.state.setdefault("studioConfig", {})
        self.state.setdefault("revBase", 0)
        # External catalogue metadata survives saves made by an older running GUI.
        registry = self.root / "data" / "catalog_registry.json"
        if registry.exists():
            try:
                entries = json.loads(registry.read_text(encoding="utf-8"))
                known = {d["id"] for d in self.state["datasets"]}
                self.state["datasets"].extend(d for d in entries if d.get("id") not in known)
            except (ValueError, TypeError, AttributeError):
                pass

    SMALL_STORE = 5000  # records: a store this small is read in place on the Tk thread (WorkspaceServices.ready), a larger one on a worker

    def is_small(self):
        return len(self.state["examples"]) + sum(d.get("recordCount") or 0 for d in self.state["datasets"]) <= self.SMALL_STORE

    def save(self):
        with self.lock:
            atomic_json(self.path, self.state)

    # Each dataset carries a persisted `rev` that goes up with every change to its data. A reader that cached what a read of
    # one dataset costs keeps that dataset's rev beside it and rereads only the datasets whose rev moved. The bump is in memory;
    # the save that follows every write persists it.
    @property
    def revision(self):
        """Goes up with every change to any dataset's data (and never down, even when a dataset is removed)."""
        return self.state.get("revBase", 0) + sum(d.get("rev", 0) for d in list(self.state["datasets"]))  # no lock: a UI thread asks this while a worker holds self.lock

    def _bump(self, *dataset_ids):
        with self._rev_lock:  # held for nothing else, so two writers never lose a bump
            for dataset_id in dataset_ids:
                dataset = self._dataset(dataset_id)
                if dataset is not None:
                    dataset["rev"] = dataset.get("rev", 0) + 1

    def next_rev(self, old=None):
        """The rev of a dataset's metadata that is replaced as a whole: above the old one's, or above every rev a removed dataset had."""
        return (old.get("rev", 0) if old else self.state.get("revBase", 0)) + 1

    def apply(self, mutate, save=True):
        """Run mutate(state) and the save under the lock; how worker threads change workspace state.

        The lock (an RLock) covers only in-memory state changes and the JSON write. Do long I/O (SQLite
        indexing, downloads, hashing) before calling this, never inside mutate."""
        with self.lock:
            result = mutate(self.state)
            if save:
                self.save()
            return result

    def create_dataset(self, name, description="", storage="json", **metadata):
        if not name.strip():
            raise ValueError("Enter a dataset name.")
        dataset = {"id": uid("data"), "name": name.strip(), "description": description, "createdAt": now(), "rev": self.next_rev()}
        if storage == "sqlite":
            dataset.update(storage="sqlite", recordCount=0)
        dataset.update(metadata)
        self.state["datasets"].append(dataset)
        self.state["selectedDatasetId"] = dataset["id"]
        self.save()
        return dataset["id"]

    def _dataset(self, dataset_id):
        return next((d for d in self.state["datasets"] if d["id"] == dataset_id), None)

    @contextmanager
    def _connect(self, touched=None):
        """A connection that commits on exit. `touched` is the set of dataset ids this block writes to (the block may add to it
        while it runs): their revs go up when it changes anything. Without it a write bumps every disk dataset, which is safe, not cheap."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=30)  # one connection per call, so none is ever used by two threads; a busy database is waited for up to 30 s
        try:
            connection.execute("PRAGMA synchronous=NORMAL")  # per connection; in WAL a power cut can lose the last commits but never corrupts
            if connection.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
                self._migrate(connection)
        except BaseException:
            connection.close()
            raise
        baseline = connection.total_changes  # a migration changes rows without changing what they say
        try:
            with connection:
                yield connection
        finally:
            changed = connection.total_changes > baseline
            connection.close()
            if changed:
                self._bump(*(touched if touched is not None else [d["id"] for d in self.state["datasets"] if d.get("storage") == "sqlite"]))

    def schema_pending(self):
        """True when records.sqlite exists and still has to be brought to the current schema (a launch can start that early)."""
        if not self.db_path.exists():
            return False
        connection = sqlite3.connect(self.db_path, timeout=30)
        try:
            return connection.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION
        finally:
            connection.close()

    def ensure_schema(self):
        with self._connect():
            pass

    def _migrate(self, connection):
        """Bring records.sqlite to SCHEMA_VERSION (0 = every database made before the version was recorded).

        Safe to interrupt and to repeat: columns are added only when missing, the fill commits every MIGRATE_BATCH rows
        and picks up at the rows still without a kind, and user_version moves only after everything else is done.
        Payloads are not rewritten here (new and edited rows write the lean form; _row_to_example reads both)."""
        import studio_data_kinds as dk
        with self._migrate_lock:
            if connection.execute("PRAGMA user_version").fetchone()[0] >= SCHEMA_VERSION:
                return  # another thread did it while this one waited
            connection.execute("PRAGMA journal_mode=WAL")  # kept in the file, so it is set here once; readers then never block the writer
            connection.execute(_CREATE_RECORDS)  # a no-op on an older table: its columns are added below
            connection.execute("CREATE INDEX IF NOT EXISTS records_dataset ON records(dataset_id)")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(records)")}
            for name, kind in (("kind", "TEXT"), ("holdout", "INTEGER")):
                if name not in columns:
                    connection.execute(f"ALTER TABLE records ADD COLUMN {name} {kind}")
            connection.commit()
            total = connection.execute("SELECT COUNT(*) FROM records WHERE kind IS NULL").fetchone()[0]
            done, last = 0, 0
            while True:
                batch = connection.execute("SELECT rowid, fingerprint, payload FROM records WHERE kind IS NULL AND rowid > ? "
                                           "ORDER BY rowid LIMIT ?", (last, MIGRATE_BATCH)).fetchall()
                if not batch:
                    break
                updates = []
                for rowid, key, payload in batch:
                    try:
                        kind = dk.classify(json.loads(payload))
                    except ValueError:
                        kind = "conversations"  # an unreadable payload fails where it is read, not here
                    updates.append((kind, int(dk.held_key(key)), rowid))
                connection.executemany("UPDATE records SET kind=?, holdout=? WHERE rowid=?", updates)
                connection.commit()
                done, last = done + len(batch), batch[-1][0]
                if self.progress:
                    self.progress(f"Updating your stored data ({done:,} of {total:,})", done / max(total, done))
            connection.execute(_PICK_INDEX)
            if self.progress:
                self.progress("Indexing your stored data", None)
            connection.execute(_STATS_INDEX)
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            connection.commit()

    @staticmethod
    def _where(dataset_id, query=""):
        if not query.strip():
            return "dataset_id=?", [dataset_id]
        escaped = query.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return "dataset_id=? AND (user LIKE ? ESCAPE '\\' OR assistant LIKE ? ESCAPE '\\' OR intent LIKE ? ESCAPE '\\')", [dataset_id] + [f"%{escaped}%"] * 3

    @staticmethod
    def _db_row(example):
        """One records row, in _COLUMNS order. The payload leaves out user/assistant (their columns hold them) and the
        in-memory kind/holdout stamps; _row_to_example puts the first two back."""
        import studio_data_kinds as dk
        key = fingerprint(example)
        payload = {k: v for k, v in example.items() if k not in ("kind", "holdout") and not (k in ("user", "assistant") and isinstance(v, str))}
        return (example["id"], example["datasetId"], key, json.dumps(payload, ensure_ascii=False),
                example.get("user") or "", example.get("assistant") or "", example.get("intent") or "general",
                example.get("quality") or "Review", len(example.get("user", "")) + len(example.get("assistant", "")),
                dk.classify(example), int(dk.held_key(key)))

    @staticmethod
    def _insert_rows(connection, rows, ignore=True):
        """The one INSERT into records. rows are _db_row tuples; returns how many were stored (ignore=True skips duplicates)."""
        return connection.executemany(f"INSERT {'OR IGNORE ' if ignore else ''}INTO records ({_COLUMNS}) VALUES ({','.join('?' * 11)})",
                                      rows).rowcount

    @staticmethod
    def _row_to_example(row):
        """(payload, user, assistant) as stored -> the example dict. Rows written before schema 2 still carry user/assistant in the payload."""
        example = json.loads(row[0])
        example.setdefault("user", row[1])
        example.setdefault("assistant", row[2])
        return example

    def set_dataset_metadata(self, dataset_id, **fields):
        dataset = self._dataset(dataset_id)
        if dataset is None:
            raise ValueError("Dataset no longer exists.")
        renamed = "name" in fields and fields["name"] != dataset.get("name")  # the name is in every row a summary shows
        dataset.update({key: value for key, value in fields.items() if key != "id"})
        if renamed:
            self._bump(dataset_id)
        self.save()

    def use_disk_storage(self, dataset_id):
        """Move one existing local dataset into SQLite without changing record IDs."""
        dataset = self._dataset(dataset_id)
        if dataset is None:
            raise ValueError("Choose a dataset first.")
        if dataset.get("storage") == "sqlite":
            return
        with self.lock:
            rows = self.examples(dataset_id)
        with self._connect({dataset_id}) as connection:  # the copy runs without the lock: a save on the UI thread never waits on it
            self._insert_rows(connection, (self._db_row(row) for row in rows))
            count = connection.execute("SELECT COUNT(*) FROM records WHERE dataset_id=?", (dataset_id,)).fetchone()[0]
            if count < len({fingerprint(row) for row in rows}):
                raise ValueError("Some records could not be moved to disk storage; nothing was changed.")
        def swap(state):
            if [fingerprint(row) for row in state["examples"] if row.get("datasetId") == dataset_id] != [fingerprint(row) for row in rows]:
                raise ValueError("The dataset changed while it was being moved to disk storage; try again.")
            state["examples"] = [row for row in state["examples"] if row.get("datasetId") != dataset_id]
            dataset.update(storage="sqlite", recordCount=count)
        self.apply(swap)

    def examples(self, dataset_id, limit=None, offset=0, query=""):
        dataset = self._dataset(dataset_id) or {}
        if dataset.get("storage") == "sqlite":
            where, parameters = self._where(dataset_id, query)
            with self._connect() as connection:
                cursor = connection.execute(f"SELECT {_READ} FROM records WHERE {where} ORDER BY rowid LIMIT ? OFFSET ?",
                                            parameters + [-1 if limit is None else max(0, int(limit)), max(0, int(offset))])
                return [self._row_to_example(row) for row in cursor]
        rows = [e for e in self.state["examples"] if e.get("datasetId") == dataset_id]
        if query.strip():
            needle = query.casefold().strip()
            rows = [e for e in rows if needle in " ".join(str(e.get(k, "")) for k in ("user", "assistant", "intent")).casefold()]
        return rows[max(0, int(offset)):None if limit is None else max(0, int(offset)) + max(0, int(limit))]

    def count(self, dataset_id, query=""):
        if (self._dataset(dataset_id) or {}).get("storage") == "sqlite":
            where, parameters = self._where(dataset_id, query)
            with self._connect() as connection:
                return connection.execute(f"SELECT COUNT(*) FROM records WHERE {where}", parameters).fetchone()[0]
        return len(self.examples(dataset_id, query=query))

    @slow("read a whole dataset from disk", when=lambda self, *args, **kwargs: not self.is_small())
    def iter_examples(self, dataset_id):
        if (self._dataset(dataset_id) or {}).get("storage") == "sqlite":
            with self._connect() as connection:
                cursor = connection.execute(f"SELECT {_READ} FROM records WHERE dataset_id=? ORDER BY rowid", (dataset_id,))
                while batch := cursor.fetchmany(2000):  # a read never holds more than one batch; WAL keeps writers unblocked
                    for row in batch:
                        yield self._row_to_example(row)
        else:
            yield from self.examples(dataset_id)

    @slow("read a whole dataset from disk", when=lambda self, *args, **kwargs: not self.is_small())
    def iter_kind(self, dataset_id, kind):
        """The examples of one dataset that are of `kind`, in storage order (disk datasets read the kind column; others filter in memory)."""
        import studio_data_kinds as dk
        if (self._dataset(dataset_id) or {}).get("storage") == "sqlite":
            with self._connect() as connection:
                cursor = connection.execute(f"SELECT {_READ} FROM records WHERE dataset_id=? AND kind=? ORDER BY rowid", (dataset_id, kind))
                while batch := cursor.fetchmany(2000):
                    for row in batch:
                        yield self._row_to_example(row)
        else:
            yield from (e for e in self.examples(dataset_id) if dk.classify(e) == kind)

    def kind_rows(self, dataset_id, kind, query="", after=0, limit=200, rules=()):
        """[(rowid, example)] of one disk dataset: up to `limit` rows of `kind` after rowid `after`, in storage order, whose user,
        assistant or intent hold `query` (SQLite folds case for ASCII letters only) or whose stored rule is one of `rules`."""
        where, parameters = self._where(dataset_id, query)
        if query.strip() and rules:
            where = f"({where} OR dataset_id=? AND ({' OR '.join(['payload LIKE ?'] * len(rules))}))"
            parameters = parameters + [dataset_id] + [f'%"rule": "{rule}"%' for rule in rules]
        with self._connect() as connection:
            cursor = connection.execute(f"SELECT rowid,{_READ} FROM records WHERE {where} AND kind=? AND rowid>? ORDER BY rowid LIMIT ?",
                                        parameters + [kind, after, limit])
            return [(row[0], self._row_to_example(row[1:])) for row in cursor]

    @slow("count a whole dataset on disk", when=lambda self, *args, **kwargs: not self.is_small())
    def kind_stats(self, dataset_id):
        """{kind: (records, characters, held back)} of one disk dataset from one GROUP BY on the kind column (read from the records_stats
        index, not the rows); None when some row has no kind yet (then the caller reads the rows)."""
        with self._connect() as connection:
            groups = connection.execute("SELECT kind,COUNT(*),COALESCE(SUM(characters),0),COALESCE(SUM(holdout),0) "
                                        "FROM records WHERE dataset_id=? GROUP BY kind", (dataset_id,)).fetchall()
        if any(kind is None for kind, *_ in groups):
            return None
        return {kind: tuple(rest) for kind, *rest in groups}

    @slow("read whole datasets from disk", when=lambda self, *args, **kwargs: not self.is_small())
    def pick(self, kind, holdout, dataset_ids):
        """{dataset_id: examples in storage order} of one kind and holdout flag among SQLite datasets: one query on the records_pick index."""
        found = {dataset_id: [] for dataset_id in dataset_ids}
        if not found:
            return found
        with self._connect() as connection:
            cursor = connection.execute(f"SELECT dataset_id,{_READ} FROM records INDEXED BY records_pick WHERE kind=? AND holdout=? AND dataset_id IN ({','.join('?' * len(found))}) "
                                        "ORDER BY rowid", (kind, int(holdout), *found))
            while batch := cursor.fetchmany(2000):
                for dataset_id, *row in batch:
                    found[dataset_id].append(self._row_to_example(row))
        return found

    def review_count(self, dataset_id):
        """How many examples of one dataset are still marked Review."""
        if (self._dataset(dataset_id) or {}).get("storage") == "sqlite":
            with self._connect() as connection:
                return connection.execute("SELECT COUNT(*) FROM records WHERE dataset_id=? AND quality='Review'", (dataset_id,)).fetchone()[0]
        return sum(1 for e in self.examples(dataset_id) if e.get("quality") == "Review")

    @slow("read a whole dataset from disk", when=lambda self, *args, **kwargs: not self.is_small())
    def sample(self, dataset_id, limit=20000, seed=42, reviewed_only=False, kind=None):
        """Uniform deterministic sampling with O(limit) record memory, including disk datasets. `kind` keeps one kind of example."""
        limit = max(0, int(limit))
        if not limit:
            return []
        rng = random.Random(seed)
        if (self._dataset(dataset_id) or {}).get("storage") != "sqlite":
            import studio_data_kinds as dk
            rows = [e for e in self.examples(dataset_id) if (not reviewed_only or e.get("quality") == "Good") and (kind is None or dk.classify(e) == kind)]
            return rng.sample(rows, min(limit, len(rows)))
        chosen = []
        with self._connect() as connection:
            suffix = (" AND quality='Good'" if reviewed_only else "") + (" AND kind=?" if kind else "")
            for index, (rowid,) in enumerate(connection.execute("SELECT rowid FROM records WHERE dataset_id=?" + suffix + " ORDER BY rowid", (dataset_id, *([kind] if kind else [])))):
                if index < limit:
                    chosen.append(rowid)
                else:
                    slot = rng.randint(0, index)
                    if slot < limit:
                        chosen[slot] = rowid
            values = {}
            for start in range(0, len(chosen), 800):
                ids = chosen[start:start + 800]
                for rowid, *row in connection.execute(f"SELECT rowid,{_READ} FROM records WHERE rowid IN (" + ",".join("?" for _ in ids) + ")", ids):
                    values[rowid] = self._row_to_example(row)
        rng.shuffle(chosen)
        return [values[i] for i in chosen]

    def get_example(self, example_id):
        found = next((e for e in self.state["examples"] if e["id"] == example_id), None)
        if found is not None:
            return found
        if not self.db_path.exists():
            return None
        with self._connect() as connection:
            row = connection.execute(f"SELECT {_READ} FROM records WHERE id=?", (example_id,)).fetchone()
            return self._row_to_example(row) if row else None

    def update_example(self, example, save=True):
        old = self.get_example(example.get("id"))
        if old is None:
            raise ValueError("Example no longer exists.")
        value = normalize({**old, **example}, old["datasetId"], old.get("source", "manual"))
        if value is None:
            raise ValueError("An example needs text or a user/assistant exchange.")
        value["id"] = old["id"]
        dataset_id = old["datasetId"]
        if (self._dataset(dataset_id) or {}).get("storage") == "sqlite":
            row = self._db_row(value)
            try:
                with self._connect({old.get("datasetId"), value["datasetId"]}) as connection:
                    connection.execute("UPDATE records SET dataset_id=?,fingerprint=?,payload=?,user=?,assistant=?,intent=?,quality=?,characters=?,kind=?,holdout=? WHERE id=?", row[1:] + (row[0],))
            except sqlite3.IntegrityError as exc:
                raise ValueError("This edit duplicates another example in this dataset.") from exc
        else:
            if any(e["id"] != value["id"] and fingerprint(e) == fingerprint(value) for e in self.examples(dataset_id)):
                raise ValueError("This edit duplicates another example in this dataset.")
            import studio_data_kinds as dk
            old.clear()
            old.update(dk.stamp(value, fingerprint(value)))
            self._bump(dataset_id)
        if save:
            self.save()
        return value

    def remove_examples(self, ids):
        ids = set(ids)
        touched = {e.get("datasetId") for e in self.state["examples"] if e["id"] in ids}
        self.state["examples"] = [e for e in self.state["examples"] if e["id"] not in ids]
        self._bump(*touched)
        if self.db_path.exists():
            with self._connect(touched) as connection:
                for example_id in ids:
                    row = connection.execute("SELECT dataset_id FROM records WHERE id=?", (example_id,)).fetchone()
                    if row:
                        touched.add(row[0])
                connection.executemany("DELETE FROM records WHERE id=?", ((i,) for i in ids))
                for dataset in self.state["datasets"]:
                    if dataset.get("storage") == "sqlite":
                        dataset["recordCount"] = connection.execute("SELECT COUNT(*) FROM records WHERE dataset_id=?", (dataset["id"],)).fetchone()[0]
        self.save()

    def mark_reviewed(self, ids):
        try:
            for example_id in ids:
                example = self.get_example(example_id)
                if example:
                    self.update_example({**example, "quality": "Good"}, save=False)
        finally:
            self.save()  # once, not once per id

    def audit(self, dataset_id):
        if (self._dataset(dataset_id) or {}).get("storage") == "sqlite":
            with self._connect() as connection:
                row = connection.execute("SELECT COUNT(*),COALESCE(SUM(quality!='Good'),0),COALESCE(SUM(characters),0),COUNT(DISTINCT intent) FROM records WHERE dataset_id=?", (dataset_id,)).fetchone()
            return {"records": row[0], "duplicates": 0, "review": row[1], "characters": row[2], "categories": row[3]}
        records = self.examples(dataset_id)
        if not records:
            return {"records": 0, "duplicates": 0, "review": 0, "characters": 0, "categories": 0}
        frame = pd.DataFrame(records).fillna("")
        texts = frame.get("user", pd.Series("", index=frame.index)).astype(str) + frame.get("assistant", pd.Series("", index=frame.index)).astype(str)
        return {"records": len(frame), "duplicates": int(pd.Series([fingerprint(e) for e in records]).duplicated().sum()),
                "review": int((frame.get("quality", pd.Series("Review", index=frame.index)) != "Good").sum()),
                "characters": int(texts.str.len().sum()), "categories": int(frame.get("intent", pd.Series("general", index=frame.index)).nunique())}

    def add(self, dataset_id, rows, source="manual"):
        dataset = self._dataset(dataset_id)
        if dataset is None:
            raise ValueError("Choose or create a dataset first.")
        if dataset.get("storage") != "sqlite" and hasattr(rows, "__len__") and len(rows) + self.count(dataset_id) > 10000:
            self.use_disk_storage(dataset_id)
        if dataset.get("storage") == "sqlite":
            added, duplicate, invalid = 0, 0, 0
            with self._connect({dataset_id}) as connection:  # one transaction for the whole import
                batch = []
                for row in rows:
                    example = normalize(row, dataset_id, source)
                    if example is None:
                        invalid += 1
                        continue
                    batch.append(self._db_row(example))
                    if len(batch) >= INSERT_BATCH:
                        stored = self._insert_rows(connection, batch)
                        added, duplicate, batch = added + stored, duplicate + len(batch) - stored, []
                stored = self._insert_rows(connection, batch)
                added, duplicate = added + stored, duplicate + len(batch) - stored
                dataset["recordCount"] = connection.execute("SELECT COUNT(*) FROM records WHERE dataset_id=?", (dataset_id,)).fetchone()[0]
            self.save()
            return added, duplicate, invalid
        import studio_data_kinds as dk
        known = {fingerprint(e) for e in self.examples(dataset_id)}
        added, duplicate, invalid = 0, 0, 0
        for row in rows:
            example = normalize(row, dataset_id, source)
            if example is None:
                invalid += 1
                continue
            key = fingerprint(example)
            if key in known:
                duplicate += 1
                continue
            known.add(key)
            self.state["examples"].append(dk.stamp(example, key))
            added += 1
        if added:
            self._bump(dataset_id)
        self.save()
        return added, duplicate, invalid

    def import_files(self, paths, dataset_id=None, describe=str):
        """One report line per file; `describe` turns an exception into the words shown after the file name."""
        report = []
        for path in paths:
            try:
                path = Path(path)
                large = path.stat().st_size > 10 * 1024 * 1024
                if large and path.suffix.lower() in {".jsonl", ".ndjson"}:
                    def streamed(source_path=path):
                        with source_path.open(encoding="utf-8-sig") as stream:
                            for line_number, line in enumerate(stream, 1):
                                if len(line) > 16_000_000:
                                    raise ValueError(f"{source_path.name}, line {line_number}: record exceeds 16 MB")
                                if line.strip():
                                    try:
                                        yield json.loads(line)
                                    except ValueError as exc:
                                        raise ValueError(f"{source_path.name}, line {line_number}: invalid JSON") from exc
                    rows = streamed()
                else:
                    rows = parse_file(path)
                    if not any(normalize(row) for row in rows):
                        raise ValueError("No chat pairs or text records found.")
                selected, created = self.state.get("selectedDatasetId"), dataset_id is None
                target = dataset_id or self.create_dataset(path.stem, storage="sqlite" if large else "json")
                try:
                    if large:
                        self.use_disk_storage(target)
                    added, duplicate, invalid = self.add(target, rows, "import")
                except BaseException:
                    if created:
                        self.remove_dataset(target)
                        if selected is None:
                            self.state.pop("selectedDatasetId", None)
                        else:
                            self.state["selectedDatasetId"] = selected
                        self.save()
                    raise
                report.append(f"{Path(path).name}: {added} added, {duplicate} duplicates, {invalid} invalid")
            except (OSError, ValueError, csv.Error) as exc:
                report.append(f"{Path(path).name}: {describe(exc)}")
        return report

    def remove_dataset(self, dataset_id):
        dataset = self._dataset(dataset_id)
        if self.db_path.exists():
            with self._connect({dataset_id}) as connection:
                connection.execute("DELETE FROM records WHERE dataset_id=?", (dataset_id,))
        if dataset:  # what a later dataset of the same id starts above, and what keeps `revision` from ever going down
            self.state["revBase"] = self.state.get("revBase", 0) + dataset.get("rev", 0) + 1
        self.state["datasets"] = [d for d in self.state["datasets"] if d["id"] != dataset_id]
        self.state["examples"] = [e for e in self.state["examples"] if e.get("datasetId") != dataset_id]
        registry = self.root / "data" / "catalog_registry.json"
        if registry.exists():
            entries = json.loads(registry.read_text(encoding="utf-8"))
            atomic_json(registry, [entry for entry in entries if entry.get("id") != dataset_id])
        if dataset and dataset.get("catalogId"):
            tombstones = self.root / "data" / "catalog_removed.json"
            removed = json.loads(tombstones.read_text(encoding="utf-8")) if tombstones.exists() else []
            atomic_json(tombstones, sorted(set(removed + [dataset_id])))
        self.save()

    def export(self, dataset_id, path):
        if not self.count(dataset_id):
            raise ValueError("This dataset is empty.")
        examples = self.iter_examples(dataset_id)
        path = Path(path)
        if path.suffix.lower() == ".csv":
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["user", "assistant", "text", "intent", "messages"])
                for e in examples:
                    turns = [] if e.get("text") else messages_of(e)
                    writer.writerow([e.get("user", ""), e.get("assistant", ""), e.get("text", ""), e.get("intent", ""), json.dumps(turns, ensure_ascii=False)])
        else:
            with path.open("w", encoding="utf-8") as stream:
                for e in examples:
                    record = {key: value for key, value in e.items() if key not in {"id", "datasetId", "updatedAt", "user", "assistant", "kind", "holdout"}}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def runs(self):
        runs = []
        if self.run_dir.exists():
            for path in self.run_dir.glob("*/run.json"):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    record["directory"] = str(path.parent)
                    runs.append(record)
                except (ValueError, OSError):
                    continue
        return sorted(runs, key=lambda r: r.get("createdAt", ""), reverse=True)

