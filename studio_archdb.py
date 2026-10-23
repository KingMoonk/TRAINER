"""Versioned architecture store: revisions, weight files, backups and spec export.

Layout under the workspace root:
  data/architectures.sqlite     architectures, revisions, weight index, meta
  data/weights/<id>/vNNNN.npz   weight files (paths in the database are relative to the root)
  data/backups/arch-*.zip       online snapshot + referenced weights + manifest.json
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

from studio_arch import validate_spec
from studio_core import atomic_json, now, replace_file, uid
from studio_spec import is_model_spec, to_legacy_view, upgrade, validate_spec as validate_model_spec

BACKUP_FORMAT = "groundwork-arch-backup/1"
EXPORT_FORMAT = "groundwork-arch-export/1"
MODEL_EXPORT_FORMAT = "groundwork-model-spec-export/1"
CHUNK = 1 << 20
MAX_MEMBER = 4 << 30  # per backup member; a larger declared size is refused before extraction
MEMBER = re.compile(r"^(architectures\.sqlite|weights/[A-Za-z0-9_-]+/v\d{4}\.npz)\Z")
SUMMARY = """SELECT a.id, a.name, a.created_at, a.updated_at, a.head_revision,
    (SELECT COUNT(*) FROM revisions r WHERE r.architecture_id = a.id),
    (SELECT COUNT(*) FROM weight_versions w WHERE w.architecture_id = a.id)
    FROM architectures a"""
SUMMARY_KEYS = ("id", "name", "created_at", "updated_at", "head_revision", "revisions", "weight_versions")
WEIGHT_KEYS = ("version", "revision", "step", "loss", "source", "run_dir", "note", "sha256", "bytes", "created_at", "parent", "path")
SCHEMA = """
CREATE TABLE IF NOT EXISTS architectures (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT, updated_at TEXT,
    head_revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    architecture_id TEXT NOT NULL REFERENCES architectures(id) ON DELETE CASCADE,
    number INTEGER NOT NULL, spec_json TEXT NOT NULL, spec_hash TEXT NOT NULL,
    note TEXT, parent INTEGER, created_at TEXT, legacy_spec_json TEXT, UNIQUE(architecture_id, number));
CREATE TABLE IF NOT EXISTS weight_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    architecture_id TEXT NOT NULL REFERENCES architectures(id) ON DELETE CASCADE,
    version INTEGER NOT NULL, revision INTEGER NOT NULL, path TEXT NOT NULL,
    sha256 TEXT NOT NULL, bytes INTEGER, step INTEGER, loss REAL, source TEXT,
    run_dir TEXT, note TEXT, parent INTEGER, created_at TEXT, UNIQUE(architecture_id, version));
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '1');
"""


def spec_hash(spec):
    if is_model_spec(spec):
        spec = validate_model_spec(spec)
    text = json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_member(archive, name):
    digest = hashlib.sha256()
    with archive.open(name) as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_hashed(source, target):
    digest = hashlib.sha256()
    size = 0
    with open(source, "rb") as src, open(target, "wb") as dst:
        while chunk := src.read(CHUNK):
            digest.update(chunk)
            dst.write(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _write_member(archive, arcname, source):
    digest = hashlib.sha256()
    with open(source, "rb") as src, archive.open(arcname, "w", force_zip64=True) as dst:
        for chunk in iter(lambda: src.read(CHUNK), b""):
            digest.update(chunk)
            dst.write(chunk)
    return digest.hexdigest()


class ArchitectureStore:
    """`lock` serialises this instance only. Other instances or processes on the same root are
    coordinated by SQLite, except restore(), which must not run while they have the database open."""

    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / "data" / "architectures.sqlite"
        self.weights_dir = self.root / "data" / "weights"
        self.backup_dir = self.root / "data" / "backups"
        self.lock = threading.RLock()
        self._ready = False  # schema/WAL set up; reset by restore()
        self._migrated = False  # migrate_model_specs ran; reset by restore()
        self._verified = {}  # absolute weight path -> (mtime_ns, size, sha256)

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            if not self._ready:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.executescript(SCHEMA)
                self._ready = True
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _require(conn, architecture_id):
        row = conn.execute("SELECT head_revision FROM architectures WHERE id=?", (architecture_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown architecture: {architecture_id}")
        return row[0]

    @staticmethod
    def _revision_spec(conn, architecture_id, number):
        columns = {row[1] for row in conn.execute("PRAGMA table_info(revisions)")}
        legacy = ", legacy_spec_json" if "legacy_spec_json" in columns else ""
        row = conn.execute(f"SELECT spec_json{legacy} FROM revisions WHERE architecture_id=? AND number=?", (architecture_id, number)).fetchone()
        if row is None:
            raise ValueError(f"Unknown revision {number} for architecture {architecture_id}")
        raw = json.loads(row[0])
        return json.loads(row[1]) if len(row) > 1 and row[1] else to_legacy_view(raw) if is_model_spec(raw) else raw

    @staticmethod
    def _revision_stored(conn, architecture_id, number):
        row = conn.execute("SELECT spec_json FROM revisions WHERE architecture_id=? AND number=?", (architecture_id, number)).fetchone()
        if row is None:
            raise ValueError(f"Unknown revision {number} for architecture {architecture_id}")
        return json.loads(row[0])

    @staticmethod
    def _revision_model_spec(conn, architecture_id, number):
        return upgrade(ArchitectureStore._revision_stored(conn, architecture_id, number))

    @staticmethod
    def _commit_in(conn, architecture_id, spec, note):
        if not isinstance(spec, dict):
            raise ValueError("Architecture spec must be a dict.")
        head = ArchitectureStore._require(conn, architecture_id)
        version = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        legacy = None
        if version == "2":
            try:
                normalized = upgrade(spec)
            except ValueError as error:
                if is_model_spec(spec):
                    raise
                raise ValueError(f"This legacy architecture cannot be converted to the current format: {error}") from error
            legacy = json.dumps(spec, ensure_ascii=False) if not is_model_spec(spec) else None
            spec = normalized
        digest = spec_hash(spec)
        if head:
            (head_digest,) = conn.execute("SELECT spec_hash FROM revisions WHERE architecture_id=? AND number=?", (architecture_id, head)).fetchone()
            if head_digest == digest:
                return head
        number = head + 1
        stamp = now()
        values = (architecture_id, number, json.dumps(spec, ensure_ascii=False), digest, note or "", head or None, stamp)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(revisions)")}
        if "legacy_spec_json" in columns:
            conn.execute(
                "INSERT INTO revisions(architecture_id, number, spec_json, spec_hash, note, parent, created_at, legacy_spec_json) VALUES (?,?,?,?,?,?,?,?)",
                (*values, legacy),
            )
        else:
            conn.execute(
                "INSERT INTO revisions(architecture_id, number, spec_json, spec_hash, note, parent, created_at) VALUES (?,?,?,?,?,?,?)",
                values,
            )
        conn.execute("UPDATE architectures SET head_revision=?, updated_at=? WHERE id=?", (number, stamp, architecture_id))
        return number

    @staticmethod
    def _name(name):
        name = str(name).strip()
        if not name:
            raise ValueError("Enter an architecture name.")
        return name

    def create(self, name, spec, note="created"):
        name = self._name(name)
        if not isinstance(spec, dict):
            raise ValueError("Architecture spec must be a dict.")
        architecture_id = uid("arch")
        stamp = now()
        with self.lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO architectures(id, name, created_at, updated_at, head_revision) VALUES (?,?,?,?,0)", (architecture_id, name, stamp, stamp))
            self._commit_in(conn, architecture_id, spec, note)
        return architecture_id

    def list(self):
        with self.lock, self._connect() as conn:
            rows = conn.execute(SUMMARY + " ORDER BY a.rowid").fetchall()
        return [dict(zip(SUMMARY_KEYS, row)) for row in rows]

    def get(self, architecture_id):
        with self.lock, self._connect() as conn:
            row = conn.execute(SUMMARY + " WHERE a.id=?", (architecture_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown architecture: {architecture_id}")
            spec = self._revision_spec(conn, architecture_id, row[4])
        return {**dict(zip(SUMMARY_KEYS, row)), "spec": spec}

    def get_model_spec(self, architecture_id):
        """Read a normalized ModelSpec without changing the legacy UI's get() result."""
        with self.lock, self._connect() as conn:
            head = self._require(conn, architecture_id)
            return self._revision_model_spec(conn, architecture_id, head)

    def revision_model_spec(self, architecture_id, number):
        with self.lock, self._connect() as conn:
            self._require(conn, architecture_id)
            return self._revision_model_spec(conn, architecture_id, number)

    def migrate_model_specs(self):
        """Atomically convert valid legacy revisions; preserve their exact UI JSON and all links.

        Incomplete historical records remain untouched and are reported. Repeating the migration
        makes no changes, including to revision numbers, hashes, and timestamps.
        """
        with self.lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(revisions)")}
            if "legacy_spec_json" not in columns:
                conn.execute("ALTER TABLE revisions ADD COLUMN legacy_spec_json TEXT")
            migrated, skipped = 0, 0
            rows = conn.execute("SELECT id, spec_json FROM revisions ORDER BY id").fetchall()
            for revision_id, source in rows:
                raw = json.loads(source)
                try:
                    normalized = upgrade(raw)
                except ValueError:
                    skipped += 1
                    continue
                if is_model_spec(raw):
                    if raw != normalized:
                        conn.execute("UPDATE revisions SET spec_json=?, spec_hash=? WHERE id=?",
                                     (json.dumps(normalized, ensure_ascii=False), spec_hash(normalized), revision_id))
                        migrated += 1
                    continue
                conn.execute("UPDATE revisions SET spec_json=?, spec_hash=?, legacy_spec_json=? WHERE id=?",
                             (json.dumps(normalized, ensure_ascii=False), spec_hash(normalized), source, revision_id))
                migrated += 1
            conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
        self._migrated = True
        return {"migrated": migrated, "skipped": skipped}

    def create_model_spec(self, name, spec, note="created"):
        spec = validate_model_spec(spec)
        if not self._migrated:
            self.migrate_model_specs()
        return self.create(name, spec, note)

    def commit_model_spec(self, architecture_id, spec, note=""):
        spec = validate_model_spec(spec)
        if not self._migrated:
            self.migrate_model_specs()
        return self.commit(architecture_id, spec, note)

    def rename(self, architecture_id, name):
        name = self._name(name)
        with self.lock, self._connect() as conn:
            self._require(conn, architecture_id)
            conn.execute("UPDATE architectures SET name=?, updated_at=? WHERE id=?", (name, now(), architecture_id))

    def delete(self, architecture_id):
        with self.lock:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._require(conn, architecture_id)
                conn.execute("DELETE FROM architectures WHERE id=?", (architecture_id,))
            folder = self.weights_dir / architecture_id
            if folder.exists():
                shutil.rmtree(folder)

    def commit(self, architecture_id, spec, note=""):
        with self.lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self._commit_in(conn, architecture_id, spec, note)

    def revisions(self, architecture_id):
        with self.lock, self._connect() as conn:
            self._require(conn, architecture_id)
            rows = conn.execute(
                "SELECT number, note, created_at, spec_hash, parent FROM revisions WHERE architecture_id=? ORDER BY number DESC",
                (architecture_id,),
            ).fetchall()
        return [dict(zip(("number", "note", "created_at", "spec_hash", "parent"), row)) for row in rows]

    def revision(self, architecture_id, number):
        with self.lock, self._connect() as conn:
            self._require(conn, architecture_id)
            return self._revision_spec(conn, architecture_id, number)

    def checkout(self, architecture_id, number, note=None):
        with self.lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require(conn, architecture_id)
            spec = self._revision_stored(conn, architecture_id, number)
            return self._commit_in(conn, architecture_id, spec, note or f"restored from r{number}")

    def add_weights(self, architecture_id, npz_path, revision=None, step=None, loss=None, source="train", run_dir=None, note="", parent=None):
        source_path = Path(npz_path)
        if not source_path.is_file():
            raise ValueError(f"Weights file not found: {npz_path}")
        with self.lock, self._connect() as conn:
            self._require(conn, architecture_id)
        folder = self.weights_dir / architecture_id
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / f".incoming.{uid('tmp')}"
        destination = None
        try:
            # Copy and hash (possibly ~1 GB) before taking the lock or the write transaction.
            digest, size = _copy_hashed(source_path, temporary)
            with self.lock, self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                head = self._require(conn, architecture_id)
                revision = head if revision is None else int(revision)
                if conn.execute("SELECT 1 FROM revisions WHERE architecture_id=? AND number=?", (architecture_id, revision)).fetchone() is None:
                    raise ValueError(f"Unknown revision {revision} for architecture {architecture_id}")
                if parent is not None and conn.execute("SELECT 1 FROM weight_versions WHERE architecture_id=? AND version=?", (architecture_id, parent)).fetchone() is None:
                    raise ValueError(f"Unknown parent weight version {parent}")
                version = conn.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM weight_versions WHERE architecture_id=?", (architecture_id,)).fetchone()[0]
                destination = folder / f"v{version:04d}.npz"
                relative = f"{self.weights_dir.relative_to(self.root).as_posix()}/{architecture_id}/{destination.name}"
                replace_file(temporary, destination)
                conn.execute(
                    """INSERT INTO weight_versions(architecture_id, version, revision, path, sha256, bytes, step, loss, source,
                       run_dir, note, parent, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (architecture_id, version, revision, relative, digest, size,
                     None if step is None else int(step), None if loss is None else float(loss),
                     source, run_dir, note or "", parent, now()),
                )
            # The COMMIT runs when the block above exits, so a failed commit is cleaned up below too.
        except BaseException:
            temporary.unlink(missing_ok=True)
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise
        return version

    def weights(self, architecture_id):
        with self.lock, self._connect() as conn:
            self._require(conn, architecture_id)
            rows = conn.execute(
                "SELECT version, revision, step, loss, source, run_dir, note, sha256, bytes, created_at, parent, path "
                "FROM weight_versions WHERE architecture_id=? ORDER BY version DESC",
                (architecture_id,),
            ).fetchall()
        return [dict(zip(WEIGHT_KEYS, row)) for row in rows]

    def weight_path(self, architecture_id, version):
        with self.lock, self._connect() as conn:
            self._require(conn, architecture_id)
            row = conn.execute("SELECT path, sha256 FROM weight_versions WHERE architecture_id=? AND version=?", (architecture_id, version)).fetchone()
            if row is None:
                raise ValueError(f"Unknown weight version {version} for architecture {architecture_id}")
        absolute = self.root.joinpath(*row[0].split("/"))
        try:
            stat = absolute.stat()
        except OSError:
            stat = None
        if stat is not None:
            key = (stat.st_mtime_ns, stat.st_size)
            cached = self._verified.get(str(absolute))
            if cached is None or cached[:2] != key:
                cached = (*key, _hash_file(absolute))
                self._verified[str(absolute)] = cached
        if stat is None or cached[2] != row[1]:
            raise ValueError(f"Weight version {version} of {architecture_id} is corrupted or missing.")
        return str(absolute)

    def _backup_name(self, label):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        candidate = self.backup_dir / f"arch-{stamp}-{label}.zip"
        number = 2
        while candidate.exists():
            candidate = self.backup_dir / f"arch-{stamp}-{label}-{number}.zip"
            number += 1
        return candidate

    def backup(self, label="manual", include_weights=True):
        label = re.sub(r"[^A-Za-z0-9_-]+", "-", str(label)).strip("-") or "manual"
        with self.lock:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            snapshot = self.backup_dir / f".snapshot-{uid('tmp')}.sqlite"
            with self._connect() as conn:
                count = conn.execute("SELECT COUNT(*) FROM architectures").fetchone()[0]
                weights = []
                if include_weights:
                    weights = conn.execute("SELECT architecture_id, version, path, sha256 FROM weight_versions ORDER BY architecture_id, version").fetchall()
                target = sqlite3.connect(snapshot)
                try:
                    conn.backup(target)
                finally:
                    target.close()
            final = self._backup_name(label)
            partial = final.with_name(final.name + ".partial")
            files, skipped = {}, []
            try:
                with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
                    files["architectures.sqlite"] = _write_member(archive, "architectures.sqlite", snapshot)
                    for architecture_id, version, relative, expected in weights:
                        arcname = f"weights/{architecture_id}/v{version:04d}.npz"
                        source = self.root.joinpath(*relative.split("/"))
                        # Missing or corrupt weights are skipped and recorded, so one bad file cannot block backup or restore.
                        if not source.is_file() or _hash_file(source) != expected:
                            skipped.append(arcname)
                            continue
                        files[arcname] = _write_member(archive, arcname, source)
                    manifest = {"format": BACKUP_FORMAT, "created_at": now(), "label": label,
                                "include_weights": bool(include_weights), "files": files, "architectures": count,
                                "skipped_weights": skipped}
                    archive.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
                replace_file(partial, final)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
            finally:
                snapshot.unlink(missing_ok=True)
        return str(final)

    def backups(self):
        with self.lock:
            if not self.backup_dir.is_dir():
                return []
            found = []
            for path in self.backup_dir.glob("arch-*.zip"):
                try:
                    with zipfile.ZipFile(path) as archive:
                        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
                    if not isinstance(manifest, dict) or manifest.get("format") != BACKUP_FORMAT:
                        continue
                    found.append({"path": str(path), "label": manifest.get("label", ""),
                                  "created_at": manifest.get("created_at", ""), "bytes": path.stat().st_size,
                                  "skipped_weights": len(manifest.get("skipped_weights", []))})
                except (OSError, KeyError, ValueError, AttributeError, zipfile.BadZipFile):
                    continue
        return sorted(found, key=lambda entry: (entry["created_at"], entry["path"]), reverse=True)

    @staticmethod
    def _verify_backup(archive):
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Backup archive has duplicate members.")
        for info in archive.infolist():
            if info.filename != "manifest.json" and not MEMBER.match(info.filename):
                raise ValueError(f"Unsafe or unexpected member in backup: {info.filename!r}")
            if info.file_size > MAX_MEMBER:
                raise ValueError(f"Backup member {info.filename} is too large ({info.file_size} bytes).")
        try:
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        except (KeyError, ValueError) as error:
            raise ValueError("Backup manifest is missing or unreadable.") from error
        if not isinstance(manifest, dict) or manifest.get("format") != BACKUP_FORMAT:
            raise ValueError("Not a groundwork architecture backup.")
        files = manifest.get("files")
        if not isinstance(files, dict) or "architectures.sqlite" not in files:
            raise ValueError("Backup manifest lists no database.")
        if set(files) != set(names) - {"manifest.json"}:
            raise ValueError("Backup members do not match its manifest.")
        for name, expected in files.items():
            if _hash_member(archive, name) != expected:
                raise ValueError(f"Backup file {name} is corrupted; nothing was changed.")
        return manifest, files

    @staticmethod
    def _extract(archive, files, staging):
        for name in files:
            target = staging.joinpath(*name.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)

    @staticmethod
    def _check_database(path):
        connection = sqlite3.connect(path)
        try:
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Backup database failed its integrity check.")
            return connection.execute("SELECT COUNT(*) FROM architectures").fetchone()[0]
        except sqlite3.DatabaseError as error:
            raise ValueError("Backup database is not a valid architecture store.") from error
        finally:
            connection.close()

    def _swap(self, staging, include_weights=True):  # noqa: ARG002 - the folder always follows the backup now
        if self.path.exists():
            # Fold the WAL into the main file and leave WAL mode so SQLite removes -wal/-shm itself;
            # this fails with "database is locked" if another connection still has it open.
            connection = sqlite3.connect(self.path, timeout=30)
            try:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                connection.execute("PRAGMA journal_mode=DELETE")
            finally:
                connection.close()
        # The weights folder always follows the backup: a weights-less backup leaves it empty, so stale
        # files cannot masquerade as (or be overwritten by) the restored database's versions.
        old = None
        if self.weights_dir.exists():
            old = self.path.parent / f"weights-old-{uid('tmp')}"
            replace_file(self.weights_dir, old)
        try:
            incoming = staging / "weights"
            if incoming.exists():
                replace_file(incoming, self.weights_dir)
            else:
                self.weights_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("-wal", "-shm"):
                self.path.with_name(self.path.name + suffix).unlink(missing_ok=True)
            replace_file(staging / "architectures.sqlite", self.path)
        except BaseException:
            if old is not None:  # put the old weights back so they still match the old database
                shutil.rmtree(self.weights_dir, ignore_errors=True)
                replace_file(old, self.weights_dir)
            raise
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)

    def restore(self, zip_path):
        path = Path(zip_path)
        if not path.is_file():
            raise ValueError(f"Backup not found: {zip_path}")
        with self.lock:
            try:
                archive = zipfile.ZipFile(path)
            except zipfile.BadZipFile as error:
                raise ValueError(f"Not a valid backup archive: {path}") from error
            with archive:
                manifest, files = self._verify_backup(archive)
                include_weights = bool(manifest.get("include_weights", True))
                pre_restore = self.backup("pre-restore")
                staging = self.path.parent / f"restore-{uid('tmp')}"
                try:
                    staging.mkdir(parents=True)
                    self._extract(archive, files, staging)
                    count = self._check_database(staging / "architectures.sqlite")
                    self._swap(staging, include_weights)
                finally:
                    shutil.rmtree(staging, ignore_errors=True)
                    self._ready = self._migrated = False
                    self._verified.clear()
        return {"restored": str(path), "pre_restore_backup": pre_restore,
                "architectures": count, "weights_restored": include_weights}

    def export_spec(self, architecture_id, path):
        with self.lock:
            record = self.get(architecture_id)
        atomic_json(path, {"format": EXPORT_FORMAT, "name": record["name"], "spec": record["spec"]})
        return str(path)

    def import_spec(self, path, name=None):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read spec file: {path}") from error
        if not isinstance(data, dict) or data.get("format") != EXPORT_FORMAT or not isinstance(data.get("spec"), dict):
            raise ValueError("Not a groundwork architecture export.")
        validate_spec(data["spec"])
        return self.create(name or data.get("name") or "Imported network", data["spec"], note=f"imported from {Path(path).name}")

    def export_model_spec(self, architecture_id, path):
        record = self.get(architecture_id)
        atomic_json(path, {"format": MODEL_EXPORT_FORMAT, "name": record["name"],
                           "spec": self.get_model_spec(architecture_id)})
        return str(path)

    def import_model_spec(self, path, name=None):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read spec file: {path}") from error
        if not isinstance(data, dict) or data.get("format") != MODEL_EXPORT_FORMAT:
            raise ValueError("Not a groundwork ModelSpec export.")
        spec = validate_model_spec(data.get("spec"))
        return self.create_model_spec(name or data.get("name") or "Imported network", spec,
                                      note=f"imported from {Path(path).name}")
