"""Free disk space: delete what training and iterating left behind, all of it or only what is older than N days.

History = run folders, checkpoints, versions, logs, exports, backups, token shards and stores of earlier designs. Your data
(datasets, downloads, the held-back set) is deleted only when asked. Settings and chat ratings are never touched.
"""
import re
import shutil
import time
from pathlib import Path

import studio_dataset_snapshot as snapshots
from studio_versions import VersionStore

AGED = ("runs", "logs", "exports", "artifacts", "data/backups", "data/shards")          # each child is judged by its own age
WHOLE = ("data/weights", "data/pulled", "data/curriculum", "data/versions.json", "data/architectures.sqlite",
         "data/passive.sqlite",  # the removed passive training's lessons (#486): never read, but "delete all" still removes them
         "studio-error.log")                           # only "delete all" removes these
DATA = ("data/records.sqlite", "data/records.sqlite-wal", "data/records.sqlite-shm", "data/downloads", "data/heldout.json", "data/catalog_registry.json", "data/catalog_removed.json")


def _files(path):
    return [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]


def size(paths):
    """Bytes deleting `paths` frees. A file with several hard links (a dataset snapshot shared by run folders) frees its bytes
    only when every one of its links is among the paths."""
    total, linked = 0, {}
    for p in paths:
        for f in _files(p):
            stat = f.stat()
            if stat.st_nlink <= 1:
                total += stat.st_size
            else:
                found = linked.setdefault((stat.st_dev, stat.st_ino), [stat.st_size, stat.st_nlink, 0])
                found[2] += 1
    return total + sum(n for n, links, seen in linked.values() if seen >= links)


def _newest(path):
    return max((f.stat().st_mtime for f in _files(path)), default=path.stat().st_mtime)


def plan(root, days=None, data=False, now=None):
    """Paths that clean() would delete. days=None means all history; data=True adds your data (never with days)."""
    root = Path(root)
    aged = [c for d in AGED if (root / d).is_dir() for c in (root / d).iterdir()] + list((root / "data").glob("workspace.before-*.json"))
    if days is not None:
        cutoff = (now or time.time()) - days * 86400
        store = root / "runs" / snapshots.FOLDER  # the store is judged file by file: one a recent run still links to stays
        old = [p for p in aged if p != store and _newest(p) < cutoff]
        return old + [path for path, _, links in snapshots.listing(store) if links <= 1 and path.stat().st_mtime < cutoff]
    return [p for p in aged + [root / w for w in WHOLE] + ([root / d for d in DATA] if data else []) if p.exists()]


def clean(workspace, days=None, data=False, now=None):
    """Delete plan(); then drop every reference to what is gone. Returns (items deleted, bytes freed)."""
    root = Path(workspace.root)
    paths = plan(root, days, data, now)
    freed = size(paths)
    for p in paths:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    store = VersionStore(root)
    if store.path.exists():  # versions survive an age-based clean; they forget runs and checkpoints that are gone
        def there(p):
            return bool(p) and (Path(p) if Path(p).is_absolute() else root / p).exists()
        versions = store.versions()
        for v in versions:
            v["runs"] = [r for r in v.get("runs", []) if there(r["dir"] if isinstance(r, dict) else r)]
            if not there(v.get("best_checkpoint")):
                v["best_checkpoint"] = None
                if v.get("status") == "trained":
                    v["status"] = "saved"
        store._write(versions)

    def forget(state):
        state["runs"] = [r for r in state.get("runs", []) if days is not None and (root / "runs" / str(r.get("id", ""))).exists()]
        if days is None:
            state["generated"] = []
        if data and days is None:
            state.update(datasets=[], examples=[], selectedDatasetId=None)
    workspace.apply(forget)
    return len(paths), freed


# What runs/ keeps that no longer serves anything --------------------------------------------------------------------------

ASIDE = re.compile(r"^queue_(?P<job>.+?)-before-restart(?:-\d+)?$")  # studio_jobs.continue_here moves an interrupted attempt here
ACTIVE = ("waiting", "held", "running", "stopped")  # a stopped job can still be continued from its checkpoint
JOB_PATHS = ("output_dir", "resume", "init_from", "previous_best")  # folders a job that has yet to finish reads or writes
UNFINISHED = 3600  # seconds: a half-written snapshot older than this is a crashed write


def _protected(root, jobs):
    """Names of the folders in runs/ nothing may be offered from: where a job that has not finished writes or continues from
    (the checkpoint a waiting job will resume is in an interrupted attempt's folder), and every run and best checkpoint a version
    keeps."""
    runs = (root / "runs").resolve()
    names = set()

    def mark(value):
        if not value:
            return
        path = Path(value) if Path(value).is_absolute() else root / value
        try:
            parts = path.resolve().relative_to(runs).parts
        except (OSError, ValueError):
            return
        if parts:
            names.add(parts[0])
    for job in jobs:
        if job["status"] in ACTIVE:
            names.update((f"queue_{job['id']}", f"compare_{job['id']}"))  # where a job writes when its params do not say
            for key in JOB_PATHS:
                mark(job["params"].get(key))
    store = VersionStore(root)
    for version in store.versions() if store.path.exists() else []:
        for run in version.get("runs", []):
            mark(run["dir"] if isinstance(run, dict) else run)
        mark(version.get("best_checkpoint"))
    return names


def _row(root, path, kind, name, belongs_to):
    freed = size([path])
    return {"id": path.relative_to(root).as_posix(), "kind": kind, "name": name, "belongs_to": belongs_to,
            "bytes": freed, "shared": sum(f.stat().st_size for f in _files(path)) - freed}


def leftovers(root, jobs):
    """What can be removed without losing a result, as [{id, kind, name, belongs_to, bytes, shared}], largest first:
    "interrupted" the folders of attempts the app moved aside when it restarted a job (queue_<id>-before-restart[-n]); "failed"
    the folder of a job that failed; "snapshot" a dataset snapshot no run folder links to any more. `bytes` is what removing it
    frees, `shared` what it holds that other runs still link to. `jobs` are {id, label, status, params} (the queue's); a running
    job's folder, the checkpoint a waiting job will continue from and everything a version keeps are never listed."""
    root = Path(root)
    runs, by_id, protected, out = root / "runs", {job["id"]: job for job in jobs}, _protected(root, jobs), []
    for child in sorted(runs.iterdir()) if runs.is_dir() else []:
        if not child.is_dir() or child.name in protected or child.name == snapshots.FOLDER:
            continue
        aside = ASIDE.match(child.name)
        owner = by_id.get(aside["job"] if aside else child.name.removeprefix("queue_") if child.name.startswith("queue_") else None)
        if aside:
            out.append(_row(root, child, "interrupted", child.name, owner["label"] if owner else "A job that is no longer in the queue"))
        elif owner and owner["status"] == "failed":
            out.append(_row(root, child, "failed", child.name, owner["label"]))
    store, cutoff = runs / snapshots.FOLDER, time.time() - UNFINISHED
    for path, _, links in snapshots.listing(store):
        if links <= 1:
            out.append(_row(root, path, "snapshot", path.stem[:12], "No run uses this training data"))
    for path in store.glob(".tmp-*") if store.is_dir() else []:
        if path.stat().st_mtime < cutoff:
            out.append(_row(root, path, "snapshot", "Unfinished write", "A dataset snapshot that was never completed"))
    return sorted(out, key=lambda row: -row["bytes"])


def remove_leftovers(root, ids, jobs):
    """Delete the leftovers whose id is in `ids` and still in leftovers(): an id that is not (protected, already gone, made up)
    is ignored, so a stale page can never remove anything else. Returns (items removed, bytes freed)."""
    root = Path(root)
    wanted = set(ids)
    chosen = [row for row in leftovers(root, jobs) if row["id"] in wanted]
    for row in chosen:
        path = root / row["id"]
        shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)
    return len(chosen), sum(row["bytes"] for row in chosen)
