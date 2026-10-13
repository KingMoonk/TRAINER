"""Free disk space: delete what training and iterating left behind, all of it or only what is older than N days.

History = run folders, checkpoints, versions, logs, exports, backups and stores of earlier designs. Your data
(datasets, downloads, the held-back set) is deleted only when asked. Settings and chat ratings are never touched.
"""
import shutil
import time
from pathlib import Path

from studio_versions import VersionStore

AGED = ("runs", "logs", "exports", "artifacts", "data/backups")          # each child is judged by its own age
WHOLE = ("data/weights", "data/curriculum", "data/versions.json", "data/architectures.sqlite", "data/passive.sqlite",
         "studio-error.log")                           # only "delete all" removes these
DATA = ("data/records.sqlite", "data/downloads", "data/heldout.json", "data/catalog_registry.json", "data/catalog_removed.json")


def _files(path):
    return [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]


def size(paths):
    return sum(f.stat().st_size for p in paths for f in _files(p))


def _newest(path):
    return max((f.stat().st_mtime for f in _files(path)), default=path.stat().st_mtime)


def plan(root, days=None, data=False, now=None):
    """Paths that clean() would delete. days=None means all history; data=True adds your data (never with days)."""
    root = Path(root)
    aged = [c for d in AGED if (root / d).is_dir() for c in (root / d).iterdir()] + list((root / "data").glob("workspace.before-*.json"))
    if days is not None:
        cutoff = (now or time.time()) - days * 86400
        return [p for p in aged if _newest(p) < cutoff]
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
