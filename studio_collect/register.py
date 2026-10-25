"""Hand collected shards to the workspace: each group becomes a text set in records.sqlite that Pretraining and Post-training read.

Runs inside the app (the process that owns the workspace), never in the collector: two processes saving workspace.json would
overwrite each other's changes. Rows are written the way studio_catalog.register_download writes them (`_db_row`, `_insert_rows`),
in one transaction per group, then the shard files are deleted. The unique (dataset, fingerprint) key drops a record registered twice,
so a crash between the write and the delete is harmless.
"""
from __future__ import annotations

import json
from pathlib import Path

from studio_core import atomic_json, now, read_json, uid
from studio_collect.config import collect_dir
from studio_collect.pipeline import shard_dir

BATCH = 5000


def dataset_id(group):
    return f"collect_{group}"


def dataset_name(group):
    """"topic-science" -> "Collected · science"; "stackexchange-math.stackexchange.com" -> "Collected · Stack Exchange math.stackexchange.com"."""
    for prefix, label in (("topic-", ""), ("stackexchange-", "Stack Exchange "), ("github", "GitHub"), ("web-", "Web ")):
        if group.startswith(prefix):
            return f"Collected · {label}{group[len(prefix):].replace('-', ' ')}".strip(" ·")
    return f"Collected · {group.replace('-', ' ')}"


def pending(root):
    """{group: [shard paths]} of finished shards (a .part file is still being written)."""
    base = shard_dir(root)
    if not base.is_dir():
        return {}
    found = {}
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        files = sorted(folder.glob("*.jsonl"))
        if files:
            found[folder.name] = files
    return found


def totals(root):
    try:
        return read_json(collect_dir(root) / "registered.json")
    except (OSError, ValueError):
        return {}


def register(workspace, emit=lambda text: None):
    """Add every finished shard to its set. Returns {"added", "duplicate", "shards", "sets": {dataset id: records added}}."""
    root = Path(workspace.root)
    report = {"added": 0, "duplicate": 0, "shards": 0, "sets": {}}
    saved = totals(root)
    for group, files in pending(root).items():
        target = dataset_id(group)
        if workspace._dataset(target) is None:
            metadata = {"id": target, "name": dataset_name(group), "description": "Collected passively from sources whose terms allow it. Every record keeps its source URL, licence, retrieval time and attribution (provenance).",
                        "createdAt": now(), "storage": "sqlite", "recordCount": 0, "collected": True, "license": "per record (see provenance)", "rev": 0}
            with workspace.lock:
                metadata["rev"] = workspace.next_rev()
                workspace.state["datasets"].append(metadata)
                workspace.save()
        added = duplicate = 0
        stamp = now()
        with workspace._connect({target}) as connection:
            for path in files:
                batch = []
                for line in path.read_text(encoding="utf-8").split("\n"):  # not splitlines(): JSON leaves U+2028 unescaped inside strings
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row.update(id=uid("ex"), datasetId=target, updatedAt=stamp)
                    batch.append(workspace._db_row(row))
                    if len(batch) >= BATCH:
                        stored = workspace._insert_rows(connection, batch)
                        added, duplicate, batch = added + stored, duplicate + len(batch) - stored, []
                stored = workspace._insert_rows(connection, batch)
                added, duplicate = added + stored, duplicate + len(batch) - stored
            count = connection.execute("SELECT COUNT(*) FROM records WHERE dataset_id=?", (target,)).fetchone()[0]
        for path in files:        # after the commit: a crash before this line re-registers the same rows, which the unique key drops
            path.unlink(missing_ok=True)
        with workspace.lock:
            workspace._dataset(target)["recordCount"] = count
            workspace.save()
        entry = saved.setdefault(target, {"name": dataset_name(group), "records": 0, "added": 0, "duplicate": 0})
        entry.update(records=count, added=entry["added"] + added, duplicate=entry["duplicate"] + duplicate)
        report["added"] += added
        report["duplicate"] += duplicate
        report["shards"] += len(files)
        report["sets"][target] = added
        emit(f"Added {added:,} collected records to {dataset_name(group)}")
    if report["shards"]:
        atomic_json(collect_dir(root) / "registered.json", saved)
    return report
