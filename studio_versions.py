"""Design versions: data/versions.json, a plain-words diff between specs, and the status life cycle.

Kept independent of studio_archdb (that store is keyed by architecture revisions); a version only
references run directories and checkpoint paths. Version = {number, name, change, parent, spec, created_at,
runs, best_checkpoint, status, scorecard, verdict, prediction, published, train_device, score_device}. prediction is the guess made on saving ({dimension: direction}, Build your model).
train_device is the device kind (cuda, mps, cpu) that trained the best checkpoint, score_device the one that scored it on the held-back text;
both are absent until known. published (absent until used) lists the version-store keys this version was published under, or was pulled from (studio_publish).
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import time
from pathlib import Path

from studio_core import replace_file
from studio_spec import block_at, structure_key, validate_spec

STATUSES = ("draft", "saved", "queued", "training", "trained")
NEXT = {"draft": {"saved"}, "saved": {"queued"}, "queued": {"saved", "training"},
        "training": {"trained", "queued", "saved"}, "trained": {"queued"}}


class NoCheckpoint(ValueError):
    """A version has no trained weights yet. The message says which one and where to fix it; it is already written for people."""

    def __init__(self, version=None):
        self.version = version
        super().__init__(f"Version {version} has not trained yet. Queue it from Build your model, or pick a trained version."
                         if version else "No version has trained yet. Queue one from Build your model.")


_WORDS = {"byte": "letters", "character": "letters", "bpe_gpt2": "large word list", "bpe_custom": "small word list", "bpe_fast": "small word list",
          "word": "whole words"}


def _facts(spec):
    spec = validate_spec(spec)
    d = spec["dims"]["d_model"]
    block = block_at(spec, 0)
    return dict(blocks=spec["dims"]["n_layer"], heads=block["mixer"]["config"].get("heads"), width=d, context=spec["dims"]["n_ctx"],
                room=(block["channel"]["config"].get("d_ff") or 4 * d) / d, words=_WORDS.get(spec["tokenizer"]["id"], spec["tokenizer"]["id"]))


def _room_label(ratio):
    return "wide" if ratio > 4 else "narrow" if ratio < 4 else "normal"


OTHER_PARTS = "other parts changed"


def spec_diff(old, new):
    """Plain-words changes from `old` to `new` as a list of strings (empty when nothing changed)."""
    a, b = _facts(old), _facts(new)
    out = []
    if a["blocks"] != b["blocks"]:
        out.append(f"{'more' if b['blocks'] > a['blocks'] else 'fewer'} blocks ({a['blocks']} → {b['blocks']})")
    if a["heads"] != b["heads"] and None not in (a["heads"], b["heads"]):  # a mixer without heads is a module change
        out.append(f"look-back heads {a['heads']} → {b['heads']}")
    if a["room"] != b["room"]:
        out.append(f"{_room_label(b['room'])} thinking room")
    if a["width"] != b["width"]:
        out.append(f"{'wider' if b['width'] > a['width'] else 'narrower'} words ({a['width']} → {b['width']})")
    if a["context"] != b["context"]:
        out.append(f"{'longer' if b['context'] > a['context'] else 'shorter'} reading window ({a['context']} → {b['context']})")
    if a["words"] != b["words"]:
        out.append(b["words"])
    if not out and structure_key(old) != structure_key(new):
        out.append(OTHER_PARTS)
    return out


def change_text(old, new):
    return ", ".join(spec_diff(old, new)) or "no change"


class VersionStore:
    def __init__(self, root):
        self.path = Path(root) / "data" / "versions.json"
        self._lock = threading.RLock()

    def versions(self):
        with self._lock:
            return json.loads(self.path.read_text())["versions"] if self.path.exists() else []

    def _write(self, versions):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            json.dump({"versions": versions}, handle, indent=1)
        replace_file(tmp, self.path)  # a reader with the file open fails a bare os.replace on Windows

    def get(self, number):
        for v in self.versions():
            if v["number"] == number:
                return v
        raise KeyError(f"No version {number}.")

    def current_version(self):
        versions = self.versions()
        return versions[-1] if versions else None

    def best_version(self):
        """Lowest held-back bits/char (scorecard["bpc"]) among versions that have a checkpoint."""
        held = [v for v in self.versions() if v["best_checkpoint"] and isinstance(v["scorecard"].get("bpc"), (int, float))]
        return min(held, key=lambda v: v["scorecard"]["bpc"], default=None)

    def save_version(self, spec, change=None, prediction=None):
        """Save `spec` as the next version (status "saved"); `change` defaults to the plain-words diff from its parent.
        `prediction` is what the person guessed the change would do ({"error": "lower", ...}); none is stored as {}."""
        spec = validate_spec(spec)
        with self._lock:
            versions = self.versions()
            parent = versions[-1] if versions else None
            if change is None:
                change = change_text(parent["spec"], spec) if parent else "first design"
            version = dict(number=len(versions) + 1, name=f"Version {len(versions) + 1}", change=change,
                           parent=parent["number"] if parent else None, spec=spec,
                           created_at=time.strftime("%Y-%m-%dT%H:%M:%S"), runs=[], best_checkpoint=None,
                           status="saved", scorecard={}, verdict=None, prediction=dict(prediction or {}))
            self._write(versions + [version])
            return copy.deepcopy(version)

    def add_pulled(self, spec, name, change, checkpoint, key, train_device=None):
        """A version another machine published, as a trained version here: its design, and `checkpoint` (the pulled file) as its weights.
        Its scorecard stays empty, because the numbers measured there were on that machine's held-back text. Returns the new row."""
        spec = validate_spec(spec)
        with self._lock:
            versions = self.versions()
            version = dict(number=len(versions) + 1, name=name, change=change, parent=None, spec=spec,
                           created_at=time.strftime("%Y-%m-%dT%H:%M:%S"), runs=[], best_checkpoint=str(checkpoint),
                           status="trained", scorecard={}, verdict=None, prediction={}, published=[key],
                           **({"train_device": train_device} if train_device else {}))
            self._write(versions + [version])
            return copy.deepcopy(version)

    def add_published(self, number, key):
        """Remember that version `number` is in the version store under `key`."""
        with self._lock:
            keys = self.get(number).get("published") or []
            return self._update(number, published=keys if key in keys else [*keys, key])

    def _update(self, number, **fields):
        with self._lock:
            versions = self.versions()
            for v in versions:
                if v["number"] == number:
                    v.update(fields)
                    self._write(versions)
                    return v
        raise KeyError(f"No version {number}.")

    def remove_last(self, number):
        """Remove version `number` when it is the newest and has no run or checkpoint (numbers follow the order, so an older one
        cannot go). Returns whether it was removed."""
        with self._lock:
            versions = self.versions()
            if not versions or versions[-1]["number"] != number or versions[-1]["runs"] or versions[-1]["best_checkpoint"]:
                return False
            self._write(versions[:-1])
            return True

    def set_change(self, number, change):
        return self._update(number, change=change)

    def leave_saved(self, number, note):
        """A version whose job ended without weights: back to "saved" (when its status can go there) with `note` added to its
        description, in one write, so a failure between two writes cannot leave the status changed and the note missing."""
        with self._lock:
            row = self.get(number)
            status = "saved" if "saved" in NEXT.get(row["status"], ()) else row["status"]
            return self._update(number, status=status, change=f"{row['change']}{note}")

    def set_status(self, number, status):
        now = self.get(number)["status"]
        if status not in NEXT.get(now, ()):
            raise ValueError(f"A {now} version cannot become {status}.")
        return self._update(number, status=status)

    def add_run(self, number, run_dir):
        return self._update(number, runs=[*self.get(number)["runs"], str(run_dir)])

    def set_best_checkpoint(self, number, path):
        return self._update(number, best_checkpoint=str(path))

    def set_source(self, number, source):
        """Say where version `number`'s weights came from: a dict kept on the row as `source`."""
        return self._update(number, source=dict(source))

    def set_devices(self, number, train=None, score=None):
        """Record the device kind that trained the version's checkpoint and/or the one that scored it; None leaves that one as it was."""
        fields = {k: v for k, v in (('train_device', train), ('score_device', score)) if v}
        return self._update(number, **fields) if fields else self.get(number)

    def set_scorecard(self, number, scorecard):
        return self._update(number, scorecard=dict(scorecard))

    def set_verdict(self, number, verdict):
        return self._update(number, verdict=verdict)
