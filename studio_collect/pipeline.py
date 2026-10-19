"""Candidate -> stored record: licence, quality, personal data, exact and near duplicates, held-out text. Then a shard on disk.

A source adapter turns what it reads into `Candidate`s and offers each to `Pipeline.offer`, which returns "" when the record was
kept or the reason it was dropped. Kept records are buffered and written as a shard (data/collect/shards/<group>/*.jsonl), the
hand-off to the app, which registers them in records.sqlite (studio_collect.register). Every record carries `provenance`:
source, source URL, licence, retrieval time and attribution.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import studio_core
import studio_decontam
from studio_collect import clean, licences
from studio_collect.config import collect_dir, slug
from studio_collect.store import LimitReached, blank_stats

SHARD_RECORDS = 2000
SHARD_BYTES = 16 * 1024 * 1024


@dataclass
class Candidate:
    """One thing a source read. kind "text": `text`. kind "qa": `question` and `answer`."""
    kind: str
    source: str                 # "hub:owner/name", "stackexchange:math.stackexchange.com", "github:owner/repo", "web:host"
    group: str                  # the dataset it lands in (a topic, a site)
    url: str
    licence: str                # as the source states it (raw)
    attribution: str = ""
    text: str = ""
    question: str = ""
    answer: str = ""
    source_id: str = ""
    meta: dict = field(default_factory=dict)


def shard_dir(root, group=None):
    base = collect_dir(root) / "shards"
    return base / slug(group) if group else base


def eval_texts(root):
    """The held-out texts the collector must not train on: held-back documents (data/heldout.json), and the held-back conversations
    and problems of every stored set. Read-only."""
    texts = []
    try:
        texts += [row["text"] if isinstance(row, dict) else row for row in json.loads((Path(root) / "data" / "heldout.json").read_text(encoding="utf-8"))]
    except (OSError, ValueError, TypeError, KeyError):
        pass
    try:
        workspace = studio_core.Workspace(root)
        ids = [d["id"] for d in workspace.state["datasets"] if d.get("storage") == "sqlite"]
        for kind in ("text", "conversations", "problems"):
            for rows in workspace.pick(kind, True, ids).values():
                texts += [f"{r.get('user', '')}\n{r.get('assistant', '')}" if kind != "text" else (r.get("text") or "") for r in rows]
        for example in workspace.state["examples"]:
            import studio_data_kinds as dk
            if dk.stamp(example).get("holdout"):
                texts.append(example.get("text") or f"{example.get('user', '')}\n{example.get('assistant', '')}")
    except (OSError, ValueError, KeyError):
        pass
    return [t for t in texts if t and t.strip()]


def eval_index(texts):
    return studio_decontam.Index().add_all(texts)


class Pipeline:
    """seen: dedup.Seen. index: a studio_decontam.Index of the held-out text. opt_in: the config's "licences"."""

    def __init__(self, root, seen, index, opt_in, clock=time.time):
        self.root, self.seen, self.index, self.opt_in, self.clock = Path(root), seen, index, opt_in, clock
        self.buffers = {}     # group -> [rows]
        self.buffer_bytes = 0
        self.stats = blank_stats()
        self.group_stats = {}
        self.shards_written = 0

    def _count(self, group, key, reason=None, n=1):
        for target in (self.stats, self.group_stats.setdefault(group, blank_stats())):
            if reason:
                target["dropped"][reason] += n
            else:
                target[key] += n

    def drop(self, group, reason):
        self._count(group, "seen")
        self._count(group, "", reason)
        return reason

    def offer(self, c):
        """"" when kept (buffered for the next shard), else the reason it was dropped."""
        ident, klass = licences.classify(c.licence)
        if not licences.allowed(klass, self.opt_in):
            return self.drop(c.group, "licence")
        if c.kind == "qa":
            problem = clean.pair_problem(c.question, c.answer)
        else:
            problem = clean.text_problem(c.text)
        if problem:
            return self.drop(c.group, "quality")
        redactions = 0
        if c.kind == "qa":
            question, n1 = clean.strip_pii(c.question)
            answer, n2 = clean.strip_pii(c.answer)
            redactions = n1 + n2
            body = {"user": question, "assistant": answer}
        else:
            text, redactions = clean.strip_pii(c.text)
            body = {"text": text}
        if redactions >= clean.PII_DROP:
            return self.drop(c.group, "pii")
        retrieved = datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(timespec="seconds")
        provenance = {"source": c.source, "source_url": c.url, "licence": ident, "licence_class": klass, "retrieved_at": retrieved,
                      "attribution": c.attribution, **c.meta}
        example = studio_core.normalize({**body, "quality": "Review", "synthetic": False, "source": f"collect:{c.source}", "license": ident,
                                         "source_url": c.url, "sourceId": c.source_id, "provenance": provenance,
                                         "groupId": studio_decontam.group_id(body.get("text") or body["user"], "collect")},
                                        source=f"collect:{c.source}")
        if example is None:
            return self.drop(c.group, "invalid")
        keys, known = self.seen.exact(example)
        if known:
            return self.drop(c.group, "duplicate")
        document = example.get("text") or f"{example['user']}\n{example['assistant']}"
        if self.index.overlaps(document):
            return self.drop(c.group, "decontam")
        near, signature = self.seen.near(example)
        if near:
            return self.drop(c.group, "near_duplicate")
        self.seen.add(keys, signature)
        for name in ("id", "datasetId", "updatedAt"):
            example.pop(name, None)
        line = json.dumps(example, ensure_ascii=False)
        self.buffers.setdefault(c.group, []).append(line)
        self.buffer_bytes += len(line)
        self._count(c.group, "seen")
        self._count(c.group, "kept")
        self._count(c.group, "bytes_kept", n=len(line))
        if redactions:
            self._count(c.group, "redactions", n=redactions)
        return ""

    def full(self):
        return self.buffer_bytes >= SHARD_BYTES or sum(map(len, self.buffers.values())) >= SHARD_RECORDS

    def flush(self):
        """Write the buffered records as shards, then commit the dedup index. Returns the stats since the last flush
        ({"stats": ..., "groups": {group: stats}}) for the caller to add to the state."""
        for group, lines in self.buffers.items():
            if not lines:
                continue
            folder = shard_dir(self.root, group)
            folder.mkdir(parents=True, exist_ok=True)
            name = f"{datetime.fromtimestamp(self.clock(), timezone.utc).strftime('%Y%m%dT%H%M%S')}-{self.shards_written:05d}-{studio_core.uid('s')[2:8]}.jsonl"
            part = folder / (name + ".part")
            part.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            studio_core.replace_file(part, folder / name)   # a shard appears whole or not at all
            self.shards_written += 1
        self.seen.commit()
        self.buffers, self.buffer_bytes = {}, 0
        out = {"stats": self.stats, "groups": self.group_stats}
        self.stats, self.group_stats = blank_stats(), {}
        return out


__all__ = ["Candidate", "Pipeline", "eval_texts", "eval_index", "shard_dir", "LimitReached"]
