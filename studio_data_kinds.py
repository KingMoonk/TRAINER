"""Three kinds of workspace data (text, conversations, problems), summaries for the Your data page, near-duplicates.

Problems are ordinary workspace examples with an extra "rule" (and "params"); both are mirrored into "metadata"
because studio_core.normalize drops unknown keys when an example is edited.
"""
import hashlib
import re
import zlib
from dataclasses import dataclass, field

import studio_core
from studio_rules import LABELS

KINDS = ("text", "conversations", "problems")
USED_BY = {"text": "used by Pretraining", "conversations": "used by Post-training, round 1",
           "problems": "used by Post-training, round 2"}
HELD_BACK_EVERY = 20          # about 5% of text documents / conversations, chosen by a hash of the content
NEAR_DUPLICATE = 0.8
SKETCH = 128                  # bottom-k hashes kept per document for the exact-ish Jaccard check
PERMS = [(2 * i * 2654435761 + 1, 40503 * i + 7) for i in range(1, 17)]   # 16 min-hashes, 8 bands of 2
BUCKET_CAP = 8                # compare a document with at most this many earlier ones per band
TEXT_CAP = 5000               # characters per document that feed the shingles
NO_KEY = "(no answer key: the rule decides)"


def rule_of(example):
    return example.get("rule") or (example.get("metadata") or {}).get("rule") or ""


def classify(example):
    if rule_of(example):
        return "problems"
    if example.get("text") and not example.get("messages"):
        return "text"
    return "conversations"


def pretraining_examples(workspace, dataset_ids):
    """Plain-text examples only; conversations and problems never enter pretraining."""
    for dataset_id in dataset_ids:
        for example in workspace.iter_examples(dataset_id):
            if classify(example) == "text":
                yield example


def make_problem(item, quality="Good"):
    """A problem item from studio_rules.generate -> a row for normalize/Workspace.add (rule fields mirrored in metadata)."""
    meta = {"rule": item["rule"], "params": item.get("params", {}), "noKey": "answer" not in item}
    return {"prompt": item["prompt"], "answer": item.get("answer") or NO_KEY, "rule": item["rule"],
            "params": item.get("params", {}), "intent": item["rule"], "quality": quality, "synthetic": True,
            "source": "synthetic", "metadata": meta}


def add_problems(workspace, dataset_id, items, quality="Good"):
    """Store problems (rule + params kept on each example). Returns (added, duplicate)."""
    added = duplicate = 0
    known = {studio_core.fingerprint(e) for e in workspace.iter_examples(dataset_id)}
    for item in items:
        row = make_problem(item, quality)
        example = studio_core.normalize(row, dataset_id, "synthetic")
        if example is None or studio_core.fingerprint(example) in known:
            duplicate += 1
            continue
        known.add(studio_core.fingerprint(example))
        example["rule"], example["params"] = row["rule"], row["params"]
        dataset = workspace._dataset(dataset_id)
        if dataset.get("storage") == "sqlite":
            with workspace._connect() as connection:
                connection.execute("INSERT OR IGNORE INTO records VALUES (?,?,?,?,?,?,?,?,?)", workspace._db_row(example))
        else:
            workspace.state["examples"].append(example)
        added += 1
    if added and workspace._dataset(dataset_id).get("storage") == "sqlite":
        with workspace._connect() as connection:
            workspace._dataset(dataset_id)["recordCount"] = connection.execute(
                "SELECT COUNT(*) FROM records WHERE dataset_id=?", (dataset_id,)).fetchone()[0]
    workspace.save()
    return added, duplicate


def ensure_dataset(workspace, name, description=""):
    found = next((d for d in workspace.state["datasets"] if d["name"] == name), None)
    return found["id"] if found else workspace.create_dataset(name, description)


# ---- near-duplicates ----------------------------------------------------------------------------------------

def normalized_text(example):
    text = example.get("text") or f'{example.get("user", "")}\n{example.get("assistant", "")}'
    return re.sub(r"\s+", " ", text.lower()).strip()[:TEXT_CAP]


def _sketch(text):
    words = re.findall(r"\w+", text)
    grams = {" ".join(words[i:i + 3]) for i in range(max(1, len(words) - 2))} if words else set()
    hashes = {zlib.crc32(g.encode()) for g in grams}
    return hashes


def near_duplicates(items, threshold=NEAR_DUPLICATE):
    """items: iterable of (id, normalised text). Returns {id: earlier_id} for each later near-copy.

    LSH on 16 min-hashes (8 bands of 2) so only documents sharing a bucket are compared, each against at most
    BUCKET_CAP earlier ones per band: cost is linear in the number of documents, never pairwise over all.
    """
    buckets, kept, found = {}, {}, {}
    for key, text in items:
        hashes = _sketch(text)
        if not hashes:
            continue
        sig = [min((h * a + b) & 0xFFFFFFFF for h in hashes) for a, b in PERMS]
        sketch = frozenset(sorted(hashes)[:SKETCH])
        kept[key] = sketch
        bands = [(i, sig[2 * i], sig[2 * i + 1]) for i in range(8)]
        for band in bands:
            for other in buckets.get(band, ()):
                if key in found:
                    break
                theirs = kept[other]
                if len(sketch & theirs) / max(1, len(sketch | theirs)) >= threshold:
                    found[key] = other
        if key not in found:           # near-copies are not indexed, so the original stays the representative
            for band in bands:
                bucket = buckets.setdefault(band, [])
                if len(bucket) < BUCKET_CAP:
                    bucket.append(key)
    return found


# ---- summaries ----------------------------------------------------------------------------------------------

@dataclass
class DataSummary:
    kind: str
    count: int = 0
    characters: int = 0
    used_by: str = ""
    sets: list = field(default_factory=list)       # [{id, name, count}]
    rules: dict = field(default_factory=dict)      # problems: rule -> count
    near_duplicates: int = 0
    scanned: int = 0                               # examples looked at for near-duplicates (bounded)
    rows: list = field(default_factory=list)       # [{id, a, b, status, dataset}]

    @property
    def count_line(self):
        if self.kind == "text":
            return f"{self.characters:,} characters · {len(self.sets)} sets"
        if self.kind == "conversations":
            return f"{self.count:,} conversations · {len(self.sets)} sets"
        return f"{self.count:,} problems · {len(self.rules)} rules"


def held_back(example):
    return int(hashlib.sha1(studio_core.fingerprint(example).encode()).hexdigest(), 16) % HELD_BACK_EVERY == 0


def _columns(kind, example, dataset_name):
    if kind == "text":
        text = example.get("text", "")
        return (example.get("title") or dataset_name), text.strip().split("\n", 1)[0][:120]
    if kind == "conversations":
        return example.get("user", ""), example.get("assistant", "")
    return example.get("user", ""), LABELS.get(rule_of(example), rule_of(example))


def _status(kind, example, duplicate):
    quality = example.get("quality")
    if kind == "text":
        return "Held back" if held_back(example) else "Near-duplicate" if duplicate else "In use"
    if kind == "conversations":
        if str(example.get("source", "")).lower() in {"chat", "correction"}:
            return "From Chat"
        return "Held back" if held_back(example) else "Good" if quality == "Good" else "Near-duplicate" if duplicate else "Review"
    return "Review" if quality == "Review" else "Ready"


def summarize(workspace, kind, query="", limit=200, scan_limit=20000, dataset_ids=None):
    """Counts, sets and up to `limit` sample rows of one kind. Near-duplicates are looked for in the first
    `scan_limit` examples of the kind only (bounded cost)."""
    out = DataSummary(kind, used_by=USED_BY[kind])
    needle = query.casefold().strip()
    pairs, dataset_names = [], {}
    for dataset in workspace.state["datasets"]:
        if dataset_ids is not None and dataset["id"] not in dataset_ids:
            continue
        dataset_names[dataset["id"]] = dataset["name"]
        n = 0
        for example in workspace.iter_examples(dataset["id"]):
            if classify(example) != kind:
                continue
            n += 1
            out.characters += len(example.get("text") or "") + len(example.get("user", "")) + len(example.get("assistant", ""))
            if kind == "problems":
                out.rules[rule_of(example)] = out.rules.get(rule_of(example), 0) + 1
            if len(pairs) < scan_limit:
                pairs.append((example, dataset["name"]))
        if n:
            out.sets.append({"id": dataset["id"], "name": dataset["name"], "count": n})
            out.count += n
    out.scanned = len(pairs)
    dups = near_duplicates((e["id"], normalized_text(e)) for e, _ in pairs) if kind != "problems" else {}
    out.near_duplicates = len(dups)
    for example, name in pairs:
        a, b = _columns(kind, example, name)
        if needle and needle not in f"{a} {b}".casefold():
            continue
        out.rows.append({"id": example["id"], "a": a, "b": b, "status": _status(kind, example, example["id"] in dups), "dataset": name})
        if len(out.rows) >= limit:
            break
    return out
