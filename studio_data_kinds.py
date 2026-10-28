"""Three kinds of workspace data (text, conversations, problems), summaries for the Your data page, near-duplicates.

Problems are ordinary workspace examples with an extra "rule" (and "params"); both are mirrored into "metadata"
because studio_core.normalize drops unknown keys when an example is edited.
"""
import hashlib
import itertools
import re
import sys
import zlib
from array import array
from dataclasses import dataclass, field, replace

import studio_core
import studio_corpus
from studio_rules import LABELS

KINDS = ("text", "conversations", "problems")
USED_BY = {"text": "used by Pretraining", "conversations": "used by Post-training, round 1",
           "problems": "used by Post-training, round 2"}
HELD_BACK_EVERY = 20          # about 5% of text documents / conversations, chosen by a hash of the content
CHAT_SHARE_MAX = 0.10         # the most conversations a pretraining mix may hold, as a share of its characters
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


def is_draft(example):
    """Generated and not yet marked Good: shown as Review and used for nothing. Imported text has no review step."""
    return bool(example.get("synthetic")) and example.get("quality") == "Review"


def pretraining_examples(workspace, dataset_ids):
    """Plain-text examples only; conversations and problems never enter pretraining."""
    for dataset_id in dataset_ids:
        for example in workspace.iter_examples(dataset_id):
            if classify(example) == "text" and not is_draft(example):
                yield example


def chat_examples(workspace, dataset_id, characters, good_only=False):
    """Conversations of one set that may go into a pretraining mix, in storage order, until `characters` of them have been given (the
    row that crosses the line is given too). Drafts and the held-back conversations (they measure Round 1) are left out, and with
    `good_only` so is everything not marked Good."""
    given = 0
    for example in workspace.iter_kind(dataset_id, "conversations"):
        if given >= characters:
            return
        if is_draft(example) or stamp(example)["holdout"] or (good_only and example.get("quality") != "Good"):
            continue
        size = studio_corpus.characters_of(example)
        if size:
            given += size
            yield example


def interleaved_examples(workspace, dataset_ids, shares, chat=None, good_only=False):
    """pretraining_examples of several sets mixed in proportion to `shares` (a text-character count per dataset id) instead of one
    set after another, so a reader that stops early (a token budget smaller than the selection) still gets every set's share.
    Each step takes the next row of the set that has given the least text relative to its share; a set that runs out drops out.
    The order is deterministic. `chat` ({conversation set id: characters of conversations to give}) mixes conversations in as well:
    each such set is read with chat_examples and stops after its characters, so the chat is that share of the mix and no more."""
    streams = {i: pretraining_examples(workspace, [i]) for i in dataset_ids if shares.get(i, 0) > 0}
    weights = dict(shares)
    for i, characters in (chat or {}).items():
        if characters > 0:
            streams[("chat", i)] = chat_examples(workspace, i, characters, good_only=good_only)
            weights[("chat", i)] = characters
    given = dict.fromkeys(streams, 0)
    while streams:
        pick = min(streams, key=lambda i: given[i] / weights[i])
        example = next(streams[pick], None)
        if example is None:
            del streams[pick]
            continue
        given[pick] += studio_corpus.characters_of(example)
        yield example


def text_sample(workspace, dataset_ids, budget=5_000_000, exclude=()):
    """(documents, total characters) for sizing pretraining without reading it all: disk sets give their size from the
    stored character counts and an evenly spread sample of about `budget` characters (every k-th row); in-memory sets
    are read whole. `exclude` holds texts to leave out (the held-back documents)."""
    import json
    docs, total, on_disk = [], 0, []
    for dataset_id in dataset_ids:
        if (workspace._dataset(dataset_id) or {}).get("storage") == "sqlite":
            on_disk.append(dataset_id)
            continue
        for example in pretraining_examples(workspace, [dataset_id]):
            if example.get("text") and example["text"] not in exclude:
                docs.append(example["text"])
                total += len(example["text"])
    if on_disk:
        marks = ",".join("?" * len(on_disk))
        with workspace._connect() as connection:
            chars = connection.execute(f"SELECT COALESCE(SUM(characters),0) FROM records WHERE dataset_id IN ({marks})", on_disk).fetchone()[0]
            step = max(1, round(chars / budget))
            total += chars
            cursor = connection.execute(f"SELECT payload FROM records WHERE dataset_id IN ({marks}) AND rowid % ? = 0 ORDER BY rowid",
                                        (*on_disk, step))
            while batch := cursor.fetchmany(2000):
                for (payload,) in batch:
                    example = json.loads(payload)
                    if classify(example) == "text" and not is_draft(example) and example.get("text") and example["text"] not in exclude:
                        docs.append(example["text"])
    return docs, total


def text_characters(workspace, dataset_ids, exclude=()):
    """Total characters of the chosen sets' training text, counted the way studio_corpus.characters_of counts a text example and
    without tokenizing or sampling: disk sets give it from the stored per-row character counts (a text row stores its text's length),
    summed over the text rows that are not unreviewed drafts and minus the held-back documents in `exclude` (found by their
    fingerprint); in-memory sets are read whole. Conversations, problems and drafts add nothing, so the sets' shares do not
    favour record-heavy ones."""
    total, on_disk = 0, []
    for dataset_id in dataset_ids:
        if (workspace._dataset(dataset_id) or {}).get("storage") == "sqlite":
            on_disk.append(dataset_id)
            continue
        total += sum(len(e["text"]) for e in pretraining_examples(workspace, [dataset_id]) if e.get("text") and e["text"] not in exclude)
    if on_disk:
        marks = ",".join("?" * len(on_disk))
        counted = f"dataset_id IN ({marks}) AND kind='text' AND NOT (quality='Review' AND json_extract(payload, '$.synthetic')=1)"
        held = [studio_core.fingerprint({"text": text}) for text in set(exclude) if text]
        with workspace._connect() as connection:
            total += connection.execute(f"SELECT COALESCE(SUM(characters),0) FROM records WHERE {counted}", on_disk).fetchone()[0]
            for start in range(0, len(held), 500):  # SQLite caps the parameters of one statement
                chunk = held[start:start + 500]
                total -= connection.execute(f"SELECT COALESCE(SUM(characters),0) FROM records WHERE {counted} "
                                            f"AND fingerprint IN ({','.join('?' * len(chunk))})", (*on_disk, *chunk)).fetchone()[0]
    return total


def text_fingerprints(workspace, dataset_ids, exclude=()):
    """The fingerprints of the chosen sets' training text, the rows `text_characters` counts, without tokenizing or reading their text:
    disk sets give them from the records table, in-memory sets are fingerprinted as they are read. `exclude` (held-back texts) is left out."""
    held = {studio_core.fingerprint({"text": text}) for text in exclude if text}
    on_disk = []
    for dataset_id in dataset_ids:
        if (workspace._dataset(dataset_id) or {}).get("storage") == "sqlite":
            on_disk.append(dataset_id)
            continue
        for example in pretraining_examples(workspace, [dataset_id]):
            if example.get("text") and (found := studio_core.fingerprint(example)) not in held:
                yield found
    if on_disk:
        marks = ",".join("?" * len(on_disk))
        with workspace._connect() as connection:
            for (found,) in connection.execute(f"SELECT fingerprint FROM records WHERE dataset_id IN ({marks}) AND kind='text' AND NOT "
                                               "(quality='Review' AND json_extract(payload, '$.synthetic')=1)", on_disk):
                if found not in held:
                    yield found


def make_problem(item, quality="Good"):
    """A problem item from studio_rules.generate -> a row for normalize/Workspace.add (rule fields mirrored in metadata)."""
    meta = {"rule": item["rule"], "params": item.get("params", {}), "noKey": "answer" not in item}
    return {"prompt": item["prompt"], "answer": item.get("answer") or NO_KEY, "rule": item["rule"],
            "params": item.get("params", {}), "intent": item["rule"], "quality": quality, "synthetic": True,
            "source": "synthetic", "metadata": meta}


def add_problems(workspace, dataset_id, items, quality="Good"):
    """Store problems (rule + params kept on each example). Returns (added, duplicate)."""
    added = duplicate = 0
    dataset = workspace._dataset(dataset_id)
    on_disk = dataset.get("storage") == "sqlite"
    known = set() if on_disk else {studio_core.fingerprint(e) for e in workspace.iter_examples(dataset_id)}
    examples = []
    for item in items:
        row = make_problem(item, quality)
        example = studio_core.normalize(row, dataset_id, "synthetic")
        if example is None:
            duplicate += 1
            continue
        example["rule"], example["params"] = row["rule"], row["params"]
        examples.append(example)
    if on_disk:
        with workspace._connect({dataset_id}) as connection:  # one transaction; UNIQUE(dataset_id, fingerprint) rejects the duplicates
            for example in examples:
                stored = workspace._insert_rows(connection, [workspace._db_row(example)])
                added, duplicate = added + stored, duplicate + 1 - stored
            if added:
                dataset["recordCount"] = connection.execute("SELECT COUNT(*) FROM records WHERE dataset_id=?", (dataset_id,)).fetchone()[0]
    else:
        for example in examples:
            key = studio_core.fingerprint(example)
            if key in known:
                duplicate += 1
                continue
            known.add(key)
            workspace.state["examples"].append(stamp(example, key))
            workspace._bump(dataset_id)
            added += 1
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


def signature(text):
    """(16 min-hashes, bottom-k sketch) of one normalised text, or None when it has no words to hash."""
    hashes = _sketch(text)
    if not hashes:
        return None
    return [min((h * a + b) & 0xFFFFFFFF for h in hashes) for a, b in PERMS], frozenset(sorted(hashes)[:SKETCH])


def pack_signature(sig, sketch):
    """A signature as bytes (little-endian uint32s: the 16 min-hashes, then the sketch) so the summary cache can keep it."""
    values = array("I", [*sig, *sorted(sketch)])
    if sys.byteorder == "big":
        values.byteswap()
    return values.tobytes()


def unpack_signature(packed):
    values = array("I")
    values.frombytes(packed)
    if sys.byteorder == "big":
        values.byteswap()
    return list(values[:len(PERMS)]), frozenset(values[len(PERMS):])


def match_signatures(items, threshold=NEAR_DUPLICATE):
    """items: iterable of (id, signature, sketch), in order. Returns {id: earlier_id} for each later near-copy.

    LSH on 16 min-hashes (8 bands of 2) so only documents sharing a bucket are compared, each against at most
    BUCKET_CAP earlier ones per band: cost is linear in the number of documents, never pairwise over all.
    """
    buckets, kept, found = {}, {}, {}
    for key, sig, sketch in items:
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


def near_duplicates(items, threshold=NEAR_DUPLICATE):
    """items: iterable of (id, normalised text). Returns {id: earlier_id} for each later near-copy."""
    def signed():
        for key, text in items:
            found = signature(text)
            if found:
                yield key, *found
    return match_signatures(signed(), threshold)


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
    held_back: int = 0                             # examples kept out of training (the holdout)
    scanned: int = 0                               # examples looked at for near-duplicates (bounded)
    rows: list = field(default_factory=list)       # [{id, a, b, status, dataset}]

    @property
    def count_line(self):
        if self.kind == "text":
            return f"{self.characters:,} characters · {len(self.sets)} sets"
        if self.kind == "conversations":
            return f"{self.count:,} conversations · {len(self.sets)} sets"
        return f"{self.count:,} problems · {len(self.rules)} rules"


def held_key(fingerprint):
    """Held back or not, from a fingerprint already in hand (the records table stores it as the `holdout` column)."""
    return int(hashlib.sha1(fingerprint.encode()).hexdigest(), 16) % HELD_BACK_EVERY == 0


def held_back(example):
    return held_key(studio_core.fingerprint(example))


def stamp(example, fingerprint=None):
    """Write `kind` and `holdout` into an in-memory (JSON dataset) example so post_rows reads two keys, not a classify and a hash.
    Without `fingerprint` only missing keys are filled (examples saved before the stamp existed)."""
    if fingerprint is not None or "kind" not in example:
        example["kind"] = classify(example)
    if fingerprint is not None or "holdout" not in example:
        example["holdout"] = int(held_key(fingerprint or studio_core.fingerprint(example)))
    return example


def _columns(kind, example, dataset_name):
    if kind == "text":
        text = example.get("text", "")
        return (example.get("title") or dataset_name), text.strip().split("\n", 1)[0][:120]
    if kind == "conversations":
        return example.get("user", ""), example.get("assistant", "")
    return example.get("user", ""), LABELS.get(rule_of(example), rule_of(example))


def _status(kind, example, duplicate):
    if is_draft(example):
        return "Review"
    quality = example.get("quality")
    if kind == "text":
        return "Review" if is_draft(example) else "Held back" if held_back(example) else "Near-duplicate" if duplicate else "In use"
    if kind == "conversations":
        if str(example.get("source", "")).lower() in {"chat", "correction"}:
            return "From Chat"
        return "Held back" if held_back(example) else "Good" if quality == "Good" else "Near-duplicate" if duplicate else "Review"
    return "Review" if quality == "Review" else "Ready"


def scan(workspace, scan_limit=20000, dataset_ids=None, progress=None):
    """One pass over the workspace for all three kinds: ({kind: DataSummary without rows}, {kind: the first
    `scan_limit` (example, dataset name)}). progress(text, fraction) is told what is being read.

    A disk dataset whose rows all carry a kind is counted by one GROUP BY on that column and read only as far as the
    first `scan_limit` examples of each kind; other datasets are read whole."""
    out = {kind: DataSummary(kind, used_by=USED_BY[kind]) for kind in KINDS}
    pairs = {kind: [] for kind in KINDS}
    datasets = [d for d in list(workspace.state["datasets"]) if dataset_ids is None or d["id"] in dataset_ids]
    total, seen = sum(d.get("recordCount") or 0 for d in datasets), 0
    for dataset in datasets:
        counts = dict.fromkeys(KINDS, 0)
        stats = workspace.kind_stats(dataset["id"]) if dataset.get("storage") == "sqlite" else None
        if stats is not None:
            if progress:
                progress(f"Reading {dataset['name']}", min(1, seen / total) if total else None)
            seen += sum(n for n, *_ in stats.values())
            for kind, (n, characters, held) in stats.items():
                counts[kind] = n
                out[kind].characters += characters
                out[kind].held_back += held
                if kind == "problems" and n:
                    for example in workspace.iter_kind(dataset["id"], kind):
                        out[kind].rules[rule_of(example)] = out[kind].rules.get(rule_of(example), 0) + 1
                        if len(pairs[kind]) < scan_limit:
                            pairs[kind].append((example, dataset["name"]))
                elif n and len(pairs[kind]) < scan_limit:
                    for example in itertools.islice(workspace.iter_kind(dataset["id"], kind), scan_limit - len(pairs[kind])):
                        pairs[kind].append((example, dataset["name"]))
        else:
            for example in workspace.iter_examples(dataset["id"]):
                if progress and seen % 2000 == 0:
                    progress(f"Reading {dataset['name']}", min(1, seen / total) if total else None)
                seen += 1
                kind = classify(example)
                counts[kind] += 1
                summary = out[kind]
                # a text example repeats its text as "user", so it counts once
                summary.characters += (len(example.get("text") or "") if kind == "text"
                                       else len(example.get("user", "")) + len(example.get("assistant", "")))
                held = example.get("holdout")
                summary.held_back += int(held_back(example) if held is None else held)
                if kind == "problems":
                    summary.rules[rule_of(example)] = summary.rules.get(rule_of(example), 0) + 1
                if len(pairs[kind]) < scan_limit:
                    pairs[kind].append((example, dataset["name"]))
        for kind, n in counts.items():
            if n:
                out[kind].sets.append({"id": dataset["id"], "name": dataset["name"], "count": n})
                out[kind].count += n
    return out, pairs


def matching(rows, query="", limit=200):
    """The first `limit` rows whose two columns contain `query` (limit None = all)."""
    needle = query.casefold().strip()
    return [r for r in rows if not needle or needle in f"{r['a']} {r['b']}".casefold()][:limit]


def page(workspace, kind, query="", sets=(), cursor=None, limit=200, dups=(), review_only=False):
    """(rows, next cursor) of one page of `kind` rows read from storage, not from the summary's sample: every row of
    `sets` ((dataset id, name) in order) whose text holds `query` can be reached. cursor None = the first page; the
    next cursor is None when nothing follows. `dups` = ids to show as near-duplicates. Disk datasets match in SQL, which
    folds case for ASCII letters only; others match in Python."""
    needle = query.strip().casefold()
    rules = [key for key, label in LABELS.items() if needle in f"{key} {label}".casefold()] if needle and kind == "problems" else []
    index, after = cursor or (0, 0)
    found = []  # (set index, position in the set, row)
    while index < len(sets) and len(found) <= limit:
        dataset_id, name = sets[index]
        want = limit + 1 - len(found)
        if (workspace._dataset(dataset_id) or {}).get("storage") == "sqlite":
            examples = workspace.kind_rows(dataset_id, kind, query, after, want, rules, review_only=review_only)
        else:
            examples = [(i, e) for i, e in enumerate(workspace.examples(dataset_id), 1) if i > after and classify(e) == kind and (not review_only or is_draft(e)) and (
                not needle or rule_of(e) in rules or needle in " ".join(str(e.get(k) or "") for k in ("user", "assistant", "intent", "text")).casefold())][:want]
        for position, example in examples:
            a, b = _columns(kind, example, name)
            found.append((index, position, {"id": example["id"], "a": a, "b": b, "status": _status(kind, example, example["id"] in dups), "dataset": name}))
        index, after = index + 1, 0
    more = len(found) > limit
    return [row for *_, row in found[:limit]], (found[limit - 1][:2] if more else None)


def summarize(workspace, kind, query="", limit=200, scan_limit=20000, dataset_ids=None, scanned=None, progress=None):
    """Counts, sets and up to `limit` sample rows of one kind. Near-duplicates are looked for in the first
    `scan_limit` examples of the kind only (bounded cost). `scanned` reuses an earlier scan() instead of reading again."""
    summaries, pairs = scanned or scan(workspace, scan_limit, dataset_ids, progress)
    out, pairs = replace(summaries[kind]), pairs[kind]
    out.scanned = len(pairs)

    def texts():
        for i, (example, _) in enumerate(pairs):
            if progress and i % 500 == 0:
                progress(f"Looking for near-copies in your {kind}", i / len(pairs))
            yield example["id"], normalized_text(example)
    dups = near_duplicates(texts()) if kind != "problems" else {}
    out.near_duplicates = len(dups)
    rows = []
    for example, name in pairs:
        a, b = _columns(kind, example, name)
        rows.append({"id": example["id"], "a": a, "b": b, "status": _status(kind, example, example["id"] in dups), "dataset": name})
    out.rows = matching(rows, query, limit)
    return out


# ---- one dataset's share of the summaries -------------------------------------------------------------------
# A summary of the whole workspace is the sum of one entry per dataset. An entry is a small JSON of counts beside the
# dataset's `rev` (studio_data_cache keeps it in data/summaries.sqlite), so only a dataset whose rev moved is read again.
# The sample rows and near-duplicate signatures are not in the entry: the cache keeps them apart, a row each, so a launch
# or a page read never loads them. Order matters: the near-duplicate check and the sample rows cover the first
# `scan_limit` examples of a kind across datasets in workspace order, so an entry keeps as many rows as that budget
# could take from it, and no more.

def budgets(datasets, entries, scan_limit):
    """Yield (dataset, entry, {kind: rows this dataset contributes}) in order, spending the per-kind budget as it goes."""
    left = dict.fromkeys(KINDS, scan_limit)
    for dataset in datasets:
        entry = entries[dataset["id"]]
        take = {kind: min(entry["kind_counts"][kind], left[kind]) for kind in KINDS}
        for kind in KINDS:
            left[kind] -= take[kind]
        yield dataset, entry, take


def dataset_entry(workspace, dataset, scan_limit, progress=None):
    """{kind_counts, chars, holdout, rules, example, limit, kept, rows} for one dataset: counts of everything in it, and for
    its first `scan_limit` examples of each kind a row [id, a, b, status, can, signature] in `rows` (`kept` says how many)."""
    summaries, pairs = scan(workspace, scan_limit, [dataset["id"]], progress)
    entry = {"limit": scan_limit, "rules": summaries["problems"].rules, "rows": {}, "kept": {},
             "kind_counts": {k: summaries[k].count for k in KINDS}, "chars": {k: summaries[k].characters for k in KINDS},
             "holdout": {k: summaries[k].held_back for k in KINDS}}
    for kind in KINDS:
        rows = []
        for example, name in pairs[kind]:
            a, b = _columns(kind, example, name)
            status = _status(kind, example, False)
            found = signature(normalized_text(example)) if kind != "problems" else None
            # can: a near-copy would change the status
            rows.append((example["id"], a, b, status, int(_status(kind, example, True) != status), pack_signature(*found) if found else None))
        entry["rows"][kind], entry["kept"][kind] = rows, len(rows)
    first = next((e for e, _ in pairs["conversations"] if not held_back(e)), None)  # the first of post_rows("conversations")
    entry["example"] = {"user": first.get("user", ""), "assistant": first.get("assistant", "")} if first else None
    return entry


def refresh(workspace, cached, scan_limit=20000, rescan_all=False, progress=None):
    """Bring per-dataset entries up to date: ({dataset_id: entry}, [ids read again]). `cached` entries are kept while their
    dataset's `rev` is the entry's and they hold the rows the budget asks of them; the rest are read with scan(dataset_ids=[id])
    and carry their `rows` until the cache takes them."""
    entries, reread, left = {}, [], dict.fromkeys(KINDS, scan_limit)
    for dataset in list(workspace.state["datasets"]):
        rev = dataset.get("rev", 0)  # read before the scan: a write during it leaves the entry behind the rev, so the next refresh rereads
        entry = None if rescan_all else cached.get(dataset["id"])
        # an entry also has to hold the rows this dataset's turn in the budget asks for (an earlier dataset may have shrunk)
        if entry is None or entry.get("rev") != rev or any(entry["kept"][k] < min(entry["kind_counts"][k], left[k]) for k in KINDS):
            entry = dataset_entry(workspace, dataset, max(left.values()), progress)
            entry["rev"] = rev
            reread.append(dataset["id"])
        entries[dataset["id"]] = entry
        for kind in KINDS:
            left[kind] -= min(entry["kind_counts"][kind], left[kind])
    return entries, reread


def assemble(datasets, entries, view, scan_limit=20000):
    """({kind: DataSummary with the first page of sample rows}, the first training conversation as {prompt, answer}):
    what summarize() reads over the whole workspace, without reading it. `view` is what the cache made of the rows
    ({"duplicates": {kind: n}, "rows": {kind: [row]}})."""
    out = {kind: DataSummary(kind, used_by=USED_BY[kind]) for kind in KINDS}
    example = None
    for dataset, entry, take in budgets(datasets, entries, scan_limit):
        for kind in KINDS:
            summary, n = out[kind], entry["kind_counts"][kind]
            if n:
                summary.sets.append({"id": dataset["id"], "name": dataset["name"], "count": n})
                summary.count += n
            summary.characters += entry["chars"][kind]
            summary.held_back += entry["holdout"][kind]
            summary.scanned += take[kind]
            if kind == "problems":
                for rule, count in entry["rules"].items():
                    summary.rules[rule] = summary.rules.get(rule, 0) + count
        if example is None and entry["example"] is not None:
            example = entry["example"]
    for kind in KINDS:
        out[kind].near_duplicates, out[kind].rows = view["duplicates"][kind], list(view["rows"][kind])
    return out, ({"prompt": example["user"], "answer": example["assistant"]} if example and example["user"] else {})
