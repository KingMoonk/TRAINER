"""Open datasets by topic from the Hugging Face Hub: search, rank, plan, download (streamed, capped, resumable), read rows.

Only ungated datasets are collected. A gated dataset needs its terms accepted on the hub with the person's own token, and this
app has no settings or keyring mechanism for such a token, so none is read, sent or stored (issue #448 follow-up).
Nothing is fetched except dataset files. Datasets tagged as benchmarks are skipped: they are evaluation material.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import studio_catalog
from studio_collect import licences
from studio_collect.config import collect_dir, slug, topic_list
from studio_collect.pipeline import Candidate
from studio_collect.store import LimitReached

API = "https://huggingface.co/api/datasets"
PAGE = "https://huggingface.co/datasets"
PER_TERM = 25
DATA_SUFFIXES = (".parquet", ".jsonl", ".json", ".jsonl.gz", ".json.gz", ".csv", ".csv.gz", ".tsv", ".txt", ".txt.gz")
SKIP_WORDS = re.compile(r"(^|[^a-z])(test|tests|val|valid|validation|dev|eval|evaluation|benchmark)([^a-z]|$)", re.I)
TEXT_FIELDS = ("text", "content", "body", "document", "article", "passage", "markdown", "raw_content", "abstract", "code", "output_text", "story", "contents")
PAIRS = (("question", "answer"), ("prompt", "response"), ("prompt", "completion"), ("instruction", "output"), ("instruction", "response"),
         ("query", "answer"), ("input", "output"), ("problem", "solution"), ("question", "response"), ("user", "assistant"), ("human", "assistant"))
CHAT_FIELDS = ("messages", "conversation", "conversations")
NOT_TEXT = ("modality:image", "modality:audio", "modality:video", "modality:3d", "modality:geospatial", "modality:timeseries", "modality:tabular")


@dataclass
class Item:
    """One dataset in the plan."""
    id: str
    topic: str
    sha: str
    licence: str
    licence_class: str
    downloads: int = 0
    likes: int = 0
    tags: list = field(default_factory=list)
    score: float = 0.0
    fit: float = 0.0
    files: list = field(default_factory=list)      # [{"path", "size", "sha256"?}] to fetch, in order
    bytes: int = 0                                  # what the plan would download
    total_bytes: int = 0                            # the dataset's data files
    status: str = "planned"                         # planned | needs_approval | skipped
    reason: str = ""                                # why it is skipped or needs approval
    author: str = ""


def _tag(tags, prefix):
    return [t.split(":", 1)[1] for t in tags if t.startswith(prefix + ":")]


def licence_of(raw):
    card = (raw.get("cardData") or {}).get("license")
    return card if card else (_tag(raw.get("tags") or [], "license") or "")


def fit_score(raw, terms):
    """0..1: how well the name, tags and description match the topic's search terms (the name counts most)."""
    name = raw.get("id", "").lower().replace("-", " ").replace("_", " ")
    tags = " ".join(raw.get("tags") or []).lower() + " " + " ".join((raw.get("cardData") or {}).get("tags") or []).lower()
    about = str(raw.get("description") or "").lower()
    hit = 0.0
    for term in terms:
        term = term.lower()
        hit = max(hit, 1.0 if term in name else 0.0, 0.7 if term in tags else 0.0, 0.4 if term in about else 0.0)
    return hit


def screen(raw, config):
    """Why a search result must not be planned at all, or ""."""
    tags = raw.get("tags") or []
    if raw.get("private") or raw.get("disabled"):
        return "private or disabled"
    if raw.get("gated"):
        return "gated: needs your own token and acceptance of its terms on the hub, which this app does not hold"
    if any(t == "benchmark" or t.startswith("benchmark:") for t in tags) or "benchmark" in (raw.get("cardData") or {}).get("tags", []):
        return "benchmark: evaluation material, kept out of training"
    if any(t in NOT_TEXT for t in tags) and "modality:text" not in tags:
        return "not text"
    ident, klass = licences.classify(licence_of(raw))
    if not licences.allowed(klass, config["licences"]):
        return licences.refusal(ident, klass)
    if raw["id"] in config["blocked"]:
        return "you blocked it"
    return ""


def pick_files(tree, limits):
    """The data files of a dataset to fetch: training files only, small ones first within the per-dataset cap.
    Returns (files, bytes planned, bytes of all data files)."""
    cap, per_file = limits["max_dataset_mb"] * 1_000_000, limits["max_file_mb"] * 1_000_000
    data = []
    for entry in tree:
        path = entry.get("path", "")
        if entry.get("type") != "file" or not path.lower().endswith(DATA_SUFFIXES) or path.lower().endswith("readme.md"):
            continue
        if SKIP_WORDS.search(path):
            continue
        size = int((entry.get("lfs") or {}).get("size") or entry.get("size") or 0)
        data.append({"path": path, "size": size, "sha256": (entry.get("lfs") or {}).get("oid", "")})
    data.sort(key=lambda f: (0 if "train" in f["path"].lower() else 1, f["path"]))
    chosen, total = [], 0
    for f in data:
        if f["size"] > per_file or f["size"] <= 0:
            continue
        if total + f["size"] > cap and chosen:
            continue
        if f["size"] > cap:      # a single file bigger than the dataset cap is read only up to the cap, so it is not worth a plan
            continue
        chosen.append(f)
        total += f["size"]
    return chosen, total, sum(f["size"] for f in data)


def rank_score(item, raw, terms, planned):
    """Licence 25%, topic fit 35%, popularity 20%, size 10% (a corpus worth reading, not a stub), card/quality 10%."""
    licence = {licences.PERMISSIVE: 1.0, licences.SHARE_ALIKE: 0.85, licences.COPYLEFT: 0.5, licences.NON_COMMERCIAL: 0.3}.get(item.licence_class, 0.1)
    popular = min(1.0, math.log10(item.downloads + 1) / 5)
    size = 0.0 if planned <= 0 else min(1.0, math.log10(max(planned, 1) / 100_000) / 3) if planned >= 100_000 else 0.05
    card = 0.3 + 0.3 * bool(raw.get("cardData")) + 0.2 * any(t.startswith("task_categories:") for t in item.tags) + 0.2 * (item.likes >= 3)
    return round(0.25 * licence + 0.35 * item.fit + 0.2 * popular + 0.1 * size + 0.1 * card, 4)


def search(http, term, limit=PER_TERM):
    url = f"{API}?search={quote(term)}&sort=downloads&direction=-1&limit={int(limit)}&full=true"
    status, _, body = http.get(url)
    if status != 200:
        raise OSError(f"The hub search answered {status}.")
    rows = json.loads(body)
    return rows if isinstance(rows, list) else []


def tree(http, dataset_id, sha):
    url = f"{API}/{quote(dataset_id, safe='/')}/tree/{sha or 'main'}?recursive=true"
    status, _, body = http.get(url)
    if status != 200:
        raise OSError(f"The hub file list for {dataset_id} answered {status}.")
    return json.loads(body)


def build_plan(http, config, stop=lambda: None, note=lambda text: None):
    """Search every chosen topic, screen, rank and list the files: [Item as dict], best first. Nothing large is fetched here."""
    seen, items, searches, failed = {}, [], 0, 0
    for topic, label, terms in topic_list(config):
        for term in terms[:3]:
            stop()
            searches += 1
            try:
                results = search(http, term)
            except OSError as error:
                note(f"Search for {term} failed: {error}")
                failed += 1
                continue
            for raw in results:
                if not isinstance(raw, dict) or "id" not in raw or raw["id"] in seen:
                    continue
                seen[raw["id"]] = True
                fit = fit_score(raw, terms)
                if fit <= 0:
                    continue
                ident, klass = licences.classify(licence_of(raw))
                item = Item(id=raw["id"], topic=topic, sha=raw.get("sha", ""), licence=ident, licence_class=klass, downloads=int(raw.get("downloads") or 0),
                            likes=int(raw.get("likes") or 0), tags=list(raw.get("tags") or [])[:30], fit=fit, author=raw.get("author", ""))
                reason = screen(raw, config)
                if reason:
                    item.status, item.reason = "skipped", reason
                    items.append((item, raw, terms))
                    continue
                items.append((item, raw, terms))
    if searches and failed == searches:
        raise OSError("The Hugging Face Hub could not be reached.")      # not an empty plan: try again later
    # files are listed only for the best candidates per topic (one request each), so the plan stays cheap
    ranked = sorted((x for x in items if x[0].status != "skipped"), key=lambda x: -(x[0].fit * 0.6 + min(1, math.log10(x[0].downloads + 1) / 5) * 0.4))
    per_topic, kept = {}, []
    for item, raw, terms in ranked:
        if per_topic.get(item.topic, 0) >= 6:
            continue
        stop()
        try:
            files, planned, total = pick_files(tree(http, item.id, item.sha), config["limits"])
        except (OSError, ValueError) as error:
            item.status, item.reason = "skipped", f"file list unavailable: {error}"
            kept.append((item, raw, terms))
            continue
        item.files, item.bytes, item.total_bytes = files, planned, total
        if not files:
            item.status, item.reason = "skipped", "no training data files small enough"
        elif item.id in config["approved"] or planned <= config["limits"]["auto_approve_mb"] * 1_000_000:
            item.status = "planned"
        else:
            item.status, item.reason = "needs_approval", f"{planned / 1e6:,.0f} MB to download: approve it to start"
        item.score = rank_score(item, raw, terms, planned)
        per_topic[item.topic] = per_topic.get(item.topic, 0) + 1
        kept.append((item, raw, terms))
    skipped = [x for x in items if x[0].status == "skipped" and x not in kept][:60]
    everything = sorted((x[0] for x in kept), key=lambda i: (i.status == "skipped", -i.score)) + [x[0] for x in skipped]
    return [asdict(i) for i in everything]


# ---- download (streamed, capped, resumable) ----------------------------------------------------------------

def download_dir(root, dataset_id):
    return collect_dir(root) / "downloads" / slug(dataset_id)


def resolve_url(dataset_id, sha, path):
    return f"{PAGE}/{quote(dataset_id, safe='/')}/resolve/{sha or 'main'}/{quote(path)}"


def fetch(http, url, dest, size, sha256="", stop=lambda: None, progress=lambda n: None):
    """Download `url` to `dest`, resuming a `.part` file with a Range request. The size must match, and an LFS hash is verified.
    LimitReached or Stopped leave the .part file for the next attempt."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size == size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    if have < size:
        reply = http.open(url, byte_range=(have, None) if have else None)
        try:
            if reply.status == 200 and have:      # the server ignored Range: start over rather than append the whole file
                have = 0
            elif reply.status not in (200, 206):
                raise OSError(f"The download answered {reply.status}.")
            with part.open("ab" if have else "wb") as out:
                for block in reply.chunks():
                    stop()
                    room = size - have
                    out.write(block[:room])
                    have += min(len(block), room)
                    progress(have)
                    if have >= size:
                        break
        finally:
            reply.close()
    if part.stat().st_size != size:
        raise OSError(f"Incomplete download of {dest.name}: {part.stat().st_size} of {size} bytes.")
    if sha256:
        digest = hashlib.sha256()
        with part.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != sha256:
            part.unlink()
            raise ValueError(f"{dest.name} does not match its published checksum; it was discarded.")
    part.replace(dest)
    return dest


# ---- rows --------------------------------------------------------------------------------------------------

def iter_rows(path):
    """Dict rows of a downloaded data file (parquet, jsonl, json array, csv/tsv, plain text paragraphs), gzip allowed."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".parquet"):
        yield from studio_catalog.iter_parquet(path)
        return
    opener = (lambda: gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")) if name.endswith(".gz") else (
        lambda: path.open("r", encoding="utf-8", errors="replace", newline=""))
    stem = name[:-3] if name.endswith(".gz") else name
    with opener() as stream:
        if stem.endswith(".jsonl"):
            for line in stream:
                if line.strip():
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    yield row if isinstance(row, dict) else {"text": row} if isinstance(row, str) else {}
        elif stem.endswith(".json"):
            first = stream.read(1)
            stream.seek(0)
            if first == "[":
                for row in studio_catalog.iter_json_array(_Bytes(stream)):
                    yield row if isinstance(row, dict) else {"text": row} if isinstance(row, str) else {}
            else:
                data = json.load(stream)
                if isinstance(data, dict) and isinstance(next(iter(data.values()), None), list):
                    data = next(iter(data.values()))
                yield from (r for r in data if isinstance(r, dict)) if isinstance(data, list) else iter(())
        elif stem.endswith((".csv", ".tsv")):
            csv.field_size_limit(1 << 26)
            yield from csv.DictReader(stream, delimiter="\t" if stem.endswith(".tsv") else ",")
        else:
            for block in re.split(r"\n\s*\n", stream.read()):
                if block.strip():
                    yield {"text": block.strip()}


class _Bytes:
    """iter_json_array reads bytes; this reads a text stream as UTF-8 bytes."""

    def __init__(self, stream):
        self.stream = stream

    def read(self, size=-1):
        return self.stream.read(size).encode("utf-8")


def _str(value):
    return value.strip() if isinstance(value, str) else ""


def row_to_candidates(row, item, source_id):
    """Zero or one Candidate for a dataset row: a chat or a question/answer pair becomes "qa", a long text field "text"."""
    base = dict(source=f"hub:{item['id']}", group=item["group"], url=f"{PAGE}/{item['id']}", licence=item["licence"], source_id=source_id,
                attribution=f"{item['id']} on the Hugging Face Hub" + (f" by {item['author']}" if item.get("author") else "") + f", licence {item['licence'] or 'not stated'}",
                meta={"dataset": item["id"], "revision": item["sha"], "topic": item["topic"]})
    for name in CHAT_FIELDS:
        chat = row.get(name)
        if isinstance(chat, str) and chat.strip().startswith("["):
            try:
                chat = json.loads(chat)
            except ValueError:
                chat = None
        if isinstance(chat, list) and chat:
            turns = [(str(t.get("role", t.get("from", ""))).lower(), _str(t.get("content", t.get("value")))) for t in chat if isinstance(t, dict)]
            user = [c for r, c in turns if r in ("user", "human") and c]
            bot = [c for r, c in turns if r in ("assistant", "gpt", "bot") and c]
            if user and bot:    # one pair per record keeps the question and its answer together; longer chats keep their first turn pair
                return [Candidate(kind="qa", question=user[0], answer=bot[0], **base)]
    for q, a in PAIRS:
        if _str(row.get(q)) and _str(row.get(a)):
            question = _str(row[q]) + (("\n" + _str(row["input"])) if q == "instruction" and _str(row.get("input")) else "")
            return [Candidate(kind="qa", question=question, answer=_str(row[a]), **base)]
    for name in TEXT_FIELDS:
        if _str(row.get(name)):
            return [Candidate(kind="text", text=_str(row[name]), **base)]
    strings = [v.strip() for v in row.values() if isinstance(v, str) and len(v.strip()) >= 200]
    return [Candidate(kind="text", text=strings[0], **base)] if len(strings) == 1 else []


# ---- the adapter -------------------------------------------------------------------------------------------

class Adapter:
    """Works the plan one file at a time. State per dataset: file index, rows already read from the current file, done."""

    key = "hub"

    def __init__(self, ctx):
        self.ctx = ctx

    @staticmethod
    def plan_inputs(config):
        return {"topics": config["topics"], "custom": config["custom_topics"], "licences": config["licences"],
                "limits": {k: config["limits"][k] for k in ("max_dataset_mb", "max_file_mb", "auto_approve_mb")}}

    def plan(self):
        ctx = self.ctx
        ctx.journal.append("plan", source="hub", note="Searching the Hugging Face Hub")
        items = build_plan(ctx.http, ctx.config, ctx.stop, lambda text: ctx.journal.append("note", text=text))
        for item in items:
            item["group"] = f"topic-{slug(item['topic'])}"
            item["source"] = "hub"
        ctx.set_plan("hub", items)
        ctx.journal.append("plan", source="hub", count=len([i for i in items if i["status"] != "skipped"]))
        return items

    def pending(self):
        ctx = self.ctx
        for item in ctx.plan_items("hub"):
            if item["id"] in ctx.config["blocked"]:
                continue
            if item["status"] == "needs_approval" and item["id"] not in ctx.config["approved"]:
                continue
            if item["status"] == "skipped" or ctx.state.source(f"hub:{item['id']}").get("done"):
                continue
            yield item

    def step(self):
        """Download and read the next file of the next pending dataset. False when there is nothing left to do."""
        ctx = self.ctx
        item = next(self.pending(), None)
        if item is None:
            return False
        key = f"hub:{item['id']}"
        saved = ctx.state.source(key)
        index, offset = saved.get("file", 0), saved.get("offset", 0)
        if index >= len(item["files"]):
            ctx.state.update_source(key, done=True, status="done")
            ctx.journal.append("done", source=key)
            return True
        f = item["files"][index]
        ctx.state.update_source(key, status="downloading", file=index, offset=offset, name=item["id"], topic=item["topic"], files=len(item["files"]))
        ctx.journal.append("download", source=key, file=f["path"], bytes=f["size"])
        dest = download_dir(ctx.root, item["id"]) / PurePosixPath(f["path"]).name
        ctx.reserve(f["size"])
        try:
            fetch(ctx.http, resolve_url(item["id"], item["sha"], f["path"]), dest, f["size"], f.get("sha256", ""), ctx.stop,
                  lambda n: ctx.state.update_source(key, downloaded=n))
        except ValueError as error:      # checksum mismatch: skip this file, keep going
            ctx.journal.append("error", source=key, file=f["path"], message=str(error))
            ctx.state.update_source(key, file=index + 1, offset=0)
            return True
        ctx.state.update_source(key, status="reading")
        done = offset
        try:
            for n, row in enumerate(iter_rows(dest)):
                if n < offset:
                    continue
                ctx.stop()
                for cand in row_to_candidates(row, item, f"{f['path']}#{n}"):
                    ctx.pipeline.offer(cand)
                done = n + 1
                if ctx.pipeline.full():
                    ctx.flush()
                    ctx.state.update_source(key, offset=done)
        except (OSError, ValueError, EOFError, RuntimeError) as error:
            ctx.journal.append("error", source=key, file=f["path"], message=f"unreadable: {error}")
        finally:
            ctx.flush()
            ctx.state.update_source(key, offset=done)    # an interruption resumes here; a finished file resets it below
        dest.unlink(missing_ok=True)
        ctx.state.update_source(key, file=index + 1, offset=0, downloaded=0, status="reading")
        return True


__all__ = ["Adapter", "Item", "build_plan", "fetch", "iter_rows", "row_to_candidates", "LimitReached"]
