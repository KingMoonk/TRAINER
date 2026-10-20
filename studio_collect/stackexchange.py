"""Stack Exchange question-and-answer pairs from the official data dumps (CC BY-SA, attribution kept per record).

Only the dump is read: no page of any Stack Exchange site is crawled. A dump is one archive per site on archive.org
(`<site>.7z`; Stack Overflow splits it per table, `stackoverflow.com-Posts.7z`), holding Posts.xml. Each question is paired with
its accepted answer, or else its top answer scoring at least `min_score`. The licence is CC BY-SA 2.5, 3.0 or 4.0 by the date
the post was made (Stack Exchange's attribution policy), and every record names the question and answer links and authors.

Reading a .7z needs the py7zr package or a 7-Zip command (`7z`, `7zz`, `7za`) on the PATH; without either the site is reported as
unreadable with that reason. A .zip or an already extracted Posts.xml is read directly.
"""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from studio_collect import htmltext
from studio_collect.config import collect_dir, slug
from studio_collect.hub import fetch
from studio_collect.pipeline import Candidate

CC_BY_SA = (("2011-04-08", "cc-by-sa-2.5"), ("2018-05-02", "cc-by-sa-3.0"))   # posts before the first date, before the second, else 4.0


def dump_name(site):
    return "stackoverflow.com-Posts.7z" if site == "stackoverflow.com" else f"{site}.7z"


def dump_url(base, site):
    return f"{base.rstrip('/')}/{dump_name(site)}"


def licence_for(created):
    """The CC BY-SA version a post made on `created` (an ISO date) is under."""
    day = (created or "")[:10]
    for before, licence in CC_BY_SA:
        if day and day < before:
            return licence
    return "cc-by-sa-4.0"


def older(a, b):
    """The stricter (older) of two CC BY-SA versions: a pair is only as free as its older half."""
    return min(a, b, key=lambda v: v.rsplit("-", 1)[-1])


# ---- reading the archive -----------------------------------------------------------------------------------

def seven_zip_command():
    return next((c for c in ("7z", "7zz", "7za") if shutil.which(c)), None)


def extract_posts(archive, out_dir, stop=lambda: None):
    """Posts.xml from a dump archive (7-Zip or zip, recognised by content, or the .xml itself) as a file in out_dir. RuntimeError, in words, when it cannot be read."""
    archive, out_dir = Path(archive), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "Posts.xml"
    with archive.open("rb") as stream:
        head = stream.read(16)
    if head.lstrip()[:5] in (b"<?xml", b"<post"):
        return archive
    if target.exists() and target.stat().st_size:
        return target
    part = out_dir / "Posts.xml.part"
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z, z.open("Posts.xml") as src, part.open("wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
    else:
        try:
            import py7zr
        except ImportError:
            py7zr = None
        if py7zr is not None:
            with py7zr.SevenZipFile(archive) as z:
                z.extract(path=out_dir, targets=["Posts.xml"])
            part = target
        elif (command := seven_zip_command()):
            with part.open("wb") as dst:
                done = subprocess.run([command, "e", "-so", str(archive), "Posts.xml"], stdout=dst, stderr=subprocess.PIPE, check=False)
            if done.returncode != 0:
                part.unlink(missing_ok=True)
                raise RuntimeError(f"7-Zip could not read {archive.name}: {done.stderr.decode(errors='replace')[:200]}")
        else:
            raise RuntimeError("Reading Stack Exchange dumps needs the py7zr package or 7-Zip (7z) on the PATH; neither was found.")
    stop()
    if part != target:
        part.replace(target)
    return target


# ---- pairing -----------------------------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS q (id INTEGER PRIMARY KEY, accepted INTEGER, score INTEGER, title TEXT, body TEXT, tags TEXT, owner INTEGER, created TEXT);
CREATE TABLE IF NOT EXISTS a (id INTEGER PRIMARY KEY, parent INTEGER, score INTEGER, body TEXT, owner INTEGER, created TEXT);
CREATE INDEX IF NOT EXISTS a_parent ON a(parent, score DESC);
"""


def load_posts(posts_xml, db_path, min_score, stop=lambda: None):
    """Stream Posts.xml into a scratch SQLite file: questions with a non-negative score, and answers that are accepted or score at least min_score."""
    Path(db_path).unlink(missing_ok=True)
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)
    n = 0
    for _, row in ElementTree.iterparse(posts_xml, events=("end",)):
        if row.tag != "row":
            continue
        a = row.attrib
        n += 1
        if n % 20000 == 0:
            stop()
            db.commit()
        kind, score = a.get("PostTypeId"), int(a.get("Score") or 0)
        if kind == "1" and score >= 0:
            db.execute("INSERT OR REPLACE INTO q VALUES (?,?,?,?,?,?,?,?)", (int(a["Id"]), int(a["AcceptedAnswerId"]) if a.get("AcceptedAnswerId") else None, score,
                                                                              a.get("Title", ""), a.get("Body", ""), a.get("Tags", ""), int(a.get("OwnerUserId") or 0), a.get("CreationDate", "")))
        elif kind == "2" and a.get("ParentId"):
            keep = score >= min_score or db.execute("SELECT 1 FROM q WHERE accepted=?", (int(a["Id"]),)).fetchone()
            if keep:
                db.execute("INSERT OR REPLACE INTO a VALUES (?,?,?,?,?,?)", (int(a["Id"]), int(a["ParentId"]), score, a.get("Body", ""), int(a.get("OwnerUserId") or 0), a.get("CreationDate", "")))
        row.clear()
    db.commit()
    return db


def pairs(db, min_score, after=0):
    """(rowid, question row, answer row) for each question with an accepted answer or a top answer scoring min_score or more, in order."""
    cursor = db.execute("SELECT rowid, id, accepted, score, title, body, tags, owner, created FROM q WHERE rowid > ? ORDER BY rowid", (after,))
    for rowid, qid, accepted, qscore, title, body, tags, owner, created in cursor:
        answer = db.execute("SELECT id, score, body, owner, created FROM a WHERE id=?", (accepted,)).fetchone() if accepted else None
        if answer is None:
            answer = db.execute("SELECT id, score, body, owner, created FROM a WHERE parent=? AND score>=? ORDER BY score DESC, id LIMIT 1", (qid, min_score)).fetchone()
        if answer:
            yield rowid, (qid, qscore, title, body, tags, owner, created), answer


def candidate(site, q, a):
    qid, qscore, title, body, tags, qowner, qcreated = q
    aid, ascore, abody, aowner, acreated = a
    url = f"https://{site}/questions/{qid}"
    licence = older(licence_for(qcreated), licence_for(acreated))
    question = htmltext.to_text(f"<p>{title}</p>{body}")
    author = lambda uid: f"https://{site}/users/{uid}" if uid else f"a deleted user of {site}"  # noqa: E731
    return Candidate(
        kind="qa", source=f"stackexchange:{site}", group=f"stackexchange-{slug(site)}", url=url, licence=licence, source_id=f"{qid}/{aid}",
        question=question, answer=htmltext.to_text(abody),
        attribution=f"Question {url} by {author(qowner)} and answer {url}/{aid}#{aid} by {author(aowner)} on {site} (Stack Exchange), licensed {licence.upper().replace('-SA-', '-SA ').replace('CC-BY', 'CC BY')}.",
        meta={"site": site, "question_id": qid, "answer_id": aid, "question_score": qscore, "answer_score": ascore, "tags": [t for t in tags.strip("|").split("|") if t],
              "question_licence": licence_for(qcreated), "answer_licence": licence_for(acreated)})


# ---- the adapter -------------------------------------------------------------------------------------------

class Adapter:
    key = "stackexchange"

    def __init__(self, ctx):
        self.ctx = ctx

    @staticmethod
    def plan_inputs(config):
        return {"sites": config["stackexchange"]["sites"], "url": config["stackexchange"]["dump_url"], "approve": config["limits"]["auto_approve_mb"],
                "file": config["limits"]["max_file_mb"]}

    def plan(self):
        ctx, settings = self.ctx, self.ctx.config["stackexchange"]
        items = []
        for site in settings["sites"]:
            ctx.stop()
            url = dump_url(settings["dump_url"], site)
            item = {"id": site, "source": "stackexchange", "topic": "Q&A", "group": f"stackexchange-{slug(site)}", "licence": "cc-by-sa", "licence_class": "share-alike",
                    "url": url, "downloads": 0, "likes": 0, "tags": [], "score": 0.5, "fit": 1.0, "files": [], "total_bytes": 0, "author": "Stack Exchange contributors"}
            try:
                size = ctx.http.size(url)
            except (OSError, ValueError) as error:
                size = None
                item.update(status="skipped", reason=f"dump not reachable: {error}", bytes=0)
            if size is not None:
                item.update(bytes=size, total_bytes=size, files=[{"path": dump_name(site), "size": size}])
                if size > ctx.config["limits"]["max_file_mb"] * 1_000_000:
                    item.update(status="skipped", reason=f"{size / 1e6:,.0f} MB is above the {ctx.config['limits']['max_file_mb']:g} MB file limit")
                elif site in ctx.config["approved"] or size <= ctx.config["limits"]["auto_approve_mb"] * 1_000_000:
                    item.update(status="planned", reason="")
                else:
                    item.update(status="needs_approval", reason=f"{size / 1e6:,.0f} MB to download: approve it to start")
            elif "status" not in item:
                item.update(status="skipped", reason="the dump's size is unknown", bytes=0)
            items.append(item)
        ctx.set_plan("stackexchange", items)

    def pending(self):
        ctx = self.ctx
        for item in ctx.plan_items("stackexchange"):
            if item["status"] == "skipped" or item["id"] in ctx.config["blocked"] or ctx.state.source(f"stackexchange:{item['id']}").get("done"):
                continue
            if item["status"] == "needs_approval" and item["id"] not in ctx.config["approved"]:
                continue
            yield item

    def step(self):
        ctx = self.ctx
        item = next(self.pending(), None)
        if item is None:
            return False
        site, key = item["id"], f"stackexchange:{item['id']}"
        folder = collect_dir(ctx.root) / "downloads" / f"stackexchange-{slug(site)}"
        archive = folder / dump_name(site)
        ctx.state.update_source(key, status="downloading", name=site, topic="Q&A")
        ctx.reserve(item["bytes"] * 4)         # the archive, Posts.xml and the scratch database
        fetch(ctx.http, item["url"], archive, item["bytes"], "", ctx.stop, lambda n: ctx.state.update_source(key, downloaded=n))
        ctx.state.update_source(key, status="unpacking")
        try:
            posts = extract_posts(archive, folder, ctx.stop)
        except RuntimeError as error:
            ctx.journal.append("error", source=key, message=str(error))
            ctx.state.update_source(key, status="failed", done=True, error=str(error))
            return True
        ctx.state.update_source(key, status="reading")
        min_score = ctx.config["stackexchange"]["min_score"]
        scratch = folder / "pairs.sqlite"
        saved = ctx.state.source(key)
        if not scratch.exists() or saved.get("phase") != "pairs":
            db = load_posts(posts, scratch, min_score, ctx.stop)
            ctx.state.update_source(key, phase="pairs", offset=0)
        else:
            db = sqlite3.connect(scratch)
        done = saved.get("offset", 0)
        try:
            for rowid, q, a in pairs(db, min_score, done):
                ctx.stop()
                ctx.pipeline.offer(candidate(site, q, a))
                done = rowid
                if ctx.pipeline.full():
                    ctx.flush()
                    ctx.state.update_source(key, offset=done)
        finally:
            ctx.flush()
            ctx.state.update_source(key, offset=done)
            db.close()
        for leftover in (scratch, posts if posts.parent == folder else None, archive):
            if leftover is not None:
                Path(leftover).unlink(missing_ok=True)
        ctx.state.update_source(key, status="done", done=True, downloaded=0)
        ctx.journal.append("done", source=key)
        return True
