"""A polite crawler for the websites the person allows (and only those).

Each site is listed in the settings with the licence its terms give (the person read them; the Collect page says so), seed pages and a page
cap. The crawler fetches a page at a time, follows links on the same host only, and:
- asks robots.txt before every fetch and obeys Disallow and Crawl-delay (studio_collect.robots);
- spaces requests to a host (the settings' pause, or the site's crawl-delay if longer) and counts them against the daily caps;
- never requests X, Quora or Reddit, whatever a seed, a redirect or a link says (Http refuses them);
- sends the User-Agent that names the app;
- skips pages that say noindex, noai or none in a robots meta tag or X-Robots-Tag header, and follows no links where they say nofollow.
A page that marks up a question and its answers (schema.org QAPage, in JSON-LD) becomes question-and-answer pairs; any other page
becomes a text document. The frontier is a SQLite file, so a crawl resumes where it stopped.
"""
from __future__ import annotations

import sqlite3
from urllib.parse import urlsplit

from studio_collect import htmltext
from studio_collect.config import collect_dir, slug
from studio_collect.netio import Refused
from studio_collect.pipeline import Candidate
from studio_collect.robots import Robots

MAX_DEPTH = 5
MAX_PAGE_BYTES = 5_000_000
BINARY = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".zip", ".gz", ".7z", ".tar", ".mp3", ".mp4", ".avi", ".mov", ".css", ".js", ".ico", ".xml", ".json", ".woff", ".woff2")
STOP_WORDS = ("noindex", "noai", "none")


class Frontier:
    def __init__(self, root):
        path = collect_dir(root) / "crawl.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("CREATE TABLE IF NOT EXISTS pages (host TEXT NOT NULL, url TEXT NOT NULL, depth INTEGER NOT NULL, status TEXT NOT NULL, PRIMARY KEY(host, url))")

    def add(self, host, url, depth, limit):
        if self.db.execute("SELECT COUNT(*) FROM pages WHERE host=?", (host,)).fetchone()[0] < limit:
            self.db.execute("INSERT OR IGNORE INTO pages VALUES (?,?,?,'new')", (host, url, depth))

    def next(self, host):
        return self.db.execute("SELECT url, depth FROM pages WHERE host=? AND status='new' ORDER BY depth, rowid LIMIT 1", (host,)).fetchone()

    def mark(self, host, url, status):
        self.db.execute("UPDATE pages SET status=? WHERE host=? AND url=?", (status, host, url))
        self.db.commit()

    def done(self, host):
        return self.db.execute("SELECT COUNT(*) FROM pages WHERE host=? AND status='done'", (host,)).fetchone()[0]

    def close(self):
        self.db.commit()
        self.db.close()


def followable(url, host):
    parts = urlsplit(url)
    return (parts.scheme in ("http", "https") and (parts.hostname or "").lower() == host and not parts.query
            and not parts.path.lower().endswith(BINARY))


class Adapter:
    key = "web"

    def __init__(self, ctx):
        self.ctx = ctx
        self.robots = Robots(ctx.http, ctx.clock)
        ctx.http.guard = self.robots.guard
        self.frontier = Frontier(ctx.root)

    @staticmethod
    def plan_inputs(config):
        return {"sites": config["web"]["sites"]}

    def close(self):
        self.frontier.close()

    def sites(self):
        return {s["host"]: s for s in self.ctx.config["web"]["sites"]}

    def plan(self):
        ctx, items = self.ctx, []
        for host, site in self.sites().items():
            for seed in site["seeds"]:
                self.frontier.add(host, seed, 0, site["pages"] * 3)
            self.frontier.db.commit()
            items.append({"id": host, "source": "web", "topic": "Web", "group": f"web-{slug(host)}", "licence": site["licence"], "licence_class": "",
                          "url": site["seeds"][0], "bytes": 0, "total_bytes": 0, "files": [], "downloads": 0, "likes": 0, "tags": [], "fit": 1.0, "score": 0.5,
                          "author": host, "status": "planned", "reason": f"up to {site['pages']} pages; licence as you stated it", "pages": site["pages"]})
        ctx.set_plan("web", items)

    def pending(self):
        ctx = self.ctx
        for item in ctx.plan_items("web"):
            site = self.sites().get(item["id"])
            if site is None or item["id"] in ctx.config["blocked"] or ctx.state.source(f"web:{item['id']}").get("done"):
                continue
            yield item, site

    def step(self):
        ctx = self.ctx
        for item, site in self.pending():
            host, key = item["id"], f"web:{item['id']}"
            if self.frontier.done(host) >= site["pages"] or (page := self.frontier.next(host)) is None:
                ctx.state.update_source(key, status="done", done=True, name=host, topic="Web")
                ctx.journal.append("done", source=key, pages=self.frontier.done(host))
                continue
            self.fetch(host, site, key, *page)
            return True
        return False

    def fetch(self, host, site, key, url, depth):
        ctx = self.ctx
        ctx.state.update_source(key, status="crawling", name=host, topic="Web", pages=self.frontier.done(host))
        try:
            reply = ctx.http.open(url, robots=True, headers={"Accept": "text/html,application/xhtml+xml"})
        except Refused as refused:
            ctx.journal.append("skip", source=key, url=url, why=str(refused))
            self.frontier.mark(host, url, "blocked")
            return
        except OSError as error:
            ctx.journal.append("note", text=f"{url}: {error}")
            self.frontier.mark(host, url, "failed")
            return
        try:
            kind = reply.headers.get("content-type", "").lower()
            if reply.status != 200 or not ("html" in kind):
                self.frontier.mark(host, url, "skipped")
                return
            body = bytearray()
            for block in reply.chunks():
                body += block
                if len(body) > MAX_PAGE_BYTES:
                    break
        finally:
            reply.close()
        header = reply.headers.get("x-robots-tag", "").lower()
        page = htmltext.parse(bytes(body).decode("utf-8", errors="replace"), url)
        directives = " ".join([header, *page.robots])
        if any(word in directives for word in STOP_WORDS):
            self.frontier.mark(host, url, "skipped")
            ctx.journal.append("skip", source=key, url=url, why="the page asks not to be indexed or used for AI")
            return
        self.extract(host, site, key, url, page)
        if "nofollow" not in directives and depth < MAX_DEPTH:
            for link in page.links:
                if followable(link, host):
                    self.frontier.add(host, link, depth + 1, site["pages"] * 3)
        self.frontier.mark(host, url, "done")
        if ctx.pipeline.full():
            ctx.flush()

    def extract(self, host, site, key, url, page):
        ctx = self.ctx
        title = page.title.strip()
        attribution = f"{title or url} ({url}), content of {host} under {site['licence']}"
        base = dict(source=key, group=f"web-{slug(host)}", url=url, licence=site["licence"], meta={"host": host, "title": title})
        pairs = [pair for blob in page.jsonld for pair in htmltext.qa_pairs(blob)]
        if pairs:
            for question, answer, answer_url, author in pairs:
                ctx.pipeline.offer(Candidate(kind="qa", question=question, answer=answer, source_id=answer_url or url,
                                             attribution=attribution + (f"; answer by {author}" if author else ""), **base))
        else:
            ctx.pipeline.offer(Candidate(kind="text", text=(f"{title}\n\n{page.text}" if title and not page.text.startswith(title) else page.text), source_id=url,
                                         attribution=attribution, **base))
