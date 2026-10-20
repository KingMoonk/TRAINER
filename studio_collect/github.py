"""Code and README text from GitHub repositories with a permissive licence, through the official API.

Only permissive licences (MIT, Apache-2.0, BSD, ISC, Unlicense, CC0 ...) are read, whatever the licence opt-ins say: a repository with
a copyleft, unknown or missing licence is listed in the plan as skipped and never fetched. Searching and listing files use the API
with its rate limit honoured (x-ratelimit-* and retry-after; when it is used up the collector waits for the reset the API names);
file contents come from raw.githubusercontent.com. No token is read or sent: the repo has no settings or keyring mechanism for one,
so the unauthenticated limit applies (60 API calls an hour, 10 searches a minute).

Issues and discussions are not collected. A repository's licence covers its files, not what other people write in its issue tracker
(GitHub's terms leave that with its authors), so the "where the repository licence covers them" condition is never met.
"""
from __future__ import annotations

import json
from pathlib import PurePosixPath
from urllib.parse import quote

from studio_collect import licences
from studio_collect.config import topic_list
from studio_collect.pipeline import Candidate
from studio_collect.store import LimitReached

API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"
HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
PER_QUERY = 30
MAX_FILES, MAX_FILE_BYTES, MAX_REPO_BYTES = 40, 100_000, 1_500_000
CODE = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".scala",
        ".sh", ".lua", ".r", ".jl", ".hs", ".ml", ".sql"}
DOCS = {".md", ".rst", ".txt"}
SKIP_PARTS = {"node_modules", "vendor", "third_party", "dist", "build", "target", "__pycache__", ".git", "migrations", "fixtures", "testdata", "generated", "site-packages"}
SKIP_NAMES = ("package-lock.json", "yarn.lock", "poetry.lock", "cargo.lock", ".min.js", ".min.css", ".pb.go", "_pb2.py", ".d.ts", "license", "licence", "copying", "notice")


def wanted(path, size):
    """Whether a file of a repository is worth reading: README and docs text, and source files that are not vendored or generated."""
    p = PurePosixPath(path)
    low = path.lower()
    if size <= 0 or size > MAX_FILE_BYTES or SKIP_PARTS & {part.lower() for part in p.parts[:-1]} or low.endswith(SKIP_NAMES) or p.name.lower() in SKIP_NAMES:
        return False
    if p.suffix.lower() in DOCS:
        return p.name.lower().startswith("readme") or (p.parts[0].lower() in ("docs", "doc") and p.suffix.lower() == ".md")
    return p.suffix.lower() in CODE


def choose(tree):
    """The files to read from a git tree listing: READMEs first, then docs, then source, smallest first, within the per-repository caps."""
    files = [e for e in tree if e.get("type") == "blob" and wanted(e["path"], int(e.get("size") or 0))]
    files.sort(key=lambda e: (0 if PurePosixPath(e["path"]).name.lower().startswith("readme") else 1 if PurePosixPath(e["path"]).suffix.lower() in DOCS else 2, int(e.get("size") or 0)))
    chosen, total = [], 0
    for e in files:
        if len(chosen) >= MAX_FILES or total + int(e["size"]) > MAX_REPO_BYTES:
            continue
        chosen.append(e)
        total += int(e["size"])
    return chosen


class Adapter:
    key = "github"

    def __init__(self, ctx):
        self.ctx = ctx

    @staticmethod
    def plan_inputs(config):
        return {"queries": config["github"]["queries"], "max": config["github"]["max_repos"], "topics": config["topics"], "custom": config["custom_topics"]}

    # ---- the API with its rate limit honoured --------------------------------------------------------------
    def api(self, url):
        """The parsed JSON of an API call. LimitReached when the rate limit is used up (until the reset the API names)."""
        ctx = self.ctx
        saved = ctx.state.source("github:api")
        if saved.get("blocked_until", 0) > ctx.clock():
            raise LimitReached(self._wait_text(saved["blocked_until"], ctx.clock()))
        reply = ctx.http.open(url, headers=HEADERS)
        try:
            body = b"".join(reply.chunks())
        finally:
            reply.close()
        h = reply.headers
        remaining, reset = h.get("x-ratelimit-remaining"), h.get("x-ratelimit-reset")
        retry = h.get("retry-after")
        until = ctx.clock() + float(retry) if retry and retry.replace(".", "", 1).isdigit() else float(reset) if reset and reset.isdigit() else 0
        if reply.status in (403, 429) and (remaining == "0" or retry):
            ctx.state.update_source("github:api", blocked_until=until or ctx.clock() + 3600)
            raise LimitReached(self._wait_text(until or ctx.clock() + 3600, ctx.clock()))
        if remaining == "0":
            ctx.state.update_source("github:api", blocked_until=until)     # this answer is good; the next call waits
        if reply.status != 200:
            raise OSError(f"GitHub answered {reply.status} for {url.split('?')[0]}")
        return json.loads(body)

    @staticmethod
    def _wait_text(until, now):
        return f"GitHub's API rate limit is used up; it resets in about {max(1, round((until - now) / 60))} minutes."

    # ---- plan ----------------------------------------------------------------------------------------------
    def queries(self):
        settings = self.ctx.config["github"]
        return settings["queries"] or [terms[0] for _, _, terms in topic_list(self.ctx.config)]

    def plan(self):
        ctx, items, seen, failed = self.ctx, [], set(), 0
        limit = ctx.config["github"]["max_repos"]
        for query in self.queries():
            ctx.stop()
            try:
                found = self.api(f"{API}/search/repositories?q={quote(query)}+fork:false+archived:false&sort=stars&order=desc&per_page={PER_QUERY}")
            except LimitReached:
                if not items:
                    raise          # nothing planned yet: do not record an empty plan, try again after the reset
                break
            except (OSError, ValueError) as error:
                ctx.journal.append("note", text=f"GitHub search for {query} failed: {error}")
                failed += 1
                continue
            for repo in found.get("items", []):
                name = repo["full_name"]
                if name in seen or repo.get("fork") or repo.get("archived"):
                    continue
                seen.add(name)
                spdx = (repo.get("license") or {}).get("spdx_id") or ""
                ident, klass = licences.classify(spdx)
                item = {"id": name, "source": "github", "topic": query, "group": "github", "licence": ident, "licence_class": klass, "url": repo["html_url"],
                        "downloads": 0, "likes": int(repo.get("stargazers_count") or 0), "tags": [], "files": [], "total_bytes": int(repo.get("size") or 0) * 1024,
                        "bytes": min(MAX_REPO_BYTES, int(repo.get("size") or 0) * 1024), "author": (repo.get("owner") or {}).get("login", ""),
                        "branch": repo.get("default_branch", "main"), "fit": 1.0, "score": 0.0, "status": "planned", "reason": ""}
                if klass != licences.PERMISSIVE:
                    item.update(status="skipped", reason=licences.refusal(ident, klass) + ": only permissive repositories are read")
                elif name in ctx.config["blocked"]:
                    item.update(status="skipped", reason="you blocked it")
                items.append(item)
        if failed and not items:
            raise OSError("GitHub could not be reached.")
        usable = sorted((i for i in items if i["status"] == "planned"), key=lambda i: -i["likes"])[:limit]
        for rank, item in enumerate(usable):
            item["score"] = round(1 - rank / max(1, len(usable)), 3)
        rest = [i for i in items if i["status"] == "skipped"][:40]
        ctx.set_plan("github", usable + rest)

    # ---- one repository -----------------------------------------------------------------------------------
    def pending(self):
        ctx = self.ctx
        for item in ctx.plan_items("github"):
            if item["status"] == "planned" and item["id"] not in ctx.config["blocked"] and not ctx.state.source(f"github:{item['id']}").get("done"):
                yield item

    def step(self):
        ctx = self.ctx
        item = next(self.pending(), None)
        if item is None:
            return False
        name, key = item["id"], f"github:{item['id']}"
        ctx.state.update_source(key, status="reading", name=name, topic="code")
        tree = self.api(f"{API}/repos/{name}/git/trees/{quote(item['branch'], safe='')}?recursive=1")
        base = dict(source=key, group="github", licence=item["licence"], meta={"repo": name, "branch": item["branch"]})
        files, read = choose(tree.get("tree", [])), 0
        try:
            for index, entry in enumerate(files):
                if index < ctx.state.source(key).get("file", 0):
                    continue          # read before a restart
                ctx.stop()
                path = entry["path"]
                status, _, body = ctx.http.get(f"{RAW}/{name}/{quote(item['branch'], safe='')}/{quote(path)}", limit=MAX_FILE_BYTES * 2)
                if status != 200:
                    continue
                text = body.decode("utf-8", errors="replace")
                url = f"https://github.com/{name}/blob/{item['branch']}/{path}"
                read += 1
                ctx.pipeline.offer(Candidate(kind="text", text=text, url=url, source_id=path, attribution=f"{name} (https://github.com/{name}) by {item['author'] or 'its'} contributors, file {path}, licensed {item['licence']}.",
                                             **{**base, "meta": {**base["meta"], "path": path}}))
                ctx.state.update_source(key, file=index + 1)
                if ctx.pipeline.full():
                    ctx.flush()
        finally:
            ctx.flush()
        ctx.state.update_source(key, status="done", done=True, files=read)
        ctx.journal.append("done", source=key, files=read)
        return True
