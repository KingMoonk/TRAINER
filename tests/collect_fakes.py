"""Fakes for the collector's tests: a web that serves bytes (with Range), a fake Hugging Face hub, and probes that say "idle".

No test here opens a socket: Http takes a transport, and this is it.
"""
import hashlib
import json
import random
from urllib.parse import unquote

from studio_collect.idle import Probes
from studio_collect.netio import Reply

WORDS = ("energy field particle theory proof lattice protein cell enzyme orbit planet algebra matrix vector integral theorem model data "
         "signal network memory language history empire treaty court statute patient dose trial method result system function variable "
         "process material reaction solution measure sample error range value pattern cluster surface volume angle curve series limit").split()


def prose(i, sentences=8):
    """A distinct, ordinary-looking paragraph for document number i."""
    rng = random.Random(i)
    return " ".join(" ".join(rng.choice(WORDS) for _ in range(rng.randint(9, 15))).capitalize() + "." for _ in range(sentences))


def idle_probes(**override):
    values = dict(on_battery=lambda: False, cpu=lambda: 0.0, queue_busy=lambda root: False, metered=lambda: False)
    values.update(override)
    return Probes(**values)


class FakeWeb:
    """transport(url, headers) -> Reply. add(prefix, body | callable(url, headers) -> (status, body)). `log` has every URL asked for.
    `cut_after`: bytes after which the next large body raises OSError once (a dropped connection)."""

    def __init__(self):
        self.routes, self.log, self.headers_log, self.cut_after = [], [], [], None

    def add(self, prefix, body, status=200, headers=None):
        self.routes.append((prefix, body, status, headers or {}))

    def __call__(self, url, headers):
        self.log.append(url)
        self.headers_log.append(headers)
        for prefix, body, status, extra in sorted(self.routes, key=lambda r: -len(r[0])):
            if url.startswith(prefix):
                if callable(body):
                    status, body = body(url, headers)
                start, total, extra = 0, len(body), dict(extra)
                if "Range" in headers and status == 200:
                    first, last = headers["Range"].split("=")[1].split("-")
                    start = int(first)
                    end = int(last) if last else total - 1
                    body, status = body[start:end + 1], 206
                    extra["Content-Range"] = f"bytes {start}-{start + len(body) - 1}/{total}"
                return Reply(status, {"Content-Length": str(len(body)), **extra}, self._chunks(body))
        return Reply(404, {}, iter(()))

    def _chunks(self, body):
        cut, self.cut_after = self.cut_after, None
        if cut is not None and len(body) > cut:
            def broken():
                yield body[:cut]
                raise OSError("connection dropped")
            return broken()
        return iter([body[i:i + 4096] for i in range(0, len(body), 4096)] or [b""])

    def asked(self, needle):
        return [u for u in self.log if needle in u]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def hub_entry(name, license="apache-2.0", downloads=1000, tags=(), gated=False, **extra):
    return {"id": name, "author": name.split("/")[0], "downloads": downloads, "likes": 5, "gated": gated, "private": False,
            "sha": "a" * 40, "tags": [f"license:{license}", "modality:text", *tags], "cardData": {"license": license}, "description": "", **extra}


def jsonl(rows):
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode()


def tree_entry(path, data, lfs=True):
    entry = {"type": "file", "path": path, "size": len(data)}
    if lfs:
        entry["lfs"] = {"oid": sha256(data), "size": len(data)}
    return entry


def serve_hub(web, datasets):
    """datasets: {id: (entry dict, {path: bytes})}. Serves the search, the file list and the files."""
    def search(url, headers):
        term = unquote(url.split("search=")[1].split("&")[0]).lower()
        return 200, json.dumps([e for e, _ in datasets.values() if term in e["id"].lower() or term in " ".join(e["tags"]).lower()]).encode()
    web.add("https://huggingface.co/api/datasets?", search)
    for name, (entry, files) in datasets.items():
        web.add(f"https://huggingface.co/api/datasets/{name}/tree/", json.dumps([tree_entry(p, d) for p, d in files.items()]).encode())
        for path, data in files.items():
            web.add(f"https://huggingface.co/datasets/{name}/resolve/{entry['sha']}/{path}", data)
