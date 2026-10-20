"""What the Collect page shows, as plain dicts: the options, the settings, and the live state read from data/collect/.

Read-only apart from `set_config` and the pause/resume writes; none of it starts a network request. The service process
writes the state; the app process (this one) reads it and registers the shards (studio_collect.register).
"""
from __future__ import annotations

import time
from pathlib import Path

from studio_collect import config as cfg
from studio_collect import licences, policy, register
from studio_collect.config import KIND_LABELS, SOURCES, TOPICS, collect_dir
from studio_collect.store import Journal, Ledger, State, disk_used

NOTES = {
    "hub": ("Hugging Face Hub datasets, found by topic and ranked by licence, topic fit, popularity and size.",
            "Each dataset's own licence; permissive and share-alike are kept. Gated datasets are not collected: they need your own token and your acceptance of their terms on the hub, and this app holds no token.",
            ""),
    "stackexchange": ("Question and accepted-answer pairs from the official Stack Exchange data dumps (no site page is fetched).",
                      "CC BY-SA 2.5, 3.0 or 4.0 by the date of each post, with the question and answer links and authors kept as attribution.",
                      "Reading a .7z dump needs the py7zr package or 7-Zip on this computer."),
    "github": ("Code and README text from repositories with a permissive licence, through GitHub's official API.",
               "Only MIT, Apache-2.0, BSD, ISC, Unlicense, CC0 and similar. Issues and discussions are not collected: a repository's licence covers its files, not what others write in its tracker.",
               "Without a token GitHub allows 60 API calls an hour, so this source is slow."),
    "web": ("Pages from websites you list, with question-and-answer pairs taken from QAPage markup where a page has it.",
            "The licence you read in each site's terms and enter for it. robots.txt and crawl-delay are obeyed, requests to one site are spaced, and the User-Agent names this app.",
            "Add only sites whose terms allow collecting their pages for training."),
}
EVENTS = {"start": "Collector started", "exit": "Collector stopped", "ready": "Ready", "limit": "Waiting on a cap", "done": "Finished a source",
          "download": "Downloading", "plan": "Planned", "skip": "Skipped a page", "error": "Problem", "note": "Note"}


def options():
    """Everything fixed: the topics, the sources with their terms notes, the excluded sites, the licence classes."""
    from studio_collect import stackexchange
    can7z = _can_read_7z(stackexchange)
    return {
        "topics": [{"id": key, "label": label} for key, (label, _) in TOPICS.items()],
        "sources": [{"id": key, "label": KIND_LABELS[key], "what": NOTES[key][0], "terms": NOTES[key][1], "note": NOTES[key][2] if key != "stackexchange" or not can7z else ""}
                    for key in SOURCES],
        "excluded": policy.excluded_sites(),
        "licences": [{"id": licences.PERMISSIVE, "label": "Permissive (MIT, Apache, CC BY ...)", "default": True, "opt_in": ""},
                     {"id": licences.SHARE_ALIKE, "label": "Share-alike (CC BY-SA, ODbL ...)", "default": True, "opt_in": ""},
                     {"id": licences.COPYLEFT, "label": "Copyleft (GPL family)", "default": False, "opt_in": "copyleft"},
                     {"id": licences.NON_COMMERCIAL, "label": "Non-commercial or no-derivatives", "default": False, "opt_in": "non_commercial"},
                     {"id": licences.UNKNOWN, "label": "Unknown or not stated", "default": False, "opt_in": "unknown"}],
        "user_agent": policy.user_agent(),
    }


def _can_read_7z(stackexchange):
    try:
        import py7zr  # noqa: F401
        return True
    except ImportError:
        return stackexchange.seven_zip_command() is not None


def get_config(root):
    return cfg.load(root)


def set_config(root, overrides):
    return cfg.save(root, overrides)


def set_paused(root, paused):
    return cfg.save(root, {"paused": bool(paused), **({"enabled": True} if not paused else {})})


def _event_text(row):
    kind = row.get("type", "")
    base = EVENTS.get(kind, kind)
    detail = row.get("text") or row.get("message") or row.get("why") or row.get("source") or ""
    if kind == "download" and row.get("file"):
        detail = f"{row.get('source', '')} · {row['file']}"
    if kind == "plan" and row.get("count") is not None:
        base, detail = "Plan ready", f"{row['count']} {row.get('source', '')} items"
    if kind == "ready" and row.get("known_records"):
        detail = f"{row['known_records']:,} records already in your data are skipped as duplicates"
    return f"{base}: {detail}" if detail else base


def state(root, running=None, now=time.time):
    """The live picture: settings flags, service status, counters, the plan, each source's progress, disk and today's use, the journal."""
    root = Path(root)
    config, saved = cfg.load(root), State(root).data
    done = register.totals(root)
    waiting = sum(len(files) for files in register.pending(root).values())
    ledger = Ledger(root).today()
    sets = []
    for group, stats in saved.get("groups", {}).items():
        entry = done.get(register.dataset_id(group), {})
        sets.append({"id": register.dataset_id(group), "group": group, "name": register.dataset_name(group), "kept": stats["kept"], "bytes": stats["bytes_kept"],
                     "records": entry.get("records", 0), "dropped": stats["dropped"], "redactions": stats["redactions"], "seen": stats["seen"]})
    plan = []
    for item in saved.get("plan", []):
        plan.append({"id": item["id"], "source": item.get("source", ""), "topic": item.get("topic", ""), "group": item.get("group", ""), "licence": item.get("licence", ""),
                     "licence_class": item.get("licence_class", ""), "bytes": item.get("bytes", 0), "status": item.get("status", ""), "reason": item.get("reason", ""),
                     "score": item.get("score", 0), "url": item.get("url", ""), "files": len(item.get("files", [])),
                     "approved": item["id"] in config["approved"], "blocked": item["id"] in config["blocked"]})
    sources = []
    for key, entry in saved.get("sources", {}).items():
        if key == "github:api":
            continue
        kind = key.split(":", 1)[0]
        detail = (f"file {entry.get('file', 0) + 1} of {entry['files']}" if entry.get("files") and kind == "hub" else f"{entry['pages']} pages" if entry.get("pages") else "")
        sources.append({"key": key, "kind": kind, "name": entry.get("name", key.split(":", 1)[-1]), "topic": entry.get("topic", ""), "status": entry.get("status", ""),
                        "detail": detail, "downloaded": entry.get("downloaded", 0), "done": bool(entry.get("done")), "error": entry.get("error", "")})
    heartbeat = saved.get("heartbeat")
    journal = [{"at": row.get("at", ""), "type": row.get("type", ""), "text": _event_text(row)} for row in Journal(root).tail(40)]
    status = saved.get("status", "off")
    if running is False and status in ("running", "waiting", "paused", "idle"):
        status = "off"       # the state file outlives the process: nobody is running it
    return {"enabled": config["enabled"], "paused": config["paused"], "running": bool(running), "status": status, "reason": saved.get("reason", "") if status == saved.get("status") else "",
            "heartbeat_age": None if not heartbeat else max(0.0, now() - heartbeat), "stats": saved.get("stats", {}), "sets": sets, "plan": plan, "sources": sources,
            "disk_used": disk_used(root), "disk_cap": int(config["limits"]["disk_mb"] * 1_000_000),
            "bandwidth_today": ledger["bytes"], "bandwidth_cap": int(config["limits"]["bandwidth_mb_per_day"] * 1_000_000), "bandwidth_total": ledger["total_bytes"],
            "requests_today": ledger["requests"], "waiting_shards": waiting, "journal": journal,
            "file": str(collect_dir(root))}
