"""The collector's settings (data/collect/config.json): topics, sources, licence opt-ins, caps and when it may run."""
from __future__ import annotations

import re
from pathlib import Path

from studio_core import atomic_json, read_json

# id -> (label, Hugging Face search terms). The person's own topics are free text searched as they are written.
TOPICS = {
    "science": ("Science", ("science", "physics", "chemistry", "biology")),
    "mathematics": ("Mathematics", ("mathematics", "math", "theorem")),
    "code": ("Code", ("code", "programming", "python")),
    "history": ("History", ("history", "historical")),
    "medicine": ("Medicine", ("medicine", "medical", "clinical")),
    "law": ("Law", ("law", "legal", "court")),
    "conversation": ("Conversation", ("conversation", "dialogue", "chat")),
    "instructions": ("Instructions", ("instruction", "instruct", "question answering")),
}
SOURCES = ("hub", "stackexchange", "github", "web")
KIND_LABELS = {"hub": "Hugging Face datasets", "stackexchange": "Stack Exchange dumps", "github": "GitHub (permissive repositories)", "web": "Allowed websites"}

DEFAULTS = {
    "enabled": False,           # the person turned collection on
    "paused": False,            # paused for now; Resume is instant
    "topics": list(TOPICS),     # chosen topic ids
    "custom_topics": [],        # the person's own topics (free text)
    "sources": {"hub": True, "stackexchange": False, "github": False, "web": False},
    "licences": {"non_commercial": False, "unknown": False, "copyleft": False},   # opt-ins; permissive and share-alike are always kept
    "limits": {
        "disk_mb": 4096,                    # everything the collector keeps (hand-off shards, downloads, dedup index)
        "bandwidth_mb_per_day": 2048,
        "requests_per_host_per_day": 5000,
        "seconds_between_requests": 2.0,    # per host; a robots.txt crawl-delay can only raise it
        "max_dataset_mb": 200,              # at most this much of one dataset is downloaded
        "max_file_mb": 1024,                # a single file larger than this is never fetched
        "auto_approve_mb": 100,             # a planned dataset above this waits for the person to approve it
    },
    "idle": {"only_when_idle": True, "only_on_mains": True, "unmetered_confirmed": False, "cpu_percent": 50},
    "approved": [],             # dataset ids the person approved although the plan asked first
    "blocked": [],              # dataset ids the person does not want
    "stackexchange": {"sites": ["math.stackexchange.com"], "min_score": 3, "dump_url": "https://archive.org/download/stackexchange"},
    "github": {"queries": [], "max_repos": 25},
    "web": {"sites": []},       # [{"host": "...", "seeds": ["https://..."], "licence": "cc-by-sa-4.0", "pages": 200}]
}
LIMIT_BOUNDS = {"disk_mb": (64, 10_000_000), "bandwidth_mb_per_day": (1, 10_000_000), "requests_per_host_per_day": (1, 1_000_000),
                "seconds_between_requests": (0.5, 3600), "max_dataset_mb": (1, 1_000_000), "max_file_mb": (1, 1_000_000), "auto_approve_mb": (0, 1_000_000)}


def collect_dir(root):
    return Path(root) / "data" / "collect"


def config_path(root):
    return collect_dir(root) / "config.json"


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:48] or "topic"


def _number(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number.")
    if not low <= value <= high:
        raise ValueError(f"{label} must be between {low:g} and {high:g}.")
    return value


def _strings(value, label, limit=200):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of text.")
    return [item.strip() for item in value if item.strip()][:limit]


def merge(overrides, base=None):
    """The settings with `overrides` (a partial dict, any depth) applied and checked. ValueError, in plain words, for a bad value."""
    config = {key: (dict(value) if isinstance(value, dict) else list(value) if isinstance(value, list) else value) for key, value in (base or DEFAULTS).items()}
    for key, value in (overrides or {}).items():
        if key not in DEFAULTS:
            raise ValueError(f"Unknown setting: {key}.")
        if key in ("enabled", "paused"):
            config[key] = bool(value)
        elif key == "topics":
            wanted = _strings(value, "Topics")
            bad = [t for t in wanted if t not in TOPICS]
            if bad:
                raise ValueError(f"Unknown topic: {bad[0]}.")
            config[key] = wanted
        elif key in ("custom_topics", "approved", "blocked"):
            config[key] = _strings(value, key.replace("_", " ").capitalize(), 100)
        elif key in ("sources", "licences", "idle"):
            if not isinstance(value, dict) or set(value) - set(DEFAULTS[key]):
                raise ValueError(f"{key.capitalize()} has an unknown field.")
            for name, item in value.items():
                config[key][name] = _number(item, name, 1, 100) if name == "cpu_percent" else bool(item)
        elif key == "limits":
            if not isinstance(value, dict) or set(value) - set(DEFAULTS[key]):
                raise ValueError("Limits has an unknown field.")
            for name, item in value.items():
                config[key][name] = _number(item, name.replace("_", " "), *LIMIT_BOUNDS[name])
        elif key == "stackexchange":
            config[key] = {**config[key], **_stack(value)}
        elif key == "github":
            config[key] = {**config[key], **_github(value)}
        elif key == "web":
            config[key] = {"sites": _sites(value.get("sites", []) if isinstance(value, dict) else None)}
    return config


def _stack(value):
    if not isinstance(value, dict) or set(value) - {"sites", "min_score", "dump_url"}:
        raise ValueError("Stack Exchange settings have an unknown field.")
    out = {}
    if "sites" in value:
        out["sites"] = [s.lower() for s in _strings(value["sites"], "Stack Exchange sites", 50)]
        if any(not re.fullmatch(r"[a-z0-9.-]+", s) for s in out["sites"]):
            raise ValueError("A Stack Exchange site is a host name such as math.stackexchange.com.")
    if "min_score" in value:
        out["min_score"] = int(_number(value["min_score"], "Minimum score", -100, 100000))
    if "dump_url" in value:
        if not str(value["dump_url"]).startswith("https://"):
            raise ValueError("The dump address must start with https://.")
        out["dump_url"] = str(value["dump_url"]).rstrip("/")
    return out


def _github(value):
    if not isinstance(value, dict) or set(value) - {"queries", "max_repos"}:
        raise ValueError("GitHub settings have an unknown field.")
    out = {}
    if "queries" in value:
        out["queries"] = _strings(value["queries"], "GitHub searches", 20)
    if "max_repos" in value:
        out["max_repos"] = int(_number(value["max_repos"], "Repositories", 1, 1000))
    return out


def _sites(value):
    """The websites the person allows: each names its host, seed pages and the licence the person read on the site's terms."""
    from studio_collect import policy
    if not isinstance(value, list):
        raise ValueError("Websites must be a list.")
    out = []
    for site in value[:50]:
        if not isinstance(site, dict) or not str(site.get("host", "")).strip():
            raise ValueError("A website needs a host name.")
        host = policy.host_of(site["host"] if "/" in site["host"] else "https://" + site["host"])
        reason = policy.excluded_reason(host)
        if reason:
            raise ValueError(f"{host} cannot be added. {reason}")
        seeds = _strings(site.get("seeds") or [f"https://{host}/"], "Seed pages", 20)
        if any(policy.host_of(s) != host for s in seeds):
            raise ValueError(f"Every seed page of {host} must be on {host}.")
        licence = str(site.get("licence", "")).strip()
        if not licence:
            raise ValueError(f"Say which licence {host} publishes its pages under (from its terms), or leave the site out.")
        out.append({"host": host, "seeds": seeds, "licence": licence, "pages": int(_number(site.get("pages", 200), "Pages", 1, 100000))})
    return out


def load(root):
    """The saved settings (the defaults when none are saved or the file is unreadable)."""
    try:
        return merge(read_json(config_path(root)))
    except (OSError, ValueError, TypeError):
        return merge({})


def save(root, overrides):
    config = merge(overrides, load(root))
    atomic_json(config_path(root), config)
    return config


def topic_list(config):
    """[(id, label, search terms)] for the chosen topics, then the person's own."""
    chosen = [(t, *TOPICS[t]) for t in config["topics"] if t in TOPICS]
    return chosen + [(f"custom:{slug(t)}", t, (t,)) for t in config["custom_topics"]]
