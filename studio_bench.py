"""Benchmark track: one fixed suite run on every trained version, so versions can be charted against tokens seen.

The suite is three measurements on fixed, sampled items: held-back bits per character, few-shot multiple-choice accuracy
(with its chance level) and the pass rate on held-back checkable problems. build_suite fixes the items from whatever the
caller holds back; its id hashes every item and setting, so a result says which suite produced it and a changed suite
never reuses an old result. Results are cached per version in data/bench/<version>.json. No Tk and no Services here.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import studio_eval
import studio_rules
from studio_core import atomic_json, read_json
from studio_eval import choice_item

SUITE_FORMAT = 1  # bump when what run_suite measures or how it scores changes: every cached result then goes stale
SEED = 0
DOCS = 24  # held-back documents scored, each cut to DOC_CHARS characters
DOC_CHARS = 1500
CHOICE_ITEMS = 40  # multiple-choice questions asked
CHOICE_POOL = 200  # choice problems worked examples are drawn from
PROBLEMS = 30  # held-back checkable problems answered
SHOTS = (0, 4)  # worked examples in front of each question; the last is the number the chart follows
KEPT_SUITES = 3  # suites a version's file remembers
KINDS = ("docs", "choice", "pool", "problems")
METRICS = ("bpc", "few_shot", "problems")


def keep_all(kind, items):
    """The filter_items hook of build_suite: which of `items` may enter the suite. kind is one of KINDS (docs are strings,
    the others dicts); return the items to keep. The default keeps everything; a decontamination check plugs in here."""
    return items


def _key(item):
    return json.dumps(item, sort_keys=True, ensure_ascii=False)


def _pick(items, n, seed):
    """n distinct items, chosen by hash rank: the same items give the same pick in any order, and adding an item changes
    the pick only when the new one ranks in the first n."""
    unique = {_key(item): item for item in items}
    ranked = sorted(unique, key=lambda k: hashlib.sha1(f"{seed}:{k}".encode()).hexdigest())
    return [unique[k] for k in ranked[:n]]


def build_suite(docs, choice, pool, problems, filter_items=keep_all, seed=SEED):
    """The fixed items, from the held-back documents, the multiple-choice items (studio_eval.choice_item dicts, held back),
    the pool they draw worked examples from, and the held-back checkable problems ({prompt, rule, params}).

    filter_items(kind, items) -> kept runs on each list before anything is sampled, so a dropped item is replaced by the
    next one in line. Returns {id, format, seed, docs, choice, pool, problems, dropped: {kind: n filtered out}}."""
    held = {_key(i) for i in choice}  # a question being asked is never also a worked example
    given = {"docs": [d for d in docs if d.strip()], "choice": list(choice), "pool": [i for i in pool if _key(i) not in held],
             "problems": [p for p in problems if p.get("rule") != "choice"]}  # lettered problems are the few-shot metric's
    kept = {kind: list(filter_items(kind, items)) for kind, items in given.items()}
    suite = {"format": SUITE_FORMAT, "seed": seed,
             "docs": [d[:DOC_CHARS] for d in _pick(kept["docs"], DOCS, seed)],
             "choice": _pick(kept["choice"], CHOICE_ITEMS, seed), "pool": _pick(kept["pool"], CHOICE_POOL, seed),
             "problems": _pick(kept["problems"], PROBLEMS, seed)}
    settings = {"shots": SHOTS, "doc_chars": DOC_CHARS, "score_method": studio_eval.SCORE_METHOD,  # whatever else changes the numbers
                "max_tokens": {rule: studio_rules.max_tokens(rule) for rule in sorted({p.get("rule") or "" for p in suite["problems"]})}}
    suite["id"] = hashlib.sha1(_key({**suite, **settings}).encode()).hexdigest()[:12]
    suite["dropped"] = {kind: len(given[kind]) - len(kept[kind]) for kind in KINDS}
    return suite


def suite_from_rows(docs, held_problems, training_problems=(), filter_items=keep_all, seed=SEED):
    """build_suite from the rows the workspace holds, as post_rows gives them: held-back documents (strings), held-back
    problems (checkable ones and lettered ones) and training problems (only their lettered ones supply worked examples)."""
    held = [i for i in map(choice_item, held_problems) if i]
    pool = [i for i in map(choice_item, training_problems) if i]  # worked examples come from training problems, never the held-back questions
    return build_suite(docs, held, pool, held_problems, filter_items, seed)


# ---------------------------------------------------------------------------------------------------- running

def _facts(version):
    """(number, best checkpoint) of a Version, a versions.json row or a (number, checkpoint) pair."""
    if isinstance(version, tuple):
        return version
    get = version.get if isinstance(version, dict) else lambda key: getattr(version, key, None)
    return get("number"), get("best_checkpoint")


def run_suite(version, suite, tokens=None):
    """The suite's measurements for one version's best checkpoint, as plain numbers:
    {suite, version, tokens, bpc, few_shot {items, chance, accuracy, rows}, problems {passed, total, rate}, seconds}.
    A measurement with no items is None. `tokens` (tokens seen, see tokens_seen) is stored as given.
    Everything is seeded or greedy, so the same suite and checkpoint give the same numbers; only `seconds` varies."""
    from studio_model import LocalModel
    number, checkpoint = _facts(version)
    started = time.monotonic()
    local = LocalModel(checkpoint, device="cpu")
    result = {"suite": suite["id"], "version": number, "tokens": tokens, "bpc": None, "few_shot": None, "problems": None}
    if suite["docs"]:
        scores = studio_eval.score_docs(local, suite["docs"])
        chars = sum(c for _, c in scores)
        result["bpc"] = sum(n for n, _ in scores) / chars / studio_eval.LN2 if chars else None
    if suite["choice"]:
        table = studio_eval.few_shot(local, suite["choice"], shots=SHOTS, seed=suite["seed"], pool=suite["pool"])
        rows = [{"shots": r["shots"], "used": r["used"], "accuracy": r["accuracy"]} for r in table["rows"]]
        result["few_shot"] = {"items": table["items"], "chance": table["chance"], "accuracy": rows[-1]["accuracy"], "rows": rows}
    if suite["problems"]:
        passed = sum(studio_rules.check(q["rule"], q["prompt"], local.generate([{"role": "user", "content": q["prompt"]}],
                                                                              max_tokens=studio_rules.max_tokens(q["rule"]), temperature=0),
                                        q.get("params")) for q in suite["problems"])
        result["problems"] = {"passed": passed, "total": len(suite["problems"]), "rate": passed / len(suite["problems"])}
    result["seconds"] = time.monotonic() - started
    return result


def _same(a, b):
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def _run_rows(run):
    """[(tokens trained in the whole lineage so far, held-back loss or None)] for a run's logged rows. A resumed run's log
    starts with its parent's rows and then counts from 0 again, so a count that falls starts a new stretch on top of the last."""
    from studio_model import read_metrics
    path = Path(run) / "metrics.jsonl"
    if not path.exists():
        return []
    out, offset, previous = [], 0, 0
    for row in read_metrics(path):
        tokens = row.get("trained_tokens")
        if tokens is None:
            continue
        if tokens < previous:
            offset += previous
        previous = tokens
        out.append((offset + tokens, row.get("val_loss")))
    return out


def tokens_seen(runs, checkpoint=None, owner=None, _depth=0):
    """Tokens the weights of `checkpoint` have trained on, over every run they descend from, or None when that cannot be told.

    The run holding the checkpoint is the one whose folder it sits in (a checkpoint elsewhere, such as a post-training
    round's, descends from the newest of `runs`). Its best.npz counts to the lowest held-back loss, anything else (last.npz,
    a later round) the whole run; a resumed run's log already carries its parent's rows (see _run_rows). A run started from
    another checkpoint (run.json initialized_from, a catch-up) adds what that checkpoint had seen. `owner(checkpoint)` may
    name the runs of the version that made such a checkpoint when it does not sit in a run folder with a log."""
    runs = [Path(r) for r in runs]
    home = Path(checkpoint).parent if checkpoint else None
    run = next((r for r in runs if home and _same(r, home)), home if home and (home / "metrics.jsonl").exists() else runs[-1] if runs else None)
    rows = _run_rows(run) if run else []
    if not rows:
        return None
    held = [r for r in rows if r[1] is not None]
    own = min(held, key=lambda r: r[1])[0] if checkpoint and Path(checkpoint) == run / "best.npz" and held else rows[-1][0]
    try:
        start = read_json(run / "run.json").get("initialized_from")
    except (OSError, ValueError, AttributeError):
        start = None
    if not start:
        return own
    if _depth >= 8:
        return None
    parents = [Path(start).parent]
    if owner is not None and not (parents[0] / "metrics.jsonl").exists():
        parents = list(owner(start) or [])
    before = tokens_seen(parents, start, owner, _depth + 1) if parents else None
    return None if before is None else before + own


# ---------------------------------------------------------------------------------------------------- cache

def cache_path(root, number):
    return Path(root) / "data" / "bench" / f"{number}.json"


def _stamp(checkpoint):
    """Path, size and time of the checkpoint file: a retrained or post-trained version has a new one. None when it is not there."""
    try:
        s = Path(checkpoint).stat()
        return [str(checkpoint), s.st_mtime_ns, s.st_size]
    except (OSError, TypeError):
        return None


def _read(root, number):
    try:
        data = read_json(cache_path(root, number))
    except (OSError, ValueError):
        return {}
    return data.get("suites", {}) if isinstance(data, dict) else {}


def cached(root, version, suite):
    """The stored result of this suite for this version's current checkpoint, or None (never computes)."""
    number, checkpoint = _facts(version)
    hit = _read(root, number).get(suite["id"])
    stamp = _stamp(checkpoint) if checkpoint else None
    return hit["result"] if hit and stamp is not None and hit.get("checkpoint") == stamp else None


_locks, _locks_guard = {}, threading.Lock()


def _lock(root, number):
    with _locks_guard:
        return _locks.setdefault((str(Path(root).resolve()), number), threading.Lock())


def benchmark(root, version, suite, tokens=None, run=run_suite):
    """cached(), else run the suite and store it under data/bench/<version>.json, keyed by suite id (the last KEPT_SUITES kept).
    One run per version at a time: a second caller waits and then reads the first one's result. The checkpoint is stamped
    before the run, so a checkpoint rewritten while the suite was running is checked again next time."""
    number, checkpoint = _facts(version)
    with _lock(root, number):
        hit = cached(root, version, suite)
        if hit is not None:
            return hit
        stamp = _stamp(checkpoint)
        result = run(version, suite, tokens)
        suites = {k: v for k, v in _read(root, number).items() if k != suite["id"]}
        suites[suite["id"]] = {"checkpoint": stamp, "result": result}
        atomic_json(cache_path(root, number), {"suites": dict(list(suites.items())[-KEPT_SUITES:])})
        return result


# ---------------------------------------------------------------------------------------------------- chart data

def metric_value(result, metric):
    """The number a chart plots for a result, or None: bits per character, 4-shot accuracy, or the problems pass rate."""
    got = (result or {}).get(metric)
    return got if metric == "bpc" else None if got is None else got["accuracy" if metric == "few_shot" else "rate"]


def chart_data(rows, metric):
    """Points for one metric, one per version, grouped into lines by design.

    rows are [{number, lineage, tokens, result}] (result None until the suite has run); versions with the same lineage are
    the same model design (same structure, trained differently) and are linked, ordered by tokens seen. Returns
    {metric, lines: [{lineage, points: [(tokens, value, number)]}], chance, unplaced: [numbers with a value but no token
    count], waiting: [numbers with no result]}. chance is the guessing level (few_shot only)."""
    lines, unplaced, waiting, chance = {}, [], [], None
    for row in sorted(rows, key=lambda r: r["number"]):
        value = metric_value(row.get("result"), metric)
        if row.get("result") is None:
            waiting.append(row["number"])
        if value is None:
            continue
        if row.get("tokens") is None:
            unplaced.append(row["number"])
            continue
        lines.setdefault(row["lineage"], []).append((row["tokens"], value, row["number"]))
        if metric == "few_shot" and chance is None:
            chance = row["result"]["few_shot"]["chance"]
    return {"metric": metric, "chance": chance, "unplaced": unplaced, "waiting": waiting,
            "lines": [{"lineage": key, "points": sorted(points)} for key, points in lines.items()]}
