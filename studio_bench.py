"""Benchmark track: one fixed suite run on every trained version, so versions can be charted against tokens seen.

The suite is three measurements on fixed, sampled items: held-back bits per character, few-shot multiple-choice accuracy
(with its chance level) and the pass rate on held-back checkable problems. build_suite fixes the items from whatever the
caller holds back; its id hashes every item and setting, so a result says which suite produced it and a changed suite
never reuses an old result. Results are cached per version in data/bench/<version>.json. No Tk and no Services here.
"""
from __future__ import annotations

import hashlib
import json
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
    given = {"docs": [d for d in docs if d.strip()], "choice": list(choice), "pool": list(pool),
             "problems": [p for p in problems if p.get("rule") != "choice"]}  # lettered problems are the few-shot metric's
    kept = {kind: list(filter_items(kind, items)) for kind, items in given.items()}
    suite = {"format": SUITE_FORMAT, "seed": seed,
             "docs": [d[:DOC_CHARS] for d in _pick(kept["docs"], DOCS, seed)],
             "choice": _pick(kept["choice"], CHOICE_ITEMS, seed), "pool": _pick(kept["pool"], CHOICE_POOL, seed),
             "problems": _pick(kept["problems"], PROBLEMS, seed)}
    suite["id"] = hashlib.sha1(_key({**suite, "shots": SHOTS, "doc_chars": DOC_CHARS}).encode()).hexdigest()[:12]
    suite["dropped"] = {kind: len(given[kind]) - len(kept[kind]) for kind in KINDS}
    return suite


def suite_from_rows(docs, held_problems, training_problems=(), filter_items=keep_all, seed=SEED):
    """build_suite from the rows the workspace holds, as post_rows gives them: held-back documents (strings), held-back
    problems (checkable ones and lettered ones) and training problems (only their lettered ones supply worked examples)."""
    held = [i for i in map(choice_item, held_problems) if i]
    pool = held + [i for i in map(choice_item, training_problems) if i]
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
        rows = [{"shots": r["shots"], "accuracy": r["accuracy"]} for r in table["rows"]]
        result["few_shot"] = {"items": table["items"], "chance": table["chance"], "accuracy": rows[-1]["accuracy"], "rows": rows}
    if suite["problems"]:
        passed = sum(studio_rules.check(q["rule"], q["prompt"], local.generate([{"role": "user", "content": q["prompt"]}],
                                                                              max_tokens=studio_rules.max_tokens(q["rule"]), temperature=0),
                                        q.get("params")) for q in suite["problems"])
        result["problems"] = {"passed": passed, "total": len(suite["problems"]), "rate": passed / len(suite["problems"])}
    result["seconds"] = time.monotonic() - started
    return result


def tokens_seen(runs, checkpoint=None):
    """Tokens a version's newest run had trained on when its checkpoint was saved: the best checkpoint is the run's lowest
    held-back loss, anything else (a later round's weights, last.npz) counts the whole run. None when no run logged tokens."""
    from studio_model import read_metrics
    run = Path(runs[-1]) if runs else None
    if run is None or not (run / "metrics.jsonl").exists():
        return None
    rows = [r for r in read_metrics(run / "metrics.jsonl") if r.get("trained_tokens") is not None]
    if not rows:
        return None
    held = [r for r in rows if r.get("val_loss") is not None]
    if checkpoint and Path(checkpoint) == run / "best.npz" and held:
        return min(held, key=lambda r: r["val_loss"])["trained_tokens"]
    return rows[-1]["trained_tokens"]


# ---------------------------------------------------------------------------------------------------- cache

def cache_path(root, number):
    return Path(root) / "data" / "bench" / f"{number}.json"


def _stamp(checkpoint):
    """Size and time of the checkpoint file: a retrained or post-trained version has a new one."""
    try:
        s = Path(checkpoint).stat()
        return [s.st_mtime_ns, s.st_size]
    except OSError:
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
    return hit["result"] if hit and checkpoint and hit.get("checkpoint") == _stamp(checkpoint) else None


def benchmark(root, version, suite, tokens=None, run=run_suite):
    """cached(), else run the suite and store it under data/bench/<version>.json, keyed by suite id (the last KEPT_SUITES kept)."""
    hit = cached(root, version, suite)
    if hit is not None:
        return hit
    number, checkpoint = _facts(version)
    result = run(version, suite, tokens)
    suites = {k: v for k, v in _read(root, number).items() if k != suite["id"]}
    suites[suite["id"]] = {"checkpoint": _stamp(checkpoint), "result": result}
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
