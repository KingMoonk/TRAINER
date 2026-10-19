"""Benchmark track: one frozen suite run on every trained version, so versions can be charted against tokens seen.

The suite is three measurements on fixed, sampled items: held-back bits per character (documents spread evenly over their
data sources, so one corpus cannot dominate), cloze multiple-choice accuracy (each option scored by log-likelihood per
character, at 0 and a few worked examples, with its chance level) and the pass rate on held-back checkable problems.
build_suite fixes the items from whatever the caller holds back; freeze keeps them in data/bench/suite.json, so a version
made next year is scored on the same items as one made today. The suite's id hashes every item and setting, so a result says
which suite produced it, and a changed scoring method never reuses an old result.

Per version, items that share a 13-gram with that version's training text (studio_decontam) are dropped from its score and
counted in result["dropped"]. Trend sets versions of one lineage side by side with a paired test, so a gain is only called a gain
when luck in the choice of items cannot explain it. Results are cached per version in data/bench/<version>.json, stage by
stage, so a scoring pass that stops resumes where it was. gate_verdict is the clean entry for an automated gate. No Tk and no
Services here.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import studio_decontam
import studio_eval
import studio_rules
from studio_core import atomic_json, read_json
from studio_eval import choice_item

SUITE_FORMAT = 2  # bump when what run_suite measures or how it scores changes: every cached result then goes stale
SEED = 0
DOCS = 24  # held-back documents scored, each cut to DOC_CHARS characters
DOC_CHARS = 1500
CHOICE_ITEMS = 40  # multiple-choice questions asked
CHOICE_POOL = 200  # choice problems worked examples are drawn from
PROBLEMS = 30  # held-back checkable problems answered
SHOTS = (0, 4)  # worked examples in front of each question; the last is the number the chart follows
KEPT_SUITES = 3  # suites a version's file remembers
KINDS = ("docs", "choice", "pool", "problems")
NORM = "char"  # cloze options are scored per character of their text, so the numbers compare across tokenizers
STAGES = ("docs", "choice", "problems")
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


def _pick_docs(docs, sources, n, seed):
    """n distinct documents, spread evenly over their sources (round robin over the sources in name order, each source's
    documents in hash rank), so a big corpus cannot fill the whole sample. Returns (documents, their sources)."""
    unique = {_key(d): d for d in docs}
    groups = {}
    for key in sorted(unique, key=lambda k: hashlib.sha1(f"{seed}:{k}".encode()).hexdigest()):
        groups.setdefault(sources.get(unique[key], ""), []).append(unique[key])
    picked, names, rank = [], [], 0
    while len(picked) < n and any(rank < len(g) for g in groups.values()):
        for source in sorted(groups):
            if rank < len(groups[source]) and len(picked) < n:
                picked.append(groups[source][rank])
                names.append(source)
        rank += 1
    return picked, names


def _settings(suite):
    return {"shots": SHOTS, "doc_chars": DOC_CHARS, "score_method": studio_eval.SCORE_METHOD, "norm": NORM,  # whatever else changes the numbers
            "max_tokens": {rule: studio_rules.max_tokens(rule) for rule in sorted({p.get("rule") or "" for p in suite["problems"]})}}


def suite_id(suite):
    """Twelve hex characters over every item and every setting that changes a score. Recomputed whenever a suite is loaded, so
    a frozen suite scored under a newer method has a new id and its old results read as stale."""
    items = {k: suite[k] for k in ("format", "seed", "docs", "sources", "choice", "pool", "problems")}
    return hashlib.sha1(_key({**items, **_settings(suite)}).encode()).hexdigest()[:12]


def build_suite(docs, choice, pool, problems, filter_items=keep_all, seed=SEED, sources=None):
    """The fixed items, from the held-back documents, the multiple-choice items (studio_eval.choice_item dicts, held back),
    the pool they draw worked examples from, and the held-back checkable problems ({prompt, rule, params}).

    filter_items(kind, items) -> kept runs on each list before anything is sampled, so a dropped item is replaced by the
    next one in line. sources maps a document to the data source it came from (the sample is spread over them).
    Returns {id, format, seed, docs, sources (one per doc), choice, pool, problems, dropped: {kind: n filtered out}}."""
    held = {_key(i) for i in choice}  # a question being asked is never also a worked example
    given = {"docs": [d for d in docs if d.strip()], "choice": list(choice), "pool": [i for i in pool if _key(i) not in held],
             "problems": [p for p in problems if p.get("rule") != "choice"]}  # lettered problems are the few-shot metric's
    kept = {kind: list(filter_items(kind, items)) for kind, items in given.items()}
    picked, names = _pick_docs(kept["docs"], sources or {}, DOCS, seed)
    suite = {"format": SUITE_FORMAT, "seed": seed, "docs": [d[:DOC_CHARS] for d in picked], "sources": names,
             "choice": _pick(kept["choice"], CHOICE_ITEMS, seed), "pool": _pick(kept["pool"], CHOICE_POOL, seed),
             "problems": _pick(kept["problems"], PROBLEMS, seed)}
    suite["id"] = suite_id(suite)
    suite["dropped"] = {kind: len(given[kind]) - len(kept[kind]) for kind in KINDS}
    return suite


def suite_from_rows(docs, held_problems, training_problems=(), filter_items=keep_all, seed=SEED, sources=None):
    """build_suite from the rows the workspace holds, as post_rows gives them: held-back documents (strings), held-back
    problems (checkable ones and lettered ones) and training problems (only their lettered ones supply worked examples)."""
    held = [i for i in map(choice_item, held_problems) if i]
    pool = [i for i in map(choice_item, training_problems) if i]  # worked examples come from training problems, never the held-back questions
    return build_suite(docs, held, pool, held_problems, filter_items, seed, sources)


# ---------------------------------------------------------------------------------------------------- running

def _facts(version):
    """(number, best checkpoint) of a Version, a versions.json row or a (number, checkpoint) pair."""
    if isinstance(version, tuple):
        return version
    get = version.get if isinstance(version, dict) else lambda key: getattr(version, key, None)
    return get("number"), get("best_checkpoint")


def suite_path(root):
    return Path(root) / "data" / "bench" / "suite.json"


def frozen(root, build):
    """The workspace's suite: read from data/bench/suite.json, built once by build() and kept. Later data never changes the items,
    so two versions made months apart are scored on the same ones. A kind with no items when the suite was frozen (no
    problems yet) is filled from build() the first time it has some; a kind that has items is never replaced. A file written by
    another SUITE_FORMAT is built again. Delete the file to start a new suite (old results stay under their own suite id)."""
    path = suite_path(root)
    try:
        kept = read_json(path)
    except (OSError, ValueError):
        kept = None
    if not isinstance(kept, dict) or kept.get("format") != SUITE_FORMAT:
        kept = None
    wanted = ("docs", "choice", "problems")
    if kept is not None and all(kept.get(k) for k in wanted):
        kept["id"] = suite_id(kept)
        return kept
    fresh = build()
    if kept is None:
        merged = fresh
    else:
        merged = {**kept, "dropped": dict(kept.get("dropped") or {})}
        for kind in KINDS:
            if not kept.get(kind) and fresh[kind]:
                merged[kind] = fresh[kind]
                merged["dropped"][kind] = fresh["dropped"].get(kind, 0)
                if kind == "docs":
                    merged["sources"] = fresh["sources"]
    asked = {_key(i) for i in merged["choice"]}
    merged["pool"] = [i for i in merged["pool"] if _key(i) not in asked]  # a question being asked is never also a worked example
    merged["id"] = suite_id(merged)
    if any(merged[k] for k in wanted):
        atomic_json(path, {k: v for k, v in merged.items() if k != "id"})
    return merged


def suite_texts(suite):
    """Every text the suite asks about or shows, for keeping them out of a training pool (an automated loop's prompts, for one):
    the passages, each question with its options, each problem's prompt, and the worked examples."""
    out = list(suite["docs"])
    for item in [*suite["choice"], *suite["pool"]]:
        out.append("\n".join([item["question"], *item["choices"]]))
    out.extend(p["prompt"] for p in suite["problems"])
    return out


def suite_index(suite):
    """A studio_decontam.Index of suite_texts: index.overlaps(prompt) says a prompt is (part of) an item the judge must not see."""
    return studio_decontam.Index().add_all(suite_texts(suite))


def contaminated(suite, index):
    """{kind: [positions in the suite's list]} of the items that share a 13-gram with the training text in `index`."""
    text = {"docs": lambda d: d, "choice": lambda i: "\n".join([i["question"], *i["choices"]]), "problems": lambda p: p["prompt"]}
    return {kind: [n for n, hit in enumerate(studio_decontam.overlap_mask(suite[kind], index, text[kind])) if hit] for kind in STAGES}


def run_suite(version, suite, tokens=None, skip=None, resume=None, save=None):
    """The suite's measurements for one version's best checkpoint, as plain numbers:
    {suite, version, tokens, bpc, bpc_band [lo, hi], sources {source: {bpc, docs}}, doc_scores [[nats, chars] or None per suite document],
    few_shot {items, chance, accuracy, band, norm, rows [{shots, used, accuracy, hits}]}, problems {passed, total, rate, each},
    dropped {docs, choice, problems}, decontam ("checked", or "unknown" when skip is None), stages, seconds}. A measurement with no items is None. `tokens` (tokens seen, see tokens_seen) is
    stored as given. Everything is seeded or greedy, so the same suite and checkpoint give the same numbers; only `seconds` varies.

    skip = {kind: positions} are the items to leave out for this version (contaminated); they stay in the lists as None so
    that two versions line up item by item. resume is an earlier, unfinished result of the same suite and checkpoint: its finished
    stages are not run again. save(result) is called after each stage with `stages` listing what is done (the cache keeps it)."""
    from studio_model import LocalModel
    number, checkpoint = _facts(version)
    started = time.monotonic()
    unchecked = skip is None  # the version's training text was not on hand: nothing could be dropped, and the result says so
    skip = {kind: set((skip or {}).get(kind, ())) for kind in STAGES}
    result = dict(resume) if resume and resume.get("suite") == suite["id"] else {}
    result.update({"suite": suite["id"], "version": number, "tokens": tokens})
    for key in ("bpc", "few_shot", "problems"):
        result.setdefault(key, None)
    result.setdefault("stages", [])
    result["dropped"] = {kind: len(skip[kind]) for kind in STAGES}
    result["decontam"] = "unknown" if unchecked else "checked"
    local = None

    def loaded():
        nonlocal local
        local = local or LocalModel(checkpoint, device="cpu")
        return local

    def finish(stage):
        result["stages"] = [*result["stages"], stage]
        result["seconds"] = result.get("seconds", 0.) + time.monotonic() - started
        if save:
            save(dict(result))
    keep = {kind: [n for n in range(len(suite[kind])) if n not in skip[kind]] for kind in STAGES}
    if "docs" not in result["stages"]:
        if keep["docs"]:
            scores = studio_eval.score_docs(loaded(), [suite["docs"][n] for n in keep["docs"]])
            per = [None] * len(suite["docs"])
            for n, row in zip(keep["docs"], scores):
                per[n] = [row[0], row[1]]
            result["doc_scores"] = per
            result["bpc"] = studio_eval.corpus_bpc(scores)
            result["bpc_band"] = list(studio_eval.band_from_scores(scores))
            result["sources"] = _by_source(suite, per)
        finish("docs")
    if "choice" not in result["stages"]:
        if keep["choice"]:
            items = [suite["choice"][n] for n in keep["choice"]]
            table = studio_eval.few_shot(loaded(), items, shots=SHOTS, seed=suite["seed"], pool=suite["pool"], norm=NORM)
            rows = []
            for r in table["rows"]:
                each = [None] * len(suite["choice"])
                for n, hit in zip(keep["choice"], r["hits"]):
                    each[n] = hit
                rows.append({"shots": r["shots"], "used": r["used"], "accuracy": r["accuracy"], "hits": each})
            band = studio_eval.accuracy_band([h for h in rows[-1]["hits"] if h is not None])
            result["few_shot"] = {"items": table["items"], "chance": table["chance"], "accuracy": rows[-1]["accuracy"], "band": list(band),
                                  "norm": NORM, "rows": rows}
        finish("choice")
    if "problems" not in result["stages"]:
        if keep["problems"]:
            each = [None] * len(suite["problems"])
            for n in keep["problems"]:
                q = suite["problems"][n]
                each[n] = int(bool(studio_rules.check(q["rule"], q["prompt"], loaded().generate(
                    [{"role": "user", "content": q["prompt"]}], max_tokens=studio_rules.max_tokens(q["rule"]), temperature=0), q.get("params"))))
            passed = sum(h for h in each if h)
            result["problems"] = {"passed": passed, "total": len(keep["problems"]), "rate": passed / len(keep["problems"]), "each": each}
        finish("problems")
    return result


def _by_source(suite, per):
    """{source: {bpc, docs}} of the scored documents of each source (only when there is more than one source to tell apart)."""
    groups = {}
    for source, row in zip(suite.get("sources") or [""] * len(per), per):
        if row is not None:
            groups.setdefault(source, []).append(row)
    if len(groups) < 2:
        return {}
    return {source: {"bpc": studio_eval.corpus_bpc(rows), "docs": len(rows)} for source, rows in sorted(groups.items())}


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
    return hit["result"] if hit and "result" in hit and stamp is not None and hit.get("checkpoint") == stamp else None


def stale(root, version, suite):
    """Whether this version has results, but only from another suite (an older scoring method, or items that were changed):
    it was scored once and needs scoring again. False for a version never scored, and for one that is current."""
    number, checkpoint = _facts(version)
    kept = _read(root, number)
    stamp = _stamp(checkpoint) if checkpoint else None
    return bool(kept) and cached(root, version, suite) is None and any(
        "result" in hit and hit.get("checkpoint") == stamp for hit in kept.values())


_locks, _locks_guard = {}, threading.Lock()


def _lock(root, number):
    with _locks_guard:
        return _locks.setdefault((str(Path(root).resolve()), number), threading.Lock())


def benchmark(root, version, suite, tokens=None, run=run_suite, skip=None):
    """cached(), else run the suite and store it under data/bench/<version>.json, keyed by suite id (the last KEPT_SUITES kept).
    One run per version at a time: a second caller waits and then reads the first one's result. The checkpoint is stamped
    before the run, so a checkpoint rewritten while the suite was running is checked again next time.

    The run is resumable: run(version, suite, tokens, skip=, resume=, save=) stores each finished stage as `partial` under the
    suite id, and a later call for the same checkpoint hands it back as `resume`. `skip` (see contaminated) is
    computed by the caller from the version's training text."""
    number, checkpoint = _facts(version)
    with _lock(root, number):
        hit = cached(root, version, suite)
        if hit is not None:
            return hit
        stamp = _stamp(checkpoint)
        before = _read(root, number)
        partial = (before.get(suite["id"]) or {})
        resume = partial.get("partial") if partial.get("checkpoint") == stamp else None

        def write(entry):
            suites = {k: v for k, v in _read(root, number).items() if k != suite["id"]}
            suites[suite["id"]] = entry
            atomic_json(cache_path(root, number), {"suites": dict(list(suites.items())[-KEPT_SUITES:])})
        result = run(version, suite, tokens, skip=skip, resume=resume, save=lambda part: write({"checkpoint": stamp, "partial": part}))
        write({"checkpoint": stamp, "result": result})
        return result


# ---------------------------------------------------------------------------------------------------- trend

def _common(a, b):
    """The values of two results' per-item lists where both versions kept the item: ([a's], [b's])."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def compare_results(parent, candidate):
    """What the candidate gained over the parent on the items both versions kept, with the paired tests of studio_eval:
    {bpc: paired_gain dict or None, few_shot: paired_hits dict or None, problems: paired_hits dict or None}. A gain counts
    only where "better" is True. None where either side has no per-item scores (a result made before they were stored)."""
    out = {"bpc": None, "few_shot": None, "problems": None}
    pa, ca = (parent or {}).get("doc_scores"), (candidate or {}).get("doc_scores")
    if pa and ca and len(pa) == len(ca):
        p, c = _common(pa, ca)
        if p:
            out["bpc"] = studio_eval.paired_gain(p, c)
    for key, hits in (("few_shot", lambda r: ((r.get("few_shot") or {}).get("rows") or [{}])[-1].get("hits")),
                      ("problems", lambda r: (r.get("problems") or {}).get("each"))):
        pa, ca = hits(parent or {}), hits(candidate or {})
        if pa and ca and len(pa) == len(ca):
            p, c = _common(pa, ca)
            if p:
                out[key] = studio_eval.paired_hits(p, c)
    return out


def trend(rows):
    """The suite across versions, a lineage at a time: [{lineage, versions [numbers], points [{number, tokens, bpc, bpc_band,
    few_shot, few_shot_band, problems, dropped, gain}]}].

    rows are [{number, lineage, tokens, result}]; versions with no result yet are left out. Points are ordered by tokens seen
    (by version number when some are unknown). gain on each point after the first is compare_results against the point before,
    on the items both kept: a drop in bits per character, or a rise in accuracy, is a real gain only where its "better" is
    True. The first point of a lineage has no gain."""
    lines = {}
    for row in sorted(rows, key=lambda r: r["number"]):
        if row.get("result") is not None:
            lines.setdefault(row.get("lineage") or "", []).append(row)
    out = []
    for lineage, members in lines.items():
        placed = all(m.get("tokens") is not None for m in members)
        members = sorted(members, key=(lambda m: (m["tokens"], m["number"])) if placed else (lambda m: m["number"]))
        points = []
        for before, m in zip([None, *members], members):
            r = m["result"]
            points.append({"number": m["number"], "tokens": m.get("tokens"), "bpc": r.get("bpc"), "bpc_band": r.get("bpc_band"),
                           "few_shot": (r.get("few_shot") or {}).get("accuracy"), "few_shot_band": (r.get("few_shot") or {}).get("band"),
                           "problems": (r.get("problems") or {}).get("rate"), "dropped": r.get("dropped") or {},
                           "gain": compare_results(before["result"], r) if before else None})
        out.append({"lineage": lineage, "versions": [m["number"] for m in members], "points": points})
    return out


# ---------------------------------------------------------------------------------------------------- gate

def gate_verdict(baseline, candidate):
    """For an automated loop: did the candidate checkpoint get worse than the baseline on the frozen suite?

    baseline and candidate are run_suite results of the same suite. Returns {ok, checks: [{name, detail, ok}], gain} in the
    shape of studio_gate.passes. A check fails only on a real regression: the baseline is better by a paired test (studio_eval.
    paired_gain / paired_hits with the roles swapped), so noise in the choice of items does not block a good round and a
    real drop does. Not comparable (different suite ids): ok False with one check saying so."""
    if not baseline or not candidate or baseline.get("suite") != candidate.get("suite"):
        return {"ok": False, "gain": None, "checks": [{"name": "Same frozen suite", "ok": False,
                                                       "detail": "The two results were not made with the same suite, so they cannot be compared."}]}
    gain = compare_results(baseline, candidate)
    swapped = compare_results(candidate, baseline)
    checks = []
    for key, name, unit in (("bpc", "Frozen suite: error did not rise", "bits per character"),
                            ("few_shot", "Frozen suite: multiple-choice accuracy did not fall", "accuracy"),
                            ("problems", "Frozen suite: checked-problem pass rate did not fall", "pass rate")):
        g, w = gain[key], swapped[key]
        if g is None or w is None:
            checks.append({"name": name, "ok": True, "detail": "not measured on both sides"})
            continue
        diff = g["diff"]
        checks.append({"name": name, "ok": not w["better"],
                       "detail": f"{unit} changed by {diff:+.3f} over {g.get('docs', g.get('items'))} shared items" + ("" if not w["better"] else ": the baseline is better by a paired test")})
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "gain": gain}


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
