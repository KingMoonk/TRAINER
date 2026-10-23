"""Honest attribution for "last time blocks changed": the signed move in held-back error, what else differed between the two
versions, and whether the pair is a fair comparison (only the block count differed, or a matched Compare run agrees). No Tk."""
import copy
import math

from studio_slots import BLOCK_SLOTS
from studio_spec import block_at, structure_key, validate_spec


T = {  # what the last change in blocks did, for Build your model (studio_ia.attribution)
    "head": "Last time blocks changed ({a} → {b}, version {v_from} → {v_to}): ",
    "fell": "error fell {n} bits per character (lower is better)",
    "rose": "error rose {n} bits per character (lower is better)",
    "flat": "error did not move",
    "no_score": "one of them has no held-back score yet.",
    "clean": "{move}. Only the blocks differed, so this is the effect of the change (one run each, so a small move could be luck).",
    "unfair": "{move}. That is not a fair comparison: {why}.",
    "also": "also changed: {names}",
    "unchecked": "could not check: {names}",
    "compare": "in a matched Compare run of {seeds}, {move}.",
    "checking": "{move}; checking what else changed…",
    "inside": " That is within the spread across seeds, so it could be luck.",
}


MIN_SEEDS = 2  # a Compare run on one seed is one run each, no better than the two versions themselves
FAIR_MODES = ("tokens", "wall-clock")  # Compare modes that leave both designs as they are (the others adjust the thinking room)
SPEC_RUN_LABELS = (("steps", "steps"), ("batch_size", "batch size"), ("accumulation", "batch size"), ("seed", "random seed"))
RUN_LABELS = (("data", "data"), ("steps", "steps"), ("start", "starting weights"), ("extra", "extra training"))


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def signed_move(x, y):
    """Held-back error of the newer version minus the older one (negative = it fell), or None when either has no usable score."""
    return y - x if _number(x) and _number(y) else None


def _plain(entry, heads, room):
    """A block's norm, mixer and channel without the settings already named as differing, to see whether anything else differs."""
    e = copy.deepcopy({slot: entry[slot] for slot in BLOCK_SLOTS})
    if heads:
        e["mixer"].get("config", {}).pop("heads", None)
    if room:
        e["channel"].get("config", {}).pop("d_ff", None)
    return e


def spec_changes(old, new):
    """(what differs besides the block count, in plain words; True when nothing else differs) for two specs. Every field of
    the validated specs is compared, with the block count (n_layer and the per-block overrides past it) made the same."""
    old, new = validate_spec(old), validate_spec(new)
    a = copy.deepcopy(old)
    a["dims"]["n_layer"] = new["dims"]["n_layer"]
    a, b = validate_spec(a), new
    if {**a, "name": ""} == {**b, "name": ""}:
        return [], True
    heads = room = other = False
    da, db = a["dims"], b["dims"]
    words, width = a["tokenizer"] != b["tokenizer"], da["d_model"] != db["d_model"]
    for i in range(db["n_layer"]):
        ba, bb = block_at(a, i), block_at(b, i)
        ha, hb = ba["mixer"].get("config", {}).get("heads"), bb["mixer"].get("config", {}).get("heads")
        h = None not in (ha, hb) and ha != hb
        ra, rb = (e["channel"].get("config", {}).get("d_ff") or 4 * d for e, d in ((ba, da["d_model"]), (bb, db["d_model"])))
        r = ra / da["d_model"] != rb / db["d_model"]
        heads, room = heads or h, room or r
        other = other or _plain(ba, h, r) != _plain(bb, h, r)
    other = other or any(a[k] != b[k] for k in ("embedding", "position", "head"))
    out = [name for flag, name in ((words, "word list"), (width, "width"), (heads, "look-back heads"), (room, "thinking room"),
                                   (da["n_ctx"] != db["n_ctx"], "reading window"), (da["dropout"] != db["dropout"], "dropout"),
                                   (other, "other parts")) if flag]
    ta, tb = a["train"], b["train"]
    if ta["optimizer"]["config"].get("lr") != tb["optimizer"]["config"].get("lr"):
        out.append("learning rate")
    if ta["schedule"] != tb["schedule"]:
        out.append("learning schedule")
    rest = lambda t: (t["optimizer"]["id"], {k: v for k, v in t["optimizer"]["config"].items() if k != "lr"})  # noqa: E731
    if rest(ta) != rest(tb) or any(ta[s] != tb[s] for s in ("objective", "init", "precision")):
        out.append("training recipe")
    ra, rb = ta.get("run") or {}, tb.get("run") or {}
    for key, name in SPEC_RUN_LABELS:
        if ra.get(key) != rb.get(key) and name not in out:
            out.append(name)
    if any(ra.get(k) != rb.get(k) for k in {*ra, *rb} - {k for k, _ in SPEC_RUN_LABELS}):
        out.append("training settings")
    return out or ["other settings"], False


def run_changes(fa, fb):
    """(what the two runs did differently, what could not be checked) from Services.run_facts of each version.
    A fact missing on either side (None, or no key) is unknown, and unknown is never equal to unknown. The one exception is
    "start": None there is a run from scratch, and only an absent key is unknown. The data, the steps and the starting weights
    must be known for both runs before a pair counts as a fair comparison."""
    known = lambda f, key: key in f and (f[key] is not None or key == "start")  # noqa: E731
    changed, unchecked = [], []
    for key, name in (*RUN_LABELS, ("tokens", "tokens"), ("lr", "learning rate")):
        if not (known(fa, key) and known(fb, key)):
            unchecked.append(name)
        elif fa[key] != fb[key] and not (key == "tokens" and "steps" in changed):
            changed.append(name)
    return changed, unchecked


def matched_compare(old, new, compares):
    """{change, seeds, inside} of the best finished Compare run that trained these two designs side by side, else None.
    `compares` = Services.compare_runs(). change is new minus old in bits per character (negative = it fell); inside is whether
    that is within the spread across seeds. A run needs at least 2 seeds to count (one seed could be luck), and an entry that is
    not the expected shape is skipped. Of several, the one with the most seeds, then the newest."""
    try:
        ka, kb = structure_key(old), structure_key(new)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    best = None
    for n, c in enumerate(compares):
        try:
            res = c.get("results") or {}
            if res.get("cancelled") or c.get("mode") not in FAIR_MODES:
                continue
            base = structure_key(c["base"])
            for i, spec in enumerate(c.get("variants") or [], 1):
                row = next((r for r in res.get("variants", []) if r.get("variant") == i), None)
                seeds = row.get("seeds") if row else None
                if not row or not _number(row.get("delta")) or not isinstance(seeds, (list, tuple)) or len(seeds) < MIN_SEEDS:
                    continue
                pair = (base, structure_key(spec))
                if pair in ((ka, kb), (kb, ka)):
                    rank = (len(seeds), c.get("finished") if _number(c.get("finished")) else 0, n)
                    if best is None or rank > best[0]:
                        inside = row.get("inside_spread")
                        best = (rank, dict(change=row["delta"] if pair == (ka, kb) else -row["delta"], seeds=len(seeds),
                                           inside=inside if isinstance(inside, bool) else None))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return best[1] if best else None


def attribute(old, new, facts_old, facts_new, compares=()):
    """What can honestly be said about the change from version `old` to `new` (both services.Version).
    {change, also, unchecked, verdict, compare}: verdict is "clean" (only the block count differed, and the data, steps and
    starting weights were checked), "matched" (a Compare run on these two designs exists), "pending" (the run files have not
    been read yet: facts_old or facts_new is None) or "confounded"."""
    same_scoring = old.scorecard.get("bpc_method") == new.scorecard.get("bpc_method")  # numbers from two methods are not a move
    change = signed_move(old.scorecard.get("bpc"), new.scorecard.get("bpc")) if same_scoring else None
    if facts_old is None or facts_new is None:
        return dict(change=change, also=[], unchecked=[], verdict="pending", compare=None)
    unchecked = []
    try:
        also, clean = spec_changes(old.spec, new.spec)
    except (KeyError, TypeError, ValueError, AttributeError):
        also, clean = [], False
        unchecked.append("the saved designs")
    ran, not_known = run_changes(facts_old, facts_new)
    also = [*also, *(n for n in ran if n not in also)]
    unchecked += not_known
    compare = matched_compare(old.spec, new.spec, compares)
    verdict = "clean" if clean and not ran and not unchecked else "matched" if compare else "confounded"
    return dict(change=change, also=also, unchecked=unchecked, verdict=verdict, compare=compare)


def move_text(change):
    """"error fell 0.12 bits per character (lower is better)", "error rose …", or "error did not move"."""
    if change is None or round(abs(change), 2) == 0:
        return T["flat"]
    return T["fell" if change < 0 else "rose"].format(n=f"{abs(change):.2f}")


def blocks_change_text(a, b, v_from, v_to, info):
    """The sentence for Build your model's "what to expect": the signed move, then whether it is a fair comparison."""
    head = T["head"].format(a=a, b=b, v_from=v_from, v_to=v_to)
    if info["compare"]:
        c = info["compare"]
        seeds = f"{c['seeds']} seed{'' if c['seeds'] == 1 else 's'}"
        return head + T["compare"].format(seeds=seeds, move=move_text(c["change"])) + (T["inside"] if c["inside"] else "")
    if info["change"] is None:
        return head + T["no_score"]
    if info["verdict"] == "pending":
        return head + T["checking"].format(move=move_text(info["change"]))
    if info["verdict"] == "clean":
        return head + T["clean"].format(move=move_text(info["change"]))
    why = [T["also"].format(names=", ".join(info["also"]))] if info["also"] else []
    why += [T["unchecked"].format(names=", ".join(info["unchecked"]))] if info["unchecked"] else []
    return head + T["unfair"].format(move=move_text(info["change"]), why="; ".join(why))
