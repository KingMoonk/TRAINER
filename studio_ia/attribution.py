"""Honest attribution for "last time blocks changed": the signed move in held-back error, what else differed between the two
versions, and whether the pair is a fair comparison (only the block count differed, or a matched Compare run agrees). No Tk."""
import copy

from studio_ia.copy import ATTRIBUTION as T
from studio_spec import structure_key, validate_spec
from studio_versions import _facts

FAIR_MODES = ("tokens", "wall-clock")  # Compare modes that leave both designs as they are (the others adjust the thinking room)
SPEC_RUN_LABELS = (("steps", "steps"), ("batch_size", "batch size"), ("accumulation", "batch size"), ("seed", "random seed"))
RUN_LABELS = (("data", "data"), ("steps", "steps"), ("start", "starting weights"), ("extra", "extra training"))


def signed_move(x, y):
    """Held-back error of the newer version minus the older one (negative = it fell), or None when either has no score."""
    ok = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
    return y - x if ok(x) and ok(y) else None


def spec_changes(old, new):
    """(what differs besides the block count, in plain words; True when nothing else differs) for two specs."""
    old, new = validate_spec(old), validate_spec(new)
    a = copy.deepcopy(old)
    a["dims"]["n_layer"] = new["dims"]["n_layer"]
    a, b = validate_spec(a), new
    if structure_key(a) == structure_key(b) and a["train"] == b["train"]:
        return [], True
    fa, fb = _facts(a), _facts(b)
    out = [name for key, name in (("words", "word list"), ("width", "width"), ("heads", "look-back heads"), ("room", "thinking room"),
                                  ("context", "reading window")) if fa[key] != fb[key]]
    if not out and structure_key(a) != structure_key(b):
        out.append("other parts")
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
    The data, the steps and the starting weights must be known for both runs before a pair counts as a fair comparison."""
    changed, unchecked = [], []
    for key, name in RUN_LABELS:
        if key not in fa or key not in fb:
            unchecked.append(name)
        elif fa[key] != fb[key]:
            changed.append(name)
    if fa.get("tokens") is not None and fb.get("tokens") is not None and fa["tokens"] != fb["tokens"] and "steps" not in changed:
        changed.append("tokens")
    if fa.get("lr") is not None and fb.get("lr") is not None and fa["lr"] != fb["lr"]:
        changed.append("learning rate")
    return changed, unchecked


def matched_compare(old, new, compares):
    """{change, seeds, inside} of a finished Compare run that trained these two designs side by side, else None.
    `compares` = Services.compare_runs(). change is new minus old in bits per character (negative = it fell); inside is whether
    that is within the spread across seeds (None with one seed)."""
    ka, kb = structure_key(old), structure_key(new)
    for c in compares:
        res = c.get("results") or {}
        if res.get("cancelled") or c.get("mode") not in FAIR_MODES:
            continue
        base = structure_key(c["base"])
        for i, spec in enumerate(c.get("variants") or [], 1):
            row = next((r for r in res.get("variants", []) if r.get("variant") == i), None)
            if not row or row.get("delta") is None:
                continue
            pair = (base, structure_key(spec))
            if pair in ((ka, kb), (kb, ka)):
                return dict(change=row["delta"] if pair == (ka, kb) else -row["delta"], seeds=len(row.get("seeds") or ()),
                            inside=row.get("inside_spread"))
    return None


def attribute(old, new, facts_old, facts_new, compares=()):
    """What can honestly be said about the change from version `old` to `new` (both services.Version).
    {change, also, unchecked, verdict, compare}: verdict is "clean" (only the block count differed, and the data, steps and
    starting weights were checked), "matched" (a Compare run on these two designs exists) or "confounded"."""
    change = signed_move(old.scorecard.get("bpc"), new.scorecard.get("bpc"))
    also, clean = spec_changes(old.spec, new.spec)
    ran, unchecked = run_changes(facts_old, facts_new)
    also = [*also, *(n for n in ran if n not in also)]
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
        return head + T["compare"].format(seeds=c["seeds"], move=move_text(c["change"])) + (T["inside"] if c["inside"] else "")
    if info["change"] is None:
        return head + T["no_score"]
    if info["verdict"] == "clean":
        return head + T["clean"].format(move=move_text(info["change"]))
    why = [T["also"].format(names=", ".join(info["also"]))] if info["also"] else []
    why += [T["unchecked"].format(names=", ".join(info["unchecked"]))] if info["unchecked"] else []
    return head + T["unfair"].format(move=move_text(info["change"]), why="; ".join(why))
