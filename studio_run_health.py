"""Is the cluster helping? What a distributed (DiLoCo) run measures of each round, and what those numbers say.

`record` turns a closed round into one small dict (kept in the round's history row, so it is saved with the run). `summarize` reads
the rounds so far and answers: the aggregate speed, the speed of the fastest member alone, the speed-up between them, how much of
a round goes to waiting and to communication, and a verdict with its reason.

The single-member baseline is measured, not assumed: a member's solo speed is the tokens it trained divided by the seconds it
spent computing them, over the same rounds the aggregate covers. That is the pace one machine would keep with no exchange. The
speed-up is therefore in tokens per second (data trained), not in loss per second, and the page says so. With fewer than two
members in a round there is nothing to compare, and the verdict is "not enough data" rather than a guess.
"""

import math

WINDOW = 5  # the latest rounds (with two or more contributors) the speed-up and the shares are measured over
MIN_ROUNDS = 2  # rounds with two or more contributors before a speed-up is stated
DIVERGE = 3  # rounds in a row the loss may rise before the run is called diverging
HELPING = 1.1  # a speed-up below this is not worth the exchange
WAITING = 0.5  # a member that waits this share of a round for the slowest is held up more than it is helped
KEPT = 400  # rounds the charts carry

WORKING, NOT_HELPING, DIVERGING, NOT_ENOUGH = "working", "not_helping", "diverging", "not_enough_data"
BASIS = "Tokens trained per second against the fastest member's measured compute speed alone; not loss per second."


def _rate(tokens, seconds):
    return tokens / seconds if tokens and seconds and seconds > 0 else None


def _mb_s(row, node, way):
    """How fast `node`'s transfer went this round, in MiB per second ("down": its weights, "up": its change), None when not measured."""
    bps = (row.get("node_rates", {}).get(node) or {}).get(way) or 0.
    return bps / 2 ** 20 if bps > 0 else None


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.


def record(row, losses, arrivals, theta_nbytes):
    """One closed round, measured. `row` is the history row being written (what the coordinator already knows of the round);
    `losses` maps each contributing member to its inner loss, `arrivals` to the seconds after the round opened that its change
    arrived (a member without one has a wait of None). `theta_nbytes` is the weights' size in full precision.

    Per member: loss, steps, tokens, compute_s, tokens_per_s, wait_s (the slowest member's arrival minus its own), sync_s (the rest
    of the round: download, upload, outer step, validation), down_mb_s and up_mb_s (the speed its weights came down and its change went
    up, MiB per second; None when not measured). compute, wait and sync add up to the round's seconds. idle_s is the seconds of the
    round the member's GPU was not computing: the round minus compute_s, which wait and sync make up when the exchange stops the
    steps, and which is near 0 when the exchange is overlapped with the next steps."""
    seconds = float(row.get("seconds") or 0.)
    contributed = list(row.get("contributed", []))
    known = {node: arrivals[node] for node in contributed if arrivals.get(node) is not None}
    last = max(known.values()) if known else None
    members = {}
    for node in contributed:
        compute = float(row.get("node_seconds", {}).get(node) or 0.)
        tokens = int(row.get("node_tokens", {}).get(node) or 0)
        wait = max(0., last - known[node]) if last is not None and node in known else None
        members[node] = {"loss": losses.get(node), "steps": int(row.get("steps", {}).get(node) or 0), "tokens": tokens,
                         "compute_s": compute, "tokens_per_s": _rate(tokens, compute), "wait_s": wait,
                         "sync_s": max(0., seconds - compute - (wait or 0.)), "idle_s": max(0., seconds - compute),
                         "down_mb_s": _mb_s(row, node, "down"), "up_mb_s": _mb_s(row, node, "up")}
    slowest = max(members, key=lambda node: members[node]["compute_s"], default=None) if len(members) > 1 else None
    up = [row.get("node_bytes", {}).get(node, 0) for node in contributed]
    full = theta_nbytes or 0
    return {"round": row["round"], "seconds": seconds, "tokens": int(row.get("tokens") or 0), "members": members, "slowest": slowest,
            "val_loss": row.get("val_loss"), "loss": row.get("loss"), "lr": row.get("lr"), "update_norm": row.get("outer_step_norm"),
            "bytes_up": int(row.get("delta_bytes") or 0), "bytes_down": int(row.get("theta_bytes") or 0), "dtype": row.get("delta_dtype", ""),
            "sparsity_up": float(row.get("delta_sparsity") or 0.), "sparsity_down": 0.,  # share of each tensor's entries sent: the changes may be sparse, the outer update members download is always dense
            "compression": full / (sum(up) / len(up)) if full and up and sum(up) > 0 else None,
            "sync_s": max(0., seconds - max((m["compute_s"] for m in members.values()), default=0.))}


def _rises(values):
    """How many rounds in a row the latest values have risen."""
    count = 0
    for before, after in zip(values[-2::-1], values[:0:-1]):
        if after > before:
            count += 1
        else:
            break
    return count


def _losses(rounds):
    return [r["val_loss"] if r.get("val_loss") is not None else r.get("loss") for r in rounds]


def _verdict(state, text):
    return {"state": state, "text": text}


def summarize(history):
    """The run's health from its history rows: the per-round records for the charts and the derived numbers and verdict. Rounds
    recorded before this existed (no `health`) are skipped."""
    rounds = [row["health"] for row in history if row.get("health")][-KEPT:]
    losses = [value for value in _losses(rounds) if value is not None]
    recent = [r for r in rounds if len(r["members"]) > 1][-WINDOW:]
    out = {"rounds": rounds, "basis": BASIS, "aggregate_tokens_per_s": None, "baseline_node": "", "baseline_tokens_per_s": None, "speedup": None,
           "comm_share": None, "wait_node": "", "wait_share": None, "straggler": "", "window": len(recent),
           "idle": {}, "idle_node": "", "idle_share": None}
    if losses and not math.isfinite(losses[-1]):
        out["verdict"] = _verdict(DIVERGING, "Diverging: the loss is no longer a number.")
    elif _rises(losses) >= DIVERGE:
        out["verdict"] = _verdict(DIVERGING, f"Diverging: the loss rose {_rises(losses)} rounds running.")
    if len(recent) < MIN_ROUNDS:
        out.setdefault("verdict", _verdict(NOT_ENOUGH, "Not enough data: it needs "
                                           f"{MIN_ROUNDS} rounds with two or more members to compare with one machine alone "
                                           f"({len(recent)} so far)."))
        return out
    seconds = sum(r["seconds"] for r in recent)
    tokens = {}
    compute = {}
    for r in recent:
        for node, m in r["members"].items():
            tokens[node] = tokens.get(node, 0) + m["tokens"]
            compute[node] = compute.get(node, 0.) + m["compute_s"]
    solo = {node: _rate(tokens[node], compute[node]) for node in tokens}
    solo = {node: rate for node, rate in solo.items() if rate}
    aggregate = _rate(sum(r["tokens"] for r in recent), seconds)
    if not solo or not aggregate:
        out.setdefault("verdict", _verdict(NOT_ENOUGH, "Not enough data: no member has reported a measurable speed yet."))
        return out
    fastest = max(solo, key=solo.get)
    speedup = aggregate / solo[fastest]
    waits = {}
    for r in recent:
        for node, m in r["members"].items():
            if m["wait_s"] is not None and r["seconds"] > 0:
                waits.setdefault(node, []).append(m["wait_s"] / r["seconds"])
    shares = {node: sum(v) / len(v) for node, v in waits.items()}
    idles = {}
    for r in recent:
        for node, m in r["members"].items():
            if r["seconds"] > 0:
                idles.setdefault(node, []).append(m.get("idle_s", (m["wait_s"] or 0.) + m["sync_s"]) / r["seconds"])
    idle = {node: sum(v) / len(v) for node, v in idles.items()}
    idler = max(idle, key=idle.get) if idle else ""
    waiter = max(shares, key=shares.get) if shares else ""
    last_ones = [r["slowest"] for r in recent if r["slowest"]]
    straggler = max(set(last_ones), key=last_ones.count) if last_ones else ""
    comm = sum(r["sync_s"] for r in recent) / seconds if seconds > 0 else None
    out.update(aggregate_tokens_per_s=aggregate, baseline_node=fastest, baseline_tokens_per_s=solo[fastest], speedup=speedup, comm_share=comm,
               wait_node=waiter, wait_share=shares.get(waiter), straggler=straggler,
               idle=idle, idle_node=idler, idle_share=idle.get(idler))
    if "verdict" in out:
        return out
    held = bool(waiter) and shares[waiter] >= WAITING and idle.get(waiter, 1.) >= WAITING  # a member that trains on while it waits (overlap) is not held up
    wait_s = _mean(m["wait_s"] for r in recent for node, m in r["members"].items() if node == waiter and m["wait_s"] is not None)
    sync_s = sum(r["sync_s"] for r in recent) / len(recent)
    # the straggler's extra compute is what it costs the waiter; blame it only when that explains the wait
    extra = (_mean(r["members"][straggler]["compute_s"] for r in recent if straggler in r["members"])
             - _mean(r["members"][waiter]["compute_s"] for r in recent if waiter in r["members"]))
    # transfer dominates: the round's sync is most of the waiter's wait (or nobody waits and the pair still loses to the solo)
    transfer = comm is not None and comm >= .3 and speedup < HELPING and (not held or sync_s >= .5 * wait_s)
    if transfer:
        out["verdict"] = _verdict(NOT_HELPING, f"Not helping: {speedup:.2f}× {fastest} alone; moving the model between machines takes "
                                               f"{comm:.0%} of each round ({sync_s:.0f} s of {seconds / len(recent):.0f} s). Raise H or "
                                               "speed up the link.")
    elif held and straggler and straggler != waiter and extra >= .5 * wait_s:
        out["verdict"] = _verdict(NOT_HELPING, f"Not helping: {waiter} waits {shares[waiter]:.0%} of each round for {straggler}.")
    elif speedup < HELPING:
        out["verdict"] = _verdict(NOT_HELPING, f"Not helping: {speedup:.2f}× the speed of {fastest} alone; the members add little to each other.")
    else:
        falling = len(losses) > 1 and losses[-1] < losses[0]
        out["verdict"] = _verdict(WORKING, f"Working: {speedup:.1f}× faster than {fastest} alone, "
                                           f"{'loss falling' if falling else 'loss not falling yet'}.")
    return out
