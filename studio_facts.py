"""What each node writes into the observation store, and how the cluster-wide views read it back (docs/distributed.md, "Cluster state:
what every node knows", "Reading and writing"; studio_state has the store itself).

* `publish(cluster)` is the origin's side, run at every poll (studio_remote.Cluster.poll_once) and once at start, so a restarted node's new
  incarnation replaces its old entries at once. Every entry is rewritten every time, so its age is the time since this node last
  looked, and the heartbeat keeps the node from being pruned while it lives. Only this node's own origin is ever written; a command
  (stop a run, cancel a job) goes to the owner through PeerClient and never writes here.
* The reads (`cluster_runs`, `node_facts`, `queue_peers`) answer from the local replica. Each fact carries `origin` and `age_s`; an
  origin that is not answering is marked down and keeps its last facts, with growing age.

Entries are summaries: a run without its timelines and events, the last HISTORY_KEPT rounds, the last EVENTS_KEPT events, the last
TIMELINES_KEPT round timelines. Weights and shards never go through the store.

kind        key       written by          value
heartbeat   node      every origin        {"beat": n}
node        node      every origin        the NodeInfo payload, plus {"peers": n answering}
queue       jobs      every origin        the running and waiting jobs queued here, as GET /queue/jobs lists them
leases      granted   every origin        the leases this node granted, with seconds since the runner was heard
events      node      every origin        the last EVENTS_KEPT cluster events
runs        index     a coordinator       {"ids": the runs it publishes, "live": those it coordinates now}
run         run id    a coordinator       the run summary (the Training tab's row)
timeline    run id    a coordinator       the last TIMELINES_KEPT round timelines
run_events  run id    a coordinator       the run's last EVENTS_KEPT events
progress    run id    a member            its own latest progress report
held        run id    a replica           {"term", "round", "rank", "ready", "held"}: the round it keeps (studio_failover)
claim       run id    a coordinator       {"term", "round", "rank"}, or {"released": true}: who coordinates the run since a failover
shards      run id    a member            the ids of the shards it has cached
"""
from __future__ import annotations

import dataclasses

EVENTS_KEPT = 200
TIMELINES_KEPT = 8
HISTORY_KEPT = 50
DOWN_AFTER = 20.  # seconds without a heartbeat after which an origin is marked down, when no poll of it says so sooner
QUEUE_STATES = ("running", "waiting", "held")


# ---- writing (the origin's own facts) ---------------------------------------------------------------------------------

def _queue_rows(runner):
    with runner.lock:
        jobs = list(runner.jobs)
        running, eta = getattr(runner, "running", None), (getattr(runner, "progress", None) or {}).get("eta_s")
    return [{"id": j.id, "kind": j.kind, "label": j.label or "", "status": j.status, "node": getattr(j, "node", "") or "",
             "owner": getattr(j, "owner", "") or "", "note": j.note or "", "eta_s": eta if j is running else None,
             "params": {"pinned": j.params.get("pinned")} if j.params.get("pinned") else {}}
            for j in jobs if j.status in QUEUE_STATES]


def run_summary(row):
    """A run's row without what has its own entries (timeline, events), its history cut to the last rounds."""
    return {**{k: v for k, v in row.items() if k not in ("timeline", "events")}, "history": row["history"][-HISTORY_KEPT:]}


def publish(cluster):
    """Write this node's facts into its replica. Never raises for one bad source: the others are still written."""
    replica, now = cluster.state, cluster.clock()
    cluster.beat += 1
    replica.put("heartbeat", "node", {"beat": cluster.beat})
    steps = (
        lambda: replica.put("node", "node", {**dataclasses.asdict(dataclasses.replace(cluster.own(), busy=cluster.busy())),
                                             "dtypes": list(cluster.own().dtypes), "peers": sum(1 for i in cluster._known.values() if i.online)}),
        lambda: replica.put("queue", "jobs", _queue_rows(cluster.runner)),
        lambda: replica.put("leases", "granted", [{**lease.as_dict(), "silent_s": silent} for lease, silent in cluster.owner.leases.held()]),
        lambda: replica.put("events", "node", [{**{k: v for k, v in e.items() if k != "at"}, "ago": max(0., now - e["at"])}
                                               for e in list(cluster.events)[-EVENTS_KEPT:]]),
        lambda: _publish_runs(cluster),
        lambda: _publish_member(cluster),
        lambda: _publish_failover(cluster),
    )
    for step in steps:
        try:
            step()
        except Exception:  # noqa: BLE001 - one source failing must not stop the heartbeat or the other facts
            pass


def _publish_runs(cluster):
    import studio_distributed as runs
    replica, ids, live = cluster.state, [], []
    for run, coordinator, job in cluster.local_runs():
        if coordinator is None and run.final is None and job is None:
            continue  # the app no longer holds this run (a later term may coordinate it elsewhere): not this node's fact
        row = runs.view(run, coordinator, job, runs.peer_facts(run, cluster))
        ids.append(run.id)
        if coordinator is not None:
            live.append(run.id)
        replica.put("run", run.id, run_summary(row))
        replica.put("timeline", run.id, row["timeline"][-TIMELINES_KEPT:])
        replica.put("run_events", run.id, row["events"][-EVENTS_KEPT:])
    replica.put("runs", "index", {"ids": ids, "live": live})  # live: the ones this node coordinates now (the rest is a kept account)


def _publish_member(cluster):
    for run_id, found in list(cluster.members.items()):
        cluster.state.put("progress", run_id, {k: v for k, v in found.items() if k != "shard_ids"})
        cluster.state.put("shards", run_id, {"ids": list(found.get("shard_ids") or [])})


def _publish_failover(cluster):
    """The replica this node's member keeps (kind `held`), and the claim of the coordinator this node runs (kind `claim`), each rewritten at
    every poll, so their ages say how long ago this node last looked (studio_failover reads both)."""
    for run_id, found in list(cluster.replicas.items()):
        cluster.state.put("held", run_id, dict(found))
    for run_id, found in list(cluster.claims.items()):
        cluster.state.put("claim", run_id, dict(found))


def member_fact(event, now):
    """The 'progress' value of a `diloco_member` event a member's worker emitted."""
    return {"node": event.get("node", ""), "term": event.get("term", 0), "round": event.get("round"), "phase": event.get("phase"),
            "step_in_round": event.get("step_in_round", 0), "round_steps": event.get("round_steps", 0), "loss": event.get("loss"),
            "tokens_per_second": event.get("tokens_per_second"), "at": now, "shard_ids": list(event.get("shard_ids") or [])}


# ---- reading (any origin's facts, from the local replica) --------------------------------------------------------------

def origins(cluster):
    """Every origin the replica holds, this node's included."""
    return sorted({e["origin"] for e in cluster.state.entries(kind="heartbeat")})


def heartbeat_age(cluster, origin):
    beat = cluster.state.get(origin, "heartbeat", "node")
    return None if beat is None else beat["age"]


def stamp(cluster, origin, age):
    """The labels every fact read from the replica carries."""
    return {"origin": origin, "age_s": 0. if origin == cluster.hostname else age, "origin_down": cluster.is_down(origin)}


def overlay_members(cluster, row):
    """Each member's own latest report (the 'progress' entry its node wrote), when it is newer than what the coordinator last heard."""
    replica = cluster.state
    for member in row["members"]:
        found = replica.get(member["node"], "progress", row["id"])
        if found is None or found["value"].get("term") not in (0, row.get("term")):
            continue
        age, mine = found["age"], found["value"]
        if member.get("reported_ago") is None or age < member["reported_ago"]:
            member.update(phase=mine.get("phase"), step_in_round=mine.get("step_in_round", 0), round_steps=mine.get("round_steps", 0),
                          loss=mine.get("loss"), live_tokens_per_second=mine.get("tokens_per_second"), reported_ago=age)
    return row


def _remote_run(cluster, origin, run_id):
    replica = cluster.state
    found = replica.get(origin, "run", run_id)
    if found is None:
        return None
    row = dict(found["value"])
    row["members"] = [dict(m) for m in row.get("members", [])]
    for member in row["members"]:
        if member.get("reported_ago") is not None:
            member["reported_ago"] += found["age"]  # the coordinator's count, plus the time since it wrote it
    timeline, events = replica.get(origin, "timeline", run_id), replica.get(origin, "run_events", run_id)
    row["timeline"], row["events"] = (timeline["value"] if timeline else []), (events["value"] if events else [])
    row["coordinator"] = origin
    return {**row, **stamp(cluster, origin, found["age"])}


def cluster_runs(cluster, local_rows):
    """Every run in the cluster, newest first: `local_rows` (this node's, read live: age 0) and the runs the replica holds for the
    other origins. A run two origins both publish (a coordinator moved to a new term) is the one of the higher term, else the fresher."""
    best = {}
    for row in local_rows:
        row = {**row, "coordinator": cluster.hostname, **stamp(cluster, cluster.hostname, 0.)}
        best[row["id"]] = row
    for origin in origins(cluster):
        if origin == cluster.hostname:
            continue
        index = cluster.state.get(origin, "runs", "index")
        for run_id in (index["value"].get("ids", []) if index else []):
            row = _remote_run(cluster, origin, run_id)
            held = best.get(run_id)
            if row is not None and (held is None or (row.get("term", 1), -row["age_s"]) > (held.get("term", 1), -held["age_s"])):
                best[run_id] = row
    rows = [overlay_members(cluster, row) for row in best.values()]
    return sorted(rows, key=lambda row: row["started_at"], reverse=True)


def node_facts(cluster, hostname):
    """(origin, age_s, down, node record) for the origin named `hostname`, or None when the replica holds nothing for it."""
    if not hostname or cluster.state.get(hostname, "heartbeat", "node") is None:
        return None
    beat = cluster.state.get(hostname, "heartbeat", "node")
    node = cluster.state.get(hostname, "node", "node")
    return hostname, (0. if hostname == cluster.hostname else beat["age"]), cluster.is_down(hostname), (node["value"] if node else {})


def queue_facts(cluster, origin):
    """(jobs, age_s) of the queue entry an origin wrote, or ([], None)."""
    found = cluster.state.get(origin, "queue", "jobs")
    return (list(found["value"]), found["age"]) if found else ([], None)
