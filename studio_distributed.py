"""The distributed pretraining runs this node owns, as the Cluster page's Training tab reads and controls them.

A run is a `pretrain_distributed` job (its id is the run id): the coordinator lives in this process (studio_diloco.COORDINATORS) and
this computer takes part as one member. The machines the person chose are invited when the run starts: a `diloco_member` job is
put on each one's own queue (the scheduler never places that kind, so it is the owner that asks), and the member joins the coordinator
from there. This module keeps who was invited and what came of it, and merges that with the coordinator's own account of the run.
"""
from __future__ import annotations

import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from studio_peer import PeerClient, PeerError

KEPT = 20  # runs remembered, newest last
RESTARTED = "The app restarted while this run was going, so it was stopped."
INVITE_TIMEOUT = 5.  # seconds a peer has to take its member job
ACTIVE = ("queued", "starting", "waiting", "running")
UNPLACED = "It was busy when the run was placed. It joins a later round if it is free by then."
STATES = ("queued", "starting", "waiting", "running", "finished", "stopped", "failed")

RUNS = {}  # run id -> Run
_LOCK = threading.Lock()
_HOME = None  # data/distributed_runs.json of the workspace the runs are kept in (see load); None: they live in memory only
_WRITE = threading.Lock()
_REMEMBER = threading.Lock()


@dataclass
class Machine:
    """One machine of a run. `node` is the name its member joins under; `error` is why it could not be reached when the run was set up."""
    node: str
    address: str  # "local" for this computer, else the listed address
    device: str = ""
    device_name: str = ""
    error: str = ""
    state: str = "invited"  # invited | unplaced (the scheduler did not give it a slot: busy, or past max_members) | unreachable
    reason: str = ""
    peer_job: str = ""  # the member job's id on that machine's queue

    @property
    def local(self):
        return self.address == "local"


@dataclass
class Run:
    id: str
    label: str
    version: int | None
    started_at: float
    settings: dict
    owner: str  # the hostname of the computer that coordinates
    machines: list = field(default_factory=list)
    final: dict | None = None  # the coordinator's last snapshot, kept when the coordinator is let go
    stopped: bool = False  # the person asked to stop it before it ran
    term: int = 1  # the coordinator's term when its account was kept (archive)
    saved: int | None = None  # the version whose weights this run saved when it finished
    restarted: bool = False  # read back from the workspace while its last account still said it was going


def machines_of(rows):
    """Machines from the dicts the start route stores in the job's params; node names are made unique."""
    out, seen = [], set()
    for row in rows:
        node = row["node"]
        if node in seen:
            node = f"{node} ({row['address']})"
        seen.add(node)
        out.append(Machine(node=node, address=row["address"], device=row.get("device", ""), device_name=row.get("device_name", ""),
                           error=row.get("error", ""), **({"state": "unreachable", "reason": row["error"]} if row.get("error") else {})))
    return out


def run_of(job):
    """The Run of an owner job; made from the job's params when this process has none (the app restarted with the job waiting)."""
    with _LOCK:
        run = RUNS.get(job.id)
        made = run is None
        if made:
            p = job.params
            run = RUNS[job.id] = Run(job.id, job.label, p.get("version"), job.created_at or time.time(), dict(p.get("settings") or {}),
                                     p.get("owner") or socket.gethostname(), machines_of(p.get("machines", [])))
            for key in list(RUNS)[:-KEPT]:
                del RUNS[key]
    if made:
        persist()
    return run


def local_runs(jobs, create=False, coordinators=None):
    """[(run, coordinator, job)] for every run this process knows: its remembered ones, and with `create` an owner job in `jobs` that
    has no record yet. `coordinator` is the live one or None; `job` the owner job or None. `coordinators` is the registry to look in (the
    process's, studio_diloco.COORDINATORS, unless a test runs several nodes in one process)."""
    import studio_diloco
    owner_jobs = {job.id: job for job in jobs if job.kind == "pretrain_distributed"}
    if create:
        for job in owner_jobs.values():
            run_of(job)
    found = studio_diloco.COORDINATORS if coordinators is None else coordinators
    return [(run, found.get(run.id), owner_jobs.get(run.id)) for run in list(RUNS.values())]


def adopt(row):
    """The Run record of a run this node took over from another coordinator (studio_failover): made from the run summary that coordinator
    published, so the tab and the store list it here. A record already held is kept."""
    with _LOCK:
        run = RUNS.get(row["id"])
        if run is None:
            run = RUNS[row["id"]] = Run(row["id"], row.get("label") or "", row.get("version"), row.get("started_at") or time.time(),
                                        dict(row.get("settings") or {}), row.get("owner") or "")
            for key in list(RUNS)[:-KEPT]:
                del RUNS[key]
    persist()
    return run


def note_saved(run_id, number):
    """Remember that a finished run's weights were saved as version `number`, so the tab can name it."""
    with _LOCK:
        if run_id in RUNS:
            RUNS[run_id].saved = number


def open_run(job):
    """Remember a run the moment its job is queued, so the tab lists it while it waits for its turn."""
    return run_of(job)


def archive(run_id, coordinator):
    """Keep a finished coordinator's account of the run before it is released."""
    run = RUNS.get(run_id)
    if run is not None and run.final is None:
        run.final = coordinator.snapshot()
        run.term = getattr(coordinator, "term", run.term)
        persist()


def keeps(run_id):
    """Whether the account of this run is being kept in a workspace."""
    return _HOME is not None and run_id in RUNS


def remember(run_id, coordinator):
    """Keep the coordinator's account of the run as it stands (after each round and when it stops), so it outlives the app."""
    with _REMEMBER:  # two of these never write their snapshots out of order
        run = RUNS.get(run_id)
        if run is not None:
            run.final = coordinator.snapshot()
            persist()


# ---- keeping the records ----------------------------------------------------------------------------------------------

def persist():
    """Write the runs to the workspace, atomically. Best effort: a record that cannot be written is only lost if the app restarts."""
    if _HOME is None or not _HOME.parent.parent.exists():
        return
    from studio_core import atomic_json
    with _WRITE:
        try:
            atomic_json(_HOME, {"format": 1, "runs": [asdict(run) for run in list(RUNS.values())]})
        except (OSError, TypeError, ValueError):
            pass


def _run_from(row):
    machines = [Machine(**{k: v for k, v in m.items() if k in Machine.__dataclass_fields__}) for m in row.get("machines", [])]
    run = Run(**{k: v for k, v in row.items() if k in Run.__dataclass_fields__ and k != "machines"}, machines=machines)
    if run.final is not None and run.final.get("state") in ("running", "waiting"):
        run.restarted = True  # its coordinator went with the app; unless the queue resumes the job, the run is over
    return run


def unload(root):
    """Stop writing to `root` (its services are closed); the runs stay in memory."""
    global _HOME
    if _HOME == Path(root) / "data" / "distributed_runs.json":
        _HOME = None


def load(root):
    """Keep the runs in `root`/data/distributed_runs.json from now on, and read back the ones kept there (the newest KEPT). A missing,
    unreadable or malformed file, or a row that does not read, gives no run and never fails the start. Runs this process already holds
    are left as they are."""
    global _HOME
    from studio_core import read_json
    _HOME = Path(root) / "data" / "distributed_runs.json"
    try:
        rows = read_json(_HOME)["runs"] if _HOME.exists() else []
    except (OSError, ValueError, KeyError, TypeError):
        rows = []
    found = []
    for row in rows if isinstance(rows, list) else []:
        try:
            found.append(_run_from(row))
        except (TypeError, ValueError, AttributeError):
            continue
    with _LOCK:
        for run in sorted(found, key=lambda r: r.started_at):
            RUNS.setdefault(run.id, run)
        for key in list(RUNS)[:-KEPT]:
            del RUNS[key]


# ---- inviting ---------------------------------------------------------------------------------------------------------

def refusal(machine, exc):
    """Why a machine did not take its member job, in one sentence."""
    if isinstance(exc, PeerError) and exc.status in (401, 403):
        return (f"{machine.node} refused this computer ({exc.detail or exc.status}). Add this computer's Tailscale login to the "
                f"allowed users on {machine.node}.")
    import studio_cluster
    return studio_cluster.failure(machine.address, exc)


def loopback(address):
    from urllib.parse import urlsplit
    host = urlsplit(address if "://" in address else f"//{address}").hostname or ""
    return host in ("localhost", "127.0.0.1", "::1")


def invite(job, client_for=PeerClient, only=None):
    """Put a member job on each machine the scheduler gave a slot (`gang` in the job's params: the addresses; every chosen machine when
    absent) or, with `only`, on just those addresses. A machine that cannot be reached, or refuses, is recorded with the reason and
    the run goes on without it; one without a slot is `unplaced` until a slot opens. Never raises: the run does not depend on the peers."""
    run, p = run_of(job), job.params
    url = p.get("peer_url")
    gang = set(only) if only is not None else (set(p["gang"]) if p.get("gang") is not None else None)

    def ask(machine):
        if machine.local or (machine.state == "unreachable" and only is None):
            return
        if gang is not None and machine.address not in gang:
            if only is None and not machine.peer_job:
                machine.state, machine.reason = "unplaced", UNPLACED
            return
        machine.state, machine.reason = "invited", ""
        if not url or (loopback(url) and not loopback(machine.address)):
            machine.state, machine.reason = "unreachable", ("This computer's name on the network is not known, so " + machine.node
                                                             + " cannot be told where to join. Set TRAINER_NODE_ADDRESS to it.")
            return
        body = {"kind": "diloco_member", "label": f"{job.label} (member)",
                "params": {"coordinator_url": url, "run": job.id, "node": machine.node, "epoch": 0,
                              **({"member_options": p["member_options"]} if p.get("member_options") else {})}}
        try:
            machine.peer_job = client_for(machine.address, timeout=INVITE_TIMEOUT, retries=0).post_json("/queue/jobs", body)["id"]
        except Exception as exc:  # noqa: BLE001 - whatever went wrong is the machine's note
            machine.state, machine.reason = "unreachable", refusal(machine, exc)

    todo = [m for m in run.machines if not m.local and not (m.peer_job and m.state == "invited")]
    if todo:
        with ThreadPoolExecutor(max_workers=len(todo)) as pool:
            list(pool.map(ask, todo))
        persist()  # who answered


def release_member(run, node, client_for=PeerClient):
    """Take a stopped member's job off its machine's queue, so it does not stay there. Best effort: the member also ends at its next call."""
    machine = next((m for m in run.machines if m.node == node), None)
    if machine is None or machine.local or not machine.peer_job:
        return
    try:
        client_for(machine.address, timeout=INVITE_TIMEOUT, retries=0).delete_json(f"/queue/jobs/{machine.peer_job}")
    except Exception:  # noqa: BLE001 - the member is already fenced out of the run
        pass


def lapse(run, node, client_for=PeerClient):
    """A member's slot lapsed: take its member job off its machine's queue, and show the machine as unplaced until a slot opens for it."""
    release_member(run, node, client_for)
    machine = next((m for m in run.machines if m.node == node), None)
    if machine is not None and not machine.local:
        machine.peer_job, machine.state = "", "unplaced"
        machine.reason = "Its lease lapsed, so the run let it go. It joins a later round if it is free by then."


# ---- the read model ---------------------------------------------------------------------------------------------------

def round_rows(history):
    return [{"round": row["round"], "step": row["step"], "loss": row.get("loss"), "val_loss": row.get("val_loss"),
             "bytes_up": row.get("delta_bytes", 0), "bytes_down": row.get("theta_bytes", 0), "seconds": row.get("seconds", 0.),
             "members": len(row.get("members", [])), "contributed": len(row.get("contributed", [])),
             "dropped": list(row.get("dropped", []))} for row in history]


def member_stats(history, node):
    """What a member contributed over the rounds so far: rounds, steps, and its last round's speed and delta."""
    mine = [row for row in history if node in row.get("contributed", [])]
    out = {"rounds_done": len(mine), "steps": sum(row.get("steps", {}).get(node, 0) for row in mine),
           "tokens_per_second": None, "last_delta_bytes": None, "last_delta_dtype": ""}
    if mine:
        last = mine[-1]
        tokens, seconds = last.get("node_tokens", {}).get(node), last.get("node_seconds", {}).get(node)
        out["tokens_per_second"] = tokens / seconds if tokens and seconds and seconds > 0 else None
        out["last_delta_bytes"] = last.get("node_bytes", {}).get(node)
        out["last_delta_dtype"] = last.get("delta_dtype", "")
    return out


def left_after_finish(snap, node):
    """Whether `node` left the run only after the run had finished (the coordinator's own member does, once the last round is done):
    its `left` event comes after the run's `finished` event."""
    events = (snap or {}).get("events", [])
    finished = [e["seq"] for e in events if e["kind"] == "finished"]
    return bool(finished) and any(e["kind"] == "left" and e["node"] == node and e["seq"] > finished[0] for e in events)


NEVER_JOINED = "It was invited but never joined the run."


def member_rows(run, snap, ended):
    """One row per machine chosen, then any other node the coordinator knows. A machine the coordinator has not heard from yet is
    `invited` (or, for this computer, starting) until it joins."""
    nodes = (snap or {}).get("nodes", {})
    history = (snap or {}).get("history", [])
    rows = []
    for machine in run.machines:
        known = nodes.get(machine.node)
        rows.append(_member(machine.node, machine, known, history, ended, snap))
    chosen = {m.node for m in run.machines}
    rows.extend(_member(node, None, known, history, ended, snap) for node, known in nodes.items() if node not in chosen)
    return rows


def live_fields(known):
    """What the member last reported (studio_diloco.Coordinator.snapshot: `progress`, `shards`); empty for a node not heard from."""
    progress = (known or {}).get("progress") or {}
    return {"phase": progress.get("phase"), "step_in_round": progress.get("step", 0), "round_steps": progress.get("steps", 0),
            "loss": progress.get("loss"), "live_tokens_per_second": progress.get("tokens_per_second"),
            "reported_ago": progress.get("reported_ago"), "shards": (known or {}).get("shards", 0), "samples": list(progress.get("samples", []))}


def timeline_rows(timeline):
    """The coordinator's per-round timelines with the swimlanes as a list, in the order the members joined."""
    return [{**line, "lanes": [{"node": node, **lane} for node, lane in line["lanes"].items()]} for line in timeline]


def phase_of(snap, state):
    """The run's phase for the live view: the coordinator's while it is going, finished or stopped once it is over."""
    if state in ("finished", "stopped"):
        return state
    if state == "failed":
        return "stopped"
    return (snap or {}).get("phase") or "waiting"


def _member(node, machine, known, history, ended, snap=None):
    local = machine is not None and machine.local
    row = {"node": node, "address": machine.address if machine else "", "local": local, "device": machine.device if machine else "",
           "device_name": machine.device_name if machine else "", "missed": 0, "sent": False, **member_stats(history, node),
           **live_fields(known)}
    if machine is not None and machine.state == "unreachable":
        state, note = "unreachable", machine.reason
    elif known is not None:
        state, note, row["missed"], row["sent"] = known["state"], known["note"], known["missed"], known["sent"]
        if ended and state in ("training", "behind", "joining"):
            state = "finished" if ended == "finished" else "stopped"
        elif ended == "finished" and state == "left" and left_after_finish(snap, node):
            state, note = "finished", ""  # it left because the run was over, not because anything went wrong
    elif ended and machine is not None and not local:
        # never heard from, so it never joined: keep why (no slot, or invited and silent) instead of calling it finished
        state, note = ("unplaced", machine.reason) if machine.state == "unplaced" else ("never_joined", NEVER_JOINED)
    elif ended:
        state, note = ("finished" if ended == "finished" else "stopped"), ""
    elif machine is not None and machine.state == "unplaced":
        state, note = "invited", machine.reason
    else:
        state, note = "invited", "Starting." if local else "Invited. It joins once its own queue gets to the member job."
    row["state"], row["note"] = state, note
    placed = machine is None or machine.state != "unplaced"
    row["can_stop"] = not ended and not local and placed and state in ("invited", "training", "behind", "joining")
    return row


def view(run, coordinator, job):
    """The run as the tab shows it. `coordinator` is the live one or None; `job` is the owner job from the queue (anything with
    status and error) or None when the queue no longer lists it."""
    snap = coordinator.snapshot() if coordinator is not None else run.final
    note = ""
    if snap is not None:
        state = snap["state"]
        if state in ("running", "waiting") and coordinator is None:
            state = "stopped"  # a snapshot of a coordinator that was let go while it still ran
            if run.restarted:  # or one that went with the app: stopped, unless the queue still holds the job and resumes it
                state, note = ("queued", "") if job is not None and job.status in ("waiting", "held") else ("stopped", RESTARTED)
    elif run.stopped or job is None or job.status in ("stopped", "done"):
        state = "stopped"
        note = "" if run.stopped else "The app no longer holds this run."
    else:
        state = "queued" if job.status in ("waiting", "held") else "starting"
    if job is not None and job.status == "failed" and state != "finished":
        state, note = "failed", job.error or "This computer's own member stopped."
    if state == "waiting":
        note = "Waiting for enough members to join."
    ended = state in ("finished", "stopped", "failed") and (state if state != "failed" else "stopped")
    settings = (snap or {}).get("settings") or run.settings
    return {"id": run.id, "label": run.label, "version": run.version, "state": state, "note": note, "started_at": run.started_at,
            "owner": run.owner, "saved_version": run.saved, "term": getattr(coordinator, "term", run.term), "round": (snap or {}).get("round", 0), "rounds": settings.get("rounds", 0),
            "inner_steps": settings.get("inner_steps", 0), "outer_lr": settings.get("outer_lr", 0.), "outer_momentum": settings.get("outer_momentum", 0.),
            "delta_dtype": settings.get("delta_dtype", ""), "members": member_rows(run, snap, ended or ""),
            "history": round_rows((snap or {}).get("history", [])), "can_stop": state in ACTIVE,
            "phase": phase_of(snap, state), "deadline_s": (snap or {}).get("deadline_s"), "grace_s": (snap or {}).get("grace_s", 0.),
            "window_s": (snap or {}).get("window_s"), "contributed": (snap or {}).get("contributed", 0), "expected": (snap or {}).get("expected", 0),
            "theta_bytes": (snap or {}).get("theta_bytes", 0), "now_s": (snap or {}).get("now_s", 0.),
            "timeline": timeline_rows((snap or {}).get("timeline", [])), "events": list((snap or {}).get("events", []))}


def stop_run(run, coordinator, client_for=PeerClient):
    """End a run: its coordinator tells every member it is finished, and the member jobs are taken off the machines' queues."""
    run.stopped = True
    if coordinator is not None:
        coordinator.stop()
    persist()
    for machine in run.machines:
        release_member(run, machine.node, client_for)
