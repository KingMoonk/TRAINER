"""The merged queue: the jobs of every machine in the cluster, each with the machine it runs on or is pinned to. Pure Python: no Tk,
no server; the one network call is `fetch`, and tests pass a fake.

`Peers.survey` asks every listed peer for its `GET /node` and `GET /queue/jobs` at the same time (studio_cluster.run_together, so
the whole survey ends at the poll deadline) and remembers each answer. A peer that stops answering keeps its last queue and name, marked
stale, so its rows stay on the page (greyed, labelled) instead of vanishing. `merge` turns this machine's jobs and the peers' into
one list of rows. Slow: call it from a worker, never the Tk thread.
"""
from __future__ import annotations

import collections
import re
import socket
import threading
import time
from dataclasses import dataclass

import studio_cluster
from studio_peer import PeerClient

STATES = ("running", "waiting", "held")
BACKOFF_FIRST = 5.  # seconds before a peer that did not answer is asked again; doubled with each failure in a row
BACKOFF_MAX = 10.  # the longest wait between asks of a peer that stays silent: one that answers again is seen within this


def fetch(address, path, timeout=studio_cluster.POLL_TIMEOUT):
    """GET `path` of the node at `address` (one attempt: a peer that does not answer is offline for now)."""
    return PeerClient(address, timeout=timeout, retries=0).get_json(path)


@dataclass(frozen=True)
class Peer:
    info: studio_cluster.NodeInfo
    jobs: tuple = ()  # its GET /queue/jobs as last read: dicts
    stale: bool = True  # the jobs are the last known ones, not this survey's
    seen_at: float = 0.0  # when they were read (epoch seconds); 0 when never
    progress: dict | None = None  # its GET /queue running job as {job_id, step, total, eta_s}; None when idle, stale or unread
    limits: dict | None = None  # its GET /queue plan and verdict as `limits_of` reads them; None when stale or unread


@dataclass(frozen=True)
class Row:
    id: str
    kind: str
    label: str
    status: str  # running | waiting | held
    node: str  # the address of the machine it runs on (LOCAL for this one); for a waiting job, the machine it is pinned to; "" for any
    hostname: str  # that machine's name, or the pin as typed when no machine answers to it; "" when the job may go anywhere
    queued_on: str  # the name of the machine whose queue holds it
    queued_here: bool  # False for a job a peer queued
    pinned: str
    waiting_reason: str
    stale: bool  # the machine is not answering: this is its last known queue
    note: str  # the job's note: where a checkpoint a peer could not store is said (for example, that the bucket is full); "" when none
    queued_address: str = ""  # the address of the machine whose queue holds it (LOCAL for this one): what stopping it there needs
    step: int = 0  # a running job's progress, where its machine said it; 0 when unknown
    total: int = 0
    eta_s: float | None = None
    basis: str = ""  # a running job: why its machine was picked (it measured the fastest step for this design); "" when the rule decided


class Peers:
    """Each listed peer as last seen. One per server; `survey` is safe to call from several requests at once.

    A peer that did not answer `GET /node` is not asked again every survey: the wait doubles with each failed ask (BACKOFF_FIRST,
    then twice as long, up to BACKOFF_MAX), and meanwhile it is reported offline with the last reason, its last queue kept stale.
    When the wait is over one `GET /node` probes it, and only an answer is followed by its queue calls. A peer that answered last time
    and is only slow now keeps its last answer for this survey (stale) and is probed at the next: one stall is not offline."""

    def __init__(self, clock=time.time):
        self.clock, self._seen, self._lock = clock, {}, threading.Lock()
        self._failed = {}  # address -> (failed asks in a row, clock time of the next ask, the reason); 0 asks: slow once, probed next
        self._answered = {}  # address -> its NodeInfo when it last answered

    def survey(self, addresses, ask=None, timeout=studio_cluster.POLL_TIMEOUT):
        """One Peer per address, in order. `ask(address, path, timeout)` returns the parsed JSON (default: `fetch`)."""
        ask = ask or fetch
        addresses = list(addresses)
        now = self.clock()
        with self._lock:
            held = {a for a in addresses if a in self._failed and now < self._failed[a][1]}  # still backing off: not asked
            probing = {a for a in addresses if a in self._failed} - held  # backing off, wait over: only /node
        asked = [a for a in addresses if a not in held]
        answers = self._ask(ask, [(a, "node") for a in asked] + [(a, p) for a in asked if a not in probing for p in ("queue/jobs", "queue")],
                            timeout)
        recovered = [a for a in asked if a in probing and not isinstance(answers[(a, "node")], BaseException)]
        answers.update(self._ask(ask, [(a, p) for a in recovered for p in ("queue/jobs", "queue")], timeout))
        now, out = self.clock(), []
        with self._lock:
            self._seen = {a: self._seen[a] for a in addresses if a in self._seen}
            self._failed = {a: self._failed[a] for a in addresses if a in self._failed}
            self._answered = {a: self._answered[a] for a in addresses if a in self._answered}
            for address in addresses:
                last, seen_at, hostname = self._seen.get(address, ((), 0.0, ""))
                if address in held:
                    reason = self._failed[address][2]
                    out.append(Peer(studio_cluster.offline(address, reason, hostname), last, stale=True, seen_at=seen_at))
                    continue
                node, jobs, state = (answers.get((address, p)) for p in ("node", "queue/jobs", "queue"))
                if isinstance(node, BaseException) and studio_cluster.slow(node) and address in self._answered and address not in self._failed:
                    self._failed[address] = (0, now, studio_cluster.failure(address, node, timeout))
                    out.append(Peer(self._answered[address], last, stale=True, seen_at=seen_at))
                    continue
                if isinstance(node, BaseException):
                    info = studio_cluster.offline(address, studio_cluster.failure(address, node, timeout), hostname)
                else:
                    try:
                        info = studio_cluster.info_from_payload(address, node)
                    except ValueError as exc:
                        info = studio_cluster.offline(address, studio_cluster.failure(address, exc, timeout), hostname)
                if info.online:
                    self._failed.pop(address, None)
                    self._answered[address] = info
                    fresh = isinstance(jobs, list)
                    if fresh:
                        last, seen_at = tuple(j for j in jobs if isinstance(j, dict)), now
                    self._seen[address] = (last, seen_at, info.hostname)
                    out.append(Peer(info, last, stale=not fresh, seen_at=seen_at, progress=running_progress(state) if fresh else None,
                                    limits=limits_of(state) if fresh else None))
                else:
                    count = self._failed.get(address, (0,))[0] + 1
                    self._answered.pop(address, None)
                    self._failed[address] = (count, now + min(BACKOFF_MAX, BACKOFF_FIRST * 2 ** (count - 1)), info.error)
                    out.append(Peer(info, last, stale=True, seen_at=seen_at))
        return out

    @staticmethod
    def _ask(ask, calls, timeout):
        return dict(zip(calls, studio_cluster.run_together(calls, lambda call: ask(call[0], call[1], timeout), timeout)))


def running_progress(state):
    """{job_id, step, total, eta_s} from a machine's GET /queue, or None when it runs nothing or answered something else."""
    job = state.get("job") if isinstance(state, dict) else None
    if not isinstance(job, dict) or not isinstance(job.get("id"), str):
        return None
    step, total, eta = state.get("step"), state.get("total"), state.get("eta_s")
    number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
    return {"job_id": job["id"], "step": int(step) if number(step) else 0, "total": int(total) if number(total) else 0,
            "eta_s": float(eta) if number(eta) else None}


def limits_of(state):
    """The limits a machine's queue plan puts on its jobs, from its GET /queue: {"plan": the plan's settings, "blocked": why it is
    holding its jobs now (None when it is not), "paused": the person paused its queue}; None when it answered something else."""
    plan = state.get("plan") if isinstance(state, dict) else None
    if not isinstance(plan, dict):
        return None
    blocked = state.get("blocked")
    return {"plan": dict(plan), "blocked": blocked if isinstance(blocked, str) and blocked else None, "paused": bool(state.get("paused"))}


def own_node(cluster=None):
    """This machine as a NodeInfo: the cluster's cached probe when the queue has one, else just its name (the GPU probe is slow)."""
    if cluster is not None:
        return cluster.own()
    return studio_cluster.NodeInfo(address=studio_cluster.LOCAL, hostname=socket.gethostname())


def _find(nodes, ref):
    """The one machine `ref` (an address or a typed pin) names, or None when none does or a hostname two machines share does."""
    found = studio_cluster.pinned_nodes(ref, nodes)
    return found[0] if len(found) == 1 else None


def _name(node):
    return node.hostname or node.address


def _names(nodes):
    """Each listed machine's name by address: its hostname, with its address added when another listed machine has the same hostname
    (the pin uses the address, so the address tells them apart). This machine keeps its bare hostname: its address is only LOCAL."""
    counts = collections.Counter(n.hostname.lower() for n in nodes if n.hostname)
    return {n.address: f"{n.hostname} ({n.address})" if n.hostname and counts[n.hostname.lower()] > 1 and n.address != studio_cluster.LOCAL
            else _name(n) for n in nodes}


def _pinned(params):
    value = params.get("pinned") if isinstance(params, dict) else None
    return value.strip() if isinstance(value, str) else ""


def merge(own, peers, jobs, waiting=None, progress=None, basis=None):
    """The rows of the whole cluster: `jobs` are this machine's (running, running on a peer, Up next), the peers' come from their
    surveys. A job a peer runs for another node is left to its owner's row, and a peer's job that runs here is already in `jobs`.
    `progress` is this machine's running job as {job_id, step, total, eta_s}; a running row takes the progress of the machine that runs
    it (a job placed on a peer runs there as "<id>-e<epoch>"). A running row says why its machine was picked: the sentence its owner
    kept when it dispatched the job (`params["cluster"]["basis"]`), or, for one running here, `basis` (job id -> the sentence)."""
    waiting, basis = waiting or {}, basis or {}
    reports = [p for p in (progress, *(peer.progress for peer in peers)) if p]
    nodes = [own, *(p.info for p in peers)]
    names = _names(nodes)

    def row(job_id, kind, label, status, params, ref, queued_on, here, why, note, peer=None, queued_address=studio_cluster.LOCAL):
        node = _find(nodes, ref) if ref else None
        report = next((r for r in reports if r["job_id"] == job_id or re.fullmatch(re.escape(job_id) + r"-e\d+", r["job_id"])), None) \
            if status == "running" else None
        notes = params.get("cluster") if isinstance(params, dict) else None
        picked = (notes.get("basis") if isinstance(notes, dict) else "") or (basis.get(job_id, "") if here else "")
        return Row(job_id, kind, label, status, node.address if node else "", names[node.address] if node else ref, queued_on, here,
                   _pinned(params), why, bool(peer and peer.stale), note, queued_address,
                   report["step"] if report else 0, report["total"] if report else 0, report["eta_s"] if report else None,
                   picked if status == "running" and isinstance(picked, str) else "")

    rows = []
    for job in jobs:
        if job.status in STATES:
            ref = (getattr(job, "node", "") or studio_cluster.LOCAL) if job.status == "running" else _pinned(job.params)
            rows.append(row(job.id, job.kind, job.label or "", job.status, job.params, ref, names[own.address], True, waiting.get(job.id, ""),
                            job.note or ""))
    for peer in peers:
        for job in peer.jobs:
            status, params = job.get("status"), job.get("params")
            if job.get("owner") or status not in STATES or not isinstance(job.get("id"), str):
                continue
            ref = (job.get("node") or peer.info.address) if status == "running" else _pinned(params)
            if status == "running" and _find(nodes, ref) is own:
                continue
            rows.append(row(job["id"], str(job.get("kind", "")), str(job.get("label") or ""), status, params, ref, names[peer.info.address],
                            False, "", str(job.get("note") or ""), peer, peer.info.address))
    return rows
