"""The merged queue: the jobs of every machine in the cluster, each with the machine it runs on or is pinned to. Pure Python: no Tk,
no server; the one network call is `fetch`, and tests pass a fake.

`Peers.survey` asks every listed peer for its `GET /node` and `GET /queue/jobs` at the same time (studio_cluster.run_together, so
the whole survey ends at the poll deadline) and remembers each answer. A peer that stops answering keeps its last queue and name, marked
stale, so its rows stay on the page (greyed, labelled) instead of vanishing. `merge` turns this machine's jobs and the peers' into
one list of rows. Slow: call it from a worker, never the Tk thread.
"""
from __future__ import annotations

import collections
import dataclasses
import socket
import threading
import time
from dataclasses import dataclass

import studio_cluster
from studio_peer import PeerClient

STATES = ("running", "waiting", "held")


def fetch(address, path, timeout=studio_cluster.POLL_TIMEOUT):
    """GET `path` of the node at `address` (one attempt: a peer that does not answer is offline for now)."""
    return PeerClient(address, timeout=timeout, retries=0).get_json(path)


@dataclass(frozen=True)
class Peer:
    info: studio_cluster.NodeInfo
    jobs: tuple = ()  # its GET /queue/jobs as last read: dicts
    stale: bool = True  # the jobs are the last known ones, not this survey's
    seen_at: float = 0.0  # when they were read (epoch seconds); 0 when never


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


class Peers:
    """Each listed peer as last seen. One per server; `survey` is safe to call from several requests at once."""

    def __init__(self, clock=time.time):
        self.clock, self._seen, self._lock = clock, {}, threading.Lock()

    def survey(self, addresses, ask=None, timeout=studio_cluster.POLL_TIMEOUT):
        """One Peer per address, in order. `ask(address, path, timeout)` returns the parsed JSON (default: `fetch`)."""
        ask = ask or fetch
        addresses = list(addresses)
        calls = [(a, path) for a in addresses for path in ("node", "queue/jobs")]
        answers = iter(studio_cluster.run_together(calls, lambda call: ask(call[0], call[1], timeout), timeout))
        now, out = self.clock(), []
        with self._lock:
            self._seen = {a: self._seen[a] for a in addresses if a in self._seen}
            for address in addresses:
                node, jobs = next(answers), next(answers)
                if isinstance(node, BaseException):
                    info = studio_cluster.offline(address, studio_cluster.failure(address, node, timeout))
                else:
                    try:
                        info = studio_cluster.info_from_payload(address, node)
                    except ValueError as exc:
                        info = studio_cluster.offline(address, studio_cluster.failure(address, exc, timeout))
                if info.online:
                    last, seen_at, _ = self._seen.get(address, ((), 0.0, ""))
                    fresh = isinstance(jobs, list)
                    if fresh:
                        last, seen_at = tuple(j for j in jobs if isinstance(j, dict)), now
                    self._seen[address] = (last, seen_at, info.hostname)
                    out.append(Peer(info, last, stale=not fresh, seen_at=seen_at))
                else:
                    last, seen_at, hostname = self._seen.get(address, ((), 0.0, ""))
                    out.append(Peer(dataclasses.replace(info, hostname=hostname), last, stale=True, seen_at=seen_at))
        return out


def own_node(cluster=None):
    """This machine as a NodeInfo: the cluster's cached probe when the queue has one, else just its name (the GPU probe is slow)."""
    if cluster is not None:
        return cluster.own()
    return studio_cluster.NodeInfo(address=studio_cluster.LOCAL, hostname=socket.gethostname())


def _find(nodes, ref):
    return next((n for n in nodes if studio_cluster.matches(ref, n)), None)


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


def merge(own, peers, jobs, waiting=None):
    """The rows of the whole cluster: `jobs` are this machine's (running, running on a peer, Up next), the peers' come from their
    surveys. A job a peer runs for another node is left to its owner's row, and a peer's job that runs here is already in `jobs`."""
    waiting = waiting or {}
    nodes = [own, *(p.info for p in peers)]
    names = _names(nodes)

    def row(job_id, kind, label, status, params, ref, queued_on, here, why, note, peer=None):
        node = _find(nodes, ref) if ref else None
        return Row(job_id, kind, label, status, node.address if node else "", names[node.address] if node else ref, queued_on, here,
                   _pinned(params), why, bool(peer and peer.stale), note)

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
                            False, "", str(job.get("note") or ""), peer))
    return rows
