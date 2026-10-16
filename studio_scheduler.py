"""The scheduler's rules (docs/distributed.md, "Scheduler: ownership, leases and fencing"). Pure logic: no threads, no sockets, no
files. Every clock and every peer is injected, so each rule is a window-free test with a fake clock and fake peers.

* `Health`: the failure detector. A node is *suspect* after one failed poll and *down* after `DOWN_AFTER` seconds without an answer.
* `LeaseTable`: the owner's side of a lease. `Lease(job_id, node, epoch, duration_s)` is granted when a job is dispatched and
  renewed by the runner; every grant increments the job's epoch, and any write that carries another epoch is refused (`StaleEpoch`,
  an HTTP 409). The owner measures a lease from the moment it received the renewal.
* `SelfFence`: the runner's side. It measures the lease on its own monotonic clock from the moment it *sent* the renewal that
  extended it, so it always gives up before the owner does. No clocks are compared between machines.
* `decide`: where a waiting job goes: here, a peer, or nowhere yet.
* `Owner` and `Guest`: the two ends of a dispatch, as state machines that call `client_for(address)` (a `studio_peer.PeerClient`).
  `Owner.tick()` says which leases lapsed; the I/O edge (studio_remote.py) acts on it.
* `resume_params`: the params a job has when it is placed again after its node was lost.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import studio_cluster
from studio_peer import PeerError

DOWN_AFTER = 30.  # seconds without an answer before a node counts as down
LEASE_S = 30.  # a lease's duration; the runner renews it every third of this
DISPATCH_TIMEOUT = 120.  # the job spec carries its training data, so handing it over can take a while
REFUSAL_COOLDOWN = 30.  # seconds before a job is offered again to a node that refused it
UP, SUSPECT, DOWN = "up", "suspect", "down"
REMOTE_KINDS = ("pretrain",)  # kinds whose inputs are the dataset and a published version, so they can run on another machine
EVENTS_RELAYED = ("metric", "telemetry", "info", "sample")  # what a runner passes to the owner; the rest names files on the runner


class StaleEpoch(RuntimeError):
    """A write that carries an epoch other than the job's current one. The API answers 409 and the writer stops its copy."""

    def __init__(self, job_id, epoch, current=None):
        self.job_id, self.epoch, self.current = job_id, epoch, current
        super().__init__(f"Job {job_id} is no longer yours to run: the owner has moved on from epoch {epoch}."
                         if current is None else f"Job {job_id} is at epoch {current}, not {epoch}: it has been placed again.")


class Refused(RuntimeError):
    """A node would not take a job. The message is already a sentence for people."""


# ---- failure detector ----------------------------------------------------------------------------------------------

class Health:
    """Per-node answer history. `observe(address, ok)` records one poll; `state(address)` is UP after a poll that answered, SUSPECT
    after one that did not, and DOWN once `down_after` seconds have passed since the last answer (or since the first failure,
    for a node that never answered). Only DOWN moves work."""

    def __init__(self, clock=time.monotonic, down_after=DOWN_AFTER):
        self.clock, self.down_after = clock, down_after
        self._seen = {}  # address -> (last poll answered, time of the last answer or of the first failure)

    def observe(self, address, ok):
        now, entry = self.clock(), self._seen.get(address)
        if ok:
            self._seen[address] = (True, now)
        elif entry is None:
            self._seen[address] = (False, now)
        elif entry[0]:
            self._seen[address] = (False, entry[1])

    def state(self, address):
        entry = self._seen.get(address)
        if entry is None:
            return SUSPECT  # never heard from: not known to be up
        answered, since = entry
        if answered:
            return UP
        return DOWN if self.clock() - since >= self.down_after else SUSPECT


# ---- leases --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Lease:
    job_id: str
    node: str
    epoch: int
    duration_s: float = LEASE_S

    @property
    def renew_every(self):
        return self.duration_s / 3

    def as_dict(self):
        return {"job_id": self.job_id, "node": self.node, "epoch": self.epoch, "duration_s": self.duration_s}

    @classmethod
    def from_dict(cls, value):
        return cls(str(value["job_id"]), str(value["node"]), int(value["epoch"]), float(value["duration_s"]))


class LeaseTable:
    """The leases an owner has granted. Epochs only go up, per job, even across releases; `seed` restores them after a restart so an
    old runner's epoch can never match a new grant."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._held = {}  # job id -> [lease, time the owner last heard from the runner]
        self._epochs = {}  # job id -> the highest epoch ever granted

    def seed(self, job_id, epoch):
        self._epochs[job_id] = max(self._epochs.get(job_id, 0), int(epoch))

    def epoch_of(self, job_id):
        return self._epochs.get(job_id, 0)

    def grant(self, job_id, node, duration_s=LEASE_S):
        """A lease at the job's next epoch. It replaces any lease still held for the job: the earlier one is fenced from now on."""
        lease = Lease(job_id, node, self._epochs.get(job_id, 0) + 1, duration_s)
        self._epochs[job_id] = lease.epoch
        self._held[job_id] = [lease, self.clock()]
        return lease

    def get(self, job_id):
        held = self._held.get(job_id)
        return held[0] if held else None

    def check(self, job_id, epoch):
        """The current lease, or StaleEpoch when `epoch` is not the epoch it was granted at (or no lease is held)."""
        held = self._held.get(job_id)
        if held is None or held[0].epoch != epoch:
            raise StaleEpoch(job_id, epoch, held[0].epoch if held else None)
        return held[0]

    def renew(self, job_id, epoch):
        lease = self.check(job_id, epoch)
        self._held[job_id][1] = self.clock()
        return lease

    touch = renew

    def release(self, job_id):
        self._held.pop(job_id, None)

    def lapsed(self):
        """Jobs whose runner has been silent for longer than the lease: the runner has certainly stopped, or is about to."""
        return [j for j, (lease, heard) in self._held.items() if self.clock() - heard > lease.duration_s]

    def expired(self):
        """Jobs silent for two lease durations: the runner's self-fence has certainly fired, so the job may be placed again."""
        return [j for j, (lease, heard) in self._held.items() if self.clock() - heard > 2 * lease.duration_s]


class SelfFence:
    """The runner's view of its lease. It starts when the dispatch arrives, and each acknowledged renewal moves the deadline to
    (the time the renewal was *sent*) + duration, on this machine's monotonic clock."""

    def __init__(self, lease, clock=time.monotonic):
        self.lease, self.clock = lease, clock
        start = clock()
        self._deadline, self._last_sent = start + lease.duration_s, start

    def due(self):
        return self.clock() - self._last_sent >= self.lease.renew_every

    def sending(self):
        """Call just before a renewal goes out; pass the result to `acked` if the owner answered."""
        self._last_sent = self.clock()
        return self._last_sent

    def acked(self, sent_at):
        self._deadline = max(self._deadline, sent_at + self.lease.duration_s)

    def remaining(self):
        return self._deadline - self.clock()

    def expired(self):
        return self.remaining() <= 0


# ---- placement -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Placement:
    where: str  # "local" | "remote" | "wait"
    node: str = ""  # the peer's address when remote
    reason: str = ""  # when waiting: a sentence saying what the job is waiting for


def decide(kind, nodes, health, pinned=None, store_ready=True, requirements=None, exclude=()):
    """Where a waiting job goes. `nodes` is the cluster as NodeInfo, this machine first (address `studio_cluster.LOCAL`).
    A peer is a candidate only when it is up, the kind can run away from its owner (REMOTE_KINDS), the version store is there to carry
    the result back, and it did not just refuse this job (`exclude`). A pinned job goes to its node or waits: it is never moved to
    a better one."""
    requirements = requirements or studio_cluster.defaults_for(kind)
    remote_kind = kind in REMOTE_KINDS
    candidates = []
    for node in nodes:
        local = node.address == studio_cluster.LOCAL
        if not local and not (remote_kind and store_ready and health.state(node.address) == UP and node.address not in exclude):
            node = replace(node, online=False)
        candidates.append(node)
    target = studio_cluster.place(requirements, candidates, pinned)
    if target is not None:
        return Placement("local") if target.address == studio_cluster.LOCAL else Placement("remote", target.address)
    if pinned:
        return Placement("wait", reason=_why_pinned(pinned, kind, nodes, health, remote_kind, store_ready, exclude))
    return Placement("wait", reason="No machine that fits this job is free yet.")


def _why_pinned(pinned, kind, nodes, health, remote_kind, store_ready, exclude):
    node = next((n for n in nodes if studio_cluster.matches(pinned, n)), None)
    if node is None:
        return f"It is pinned to {pinned}, which is not one of this cluster's machines."
    if node.address == studio_cluster.LOCAL:
        return "It is pinned to this machine, which is busy."
    if not remote_kind:
        return f"A {kind} job only runs on the machine that queued it, so it cannot go to {pinned}."
    if not store_ready:
        return f"It is pinned to {pinned}, and a job that leaves this machine needs the version store set up (docs/cluster.md)."
    if health.state(node.address) != UP:
        return f"It is pinned to {pinned}, which is not answering."
    if node.address in exclude:
        return f"It is pinned to {pinned}, which just refused it."
    return f"It is pinned to {pinned}, which is busy or does not fit it."


# ---- the two ends of a dispatch ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Dispatch:
    lease: Lease
    confirmed: bool  # False when the peer did not answer: it may have started the job anyway, so the lease stays until it lapses
    detail: str = ""


class Owner:
    """The node a job was queued on, as far as leases go: grants them, answers renewals, and says which have lapsed. `client_for(address,
    **options)` returns a studio_peer.PeerClient (or a fake)."""

    def __init__(self, client_for, clock=time.monotonic, lease_s=LEASE_S):
        self.client_for, self.clock, self.lease_s = client_for, clock, lease_s
        self.leases = LeaseTable(clock)
        self._lapsed = set()  # jobs already reported as lapsed
        self._refused = {}  # (job id, address) -> when the node may be offered the job again

    def dispatch(self, job_id, address, body):
        """Grant a lease and hand the job to `address` (`POST /cluster/jobs`). A refusal (any 4xx) revokes the lease and raises Refused
        with the peer's sentence; an answer that never came leaves the lease in place, unconfirmed, because the peer may have started."""
        lease = self.leases.grant(job_id, address, self.lease_s)
        try:
            self.client_for(address).post_json("cluster/jobs", {**body, "lease": lease.as_dict()}, timeout=DISPATCH_TIMEOUT)
        except PeerError as exc:
            if exc.status is not None and 400 <= exc.status < 500:
                self.leases.release(job_id)
                self._refused[(job_id, address)] = self.clock() + REFUSAL_COOLDOWN
                raise Refused(exc.detail or str(exc)) from None
            self.leases.touch(job_id, lease.epoch)
            return Dispatch(lease, False, exc.detail or str(exc))
        self.leases.touch(job_id, lease.epoch)  # counted from here: the runner's count began when the request arrived, before this
        return Dispatch(lease, True)

    def refused_by(self, job_id):
        now = self.clock()
        return {address for (job, address), until in self._refused.items() if job == job_id and until > now}

    def renew(self, job_id, epoch):
        """(lease, recovered): StaleEpoch for any epoch but the current. `recovered` is True when the lease had lapsed."""
        recovered = job_id in self._lapsed
        lease = self.leases.renew(job_id, epoch)
        self._lapsed.discard(job_id)
        return lease, recovered

    def check(self, job_id, epoch):
        return self.leases.check(job_id, epoch)

    def end(self, job_id, epoch=None):
        """The job is over (finished, failed or cancelled): drop its lease. With `epoch`, only if that is the current one."""
        if epoch is not None:
            self.leases.check(job_id, epoch)
        self.leases.release(job_id)
        self._lapsed.discard(job_id)

    def cancel(self, job_id):
        """Tell the runner to stop the job (best effort) and drop its lease. A runner that is never told finds out at its next renewal."""
        lease = self.leases.get(job_id)
        self.end(job_id)
        if lease is not None:
            try:
                self.client_for(lease.node, retries=0, timeout=5.).post_json(f"cluster/jobs/{job_id}/stop", {"epoch": lease.epoch})
            except PeerError:
                pass

    def tick(self):
        """[(job id, "lapsed" | "reassign")]. "lapsed" is reported once, when a lease has not been renewed for its duration;
        "reassign" once the owner has waited a second duration, which releases the lease so the old runner is fenced."""
        out = []
        expired = set(self.leases.expired())
        for job_id in self.leases.lapsed():
            if job_id in expired:
                continue
            if job_id not in self._lapsed:
                self._lapsed.add(job_id)
                out.append((job_id, "lapsed"))
        for job_id in sorted(expired):
            self.leases.release(job_id)
            self._lapsed.discard(job_id)
            out.append((job_id, "reassign"))
        return out


class Guest:
    """A runner's end of one lease: renews it on schedule and tells the edge when to stop the job. `owner` is a client for the owner
    (retries off: a renewal that fails is simply sent again a third of a lease later)."""

    def __init__(self, lease, owner, clock=time.monotonic):
        self.lease, self.owner, self.fence = lease, owner, SelfFence(lease, clock)

    @property
    def timeout(self):
        return self.lease.renew_every

    def renew(self, progress, acked=None):
        """Send a renewal when one is due. `progress()` builds the body's progress (called only when sending) and `acked()` is called once
        the owner has taken it. Returns "stale" when the owner refused the epoch, "fenced" when the lease ran out before it could be
        renewed, else None."""
        if self.fence.expired():
            return "fenced"
        if not self.fence.due():
            return None
        sent = self.fence.sending()
        try:
            self.owner.post_json(f"cluster/jobs/{self.lease.job_id}/renew", {"epoch": self.lease.epoch, **progress()}, timeout=self.timeout)
        except PeerError as exc:
            if exc.status == 409:
                return "stale"
            return "fenced" if self.fence.expired() else None
        self.fence.acked(sent)
        if acked:
            acked()
        return None

    def finish(self, body):
        """Report the end of the job. "ok", "stale" (the owner has moved on) or "retry" (not delivered; try again until the fence fires)."""
        try:
            self.owner.post_json(f"cluster/jobs/{self.lease.job_id}/finished", {"epoch": self.lease.epoch, **body}, timeout=self.timeout * 3)
        except PeerError as exc:
            return "stale" if exc.status == 409 else "retry"
        return "ok"


# ---- placing a job again -------------------------------------------------------------------------------------------

def attempt_state(params):
    """The scheduler's notes in a job's params: the run's planned steps, the steps done before this attempt, the last checkpoint."""
    return params.get("cluster") or {}


def resume_params(params, node):
    """(params, note, finished) for a job whose runner was lost. A published checkpoint is the weights only, so the job continues from
    it with a fresh optimizer and the steps that are left; with none, it starts again. `finished` is True when the checkpoint is
    the last step, so there is nothing left to run."""
    notes = attempt_state(params)
    checkpoint = notes.get("checkpoint")
    out = {k: v for k, v in params.items() if k != "fetch"}
    if not checkpoint:
        out["cluster"] = {**notes, "offset": 0}
        return out, f"It was interrupted on {node} before its first checkpoint, so it starts again from the beginning.", False
    total, done = int(notes.get("steps") or 0), int(checkpoint["step"])
    if total and done >= total:
        return out, "", True
    config = dict(out.get("config") or {})
    if total:
        config["steps"] = total - done
        out["config"] = config
    out.update(mode="catch_up", fetch={"init_from": checkpoint["key"]})
    out.pop("init_from", None)
    out["cluster"] = {**notes, "offset": done}
    return out, (f"Interrupted on {node} at step {done}. It continues from that checkpoint, and the optimizer starts again from zero "
                 "because only the weights are published."), False


def relayed(event, offset=0):
    """An event a runner sent, as the owner re-emits it: None for kinds that name files on the runner; steps shifted by `offset`
    (the steps an earlier attempt already did) so progress does not start over after a resume."""
    if not isinstance(event, dict) or event.get("type") not in EVENTS_RELAYED:
        return None
    out = dict(event)
    for key in ("step", "total"):
        if isinstance(out.get(key), int) and not isinstance(out.get(key), bool):
            out[key] += offset
    return out
