"""The scheduler's rules (docs/distributed.md, "Scheduler: ownership, leases and fencing"). Pure logic: no threads, no sockets, no
files. Every clock and every peer is injected, so each rule is a window-free test with a fake clock and fake peers.

* `Health`: the failure detector. A node is *suspect* after one failed poll and *down* after `DOWN_AFTER` seconds without an answer.
* `LeaseTable`: the owner's side of a lease. `Lease(job_id, node, epoch, duration_s)` is granted when a job is dispatched and
  renewed by the runner; every grant increments the job's epoch, and any write that carries another epoch is refused (`StaleEpoch`,
  an HTTP 409). The owner measures a lease from the moment it received the renewal.
* `SelfFence`: the runner's side. It measures the lease on its own monotonic clock from the moment it *sent* the renewal that
  extended it, so it always gives up before the owner does. No clocks are compared between machines.
* `decide`: where a waiting job goes: here, a peer, or nowhere yet.
* `decide_gang` and `Gangs`: where a distributed run goes. A run takes `min_members` to `max_members` free machines in one decision,
  or none, and each member slot is a lease. A slot that lapses is freed and refilled from the machines that are free then.
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
JOIN_GRACE = 4  # a member slot gets this many lease durations to be heard from the first time: its machine has a member to start
REMOTE_KINDS = ("pretrain", "compare", "post_round", "restore_answers")  # kinds whose inputs travel in the dispatch (rows) or the version store (checkpoints), so they can run on another machine; docs/distributed.md says why the rest stay
EVENTS_RELAYED = ("metric", "telemetry", "info", "sample", "compare", "compare_run", "moved")  # what a runner passes to the owner; the rest names files on the runner


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

    def held(self):
        """[(lease, seconds since the runner was last heard from)] for every lease held, as this owner sees them."""
        now = self.clock()
        return [(lease, now - heard) for lease, heard in list(self._held.values())]

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
    basis: str = ""  # when a node's measured speed picked it: the sentence saying so (empty: the old rule decided)


def decide(kind, nodes, health, pinned=None, store_ready=True, requirements=None, exclude=(), shape=None):
    """Where a waiting job goes. `nodes` is the cluster as NodeInfo, this machine first (address `studio_cluster.LOCAL`).
    A peer is a candidate only when it is up, the kind can run away from its owner (REMOTE_KINDS), the version store is there to carry
    the result back, and it did not just refuse this job (`exclude`). A pinned job goes to its node or waits: it is never moved to
    a better one. `shape` is the job's design shape (`job_shape`): among the free machines that fit, the one that measured the
    fastest step for it wins (studio_cluster.place_why); with none given, or a machine that has not measured it, the old rule."""
    requirements = requirements or studio_cluster.defaults_for(kind)
    remote_kind = kind in REMOTE_KINDS
    candidates = []
    for node in nodes:
        local = node.address == studio_cluster.LOCAL
        if not local and not (remote_kind and store_ready and health.state(node.address) == UP and node.address not in exclude):
            node = replace(node, online=False)
        candidates.append(node)
    target, basis = studio_cluster.place_why(requirements, candidates, pinned, shape)
    if target is not None:
        return Placement("local", basis=basis) if target.address == studio_cluster.LOCAL else Placement("remote", target.address, basis=basis)
    if pinned:
        return Placement("wait", reason=_why_pinned(pinned, kind, nodes, health, remote_kind, store_ready, exclude))
    return Placement("wait", reason="No machine that fits this job is free yet.")


def job_shape(kind, params):
    """The design shape of a job (`studio_probe.shape_key`, the key nodes report speeds under), or None for a job with no design to time
    or a config that does not validate (placement then falls back to the old rule)."""
    if kind != "pretrain" or not isinstance(params.get("config"), dict):
        return None
    try:
        import studio_memory
        return studio_memory.shape_of(studio_memory.validated(params["config"]))
    except Exception:  # noqa: BLE001 - a job we cannot describe is placed without a speed
        return None


def _why_pinned(pinned, kind, nodes, health, remote_kind, store_ready, exclude):
    found = studio_cluster.pinned_nodes(pinned, nodes)
    if len(found) > 1:
        names = " and ".join(f"{n.hostname} ({n.address})" for n in found)
        return f"It is pinned to {pinned}, a name {len(found)} machines share: {names}. Pin it to one by its address."
    if not found:
        return f"It is pinned to {pinned}, which is not one of this cluster's machines."
    node = found[0]
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


# ---- gang placement ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GangPlacement:
    nodes: tuple = ()  # the addresses that got a slot, in the order given; empty when the run waits
    reason: str = ""  # when waiting: a sentence saying what the run is waiting for


def free_for_gang(nodes, health, taken=(), exclude=()):
    """The addresses a distributed run could use now, in the order given: online, not busy, not held by another run (`taken`) or just
    lost (`exclude`), and a peer must be up by the failure detector. Members are `diloco_member` jobs on the peer's own queue, so
    unlike a `pretrain` job they need no version store."""
    out = []
    for node in nodes:
        local = node.address == studio_cluster.LOCAL
        if node.free and node.address not in taken and node.address not in exclude and (local or health.state(node.address) == UP):
            out.append(node.address)
    return out


def decide_gang(nodes, health, min_members, max_members, taken=(), exclude=()):
    """A GangPlacement: the first `max_members` free machines when at least `min_members` are free, else none. All or nothing, so two
    runs can never each hold part of what the other needs."""
    free = free_for_gang(nodes, health, taken, exclude)
    if len(free) < min_members:
        listed = ", ".join(n.hostname or n.address for n in nodes if n.address in free) or "none"
        return GangPlacement((), f"It needs {min_members} machine{'s' if min_members != 1 else ''} free at once; {len(free)} "
                                 f"of the {len(nodes)} chosen {'is' if len(free) == 1 else 'are'} free ({listed}).")
    return GangPlacement(tuple(free[:max_members]))


@dataclass
class _Slot:
    lease: Lease
    last: float  # when the owner last heard the member (or granted the slot)
    heard: bool = False  # heard at least once since the grant: a slot that never was gets JOIN_GRACE durations


class Gangs:
    """The member slots of the distributed runs this node owns. A slot is a lease on one machine for one run; a run's slots are granted
    together by `place` or not at all, and no machine holds slots of two runs. `heard` renews a slot; `lapsed` lists the slots whose
    member has been silent for longer than its lease, which the owner then frees (`drop`) and offers to another free machine (`fill`),
    which joins the run at the round after. Epochs are per slot and only go up."""

    def __init__(self, clock=time.monotonic, lease_s=LEASE_S):
        self.clock, self.lease_s = clock, lease_s
        self._slots = {}  # run id -> {address: _Slot}
        self._epochs = {}  # (run id, address) -> the highest epoch granted
        self._lost = {}  # (run id, address) -> when it may be offered the run again

    def runs(self):
        return list(self._slots)

    def held(self, run):
        """{address: Lease} for the run's slots."""
        return {a: slot.lease for a, slot in self._slots.get(run, {}).items()}

    def taken(self, run=None):
        """The addresses held by runs other than `run`."""
        return {a for r, slots in self._slots.items() if r != run for a in slots}

    def _grant(self, run, address):
        epoch = self._epochs.get((run, address), 0) + 1
        self._epochs[(run, address)] = epoch
        lease = Lease(f"{run}@{address}", address, epoch, self.lease_s)
        self._slots.setdefault(run, {})[address] = _Slot(lease, self.clock())
        return lease

    def place(self, run, nodes, health, min_members, max_members):
        """Take the run's slots in one decision. A run that holds slots already gets them back unchanged, so asking again is harmless."""
        if self._slots.get(run):
            return GangPlacement(tuple(self._slots[run]))
        placement = decide_gang(nodes, health, min_members, max_members, taken=self.taken(), exclude=self._excluded(run))
        for address in placement.nodes:
            self._grant(run, address)
        return placement

    def fill(self, run, nodes, health, max_members):
        """Offer the run's open slots (up to `max_members` in all) to the machines that are free now. Returns the addresses that got one."""
        slots = self._slots.get(run)
        if slots is None:
            return []
        free = free_for_gang(nodes, health, taken=self.taken() | set(slots), exclude=self._excluded(run))
        return [address for address in free[:max(0, max_members - len(slots))] if self._grant(run, address)]

    def heard(self, run, address):
        slot = self._slots.get(run, {}).get(address)
        if slot is not None:
            slot.last, slot.heard = self.clock(), True

    def lapsed(self):
        """[(run id, address)]: slots whose member has been silent for longer than its lease. This machine's own slot never lapses."""
        now = self.clock()
        return [(run, a) for run, slots in self._slots.items() for a, slot in slots.items()
                if a != studio_cluster.LOCAL and now - slot.last > slot.lease.duration_s * (1 if slot.heard else JOIN_GRACE)]

    def drop(self, run, address, ban=True):
        """Free a slot. The machine is not offered this run again for two leases, so one that is merely slow is not taken straight back."""
        if self._slots.get(run, {}).pop(address, None) is not None:
            if ban:
                self._lost[(run, address)] = self.clock() + 2 * self.lease_s

    def end(self, run):
        """The run is over: free all its slots."""
        self._slots.pop(run, None)
        for key in [k for k in self._lost if k[0] == run]:
            del self._lost[key]

    def _excluded(self, run):
        now = self.clock()
        return {a for (r, a), until in self._lost.items() if r == run and until > now}


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
    return out, f"Interrupted on {node} at step {done}. It continues from that checkpoint.", False


def resume_here(notes, address):
    """{"epoch", "steps", "base"} when the job's last checkpoint was written on the node at `address`, which still holds the full one
    (`last.npz` of that attempt's folder: optimizer, sampler and RNG), else None. `steps` is the length of the attempt that wrote it
    and `base` the steps done before that attempt."""
    checkpoint = notes.get("checkpoint") or {}
    if checkpoint.get("node") != address or "epoch" not in checkpoint:
        return None
    base = int(checkpoint.get("base") or 0)
    steps = int(notes.get("steps") or 0) - base
    return {"epoch": int(checkpoint["epoch"]), "steps": steps, "base": base} if steps > 0 else None


def interrupt(job):
    """Put a job whose runner is gone back on the owner: waiting for a new node, resuming from its last checkpoint (`resume_params`), or
    done when that checkpoint was the last step. Sets `params`, `node`, `status` and `note`; returns (note, finished). The owner holds
    its queue lock."""
    node = job.node
    params, note, finished = resume_params(job.params, node)
    job.params, job.node = params, ""
    if finished:
        checkpoint = attempt_state(job.params)["checkpoint"]
        job.status = "done"
        job.params["remote_result"] = {"key": checkpoint["key"], "node": node, "step": checkpoint["step"]}
    else:
        job.status = "waiting"
        job.note = f"{job.note} {note}".strip()
    return note, finished


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
