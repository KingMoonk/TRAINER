"""The I/O edge of the scheduler (studio_scheduler.py has the rules): one `Cluster` per queue Runner.

As the **owner** of the jobs queued here it polls the peers' `GET /node`, places each waiting job with `decide`, hands a job that
belongs on a peer over (`POST /cluster/jobs` on the peer, the job spec carrying its training data and a lease), relays the progress
the peer reports into this node's event stream, and places a job again when its lease lapses.

As a **runner** of a job another node owns it checks the lease and the version store, pulls the version the job starts from, puts the
job in this node's own queue (it runs in its own worker process, studio_worker, like any other), renews the lease on the owner every
third of its duration with the job's progress, publishes the job's checkpoints to the version store, and reports the end. It stops the
job itself when the lease cannot be renewed in time (self-fencing) or the owner answers 409 (the job has moved on).

The version store is the shared folder the workspace names (`studioConfig.store.folder`), else a shared folder (`TRAINER_STORE_FOLDER`)
or an S3-compatible bucket (`TRAINER_STORE_*`, docs/cluster.md). A job that leaves this machine needs it: the checkpoints and the
finished version travel through it, so without it nothing is dispatched.
`TRAINER_NODE_ADDRESS` is how peers reach this node when the address they see the request come from is not it (behind a proxy).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import shutil
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import studio_cluster
import studio_cluster_watch
import studio_publish
import studio_facts
import studio_failover
import studio_scheduler as sched
import studio_state
from studio_peer import PeerClient

POLL_EVERY = studio_failover.POLL_EVERY  # seconds between polls of the peers
POLL_BACKOFF_MAX = studio_failover.POLL_BACKOFF_MAX  # the longest wait between polls of a peer that does not answer: a peer that answers again is seen within this
TICK_EVERY = 1.  # seconds between lease checks and renewals
RUN_EVENT_EVERY = .5  # the least seconds between two `run` events of one run (a coordinator changes at every member report)
FACTS_EVENT_EVERY = 1.  # the least seconds between two `facts` events
QUIET_KINDS = ("heartbeat",)  # observation kinds whose change is no news to a page
RUN_KINDS = ("run", "timeline", "run_events", "progress", "shards", "held", "claim")  # observation kinds keyed by run id
EVENT_BACKLOG = 200  # events a runner keeps for its next renewal
RESUME_EVENTS = ("resumed", "resume_refused")  # which way a job placed again on the node that wrote its checkpoint went (`_own_checkpoint`)
LOCAL_ONLY = ("version", "user_text", "previous_best", "output_dir", "result", "max_batch_size", "pinned", "remote_result")  # not for the runner
TERMINAL = ("done", "failed", "stopped")
SWEEP_RETRY_S = 60.  # seconds before the owner tries again to delete checkpoints the store would not let go of
USAGE_RETRY_S = 30.  # seconds before a checkpoint the store could not check (its size would not read) is offered again
ROUND_KINDS = ("post_round", "restore_answers")  # their finished weights are the result's `checkpoint`, not the newest file in the folder
RESULT_KINDS = ("compare",) + ROUND_KINDS  # kinds whose finished report carries `result`, which the app keeps in the job's params
PULLED_KINDS = ("pretrain",) + ROUND_KINDS  # kinds whose finished weights the owner pulls from the store ("landing"), so its version is local
RESULT_MAX_BYTES = 4 << 20  # the largest `result` a runner reports
LAND_TRIES = 3  # attempts to bring a finished job's weights from the store before the job is failed
LAND_RETRY_S = 30.  # seconds between those attempts
_log = logging.getLogger(__name__)


def client_for(address, **options):
    return PeerClient(address, **options)


def checkpoint_step(path):
    """The step a checkpoint was saved at, 0 when it cannot be read."""
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as archive:
            return int(json.loads(str(archive["metadata"].item())).get("step") or 0)
    except (OSError, ValueError, KeyError):
        return 0


class GuestRun:
    """One job this node runs for another. `id` is the owner's job id; the job in this node's queue is `local_id`, which carries the
    epoch so a job that comes back to this node starts in a folder of its own."""

    def __init__(self, lease, spec, fetch, owner_address, guest):
        self.lease, self.spec, self.fetch, self.owner_address, self.guest = lease, spec, fetch, owner_address, guest
        self.id, self.local_id = lease.job_id, f"{lease.job_id}-e{lease.epoch}"
        self.lock = threading.Lock()
        self.events, self.sent = [], 0  # events not yet acknowledged by the owner
        self.checkpoint, self.published_mtime, self.publishing = None, 0., False
        self.refused_mtime, self.refusal = 0., ""  # the newest checkpoint the store refused (over its limit), and why: it is not offered again
        self.retry_at = 0.  # the clock time a checkpoint the store could not check may be offered again
        self.stored = []  # the keys of the checkpoints this run put in the store (not ones already there), oldest first: the only ones it deletes
        self.enqueued = self.cancelled = self.ended = self.reported = False
        self.stopped_by = None  # "stale" | "fenced" | "cancelled" | "replaced": the owner no longer wants this copy
        self.problem = ""
        self.shift = 0  # added to the steps this run reports: the owner's offset counts the published checkpoint it handed over, and a
        #                 run that continues from its own, further-on checkpoint (`Cluster._own_checkpoint`) must read the same progress

    def keep(self, event):
        with self.lock:
            self.events.append(event)
            del self.events[:-EVENT_BACKLOG]


class Coalescer:
    """Hands `send(key, payload)` on at most once per `every` seconds per key: the first at once, the rest folded (`fold`) into one
    trailing call. Pages refetch on these events, so a burst of changes costs one refetch, and the last change is never held back."""

    def __init__(self, send, every, fold=lambda old, new: new, clock=time.monotonic, later=None):
        self.send, self.every, self.fold, self.clock = send, every, fold, clock
        self.later = later or (lambda delay, call: _timer(delay, call))
        self._last, self._held = {}, {}  # key -> clock time of its last send; key -> payload folded while it waits
        self._lock = threading.Lock()

    def __call__(self, key, payload):
        with self._lock:
            if key in self._held:
                self._held[key] = self.fold(self._held[key], payload)
                return
            wait = self._last.get(key, -math.inf) + self.every - self.clock()
            if wait > 0:
                self._held[key] = payload
                self.later(wait, lambda: self._flush(key))
                return
            self._last[key] = self.clock()
        self.send(key, payload)

    def _flush(self, key):
        with self._lock:
            payload = self._held.pop(key)
            self._last[key] = self.clock()
        self.send(key, payload)


def _timer(delay, call):
    timer = threading.Timer(delay, call)
    timer.daemon = True
    timer.start()


def _shown(info):
    """What a machine list shows of a peer, so a poll that changed none of it sends no `nodes` event."""
    return None if info is None else (info.online, info.busy, info.hostname, info.version, info.device, info.device_name)


def _fold_facts(old, new):
    return {name: sorted({*old[name], *new[name]}) for name in old}


class Cluster:
    """Everything a Runner needs to take part in a cluster. Inert (no polling, no threads of consequence) while no peers are listed in
    the workspace config. `client_for(address, **options)` makes the PeerClient; tests pass fakes for it, for `store`, `own` and `clock`."""

    def __init__(self, runner, client_for=client_for, store=None, own=None, clock=time.monotonic, address=None, environ=None,
                 lease_s=sched.LEASE_S, threads=True, hostname=None, local_runs=None, coordinators=None):
        self.runner, self.client_for, self.clock, self.lease_s, self.threads = runner, client_for, clock, lease_s, threads
        self.environ = os.environ if environ is None else environ
        self.hostname = hostname or socket.gethostname()  # this node's origin in the observation store
        self._local_runs = local_runs  # tests give each in-process node its own; None: this process's runs (studio_distributed)
        self.address = address if address is not None else (self.environ.get("TRAINER_NODE_ADDRESS") or "").strip() or None
        self.api_port, self._tailnet = studio_cluster.DEFAULT_PORT, lambda: ""  # serve_at: how this node's API is reached
        self._store_factory = store or self.workspace_store
        self._own_factory = own or (lambda: studio_cluster.own_info(version=_version()))
        self._own = None
        self.health = sched.Health(clock)
        self.owner = sched.Owner(client_for, clock, lease_s)
        self.gangs = sched.Gangs(clock, lease_s)  # the member slots of the distributed runs queued here
        self.guests = {}  # owner's job id -> GuestRun
        self.waiting = {}  # job id -> why it is waiting
        self.basis = {}  # job id -> why the machine it goes to was picked (its measured speed), until it is dispatched; none when the rule decided
        self._landing, self._tries, self._retry_at = set(), {}, {}  # finished jobs being pulled now; failed pulls so far; clock time of the next try (`_land`)
        self._removed = {}  # job id -> checkpoint keys of a job taken out of the queue, still to be deleted from the store
        self._results = set()  # result keys of jobs taken out of the queue: a result can be the bytes of a checkpoint, and stays
        self._begun = set()  # keys a delete was started for: a half-finished one (its listing gone) is finished, no other unlisted key is
        self._keep_lock = threading.Lock()  # guards _removed and _results only; a leaf lock (held with nothing else taken), so the queue may call in under its own
        self._delete_lock = threading.Lock()  # a sweep checks a key is free and deletes it under this; `finished` records a result under it
        self._sweeping, self._sweep_at = False, 0.  # a sweep of checkpoints is running; the clock time the next may start
        self._known = {}  # peer address -> its last NodeInfo
        self._slow = set()  # peer addresses whose last poll timed out: busy and answering slowly, not refused or unknown
        self._silent = {}  # peer address -> (polls in a row it did not answer, clock time of the next poll): it is left alone until then; 0: slow once
        self._answered, self._probed = {}, {}  # peer address -> clock time of its last answer; of the last time it was asked
        self._peers, self._peers_mtime, self.config_problem = [], None, ""
        self._store, self._store_problem, self._store_mtime = None, "", object()  # object(): not yet read, equal to no mtime
        self.state = studio_state.Replica(self.hostname, studio_state.next_incarnation(runner.root / "data" / studio_state.INCARNATION_FILE),
                                          clock)  # the observation store; `publish_state` writes this node's facts into it
        self.beat, self.events, self.members = 0, [], {}  # heartbeat count; recent cluster events; run id -> this node's member report
        self.replicas, self.claims = {}, {}  # run id -> the replica this node's member keeps; run id -> the claim of the coordinator run here
        # the coordinators this node serves: the process's (studio_diloco.COORDINATORS) unless a test runs several nodes in one process
        import studio_diloco
        self.coordinators = studio_diloco.COORDINATORS if coordinators is None else coordinators
        self.failover = studio_failover.Failover(studio_failover.ClusterHost(self))
        self._names = {}  # peer address -> the hostname it last answered with
        self.polled = threading.Event()
        self._threads = []
        self._lock = threading.RLock()
        for job in runner.jobs:  # an old runner's epoch must never match a new grant
            self.owner.leases.seed(job.id, job.epoch)
        runner.subscribe(self._on_event)
        # pages refresh on these instead of polling: `run` when a coordinator here changes, `facts` when the observation store does
        self._run_events = Coalescer(lambda run, payload: runner._emit("run", payload), RUN_EVENT_EVERY)
        self._facts_events = Coalescer(lambda _, payload: runner._emit("facts", payload), FACTS_EVENT_EVERY, _fold_facts)
        self._unwatch = [studio_diloco.watch(self._coordinator_changed), self.state.watch(self._facts_changed)]

    # ---- configuration ----------------------------------------------------------------------------------------------
    def peers(self):
        """The peers listed in the workspace config (re-read when the file changes)."""
        path = self.runner.root / "data" / "workspace.json"
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            return []
        if mtime != self._peers_mtime:
            self._peers_mtime, self.config_problem = mtime, ""
            try:
                self._peers = studio_cluster.configured_nodes(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                self._peers, self.config_problem = [], str(exc)
        return self._peers

    def workspace_store(self):
        """The version store the workspace names (`studioConfig.store.folder`), else the one the environment describes."""
        return studio_publish.configured_store(self._studio_config(), self.environ)

    def store(self):
        """The version store, or None (see `store_problem`). It is looked up again when the workspace file changes, as the peers are."""
        mtime = self._workspace_mtime()
        if mtime != self._store_mtime:
            self._store_mtime, self._store_problem = mtime, ""
            try:
                self._store = self._store_factory()
            except ValueError as exc:
                self._store, self._store_problem = None, str(exc)
            if self._store is None and not self._store_problem:
                self._store_problem = studio_publish.NO_STORE
        return self._store

    def _workspace_mtime(self):
        try:
            return (self.runner.root / "data" / "workspace.json").stat().st_mtime_ns
        except OSError:
            return None

    def _studio_config(self):
        try:
            config = json.loads((self.runner.root / "data" / "workspace.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return config.get("studioConfig") or {} if isinstance(config, dict) else {}

    @property
    def store_problem(self):
        self.store()
        return self._store_problem

    def own(self):
        if self._own is None:
            self._own = self._own_factory()
        return self._own

    # ---- threads ----------------------------------------------------------------------------------------------------
    def start(self):
        if self._threads or not self.threads:
            return
        self.publish_state()  # promptly, so this incarnation replaces the old one's entries in the other replicas
        for name, work, every in (("poll", self.poll_once, POLL_EVERY), ("tick", self.tick, TICK_EVERY)):
            thread = threading.Thread(target=self._every, args=(work, every), daemon=True, name=f"cluster {name}")
            thread.start()
            self._threads.append(thread)

    def _every(self, work, seconds):
        while not self.runner.closed.is_set():
            try:
                work()
            except Exception:  # noqa: BLE001 - a bad poll or tick must not end the loop
                pass
            self.runner.closed.wait(seconds)

    def close(self):
        """Tell the peers running this node's jobs to stop them (best effort) and put those jobs back to waiting for the next start."""
        away = self._remote_jobs()
        for job in away:
            self.owner.cancel(job.id)
        if away:
            with self.runner.lock:
                for job in away:
                    sched.interrupt(job)  # (steps, offset and the key to start from stay together, as when a lease lapses)
                self.runner._save()
        for run in list(self.guests.values()):
            run.ended = True
        for stop in self._unwatch:
            stop()

    def _coordinator_changed(self, coordinator):
        run = next((key for key, held in list(self.coordinators.items()) if held is coordinator), None)
        if run is not None:  # another node's coordinator in this process (tests run several) is that node's news
            self._run_events(run, {"run": run})

    def _facts_changed(self, changes):
        changes = [(origin, kind, key) for origin, kind, key in changes if kind not in QUIET_KINDS]
        if changes:
            self._facts_events("facts", {"kinds": sorted({kind for _, kind, _ in changes}),
                                         "runs": sorted({key for _, kind, key in changes if kind in RUN_KINDS}),
                                         "origins": sorted({origin for origin, _, _ in changes})})

    # ---- the cluster as this node sees it ---------------------------------------------------------------------------
    def publish_state(self):
        """Write this node's facts into its replica (studio_facts.publish): at every poll, which is also the heartbeat."""
        studio_facts.publish(self)

    def busy(self):
        return self.runner.running is not None or bool(self.runner.blocked)

    def local_runs(self):
        """[(run, coordinator, job)] of the runs this node coordinates or queued."""
        if self._local_runs is not None:
            return self._local_runs()
        import studio_distributed
        with self.runner.lock:
            jobs = list(self.runner.jobs)
        return studio_distributed.local_runs(jobs, create=True, coordinators=self.coordinators)

    def serve_at(self, port, tailnet=lambda: ""):
        """Say how this node's API is reached (studio_api.__main__): on loopback at `port` by the workers here, and by the peers at
        TRAINER_NODE_ADDRESS, else https://<tailnet()>, where `tailscale serve` listens (tailnet() is this node's MagicDNS name or '')."""
        self.api_port, self._tailnet = int(port), tailnet

    def url_for_peers(self):
        """The address the peers reach this node's API at (serve_at), or None when it is not known."""
        if self.address:
            return self.address
        name = self._tailnet()
        return f"https://{name}" if name else None

    def address_of(self, origin):
        """The listed address of the peer that is `origin`, or None when this node's list has none by that name."""
        return next((a for a, name in self._names.items() if name == origin and a in self.peers()), None)

    def is_down(self, origin):
        """True for an origin that is not answering: its last poll failed, or its heartbeat is older than DOWN_AFTER."""
        if origin == self.hostname:
            return False
        address = self.address_of(origin)
        known = self._known.get(address) if address else None
        if known is not None and not known.online:
            return True
        age = studio_facts.heartbeat_age(self, origin)
        return age is None or age > studio_facts.DOWN_AFTER

    def answers_slowly(self, origin):
        """True for a down origin whose last poll timed out: it is there but slow, rather than off or refusing."""
        return self.is_down(origin) and self.address_of(origin) in self._slow

    def contact(self, address):
        """(seconds since the peer at `address` was last heard from, seconds until it is asked again): either is None when not known.
        Heard from is its last answer to a poll or its newest heartbeat in this node's copy of the store, whichever is later."""
        now, heard = self.clock(), []
        if address in self._answered:
            heard.append(now - self._answered[address])
        name = self._names.get(address)
        age = studio_facts.heartbeat_age(self, name) if name else None
        if age is not None:
            heard.append(age)
        if address in self._silent:
            check = max(0., self._silent[address][1] - now)
        elif address in self._probed:
            check = max(0., POLL_EVERY - (now - self._probed[address]))
        else:
            check = None
        return (max(0., min(heard)) if heard else None), check

    def poll_once(self):
        try:
            self.publish_state()
        except Exception:  # noqa: BLE001 - facts are a courtesy; the poll still runs
            _log.debug("publishing state failed", exc_info=True)
        peers = self.peers()
        if not peers:
            self.polled.set()
            return

        def ask(address):
            try:
                payload = self.client_for(address, retries=0, timeout=studio_cluster.POLL_TIMEOUT).get_json("node")
                info = studio_cluster.info_from_payload(address, payload)
            except Exception as exc:  # noqa: BLE001 - unreachable, slow or nonsense all mean "did not answer"
                return address, None, exc
            try:  # the observation store rides this poll; a peer without /state/sync (an older version) is still online
                studio_state.sync_peer(self.state, self.client_for(address, retries=0, timeout=studio_cluster.POLL_TIMEOUT))
            except Exception as exc:  # noqa: BLE001 - the next poll tries again
                _log.debug("state sync with %s failed: %s", address, exc)
            return address, info, None

        now = self.clock()
        self._silent = {a: v for a, v in self._silent.items() if a in peers}
        self._answered = {a: v for a, v in self._answered.items() if a in peers}
        self._probed = {a: v for a, v in self._probed.items() if a in peers}
        due = [a for a in peers if a not in self._silent or now >= self._silent[a][1]]  # a silent peer is asked less and less often
        with ThreadPoolExecutor(max_workers=min(8, len(due) or 1)) as pool:
            results = list(pool.map(ask, due))
        before = {address: _shown(self._known.get(address)) for address in peers}
        for address, info, exc in results:
            known = self._known.get(address)
            if info is None and studio_cluster.slow(exc) and known is not None and known.online and address not in self._silent:
                self._silent[address] = (0, self.clock())  # slow once: it keeps its last answer, and the next poll asks it again
                continue
            error = "" if exc is None else f"{type(exc).__name__}: {exc}"
            (self._slow.add if studio_cluster.slow(exc) else self._slow.discard)(address)
            self.health.observe(address, info is not None)
            if info is not None and info.hostname:
                self._names[address] = info.hostname
            self._known[address] = info or studio_cluster.offline(address, error)
            self._probed[address] = self.clock()
            if info is not None:
                self._silent.pop(address, None)
                self._answered[address] = self._probed[address]
            else:
                count = self._silent.get(address, (0,))[0] + 1
                self._silent[address] = (count, self._probed[address] + min(POLL_BACKOFF_MAX, 2 * POLL_EVERY * 2 ** (count - 1)))
        changed = sorted(address for address in peers if _shown(self._known.get(address)) != before[address])
        if changed:  # a machine came online, went offline or changed what it is: the machine lists refresh on this
            self.runner._emit("nodes", {"addresses": changed})
        self.polled.set()
        try:  # who coordinates which run, from the store as it is after this poll's syncs (studio_failover)
            self.failover.step()
        except Exception:  # noqa: BLE001 - a failover step is retried at the next poll
            _log.debug("failover step failed", exc_info=True)

    def nodes(self):
        """This node first, then each listed peer as last polled."""
        own = dataclasses.replace(self.own(), busy=self.busy())
        return [own, *(self._known.get(a) or studio_cluster.offline(a, "not polled yet") for a in self.peers())]

    # ---- owner: placing and dispatching -----------------------------------------------------------------------------
    def place(self, job):
        """Where `job` goes: a Placement. Without peers, or for a job another node owns, always here."""
        if job.owner:
            return sched.Placement("local")
        if job.kind == "pretrain_distributed":
            return self._place_gang(job)
        if not self.peers():
            return sched.Placement("local")
        if self._threads and not self.polled.is_set():
            self.polled.wait(studio_cluster.POLL_TIMEOUT + 1)
        placement = sched.decide(job.kind, self.nodes(), self.health, job.params.get("pinned"), store_ready=self.store() is not None,
                                 exclude=self.owner.refused_by(job.id), shape=sched.job_shape(job.kind, job.params))
        if placement.where == "wait":
            self.waiting[job.id] = placement.reason
        else:
            self.waiting.pop(job.id, None)
        if placement.basis:
            self.basis[job.id] = placement.basis
        else:
            self.basis.pop(job.id, None)
        return placement

    def _place_gang(self, job):
        """A distributed run goes on this machine (it coordinates and is a member) once `min_members` of the machines it was given are
        free at once; it then holds a slot on up to `max_members` of them (`job.params["gang"]`, the addresses), or on none and waits."""
        import studio_diloco
        import studio_distributed
        studio_distributed.release_stale(job, self.client_for)  # a run resuming after a restart frees the machines its old members hold
        if self.peers() and self._threads and not self.polled.is_set():
            self.polled.wait(studio_cluster.POLL_TIMEOUT + 1)
        here =sched.decide(job.kind, self.nodes(), self.health, job.params.get("pinned"))
        if here.where != "local":
            self.waiting[job.id] = here.reason
            return here
        try:
            settings = studio_diloco.Settings.from_params(job.params)
        except ValueError as exc:
            self.waiting[job.id] = str(exc)
            return sched.Placement("wait", reason=str(exc))
        chosen = {m.get("address") for m in job.params.get("machines") or []}
        nodes = [n for n in self.nodes() if n.address == studio_cluster.LOCAL or n.address in chosen]
        gang = self.gangs.place(job.id, nodes, self.health, settings.min_members, settings.max_members)
        if not gang.nodes:
            self.waiting[job.id] = gang.reason
            return sched.Placement("wait", reason=gang.reason)
        self.waiting.pop(job.id, None)
        if job.params.get("gang") != list(gang.nodes):
            with self.runner.lock:
                job.params = {**job.params, "gang": list(gang.nodes)}
        return sched.Placement("local")

    def dispatch(self, job, address):
        """Hand `job` to the peer at `address`. On success the job is running there, under a lease; a peer that refuses it, or a job that
        cannot be packaged, leaves it waiting with the reason (and that peer is not offered it again for a while)."""
        runner = self.runner
        try:
            store = self.store()
            if store is None:
                raise sched.Refused(self.store_problem)
            body, notes = self._package(job, store, address)
            if self.basis.get(job.id):
                notes = {**notes, "basis": self.basis[job.id]}
            result = self.owner.dispatch(job.id, address, body)
        except sched.Refused as exc:
            return self._refused(job, address, str(exc))
        except studio_publish.StoreLimitError as exc:  # the store would not take the job's input: not the peer's doing, so no cooldown for it
            self.waiting[job.id] = str(exc)
            runner._emit("cluster", {"job": job.id, "state": "store_refused", "node": address, "detail": str(exc)})
            return False
        except Exception as exc:  # noqa: BLE001 - reading the data or publishing the input failed: the sentence is the reason
            self.owner._refused[(job.id, address)] = self.clock() + sched.REFUSAL_COOLDOWN
            return self._refused(job, address, str(exc) or type(exc).__name__)
        with runner.lock:
            job.status, job.node, job.epoch = "running", address, result.lease.epoch
            job.params = {**job.params, "cluster": notes}
            if notes.get("checkpoint") and body["fetch"] and "resume_from" not in body["job"]["params"]:  # not on the node that has the full one
                job.note = f"{job.note} On {address} the optimizer starts again from zero because only the weights are published.".strip()
            self.waiting.pop(job.id, None)
            self.basis.pop(job.id, None)
            runner._save()
        runner._emit("job", {"job": job.id, "status": "running"})
        runner._emit("cluster", {"job": job.id, "state": "dispatched" if result.confirmed else "unconfirmed", "node": address,
                                 "epoch": result.lease.epoch, "detail": result.detail})
        return True

    def _refused(self, job, address, reason):
        self.waiting[job.id] = f"{address} would not take it: {reason}"
        self.runner._emit("cluster", {"job": job.id, "state": "refused", "node": address, "detail": reason})
        return False

    def _package(self, job, store, address=None):
        """The body of `POST /cluster/jobs`: the job with its training data inside, and the store key of the version it starts from.
        Sent to the node that wrote the job's last checkpoint, it also names that checkpoint (`resume_from`), so the node can continue
        from its own full copy, optimizer included, and fall back to the published weights when it cannot."""
        import studio_jobs
        import studio_rl
        import studio_worker
        rounds = job.kind in studio_jobs.ROUNDS
        if rounds and self.runner.prepare:
            self.runner.prepare(job)  # the checkpoint and teacher a round starts from are versions of this node (studio_rl.resolve)
        params = {k: v for k, v in job.params.items() if k not in LOCAL_ONLY}
        notes = dict(sched.attempt_state(job.params))
        fetch = dict(params.pop("fetch", None) or {})
        params.pop("cluster", None)
        context = studio_worker.JobContext(self.runner.root, self.runner.root / "runs")
        if rounds:  # its rows ship in the body; the checkpoint and teacher are published and handed over as keys
            for kind in studio_rl.rows_needed(studio_rl.number_of(job), params):
                params[kind] = context.post_rows(kind)
            before = dict(fetch)
            for name in ("checkpoint", "teacher"):
                path = params.pop(name, None)
                if isinstance(path, str) and path and name not in fetch:
                    fetch[name] = store.publish({"number": 0, "name": f"{name} of job {job.id}"}, path)
            if fetch != before:
                with self.runner.lock:  # recorded on the job, so a sweep sees the keys the runner is about to pull
                    job.params = {**job.params, "fetch": dict(fetch)}
        elif "examples" not in params:
            params["examples"] = context.examples(params)
        if not fetch and params.get("mode") == "catch_up" and params.get("init_from"):
            fetch["init_from"] = store.publish({"number": 0, "name": f"start of job {job.id}"}, params["init_from"])
            with self.runner.lock:  # recorded on the job, so a sweep sees the key the runner is about to pull
                job.params = {**job.params, "fetch": dict(fetch)}
        params.pop("init_from", None)
        notes.setdefault("steps", int((params.get("config") or {}).get("steps") or 0))
        notes.setdefault("offset", 0)
        resume = sched.resume_here(notes, address) if address and fetch else None
        if resume:
            params["resume_from"] = {**resume, "offset": notes["offset"]}
        notes["attempt_base"] = resume["base"] if resume else notes["offset"]  # the steps done before the attempt this epoch runs
        spec = {"id": job.id, "kind": job.kind, "label": job.label, "note": job.note, "params": params, "created_at": job.created_at,
                "eta_s": job.eta_s}
        return {"job": spec, "owner_address": self.address, "fetch": fetch}, notes

    def _remote_jobs(self):
        with self.runner.lock:
            return [j for j in self.runner.jobs if j.node and j.status == "running" and not j.owner]

    def _remote(self, job_id):
        return next((j for j in self._remote_jobs() if j.id == job_id), None)

    # ---- owner: what runners tell us --------------------------------------------------------------------------------
    def renewed(self, job_id, body):
        """`POST /cluster/jobs/{id}/renew`: extend the lease (StaleEpoch -> 409 for any other epoch) and take in the progress."""
        epoch = int(body["epoch"])
        lease, recovered = self.owner.renew(job_id, epoch)
        job = self._remote(job_id)
        if job is None:
            raise sched.StaleEpoch(job_id, epoch)
        problem = (body.get("problem") or "").strip()
        checkpoint = body.get("checkpoint")
        with self.runner.lock:  # (read and written under the lock: `_reassign` may have changed the notes since `_remote` looked)
            if self._remote(job_id) is not job or job.epoch != epoch:
                raise sched.StaleEpoch(job_id, epoch)
            notes = dict(sched.attempt_state(job.params))
            said = problem != notes.get("problem", "")  # a refused checkpoint is said once on the job, and again only if the reason changes
            changed = said
            if said:
                notes = {**notes, "problem": problem} if problem else {k: v for k, v in notes.items() if k != "problem"}
                if problem:
                    job.note = f"{job.note} {problem}".strip()
            if checkpoint and checkpoint.get("key") and (notes.get("checkpoint") or {}).get("key") != checkpoint["key"]:
                notes["checkpoint"] = {"key": checkpoint["key"], "step": int(notes.get("offset", 0)) + int(checkpoint.get("step") or 0),
                                       "node": lease.node, "epoch": epoch, "base": int(notes.get("attempt_base", notes.get("offset", 0)))}  # and where the full one is
                notes["checkpoints"] = [*(k for k in notes.get("checkpoints") or () if k != checkpoint["key"]), checkpoint["key"]]
                job.params = {k: v for k, v in job.params.items() if k != "fetch"}  # a new checkpoint: it has pulled the one it started from
                changed = True
            if changed:
                job.params = {**job.params, "cluster": notes}
                self.runner._save()
        if said and problem:
            self.runner._emit("cluster", {"job": job_id, "state": "checkpoint_refused", "node": lease.node, "detail": problem})
        for event in body.get("events") or ():
            if isinstance(event, dict) and event.get("type") in RESUME_EVENTS:
                notes = self._resume_report(job, epoch, event) or notes
                continue
            shifted = sched.relayed(event, int(notes.get("offset", 0)))
            if shifted is not None:
                self.runner._emit("train", {"job": job_id, **shifted})
        self.runner._emit("cluster", {"job": job_id, "state": body.get("state") or "running", "node": lease.node, "epoch": epoch,
                                      "recovered": recovered})
        return lease

    def _resume_report(self, job, epoch, event):
        """The runner says which way it went: from its own checkpoint with the optimizer ("resumed") or from the published weights
        ("resume_refused", with why). Said once per epoch on the job's note. A refusal also corrects `attempt_base`, which `_package`
        set for a resume. Returns the job's notes, or None when the job is no longer that runner's."""
        with self.runner.lock:
            if self._remote(job.id) is not job or job.epoch != epoch:
                return None
            notes = dict(sched.attempt_state(job.params))
            kind = event["type"]
            if notes.get(kind) == epoch:
                return notes
            notes[kind] = epoch
            if kind == "resumed":
                job.note = (f"{job.note} {job.node} continues from its own checkpoint at step "
                            f"{int(event.get('base') or 0) + int(event.get('step') or 0)}, with the optimizer state it had.").strip()
            else:
                notes["attempt_base"] = int(notes.get("offset", 0))
                if (notes.get("checkpoint") or {}).get("epoch") == epoch:
                    notes["checkpoint"] = {**notes["checkpoint"], "base": notes["attempt_base"]}
                job.note = (f"{job.note} {job.node} could not continue from its own checkpoint ({event.get('reason') or 'it is gone'}): "
                            "it continues from the published weights and the optimizer starts again from zero.").strip()
            job.params = {**job.params, "cluster": notes}
            self.runner._save()
            return notes

    def finished(self, job_id, body):
        """`POST /cluster/jobs/{id}/finished`: the runner's job ended. The result is a published version, listed here as available but
        not local."""
        epoch = int(body["epoch"])
        self.owner.end(job_id, epoch)
        job = self._remote(job_id)
        if job is None:
            raise sched.StaleEpoch(job_id, epoch)
        key = body.get("key")
        result = body.get("result") if job.kind in RESULT_KINDS and isinstance(body.get("result"), dict) else None
        with self._delete_lock:  # a sweep is never between "this key is free" and deleting it while the result is being recorded
            lost = bool(key) and self._missing(key)
            with self.runner.lock:
                if self._remote(job_id) is not job or job.epoch != epoch:
                    raise sched.StaleEpoch(job_id, epoch)
                notes = sched.attempt_state(job.params)
                job.status, job.error = body["status"], body.get("error") or ""
                if lost:  # the result has the bytes of an older checkpoint and a cleanup took that key before it was recorded
                    if job.status == "done":
                        job.status = "failed"
                    job.error = (f"{job.error} " if job.error else "") + (f"The result ({key}) is gone from the version store: a cleanup of "
                                                                          "old checkpoints removed it before this machine recorded it. Run the job again.")
                elif key:
                    job.params = {**job.params, "remote_result": {"key": key, "node": job.node,
                                                                  "step": int(notes.get("offset", 0)) + int(body.get("step") or 0)}}
                if result is not None and job.status != "failed":  # (a round's `checkpoint` is a path on the runner: the landing sets this node's)
                    job.params = {**job.params, "result": {**result, "checkpoint": None} if job.kind in ROUND_KINDS else result}
                self.runner._save()
        self._announce(job)

    @staticmethod
    def _result_key(job):
        key = (job.params.get("remote_result") or {}).get("key")
        return key if isinstance(key, str) else None

    def _announce(self, job):
        """Owner: a job that ran on a peer has ended. The weights of a job that finished (a pretrain's version, a round's checkpoint)
        are pulled from the store first (`_land`), so what hears of the end finds them here as it would after a local run; the rest
        is said at once."""
        with self.runner.lock:
            pull = job.status == "done" and job.kind in PULLED_KINDS and bool(self._result_key(job))
            if pull:
                job.params = {**job.params, "landing": True}  # persisted: a restart, or a pull that failed, finds it again (`_land_pending`)
                self.runner._save()
        if pull:
            self._land_in_background(job)
        else:
            self._settle(job)

    def _settle(self, job):
        """Say the job ended. A compare's result is written where a local compare's would be, which is where the app reads it."""
        if job.status == "done" and job.kind == "compare" and isinstance(job.params.get("result"), dict):
            try:
                self._write_compare(job)
            except (OSError, ValueError, TypeError) as exc:
                with self.runner.lock:
                    job.status, job.error = "failed", f"The comparison finished on {job.node} but its result could not be saved here: {exc}"
                    self.runner._save()
        self.runner._emit("job", {"job": job.id, "status": job.status, "error": job.error})
        self.runner.wake.set()

    def _write_compare(self, job):
        import studio_experiments
        from studio_core import atomic_json
        out = self._folder(job)
        out.mkdir(parents=True, exist_ok=True)
        studio_experiments.ExperimentSpec(**job.params["experiment"]).normalized().save(out / "experiment.json")
        atomic_json(out / "results.json", job.params["result"])
        with self.runner.lock:
            job.params = {**job.params, "output_dir": str(out)}
            self.runner._save()

    def _folder(self, job):
        import studio_jobs
        return studio_jobs.output_dir(job, self.runner.root / "runs")

    def _land_in_background(self, job):
        with self._keep_lock:
            if job.id in self._landing:
                return
            self._landing.add(job.id)
        if not self.threads:
            return self._land(job)
        threading.Thread(target=self._land, args=(job,), daemon=True, name=f"land {job.id}").start()

    def _land_pending(self):
        """Owner: bring home the weights of finished jobs whose pull has not happened (the app restarted, or the store did not answer)."""
        with self.runner.lock:
            due = [j for j in self.runner.jobs if not j.owner and j.params.get("landing") and self._retry_at.get(j.id, 0.) <= self.clock()]
        for job in due:
            self._land_in_background(job)

    def _land(self, job):
        """Pull the weights a peer finished into the folder a local run would have written them to (`best.npz`), point the job's
        `output_dir` and, for a round, its result's `checkpoint` there, and only then say the job ended: the version is local by then.
        A pretrain on new text then leaves its restore_answers round, as a local one does. A pull that fails is tried again, and then
        fails the job with the key, which stays in the store."""
        key = self._result_key(job)
        try:
            store = self.store()
            if store is None:
                raise studio_publish.StoreError(self.store_problem)
            pulled = Path(store.pull(key, self.runner.root / "data" / "pulled"))
            out = self._folder(job)
            out.mkdir(parents=True, exist_ok=True)
            shutil.move(str(pulled), out / "best.npz")
        except Exception as exc:  # noqa: BLE001 - the store or the disk said no: the sentence is the reason
            return self._land_failed(job, key, str(exc) or type(exc).__name__)
        with self._keep_lock:
            self._landing.discard(job.id)
            self._tries.pop(job.id, None)
            self._retry_at.pop(job.id, None)
        with self.runner.lock:
            params = {k: v for k, v in job.params.items() if k != "landing"}
            params["output_dir"] = str(out)
            if job.kind in ROUND_KINDS and isinstance(params.get("result"), dict):
                params["result"] = {**params["result"], "checkpoint": str(out / "best.npz")}
            job.params = params
            self.runner._save()
        self._settle(job)  # (the app records the version on this event, before the follow-up below can be placed)
        with self.runner.lock:
            self.runner.follow_up(job)
            self.runner._save()
        self.runner.wake.set()

    def _land_failed(self, job, key, reason):
        with self._keep_lock:
            self._landing.discard(job.id)
            tries = self._tries[job.id] = self._tries.get(job.id, 0) + 1
            self._retry_at[job.id] = self.clock() + LAND_RETRY_S
        _log.warning("Could not bring the result %s of job %s here (try %s of %s): %s", key, job.id, tries, LAND_TRIES, reason)
        if tries < LAND_TRIES:
            return
        with self.runner.lock:
            job.status = "failed"
            job.error = f"The job finished on {job.node}, but its result ({key}) could not be brought here: {reason} It stays in the version store."
            job.params = {k: v for k, v in job.params.items() if k != "landing"}
            self.runner._save()
        self._settle(job)

    def _missing(self, key):
        """True only when the store says `key` is not there; a store that cannot say (no `exists`, or it fails) is believed."""
        exists = getattr(self.store(), "exists", None)
        try:
            return exists is not None and not exists(key)
        except Exception:  # noqa: BLE001 - could not look: the runner said it published it
            return False

    def cancel(self, job):
        """The user removed a job that is running on a peer: tell the peer, drop the lease, keep the record as stopped."""
        self.owner.cancel(job.id)
        with self.runner.lock:
            job.status = "stopped"
            self.runner._save()
        self.runner._emit("job", {"job": job.id, "status": "stopped", "error": ""})
        self.runner.wake.set()

    def tick(self):
        """Owner: act on leases that lapsed. Runner: renew leases, publish checkpoints, stop jobs the owner no longer wants."""
        for job_id, what in self.owner.tick():
            if what == "lapsed":
                job = self._remote(job_id)
                self.runner._emit("cluster", {"job": job_id, "state": "lapsed", "node": job.node if job else ""})
            else:
                self._reassign(job_id)
        for run in list(self.guests.values()):
            self._tick_guest(run)
        self._tick_gangs()
        studio_cluster_watch.check(self.runner, self.coordinators, self.client_for)  # a run whose driver stopped ends, its members released
        self._land_pending()
        self._sweep_checkpoints()

    def _tick_gangs(self):
        """Owner: free the slots of runs that ended, renew the slots whose member the coordinator hears from, and for each slot that
        lapsed let its member go at the round boundary and free the slot. Open slots are offered each tick to the machines that are free
        then (`offer`), so a machine that frees up joins the run at the round after it is invited. A machine holding a slot whose
        invitation failed is asked again once per round (`studio_distributed.reinvite`)."""
        import studio_distributed
        import studio_diloco
        with self.runner.lock:
            live = {j.id: j for j in self.runner.jobs if j.kind == "pretrain_distributed" and not j.owner and j.status in ("waiting", "running")}
        for run_id in self.gangs.runs():
            job = live.get(run_id)
            coordinator = studio_diloco.COORDINATORS.get(run_id)
            if job is None or (coordinator is not None and coordinator.finished):
                self.gangs.end(run_id)
            elif job.status == "running" and coordinator is not None:
                run = studio_distributed.run_of(job)
                nodes = coordinator.snapshot()["nodes"]
                for address in list(self.gangs.held(run_id)):
                    machine = next((m for m in run.machines if m.address == address), None)
                    if machine is not None and machine.peer_job:
                        import studio_facts
                        failed = next((f for f in studio_facts.member_failures(self, machine.node) if f["id"] == machine.peer_job), None)
                        if failed is not None:
                            coordinator.lose(machine.node)
                            studio_distributed.lapse(run, machine.node, self.client_for)
                            chosen = {m.get("address") for m in (job.params.get("machines") if job else ()) or []}
                            self.gangs.drop(run_id, address, ban=address not in chosen)
                            self.runner._emit("cluster", {"job": run_id, "state": "member_lapsed", "node": address, "detail": machine.node})
                            continue
                    known = nodes.get(machine.node) if machine is not None else None
                    reported = ((known or {}).get("progress") or {}).get("reported_ago")
                    if address == studio_cluster.LOCAL or (known is not None and known["state"] in ("training", "behind", "joining")
                                                           and (reported is None or reported <= self.lease_s)):
                        self.gangs.heard(run_id, address)
                studio_distributed.reinvite(job, coordinator.next_round, self.client_for, among=set(self.gangs.held(run_id)))
                self.offer(job)
        for run_id, address in self.gangs.lapsed():
            job, coordinator = live.get(run_id), studio_diloco.COORDINATORS.get(run_id)
            self.gangs.drop(run_id, address)
            if job is None or coordinator is None:
                continue
            run = studio_distributed.run_of(job)
            machine = next((m for m in run.machines if m.address == address), None)
            if machine is not None:
                coordinator.lose(machine.node)
                studio_distributed.lapse(run, machine.node, self.client_for)
                self.runner._emit("cluster", {"job": run_id, "state": "member_lapsed", "node": address, "detail": machine.node})

    def offer(self, job):
        """Offer the open slots of a running distributed run to the machines that are free now, and invite them. Returns the addresses
        that got a slot. A machine that comes online after the run began is offered one while the run has room (`max_members`) and its
        `auto_join` switch is on (the default); with it off only the machines the person chose, or asked to join, are offered. A machine
        that cannot join (another app version, busy, too little memory) gets no slot, and the run's row for it says why. The ones that
        fit are sized like the first members (studio_fit.fit_members) and join at the round after their invitation."""
        import studio_distributed
        import studio_diloco
        try:
            settings = studio_diloco.Settings.from_params(job.params)
        except ValueError:
            return []
        held = self.gangs.held(job.id)
        if not held or len(held) >= settings.max_members:
            return []
        chosen = {m.get("address") for m in job.params.get("machines") or []}
        auto = job.params.get("auto_join", True)
        candidates = [n for n in self.nodes() if n.address == studio_cluster.LOCAL or n.address in chosen or auto]
        run, coordinator = studio_distributed.run_of(job), studio_diloco.COORDINATORS.get(job.id)
        version, fits, eligible = self.own().version, {}, []
        for n in candidates:
            if n.address == studio_cluster.LOCAL or n.address in held:
                continue
            if not n.online or self.health.state(n.address) != sched.UP:
                continue
            reason, fit = studio_distributed.late_check(job, n, version, coordinator)
            if reason:
                studio_distributed.note_unplaced(run, n, reason)
            else:
                eligible.append(n)
                if fit is not None:
                    fits[n.address] = fit
        added = self.gangs.fill(job.id, eligible, self.health, settings.max_members)
        if added:
            with self.runner.lock:
                member_fit = {**(job.params.get("member_fit") or {}), **{a: fits[a] for a in added if a in fits}}
                job.params = {**job.params, "member_fit": member_fit} if member_fit else job.params
            with studio_distributed._LOCK:
                for addr in added:
                    machine = next((m for m in run.machines if m.address == addr), None)
                    if machine is None:
                        node_name = next((n.hostname or n.address for n in eligible if n.address == addr), addr)
                        machine = studio_distributed.Machine(node=node_name, address=addr, state="unplaced", reason=studio_distributed.UNPLACED)
                        run.machines.append(machine)
                    machine.late = True
                    slot = self.gangs.held(job.id).get(addr)
                    if slot:
                        machine.epoch = slot.epoch
                    machine.invite_id = ""
            studio_distributed.invite(job, self.client_for, only=added)
            self.runner._emit("cluster", {"job": job.id, "state": "member_added", "node": ",".join(added), "detail": ""})
        return added

    def _reassign(self, job_id):
        """The runner has been silent for two leases, so its self-fence has fired: the job is interrupted at its last checkpoint and
        goes back to waiting with the next epoch, to be placed again (the same node when it returns, or another that fits)."""
        job = self._remote(job_id)
        if job is None:
            return
        node = job.node
        with self.runner.lock:
            if self._remote(job_id) is not job:  # it ended, or was cancelled or placed again, since it was looked up: leave it as it is
                return
            note, finished = sched.interrupt(job)
            self.runner._save()
        self.runner._emit("cluster", {"job": job_id, "state": "interrupted", "node": node, "detail": note})
        if finished:
            self._announce(job)
        self.runner.wake.set()

    def jobs_view(self):
        """Jobs owned here that run (or wait to run) away from this node, and jobs this node runs for others."""
        rows = []
        with self.runner.lock:
            owned = [j for j in self.runner.jobs if not j.owner and (j.node or j.id in self.waiting) and j.status in ("running", "waiting")]
        for job in owned:
            lease = self.owner.leases.get(job.id)
            notes = sched.attempt_state(job.params)
            rows.append({"id": job.id, "kind": job.kind, "label": job.label, "role": "owner", "node": job.node, "epoch": job.epoch,
                         "status": job.status, "lease": ("lapsed" if job.id in self.owner._lapsed else "held") if lease else "",
                         "waiting_reason": self.waiting.get(job.id, ""), "checkpoint_step": (notes.get("checkpoint") or {}).get("step"),
                         "basis": notes.get("basis", "") if job.status == "running" else ""})
        for run in self.guests.values():
            if not run.ended:
                rows.append({"id": run.id, "kind": run.spec["kind"], "label": run.spec.get("label", ""), "role": "runner",
                             "node": run.owner_address, "epoch": run.lease.epoch, "status": self._guest_state(run),
                             "lease": "expired" if run.guest.fence.expired() else "held", "waiting_reason": run.problem or run.refusal,
                             "checkpoint_step": (run.checkpoint or {}).get("step")})
        return rows

    # ---- owner: the checkpoints of jobs that cannot resume ----------------------------------------------------------
    def removed(self, job):
        """A job was taken out of the queue (`Runner.remove`): whatever checkpoints it had recorded can never be resumed from now, but the
        record that held them is gone, so they are kept here until the store has let them go (and so is its result's key, which can be the
        bytes of one of them). A restart forgets both (`--check` lists what is left). No store I/O: this runs on the UI thread."""
        keys = self._checkpoint_keys(job)
        result = (job.params.get("remote_result") or {}).get("key")
        if job.owner or not (keys or result):
            return
        with self._keep_lock:
            if isinstance(result, str):
                self._results.add(result)
            if keys:
                self._removed[job.id] = [*self._removed.get(job.id, ()), *(k for k in keys if k not in self._removed.get(job.id, ()))]

    def trimmed(self, jobs):
        """Jobs the queue dropped from its record of finished ones (`Runner._save` keeps 50): their result keys stay protected. No I/O."""
        with self._keep_lock:
            self._results.update(r for r in ((j.params.get("remote_result") or {}).get("key") for j in jobs if not j.owner)
                                 if isinstance(r, str))

    @staticmethod
    def _checkpoint_keys(job):
        return [k for k in sched.attempt_state(job.params).get("checkpoints") or () if isinstance(k, str)]

    def _protected(self):
        """The keys that must not be deleted now: the result of any job (a result can be the bytes of a checkpoint), and for every job that
        has not ended the key it resumes from (`fetch`, which the next runner pulls), the one its notes name and the newest it reported."""
        with self.runner.lock:
            keep = {(j.params.get("remote_result") or {}).get("key") for j in self.runner.jobs}
            for job in self.runner.jobs:
                if job.status not in TERMINAL:
                    keep.update((job.params.get("fetch") or {}).values())
                    keep.add((sched.attempt_state(job.params).get("checkpoint") or {}).get("key"))
                    keep.update(self._checkpoint_keys(job)[-1:])
        with self._keep_lock:
            keep |= self._results
        return {k for k in keep if isinstance(k, str)}

    def _doomed(self):
        """[(job id, [keys])] to delete now: every checkpoint a finished, failed or cancelled job reported, and all but the newest of a job
        that can still resume (only the newest is resumed from). Anything `_protected` stays. Only keys a runner reported for the job are
        ever listed."""
        kept, doomed = self._protected(), []
        with self.runner.lock:
            for job in (j for j in self.runner.jobs if not j.owner):
                keys = self._checkpoint_keys(job)
                if job.status not in TERMINAL:
                    keys = keys[:-1]
                keys = [k for k in keys if k not in kept]
                if keys:
                    doomed.append((job.id, keys))
        with self._keep_lock:
            held = {k for _, keys in doomed for k in keys}
            for job_id, keys in self._removed.items():
                keys = [k for k in keys if k not in kept and k not in held]
                if keys:
                    doomed.append((job_id, keys))
        return doomed

    def _sweep_checkpoints(self):
        """Owner: delete, in the background and not more than once in SWEEP_RETRY_S after a failure, the checkpoints of jobs that can no
        longer resume from them. Deleting never uploads; nothing here goes through the size check because nothing here adds a byte."""
        if self._sweeping or self.clock() < self._sweep_at:
            return
        doomed = self._doomed()
        store = self.store() if doomed else None
        if store is None or getattr(store, "delete", None) is None or getattr(store, "metadata", None) is None:
            return  # (a store that cannot delete, or cannot say what a key is: nothing is deleted, nothing to repeat)
        self._sweeping = True
        if not self.threads:
            return self._sweep_now(doomed)
        threading.Thread(target=self._sweep_now, args=(doomed,), daemon=True, name="sweep checkpoints").start()

    # Lock order: `_delete_lock` is taken before `runner.lock`, never the other way round. `renewed` and `_reassign` do not take it, and need
    # not: they only add checkpoint keys, or point `fetch` at the newest one, which `_protected` already counts; the one key they stop
    # protecting is a `fetch` the runner has pulled. So a sweep that checked a key is free cannot have it taken by them before it deletes.
    def _sweep_now(self, doomed):
        try:
            store, failed = self.store(), False
            for job_id, keys in doomed:
                for key in keys:
                    with self._delete_lock:  # (what the list said was free may have been taken since: look again, and delete before anyone can)
                        if key in self._protected():
                            continue
                        done = self._delete_checkpoint(store, job_id, key)
                    if done:
                        self._forgotten(job_id, key)
                    else:
                        failed = True
            if failed:
                self._sweep_at = self.clock() + SWEEP_RETRY_S
        except Exception as exc:  # noqa: BLE001 - a sweep that fails is tried again; it never stops the queue
            self._sweep_at = self.clock() + SWEEP_RETRY_S
            _log.warning("Could not delete checkpoints of finished jobs from the store: %s", exc)
        finally:
            self._sweeping = False

    def _delete_checkpoint(self, store, job_id, key):
        """True when `key` is gone from the store or is left alone for good (not a checkpoint of job `job_id`, or not listed); False,
        logged, when the store would not do it. Fails closed: a key with no listing is not deleted, unless a delete of it was begun here
        and the store cut it short (the listing goes first)."""
        try:
            meta = store.metadata(key)
            if meta is None and key not in self._begun:
                _log.warning("Not deleting %s: the store has no listing for it, so it is not known to be a checkpoint of job %s.", key, job_id)
                return True
            if meta is not None and studio_publish.checkpoint_job(meta) != job_id:
                _log.warning("Not deleting %s: the store does not say it is a checkpoint of job %s.", key, job_id)
                return True
            self._begun.add(key)
            store.delete(key)
            self._begun.discard(key)
            return True
        except Exception as exc:  # noqa: BLE001 - a checkpoint left behind costs space, not the job
            _log.warning("Could not delete checkpoint %s of job %s from the store: %s", key, job_id, exc)
            return False

    def _forgotten(self, job_id, key):
        """`key` is deleted (or was never the job's): take it off the job's record, so it is not tried again."""
        with self._keep_lock:  # (a leaf lock: nothing else is taken while it is held)
            if job_id in self._removed:
                self._removed[job_id] = [k for k in self._removed[job_id] if k != key]
                if not self._removed[job_id]:
                    del self._removed[job_id]
        with self.runner.lock:
            job = next((j for j in self.runner.jobs if j.id == job_id), None)
            notes = sched.attempt_state(job.params) if job else {}
            if job is not None and key in (notes.get("checkpoints") or ()):
                job.params = {**job.params, "cluster": {**notes, "checkpoints": [k for k in notes["checkpoints"] if k != key]}}
                self.runner._save()

    # ---- runner: taking a job in ------------------------------------------------------------------------------------
    def accept(self, spec, lease, owner_address, fetch):
        """`POST /cluster/jobs`: take a job another node owns. Refused (409) without a version store or for a kind that cannot leave its
        owner; a lease at an epoch below the one held is StaleEpoch. The version it starts from is pulled in the background."""
        if spec["kind"] not in sched.REMOTE_KINDS:
            raise sched.Refused(f"A {spec['kind']} job only runs on the machine that queued it.")
        store = self.store()
        if store is None:
            raise sched.Refused(self.store_problem)
        if not owner_address:
            raise sched.Refused("The job did not say how to reach its owner, so this machine could not renew its lease.")
        with self._lock:
            held = self.guests.get(lease.job_id)
            if held and not held.ended:
                if lease.epoch <= held.lease.epoch:
                    raise sched.StaleEpoch(lease.job_id, lease.epoch, held.lease.epoch)
                self._stop_guest(held, "replaced")
            owner = self.client_for(owner_address, retries=0, timeout=lease.renew_every)
            run = GuestRun(lease, spec, fetch, owner_address, sched.Guest(lease, owner, self.clock))
            self.guests[lease.job_id] = run
        self.start()
        threading.Thread(target=self._launch, args=(run, store), daemon=True, name=f"guest {run.id}").start()
        return run

    def _launch(self, run, store):
        """Pull what the job starts from, then queue it here like any other job."""
        import studio_queue
        try:
            params = dict(run.spec.get("params") or {})
            for name, key in (run.fetch or {}).items():
                params[name] = str(store.pull(key, self.runner.root / "data" / "pulled"))
            self._own_checkpoint(run, params)
            if run.cancelled:
                return
            job = studio_queue.Job(kind=run.spec["kind"], label=f"{run.spec.get('label') or run.spec['kind']} (for {run.owner_address})",
                                   note=run.spec.get("note") or "", params=params, id=run.local_id, owner=run.owner_address,
                                   epoch=run.lease.epoch, created_at=run.spec.get("created_at") or time.time())
            if self.runner.thread is None:
                self.runner.start()
            run.enqueued = True
            self.runner.enqueue(job)
        except Exception as exc:  # noqa: BLE001 - the pull or the queueing failed: the owner is told in the plain
            run.problem = str(exc) or type(exc).__name__
            self._report(run, "failed", run.problem)

    def _own_checkpoint(self, run, params):
        """The owner named the checkpoint this node wrote (`resume_from`): continue from its full file when the attempt's folder still
        has it (`studio_jobs.pretrain` falls back to the pulled weights if training refuses it). Otherwise the owner is told why."""
        named = params.pop("resume_from", None)
        if not named:
            return
        path = self.runner.root / "runs" / f"queue_{run.id}-e{int(named['epoch'])}" / "last.npz"
        if path.is_file():
            params.update(resume=str(path), resume_steps=int(named["steps"]), resume_base=int(named["base"]))
            run.shift = int(named["base"]) - int(named["offset"])
        else:
            run.keep({"type": "resume_refused", "reason": "its folder no longer has it"})

    def stop_guest(self, job_id, epoch):
        """`POST /cluster/jobs/{id}/stop`: the owner no longer wants this copy."""
        run = self.guests.get(job_id)
        if run is None or run.ended or run.lease.epoch != epoch:
            raise sched.StaleEpoch(job_id, epoch, run.lease.epoch if run else None)
        self._stop_guest(run, "cancelled")

    def _stop_guest(self, run, why):
        """Stop this node's copy of a job: it is not told again and its end is not reported."""
        run.stopped_by, run.cancelled, run.ended = why, True, True
        self.runner._emit("cluster", {"job": run.id, "state": why, "node": run.owner_address, "epoch": run.lease.epoch})
        job = next((j for j in self.runner.jobs if j.id == run.local_id and j.status in ("waiting", "held", "running")), None)
        if job is not None:
            self.runner.remove(job.id)

    def _guest_state(self, run):
        job = next((j for j in self.runner.jobs if j.id == run.local_id), None)
        if job is None:
            return "fetching" if not run.enqueued else "ended"
        return {"waiting": "queued", "held": "queued", "running": "running"}.get(job.status, job.status)

    def _tick_guest(self, run):
        if run.ended:
            return
        self._publish_checkpoint(run)
        with run.lock:
            count = len(run.events)

        sent = []  # the checkpoint key this renewal tells the owner about

        def progress():
            job = next((j for j in self.runner.jobs if j.id == run.local_id), None)
            with run.lock:
                events = list(run.events[:count])
            sent[:] = [(run.checkpoint or {}).get("key")]
            step = max((e["step"] for e in events if isinstance(e.get("step"), int)), default=0)
            total = max((e["total"] for e in events if isinstance(e.get("total"), int)), default=0)
            return {"state": self._guest_state(run), "step": step, "total": total, "events": events, "checkpoint": run.checkpoint,
                    "waiting": bool(job and job.status != "running"), "problem": run.refusal}

        def acked():
            with run.lock:
                del run.events[:count]
            if sent and sent[0]:  # the owner has this checkpoint now: the ones before it are no use to anyone
                threading.Thread(target=self._prune, args=(run, sent[0]), daemon=True, name=f"prune {run.id}").start()

        verdict = run.guest.renew(progress, acked)
        if verdict:
            self._stop_guest(run, verdict)

    def _publish_checkpoint(self, run):
        """Upload the job's newest checkpoint (weights only) so the owner can resume from it if this node is lost."""
        job = next((j for j in self.runner.jobs if j.id == run.local_id and j.status == "running"), None)
        store = self.store()
        if job is None or store is None or run.publishing or not job.params.get("output_dir") or run.spec["kind"] != "pretrain":
            return  # (only a pretrain can continue from a checkpoint; a round or a comparison starts again)
        path = Path(job.params["output_dir"]) / "last.npz"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if mtime <= max(run.published_mtime, run.refused_mtime) or self.clock() < run.retry_at:
            return
        run.publishing = True

        def work():
            step = 0
            try:
                step = checkpoint_step(path) + run.shift
                key, created = self._publish(store, studio_publish.checkpoint_version(run.id, step), path)
                with run.lock:
                    run.checkpoint, run.published_mtime, run.refusal = {"key": key, "step": step}, mtime, ""
                    if created:
                        run.stored.append(key)
            except studio_publish.StoreUsageError as exc:  # not the limit: the store could not be checked, so this one is offered again soon
                run.refusal, run.retry_at = f"Checkpoint at step {step} not shared: {exc}", self.clock() + USAGE_RETRY_S
            except studio_publish.StoreLimitError as exc:  # the job keeps training; the owner is told through the next renewal (`problem`)
                run.refused_mtime, run.refusal = mtime, f"Checkpoint at step {step} not shared: {exc}"
            except Exception as exc:  # noqa: BLE001 - the next checkpoint tries again; the owner resumes from the last one that made it
                run.problem = str(exc) or type(exc).__name__
            finally:
                run.publishing = False
        threading.Thread(target=work, daemon=True, name=f"publish {run.id}").start()

    @staticmethod
    def _publish(store, version, path):
        """(key, created): `store.publish`, and whether this call put the objects up. A store that cannot say counts as not having, so
        nothing it holds is ever deleted on its word."""
        report = getattr(store, "publish_reporting", None)
        if report is not None:
            return report(version, path)
        return store.publish(version, path), False

    def _prune(self, run, upto):
        """Delete the checkpoints this run stored before `upto` (the one the owner has now)."""
        with run.lock:
            if upto not in run.stored:
                return
            doomed = run.stored[:run.stored.index(upto)]
            del run.stored[:len(doomed)]
        self._delete_all(run, doomed)

    def _forget_checkpoints(self, run, keep=None):
        """Delete every checkpoint this run stored except `keep` (the result, when it is the same bytes); returns how many went."""
        with run.lock:
            doomed = [key for key in run.stored if key != keep]
            run.stored[:] = [key for key in run.stored if key == keep]
        return self._delete_all(run, doomed)

    def _delete_all(self, run, keys):
        """Delete `keys` from the store. One that will not go is logged and kept in the run's list for the next time; it is never fatal."""
        delete = getattr(self.store(), "delete", None)
        gone, failed = 0, []
        for key in keys:
            try:
                if delete is None:
                    raise AttributeError("this store cannot delete")
                delete(key)
                gone += 1
            except Exception as exc:  # noqa: BLE001 - a checkpoint left behind costs space, not the job
                failed.append(key)
                _log.warning("Could not delete checkpoint %s of job %s from the store: %s", key, run.id, exc)
        if failed:
            with run.lock:
                run.stored[:0] = failed
        return gone

    # ---- runner: the end of a job -----------------------------------------------------------------------------------
    def _on_event(self, kind, payload):
        if isinstance(payload, dict):
            if kind == "cluster":
                self.events.append({**payload, "at": self.clock()})
                del self.events[:-studio_facts.EVENTS_KEPT]
            elif kind == "train" and payload.get("type") == "diloco_member" and payload.get("run"):
                self.members[payload["run"]] = studio_facts.member_fact(payload, self.clock())
            elif kind == "train" and payload.get("type") == "diloco_replica" and payload.get("run"):
                self.replicas[payload["run"]] = {k: payload.get(k) for k in ("term", "round", "rank", "ready", "held")}
        run = next((r for r in self.guests.values() if payload.get("job") == r.local_id), None) if isinstance(payload, dict) else None
        if run is None or run.ended:
            return
        if kind == "train":
            event = {k: v for k, v in payload.items() if k != "job"}
            if event.get("type") == "resume_refused":
                run.shift = 0  # (before any step: training refused the checkpoint and goes on from the published weights)
            elif event.get("type") in sched.EVENTS_RELAYED:
                event = sched.relayed(event, run.shift)
            run.keep(event)
        elif kind == "job" and payload.get("status") in TERMINAL:
            threading.Thread(target=self._conclude, args=(run, payload["status"], payload.get("error") or ""), daemon=True,
                             name=f"finish {run.id}").start()

    def _conclude(self, run, status, error):
        """Publish what the job made, then tell the owner. The lease keeps being renewed meanwhile, by `tick`. Once the owner has the
        result, the job's checkpoints are of no use and are deleted. A job that resumed (or could not) is not reported before the owner
        has been told which way it went, so a short job still gets that on its note."""
        while not (run.stopped_by or run.guest.fence.expired()) and any(e.get("type") in RESUME_EVENTS for e in list(run.events)):
            if self.runner.closed.wait(.2):
                break
        key, step, kind = None, 0, run.spec["kind"]
        job = next((j for j in self.runner.jobs if j.id == run.local_id), None)
        out = Path((job.params.get("output_dir") if job else "") or "")
        result = job.params.get("result") if job and kind in RESULT_KINDS and status == "done" else None
        if result is not None and len(json.dumps(result, default=str)) > RESULT_MAX_BYTES:
            result, status, error = None, "failed", f"The result is larger than the {RESULT_MAX_BYTES >> 20} MB a job may report."
        if kind in ROUND_KINDS:  # the weights that count are the round's candidate, and only when it passed its gate
            named = (result or {}).get("checkpoint") if (result or {}).get("ok") else None
            path = Path(named) if named and Path(named).is_file() else None
        elif kind == "compare":
            path = None  # (a comparison keeps its checkpoints; the owner needs only what it measured)
        else:
            path = next((p for p in (out / "best.npz", out / "last.npz") if p.is_file()), None)
        if path is not None:
            try:
                step = checkpoint_step(path) + run.shift
                key = self._publish_result(run, {"number": 0, "name": f"job:{run.id}", "change": f"result at step {step}"}, path)
            except Exception as exc:  # noqa: BLE001 - the owner gets the reason instead of a result it could not use
                where = self.address or "the machine that ran it"
                status = "failed"
                error = f"{str(exc) or type(exc).__name__} The finished weights are kept on {where}, at {path}."
        if self._report(run, status, error, key, step, result) and key:
            self._forget_checkpoints(run, keep=key)

    def _publish_result(self, run, version, path):
        """Publish the finished weights. A store at its limit gets one more try after the job's own checkpoints are deleted, which are
        the one thing here that can free room; a store that could not be checked, or has nothing of this job's to delete, does not."""
        store = self.store()
        try:
            return store.publish(version, path)
        except studio_publish.StoreUsageError:
            raise
        except studio_publish.StoreLimitError:
            if not self._forget_checkpoints(run):
                raise
            return store.publish(version, path)

    def _report(self, run, status, error, key=None, step=0, result=None):
        """Tell the owner how the job ended; True when it took the report (a "stale" or undelivered report is False)."""
        body = {"status": status, "error": error, "key": key, "step": step, **({"result": result} if result is not None else {})}
        delivered = False
        while not run.stopped_by and not run.guest.fence.expired():
            outcome = run.guest.finish(body)
            if outcome == "retry":
                if self.runner.closed.wait(1.):
                    break
                continue
            if outcome == "stale":
                self._stop_guest(run, "stale")
            delivered = outcome == "ok"
            break
        run.ended = True
        return delivered


def _version():
    from studio_api.version import app_version
    return app_version()
