"""The I/O edge of the scheduler (studio_scheduler.py has the rules): one `Cluster` per queue Runner.

As the **owner** of the jobs queued here it polls the peers' `GET /node`, places each waiting job with `decide`, hands a job that
belongs on a peer over (`POST /cluster/jobs` on the peer, the job spec carrying its training data and a lease), relays the progress
the peer reports into this node's event stream, and places a job again when its lease lapses.

As a **runner** of a job another node owns it checks the lease and the version store, pulls the version the job starts from, puts the
job in this node's own queue (it runs in its own worker process, studio_worker, like any other), renews the lease on the owner every
third of its duration with the job's progress, publishes the job's checkpoints to the version store, and reports the end. It stops the
job itself when the lease cannot be renewed in time (self-fencing) or the owner answers 409 (the job has moved on).

The version store is a shared folder (`TRAINER_STORE_FOLDER`) or an S3-compatible bucket (`TRAINER_STORE_*`, docs/cluster.md). A job
that leaves this machine needs it: the checkpoints and the finished version travel through it, so without it nothing is dispatched.
`TRAINER_NODE_ADDRESS` is how peers reach this node when the address they see the request come from is not it (behind a proxy).
"""
from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import studio_cluster
import studio_scheduler as sched
from studio_peer import PeerClient

POLL_EVERY = 5.  # seconds between polls of the peers
TICK_EVERY = 1.  # seconds between lease checks and renewals
EVENT_BACKLOG = 200  # events a runner keeps for its next renewal
NO_STORE = ("A job can only leave this machine once a version store is set up, because its checkpoints and its result travel through "
            "it. Set TRAINER_STORE_FOLDER to a shared folder, or the TRAINER_STORE_* bucket variables (docs/cluster.md).")
LOCAL_ONLY = ("version", "user_text", "previous_best", "output_dir", "result", "max_batch_size", "pinned", "remote_result")  # not for the runner
TERMINAL = ("done", "failed", "stopped")


def client_for(address, **options):
    return PeerClient(address, **options)


def default_store(environ=None):
    """The version store this machine's environment describes: a shared folder, else a bucket, else None. A half-set bucket is a
    ValueError that names the missing variables."""
    environ = os.environ if environ is None else environ
    folder = (environ.get("TRAINER_STORE_FOLDER") or "").strip()
    if folder:
        from studio_publish import FolderStore
        return FolderStore(folder)
    from studio_publish import S3Store
    return S3Store.from_env(environ)


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
        self.enqueued = self.cancelled = self.ended = self.reported = False
        self.stopped_by = None  # "stale" | "fenced" | "cancelled" | "replaced": the owner no longer wants this copy
        self.problem = ""

    def keep(self, event):
        with self.lock:
            self.events.append(event)
            del self.events[:-EVENT_BACKLOG]


class Cluster:
    """Everything a Runner needs to take part in a cluster. Inert (no polling, no threads of consequence) while no peers are listed in
    the workspace config. `client_for(address, **options)` makes the PeerClient; tests pass fakes for it, for `store`, `own` and `clock`."""

    def __init__(self, runner, client_for=client_for, store=None, own=None, clock=time.monotonic, address=None, environ=None,
                 lease_s=sched.LEASE_S, threads=True):
        self.runner, self.client_for, self.clock, self.lease_s, self.threads = runner, client_for, clock, lease_s, threads
        self.environ = os.environ if environ is None else environ
        self.address = address if address is not None else (self.environ.get("TRAINER_NODE_ADDRESS") or "").strip() or None
        self._store_factory = store or (lambda: default_store(self.environ))
        self._own_factory = own or (lambda: studio_cluster.own_info(version=_version()))
        self._own = None
        self.health = sched.Health(clock)
        self.owner = sched.Owner(client_for, clock, lease_s)
        self.guests = {}  # owner's job id -> GuestRun
        self.waiting = {}  # job id -> why it is waiting
        self._known = {}  # peer address -> its last NodeInfo
        self._peers, self._peers_mtime, self.config_problem = [], None, ""
        self._store, self._store_problem, self._store_made = None, "", False
        self.polled = threading.Event()
        self._threads = []
        self._lock = threading.RLock()
        for job in runner.jobs:  # an old runner's epoch must never match a new grant
            self.owner.leases.seed(job.id, job.epoch)
        runner.subscribe(self._on_event)

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

    def store(self):
        """The version store, or None (see `store_problem`)."""
        if not self._store_made:
            try:
                self._store, self._store_problem = self._store_factory(), ""
            except ValueError as exc:
                self._store, self._store_problem = None, str(exc)
            if self._store is None and not self._store_problem:
                self._store_problem = NO_STORE
            self._store_made = True
        return self._store

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
                    job.status, job.node = "waiting", ""
                self.runner._save()
        for run in list(self.guests.values()):
            run.ended = True

    # ---- the cluster as this node sees it ---------------------------------------------------------------------------
    def poll_once(self):
        peers = self.peers()
        if not peers:
            self.polled.set()
            return

        def ask(address):
            try:
                payload = self.client_for(address, retries=0, timeout=studio_cluster.POLL_TIMEOUT).get_json("node")
                return address, studio_cluster.info_from_payload(address, payload), ""
            except Exception as exc:  # noqa: BLE001 - unreachable, slow or nonsense all mean "did not answer"
                return address, None, f"{type(exc).__name__}: {exc}"

        with ThreadPoolExecutor(max_workers=min(8, len(peers))) as pool:
            results = list(pool.map(ask, peers))
        for address, info, error in results:
            self.health.observe(address, info is not None)
            self._known[address] = info or studio_cluster.offline(address, error)
        self.polled.set()

    def nodes(self):
        """This node first, then each listed peer as last polled."""
        own = dataclasses.replace(self.own(), busy=self.runner.running is not None or bool(self.runner.blocked))
        return [own, *(self._known.get(a) or studio_cluster.offline(a, "not polled yet") for a in self.peers())]

    # ---- owner: placing and dispatching -----------------------------------------------------------------------------
    def place(self, job):
        """Where `job` goes: a Placement. Without peers, or for a job another node owns, always here."""
        if job.owner or not self.peers():
            return sched.Placement("local")
        if self._threads and not self.polled.is_set():
            self.polled.wait(studio_cluster.POLL_TIMEOUT + 1)
        placement = sched.decide(job.kind, self.nodes(), self.health, job.params.get("pinned"), store_ready=self.store() is not None,
                                 exclude=self.owner.refused_by(job.id))
        if placement.where == "wait":
            self.waiting[job.id] = placement.reason
        else:
            self.waiting.pop(job.id, None)
        return placement

    def dispatch(self, job, address):
        """Hand `job` to the peer at `address`. On success the job is running there, under a lease; a peer that refuses it, or a job that
        cannot be packaged, leaves it waiting with the reason (and that peer is not offered it again for a while)."""
        runner = self.runner
        try:
            store = self.store()
            if store is None:
                raise sched.Refused(self.store_problem)
            body, notes = self._package(job, store)
            result = self.owner.dispatch(job.id, address, body)
        except sched.Refused as exc:
            return self._refused(job, address, str(exc))
        except Exception as exc:  # noqa: BLE001 - reading the data or publishing the input failed: the sentence is the reason
            self.owner._refused[(job.id, address)] = self.clock() + sched.REFUSAL_COOLDOWN
            return self._refused(job, address, str(exc) or type(exc).__name__)
        with runner.lock:
            job.status, job.node, job.epoch = "running", address, result.lease.epoch
            job.params = {**job.params, "cluster": notes}
            self.waiting.pop(job.id, None)
            runner._save()
        runner._emit("job", {"job": job.id, "status": "running"})
        runner._emit("cluster", {"job": job.id, "state": "dispatched" if result.confirmed else "unconfirmed", "node": address,
                                 "epoch": result.lease.epoch, "detail": result.detail})
        return True

    def _refused(self, job, address, reason):
        self.waiting[job.id] = f"{address} would not take it: {reason}"
        self.runner._emit("cluster", {"job": job.id, "state": "refused", "node": address, "detail": reason})
        return False

    def _package(self, job, store):
        """The body of `POST /cluster/jobs`: the job with its training data inside, and the store key of the version it starts from."""
        import studio_worker
        params = {k: v for k, v in job.params.items() if k not in LOCAL_ONLY}
        notes = dict(sched.attempt_state(job.params))
        fetch = dict(params.pop("fetch", None) or {})
        params.pop("cluster", None)
        if "examples" not in params:
            params["examples"] = studio_worker.JobContext(self.runner.root, self.runner.root / "runs").examples(params)
        if not fetch and params.get("mode") == "catch_up" and params.get("init_from"):
            fetch["init_from"] = store.publish({"number": 0, "name": f"start of job {job.id}"}, params["init_from"])
        params.pop("init_from", None)
        notes.setdefault("steps", int((params.get("config") or {}).get("steps") or 0))
        notes.setdefault("offset", 0)
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
        notes = dict(sched.attempt_state(job.params))
        checkpoint = body.get("checkpoint")
        if checkpoint and checkpoint.get("key") and (notes.get("checkpoint") or {}).get("key") != checkpoint["key"]:
            with self.runner.lock:
                notes["checkpoint"] = {"key": checkpoint["key"], "step": int(notes.get("offset", 0)) + int(checkpoint.get("step") or 0)}
                job.params = {**job.params, "cluster": notes}
                self.runner._save()
        for event in body.get("events") or ():
            shifted = sched.relayed(event, int(notes.get("offset", 0)))
            if shifted is not None:
                self.runner._emit("train", {"job": job_id, **shifted})
        self.runner._emit("cluster", {"job": job_id, "state": body.get("state") or "running", "node": lease.node, "epoch": epoch,
                                      "recovered": recovered})
        return lease

    def finished(self, job_id, body):
        """`POST /cluster/jobs/{id}/finished`: the runner's job ended. The result is a published version, listed here as available but
        not local."""
        epoch = int(body["epoch"])
        self.owner.end(job_id, epoch)
        job = self._remote(job_id)
        if job is None:
            raise sched.StaleEpoch(job_id, epoch)
        notes = sched.attempt_state(job.params)
        with self.runner.lock:
            job.status, job.error = body["status"], body.get("error") or ""
            if body.get("key"):
                job.params = {**job.params, "remote_result": {"key": body["key"], "node": job.node,
                                                              "step": int(notes.get("offset", 0)) + int(body.get("step") or 0)}}
            self.runner._save()
        self.runner._emit("job", {"job": job_id, "status": job.status, "error": job.error})
        self.runner.wake.set()

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

    def _reassign(self, job_id):
        """The runner has been silent for two leases, so its self-fence has fired: the job is interrupted at its last checkpoint and
        goes back to waiting with the next epoch, to be placed again (the same node when it returns, or another that fits)."""
        job = self._remote(job_id)
        if job is None:
            return
        node = job.node
        with self.runner.lock:
            params, note, finished = sched.resume_params(job.params, node)
            job.params, job.node = params, ""
            if finished:
                checkpoint = sched.attempt_state(job.params)["checkpoint"]
                job.status = "done"
                job.params["remote_result"] = {"key": checkpoint["key"], "node": node, "step": checkpoint["step"]}
            else:
                job.status = "waiting"
                job.note = f"{job.note} {note}".strip()
            self.runner._save()
        self.runner._emit("cluster", {"job": job_id, "state": "interrupted", "node": node, "detail": note})
        if finished:
            self.runner._emit("job", {"job": job_id, "status": "done", "error": ""})
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
                         "waiting_reason": self.waiting.get(job.id, ""), "checkpoint_step": (notes.get("checkpoint") or {}).get("step")})
        for run in self.guests.values():
            if not run.ended:
                rows.append({"id": run.id, "kind": run.spec["kind"], "label": run.spec.get("label", ""), "role": "runner",
                             "node": run.owner_address, "epoch": run.lease.epoch, "status": self._guest_state(run),
                             "lease": "expired" if run.guest.fence.expired() else "held", "waiting_reason": run.problem,
                             "checkpoint_step": (run.checkpoint or {}).get("step")})
        return rows

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

        def progress():
            job = next((j for j in self.runner.jobs if j.id == run.local_id), None)
            with run.lock:
                events = list(run.events[:count])
            step = max((e["step"] for e in events if isinstance(e.get("step"), int)), default=0)
            total = max((e["total"] for e in events if isinstance(e.get("total"), int)), default=0)
            return {"state": self._guest_state(run), "step": step, "total": total, "events": events, "checkpoint": run.checkpoint,
                    "waiting": bool(job and job.status != "running")}

        def acked():
            with run.lock:
                del run.events[:count]

        verdict = run.guest.renew(progress, acked)
        if verdict:
            self._stop_guest(run, verdict)

    def _publish_checkpoint(self, run):
        """Upload the job's newest checkpoint (weights only) so the owner can resume from it if this node is lost."""
        job = next((j for j in self.runner.jobs if j.id == run.local_id and j.status == "running"), None)
        store = self.store()
        if job is None or store is None or run.publishing or not job.params.get("output_dir"):
            return
        path = Path(job.params["output_dir"]) / "last.npz"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if mtime <= run.published_mtime:
            return
        run.publishing = True

        def work():
            try:
                step = checkpoint_step(path)
                key = store.publish({"number": 0, "name": f"job:{run.id}", "change": f"checkpoint at step {step}"}, path)
                run.checkpoint, run.published_mtime = {"key": key, "step": step}, mtime
            except Exception as exc:  # noqa: BLE001 - the next checkpoint tries again; the owner resumes from the last one that made it
                run.problem = str(exc) or type(exc).__name__
            finally:
                run.publishing = False
        threading.Thread(target=work, daemon=True, name=f"publish {run.id}").start()

    # ---- runner: the end of a job -----------------------------------------------------------------------------------
    def _on_event(self, kind, payload):
        run = next((r for r in self.guests.values() if payload.get("job") == r.local_id), None) if isinstance(payload, dict) else None
        if run is None or run.ended:
            return
        if kind == "train":
            run.keep({k: v for k, v in payload.items() if k != "job"})
        elif kind == "job" and payload.get("status") in TERMINAL:
            threading.Thread(target=self._conclude, args=(run, payload["status"], payload.get("error") or ""), daemon=True,
                             name=f"finish {run.id}").start()

    def _conclude(self, run, status, error):
        """Publish what the job made, then tell the owner. The lease keeps being renewed meanwhile, by `tick`."""
        key, step = None, 0
        job = next((j for j in self.runner.jobs if j.id == run.local_id), None)
        out = Path((job.params.get("output_dir") if job else "") or "")
        path = next((p for p in (out / "best.npz", out / "last.npz") if p.is_file()), None)
        if path is not None:
            try:
                step = checkpoint_step(path)
                key = self.store().publish({"number": 0, "name": f"job:{run.id}", "change": f"result at step {step}"}, path)
            except Exception as exc:  # noqa: BLE001 - the owner gets the reason instead of a result it could not use
                status, error = "failed", str(exc) or type(exc).__name__
        self._report(run, status, error, key, step)

    def _report(self, run, status, error, key=None, step=0):
        body = {"status": status, "error": error, "key": key, "step": step}
        while not run.stopped_by and not run.guest.fence.expired():
            outcome = run.guest.finish(body)
            if outcome == "retry":
                if self.runner.closed.wait(1.):
                    break
                continue
            if outcome == "stale":
                self._stop_guest(run, "stale")
            break
        run.ended = True


def _version():
    from studio_api.version import app_version
    return app_version()
