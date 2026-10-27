"""The Auto Run orchestrator: walks a plan (studio_autorun) over the EXISTING job queue. No runner and no worker of its own.

It enqueues the queue's own job kinds, each tagged `params["auto_run"] = <run id>` and carrying a deterministic id
(`studio_autorun.job_id`), so asking for a phase's job twice is the same job: a restart in the middle of a phase finds the job by
its id and waits for it instead of enqueueing a second one. It listens to the runner's job events (Services.watch for the fake, the
runner's own subscribers after `_record_job` for the real app, so `record_round` has already run), feeds each end to
`studio_autorun.apply_outcome`, and persists the state after every transition (`data/auto_runs/<id>/`). Every transition emits an
`auto_run` event, {"id", "status", "current", "reason"}.

Phases and the jobs they become (`JOB_KIND` in studio_autorun is the plan's view; this is the queue's):
* data: no job. A readiness check run in process: the workspace needs text a run may train on, else the run waits for the person;
* pretrain: `build_pretrain(settings, version)` makes the job, the same way POST /pretraining/start does (sizing and fitting
  included); it counts as done only once the restore_answers round a pretrain on new text leaves behind is no longer waiting or running;
* sft, rlvr, distill, dpo: `post_round` jobs for rounds 1-4, built by `studio_rl.round_job`, on the version's newest checkpoint;
* midtrain: no queue job of that name exists yet (studio_training_stages is not merged), so an enabled midtrain phase waits for the
  person (waiting_user), who skips it.

What the budget covers today: the overall deadline is checked between phases, and a pretrain phase's `hours` is capped to what is
left. Rounds 1-4 honour no wall-clock budget of their own (their steps decide how long they take).

Controls: pause (the job that runs finishes; a job that only waits is held with `Services.hold`, never the global queue pause),
resume, stop (the job is removed from the queue; checkpoints are never deleted), skip, continue {next | retry | stop | adopt}.
One run per version may be running at a time, and a run does not start while a post-training round the person added is queued for
its version.
"""
from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path

import studio_autorun as ar
from studio_core import now as clock

LOG = logging.getLogger("groundwork.autorun")
ROUND = {"sft": 1, "rlvr": 2, "distill": 3, "dpo": 4}      # the post-training round each phase runs
ACTIVE = ("running", "pausing")
FINISHED = ("done", "failed", "stopped")                    # the queue's statuses for a job that has ended
CHOICES = ("next", "retry", "stop", "adopt")


class Orchestrator:
    """`services` supplies the queue (find_job, live_jobs, enqueue, hold, remove, versions, text_sets, has_reviewed_text, adopt_round)
    and `emit`. `directory` is data/auto_runs. `build_pretrain(settings, version)` returns the pretrain Job to enqueue or raises
    ValueError with the sentence to show; without it a pretrain phase waits for the person. `spawn(fn, *args)` runs the work a runner
    event triggers (the real app passes a thread starter so the runner thread is never held up; the default runs it right there)."""

    def __init__(self, services, directory, build_pretrain=None, spawn=None, now=clock):
        self.services, self.directory, self.build_pretrain, self.now = services, Path(directory), build_pretrain, now
        self.spawn = spawn or (lambda fn, *args: fn(*args))
        self.lock = threading.RLock()
        self.runs = {}                 # id -> {"plan", "state"}: what is on disk, kept in memory
        self.inside = False            # a step is running (an event it caused only asks for another pass)
        self.again = False
        self._loaded = False

    # ---- storage
    def _folder(self, run_id):
        return self.directory / run_id

    def _load_all(self):
        if self._loaded:
            return
        self._loaded = True
        if self.directory.is_dir():
            for folder in sorted(self.directory.iterdir()):
                plan, state = ar.load_plan(folder), ar.load_state(folder)
                if plan and state and not ar.validate(plan):
                    self.runs[plan["id"]] = {"plan": plan, "state": state}

    def _get(self, run_id):
        self._load_all()
        run = self.runs.get(run_id)
        if run is None:
            raise ar.AutorunError(f"There is no Auto Run {run_id}.")
        return run

    def _put(self, run_id, state):
        """Keep `state` as the run's, write it (and its journal lines), and tell the watchers."""
        run = self.runs[run_id]
        before = run["state"]
        run["state"] = state
        ar.save_state(self._folder(run_id), state)
        self.services.emit("auto_run", {"id": run_id, "status": state["status"], "current": state["current"], "reason": state["reason"]})
        return before

    # ---- reads
    def ids(self):
        with self.lock:
            self._load_all()
            return sorted(self.runs)

    def view(self, run_id):
        """{"plan", "state", "journal"} of one run: the journal is its last 50 lines."""
        with self.lock:
            run = self._get(run_id)
            return {"plan": run["plan"], "state": run["state"], "journal": ar.read_journal(self._folder(run_id))[-50:]}

    # ---- making a run
    def create(self, plan):
        """Save `plan` as a new draft run and return its id. A plan with an empty id is given one. AutorunError lists what is wrong."""
        with self.lock:
            self._load_all()
            plan = dict(plan, id=plan.get("id") or f"ar-{uuid.uuid4().hex[:8]}")
            if plan["id"] in self.runs:
                raise ar.AutorunError(f"There is already an Auto Run {plan['id']}.")
            state = ar.new_state(plan)
            ar.save_plan(self._folder(plan["id"]), plan)
            self.runs[plan["id"]] = {"plan": plan, "state": state}
            self._put(plan["id"], state)
            return plan["id"]

    def start(self, run_id):
        with self.lock:
            run = self._get(run_id)
            self._claim(run["plan"])
            self._put(run_id, ar.start(run["plan"], run["state"], self.now()))
        self._advance()
        return self.view(run_id)

    def _claim(self, plan):
        """One run per version may be going, and the person's own post-training rounds for it must not be in the queue."""
        for other_id, other in self.runs.items():
            if other_id != plan["id"] and other["plan"]["version"] == plan["version"] and other["state"]["status"] in ACTIVE:
                raise ar.AutorunError(f"Auto Run {other['plan'].get('name') or other_id} is already running on version {plan['version']}.")
        for job in self.services.live_jobs():
            if job.kind in ("post_round", "restore_answers") and not job.params.get("auto_run") and job.params.get("version") == plan["version"]:
                raise ar.AutorunError(f"A post-training round is already queued for version {plan['version']}. Let it finish or remove it first.")

    # ---- controls
    def pause(self, run_id):
        with self.lock:
            run = self._get(run_id)
            self._put(run_id, ar.request_pause(run["state"], self.now()))
        self._advance()
        return self.view(run_id)

    def resume(self, run_id):
        with self.lock:
            run = self._get(run_id)
            if run["state"]["status"] not in ("paused", "waiting_user"):
                raise ar.AutorunError("Only a paused run can be resumed.")
            self._claim(run["plan"])
            self._put(run_id, ar.resume(run["plan"], run["state"], self.now()))
        self._advance()
        return self.view(run_id)

    def stop(self, run_id, reason=""):
        """End the run. Its queued or running jobs are removed from the queue (a running one stops at its next safe point); every
        checkpoint stays."""
        with self.lock:
            run = self._get(run_id)
            jobs = [p["job_id"] for p in run["state"]["phases"] if p["status"] in ("queued", "running") and p["job_id"]]
            self._put(run_id, ar.stop(run["state"], reason, self.now()))   # first: the removals' events then find nothing to change
            for job_id in jobs:
                self.services.remove(job_id)
        return self.view(run_id)

    def skip(self, run_id, phase=None):
        """Skip a failed, stopped, blocked or not yet started phase (the current one by default). The run stays as it is; resume it."""
        with self.lock:
            run = self._get(run_id)
            phase = phase or run["state"]["current"]
            if not phase:
                raise ar.AutorunError("There is no phase to skip.")
            self._put(run_id, ar.skip_phase(run["plan"], run["state"], phase, self.now()))
        return self.view(run_id)

    def continue_run(self, run_id, choice):
        """What the person decides when the run waits for them (paused on a failed phase, or waiting_user): `next` skips the phase and
        goes on, `retry` runs it again as the next attempt, `stop` ends the run, `adopt` first makes the checkpoint a failed round kept
        the version's own (Services.adopt_round), then skips the phase and goes on."""
        if choice not in CHOICES:
            raise ar.AutorunError(f"Choose one of {', '.join(CHOICES)}.")
        with self.lock:
            run = self._get(run_id)
            state = run["state"]
            if state["status"] not in ("paused", "waiting_user"):
                raise ar.AutorunError("This run is not waiting for you.")
            if choice == "stop":
                return self.stop(run_id)
            phase = state["current"]
            if choice == "retry":
                return self.resume(run_id)
            if not phase:
                raise ar.AutorunError("There is no phase to skip.")
            if choice == "adopt":
                if phase not in ROUND:
                    raise ar.AutorunError("Only a post-training round keeps a checkpoint to adopt.")
                self.services.adopt_round(run["plan"]["version"], ROUND[phase])   # ValueError with a sentence when there is none
            self._put(run_id, ar.skip_phase(run["plan"], state, phase, self.now()))
            return self.resume(run_id)

    # ---- events and restart
    def on_event(self, kind, payload=None):
        """A runner or services event. Cheap; the work goes through `spawn`. A "job" event moves the phase it belongs to; a "queue"
        event also catches a job the person deleted from the queue."""
        if kind not in ("job", "queue") or not self._anything_active():
            return
        self.spawn(self._advance)

    def _anything_active(self):
        return any(r["state"]["status"] in ACTIVE for r in list(self.runs.values()))

    def reconcile(self):
        """After a start of the app: for every run that was going, find where each queued or running phase's job is (the queue, or the
        history it keeps) and carry on: wait for it, record how it ended, or ask for the same job again (same id: never a second one).
        Safe to call more than once."""
        with self.lock:
            self._load_all()
            self.inside = True
            try:
                for run_id in list(self.runs):
                    run = self.runs[run_id]
                    if run["state"]["status"] not in ACTIVE:
                        continue
                    found = self.services.find_job
                    for action in ar.reconcile(run["state"], found, found):
                        self._reconciled(run_id, action)
            finally:
                self.inside = False
        self._advance()

    def _reconciled(self, run_id, action):
        """A job still in the queue is waited for, and one that ended is recorded, by the pass that follows (`_sync`, which reads the
        result from the job's params); only a job that is in neither place is asked for again, and a data check is run again."""
        if action[0] == "enqueue":
            self._enqueue(run_id, action[1], action[2])
        elif action[0] == "data":
            self._data(run_id)

    # ---- the loop
    def _advance(self):
        """Take every step the active runs can take now. Re-entrant calls (an event the step itself caused) only ask for one more pass."""
        with self.lock:
            if self.inside:
                self.again = True
                return
            self.inside = True
            try:
                self.again = True
                while self.again:
                    self.again = False
                    for run_id in list(self.runs):
                        if self.runs[run_id]["state"]["status"] in ACTIVE:
                            try:
                                self._step(run_id)
                            except Exception:  # noqa: BLE001 - one broken run must not stop the others
                                LOG.exception("auto run %s could not take its next step", run_id)
            finally:
                self.inside = False

    def _step(self, run_id):
        for _guard in range(32):
            run = self.runs[run_id]
            plan, state = run["plan"], run["state"]
            if state["status"] not in ACTIVE:
                return
            if self._sync(run_id):
                continue
            if state["status"] == "pausing" and self._park(run_id):
                continue
            action = ar.next_action(plan, state, self.now())
            if action[0] == "wait":
                return
            if action[0] == "pause":
                self._put(run_id, ar.settle_pause(state, self.now()))
            elif action[0] == "finish":
                self._put(run_id, ar.finish(state, action[1], action[2], self.now()))
            elif action[0] == "data":
                self._data(run_id)
            elif action[0] == "enqueue":
                self._enqueue(run_id, action[1], action[2])
            else:
                return

    def _sync(self, run_id):
        """Look at the job of each queued or running phase; True when the state changed."""
        run = self.runs[run_id]
        for entry in run["state"]["phases"]:
            if entry["status"] not in ("queued", "running") or not entry["job_id"]:
                continue
            job = self.services.find_job(entry["job_id"])
            kind = entry["kind"]
            if job is None:
                self._finish_phase(run_id, kind, "missing", {})
                return True
            status = job.status
            if status in FINISHED:
                if kind == "pretrain" and status == "done" and self._restoring(run["plan"]["version"]):
                    continue                    # not done until the answers are restored
                self._finish_phase(run_id, kind, status, self._result(run_id, kind, job))
                return True
            if status == "running" and entry["status"] == "queued":
                self._put(run_id, ar.mark_running(run["plan"], run["state"], kind, self.now()))
                return True
        return False

    def _restoring(self, version):
        """Whether a restore_answers round for `version` is waiting or running (the one a pretrain on new text leaves behind)."""
        return any(j.kind == "restore_answers" and j.params.get("version") in (None, version) for j in self.services.live_jobs())

    def _result(self, run_id, kind, job):
        result = {}
        if kind in ROUND:
            result = dict(job.params.get("result") or {})
            if job.status == "done" and not result:
                result = {"passed": False, "error": f"Round {ROUND[kind]} left no result."}
        elif kind == "pretrain" and job.status == "done":
            found = self._version(self.runs[run_id]["plan"]["version"])
            result = {"checkpoint": str(found.best_checkpoint)} if found is not None and found.best_checkpoint else {}
        if getattr(job, "error", ""):
            result.setdefault("error", job.error)
        return result

    def _finish_phase(self, run_id, kind, status, result):
        run = self.runs[run_id]
        self._put(run_id, ar.apply_outcome(run["plan"], run["state"], kind, status, result, self.now()))

    def _park(self, run_id):
        """A pause is waiting for a phase whose job has not started: hold the job and park the phase. True when one was parked."""
        run = self.runs[run_id]
        for entry in run["state"]["phases"]:
            if entry["status"] == "queued" and entry["job_id"]:
                job = self.services.find_job(entry["job_id"])
                if job is not None and job.status in ("waiting", "held"):
                    self.services.hold(entry["job_id"], True)
                    self._put(run_id, ar.park(run["plan"], run["state"], entry["kind"], self.now()))
                    return True
        return False

    def _block(self, run_id, kind, reason, action):
        run = self.runs[run_id]
        began = ar.begin(run["plan"], run["state"], action, self.now())
        self._put(run_id, began)
        self._finish_phase(run_id, kind, "blocked", {"reason": reason})

    def _version(self, number):
        return next((v for v in self.services.versions() if v.number == number), None)

    # ---- doing a phase
    def _data(self, run_id):
        run = self.runs[run_id]
        action = ("data", "data")
        reason = self._no_text()
        if reason:
            return self._block(run_id, "data", reason, action)
        self._put(run_id, ar.begin(run["plan"], run["state"], action, self.now()))
        self._finish_phase(run_id, "data", "done", {})

    def _no_text(self):
        """Why no run could train on the workspace's text yet, or ""."""
        try:
            from studio_ia import pretraining_plan as plan
            ids = [s["id"] for s in self.services.text_sets() if not plan.held_out(s["id"])]
            if ids and self.services.has_reviewed_text(ids):
                return ""
        except NotImplementedError:
            return ""
        return "There is no text to train on yet. Add text in Your data and mark it Good."

    def _enqueue(self, run_id, kind, job_id):
        """Ask the queue for the phase's job (by its deterministic id) and record it. A phase that cannot be built waits for the person."""
        run = self.runs[run_id]
        plan = run["plan"]
        action = ("enqueue", kind, job_id)
        there = self.services.find_job(job_id)
        if there is not None:
            if there.status == "held":
                self.services.hold(job_id, False)       # a parked job (pause) goes on
        else:
            try:
                job = self._build(run_id, kind, job_id)
            except (ValueError, RuntimeError, OSError) as exc:   # (the plan's reason, or the size of the text could not be worked out)
                return self._block(run_id, kind, str(exc) or f"The {kind} phase could not be set up.", action)
            self.services.enqueue(job)
        self._put(run_id, ar.begin(plan, self.runs[run_id]["state"], action, self.now()))   # (a phase already queued is only re-recorded)

    def _build(self, run_id, kind, job_id):
        run = self.runs[run_id]
        plan, state = run["plan"], run["state"]
        if kind not in ar.JOB_KIND or kind == "midtrain":
            raise ValueError("Mid-training is not available yet. Skip this phase." if kind == "midtrain" else f"The {kind} phase has no job.")
        version = self._version(plan["version"])
        if version is None:
            raise ValueError(f"There is no version {plan['version']}.")
        settings = ar.phase_settings(plan, state, kind, self.now())
        if kind == "pretrain":
            build = self.build_pretrain or getattr(self.services, "build_pretrain", None)   # the API app sets the services' one
            if build is None:
                raise ValueError("This app cannot start a pretraining run from an Auto Run.")
            job = build(settings, version.number)
        else:
            if not version.best_checkpoint:
                raise ValueError(f"Version {version.number} has no trained weights yet. Pretrain it first, or skip this phase.")
            import studio_rl
            number = ROUND[kind]
            if number == 3:
                settings.setdefault("teacher", None)
                settings.setdefault("teacher_choice", "previous best" if settings.get("teacher") else "correction")
            job = studio_rl.round_job(number, version, settings)
        job.id = job_id
        job.params = {**job.params, "auto_run": plan["id"], "auto_phase": kind}
        return job
