"""The Auto Run orchestrator: walks a plan (studio_autorun) over the EXISTING job queue. No runner and no worker of its own.

It enqueues the queue's own job kinds, each tagged `params["auto_run"] = <run id>` and carrying a deterministic id
(`studio_autorun.job_id`), so asking for a phase's job twice is the same job: a restart in the middle of a phase finds the job by
its id and waits for it instead of enqueueing a second one. It listens to the runner's job events (Services.watch for the fake, the
runner's own subscribers after `_record_job` for the real app, so `record_round` has already run), feeds each end to
`studio_autorun.apply_outcome`, and persists the state after every transition (`data/auto_runs/<id>/`). Every transition emits an
`auto_run` event, {"id", "status", "current", "reason"}.

Phases and the jobs they become (`JOB_KIND` in studio_autorun is the plan's view; this is the queue's):
* data: no job. A readiness check run in process, for what the plan's own later phases train on: text a run may train on for
  pretrain and midtrain, conversations marked Good for sft. When something is missing the run waits for the person with a sentence
  naming it and the phase that needs it;
* pretrain: `build_pretrain(settings, version)` makes the job, the same way POST /pretraining/start does (sizing and fitting
  included); it counts as done only once the restore_answers round a pretrain on new text leaves behind is no longer waiting or running;
* sft, rlvr, distill, dpo: `post_round` jobs for rounds 1-4, built by `studio_rl.round_job`, on the version's newest checkpoint;
* rlhf: the `rlhf_loop` job (`Services.loop_job`, studio_loop) on the sft phase's checkpoint when sft passed, else the version's best weights.
  Its loop id is `<run>-rlhf`, so a retry carries on from the loop's folder. The phase's settings are the loop's config (studio_loop.DEFAULTS:
  max_rounds, patience, card_patience, ...); the loop gates every round on the report card as well and stops when the card stops
  improving. When it kept a round, the phase asks `Services.adopt_loop` (the gate on all the held-back data) and passes only when that passes;
  a loop that kept nothing passes without a checkpoint. While it runs, the run's view shows its rounds so far in the phase's result;
* midtrain: the `midtrain` stage job (what POST /post-training/stages queues) on the version's best weights; the plan's `steps`
  setting becomes `params["config"]["steps"]`. Its result (ok, checkpoint, checks) lands in the job's params like a round's; only a
  passing stage becomes the version's best (Services.record_stage), and a failing one fails the phase by the plan's on_fail policy.

What the budget covers today: the overall deadline is checked between phases, and a pretrain phase's `hours` is capped to what is
left. Rounds 1-4 honour no wall-clock budget of their own (their steps decide how long they take).

Controls: pause (the job that runs finishes; a job that only waits is held with `Services.hold`, never the global queue pause),
resume, stop (the job is removed from the queue; checkpoints are never deleted), skip, continue {next | retry | stop | adopt}.
One run per version may be running at a time, and a run does not start while a post-training round the person added is queued for
its version.
"""
from __future__ import annotations

import copy
import logging
import threading
import uuid
from pathlib import Path

import studio_autorun as ar
import studio_autotune as tune
from studio_core import now as clock

LOG = logging.getLogger("groundwork.autorun")
ROUND = {"sft": 1, "rlvr": 2, "distill": 3, "dpo": 4}      # the post-training round each phase runs
ACTIVE = ("running", "pausing")
FINISHED = ("done", "failed", "stopped")                    # the queue's statuses for a job that has ended
CHOICES = ("next", "retry", "stop", "adopt")


class PlanLocked(ar.AutorunError):
    """The plan cannot be changed in the run's state (a conflict, not a bad plan)."""


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
            return {"plan": run["plan"], "state": self._live(run), "journal": ar.read_journal(self._folder(run_id))[-50:]}

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

    def replace_plan(self, run_id, plan):
        """Replace a draft or paused run's plan and return its view. A draft may be changed freely (its phases are made again from the
        plan). A paused run keeps its id, version, budget (the deadline is already set) and its phases in order; a phase that has
        passed, has a job (queued, running or parked by the pause), or comes before the current one is kept as it was, and the others (their settings, policies and
        enabled) may change. PlanLocked when the run is in any other state; AutorunError lists what is wrong with the plan."""
        with self.lock:
            run = self._get(run_id)
            old, state = run["plan"], run["state"]
            if state["status"] not in ("draft", "paused"):
                raise PlanLocked("The plan can only be changed while the run is a draft or paused.")
            plan = dict(plan, id=plan.get("id") or run_id)
            problems = ar.validate(plan)
            if problems:
                raise ar.AutorunError(" ".join(problems))
            if plan["id"] != run_id:
                raise ar.AutorunError("The plan id cannot change.")
            if state["status"] == "draft":
                state = ar.new_state(plan)
            else:
                state = self._edited_state(old, plan, state)
            ar.save_plan(self._folder(run_id), plan)
            run["plan"] = plan
            self._put(run_id, state)
        return self.view(run_id)

    def _edited_state(self, old, plan, state):
        if plan["version"] != old["version"] or plan["budget"] != old["budget"]:
            raise ar.AutorunError("The version and the budget cannot change once the run has started.")
        if [p["kind"] for p in plan["phases"]] != [p["kind"] for p in old["phases"]]:
            raise ar.AutorunError("A started run keeps its phases and their order; switch a phase on or off instead.")
        first = next((i for i, p in enumerate(old["phases"]) if p["kind"] == state["current"]), len(old["phases"]))
        state = copy.deepcopy(state)
        for i, (was, now, entry) in enumerate(zip(old["phases"], plan["phases"], state["phases"])):
            if was == now:
                continue
            if i < first or entry["status"] in ("queued", "running", "passed") or entry["job_id"]:   # (a parked phase keeps its held job)
                raise ar.AutorunError(f"The {was['kind']} phase can no longer be changed: only the current phase and the ones after it can, "
                                      "and not one that has passed or has a job.")
            if was["enabled"] != now["enabled"] and entry["status"] in ("pending", "skipped"):
                entry["status"] = "pending" if now["enabled"] else "skipped"
        phase = next((p for p in state["phases"] if p["status"] in ("pending", "queued", "running", "blocked")), None)
        state["current"] = phase["kind"] if phase else None
        ar._event(state, "plan_changed", self.now())
        return state

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
            if choice == "adopt" and phase == "rlhf":
                adopted = self.services.adopt_loop(self._loop_id(run["plan"]))
                if not adopted.get("ok"):
                    raise ar.AutorunError(adopted.get("verdict") or "The loop's best did not pass the check.")
            elif choice == "adopt":
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
                self._put(run_id, ar.finish(state, action[1], action[2], self.now(), plan))
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
        if kind == "rlhf":
            result = dict(job.params.get("result") or {})
            if job.status == "done" and not result:
                result = {"passed": False, "error": "The RLHF loop left no result."}
            elif job.status == "done" and result.get("ok") and result.get("checkpoint") and result.get("loop_id"):
                adopted = self.services.adopt_loop(result["loop_id"])    # the gate on all the held-back data, then the version's own
                result.update(ok=bool(adopted.get("ok")), verdict=adopted.get("verdict") or result.get("verdict", ""))
                if not adopted.get("ok"):
                    result.pop("checkpoint", None)
        elif kind in ROUND or kind == "midtrain":
            result = dict(job.params.get("result") or {})
            if job.status == "done" and not result:
                what = f"Round {ROUND[kind]}" if kind in ROUND else "Mid-training"
                result = {"passed": False, "error": f"{what} left no result."}
        elif kind == "pretrain" and job.status == "done":
            found = self._version(self.runs[run_id]["plan"]["version"])
            result = {"checkpoint": str(found.best_checkpoint)} if found is not None and found.best_checkpoint else {}
            result.update(self._curve(found))
        if getattr(job, "error", ""):
            result.setdefault("error", job.error)
        return result

    def _curve(self, version):
        """{"curve": [[tokens read, validation loss]...], "curve_report": {...}} from the pretrain run's own validation points
        (Services.loss_evidence), so the controller and the journal read the same evidence; {} when the run left none to fit."""
        try:
            found = self.services.loss_evidence(version) if version is not None else None
        except (NotImplementedError, OSError, ValueError):
            return {}
        points = [[float(x), float(y)] for x, y in (found or {}).get("points", [])][-tune.MAX_POINTS:]
        if not points:
            return {}
        report = tune.curve_report(points)
        return {"curve": points, **({"curve_report": report} if report else {})}

    def _finish_phase(self, run_id, kind, status, result):
        run = self.runs[run_id]
        state = tune.record_attempt(run["plan"], run["state"], kind, status, result, self.now())   # (what the controller reads of it later)
        self._put(run_id, ar.apply_outcome(run["plan"], state, kind, status, result, self.now()))

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
        reason = self._missing_data({p["kind"] for p in run["plan"]["phases"] if p["enabled"]})
        if reason:
            return self._block(run_id, "data", reason, action)
        self._put(run_id, ar.begin(run["plan"], run["state"], action, self.now()))
        self._finish_phase(run_id, "data", "done", {})

    def _missing_data(self, kinds):
        """Why the plan's enabled phases cannot train on the workspace's data yet, or "": text a run may train on for pretrain and
        midtrain, conversations marked Good for sft. Every missing piece is named with the phase that needs it."""
        problems = []
        try:
            if kinds & {"pretrain", "midtrain"}:
                from studio_ia import pretraining_plan as plan
                ids = [s["id"] for s in self.services.text_sets() if not plan.held_out(s["id"])]
                if not (ids and self.services.has_reviewed_text(ids)):
                    needs = " and ".join(k for k in ("pretrain", "midtrain") if k in kinds)
                    problems.append(f"There is no text to train on yet for {needs}. Add text in Your data and mark it Good.")
            if "sft" in kinds:
                ids = [s["id"] for s in self.services.conversation_sets()]
                if not (ids and self.services.has_conversations(ids, reviewed_only=True)):
                    problems.append("There are no conversations marked Good yet for sft. Add some in Your data and mark them Good.")
        except NotImplementedError:
            return ""
        return " ".join(problems)

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

    def _tuned(self, run_id, version, kind, settings):
        """`settings` with what the parameter controller (studio_autotune) chose for this phase from the evidence so far; the choices
        are kept in the run's state and journal, with that evidence, before the job is asked for."""
        run = self.runs[run_id]
        plan = run["plan"]
        if not tune.enabled(plan):
            return settings
        points = None
        if any(p["kind"] == "pretrain" and p["enabled"] for p in plan["phases"]):
            try:
                found = self.services.loss_evidence(version)
            except (NotImplementedError, OSError, ValueError):
                found = None
            points = found["points"] if found else None
        width = ((version.spec or {}).get("dims") or {}).get("d_model")
        choices = tune.decide(plan, run["state"], kind, points, width)
        if choices:
            self._put(run_id, tune.add_choices(run["state"], choices, self.now()))
        return tune.settings_with(settings, choices)

    @staticmethod
    def _loop_id(plan):
        return f"{plan['id']}-rlhf"

    def _loop_start(self, run_id, version):
        """The weights the RLHF loop starts from: the sft phase's checkpoint when that phase passed, else the version's best."""
        run = self.runs[run_id]
        sft = next((e for e, p in zip(run["state"]["phases"], run["plan"]["phases"]) if p["kind"] == "sft" and e["status"] == "passed"), None)
        return (sft or {}).get("checkpoint") or version.best_checkpoint

    def _live(self, run):
        """The run's state for a reader: while the rlhf phase has its job, its result shows the loop's rounds so far (read from the loop's
        folder, never written to the run's state)."""
        state = run["state"]
        entry = next((e for e in state["phases"] if e["kind"] == "rlhf" and e["status"] in ("queued", "running")), None)
        if entry is None:
            return state
        try:
            loop = next((x for x in self.services.post_loops() if x.get("id") == self._loop_id(run["plan"])), None)
        except NotImplementedError:
            loop = None
        if loop is None:
            return state
        import studio_loop
        live = studio_loop.phase_result(loop)
        for key in ("ok", "checkpoint", "stopped", "loop_id"):
            live.pop(key, None)
        live["verdict"] = loop.get("phase") or loop.get("note") or ""
        live["live"] = True
        state = copy.deepcopy(state)
        next(e for e in state["phases"] if e["kind"] == "rlhf").update(result=live)
        return state

    def _build(self, run_id, kind, job_id):
        run = self.runs[run_id]
        plan, state = run["plan"], run["state"]
        if kind not in ar.JOB_KIND:
            raise ValueError(f"The {kind} phase has no job.")
        version = self._version(plan["version"])
        if version is None:
            raise ValueError(f"There is no version {plan['version']}.")
        settings = ar.phase_settings(plan, state, kind, self.now())
        settings = self._tuned(run_id, version, kind, settings)
        if kind == "pretrain":
            build = self.build_pretrain or getattr(self.services, "build_pretrain", None)   # the API app sets the services' one
            if build is None:
                raise ValueError("This app cannot start a pretraining run from an Auto Run.")
            job = build(settings, version.number)
        elif kind == "rlhf":
            start = self._loop_start(run_id, version)
            if not start:
                raise ValueError(f"Version {version.number} has no trained weights yet. Pretrain it first, or skip this phase.")
            import studio_loop
            config = {k: v for k, v in settings.items() if k in studio_loop.DEFAULTS and v is not None}
            job = self.services.loop_job(version.number, start, config, self._loop_id(plan))
            if settings.get("judge_model"):
                job.params["judge_model"] = settings["judge_model"]
        elif kind == "midtrain":
            if not version.best_checkpoint:
                raise ValueError(f"Version {version.number} has no trained weights yet. Pretrain it first, or skip this phase.")
            params = {"version": version.number}
            if isinstance(settings.get("steps"), int) and settings["steps"] > 0:
                params["config"] = {"steps": settings["steps"]}
            from studio_queue import Job
            job = Job("midtrain", f"Version {version.number} \u00b7 mid-training", "midtrain stage", params=params)
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
