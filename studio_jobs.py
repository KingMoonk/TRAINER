"""Job handlers: (job, context, emit, stop, pause), run in a worker process (studio_worker). The context is a
studio_worker.JobContext: root, run_dir, examples(params) and post_rows(kind). Anything else a job needs from the app, such as
which checkpoint a round starts from, prepare() puts in job.params before the worker starts."""
import errno
import json
import shutil
import tempfile
from pathlib import Path

COOLDOWN_STEPS = 20
ROUNDS = ("post_round", "restore_answers")


def output_dir(job, run_dir):
    """Where a job writes, unless its params already say: pages show the folder while the job runs, so the app picks it first."""
    p = job.params
    if p.get("output_dir"):
        return Path(p["output_dir"])
    if job.kind == "import_pretrained":
        import studio_import
        return studio_import.new_output_dir(run_dir, p["model"])
    return Path(run_dir) / f"{'compare' if job.kind == 'compare' else 'queue'}_{job.id}"


def prepare(job, services):
    """App side, on the runner thread, just before the job's worker starts: resolve what the handler cannot ask the app for from
    its own process, and return the handler's context. A failure here fails the job with its sentence."""
    p = job.params
    run_dir = Path(services.workspace.run_dir)
    p.pop("result", None)  # a run cut short by a restart starts afresh; an old result must not be recorded again
    if job.kind in ("pretrain", "compare", "import_pretrained") + ROUNDS:
        p["output_dir"] = str(output_dir(job, run_dir))
    if job.kind == "pretrain" and not getattr(job, "owner", ""):
        continue_here(job)
    if job.kind == "pretrain_distributed":
        import studio_cluster
        import studio_diloco
        p["output_dir"] = str(output_dir(job, run_dir))
        studio_diloco.prepare(job, services, p["output_dir"], p.get("api_port", studio_cluster.DEFAULT_PORT))
    if job.kind in ROUNDS:
        import studio_rl
        studio_rl.resolve(studio_rl.number_of(job), job, services)
    return {"root": str(services.root), "run_dir": str(run_dir)}


FOLDER_FILES = ("config.json", "dataset.jsonl", "last.npz", "best.npz")  # what studio_model.train refuses to overwrite
RESUME_PARAMS = ("resume", "resume_steps", "resume_base", "resume_local", "resume_rebase")
USER_CONTINUES = ("same", "new_data")  # `params["continue_from"]["kind"]` that resume a checkpoint (studio_ia.continue_run)


def continue_here(job):
    """A pretrain job taken again on this machine finds the folder of its earlier attempt (the app stopped or crashed mid-run), which
    training refuses to write into. Move that folder aside, and when it holds this job's `last.npz` with its `dataset.jsonl` set
    `params["resume"]` so `pretrain` continues from it with the optimizer, sampler and RNG (`resume_local` lets it fall back to a
    fresh start if training still refuses the file). Otherwise start fresh and say why on the job's note. A job that has never run
    here has no folder and is left alone, unless the person asked to continue a run (`params["continue_from"]`): then its first start
    resumes that run's `last.npz` (`new_data`: rebased on the job's own text)."""
    p = job.params
    for key in RESUME_PARAMS:
        p.pop(key, None)
    out = Path(p["output_dir"])
    wanted = p.get("continue_from") or {}
    if not any((out / name).exists() for name in FOLDER_FILES):
        if wanted.get("kind") in USER_CONTINUES:
            p.update(resume=wanted["checkpoint"], resume_steps=wanted["total"], resume_base=0)
            if wanted["kind"] == "new_data":
                p["resume_rebase"] = True
        return
    aside = out.with_name(f"{out.name}-before-restart")
    for n in range(2, 1000):
        if not aside.exists():
            break
        aside = out.with_name(f"{out.name}-before-restart-{n}")
    shutil.move(str(out), str(aside))
    steps = int(wanted.get("total") or (p.get("config") or {}).get("steps") or 0)  # (a rebased job's config holds only the steps it adds)
    step, reason = _checkpoint_step(aside / "last.npz"), ""
    if step is None:
        reason = "it had not saved a usable checkpoint"
    elif not (aside / "dataset.jsonl").is_file():
        reason = "the training data it used was not saved beside its checkpoint"
    elif steps and step >= steps:
        reason = f"its checkpoint (step {step}) is not before the last planned step"
    if reason:
        job.note = f"{job.note} The app stopped while this ran; it starts again from step 0 because {reason}.".strip()
        return
    p.update(resume=str(aside / "last.npz"), resume_steps=steps or None, resume_base=0, resume_local=True)


def _checkpoint_step(path):
    """The step a checkpoint file was saved at, from its metadata alone (no weights read); None when it is missing or not readable."""
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
        return int(metadata["step"]) if metadata.get("architecture") == "groundwork_numpy_decoder" else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _examples(job, context):
    p = job.params
    return p["examples"] if "examples" in p else context.examples(p)


def _example_source(job, context):
    """The job's text as an iterator when the context can stream it (the worker's can: a text of billions of characters is read a row at
    a time, studio_corpus), else what _examples gives."""
    p = job.params
    if "examples" not in p and hasattr(context, "example_stream"):
        return context.example_stream(p)
    return _examples(job, context)


def pretrain(job, context, emit, stop, pause):
    import studio_memory
    import studio_model
    p = job.params
    cfg = dict(p["config"])
    share = p.get("memory_share", 1.)
    declared = cfg.get("batch_size", studio_model.DEFAULT_CONFIG["batch_size"])
    # the queue's fit (studio_fit) may have put the optimizer on the host already; CUDA only, and otherwise only when the state cannot fit
    offload = bool(p.get("offload_optimizer")) or studio_memory.offload_wanted(studio_model.validate_config(cfg), share)
    cfg["batch_size"] = studio_memory.batch_cap(cfg, share, offload=offload)  # the plan's share of the device's memory
    if cfg["batch_size"] < declared:
        emit({"type": "info", "text": f"A batch of {declared} does not fit the {share:.0%} of memory the plan allows this job: "
                                      f"training with batch size {cfg['batch_size']}."})
    out = output_dir(job, context.run_dir)
    init = p.get("init_from") if p.get("mode") == "catch_up" else None
    job.params["output_dir"] = str(out)
    options = dict(emit=emit, stop=stop, pause=pause, wall_seconds=p.get("wall_seconds"), token_budget=p.get("token_budget"),
                   snapshot_every=p.get("snapshot_every", 50),  # feeds the 3D page's scrubber; newest 20 kept
                   offload_optimizer=offload)
    options, watcher = _watched(job, options, stop)
    if p.get("datasets") is not None:
        import studio_ia.continue_run as continuing
        continuing.source_note(out, p)  # what a later "continue this run" needs to know of the text
    if p.get("resume"):
        resumed = _resume(job, cfg, out, init, options, context)
        if resumed is not None:
            return _plateau_end(job, out, watcher, resumed)
    return _plateau_end(job, out, watcher, studio_model.train(_weights_only_steps(job, cfg), _example_source(job, context), out, init_from=init, **options))


def _weights_only_steps(job, cfg):
    """The config for a start from published weights (no checkpoint to resume): a continued run adds only its `add` steps there,
    whatever total its config names for the resume."""
    wanted = job.params.get("continue_from") or {}
    return dict(cfg, steps=int(wanted["add"])) if wanted.get("kind") in USER_CONTINUES else cfg


def _watched(job, options, stop):
    """`options` for studio_model.train with the plateau rule (`params["plateau"]`, studio_plateau) watching the events it emits and
    ending the stage through the stop event it polls. Returns (options, the watcher or None when the job has no rule)."""
    rule = job.params.get("plateau")
    if not rule:
        return options, None
    import studio_plateau
    watcher = studio_plateau.Plateau(rule, use_history=not job.params.get("resume_rebase"))
    watcher.flag = studio_plateau.Flagged(stop, watcher)
    emit = watcher.emit = options["emit"]

    def relay(event):
        emit(event)  # the metric is reported (and its checkpoints saved) before the rule can end the stage
        watcher.feed(event)
    return {**options, "emit": relay, "stop": watcher.flag}, watcher


def _plateau_end(job, out, watcher, record):
    """When the plateau rule ended the stage (not a stop the person asked for): say why on the job (`params["ended"]`), in the run's
    own `run.json` and as an info event, and mark the run completed, not stopped. The best checkpoint was kept by training."""
    if watcher is None or watcher.hit is None or watcher.flag.user_stopped():
        return record
    import studio_plateau
    from studio_core import atomic_json, read_json
    ended = {"reason": "plateau", "text": studio_plateau.reason_text(watcher.hit), **watcher.hit}
    job.params["ended"] = ended
    watcher.emit({"type": "info", "text": ended["text"]})
    if isinstance(record, dict):
        record.update(status="completed", budget_reached="plateau", ended=ended)
        path = Path(out) / "run.json"
        try:
            atomic_json(path, {**read_json(path), **{k: record[k] for k in ("status", "budget_reached", "ended")}})
        except (OSError, ValueError):
            pass  # the run page then lacks the reason; the job still has it
    return record


DISK_ERRORS = (errno.ENOSPC, errno.EIO, errno.EDQUOT)  # a resume that fails like this fails the job; it is no refusal


def _resume(job, cfg, out, init, options, context=None):
    """Continue from `params["resume"]`, a full checkpoint (`last.npz`: weights, optimizer, sampler and RNG) this node wrote, on the data
    it was written with (the `dataset.jsonl` beside it, so a changed dataset cannot refuse it) and for the `resume_steps` its attempt
    was planned for. Emits "resumed" with `resume_base`, the steps done before that attempt, once training has accepted the checkpoint,
    so the owner counts progress from there. When training refuses it before the first step and the job has published weights to start
    from (`init`), emits "resume_refused" and returns None: the caller trains from those with a fresh optimizer. Without them the
    refusal is the job's failure.

    A run the person asked to continue (`params["continue_from"]`, studio_ia.continue_run) resumes the same way, but says so in plain
    info lines: the "resumed" and "resume_refused" events make the queue write "the app stopped while this ran". With `resume_rebase`
    the data is the job's own text and training rebases the checkpoint onto it, adding `cfg["steps"]` steps."""
    import studio_dataset_snapshot
    import studio_model
    path = Path(job.params["resume"])
    emit, started = options["emit"], []
    wanted = job.params.get("continue_from") or {}
    asked = wanted.get("kind") in USER_CONTINUES and str(path) == str(wanted.get("checkpoint"))  # (not a restart's or a lost node's own file)
    rebase = bool(job.params.get("resume_rebase"))

    def relay(event):
        if not started:
            started.append(True)
            saved = studio_model.load_checkpoint(path)
            step = saved["step"]
            if asked:
                emit({"type": "info", "text": f"Continuing from step {step:,} with the optimizer state it had" + (", on new text." if rebase else ".")})
            else:
                said = {"type": "resumed", "base": int(job.params.get("resume_base") or 0), "step": step,
                        "earlier_tokens": saved.get("trained_tokens") or 0, "earlier_seconds": saved.get("elapsed_seconds") or 0.}
                if job.params.get("resume_local"):  # this queue shows the progress itself, from the checkpoint's step
                    said["total"] = int(job.params.get("resume_steps") or cfg["steps"])
                emit(said)
        emit(event)
    try:
        if rebase:
            examples, steps = _example_source(job, context), cfg["steps"]
        else:
            examples = studio_dataset_snapshot.Snapshot(path.parent / "dataset.jsonl")  # one row at a time, read in place: a big snapshot is never held or copied
            steps = int(job.params.get("resume_steps") or cfg["steps"])
        return studio_model.train(dict(cfg, steps=steps), examples, out, resume=path, rebase=rebase, **{**options, "emit": relay})
    except (OSError, ValueError, KeyError) as exc:
        stopper = options.get("stop")
        if isinstance(exc, InterruptedError) or (stopper is not None and stopper.is_set()) or getattr(exc, "errno", None) in DISK_ERRORS:
            raise  # a Stop or a full or failing disk is not an incompatible checkpoint: never train fresh from it
        if started or (init is None and not job.params.get("resume_local")):  # (a local restart can always start fresh instead)
            raise
        if asked:
            emit({"type": "info", "text": f"The full checkpoint could not be continued ({str(exc) or type(exc).__name__}): training "
                                          f"{wanted['add']:,} more steps from its weights with a fresh optimizer."})
        else:
            emit({"type": "resume_refused", "reason": str(exc) or type(exc).__name__})
        return None


def _member(job, context, emit, stop, pause, name):
    """Run a DiLoCo member (studio_diloco) against params["coordinator_url"]; it keeps its own training folder next to the job's.
    It lives by this machine's queue plan like any job: `pause` is the plan's pause (battery, hours, idle) and params["memory_share"]
    the share of memory the plan gives it."""
    import socket
    import studio_diloco
    from studio_peer import PeerClient
    p = job.params
    node = p.get("node") or socket.gethostname()
    fit = p.get("fit") or (p.get("member_fit") or {}).get(node)  # this member's own batch, accumulation and optimizer placement (studio_fit.fit_members), from the run's coordinator
    folder = Path(p["output_dir"]) / "member" if p.get("output_dir") else Path(context.run_dir) / f"{name}_{job.id}"
    studio_diloco.run_member(PeerClient(p["coordinator_url"]), p["run"], node, int(p.get("epoch", 0)),
                             context.root, folder, emit, stop, **{"locate": studio_diloco.file_locator(context.root), "pause": pause,
                                                                    "memory_share": p.get("memory_share"), "fit": fit, **p.get("member_options", {})})


def pretrain_distributed(job, context, emit, stop, pause):
    """The owner's job: the coordinator already runs in the app (prepare); this process is its local member."""
    _member(job, context, emit, stop, pause, "diloco")


def diloco_member(job, context, emit, stop, pause):
    """A member for a run whose coordinator is on another node: params coordinator_url, run, node and epoch."""
    job.params.setdefault("output_dir", str(Path(context.run_dir) / f"diloco_member_{job.id}"))
    _member(job, context, emit, stop, pause, "diloco_member")


def import_pretrained(job, context, emit, stop, pause):
    """Download a pretrained open model (kept under data/pretrained) and turn it into a checkpoint in the job's output folder."""
    import studio_import
    out = output_dir(job, context.run_dir)
    job.params["output_dir"] = str(out)
    studio_import.run_import(job.params["model"], Path(context.root) / "data" / "pretrained", out, emit=emit, stop=stop)


def checkpoint_now(job, context, emit, stop, pause):
    """Short decaying-schedule run on a copy of the parent run's last checkpoint -> <run>/cooldown_<step>."""
    import studio_model
    run = Path(job.params["output_dir"])
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "last.npz"
        shutil.copyfile(run / "last.npz", copy)
        step = studio_model.load_checkpoint(copy)["step"]
        # resume forbids changing the schedule, so the decay comes from shrinking the horizon: the rate falls to
        # near zero over the next COOLDOWN_STEPS steps.
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        cfg["steps"] = step + job.params.get("cooldown_steps", COOLDOWN_STEPS)
        examples = [json.loads(line) for line in (run / "dataset.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
        return studio_model.train(cfg, examples, run / f"cooldown_{step}", emit=emit, stop=stop, pause=pause, resume=copy)


def post_round(job, context, emit, stop, pause):
    import studio_rl
    return studio_rl.run_round(int(job.params.get("round", 1)), job, context, emit, stop, pause)


def restore_answers(job, context, emit, stop, pause):
    """Round 3 with the previous best version as the teacher (prepare() fills params["teacher"] with the best version's checkpoint)."""
    import studio_rl
    return studio_rl.run_round(3, job, context, emit, stop, pause)


def rlhf_loop(job, context, emit, stop, pause):
    """The automated RLHF loop (studio_loop): rounds until a stop rule, resumable from its folder (params["output_dir"])."""
    import studio_loop
    studio_loop.run_job(job, context, emit, stop, pause)


def quicktest(job, context, emit, stop, pause):
    import studio_quicktest
    p = job.params
    dataset = p["examples"] if "examples" in p else p.get("dataset", "stories")
    result = studio_quicktest.run(p["current_spec"], p["draft_spec"], dataset, p.get("minutes", 1), p.get("shrink", False),
                                  emit=emit, stop=stop, config=p.get("config"), services=context)
    job.params["result"] = result


def compare(job, context, emit, stop, pause):
    import studio_experiments
    p = job.params
    out = output_dir(job, context.run_dir)
    job.params["output_dir"] = str(out)
    job.params["result"] = studio_experiments.execute(studio_experiments.ExperimentSpec(**p["experiment"]), _examples(job, context), out,
                                                      emit=emit, stop=stop, pause=pause, config=p.get("config"))


def mlperf(job, context, emit, stop, pause):
    """The training-speed benchmark (studio_mlperf): one task on this machine, result stored under data/mlperf."""
    import studio_mlperf
    studio_mlperf.job(job, context, emit, stop, pause)


def not_built(job, context, emit, stop, pause):
    emit({"type": "info", "text": "not built yet"})


HANDLERS = {"pretrain": pretrain, "checkpoint_now": checkpoint_now, "quicktest": quicktest, "compare": compare,
            "post_round": post_round, "restore_answers": restore_answers, "import_pretrained": import_pretrained,
            "pretrain_distributed": pretrain_distributed, "diloco_member": diloco_member, "rlhf_loop": rlhf_loop,
            "mlperf": mlperf}
