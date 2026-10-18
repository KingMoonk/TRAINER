"""Job handlers: (job, context, emit, stop, pause), run in a worker process (studio_worker). The context is a
studio_worker.JobContext: root, run_dir, examples(params) and post_rows(kind). Anything else a job needs from the app, such as
which checkpoint a round starts from, prepare() puts in job.params before the worker starts."""
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
RESUME_PARAMS = ("resume", "resume_steps", "resume_base", "resume_local")


def continue_here(job):
    """A pretrain job taken again on this machine finds the folder of its earlier attempt (the app stopped or crashed mid-run), which
    training refuses to write into. Move that folder aside, and when it holds this job's `last.npz` with its `dataset.jsonl` set
    `params["resume"]` so `pretrain` continues from it with the optimizer, sampler and RNG (`resume_local` lets it fall back to a
    fresh start if training still refuses the file). Otherwise start fresh and say why on the job's note. A job that has never run
    here has no folder and is left alone."""
    p = job.params
    for key in RESUME_PARAMS:
        p.pop(key, None)
    out = Path(p["output_dir"])
    if not any((out / name).exists() for name in FOLDER_FILES):
        return
    aside = out.with_name(f"{out.name}-before-restart")
    for n in range(2, 1000):
        if not aside.exists():
            break
        aside = out.with_name(f"{out.name}-before-restart-{n}")
    shutil.move(str(out), str(aside))
    steps = int((p.get("config") or {}).get("steps") or 0)
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


def pretrain(job, context, emit, stop, pause):
    import studio_memory
    import studio_model
    p = job.params
    cfg = dict(p["config"])
    share = p.get("memory_share", 1.)
    declared = cfg.get("batch_size", studio_model.DEFAULT_CONFIG["batch_size"])
    offload = studio_memory.offload_wanted(studio_model.validate_config(cfg), share)  # CUDA only, and only when the state cannot fit
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
    if p.get("resume"):
        resumed = _resume(job, cfg, out, init, options)
        if resumed is not None:
            return resumed
    return studio_model.train(cfg, _examples(job, context), out, init_from=init, **options)


def _resume(job, cfg, out, init, options):
    """Continue from `params["resume"]`, a full checkpoint (`last.npz`: weights, optimizer, sampler and RNG) this node wrote, on the data
    it was written with (the `dataset.jsonl` beside it, so a changed dataset cannot refuse it) and for the `resume_steps` its attempt
    was planned for. Emits "resumed" with `resume_base`, the steps done before that attempt, once training has accepted the checkpoint,
    so the owner counts progress from there. When training refuses it before the first step and the job has published weights to start
    from (`init`), emits "resume_refused" and returns None: the caller trains from those with a fresh optimizer. Without them the
    refusal is the job's failure."""
    import studio_model
    path = Path(job.params["resume"])
    emit, started = options["emit"], []

    def relay(event):
        if not started:
            started.append(True)
            said = {"type": "resumed", "base": int(job.params.get("resume_base") or 0), "step": studio_model.load_checkpoint(path)["step"]}
            if job.params.get("resume_local"):  # this queue shows the progress itself, from the checkpoint's step
                said["total"] = int(job.params.get("resume_steps") or cfg["steps"])
            emit(said)
        emit(event)
    try:
        examples = [json.loads(line) for line in (path.parent / "dataset.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
        return studio_model.train(dict(cfg, steps=int(job.params.get("resume_steps") or cfg["steps"])), examples, out, resume=path,
                                  **{**options, "emit": relay})
    except (OSError, ValueError, KeyError) as exc:
        if started or (init is None and not job.params.get("resume_local")):  # (a local restart can always start fresh instead)
            raise
        emit({"type": "resume_refused", "reason": str(exc) or type(exc).__name__})
        return None


def _member(job, context, emit, stop, name):
    """Run a DiLoCo member (studio_diloco) against params["coordinator_url"]; it keeps its own training folder next to the job's."""
    import socket
    import studio_diloco
    from studio_peer import PeerClient
    p = job.params
    folder = Path(p["output_dir"]) / "member" if p.get("output_dir") else Path(context.run_dir) / f"{name}_{job.id}"
    studio_diloco.run_member(PeerClient(p["coordinator_url"]), p["run"], p.get("node") or socket.gethostname(), int(p.get("epoch", 0)),
                             context.root, folder, emit, stop, **p.get("member_options", {}))


def pretrain_distributed(job, context, emit, stop, pause):
    """The owner's job: the coordinator already runs in the app (prepare); this process is its local member."""
    _member(job, context, emit, stop, "diloco")


def diloco_member(job, context, emit, stop, pause):
    """A member for a run whose coordinator is on another node: params coordinator_url, run, node and epoch."""
    job.params.setdefault("output_dir", str(Path(context.run_dir) / f"diloco_member_{job.id}"))
    _member(job, context, emit, stop, "diloco_member")


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


def not_built(job, context, emit, stop, pause):
    emit({"type": "info", "text": "not built yet"})


HANDLERS = {"pretrain": pretrain, "checkpoint_now": checkpoint_now, "quicktest": quicktest, "compare": compare,
            "post_round": post_round, "restore_answers": restore_answers, "import_pretrained": import_pretrained,
            "pretrain_distributed": pretrain_distributed, "diloco_member": diloco_member}
