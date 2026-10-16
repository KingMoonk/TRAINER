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
    if job.kind == "pretrain_distributed":
        import studio_cluster
        import studio_diloco
        p["output_dir"] = str(output_dir(job, run_dir))
        studio_diloco.prepare(job, services, p["output_dir"], p.get("api_port", studio_cluster.DEFAULT_PORT))
    if job.kind in ROUNDS:
        import studio_rl
        studio_rl.resolve(3 if job.kind == "restore_answers" else int(p.get("round", 1)), job, services)
    return {"root": str(services.root), "run_dir": str(run_dir)}


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
    cfg["batch_size"] = studio_memory.batch_cap(cfg, share)  # the plan's share of the device's memory
    if cfg["batch_size"] < declared:
        emit({"type": "info", "text": f"A batch of {declared} does not fit the {share:.0%} of memory the plan allows this job: "
                                      f"training with batch size {cfg['batch_size']}."})
    out = output_dir(job, context.run_dir)
    init = p.get("init_from") if p.get("mode") == "catch_up" else None
    job.params["output_dir"] = str(out)
    return studio_model.train(cfg, _examples(job, context), out, emit=emit, stop=stop, pause=pause, init_from=init,
                              wall_seconds=p.get("wall_seconds"), token_budget=p.get("token_budget"),
                              snapshot_every=p.get("snapshot_every", 50))  # feeds the 3D page's scrubber; newest 20 kept


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
        examples = [json.loads(line) for line in (run / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if line]
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
