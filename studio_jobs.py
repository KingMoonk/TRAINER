"""Job handlers: (job, services, emit, stop, pause). Only pretrain and the checkpoint cool-down are real so far."""
import functools
import json
import shutil
import tempfile
from pathlib import Path

COOLDOWN_STEPS = 20


def _examples(job, services):
    p = job.params
    return p["examples"] if "examples" in p else services.examples(p)


def pretrain(job, services, emit, stop, pause):
    import studio_model
    p = job.params
    cfg = dict(p["config"])
    cfg["batch_size"] = min(cfg.get("batch_size", 8), p.get("max_batch_size", 10 ** 9))  # plan's memory cap
    out = Path(p.get("output_dir") or Path(services.workspace.run_dir) / f"queue_{job.id}")
    init = p.get("init_from") if p.get("mode") == "catch_up" else None
    job.params["output_dir"] = str(out)
    return studio_model.train(cfg, _examples(job, services), out, emit=emit, stop=stop, pause=pause, init_from=init,
                              wall_seconds=p.get("wall_seconds"), token_budget=p.get("token_budget"),
                              snapshot_every=p.get("snapshot_every", 50))  # feeds the 3D page's scrubber; newest 20 kept


def import_pretrained(job, services, emit, stop, pause):
    """Download a pretrained open model (kept under data/pretrained) and turn it into a checkpoint in the job's output folder."""
    import studio_import
    p = job.params
    root = Path(getattr(services, "root", None) or services.workspace.run_dir.parent)
    out = Path(p.get("output_dir") or studio_import.new_output_dir(services.workspace.run_dir, p["model"]))
    job.params["output_dir"] = str(out)
    studio_import.run_import(p["model"], root / "data" / "pretrained", out, emit=emit, stop=stop)


def checkpoint_now(job, services, emit, stop, pause):
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


def post_round(job, services, emit, stop, pause):
    import studio_rl
    return studio_rl.run_round(int(job.params.get("round", 1)), job, services, emit, stop, pause)


def restore_answers(job, services, emit, stop, pause):
    """Round 3 with the previous best version as the teacher (params["teacher"] defaults to the best version)."""
    import studio_rl
    return studio_rl.run_round(3, job, services, emit, stop, pause)


def quicktest(job, services, emit, stop, pause):
    import studio_quicktest
    p = job.params
    dataset = p["examples"] if "examples" in p else p.get("dataset", "stories")
    result = studio_quicktest.run(p["current_spec"], p["draft_spec"], dataset, p.get("minutes", 1), p.get("shrink", False),
                                  emit=emit, stop=stop, config=p.get("config"), services=services)
    job.params["result"] = result


def compare(job, services, emit, stop, pause):
    import studio_experiments
    p = job.params
    out = Path(p.get("output_dir") or Path(services.workspace.run_dir) / f"compare_{job.id}")
    job.params["output_dir"] = str(out)
    job.params["result"] = studio_experiments.execute(studio_experiments.ExperimentSpec(**p["experiment"]), _examples(job, services), out,
                                                      emit=emit, stop=stop, pause=pause, config=p.get("config"))


def not_built(job, services, emit, stop, pause):
    emit({"type": "info", "text": "not built yet"})


def _capped(handler):
    """The handler, run with torch's thread pool capped (studio_backend.cap_threads): a job runs on a thread of the app's process,
    so left alone torch takes every core and the Tk thread waits behind it for the GIL."""
    @functools.wraps(handler)
    def run(*args):
        from studio_backend import cap_threads
        cap_threads()
        return handler(*args)
    return run


HANDLERS = {name: _capped(handler) for name, handler in {
    "pretrain": pretrain, "checkpoint_now": checkpoint_now, "quicktest": quicktest, "compare": compare,
    "post_round": post_round, "restore_answers": restore_answers, "import_pretrained": import_pretrained}.items()}
