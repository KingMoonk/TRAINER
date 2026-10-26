"""Continuing a run, without Tk: the pretrain job that takes up where a version's newest run stopped.

Three ways, each a queued `pretrain` job whose label says what it continues and from which step:
- `same`: more steps on the same data, with the optimizer, sampler and random state it had (studio_jobs resumes `last.npz`);
- `new_data`: the same weights, optimizer and step on other text (`rebase`, studio_model.train);
- `grow`: a bigger design started from these weights (studio_warmstart carries what fits; new blocks start as no-ops).

The job's `params["continue_from"]` records the kind, the checkpoint and its step. studio_jobs reads it to resume (or, on a machine
that has no copy of the checkpoint, to start from the published weights for just the steps added)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KINDS = ("same", "new_data", "grow")


class ContinueError(ValueError):
    """Why a run cannot be continued that way, already written for people."""


def source_note(out, params):
    """What the finished run's folder keeps of the job that wrote it, for continuing it later: the sets it read."""
    if params.get("datasets") is None:
        return
    Path(out).mkdir(parents=True, exist_ok=True)
    chat = {"chat_share": params["chat_share"], "chat_datasets": list(params.get("chat_datasets") or [])} if params.get("chat_share") else {}
    (Path(out) / "source.json").write_text(json.dumps({"datasets": list(params["datasets"]), "version": params.get("version"), **chat}), encoding="utf-8")


def run_of(run_dir):
    """{"dir", "checkpoint", "step", "config", "spec", "datasets"} of a run folder that can be continued; ContinueError otherwise."""
    from studio_core import read_json
    from studio_jobs import _checkpoint_step
    run = Path(run_dir)
    last = run / "last.npz"
    step = _checkpoint_step(last)
    if step is None:
        raise ContinueError("This run has no full checkpoint (last.npz) to continue from: it may have been imported or pulled from another machine.")
    try:
        config = read_json(run / "config.json")
    except (OSError, ValueError):
        raise ContinueError("This run's settings (config.json) are missing, so it cannot be continued exactly.") from None
    try:
        spec = read_json(run / "spec.json")
    except (OSError, ValueError):
        spec = None
    try:
        datasets = read_json(run / "source.json").get("datasets")
    except (OSError, ValueError):
        datasets = None  # a run from before this was kept: continuing it reads every set
    return {"dir": str(run), "checkpoint": str(last), "step": step, "config": config, "spec": spec, "datasets": datasets}


def _count(steps):
    steps = int(steps or 0)
    if steps < 1:
        raise ContinueError("Say how many steps to add: at least 1.")
    return steps


def _from(run, kind, add, version):
    return {"kind": kind, "checkpoint": run["checkpoint"], "step": run["step"], "add": add, "total": run["step"] + add, "run": run["dir"],
            "version": version}


def same_job(version, run, add, previous_best=None):
    """(label, params) to run `add` more steps on the same data from the run's last checkpoint."""
    add = _count(add)
    label = f"Version {version} continued from step {run['step']:,}"
    config = {**run["config"], "steps": run["step"] + add}
    return label, {"config": config, "datasets": run["datasets"], "mode": "catch_up", "init_from": run["checkpoint"], "output_dir": None,
                   "user_text": False, "previous_best": previous_best, "version": version, "heldout_fraction": config.get("validation"),
                   "continue_from": _from(run, "same", add, version)}


def new_data_job(version, run, add, datasets, own_text=False, previous_best=None):
    """(label, params) to run `add` more steps on other text (`datasets`) from the run's last checkpoint: same weights, tokenizer,
    optimizer state and step; the held-back split and the best score start afresh on the new text."""
    add = _count(add)
    if not datasets:
        raise ContinueError("Choose the text to continue on.")
    label = f"Version {version} continued from step {run['step']:,} on new data"
    config = {**run["config"], "steps": add}  # a rebase adds this many steps to the checkpoint's
    return label, {"config": config, "datasets": list(datasets), "mode": "catch_up", "init_from": run["checkpoint"], "output_dir": None,
                   "user_text": own_text, "previous_best": previous_best, "version": version, "heldout_fraction": config.get("validation"),
                   "continue_from": _from(run, "new_data", add, version)}


def write_warm_start(run, new_spec, out):
    """Write a checkpoint of `new_spec` carrying the run's weights (studio_warmstart) to `out`; returns the warm-start report.
    ContinueError when the word list differs, because the old word table would not fit."""
    import studio_model
    import studio_warmstart
    from studio_core import replace_file
    if run["spec"] is None:
        raise ContinueError("This run's design (spec.json) is missing, so its weights cannot be carried into another design.")
    data = studio_model.load_checkpoint(run["checkpoint"], weights_only=True)
    params, report = studio_warmstart.warm_start(data["model"], run["spec"], new_spec)
    if report["from_scratch"]:
        raise ContinueError(report["reason"])
    cfg = studio_model.validate_config({"architecture": new_spec, "name": new_spec.get("name")})
    metadata = {"architecture": "groundwork_numpy_decoder", "format_version": 1, "step": 0, "best": None, "optimizer_steps": 0, "history": [],
                "config": cfg, "tokenizer": data["tokenizer"], "device": "cpu", "grown_from": run["checkpoint"], "grown_from_step": run["step"]}
    arrays = {"metadata": np.array(json.dumps(metadata))}
    arrays.update({"weight::" + key: value for key, value in params.items()})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    replace_file(temporary, out)
    return report


def grow_job(old_version, new_version, run, warm_path, steps, datasets, rows, hold, own_text=False, previous_best=None):
    """(label, params) to train version `new_version` (the bigger design) for `steps` steps from the warm-started weights at `warm_path`."""
    from studio_ia import pretraining_plan as plan
    steps = _count(steps)
    spec = new_version.spec
    cfg = plan.pretrain_config(spec, steps, rows, hold)
    label = f"Version {new_version.number} grown from version {old_version} (step {run['step']:,})"
    return label, {"config": cfg, "datasets": list(datasets) if datasets else run["datasets"], "mode": "catch_up", "init_from": str(warm_path),
                   "output_dir": None, "user_text": own_text, "previous_best": previous_best, "version": new_version.number,
                   "heldout_fraction": cfg["validation"], "continue_from": {**_from(run, "grow", steps, old_version), "total": steps,
                                                                             "grown_to": new_version.number}}
