"""CUDA measurements owed by issues #237 (pinned-memory batch prefetch) and #238 (optimizer state offload to the host).

    python bench_cuda.py [--steps 200] [--rounds 3] [--warmup 10] [--seed 0] [--only prefetch|offload]

Run it once on a node with a CUDA GPU and paste the Markdown it prints into the issues. Without CUDA it prints why and exits 2.
Synthetic text only: no data/ folder, no network, nothing outside a temporary folder.

#237  The same model, data and seed trained for --steps steps with BatchPrefetcher staging batches on a side stream (the default
      on CUDA) and with the plain path (studio_model.batch_stage patched to None), --rounds times each, alternating. Step time
      is the gap between two optimizer updates with the device synchronized, so it covers the batch, the forward and backward
      passes and the update.
#238  (a) The same fitting model with offload_optimizer off and on: the step-time overhead of streaming the state.
      (b) A model whose optimizer state does not fit: a ballast tensor leaves the GPU less free memory than the state needs
      but more than the offloaded run needs, the plain run is tried (expected to run out of memory), then the real
      studio_jobs.pretrain handler trains it end to end; studio_memory.offload_wanted must choose offload by itself.

Checkpoint files are not written (np.savez_compressed is stubbed): compressing gigabytes of state is not what is measured."""
from __future__ import annotations

import argparse
import gc
import statistics
import sys
import tempfile
import time
import types
from pathlib import Path
from unittest import mock

import numpy as np
import torch

GIB = 1024 ** 3
FIT = {"width": 256, "layers": 6, "batch_size": 32}  # context 256 (scaled_config); a state of a few tens of MB: always fits
LADDER = [(512, 8), (640, 10), (768, 12), (896, 16), (1024, 16), (1024, 20), (1024, 24)]  # (width, layers) of the not-fitting candidates
MIN_STATE_NEED = 2 * GIB  # a not-fitting model needs at least this much on the device, so CUDA's own overhead is small beside it
HEADROOM = 1.3  # the offloaded run must have this much more than the estimate says it needs
SLACK = .9  # the plain run must have this much less than the estimate says it needs
BASE = {"dropout": 0., "steps": 1, "learning_rate": 1e-4, "warmup": 0, "validation": 0, "eval_every": 100000, "sample_every": 0,
        "objective": "all", "device": "cuda"}


# ---- pure helpers (unit-tested on the CPU) ----

def synthetic_examples(count=2000, seed=0):
    """Seeded pseudo-text, enough of it for hundreds of context windows; the same call always gives the same examples."""
    rng = np.random.default_rng(seed)
    words = ["".join(chr(97 + int(c)) for c in rng.integers(0, 26, size=int(rng.integers(2, 9)))) for _ in range(400)]
    return [{"text": " ".join(words[int(i)] for i in rng.integers(0, len(words), size=120))} for _ in range(count)]


def step_times(stamps, warmup):
    """Seconds per step from the clock readings taken after each optimizer update: the gaps, minus the first `warmup` of them
    (the first reading ends step 1, which has no gap before it, so it never counts)."""
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return gaps[warmup:]


def summarize(seconds):
    """{n, mean_ms, median_ms} of step times given in seconds."""
    if not seconds:
        return {"n": 0, "mean_ms": float("nan"), "median_ms": float("nan")}
    return {"n": len(seconds), "mean_ms": 1000 * statistics.fmean(seconds), "median_ms": 1000 * statistics.median(seconds)}


def ratio(slower, faster):
    """slower / faster, nan when either is missing or zero."""
    return slower / faster if slower and faster and slower == slower and faster == faster else float("nan")


def markdown_table(headers, rows):
    """A GitHub Markdown table from a header row and rows of cells (anything with a str())."""
    lines = ["| " + " | ".join(map(str, headers)) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines)


def gib(n):
    return f"{n / GIB:.2f} GiB"


def scaled_config(width, layers, **extra):
    """A validated config of a transformer `width` wide and `layers` deep. validate_config caps a plain width at 512 and layers at
    12; a ModelSpec goes to 1024 and 24, so the architecture is handed over as one."""
    import studio_model
    cfg = studio_model.validate_config(dict(BASE, width=min(width, 512), layers=min(layers, 12), heads=8, context=256, **extra))
    spec = studio_model.effective_spec(cfg)
    spec["dims"].update(d_model=width, n_layer=layers)
    spec["block"]["channel"]["config"]["d_ff"] = 4 * width
    spec["block"]["mixer"]["config"]["heads"] = max(1, width // 64)
    return studio_model.validate_config(dict(cfg, architecture=spec))


def fit_config(seed=0):
    """The model whose state always fits: both #237 and the #238 overhead comparison train it."""
    return scaled_config(FIT["width"], FIT["layers"], batch_size=FIT["batch_size"], seed=seed)


def plan_offload_case(free_bytes, ladder=LADDER, scale=scaled_config, batch_size=4):
    """The smallest model on `ladder` whose optimizer state cannot stay on a GPU with `free_bytes` free, and how much GPU memory to
    leave free so the plain run cannot fit but the offloaded one can. Sizes come from studio_memory.training_bytes (a batch of one).

    Returns {config, width, layers, params, need, need_offload, host, target_free, ballast}, or None when no model on the ladder fits both
    conditions (or the ladder's models are too small to matter). target_free sits between HEADROOM x what the offloaded run needs and
    SLACK x what the plain run needs; ballast is the memory to hold so only target_free is left."""
    import studio_memory
    import studio_model
    for width, layers in ladder:
        cfg = scale(width, layers, batch_size=batch_size)
        fixed, per_sample = studio_memory.training_bytes(cfg)
        fixed_off, _ = studio_memory.training_bytes(cfg, offload=True)
        need, need_offload = fixed + per_sample, fixed_off + per_sample
        low, high = need_offload * HEADROOM, need * SLACK
        if need < MIN_STATE_NEED or low >= high:
            continue
        target = int((low + high) / 2)
        if target > free_bytes:
            continue
        return {"config": cfg, "width": width, "layers": layers, "params": studio_model.parameter_estimate(cfg), "need": need,
                "need_offload": need_offload, "host": studio_memory.host_bytes(cfg), "target_free": target, "ballast": free_bytes - target}
    return None


class StepClock:
    """Readings of a synchronized clock after each optimizer update. wrap(build_optimizer) returns a build_optimizer whose optimizer
    ticks it; patch that over studio_model.build_optimizer and every train() run in the block is timed."""

    def __init__(self, sync=lambda: None, clock=time.perf_counter):
        self.sync, self.clock, self.stamps = sync, clock, []

    def tick(self):
        self.sync()
        self.stamps.append(self.clock())

    def wrap(self, build):
        def timed(model, offload=False):
            optimizer, schedule = build(model, offload=offload)
            update = optimizer.update

            def ticking(*args, **kwargs):
                result = update(*args, **kwargs)
                self.tick()
                return result
            optimizer.update = ticking
            return optimizer, schedule
        return timed


# ---- runs ----

def timed_run(cfg, examples, steps, warmup, sync=lambda: None, plain=False, offload=False, share=None, events=None):
    """Train `cfg` for `steps` steps on `examples` in a temporary folder and return (times in seconds per step, events).
    plain: no batch staging (the pre-#237 path). offload: train(offload_optimizer=True). via_job: run through
    studio_jobs.pretrain with that memory_share (a fraction of the GPU's memory), which decides on offload itself."""
    import studio_jobs
    import studio_model
    clock, events = StepClock(sync), [] if events is None else events
    cfg = dict(cfg, steps=steps)
    patches = [mock.patch.object(studio_model, "build_optimizer", clock.wrap(studio_model.build_optimizer)),
               mock.patch.object(np, "savez_compressed", lambda *a, **k: None)]
    if plain:
        patches.append(mock.patch.object(studio_model, "batch_stage", lambda device: None))
    with tempfile.TemporaryDirectory() as root:
        out = Path(root) / "run"
        for patch in patches:
            patch.start()
        try:
            if share is not None:
                job = types.SimpleNamespace(id="bench", kind="pretrain", params={"config": cfg, "examples": examples, "output_dir": str(out),
                                                                                 "memory_share": share, "snapshot_every": 0})
                studio_jobs.pretrain(job, types.SimpleNamespace(run_dir=root), events.append, None, None)
            else:
                studio_model.train(cfg, examples, out, emit=events.append, snapshot_every=0, telemetry_every=0, offload_optimizer=offload)
        finally:
            for patch in reversed(patches):
                patch.stop()
    return step_times(clock.stamps, warmup), events


def alternate(variants, rounds):
    """Run each (name, fn) of `variants` `rounds` times, in turn, and pool each one's step times: {name: [seconds]}."""
    pooled = {name: [] for name, _ in variants}
    for _ in range(rounds):
        for name, fn in variants:
            pooled[name] += fn()
    return pooled


def bench_prefetch(args, examples, sync):
    cfg = fit_config(args.seed)
    times = alternate([("plain", lambda: timed_run(cfg, examples, args.steps, args.warmup, sync, plain=True)[0]),
                       ("prefetch", lambda: timed_run(cfg, examples, args.steps, args.warmup, sync)[0])], args.rounds)
    plain, prefetch = summarize(times["plain"]), summarize(times["prefetch"])
    return plain, prefetch


def bench_offload_overhead(args, examples, sync):
    import studio_memory
    cfg = fit_config(args.seed)
    if studio_memory.offload_wanted(cfg, 1., "cuda"):
        raise SystemExit("The fitting model asks for offload by itself; this GPU is too small for the overhead comparison.")
    times = alternate([("off", lambda: timed_run(cfg, examples, args.steps, args.warmup, sync)[0]),
                       ("on", lambda: timed_run(cfg, examples, args.steps, args.warmup, sync, offload=True)[0])], args.rounds)
    return summarize(times["off"]), summarize(times["on"]), studio_memory.host_bytes(cfg)


def plain_attempt(cfg, examples, sync):
    """Try the not-fitting model with the optimizer state on the device, two steps, and say what happened."""
    outcome = "ran (the estimate was conservative: raise MIN_STATE_NEED)"
    try:
        timed_run(dict(cfg, batch_size=1, accumulation=1), examples, 2, 0, sync)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        outcome = "out of memory, as expected" if "out of memory" in str(exc).lower() else f"failed: {type(exc).__name__}: {exc}"
    gc.collect()
    torch.cuda.empty_cache()
    return outcome


def bench_offload_big(args, examples, sync):
    """Train the model whose state does not fit, end to end through the job handler, with the GPU filled up to the planned free memory."""
    free, total = torch.cuda.mem_get_info()
    plan = plan_offload_case(free)
    if plan is None:
        raise SystemExit(f"No model on the ladder suits {gib(free)} of free GPU memory (it needs a state of {gib(MIN_STATE_NEED)} or more "
                         "and a gap between what the plain and the offloaded run need).")
    ballast = [torch.empty(min(GIB, plan["ballast"] - i * GIB), dtype=torch.uint8, device="cuda") for i in range(-(-plan["ballast"] // GIB))]
    try:
        cfg = dict(plan["config"], seed=args.seed)
        left = torch.cuda.mem_get_info()[0]
        share = left / total  # the plan's share of the GPU: what the queue would hand this job
        plain = plain_attempt(cfg, examples, sync)
        torch.cuda.reset_peak_memory_stats()
        times, events = timed_run(cfg, examples, args.steps, args.warmup, sync, share=share)
        peak = torch.cuda.max_memory_allocated()
    finally:
        del ballast
        gc.collect()
        torch.cuda.empty_cache()
    texts = [e.get("text", "") for e in events if e.get("type") == "info"]
    return plan, left, plain, summarize(times), peak, any("host memory" in t for t in texts), [t for t in texts if "does not fit" in t]


# ---- report ----

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, default=200, help="training steps per run (default 200)")
    parser.add_argument("--rounds", type=int, default=3, help="alternating runs per variant for #237 and the #238 overhead (default 3)")
    parser.add_argument("--warmup", type=int, default=10, help="steps left out of each run's timing (default 10)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", choices=("prefetch", "offload"), help="run one issue's measurement only")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        print("bench_cuda.py needs a CUDA GPU and torch.cuda.is_available() is False here: run it on a CUDA node.", file=sys.stderr)
        return 2
    if args.steps <= args.warmup + 1:
        print("--steps must exceed --warmup by at least 2.", file=sys.stderr)
        return 2
    torch.manual_seed(args.seed)
    sync = torch.cuda.synchronize
    props = torch.cuda.get_device_properties(0)
    examples = synthetic_examples(seed=args.seed)
    out = [f"## CUDA measurement: {props.name}", "",
           f"- device: {props.name}, {gib(props.total_memory)}, compute capability {props.major}.{props.minor}",
           f"- torch {torch.__version__}, CUDA {torch.version.cuda}, Python {sys.version.split()[0]}",
           f"- {args.steps} steps per run ({args.warmup} left out), {args.rounds} alternating rounds, seed {args.seed}, synthetic text only", ""]
    import studio_model
    fit = studio_model.parameter_estimate(fit_config())
    shape = f"width {FIT['width']}, {FIT['layers']} layers, context 256, batch {FIT['batch_size']}, {fit:,} parameters"
    if args.only != "offload":
        plain, prefetch = bench_prefetch(args, examples, sync)
        out += ["### #237: pinned-memory batch prefetch", "", f"Model: {shape}.", "",
                markdown_table(["path", "steps timed", "mean ms/step", "median ms/step"],
                               [["plain", plain["n"], f"{plain['mean_ms']:.2f}", f"{plain['median_ms']:.2f}"],
                                ["prefetch", prefetch["n"], f"{prefetch['mean_ms']:.2f}", f"{prefetch['median_ms']:.2f}"]]), "",
                f"Speedup (plain / prefetch): {ratio(plain['mean_ms'], prefetch['mean_ms']):.3f}x by mean, "
                f"{ratio(plain['median_ms'], prefetch['median_ms']):.3f}x by median.", ""]
    if args.only != "prefetch":
        off, on, host = bench_offload_overhead(args, examples, sync)
        out += ["### #238: optimizer state offload", "", "**Step-time overhead on a model that fits**", "", f"Model: {shape}; the state ({gib(host)}) moves to pinned host memory when on.", "",
                markdown_table(["offload", "steps timed", "mean ms/step", "median ms/step"],
                               [["off", off["n"], f"{off['mean_ms']:.2f}", f"{off['median_ms']:.2f}"],
                                ["on", on["n"], f"{on['mean_ms']:.2f}", f"{on['median_ms']:.2f}"]]), "",
                f"Overhead of offload: {ratio(on['mean_ms'], off['mean_ms']):.2f}x by mean, {ratio(on['median_ms'], off['median_ms']):.2f}x by median.", ""]
        plan, left, plain, big, peak, chose, notes = bench_offload_big(args, examples, sync)
        out += ["**End to end on a model whose state does not fit**", "",
                f"A ballast tensor held {gib(plan['ballast'])}, leaving {gib(left)} free of {gib(props.total_memory)}. "
                "studio_jobs.pretrain trained it with memory_share=1; the plain run was tried first.", "",
                markdown_table(["measure", "value"],
                               [["model", f"width {plan['width']}, {plan['layers']} layers, {plan['params']:,} parameters"],
                                ["state on the device needs (estimate, batch 1)", gib(plan["need"])],
                                ["needs with offload (estimate)", gib(plan["need_offload"])],
                                ["state held in host memory", gib(plan["host"])],
                                ["GPU memory free while training", gib(left)],
                                ["plain run (state on the device)", plain],
                                ["offload chosen by studio_memory.offload_wanted", "yes" if chose else "NO"],
                                ["steps trained with offload", args.steps],
                                ["mean / median ms per step", f"{big['mean_ms']:.1f} / {big['median_ms']:.1f}"],
                                ["peak GPU memory allocated", gib(peak)]]), ""]
        out += [f"> {note}" for note in notes] + ([""] if notes else [])
        if not chose:
            print("\n".join(out))
            print("studio_jobs.pretrain did not turn offload on for a model that does not fit.", file=sys.stderr)
            return 1
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
