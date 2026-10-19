"""Pick the device for `device="auto"` from a measurement: time a few real training steps of the design on each available device.

The old rule (a constant number of values per step) was fitted on one machine. A fast CPU, a slow integrated GPU, a big batch or bf16
all move the crossover, so the probe builds the design on the CPU and the GPU, runs a warm-up and a few timed steps (forward, backward,
optimizer update) on random tokens, and takes the faster. The result is cached by design shape, batch, precision and machine (in
memory and in a small file), so a run of the same design starts without it.

Bounded: each device gets a budget of BUDGET_S seconds of timed steps (at least one step), the model built for it is freed
afterwards, and a device that fails (out of memory, a driver error) is dropped with a plain reason. If the CPU won, the GPU is timed
once more at LARGER_BATCH times the batch, to say whether a bigger batch would have changed the answer.
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import statistics
import threading
import time
from pathlib import Path

WARMUP = 2  # steps run and thrown away: allocator, kernel selection and lazy initialisation
STEPS = 3  # timed steps (the median is kept)
BUDGET_S = 4.  # stop timing a device after this many seconds (one timed step is always taken)
LARGER_BATCH = 4  # the batch multiple tried on the GPU when the CPU wins
CLOSE = 1.05  # the GPU must be this much faster to be chosen over the CPU; below it the CPU stays
MAX_AGE_S = 30 * 86400.  # a probe older than this is not reported to peers: torch, the driver or the machine may have changed since
REPORT_LIMIT = 64  # shapes a node tells its peers about (the newest)
VERSION = 1  # bump when the measurement changes meaning: old cache entries are ignored

_lock = threading.Lock()
_memory = {}


def shape_key(spec, vocabulary, batch):
    """What a step's cost depends on: the design (without learning-rate settings), the vocabulary, the batch and the precision."""
    spec = copy.deepcopy(spec)
    train = spec.get("train") or {}
    spec["train"] = {"precision": train.get("precision"), "optimizer": (train.get("optimizer") or {}).get("id"),
                     "checkpointing": (train.get("run") or {}).get("checkpointing")}
    spec.pop("name", None)
    text = json.dumps({"spec": spec, "vocabulary": int(vocabulary), "batch": int(batch), "v": VERSION}, sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def machine_key(status):
    """This machine as the probe sees it: processor, the intra-op thread count (the app caps it), torch and the GPU."""
    import torch
    return "|".join([platform.node(), platform.processor() or platform.machine(), str(torch.get_num_threads()), torch.__version__,
                     status.get("kind") or "cpu", status.get("name") or ""])


def default_path():
    from studio_core import ROOT
    return Path(ROOT) / "data" / "device_probe.json"


def _load(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(path, table):
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(table), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass  # a cache that cannot be written is just a cache that misses next time


def _why(exc):
    text = f"{type(exc).__name__}: {exc}".strip()
    if "out of memory" in text.lower():
        return "ran out of memory"
    return text.splitlines()[0][:200]


def time_steps(step, sync, clock=time.perf_counter, warmup=WARMUP, steps=STEPS, budget=BUDGET_S):
    """Seconds per step: the median of up to `steps` timed calls of `step()` after `warmup` calls, each ended by `sync()`.
    A slow device stops early: warm-up ends once it has used `budget` seconds and the timed steps end when they have, always
    keeping one measurement."""
    spent, times, n = 0., [], 0
    while n < warmup + steps:
        start = clock()
        step()
        sync()
        took = clock() - start
        if n < warmup:
            spent += took
            if spent > budget:
                warmup = n + 1  # too slow to warm up twice: the next call is the measurement
        else:
            times.append(took)
            if sum(times) >= budget:
                break
        n += 1
    return statistics.median(times)


def probe(runner, devices, tokens, clock=time.perf_counter, larger=None):
    """Compare `devices` (names, "cpu" first). `runner(device, batch_multiple)` returns `(step, sync, close)`; `tokens` is what one
    step reads. Returns a plain dict: device (the pick), seconds and tokens_per_second per device, speedup (the pick's speed over the
    other's), failed ({device: reason}), larger_batch (None, or {multiple, speedup} when the GPU wins only with a bigger batch)
    and a one-sentence `reason`. Raises RuntimeError when no device could be timed."""
    seconds, failed = {}, {}
    for device in devices:
        try:
            seconds[device] = _measure(runner, device, 1, clock)
        except Exception as exc:
            failed[device] = _why(exc)
    if not seconds:
        raise RuntimeError("no device could be timed: " + "; ".join(f"{d} {r}" for d, r in failed.items()))
    speed = {d: tokens / s for d, s in seconds.items()}
    best = min(seconds, key=lambda d: seconds[d] * (CLOSE if d != "cpu" else 1.))  # a GPU must beat the CPU by CLOSE
    others = [d for d in seconds if d != best]
    out = {"version": VERSION, "device": best, "seconds": seconds, "tokens_per_second": speed,
           "speedup": seconds[others[0]] / seconds[best] if others else None, "failed": failed, "larger_batch": None,
           "tokens": tokens, "measured_at": time.time()}
    gpu = next((d for d in seconds if d != "cpu"), None)
    if best == "cpu" and gpu and larger:
        try:
            big = _measure(runner, gpu, larger, clock)
            ratio = (larger * tokens / big) / speed["cpu"]
            if ratio > CLOSE:
                out["larger_batch"] = {"multiple": larger, "speedup": ratio}
        except Exception:
            pass  # the bigger batch did not fit or failed: nothing to add
    out["reason"] = describe(out)
    return out


def _measure(runner, device, multiple, clock):
    step, sync, close = runner(device, multiple)
    try:
        return time_steps(step, sync, clock)
    finally:
        close()


def describe(result):
    """The sentence the run and the plan show: what was picked, why, and the speeds behind it."""
    seconds, speed, failed = result["seconds"], result["tokens_per_second"], result["failed"]
    device = result["device"]
    if len(seconds) == 1:
        lost = "; ".join(f"the {d} probe {r}" for d, r in failed.items())
        return f"{lost}; using the {device.upper()}"
    other = next(d for d in seconds if d != device)
    ratio = result["speedup"]
    if device == "cpu":
        text = f"CPU is faster for this size: {speed['cpu']:,.0f} tokens/s against {speed[other]:,.0f} on the GPU, measured ({ratio:.1f}x)"
        big = result.get("larger_batch")
        if big:
            text += f"; the GPU would be {big['speedup']:.1f}x faster at a batch {big['multiple']}x larger"
        return text + "."
    return f"GPU {ratio:.1f}x faster, using it: {speed[device]:,.0f} tokens/s against {speed[other]:,.0f} on the CPU, measured."


def build_runner(spec, vocabulary, batch, seed=0):
    """The real runner for `probe`: a throwaway model of the design on `device`, stepped on random tokens with its own optimizer."""
    import numpy as np
    from studio_backend import empty_cache
    from studio_compile import CompiledModel
    from studio_model import build_optimizer

    def runner(device, multiple):
        rows, context = batch * multiple, spec["dims"]["n_ctx"]
        try:
            model = CompiledModel(spec, vocabulary, seed=seed, device=device, batch_size=rows)
            if model.device != device:
                raise RuntimeError(model.backend.reason or f"{device} is not available")
            optimizer, _ = build_optimizer(model)
            rng = np.random.default_rng(seed)
            ids = rng.integers(0, vocabulary, size=(rows, context))
            targets = rng.integers(0, vocabulary, size=(rows, context))
        except BaseException:
            empty_cache(device)
            raise

        def step():
            model.zero_grad()
            model.loss_and_backward(ids, targets)
            optimizer.update(model, 1e-5)

        def close():
            nonlocal model, optimizer
            model = optimizer = None
            empty_cache(device)
        return step, model.backend.synchronize, close
    return runner


def measure(spec, vocabulary, batch, status, path=None, runner=None, clock=time.perf_counter):
    """The cached probe of the design (a validated ModelSpec) at `batch` rows, run on first use. `status` is
    `studio_backend.gpu_status()`; a machine with no GPU needs no probe and gets None. Raises when nothing could be timed.
    `path` is the cache file (None: the workspace's; "" : memory only)."""
    if not status.get("available"):
        return None
    key = shape_key(spec, vocabulary, batch) + "|" + machine_key(status)
    path = default_path() if path is None else path
    with _lock:
        if key in _memory:
            return _memory[key]
        hit = _load(path).get(key) if path else None
        if isinstance(hit, dict) and hit.get("version") == VERSION:
            _memory[key] = hit
            return hit
        result = probe(runner or build_runner(spec, vocabulary, batch), ["cpu", status["kind"]],
                       batch * spec["dims"]["n_ctx"], clock, larger=LARGER_BATCH)
        _memory[key] = result
        if path:
            table = _load(path)
            table[key] = result
            _save(path, table)
        return result


def cached(spec, vocabulary, batch, status, path=None):
    """The stored probe for this design on this machine, or None; never measures. For the plan and the time model, which must not stall."""
    if not status.get("available"):
        return None
    key = shape_key(spec, vocabulary, batch) + "|" + machine_key(status)
    if key in _memory:
        return _memory[key]
    path = default_path() if path is None else path
    hit = _load(path).get(key) if path else None
    return hit if isinstance(hit, dict) and hit.get("version") == VERSION else None


def reported(status, path=None, now=None, max_age=MAX_AGE_S, limit=REPORT_LIMIT):
    """What this node tells its peers (`GET /node`): {shape key: {device, seconds: {device: s}, tokens, age_s}} for the newest `limit`
    probes made on THIS machine as it is now (same processor, threads, torch and GPU: the machine part of the cache key) and not older
    than `max_age`. A probe from another torch or GPU is left out, so it is never trusted. Ages, not timestamps, so no two clocks are
    compared. Never measures."""
    if not status.get("available"):
        return {}
    machine = machine_key(status)
    path = default_path() if path is None else path
    now = time.time() if now is None else now
    table = dict(_load(path)) if path else {}
    table.update(_memory)
    rows = []
    for key, hit in table.items():
        shape, _, where = key.partition("|")
        at = hit.get("measured_at") if isinstance(hit, dict) else None
        if where != machine or hit.get("version") != VERSION or not isinstance(at, (int, float)) or not 0 <= now - at <= max_age:
            continue
        seconds = {d: float(s) for d, s in (hit.get("seconds") or {}).items() if isinstance(s, (int, float)) and s > 0}
        if hit.get("device") in seconds and hit.get("tokens"):
            rows.append((at, shape, {"device": hit["device"], "seconds": seconds, "tokens": int(hit["tokens"]), "age_s": now - at}))
    rows.sort(key=lambda row: row[0], reverse=True)
    return {shape: value for _, shape, value in rows[:limit]}


def any_stored(path=None):
    """Whether any probe has been made on this machine (in memory or in the cache file): a cheap check that spares a caller which only
    wants `cached` from bringing the GPU up when there is nothing to find."""
    path = default_path() if path is None else path
    return bool(_memory) or bool(path and _load(path))


def forget():
    """Drop the in-memory cache (tests)."""
    with _lock:
        _memory.clear()
