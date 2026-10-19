"""How much a run is worth, from numbers: a fitted loss curve, the point where it goes flat, and a measured step rate.

Plain arithmetic and numpy, except `measure_step_rate`, which trains a throwaway copy of a design for a few steps. The fit is
loss(x) = floor + scale * x^-exponent over tokens (or steps) read, the form scaling-law papers use for a fixed model size
(Kaplan et al. 2020; Hoffmann et al. 2022). It is a guide drawn from one run's own validation points, not a promise."""
from __future__ import annotations

import math
import time

import numpy as np

FLAT_PER_DOUBLING = .02  # a curve is flat once doubling the tokens read lowers the error by less than this share of it (our cut-off)
MIN_POINTS = 5  # validation points a fit needs
SPAN = 500  # points earlier than this fraction of the last x are left out: the first steps are warm-up, not the trend
BIN = 1.15  # points closer than this ratio in x are thinned to one, so a densely logged start does not outweigh the rest
FLOORS = 400  # floors tried


def thin(points):
    """The last point of each log-spaced bin of x, so every stretch of the run weighs the same in a fit."""
    kept, edge = [], None
    for x, y in sorted(points):
        if edge is not None and x < edge:
            kept[-1] = (x, y)
            continue
        kept.append((x, y))
        edge = x * BIN
    return kept


def fit_curve(points):
    """Fit loss = floor + scale * x^-exponent to [(x, loss)] (x in tokens or steps, loss the validation error). Returns
    {"floor", "scale", "exponent", "rmse", "points"} or None when there are too few points or the error is not falling."""
    clean = [(float(x), float(y)) for x, y in points if x and x > 0 and y is not None and math.isfinite(y) and y > 0]
    if not clean:
        return None
    last = max(x for x, _ in clean)
    kept = thin([(x, y) for x, y in clean if x >= last / SPAN])
    if len(kept) < MIN_POINTS:
        return None
    x, y = np.array([p[0] for p in kept]), np.array([p[1] for p in kept])
    if not y[-1] < y[0]:
        return None
    design = np.vstack([np.log(x), np.ones_like(x)]).T
    best = None
    for floor in np.linspace(0., float(y.min()) * .999, FLOORS):
        (slope, intercept), *_ = np.linalg.lstsq(design, np.log(y - floor), rcond=None)
        if slope >= 0:
            continue
        error = float(((floor + np.exp(intercept) * x ** slope - y) ** 2).sum())
        if best is None or error < best[0]:
            best = (error, float(floor), float(np.exp(intercept)), float(-slope))
    if best is None:
        return None
    error, floor, scale, exponent = best
    return {"floor": floor, "scale": scale, "exponent": exponent, "rmse": math.sqrt(error / len(kept)), "points": len(kept)}


def predict(fit, x):
    """The fitted error after reading `x` tokens."""
    return fit["floor"] + fit["scale"] * max(float(x), 1.) ** -fit["exponent"]


def flat_point(fit, start, limit):
    """The first x from `start` at which doubling x lowers the fitted error by under FLAT_PER_DOUBLING of it; `limit` when it
    never does before then. Stepped in 10% increments, so it is a round figure, not a precise one."""
    x = max(float(start), 1.)
    while x < limit:
        now = predict(fit, x)
        if (now - predict(fit, 2 * x)) / now < FLAT_PER_DOUBLING:
            return x
        x *= 1.1
    return float(limit)


def curve_points(fit, start, stop, count=24):
    """[(x, fitted error)], `count` points spread evenly in log x from `start` to `stop`, for drawing the expected curve."""
    start, stop = max(float(start), 1.), max(float(stop), 2.)
    return [(x, predict(fit, x)) for x in np.geomspace(start, stop, count)]


def step_rate(run_step, steps=5, warmup=2, clock=time.perf_counter, stop=None):
    """Seconds per call of `run_step`: `warmup` calls are not timed (the first pays for allocation and kernel setup), then the mean
    of `steps` timed ones by `clock`. None when `stop` (a threading.Event) ended it first."""
    for _ in range(warmup):
        if stop is not None and stop.is_set():
            return None
        run_step()
    total = 0.
    for _ in range(steps):
        if stop is not None and stop.is_set():
            return None
        began = clock()
        run_step()
        total += clock() - began
    return total / steps


def measure_step_rate(spec, rows, vocab, steps=5, warmup=2, clock=time.perf_counter, stop=None, config=None):
    """Time optimizer steps of a throwaway copy of `spec` on random token windows, on the device a run of it would really use (the
    backend's own choice: "auto" keeps a small model on the CPU). Nothing is saved. Returns {"seconds_per_step", "tokens_per_second",
    "device", "device_name", "note", "rows", "context"}, or None when `stop` ended it. A GPU that cannot hold the model raises
    RuntimeError, which the caller turns into a sentence."""
    import studio_model as sm
    from studio_spec import config_from_spec

    cfg = sm.validate_config({**(config or {}), **config_from_spec(spec), "steps": 10, "batch_size": rows, "validation": 0., "objective": "all"})
    model = sm.ScratchTransformer(cfg, vocab)
    optimizer, _ = sm.build_optimizer(model)
    context, accumulation = int(cfg["context"]), int(cfg["accumulation"])
    x = np.random.default_rng(0).integers(0, vocab, (rows, context)).astype(np.int64)

    def one():
        model.zero_grad()
        for _ in range(accumulation):
            model.loss_and_backward(x, x, 1 / accumulation)
        optimizer.update(model, cfg["learning_rate"])
        model.backend.synchronize()
    seconds = step_rate(one, steps, warmup, clock, stop)
    if seconds is None:
        return None
    return {"seconds_per_step": seconds, "tokens_per_second": rows * context * accumulation / max(seconds, 1e-9),
            "device": model.backend.device, "device_name": model.backend.name, "note": model.backend.reason, "rows": rows, "context": context}
