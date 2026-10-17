"""The loss landscape around a model's weights (Li et al. 2018, "Visualizing the Loss Landscape of Neural Nets").

Two directions R1, R2 in parameter space; the loss at theta + a*R1 + b*R2 over a grid of (a, b) on a small fixed held-back batch.
The directions are either random and filter-normalised (every filter of R has the norm of the same filter of the weights, so a step
of 1 is "as far as the weights are long" for every layer alike), or the two main directions of the training trajectory (PCA of the
snapshots' offsets from the centre), so the path to the minimum lies in the plane and can be drawn on it.

Everything here is plain NumPy/torch on the CPU and takes weights as {name: array}: the Services layer picks the files, caches
and runs it as a job. A "live" frame is the same call with a small grid, the filter norms of the first frame and a throttle.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

MIN_GRID, MAX_GRID = 5, 41
EVALUATION_BUDGET = 4e9  # grid cells x parameters one landscape may cost; keeps a laptop answer in minutes
MAX_PARAMETERS = 200_000_000
EPSILON = 1e-10

MODES = ("random", "pca")


class Cancelled(InterruptedError):
    plain = True  # its message is a sentence for people (studio_api.jobs shows it as the job's error)


def grid_size(requested):
    """An odd grid size inside [MIN_GRID, MAX_GRID], so the centre (0, 0) is a cell."""
    size = max(MIN_GRID, min(MAX_GRID, int(requested)))
    return size if size % 2 else size + 1


def check_size(parameters, grid):
    """ValueError with a sentence when grid x grid evaluations of a model this size would not finish on a laptop."""
    if parameters > MAX_PARAMETERS:
        raise ValueError(f"This model has {parameters:,} parameters, more than the {MAX_PARAMETERS:,} a landscape is drawn for on this computer.")
    if grid * grid * parameters > EVALUATION_BUDGET:
        fits = int((EVALUATION_BUDGET / max(parameters, 1)) ** .5)
        raise ValueError(f"A {grid} by {grid} landscape of {parameters:,} parameters is too much work here. "
                         f"Use a grid of {max(fits, 1)} or smaller" + ("." if fits >= MIN_GRID else f", below the least ({MIN_GRID}); this model is too large."))


# ---- directions --------------------------------------------------------------------------------------------------------

def filter_axis(role, ndim):
    """The axis that runs over a tensor's filters (output units), or None for a vector (a bias or gain: no filters).
    An embedding is [pieces, d]: one filter per row. A linear weight is [in, out]: one filter per output column."""
    if ndim < 2:
        return None
    return 0 if role == "embedding" else ndim - 1


def _norms(array, axis):
    other = tuple(i for i in range(array.ndim) if i != axis)
    return np.sqrt((np.asarray(array, dtype=np.float64) ** 2).sum(axis=other, keepdims=True))


def filter_norms(weights, roles):
    """{name: per-filter norms (shaped to broadcast against the tensor)} of the weights; None for a vector. These set the scale of
    the random directions, so keeping them from one frame makes later frames use the same directions."""
    out = {}
    for name, value in weights.items():
        axis = filter_axis(roles.get(name), np.ndim(value))
        out[name] = None if axis is None else _norms(value, axis)
    return out


def filter_normalise(direction, norms, axis):
    """Scale each filter of `direction` to the norm in `norms` (the weights' filter norms): Li et al.'s filter normalisation."""
    return direction * norms / (_norms(direction, axis) + EPSILON)


def random_directions(weights, roles, seed, norms=None):
    """Two random, filter-normalised directions {name: array} (float32). Vectors (biases, gains) get a zero direction, as in the
    paper. The same seed and shapes give the same Gaussian draw; `norms` (default: of `weights`) only rescales its filters."""
    norms = norms if norms is not None else filter_norms(weights, roles)
    pair = []
    for index in range(2):
        rng = np.random.default_rng([int(seed), index])
        direction = {}
        for name in sorted(weights):
            draw = rng.standard_normal(np.shape(weights[name]))  # drawn for every name, so the stream does not depend on which are zeroed
            axis = filter_axis(roles.get(name), np.ndim(weights[name]))
            direction[name] = (np.zeros(np.shape(weights[name]), dtype=np.float32) if axis is None
                               else filter_normalise(draw, norms[name], axis).astype(np.float32))
        pair.append(direction)
    return pair


def _flat(tensors, names):
    return np.concatenate([np.asarray(tensors[name], dtype=np.float64).ravel() for name in names])


def _unflat(vector, shapes):
    out, start = {}, 0
    for name, shape in shapes.items():
        size = int(np.prod(shape))
        out[name] = vector[start:start + size].reshape(shape).astype(np.float32)
        start += size
    return out


def pca_directions(center, others):
    """The two main directions of the offsets of `others` (weights at other steps) from `center`, scaled so the farthest snapshot
    lies at 1.0 on each axis. Returns (d1, d2). ValueError with a sentence when the snapshots do not span a plane."""
    names = sorted(center)
    shapes = {name: np.shape(center[name]) for name in names}
    base = _flat(center, names)
    offsets = np.stack([_flat(other, names) - base for other in others]) if others else np.zeros((0, base.size))
    if len(offsets) < 2:
        raise ValueError(f"The trajectory needs at least 3 snapshots to pick two directions; this run kept {len(offsets) + 1}.")
    values, vectors = np.linalg.eigh(offsets @ offsets.T)  # the small side of the SVD: snapshots x snapshots
    order = np.argsort(values)[::-1][:2]
    axes = []
    biggest = float(np.sqrt(max(values[order[0]], 0.)))
    for index in order:
        scale = float(np.sqrt(max(values[index], 0.)))
        if scale < max(1e-9, 1e-6 * biggest):
            raise ValueError("The snapshots lie along one line, so there is no second direction. Use random directions.")
        axis = offsets.T @ vectors[:, index] / scale  # unit length
        axis *= max(float(np.abs(offsets @ axis).max()), EPSILON)  # the farthest snapshot sits at 1.0
        axes.append(_unflat(axis, shapes))
    return axes[0], axes[1]


def coordinates(center, weights, d1, d2):
    """(a, b) with center + a*d1 + b*d2 closest to `weights` (least squares in the plane)."""
    names = sorted(center)
    delta = _flat(weights, names) - _flat(center, names)
    basis = np.stack([_flat(d1, names), _flat(d2, names)], axis=1)
    solution = np.linalg.lstsq(basis, delta, rcond=None)[0]
    return float(solution[0]), float(solution[1])


# ---- the grid ----------------------------------------------------------------------------------------------------------

def _set(model, tensors):
    with torch.no_grad():
        for name, value in tensors.items():
            model.p[name].copy_(value)


def evaluate(model, ids, targets):
    with torch.no_grad():
        return float(model.loss(ids, targets))


def compute(model, ids, targets, center, others, *, mode="random", grid=21, span=1.0, seed=0, norms=None, stop=None, throttle=0.):
    """The landscape around `center` ({name: array}); `others` is [(step, weights)], the trajectory.

    Returns {axis, loss (rows are b, columns a), ceiling, current, minimum, trajectory, unit, evaluations, seconds}. Losses are nats
    per piece on (ids, targets). `norms` pin the random directions' filter norms (default: the centre's own). `throttle` sleeps
    that many times the work time after each row, so a live frame takes a fraction of a core's time rather than all of it.
    """
    if mode not in MODES:
        raise ValueError(f"Directions are {' or '.join(MODES)}.")
    grid = grid_size(grid)
    check_size(sum(int(np.size(v)) for v in center.values()), grid)
    started = time.perf_counter()
    if mode == "random":
        d1, d2 = random_directions(center, model.roles, seed, norms)
        unit = "filter"
    else:
        d1, d2 = pca_directions(center, [weights for _, weights in others])
        unit = "trajectory"
    names = sorted(center)
    base = {name: torch.as_tensor(np.asarray(center[name], dtype=np.float32)) for name in names}
    first = {name: torch.as_tensor(d1[name]) for name in names}
    second = {name: torch.as_tensor(d2[name]) for name in names}
    axis = np.linspace(-span, span, grid)
    losses = np.empty((grid, grid))
    held = {name: model.p[name].detach().clone() for name in names}
    try:
        for row, b in enumerate(axis):
            tick = time.perf_counter()
            for column, a in enumerate(axis):
                if stop is not None and stop.is_set():
                    raise Cancelled("The landscape was cancelled.")
                _set(model, {name: base[name] + float(a) * first[name] + float(b) * second[name] for name in names})
                losses[row, column] = evaluate(model, ids, targets)
            if throttle:
                time.sleep((time.perf_counter() - tick) * throttle)
        trajectory = []
        for step, weights in others:
            _set(model, {name: torch.as_tensor(np.asarray(weights[name], dtype=np.float32)) for name in names})
            a, b = coordinates(center, weights, d1, d2)
            trajectory.append({"step": int(step), "x": a, "y": b, "loss": evaluate(model, ids, targets)})
    finally:
        _set(model, held)
    finite = np.isfinite(losses)
    if not finite.any():
        raise ValueError("The loss is not finite anywhere on the grid. Try a smaller range.")
    losses = np.where(finite, losses, losses[finite].max())
    low = np.unravel_index(int(losses.argmin()), losses.shape)
    middle = grid // 2
    return {"axis": axis.tolist(), "loss": losses.tolist(), "ceiling": float(np.percentile(losses, 95)), "unit": unit,
            "current": {"x": 0., "y": 0., "loss": float(losses[middle, middle])},
            "minimum": {"x": float(axis[low[1]]), "y": float(axis[low[0]]), "loss": float(losses[low]),
                        "inside": bool(0 < low[0] < grid - 1 and 0 < low[1] < grid - 1)},
            "trajectory": trajectory, "evaluations": grid * grid + len(trajectory), "seconds": time.perf_counter() - started}


# ---- a training run's files --------------------------------------------------------------------------------------------

def read_weights(path):
    """(step, {name: array}) of a viz snapshot (viz/step_*.npz) or a checkpoint (best.npz, last.npz)."""
    path = Path(path)
    if path.parent.name == "viz":
        with np.load(path, allow_pickle=False) as archive:
            weights = {key[len("weight::"):]: archive[key] for key in archive.files if key.startswith("weight::")}
            metadata = json.loads(str(archive["metadata"]))
        return int(metadata.get("step", 0)), weights
    from studio_model import load_checkpoint
    checkpoint = load_checkpoint(path, weights_only=True)
    return int(checkpoint.get("step", 0)), checkpoint["model"]


def run_model(run_dir):
    """(model, tokenizer) of a run from its own config and tokenizer files, which exist from the first step, before any checkpoint."""
    from studio_model import ScratchTransformer, Tokenizer
    run = Path(run_dir)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer.restore(json.loads((run / "tokenizer.json").read_text(encoding="utf-8")))
    return ScratchTransformer(dict(config, device="cpu"), tokenizer.vocab_size), tokenizer


def run_snapshots(run_dir):
    """The run's viz snapshot paths, oldest first."""
    return sorted(Path(run_dir).glob("viz/step_*.npz"))


def landscape_batch(tokenizer, docs, n_ctx, rows=4, length=32):
    """(ids, targets): the first `rows` windows of held-back text, `length` pieces at most. Fixed, so frames are comparable."""
    import studio_inspect
    ids, _ = studio_inspect.heldout_batch(tokenizer, docs, min(length, n_ctx), rows=rows)
    return ids, studio_inspect._targets(ids)
