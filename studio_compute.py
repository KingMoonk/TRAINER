"""Size a run before starting it: how much text a design needs, and a short test that suggests a learning rate.

plan() is plain arithmetic (no torch, no windows). lr_range_test() trains a fresh copy of the design for about a hundred steps.
"""
from __future__ import annotations

import math
import random
import threading

TOKENS_PER_PARAM = 20  # compute-optimal training reads about 20 tokens per parameter (Hoffmann et al. 2022, "Chinchilla")
MAX_EPOCHS = 4  # repeating the text up to ~4 times is nearly as good as fresh text; beyond that returns diminish (Muennighoff et al. 2023)
UNDER_TRAINED = .25  # our own cut-off, not from a paper: planned tokens below this share of 20 per parameter (5 per parameter) are "far below"


def plan(params, data_tokens, seconds=None, tokens_per_second=None, batch_tokens=None):
    """What a design of `params` parameters can be trained on, given `data_tokens` of text and optionally a time budget.

    Rule of thumb: compute-optimal training reads about 20 tokens per parameter (TOKENS_PER_PARAM; Hoffmann et al. 2022).
    Repeating the text helps up to about 4 epochs (MAX_EPOCHS) and returns diminish after that (Muennighoff et al. 2023).
    seconds is a time budget and tokens_per_second a measured or estimated throughput; both are needed to apply it.
    batch_tokens (rows x context x accumulation) turns tokens into optimizer steps.

    The planned run reads min(20 x params, 4 x data, what the time allows) tokens. Plain fields, None where unknown:
    optimal_tokens, data_params (what one pass over the text supports), time_tokens, time_params, recommended_params (the
    size one pass over the text, within the time budget, supports), planned_tokens, steps, epochs, seconds_needed, limit
    ("data" | "time" | None: what holds the plan below the optimum), tokens_per_param (planned), epochs_for_optimal,
    under_trained (planned far below 20 per parameter), over_repeated (reaching 20 per parameter needs more than 4 epochs)."""
    params = max(0, int(params or 0))
    data = int(data_tokens) if data_tokens else None
    optimal = TOKENS_PER_PARAM * params
    time_tokens = int(seconds * tokens_per_second) if seconds and tokens_per_second and seconds > 0 and tokens_per_second > 0 else None
    caps = [(tokens, kind) for tokens, kind in ((MAX_EPOCHS * data if data else None, "data"), (time_tokens, "time")) if tokens is not None]
    planned, limit = optimal, None
    for tokens, kind in caps:
        if tokens < planned:
            planned, limit = tokens, kind
    sized = [tokens for tokens in (data, time_tokens) if tokens is not None]
    out = dict(params=params, data_tokens=data, optimal_tokens=optimal, data_params=data // TOKENS_PER_PARAM if data else None,
               time_tokens=time_tokens, time_params=time_tokens // TOKENS_PER_PARAM if time_tokens is not None else None,
               recommended_params=min(sized) // TOKENS_PER_PARAM if sized else None, planned_tokens=planned, limit=limit,
               tokens_per_param=planned / params if params else None, epochs=planned / data if data else None,
               epochs_for_optimal=optimal / data if data else None,
               steps=max(1, math.ceil(planned / batch_tokens)) if batch_tokens and planned else None,
               seconds_needed=planned / tokens_per_second if tokens_per_second and tokens_per_second > 0 else None)
    out["under_trained"] = bool(params and caps and planned < UNDER_TRAINED * optimal)
    out["over_repeated"] = bool(data and optimal > MAX_EPOCHS * data)
    return out


def throughput(batch, context, params, layers, kind="cpu", accumulation=1):
    """Estimated tokens per second: the Pretraining page's own time model (studio_ux.step_seconds), so both pages agree.
    Prefer a measured number (the newest run's tokens_per_second) when there is one."""
    from studio_ux import step_seconds
    cfg = {"batch_size": batch, "context": context, "layers": layers, "accumulation": accumulation}
    return batch * context * accumulation / step_seconds(cfg, params, kind)


# ------------------------------------------------------------------------------------------------ learning-rate range test

SWEEP_STEPS = 100
SWEEP_BATCH = 8
SMOOTH = .9  # weight of the running loss average
DIVERGED = 4.  # a smoothed loss this many times the best one so far ends the sweep early (the fastai default)
SAFETY = 10.  # the suggestion is this many times lower than the steepest point
SKIP_START, SKIP_END = 10, 5  # points left out of the slope reading: the smoothed start is noisy, the end is near divergence (fastai)


def smooth(losses, beta=SMOOTH):
    """Exponential moving average with the start bias removed."""
    out, avg = [], 0.
    for i, loss in enumerate(losses, 1):
        avg = beta * avg + (1 - beta) * loss
        out.append(avg / (1 - beta ** i))
    return out


def suggest(rates, losses):
    """(suggested rate | None, index of the steepest point | None) from an exponential sweep: `rates` rising by a constant factor, `losses` its raw losses.

    Rule (Smith 2017, "Cyclical learning rates"; the loss-slope reading is fastai's): smooth the loss, find where it falls fastest
    per step of the rising rate (the most negative slope; the rate steps by a constant factor, so a step is a step in log rate), and suggest
    a tenth of that rate: the steepest point is already near the edge of what trains stably. The sweep stops when the loss
    diverges, so only points before that count. None when the loss never falls (nothing to read)."""
    if len(losses) < 3:
        return None, None
    smoothed = smooth(losses)
    slopes = [smoothed[i + 1] - smoothed[i] for i in range(len(smoothed) - 1)]
    lo, hi = (SKIP_START, len(slopes) - SKIP_END) if len(slopes) > SKIP_START + SKIP_END + 2 else (0, len(slopes))
    steepest = min(range(lo, hi), key=slopes.__getitem__)
    if slopes[steepest] >= 0:
        return None, None
    # the slope between points i and i+1 belongs to the rate that gets there
    rate = rates[steepest + 1] / SAFETY
    return min(max(rate, rates[0]), rates[-1]), steepest + 1


def lr_range_test(spec, examples, steps=SWEEP_STEPS, lr_min=1e-6, lr_max=None, batch_size=SWEEP_BATCH, seed=0, stop=None, emit=None,
                  config=None):
    """Sweep the learning rate up exponentially from lr_min to lr_max over `steps` steps on a fresh copy of the design and
    suggest a rate (suggest() documents the rule).

    The model is built new from `spec` and the text in `examples` ({"text": ...} rows) and trained with the spec's own optimizer, at
    the swept rate, with no warm-up or schedule. Nothing is written to disk. lr_max defaults to the optimizer's own upper bound.
    The sweep ends early when the loss diverges (4x the best smoothed loss) or goes non-finite. `stop` is a threading.Event
    checked every step. emit gets {"type": "lr_range", "step", "total", "lr", "loss"}.

    Returns {"rates", "losses", "smoothed", "suggestion", "steepest" (index), "diverged", "failed_at" (the rate that diverged, or
    None), "cancelled", "lr_min", "lr_max"}."""
    import studio_model as sm
    from studio_spec import config_from_spec, module_of

    stop, emit = stop or threading.Event(), emit or (lambda event: None)
    if steps < 3:
        raise ValueError("A range test needs at least three steps.")
    cfg = sm.validate_config({**(config or {}), **config_from_spec(spec), "steps": steps, "batch_size": batch_size, "validation": 0.,
                              "objective": "all", "seed": seed})
    lr_max = lr_max or sm._learning_rate_limit(sm.effective_spec(cfg))
    if not 0 < lr_min < lr_max:
        raise ValueError("The lowest rate must be above zero and below the highest.")
    result = dict(rates=[], losses=[], smoothed=[], suggestion=None, steepest=None, diverged=False, failed_at=None, cancelled=False, lr_min=lr_min, lr_max=lr_max)
    tokenizer_module, tokenizer_cfg = module_of("tokenizer", sm.effective_spec(cfg)["tokenizer"])
    rows = [dict(e) for e in examples]
    tokenizer = tokenizer_module.fit(tokenizer_cfg, rows)
    samples = sm.samples_from(rows, tokenizer, cfg)
    if not samples:
        raise ValueError("No trainable target tokens were found.")
    model = sm.ScratchTransformer(cfg, tokenizer.vocab_size)  # fresh weights: the design's own are never touched
    optimizer, _ = sm.build_optimizer(model)
    rng = random.Random(seed)
    factor = (lr_max / lr_min) ** (1 / (steps - 1))
    best = float("inf")
    for step in range(steps):
        if stop.is_set():
            result["cancelled"] = True
            break
        rate = lr_min * factor ** step
        model.zero_grad()
        x, y = sm.batch_arrays([samples[rng.randrange(len(samples))] for _ in range(batch_size)])
        loss = model.loss_and_backward(x, y)
        if not math.isfinite(loss):
            result["diverged"], result["failed_at"] = True, rate
            break
        try:
            optimizer.update(model, rate)
        except ValueError:  # a non-finite gradient
            result["diverged"], result["failed_at"] = True, rate
            break
        result["rates"].append(rate)
        result["losses"].append(loss)
        result["smoothed"] = smooth(result["losses"])
        emit({"type": "lr_range", "step": step + 1, "total": steps, "lr": rate, "loss": loss})
        best = min(best, result["smoothed"][-1])
        if result["smoothed"][-1] > DIVERGED * best:
            result["diverged"], result["failed_at"] = True, rate
            break
    if not result["cancelled"]:
        result["suggestion"], result["steepest"] = suggest(result["rates"], result["losses"])
    return result
