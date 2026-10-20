"""Does the model fit the text and the time? The size part of the Pretraining plan, without Tk or I/O: the verdict, the warning, the
expected time and error, and the design and step count that would fit the hours the person gives. Fed by a step rate measured on this
computer (studio_scaling.measure_step_rate) and, when the design has run before, the validation points of that run."""
from __future__ import annotations

import copy
import math

import studio_scaling as scaling

RULE = 20  # word pieces per parameter a model this size uses best (studio_compute.TOKENS_PER_PARAM, Hoffmann et al. 2022)
TOO_SMALL_AT = 10 * RULE  # planned word pieces per parameter above which the model is too small for the text (our cut-off)
TOO_LITTLE_AT = RULE / 4  # below it the text is too little for the model (studio_compute.UNDER_TRAINED)
LADDER = ((32, 2), (48, 2), (64, 2), (96, 3), (128, 4), (192, 4), (256, 6), (384, 6), (512, 8), (768, 10), (1024, 12))  # (width, layers)
DEFAULT_HOURS = 4.
GPU_WORTH = .7  # a GPU is worth mentioning when a step takes under this share of the CPU's time

SIZING = {
    "too_small": "This model is too small for this much text. The plan reads {ratio} word pieces per parameter, {times}× the roughly {rule} "
                 "a model this size uses best, so most of the later steps buy little: the same hours go further in a bigger design.",
    "too_small_fit": "This model is too small for this much text, and its own last run shows it: by about step {flat} ({flat_hours}) the "
                     "validation error was within {gap} of where this plan ends. The other {rest} steps would lower it only from "
                     "{at_flat} to {at_end}.",
    "data_small": "There is little text for a model this size ({ratio} word pieces per parameter; {rule} is the rule of thumb): "
                  "it will start memorising the text before it has learned from it.",
    "fits": "The text and the model suit each other: {ratio} word pieces per parameter.",
    "keep": "For {hours}, keep this design and run {steps} steps: that reads about {ratio} word pieces per parameter.",
    "grow": "For {hours}, a design about {width} wide and {layers} layers deep ({params} parameters) fits best: {steps} steps read about "
            "{ratio} word pieces per parameter.",
    "gpu": "A {name} would run this design about {times}× faster than the {device} does. Auto keeps a model this small on the {device}; "
           "set the device to {kind} under the design's training settings to use the GPU.",
}


def hours_text(hours):
    """Hours in plain words: minutes under an hour, days from 48 hours."""
    if hours is None:
        return ""
    if hours < 1:
        return f"{max(1, round(hours * 60))} min"
    if hours >= 48:
        return f"{hours / 24:.1f} days"
    return f"{hours:.1f} h" if hours < 10 else f"{round(hours)} h"


def count_text(value):
    """A count the way a person says it: 133k, 4.2M, 545M."""
    for size, mark in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if value >= size:
            return f"{value / size:.0f}{mark}" if value >= 10 * size else f"{value / size:.1f}{mark}"
    return str(int(value))


def device_kind(spec, work, gpu):
    """The device a run of `spec` would use: its own setting, else the backend's "auto" rule (a small model stays on the CPU).
    `gpu` is studio_backend.gpu_status()."""
    from studio_backend import AUTO_GPU_MIN_WORK
    setting = ((spec.get("train") or {}).get("run") or {}).get("device") or "auto"
    if setting != "auto":
        return setting
    return gpu["kind"] if gpu and gpu.get("available") and work >= AUTO_GPU_MIN_WORK else "cpu"


def resized_spec(spec, width, layers):
    """A copy of `spec` at another width and depth (attention heads keep their size, at least 64 wide from width 256), or None when
    the design is hand-built per block or the result is not a valid design."""
    from studio_spec import validate_spec
    if spec.get("blocks"):
        return None
    new = copy.deepcopy(spec)
    new["dims"].update(d_model=width, n_layer=layers)
    mixer = new["block"]["mixer"]["config"]
    if "heads" in mixer:
        head = max(1, spec["dims"]["d_model"] // max(1, mixer["heads"]))
        head = max(head, 64) if width >= 256 else head
        mixer["heads"] = max(1, width // head)
        if "kv_heads" in mixer:
            mixer["kv_heads"] = math.gcd(mixer["heads"], mixer["kv_heads"])
    try:
        return validate_spec(new)
    except (ValueError, KeyError, TypeError):
        return None


def _seconds(spec, rows, accumulation, params, kind):
    """The page's own time model (studio_compute.step_seconds) for one step of `spec` on `kind`."""
    from studio_compute import step_seconds
    dims = spec["dims"]
    return step_seconds({"batch_size": rows, "context": dims["n_ctx"], "layers": dims["n_layer"], "accumulation": accumulation}, params, kind)


def _work(spec, rows):
    dims = spec["dims"]
    return rows * dims["n_ctx"] * dims["d_model"] * max(1, dims["n_layer"])  # what studio_backend.get_backend("auto") weighs


def suggest_design(spec, vocab, rows, accumulation, pieces, hours, measured, gpu):
    """The design and step count that fit `hours` on this computer: of the current design and the sizes in LADDER, the largest whose
    steps in that time (at most MAX_EPOCHS passes over `pieces`) still read the compute-optimal RULE word pieces per parameter; the
    smallest when none does. A step takes the measured time scaled by studio_compute.step_seconds, the page's own time model, so a
    wider design costs what it should and may move to the GPU under "auto". Returns {"width", "layers", "params", "steps", "hours",
    "tokens_per_param", "kind", "seconds_per_step", "spec"}, or None without a measured rate or for a hand-built design."""
    import studio_estimate
    from studio_compute import MAX_EPOCHS
    if not measured or spec.get("blocks") or hours <= 0:
        return None
    dims = spec["dims"]
    here = (dims["d_model"], dims["n_layer"])
    now = studio_estimate.estimate(spec, vocab)["total"]
    measured_kind = measured["device"]
    calibration = measured["seconds_per_step"] / _seconds(spec, rows, accumulation, now, measured_kind)  # the model's error on this computer
    per_step = rows * dims["n_ctx"] * accumulation
    found = []
    for width, layers in sorted({*LADDER, here}):
        candidate = spec if (width, layers) == here else resized_spec(spec, width, layers)
        if candidate is None:
            continue
        params = studio_estimate.estimate(candidate, vocab)["total"]
        kind = device_kind(candidate, _work(candidate, rows), gpu)
        seconds = _seconds(candidate, rows, accumulation, params, kind) * (calibration if kind == measured_kind else 1.)  # the other device was not measured
        tokens = min(hours * 3600 / seconds * per_step, MAX_EPOCHS * pieces if pieces else float("inf"))
        steps = max(1, int(tokens // per_step))
        found.append({"width": width, "layers": layers, "params": params, "steps": steps, "hours": steps * seconds / 3600,
                      "tokens_per_param": steps * per_step / params, "kind": kind, "seconds_per_step": seconds, "spec": candidate})
    if not found:
        return None
    fits = [c for c in found if c["tokens_per_param"] >= RULE]
    return max(fits, key=lambda c: c["params"]) if fits else min(found, key=lambda c: c["params"])


def gpu_note(found, rows, accumulation, gpu):
    """The sentence saying a GPU would be faster for design `found`, or "" when it would not, none is here, or it is already used."""
    if not gpu or not gpu.get("available") or found["kind"] != "cpu":
        return ""
    cpu = _seconds(found["spec"], rows, accumulation, found["params"], "cpu")
    fast = _seconds(found["spec"], rows, accumulation, found["params"], gpu["kind"])
    if fast >= GPU_WORTH * cpu:
        return ""
    return SIZING["gpu"].format(name=gpu.get("name") or gpu["kind"].upper(), times=f"{cpu / fast:.1f}", device="CPU", kind=gpu["kind"])


def sizing(spec, vocab, rows, accumulation, pieces, params, steps, hours, measured, evidence=None, gpu=None):
    """What the plan says about its size, as plain fields (None where unknown).

    `measured` is the step rate studio_scaling measured for this design here ({"seconds_per_step", "tokens_per_second", "device"}),
    `evidence` an earlier run of this same design from step 0 ({"points": [(word pieces read, validation loss)]}), `hours` the time
    the person gives. Returns {"verdict": model_too_small | data_too_small | balanced, "warning", "tokens_per_param",
    "seconds_per_step", "tokens_per_second", "device", "run_hours", "hours", "expected": {"basis": fit | rule, "loss_planned",
    "loss_flat", "flat_steps", "curve": [(step, loss)]} | None, "suggestion": {...} | None, "gpu_note"}."""
    per_step = rows * spec["dims"]["n_ctx"] * accumulation
    planned = steps * per_step
    ratio = planned / params
    seconds = measured["seconds_per_step"] if measured else None
    out = {"verdict": "balanced", "warning": "", "tokens_per_param": ratio, "seconds_per_step": seconds, "hours": hours,
           "tokens_per_second": measured["tokens_per_second"] if measured else None, "device": measured["device"] if measured else None,
           "run_hours": steps * seconds / 3600 if seconds else None, "expected": None, "suggestion": None, "gpu_note": ""}
    if ratio > TOO_SMALL_AT:
        out["verdict"] = "model_too_small"
        out["warning"] = SIZING["too_small"].format(ratio=count_text(ratio), times=f"{ratio / RULE:.0f}", rule=RULE)
    elif ratio < TOO_LITTLE_AT:
        out["verdict"] = "data_too_small"
        out["warning"] = SIZING["data_small"].format(ratio=f"{ratio:.1f}", rule=RULE)
    else:
        out["warning"] = SIZING["fits"].format(ratio=f"{ratio:.1f}")
    fit = scaling.fit_curve(evidence["points"]) if evidence else None
    if fit:
        first = min(x for x, _ in evidence["points"] if x > 0)
        flat = scaling.flat_point(fit, first, planned)
        flat_steps = int(flat // per_step)
        at_flat, at_end = scaling.predict(fit, flat), scaling.predict(fit, planned)
        out["expected"] = {"basis": "fit", "loss_planned": at_end, "loss_flat": at_flat, "flat_steps": flat_steps,
                           "curve": [(x / per_step, y) for x, y in scaling.curve_points(fit, max(first, per_step), planned)]}
        if flat < .6 * planned:
            out["verdict"] = "model_too_small"
            out["warning"] = SIZING["too_small_fit"].format(
                flat=f"{flat_steps:,}", flat_hours=hours_text(flat_steps * seconds / 3600) if seconds else "of this run",
                gap=f"{(at_flat - at_end) / at_flat:.1%}", rest=f"{steps - flat_steps:,}", at_flat=f"{at_flat:.2f}", at_end=f"{at_end:.2f}")
    suggestion = suggest_design(spec, vocab, rows, accumulation, pieces, hours, measured, gpu)
    if suggestion:
        same = (suggestion["width"], suggestion["layers"]) == (spec["dims"]["d_model"], spec["dims"]["n_layer"])
        out["gpu_note"] = gpu_note(suggestion, rows, accumulation, gpu)
        if not (same and out["verdict"] == "balanced" and abs(suggestion["steps"] - steps) <= .1 * steps):  # else the plan is the suggestion
            text = SIZING["keep" if same else "grow"].format(
                hours=hours_text(suggestion["hours"]), steps=f"{suggestion['steps']:,}", ratio=f"{suggestion['tokens_per_param']:.0f}",
                width=suggestion["width"], layers=suggestion["layers"], params=count_text(suggestion["params"]))
            out["suggestion"] = {"width": suggestion["width"], "layers": suggestion["layers"], "params": suggestion["params"],
                                 "steps": suggestion["steps"], "hours": suggestion["hours"], "tokens_per_param": suggestion["tokens_per_param"],
                                 "same_design": same, "text": text}
    return out
