"""Plateau rule: watch a run's validation loss and say when it has stopped improving.

`Plateau` reads the events `studio_model.train` already emits ("history" at the start, then "metric" rows) and raises a flag; `Flagged`
is the stop event training polls, which is the caller's stop or that flag. Nothing here touches the engine. A plateau ends a stage of
a run, not the run: studio_jobs decides what comes next (end it with the reason, or move on to other data).

The rule: at a validation point, the best loss over the whole stage is compared with the best loss up to `window` steps ago (or
`window_seconds` of training ago). If it improved by less than `min_gain` of the older best, the curve is flat."""
from __future__ import annotations

PARAMS = {"window_steps": None, "window_seconds": None, "min_gain": .01, "min_steps": 0}
LONG_RUN_SECONDS = 3600  # the plan turns the rule on by default for runs expected to take longer than this
WINDOW_SHARE = .05  # default window: this share of the run's planned steps
MIN_WINDOW = 500


def default_rule(total_steps, min_gain=.01):
    """The rule a plan gives a run of `total_steps`: a window of 5% of them (at least MIN_WINDOW), and no stop before two windows."""
    window = max(MIN_WINDOW, int(total_steps * WINDOW_SHARE))
    return {"window_steps": window, "window_seconds": None, "min_gain": min_gain, "min_steps": 2 * window}


class Plateau:
    """The rule, fed with one run's events. `hit` becomes a dict {"step", "best", "older_best", "gain", "window"} once it fires."""

    def __init__(self, rule, use_history=True):
        """`use_history` False ignores the "history" event: a rebased stage is judged on its own new text, not on the losses the
        checkpoint carries from the text before."""
        self.use_history = use_history
        rule = {**PARAMS, **(rule or {})}
        self.window_steps, self.window_seconds = rule["window_steps"], rule["window_seconds"]
        self.min_gain, self.min_steps = float(rule["min_gain"]), int(rule["min_steps"] or 0)
        if not (self.window_steps or self.window_seconds):
            raise ValueError("A plateau rule needs a window in steps or in seconds.")
        if not 0 < self.min_gain < 1:
            raise ValueError("The plateau gain must be between 0 and 1.")
        self.points, self.hit = [], None  # (step, seconds, val_loss)

    def feed(self, event):
        """Take one training event; returns the plateau dict when this event made the rule fire, else None."""
        kind = event.get("type")
        if kind == "history" and not self.use_history:
            return None
        rows = event.get("history") or [] if kind == "history" else [event] if kind == "metric" else []
        found = None
        for row in rows:
            if row.get("val_loss") is None or not row.get("step"):
                continue
            self.points.append((int(row["step"]), row.get("elapsed_seconds"), float(row["val_loss"])))
            found = self._check() or found
        return found

    def _check(self):
        if self.hit:
            return None
        step, seconds, _ = self.points[-1]
        if step < self.min_steps:
            return None
        if self.window_steps:
            older = [v for s, _, v in self.points if s <= step - self.window_steps]
            window = f"{self.window_steps:,} steps"
        else:
            if seconds is None:
                return None
            older = [v for s, t, v in self.points if t is not None and t <= seconds - self.window_seconds]
            window = f"{self.window_seconds / 3600:.1f} h"
        if not older:
            return None
        before, best = min(older), min(v for _, _, v in self.points)
        gain = (before - best) / before
        if gain >= self.min_gain:
            return None
        self.hit = {"step": step, "best": best, "older_best": before, "gain": gain, "window": window, "min_gain": self.min_gain}
        return self.hit


class Flagged:
    """A stop event for training: set when the caller's `stop` is, or when `flag` says the plateau rule fired. Training only polls
    `is_set()`; `user_stopped()` tells the two apart afterwards."""

    def __init__(self, stop, plateau):
        self.stop, self.plateau = stop, plateau

    def is_set(self):
        return bool(self.stop is not None and self.stop.is_set()) or self.plateau.hit is not None

    def user_stopped(self):
        return bool(self.stop is not None and self.stop.is_set())

    def set(self):
        if self.stop is not None:
            self.stop.set()

    def wait(self, timeout=None):
        return self.stop.wait(timeout) if self.stop is not None else False


def reason_text(hit, best_path=True):
    """The sentence the run page shows for a run the plateau rule ended."""
    return (f"Stopped at step {hit['step']:,} because the best validation loss ({hit['best']:.3f}) improved by only {hit['gain']:.2%} over "
            f"the last {hit['window']} (the rule stops below {hit.get('min_gain', .01):.0%}). "
            + ("The best checkpoint is kept. " if best_path else "")
            + "To learn more, continue with a bigger design (a warm start keeps what it learned) or with more text.")
