"""The rules that end or redirect a DiLoCo run, as pure functions over a small state (the port of studio_passive's stop_reason,
plateaued, next_move and _next_pivot to rounds).

`studio_diloco.Coordinator._finish` calls `observe` once a round's `val_loss` is known and acts on the answer: continue, stop with a
plain reason, or go back to the best weights and apply the next pivot. Nothing here reads a clock or a file: the coordinator passes
`now` to `accrue` (its own, injectable clock), and `State` round-trips through `to_json`/`from_json` so diloco.json keeps it across a
restart or a failover.

The plateau test is studio_plateau.Plateau (#417), the rule single runs use, fed one point per round with the window in steps
(`plateau_checks` rounds of `inner_steps`); a pivot starts a new stage, which is judged on its own rounds."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import studio_plateau

PIVOTS = ("halve_lr", "next_dataset")  # drop_lessons and target_win_rate arrive with teacher polish and lessons (#508)
PIVOT_TEXT = {"halve_lr": "halved the learning rate", "next_dataset": "switched to the next data source"}


def _number(value, label, low=None, high=None, integer=False, allow_none=False, strict_low=False):
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (integer and int(value) != value):
        raise ValueError(f"{label} must be {'a whole number' if integer else 'a number'}.")
    if low is not None and (value <= low if strict_low else value < low):
        raise ValueError(f"{label} must be {'above' if strict_low else 'at least'} {low}.")
    if high is not None and value >= high:
        raise ValueError(f"{label} must be below {high}.")


def check(s):
    """Raise ValueError, in a sentence, when the rule options of the Settings `s` are not usable together."""
    _number(s.max_seconds, "max_seconds", 0, allow_none=True, strict_low=True)
    _number(s.patience_rounds, "patience_rounds", 0, integer=True)
    _number(s.target_val_loss, "target_val_loss", 0, allow_none=True, strict_low=True)
    if s.target_win_rate is not None:
        raise ValueError("target_win_rate is not available yet: it arrives with teacher polish in #508.")
    _number(s.plateau_checks, "plateau_checks", 0, integer=True)
    if s.plateau_checks == 1:
        raise ValueError("plateau_checks must be 0 (off) or at least 2.")
    _number(s.plateau_min_improvement, "plateau_min_improvement", 0, 1, strict_low=True)
    _number(s.worse_limit, "worse_limit", 0, integer=True)
    _number(s.keep_checkpoints, "keep_checkpoints", 0, integer=True)
    if not isinstance(s.teacher, str):
        raise ValueError("teacher must be a model name.")
    if "drop_lessons" in s.pivots:
        raise ValueError("The drop_lessons pivot is not available yet: lessons arrive with #508.")
    if any(p not in PIVOTS for p in s.pivots):
        raise ValueError("Each pivot must be one of: " + ", ".join(PIVOTS) + ".")
    if s.pivots and not (s.worse_limit or s.plateau_checks):
        raise ValueError("pivots need worse_limit or plateau_checks, or they would never run.")
    if s.pivots and s.worse_limit and not s.plateau_checks and s.patience_rounds and s.patience_rounds <= s.worse_limit:
        raise ValueError("patience_rounds must be above worse_limit, or the run ends before it tries any pivot.")


@dataclass
class State:
    """What the rules remember. `tick`/`counted` (the last clock reading and whether it counted as active time) are not saved."""
    active_seconds: float = 0.
    best_round: int | None = None
    best_loss: float | None = None
    rounds_since_best: int = 0
    worse_streak: int = 0
    pivot_index: int = 0
    lr_scale: float = 1.
    source: int = 0                                      # the data source being dealt
    points: list = field(default_factory=list)           # [step, val_loss] of the rounds since the last pivot (the plateau stage)
    stop_rule: str = ""                                  # which rule ended the run, once one has
    stop_text: str = ""                                  # ... and the sentence for it
    tick: float | None = None
    counted: bool = False

    def to_json(self):
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("tick", "counted")}

    @classmethod
    def from_json(cls, data):
        known = {f.name for f in fields(cls)} - {"tick", "counted"}
        return cls(**{key: value for key, value in (data or {}).items() if key in known})


def accrue(state, now, counting):
    """Add the time since the last call to the active seconds when that stretch counted (enough members were in), then note `now` and
    whether the stretch that starts here counts. Time below min_members never counts."""
    if state.tick is not None and state.counted:
        state.active_seconds += max(0., now - state.tick)
    state.tick, state.counted = now, bool(counting)


def plateaued(state, s):
    """True when the stage's validation losses (since the last pivot) improved by less than plateau_min_improvement over the last
    plateau_checks rounds: studio_plateau's rule with a window of that many rounds."""
    if not s.plateau_checks:
        return False
    plateau = studio_plateau.Plateau({"window_steps": s.plateau_checks * s.inner_steps, "window_seconds": None,
                                      "min_gain": s.plateau_min_improvement, "min_steps": 0})
    for step, loss in state.points:
        plateau.feed({"type": "metric", "step": step, "val_loss": loss})
    return plateau.hit is not None


def next_pivot(state, s, sources=1):
    """(index, name) of the next usable pivot in the list, or None. next_dataset needs a source after the current one."""
    for index in range(state.pivot_index, len(s.pivots)):
        name = s.pivots[index]
        if name == "next_dataset" and state.source + 1 >= sources:
            continue
        return index, name
    return None


def stop_reason(state, s):
    """(rule, sentence) for the first stop rule that holds, or None. The rules are the ones the person set; none set means None."""
    if s.max_seconds is not None and state.active_seconds >= s.max_seconds:
        return "max_seconds", f"Stopped: the run trained for {state.active_seconds:,.0f} seconds, its limit of {s.max_seconds:,.0f}."
    if s.patience_rounds and state.best_loss is not None and state.rounds_since_best >= s.patience_rounds:
        return "patience", f"Stopped: {state.rounds_since_best} rounds in a row without a better validation loss."
    if s.target_val_loss is not None and state.best_loss is not None and state.best_loss <= s.target_val_loss:
        return "target_val_loss", f"Stopped: the validation loss reached {state.best_loss:.4g}, at or under the target {s.target_val_loss:.4g}."
    return None


def observe(state, s, number, step, val_loss, sources=1, final=False):
    """Fold the closed round `number` (its `step` and `val_loss`) into `state` and say what to do next. `final`: the run ends with this
    round anyway, so only the best is kept and no rule or pivot applies. A round without a finite loss counts as worse only once there
    is a best to be worse than. Returns
    {"action": "continue" | "stop" | "pivot", "improved": bool, ...}: a stop carries its "rule" and "text"; a pivot its "name",
    "before" (this round's loss), "best" (the loss it goes back to), "best_round" and "why" (what triggered it)."""
    finite = val_loss is not None and math.isfinite(val_loss)
    improved = finite and (state.best_loss is None or val_loss < state.best_loss)
    if improved:
        state.best_round, state.best_loss, state.rounds_since_best, state.worse_streak = number, float(val_loss), 0, 0
    elif state.best_loss is not None:
        state.rounds_since_best += 1
        state.worse_streak += 1
    if finite:
        state.points.append([step, float(val_loss)])
    if final:
        return {"action": "continue", "improved": improved}
    found = stop_reason(state, s)
    if found:
        state.stop_rule, state.stop_text = found
        return {"action": "stop", "improved": improved, "rule": found[0], "text": found[1]}
    why = ("" if state.best_loss is None else "worse" if s.worse_limit and state.worse_streak >= s.worse_limit
           else "plateau" if plateaued(state, s) else "")
    if not why:
        return {"action": "continue", "improved": improved}
    streak = state.worse_streak
    move = next_pivot(state, s, sources)
    if move is None:
        if why == "plateau" and not s.pivots:
            rule, text = "plateau", (f"Stopped: the validation loss improved by less than {s.plateau_min_improvement:.2%} "
                                     f"over {s.plateau_checks} rounds.")
        else:
            rule, text = "pivots", (f"Stopped: the validation loss {'got worse ' + str(streak) + ' rounds in a row' if why == 'worse' else 'stopped improving'}"
                                    f" and there is no change left to try.")
        state.stop_rule, state.stop_text = rule, text
        return {"action": "stop", "improved": improved, "rule": rule, "text": text}
    index, name = move
    state.pivot_index, state.worse_streak, state.rounds_since_best, state.points = index + 1, 0, 0, []
    if name == "halve_lr":
        state.lr_scale /= 2
    elif name == "next_dataset":
        state.source += 1
    return {"action": "pivot", "improved": improved, "name": name, "before": val_loss, "best": state.best_loss,
            "best_round": state.best_round, "why": why, "streak": streak}
