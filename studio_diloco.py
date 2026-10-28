"""DiLoCo data-parallel pretraining across nodes (Douillard et al. 2023, arXiv:2311.08105; docs/distributed.md, "Training runtime").

One run has a Coordinator, on the owner node, and Members, each a process on some node. The coordinator holds the global weights
theta (fp32), the outer momentum, the validation windows and the shard deal; a member trains. Round r:

  1. every member holds theta of round r and the training shards dealt to it;
  2. each runs H inner AdamW steps with `studio_model.train` (through its `distributed=` hook), the learning rate taken from the
     GLOBAL schedule at r * H + step, its AdamW state kept from round to round;
  3. it sends delta_k = theta - theta_k and its target-token count t_k (partial work counts for what it is, by t_k);
  4. the coordinator averages delta = sum t_k delta_k / sum t_k and takes the outer Nesterov step
     v = mu v + delta; theta = theta - lr (delta + mu v); evaluates theta on the validation windows; writes last.npz (the
     ordinary checkpoint) and diloco.json; opens round r + 1.

The coordinator is plain logic with an injected clock: every rule (deadline, drop, stale 409, elastic membership, restart) is a
window-free test. Its wire protocol is studio_api.routers.collective; members reach it through any client with
studio_peer.PeerClient's methods. `python -m studio_diloco member --coordinator URL --run ID` runs a member by hand.

A member can also overlap its communication with its compute (`overlap=True`, `--overlap`; the "overlapping" variant of Streaming
DiLoCo, arXiv:2501.18512): the delta of round r uploads, and the weights of round r + 1 download, on a background thread while the
next H inner steps already run, and the outer result is applied one round late: when the new theta arrives the member keeps the
work it did meanwhile, theta' + (w - w_start), instead of throwing it away. The coordinator is not told and does not change.
"""
from __future__ import annotations

import argparse
import bisect
import copy
import gc
import hashlib
import io
import json
import logging
import math
import random
import re
import shutil
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

import studio_run_health as run_health
import studio_shards
from studio_failover import POLL_BACKOFF_MAX, POLL_EVERY
import studio_diloco_rules as drules

FORMAT = 1
TERM_HEADER = "X-Run-Term"  # every answer on /collective/{run} carries the coordinator's term here and in its JSON body
TERM_STALE = "Stale term"  # the start of the 409 message for a term that is not the coordinator's
LONG_POLL = 30.  # seconds a GET /rounds/{r} may wait for a change
RELOCATE_SECONDS = POLL_BACKOFF_MAX + POLL_EVERY + 5.  # the wait for the claim that says where a run went: a peer in poll backoff is next asked within POLL_BACKOFF_MAX, then the failover poll, plus margin
STATE_PREFIXES = ("theta::", "momentum::")
DELTA_DTYPES = ("int8", "float16", "float32")
GONE_RUN = "No distributed run"  # the start of the 404 detail of a run the coordinator's app does not hold (routers.collective.coordinator)
FINISHED = "The run is finished."  # the 409 detail of a join to a run that has ended
STATE_FILE = "diloco.json"  # the entry of a GET /state archive that holds diloco.json's bytes
SCALE_SUFFIX = "::scale"  # an int8 array's float32 scales (one per INT8_BLOCK values) sit beside it in the archive under this suffix
INT8_BLOCK = 256  # values that share one int8 scale
INDEX_SUFFIX = "::idx"  # a sparse array's int32 flat indices (sorted) sit beside its int8 values and "::scale" under this suffix
SHAPE_SUFFIX = "::shape"  # ... and its dense shape under this one
MAX_SPARSITY = .2  # a kept entry costs 5 bytes (int8 + int32 index) against about 1 dense, so above this share sparse is no smaller
UPDATE_KEY = "__update_from__"  # in a params body: the body is the outer update from this round's theta, not theta itself
RUN_ID = r"^[A-Za-z0-9_.-]{1,64}$"
PHASES = ("downloading", "loading", "training", "uploading", "waiting")  # what a member says it is doing (POST /progress)
WORK_PHASES = ("training", "uploading")  # the phases that belong to one open round: a report in them for a closed round is stale
MARKS = ("theta_taken", "first_step", "delta_sent", "delta_received")  # a round's timeline per member, in the order they happen
PROGRESS_SECONDS = 2.  # a member reports at least this often while it runs ...
PROGRESS_STEPS = 20  # ... or after this many steps, whichever comes first (never twice within PROGRESS_GAP)
MIN_SHARE = .1  # a member is dealt at least this fraction of H steps (deal_steps) ...
MAX_SHARE = 3  # ... and at most this many times H
PROBE_SHARE = .25  # a member with no measurement yet is dealt this fraction of H: a short first round that times it
SPEED_BAND = 1.05  # measured speeds within this ratio count as equal: everyone keeps H (real hardware is never exactly equal)
STEP_TIME_SMOOTHING = .5  # the weight of the latest round in a member's smoothed seconds per step
PROGRESS_GAP = .5
PROGRESS_TIMEOUT = 5.  # seconds a progress report may take; a slow or failing one is dropped
PROGRESS_WAIT = .25  # seconds a progress report or a snapshot waits for a round that is closing before it answers without it
EVENTS_KEPT = 200  # the coordinator's event log is bounded
TIMELINES_KEPT = 8  # rounds whose timeline stays in memory (every finished round's also goes to metrics.jsonl)
SAMPLES_KEPT = 240  # progress samples kept per member for the open round
EVAL_PROMPTS = 12  # prompts of the head-to-head against the starting weights (target_win_rate)
MIN_EVAL_PROMPTS = 4  # fewest held-out prompts a win-rate stop may rest on: with 3 or fewer one prompt moves the rate by a third or more
PROMPTS_PER_POLISH = 500  # the teacher is given this many training prompts per polish, a new window each time
LESSON_ITEMS_SHOWN = 50  # how many of a lesson's answers lesson_list shows (the rest are counted)


def _prompt_of(messages):
    """The question a lesson row answers: its last user message (else its last message), as text."""
    asked = [m for m in messages if m.get("role") == "user"] or list(messages)
    return str(asked[-1].get("content", "")) if asked else ""


class Background:
    """`fn` running on a daemon thread. result() waits for it and returns what it returned, or raises what it raised."""

    def __init__(self, fn, name="diloco-exchange"):
        self.value, self.error = None, None
        self.thread = threading.Thread(target=self._run, args=(fn,), name=name, daemon=True)
        self.thread.start()

    def _run(self, fn):
        try:
            self.value = fn()
        except BaseException as exc:  # noqa: BLE001 - handed to whoever asks for the result
            self.error = exc

    def result(self):
        self.thread.join()
        if self.error is not None:
            raise self.error
        return self.value

    def abandon(self, timeout):
        self.thread.join(timeout)


@dataclass
class Entered:
    """A round a member has taken: its theta, the member's place in it and the shards it deals. `skipped`: the round it asked
    for had closed without it, so this is a later one."""
    round: int
    theta: dict
    generation: int
    rank: int
    deadline_s: float
    shards: list
    skipped: bool = False
    lr_scale: float = 1.  # the factor the run's pivots put on the global learning-rate schedule
    steps: int = 0  # the inner steps dealt to this member for the round (0: the run's H, as a coordinator that deals none says)
    train_s: float = 0.  # time-boxed round: the seconds to train for (0: train the dealt steps)
    bytes: int = 0  # what the download of theta (or its update) was on the wire
    rounds: int = 0  # the run's rounds as the coordinator had them when it opened this round (0: an older coordinator, which says none)
    plan: list | None = None  # the run's plan (plan_h) then, so a member learns of an extension from the rounds it takes


class Stale(RuntimeError):
    """A request from an earlier round, generation or lease epoch, or into a closed or full run: the API answers 409."""


class Gone(RuntimeError):
    """The run is over (stopped or finished) on this coordinator: it hands out no more weights or shards (the API answers 410)."""


class StaleTerm(Stale):
    """A request under a lower term than the coordinator's, or a coordinator that has seen a higher term and no longer coordinates
    the run. `term` is the highest term involved. The message begins TERM_STALE so a member can tell it from the other 409s."""

    def __init__(self, message, term):
        super().__init__(message)
        self.term = term


class Rebound(Exception):
    """Raised inside a member whose coordinator is gone, when the run is coordinated elsewhere now (studio_failover). `found` is the
    (node, term, client) of the coordinator in charge; Member.work catches it and rejoins there."""

    def __init__(self, found):
        super().__init__(f"The run is coordinated on {found[0]} at term {found[1]} now.")
        self.found = found


class Halt:
    """The stop studio_model.train sees: the job's stop, or the end of this attempt (the run finished, or the member was fenced out of
    its coordinator and may rebind to the one now in charge)."""

    def __init__(self, stop, ended):
        self.stop, self.ended = stop, ended

    def is_set(self):
        return self.stop.is_set() or self.ended.is_set()

    def set(self):
        self.ended.set()


REMOTE_INNER_STEPS = 1500  # H when a member is on another machine: moving the model is a fixed cost per round, so a longer round hides it


def default_inner_steps(remote: bool) -> int:
    """The H a run starts from: the usual one on this computer alone, a larger one when a member is on another machine."""
    return REMOTE_INNER_STEPS if remote else Settings.inner_steps


@dataclass(frozen=True)
class Settings:
    """A run's DiLoCo parameters (the job's params). outer_lr 1 with momentum 0 makes one member exactly plain training."""
    inner_steps: int = 100          # H
    rounds: int = 10
    outer_lr: float = .7
    outer_momentum: float = .9
    min_members: int = 1            # below this the run pauses; it continues above
    max_members: int = 8
    shards: int = 64                # S: fixed training shards, dealt to the live members each round
    delta_dtype: str = "int8"    # the precision of a member's change AND of the outer update members download (float32: members download theta itself)
    delta_sparsity: float = 0.      # int8 only: the share of each tensor's entries (the largest by magnitude) a member sends; 0: all of them. The rest is kept for the next change. The outer update download stays dense
    time_box: bool = False          # members train until the round's training deadline (the fastest member's H-step time) instead of a fixed number of steps
    slack_ratio: float = .25        # round deadline = fastest member's step time * H * (1 + ratio) + slack_seconds
    slack_seconds: float = 30.
    grace_seconds: float = 120.     # after the deadline, for the member's download of theta and upload of its delta
    first_round_seconds: float = 900.  # the deadline of a round nobody has timed yet
    missed_limit: int = 3           # rounds in a row without a contribution before a member is dropped from the run
    replicas: int = 2               # R: the lowest-ranked members keep a copy of the latest round, so it outlives the coordinator
    # stop rules, plateau and pivots (studio_diloco_rules): all off by default, so a run goes to `rounds`
    max_seconds: float | None = None   # active wall-clock; time with fewer than min_members does not count
    patience_rounds: int = 0           # rounds without a better val_loss
    target_val_loss: float | None = None
    target_win_rate: float | None = None  # stop once the best weights win this share of head-to-head replies against the starting weights (judged by the teacher)
    versus_seconds: float = 600.       # wall-clock limit of one head-to-head against the starting weights; past it the comparison is skipped and rounds go on
    teacher: str = ""                  # the teacher model's name (an Ollama model); needed by polish_every and target_win_rate
    plateau_checks: int = 0            # rounds the val_loss must have improved within, else it is a plateau (0: off)
    plateau_min_improvement: float = .01
    worse_limit: int = 0               # rounds in a row worse than the best before the coordinator goes back to the best and pivots (0: off)
    pivots: tuple = ()                 # what to try, in order: halve_lr (also the outer learning rate), next_dataset
    keep_checkpoints: int = 0          # per-round checkpoints (round_<n>.npz) kept in the run folder; 0: none besides last.npz and best.npz
    # teacher polish and lessons (#508): after every polish_every-th closed round (never the last), the coordinator has the teacher polish theta
    polish_every: int = 0              # rounds between polishes; 0: off
    polish: bool = True                # keep the polished weights when they won the teacher's head-to-head and are no worse on validation
    lessons: bool = True               # train on the teacher's better answers as extra shards

    def __post_init__(self):
        if self.inner_steps < 1 or self.rounds < 1 or self.shards < 1:
            raise ValueError("inner_steps, rounds and shards must be at least 1.")
        from studio_model import MAX_STEPS
        if self.rounds * self.inner_steps > MAX_STEPS:
            raise ValueError(f"{self.rounds:,} rounds of {self.inner_steps:,} steps is {self.rounds * self.inner_steps:,} steps; a run trains at most {MAX_STEPS:,}. "
                             "Use fewer rounds or fewer steps per round.")
        if not 1 <= self.min_members <= self.max_members:
            raise ValueError("min_members must be between 1 and max_members.")
        if self.delta_dtype not in DELTA_DTYPES:
            raise ValueError(f"delta_dtype must be one of {', '.join(DELTA_DTYPES)}.")
        if not 0 <= self.delta_sparsity <= MAX_SPARSITY:
            raise ValueError(f"delta_sparsity must be between 0 (send everything) and {MAX_SPARSITY}.")
        if self.delta_sparsity > 0 and self.delta_dtype != "int8":
            raise ValueError("delta_sparsity needs delta_dtype int8.")
        if self.replicas < 0:
            raise ValueError("replicas must not be negative.")
        if self.outer_lr <= 0 or not 0 <= self.outer_momentum < 1:
            raise ValueError("outer_lr must be positive and outer_momentum in [0, 1).")
        object.__setattr__(self, "pivots", tuple(self.pivots))  # a list from JSON
        drules.check(self)

    @classmethod
    def from_params(cls, params, steps=None):
        """Settings from a job's params; `rounds` defaults to the config's steps divided by H."""
        known = {key: params[key] for key in cls.__dataclass_fields__ if key in params}
        if "rounds" not in known and steps:
            known["rounds"] = max(1, int(steps) // int(known.get("inner_steps", cls.inner_steps)))
        return cls(**known)


# ---- the learning-rate plan -------------------------------------------------------------------------------------------
# A run's length can grow while it goes (Coordinator.extend), and so can H. The plan is the list of those changes:
# [{"round": the first round it holds for, "q": the share of the learning-rate schedule done when that round starts, "h": inner steps a
# round runs, "rounds": the run's rounds when the change was made}]. The first entry is {"round": 0, "q": 0, "h": H, "rounds": rounds}.
# The schedule of an entry's rounds is laid over its frame: the steps of the rounds before it plus its own H for every round up to
# its `rounds`. From the entry's round on, the share left (1 - q) is spread over the steps left in the frame, so the rate carries on
# from where the entry before it had it instead of jumping back up when the run gets longer.

def plan_entry(plan, number):
    """The index of the plan entry round `number` is under: the last whose round is not after it."""
    return max(k for k, entry in enumerate(plan) if entry["round"] <= number)


def plan_h(plan, number):
    """The inner steps round `number` runs (H)."""
    return plan[plan_entry(plan, number)]["h"]


def plan_before(plan, number):
    """The inner steps of rounds 0 .. number-1, together."""
    ends = [entry["round"] for entry in plan[1:]] + [number]
    return sum(entry["h"] * max(0, min(number, end) - entry["round"]) for entry, end in zip(plan, ends))


def plan_frame(plan, number):
    """The length the learning-rate schedule has in round `number`, in steps: the frame of its entry."""
    k = plan_entry(plan, number)
    return plan_before(plan[:k + 1], plan[k]["rounds"])


def plan_rounds(plan):
    """The rounds the run has as it stands."""
    return plan[-1]["rounds"]


def plan_total(plan):
    """The inner steps of the whole run as it stands: the frame of its last entry."""
    return plan_before(plan, plan[-1]["rounds"])


def plan_position(plan, number, inside=0.):
    """Where on its frame round `number` stands after `inside` of its steps (0 .. plan_frame, in steps)."""
    entry, total = plan[plan_entry(plan, number)], plan_frame(plan, number)
    start = plan_before(plan, entry["round"])
    done = plan_before(plan, number) - start + inside
    if not entry["q"] and not start:
        return min(done, total)
    return (entry["q"] + (1 - entry["q"]) * min(1., done / max(1, total - start))) * total


def plan_carry(position, old_frame, new_frame, warmup=0):
    """Where a schedule that stood at `position` of a frame of `old_frame` steps stands on a frame of `new_frame` steps, so the rate
    carries on: the warmup is a fixed number of steps, so a position inside it stays where it is, and past it the share of the
    post-warmup span done stays the same (a plain share of the whole frame would move the warmup's end)."""
    warmup = min(max(0, int(warmup)), old_frame - 1, new_frame - 1)
    if position <= warmup:
        return position
    return warmup + (position - warmup) * (new_frame - warmup) / max(1, old_frame - warmup)


# ---- pure rules -------------------------------------------------------------------------------------------------------

def token_average(contributions):
    """sum t_k delta_k / sum t_k over [(tokens, {name: delta})]; a member with no target tokens counts for nothing."""
    total = sum(tokens for tokens, _ in contributions)
    if total <= 0:
        return None
    out = {}
    for tokens, delta in contributions:
        for name, value in delta.items():
            out[name] = out.get(name, 0.) + value.astype(np.float32) * np.float32(tokens / total)
    return {name: value.astype(np.float32) for name, value in out.items()}


def outer_step(theta, momentum, delta, lr, mu):
    """The outer Nesterov step on the pseudo-gradient, in place: v = mu v + delta; theta = theta - lr (delta + mu v)."""
    for name, gradient in delta.items():
        momentum[name] *= np.float32(mu)
        momentum[name] += gradient
        theta[name] -= np.float32(lr) * (gradient + np.float32(mu) * momentum[name])


def replica_nodes(members, count, prefer=()):
    """The replicas of a round: the `count` lowest-ranked of its members (a member's rank is its place in the round's list), except
    that the members in `prefer` (the node the person chose to coordinate the run) come first, so it holds the run in time to take it over."""
    order = list(members)
    first = [node for node in order if node in prefer]
    return (first + [node for node in order if node not in first])[:max(0, count)]


def deal_ids(seed, round_number, members, ids):
    """{member: sorted shard ids}: the shards `ids` shuffled by (seed, round, membership) and dealt round-robin, so a rerun with the
    same membership history deals the same, and one member always holds every shard in order."""
    order = list(ids)
    random.Random(f"{seed}:{round_number}:{','.join(members)}").shuffle(order)
    return {member: sorted(order[index::len(members)]) for index, member in enumerate(members)}


def deal_shards(seed, round_number, members, shard_count, first=0):
    """deal_ids for the shards first..first+shard_count-1 (one data source's)."""
    return deal_ids(seed, round_number, members, range(first, first + shard_count))


def round_order(seed, rank, number, dealt):
    """The order a member takes its dealt shards in for a round: shuffled by (seed, rank, round), so the shards it trains on differ
    from round to round while the choice stays the same on every run of the deal."""
    order = sorted(dealt)
    random.Random(f"{seed}:{rank}:{number}:order").shuffle(order)
    return order


def sampler_seed(seed, rank, round_number):
    return int.from_bytes(hashlib.sha256(f"{seed}:{rank}:{round_number}".encode()).digest()[:8], "big")


def round_seconds(settings, step_time, steps=None):
    """A round's deadline from the slowest predicted finish: a member's measured seconds per step times the steps dealt to it (H when
    not given; None step_time: nobody has been timed yet)."""
    if not step_time:
        return settings.first_round_seconds
    return step_time * (steps or settings.inner_steps) * (1 + settings.slack_ratio) + settings.slack_seconds


def deal_steps(settings, members, step_times):
    """{member: inner steps} for a round, so that the members' predicted finish times (steps * measured seconds per step) match.
    The measured members share the work an equal-H round would give them (H each) in proportion to their speed (the rounding
    remainder goes to the largest fractions, so the total stays H per measured member), each share clamped to [MIN_SHARE, MAX_SHARE]
    of H. Equal speeds (within SPEED_BAND), one member, or no measurement give H to everyone; a member with no measurement gets a
    probe round of PROBE_SHARE of H beside measured ones. A clamped share breaks the match of finish times, so the round runs as long
    as the slowest clamped member needs (the deadline follows it)."""
    h = settings.inner_steps
    measured = {node: step_times[node] for node in members
                if isinstance(step_times.get(node), (int, float)) and 0 < step_times[node] < math.inf}
    if not measured:
        return {node: h for node in members}
    speeds = {node: 1. / seconds for node, seconds in measured.items()}
    probe = min(h, max(1, round(h * PROBE_SHARE)))
    if max(speeds.values()) <= SPEED_BAND * min(speeds.values()):
        return {node: h if node in measured else probe for node in members}
    low, high = max(1, math.ceil(h * MIN_SHARE)), max(1, int(h * MAX_SHARE))
    work, total = h * len(measured), sum(speeds.values())
    exact = {node: work * speed / total for node, speed in speeds.items()}
    share = {node: math.floor(value) for node, value in exact.items()}
    for node in sorted(exact, key=lambda n: exact[n] - share[n], reverse=True)[:work - sum(share.values())]:
        share[node] += 1
    return {node: min(high, max(low, share[node])) if node in share else probe for node in members}


def train_seconds(settings, members, step_times):
    """The seconds every member trains for in a time-boxed round: the fastest measured member's time for H steps, so it does H steps and
    no machine waits for a slower one (a slower one does fewer, and its change counts for the tokens it trained). 0 when the run is not
    time-boxed or nobody has been timed yet (the first round runs fixed steps, and times them)."""
    measured = [step_times[node] for node in members
                if isinstance(step_times.get(node), (int, float)) and 0 < step_times[node] < math.inf]
    return settings.inner_steps * min(measured) if settings.time_box and measured else 0.


def timed_steps(settings, members, step_times, train):
    """{member: the inner steps it is expected to do in `train` seconds}: what its learning-rate schedule is spread over (H for a member
    nobody has timed). The round ends by the clock, not by this count."""
    h = settings.inner_steps
    return {node: min(h * MAX_SHARE, max(1, round(train / step_times[node]))) if 0 < (step_times.get(node) or 0) < math.inf else h
            for node in members}


def quantize_int8(value, block=INT8_BLOCK):
    """(int8 values, float32 scales) for one tensor, one scale per `block` consecutive values of its flattened form: scale =
    max |x| / 127 over the block, values round(x / scale). A block with a non-finite value gets a NaN scale, so it decodes
    non-finite and the coordinator refuses the change as it would any other."""
    value = np.asarray(value, dtype=np.float32)
    flat = value.reshape(-1)
    count = -(-flat.size // block)
    padded = np.zeros(count * block, dtype=np.float32)
    padded[:flat.size] = flat
    blocks = padded.reshape(count, block)
    with np.errstate(invalid="ignore"):
        peak = np.max(np.abs(blocks), axis=1) if count else np.zeros(0, dtype=np.float32)
    finite = np.isfinite(peak)
    scale = np.where(finite, peak / np.float32(127.), np.float32("nan")).astype(np.float32)
    divisor = np.where(finite & (scale > 0), scale, np.float32(1.)).astype(np.float32)
    clean = np.where(finite[:, None], blocks, np.float32(0.))
    values = np.clip(np.rint(clean / divisor[:, None]), -127, 127).astype(np.int8)
    return values.reshape(-1)[:flat.size].reshape(value.shape), scale


def dequantize_int8(values, scale, block=INT8_BLOCK):
    """The fp32 tensor of quantize_int8's output; a 0-d scale (one per tensor) is also read."""
    scale = np.asarray(scale, dtype=np.float32)
    if scale.ndim == 0:
        return values.astype(np.float32) * scale
    flat = values.reshape(-1)
    if len(scale) != -(-flat.size // block):
        raise ValueError("An int8 tensor's scales do not match its size.")
    return (flat.astype(np.float32) * scale[np.arange(flat.size) // block]).reshape(values.shape)


def top_k(value, sparsity):
    """The sorted flat indices of the ceil(sparsity * size) entries of `value` with the largest magnitude, or None when keeping that
    few is not smaller than sending the tensor whole. A non-finite entry ranks first, so it is sent and the coordinator refuses it."""
    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    count = max(1, math.ceil(sparsity * flat.size))
    if count * 5 >= flat.size:
        return None
    with np.errstate(invalid="ignore"):
        magnitude = np.where(np.isfinite(flat), np.abs(flat), np.float32(np.inf))
    return np.sort(np.argpartition(magnitude, flat.size - count)[flat.size - count:]).astype(np.int32)


def encode_arrays(arrays, dtype="float32", update_from=None, sparsity=0.):
    """A {name: array} as an uncompressed .npz body; float16 is clipped to its range first; int8 stores each tensor as int8 values
    plus one float32 scale per INT8_BLOCK values (name + "::scale"), a quarter of the fp32 size. With `sparsity` (int8 only) a tensor
    keeps just its largest entries by magnitude: their int8 values and scales, their int32 flat indices (name + "::idx") and the dense
    shape (name + "::shape"); a tensor too small for that to pay is sent dense. `update_from` marks the body as an
    outer update from that round's theta (see Coordinator.params_bytes)."""
    buffer = io.BytesIO()
    extra = {} if update_from is None else {UPDATE_KEY: np.array(update_from, dtype=np.int64)}
    if dtype == "float16":
        np.savez(buffer, **{name: np.clip(value, -65504., 65504.).astype(np.float16) for name, value in arrays.items()}, **extra)
    elif dtype == "int8":
        out = {}
        for name, value in arrays.items():
            keep = top_k(value, sparsity) if sparsity > 0 else None
            if keep is None:
                out[name], out[name + SCALE_SUFFIX] = quantize_int8(value)
            else:
                out[name], out[name + SCALE_SUFFIX] = quantize_int8(np.asarray(value, dtype=np.float32).reshape(-1)[keep])
                out[name + INDEX_SUFFIX], out[name + SHAPE_SUFFIX] = keep, np.asarray(np.shape(value), dtype=np.int64)
        np.savez(buffer, **out, **extra)
    else:
        np.savez(buffer, **{name: np.asarray(value, dtype=np.float32) for name, value in arrays.items()}, **extra)
    return buffer.getvalue()


def encode_feedback(arrays, dtype, residual=None, update_from=None, sparsity=0.):
    """(body, residual) of `arrays` plus the residual of the last call, sent as `dtype` (with `sparsity`: the top-k entries only): the
    part of what was meant to go that the encoding lost (the entries not kept, then the quantization of the ones kept; non-finite
    entries are left out), to be added to the next call's arrays (error feedback). The order is residual, then top-k, then int8."""
    if residual:
        arrays = {name: value + residual[name] for name, value in arrays.items()}
    body = encode_arrays(arrays, dtype, update_from, sparsity)
    sent = decode_arrays(body)
    lost = {name: np.asarray(value, dtype=np.float32) - sent[name] for name, value in arrays.items()}
    return body, {name: np.where(np.isfinite(value), value, np.float32(0.)).astype(np.float32) for name, value in lost.items()}


def decode_arrays(data):
    """The fp32 arrays of an .npz body. ValueError when it is not one."""
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            out = {}
            for name in archive.files:
                if name == UPDATE_KEY or any(name.endswith(suffix) and name[:-len(suffix)] in archive.files
                                             for suffix in (SCALE_SUFFIX, INDEX_SUFFIX, SHAPE_SUFFIX)):
                    continue
                value = archive[name]
                if value.dtype == np.int8:
                    value = dequantize_int8(value, archive[name + SCALE_SUFFIX])
                if name + INDEX_SUFFIX in archive.files:  # sparse: the kept entries go back where they were, zero elsewhere
                    index, shape = archive[name + INDEX_SUFFIX], archive[name + SHAPE_SUFFIX]
                    dense = np.zeros(int(np.prod(shape)), dtype=np.float32)
                    if index.shape != value.shape or (index.size and (index.min() < 0 or index.max() >= dense.size)):
                        raise ValueError("A sparse tensor's indices do not match its values.")
                    dense[index] = value
                    value = dense.reshape(tuple(int(n) for n in shape))
                out[name] = value.astype(np.float32)
            return out
    except Exception as exc:  # noqa: BLE001 - zip and numpy raise many types for a bad body
        raise ValueError(f"The body is not an .npz archive of arrays ({exc}).") from None


def update_origin(data):
    """The round whose theta a params body is an update from, or None when the body is theta itself."""
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return int(archive[UPDATE_KEY]) if UPDATE_KEY in archive.files else None


def add_update(theta, applied):
    """theta + the dequantized update, new arrays: the one operation the coordinator and every member apply, so every theta is bit-identical."""
    return {name: theta[name] + applied[name] for name in theta}


def encode_state(momentum, record):
    """A replica's per-round state as one .npz body: the outer momentum (fp32, under STATE_PREFIXES[1]) and diloco.json's bytes."""
    buffer = io.BytesIO()
    np.savez(buffer, **{STATE_PREFIXES[1] + name: np.asarray(value, dtype=np.float32) for name, value in momentum.items()},
             **{STATE_FILE: np.frombuffer(record, dtype=np.uint8)})
    return buffer.getvalue()


def decode_state(data):
    """(momentum, diloco.json bytes) of an encode_state body. ValueError when it is not one."""
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            momentum = {key[len(STATE_PREFIXES[1]):]: archive[key].astype(np.float32) for key in archive.files if key.startswith(STATE_PREFIXES[1])}
            return momentum, archive[STATE_FILE].tobytes()
    except Exception as exc:  # noqa: BLE001 - zip and numpy raise many types for a bad body
        raise ValueError(f"The body is not a replica state archive ({exc}).") from None


# ---- shards -----------------------------------------------------------------------------------------------------------

def shard_blob(windows, low, high):
    """Windows low..high-1 of a WindowStore as an .npz body: tokens, flags (one per target position) and rebased offsets."""
    offsets = np.asarray(windows.offsets[low:high + 1], dtype=np.int64)
    tokens = np.asarray(windows.tokens[offsets[0]:offsets[-1]])
    flags = np.asarray(windows.flags[offsets[0] - low:offsets[-1] - high], dtype=np.uint8)
    buffer = io.BytesIO()
    np.savez(buffer, tokens=tokens, flags=flags, offsets=offsets - offsets[0])
    return buffer.getvalue()


def install_shard(root, shard, blob, vocab_size):
    """Write a downloaded shard under root as a studio_shards folder (train split only) and return its windows, memory-mapped."""
    import studio_model
    import studio_shards
    from studio_slots import BOS
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        tokens, flags, offsets = archive["tokens"], archive["flags"], archive["offsets"]
    key, directory = f"shard{shard}", Path(root) / f"shard{shard}"
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    dtype, count = studio_shards.dtype_for(vocab_size), len(offsets) - 1
    all_targets = bool(flags.all())
    (directory / "train.tokens.bin").write_bytes(tokens.astype(dtype).tobytes())
    (directory / "train.offsets.bin").write_bytes(offsets.astype(np.int64).tobytes())
    if not all_targets:
        (directory / "train.flags.bin").write_bytes(flags.tobytes())
    (directory / "val.tokens.bin").write_bytes(b"")
    (directory / "val.offsets.bin").write_bytes(np.zeros(1, dtype=np.int64).tobytes())
    (directory / "val.flags.bin").write_bytes(b"")
    train = {"windows": count, "tokens": len(tokens), "targets": int(flags.sum()), "all_targets": all_targets,
             "from_start": int((tokens[offsets[:-1]] == BOS).sum()) if count else 0}
    val = {"windows": 0, "tokens": 0, "targets": 0, "from_start": 0, "all_targets": False}
    (directory / "index.json").write_text(json.dumps({"format": studio_shards.FORMAT, "vocab_size": vocab_size, "train": train, "val": val}),
                                          encoding="utf-8")
    found = studio_shards.load(root, key, vocab_size)
    if found is None:
        raise OSError(f"Shard {shard} could not be read back after writing.")
    return studio_model.MappedWindows(*found["train"])


def cache_name(digest, vocab_size):
    """The name of a shard in the cache that outlives runs: its content hash and the vocabulary its tokens are stored for."""
    return f"{digest[:32]}-{vocab_size}"


def cached_shard(root, digest, vocab_size):
    """The windows of the shard with content hash `digest` held under `root` (install_shard's folder), or None."""
    import studio_model
    import studio_shards
    found = studio_shards.load(root, f"shard{cache_name(digest, vocab_size)}", vocab_size)
    return None if found is None else studio_model.MappedWindows(*found["train"])


class Windows:
    """The windows a member trains on: several WindowStores read as one sequence, swapped when a round deals new shards."""

    def __init__(self, parts=()):
        self.set(parts)

    def set(self, parts):
        self.parts = list(parts)
        self.ends = list(np.cumsum([len(part) for part in self.parts], dtype=np.int64)) if self.parts else []

    def __len__(self):
        return int(self.ends[-1]) if self.ends else 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Window index out of range.")
        part = bisect.bisect_right(self.ends, index)
        return self.parts[part][index - (int(self.ends[part - 1]) if part else 0)]

    def describe(self):
        windows = len(self)
        starts = sum(part.describe()["from_start"] * len(part) for part in self.parts)
        return {"windows": windows, "from_start": starts / windows if windows else 0.}

    @property
    def nbytes(self):
        return sum(part.nbytes for part in self.parts)


# ---- the coordinator --------------------------------------------------------------------------------------------------

@dataclass
class Round:
    number: int
    generation: int
    members: list
    deal: dict
    opened: float
    deadline: float
    window: float
    steps: dict = field(default_factory=dict)  # node -> the inner steps dealt to it (deal_steps)
    train_seconds: float = 0.  # time-boxed rounds: the seconds every member trains for (0: a fixed number of steps)
    downloaded: dict = field(default_factory=dict)  # node -> bytes of theta or update it took
    contributions: dict = field(default_factory=dict)  # node -> (tokens, steps, seconds, loss, delta)
    uploaded: dict = field(default_factory=dict)  # node -> bytes of its delta body on the wire
    rates: dict = field(default_factory=dict)  # node -> {"down", "up"}: bytes per second its weights came down and its change went up
    open: bool = True
    marks: dict = field(default_factory=dict)  # node -> {MARKS name: coordinator clock time}
    closed: float | None = None


def step_size(delta, momentum, lr, mu):
    """The L2 norm of the outer step just taken, lr * (delta + mu * v) over every tensor."""
    return float(np.sqrt(sum(float(((lr * (gradient + mu * momentum[name])) ** 2).sum()) for name, gradient in delta.items())))


def tokenized_root(directory):
    """Where the tokenized text of a run is cached so the next run on the same data and tokenizer finds it: data/shards beside the runs
    folder (the one Resources > Storage ages out). A run folder anywhere else (a test, a script) keeps it inside itself."""
    directory = Path(directory)
    return directory.parent.parent / "data" / "shards" if directory.parent.name == "runs" else directory / "windows"


PREPARE_EVERY = 50  # documents tokenized between two progress reports of the build


def _counted(examples, tally, say):
    """`examples`, one by one, while `tally` ([done, total], shared by the splits) counts them and `say("tokenizing", done, total)` is called every PREPARE_EVERY."""
    for example in examples:
        yield example
        tally[0] += 1
        if tally[0] % PREPARE_EVERY == 0:
            say("tokenizing", tally[0], tally[1])


class Coordinator:
    """The global state of one run. Thread-safe; every method that can change it first moves the clock's consequences along
    (_advance: close a round that is complete or past its deadline, open the next). `evaluate(theta)` returns the validation loss."""

    def __init__(self, run, directory, cfg, tokenizer, train, val, settings, theta, evaluate, digest="", clock=time.monotonic, state=None,
                 on_step_down=None):
        self.run, self.directory, self.cfg, self.tokenizer = run, Path(directory), cfg, tokenizer
        self.train, self.val, self.settings, self.evaluate, self.digest, self.clock = train, val, settings, evaluate, digest, clock
        self.theta = {name: np.array(value, dtype=np.float32) for name, value in theta.items()}
        self.momentum = {name: np.zeros_like(value) for name, value in self.theta.items()}
        self.next_round, self.generation, self.history, self.step_time = 0, 0, [], None
        self.last_steps = {}  # node -> the steps it was dealt for the round opened last
        self.step_times = {}  # node -> its smoothed seconds per step, from the rounds it contributed to (not kept over a restart)
        self.members, self.pending, self.leaving, self.last_members, self.sequence = {}, {}, set(), [], 0
        self.current, self.finished, self.touched, self.deal = None, False, time.monotonic(), {}
        self.beats = {}  # node -> the clock time of its last call: the heartbeat studio_cluster_watch reads
        self.stopped, self.banned, self.gone = False, set(), {}  # stopped by the owner; nodes the owner removed; node -> (state, sentence) of those out
        self.split, self.split_hash, self.state_file, self.record = {"strategy": "lineage-v1"}, "", None, {}
        self.shard_count = max(1, min(settings.shards, len(train)))
        self.bounds = np.linspace(0, len(train), self.shard_count + 1).astype(int)
        self.spans, self.later = [(0, self.shard_count)], []  # per data source: its first shard id and shard count; the (windows, bounds) of the sources after the first
        self.rules = drules.State()
        self.stages = None  # studio_stages.Policy: where next_dataset goes (the pool of sets, ranked by held-out loss), replay, the journal of switches
        self.paused = False  # the owner paused the run: no round opens until resume (a round in flight finishes)
        self.polishing = False  # the teacher is polishing theta, on a thread off the lock: like paused, no round opens meanwhile
        self.polisher, self.teacher_settings, self.polish_on = default_polish, {}, True  # polish_on: off on a coordinator made from a replica
        self.judging = False  # with polishing: the held phase is the head-to-head of target_win_rate, not a polish (the phase says so)
        self.versus, self.versus_on = default_versus, True  # the head-to-head of target_win_rate; off on a coordinator made from a replica (no prompts, no start.npz)
        self._polish_stop = threading.Event()  # set by stop, step-down and release: the polish in flight is discarded
        self.lessons, self.lesson_windows = [], {}  # [{shard, round, rows, status}] in shard-id order; shard id -> its windows
        self.blobs, self.hashes, self.blob_lock = {}, {}, threading.Lock()  # shard id -> the file its .npz body was written to (once); see shard_path
        self.parts = None  # a coordinator made from a replica serves the shards it holds one by one (from_replica)
        self._params, self._state = (None, b""), (None, b"")
        self.joined_round = {}  # node -> the (0-based) round it first trains in, for a node that joined after the run began
        self.late, self.late_round, self.closed = {}, None, None  # a change that came in after its round closed, waiting for the next outer step; (number, generation, members, contributors) of the last closed round
        self._update, self.update_residual = None, {}  # (round, body): the outer update that turned theta of the round before into theta of that round; what its quantization lost
        self.last_train = 0.  # the training seconds of the round opened last (time-boxed runs)
        self.started, self.theta_nbytes = clock(), sum(value.nbytes for value in self.theta.values())  # the zero of the event log; θ's size
        self.reports, self.events, self.timelines = {}, deque(maxlen=EVENTS_KEPT), deque(maxlen=TIMELINES_KEPT)  # the live view's state
        self.event_seq, self.stage, self._last = 0, "", None  # stage: "outer_step" or "validating" while a round closes (read without the lock)
        self.term, self.superseded, self.on_step_down = 1, 0, on_step_down  # superseded: the higher term this coordinator was told of, else 0
        self.prefer = ()  # nodes that are replicas before the others: the node chosen to coordinate (prepare)
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.plan = [dict(entry) for entry in (state or {}).get("plan") or [{"round": 0, "q": 0., "h": settings.inner_steps, "rounds": settings.rounds}]]  # see plan_h
        if state:
            self.term = state["term"]
            self.momentum = state["momentum"]
            self.next_round, self.generation = state["round"], state["generation"]
            self.history, self.step_time, self.state_file = state["history"], state.get("step_time"), state["state_file"]
            self.rules = drules.State.from_json(state.get("rules"))
            self.paused = bool(state.get("paused"))
            self.lessons = [dict(entry) for entry in state.get("lessons") or []]
            self.finished = self.next_round >= settings.rounds or bool(state.get("finished"))

    # -- building and resuming
    @classmethod
    def build(cls, run, directory, config, examples, settings, clock=time.monotonic, shard_dir=None, later=(), start_from=None, conversations=None, progress=None, prepared=None):
        """Tokenize `examples` once the way studio_model.train does, cut the windows, start theta from the config's seeded
        initialization, and continue from diloco.json if the folder has one. `later` are the example lists of the data sources
        the next_dataset pivot moves on to, in order: cut with the first source's tokenizer, never part of the validation. `start_from`, the
        path of a checkpoint (best.npz or last.npz of an earlier run), is where theta starts when the folder holds no run yet (a restart).
        `conversations`, when given (a list, possibly empty), are the teacher's: the polish and the head-to-head take their prompts from them, split
        by the run's seed into polish prompts and held-out ones, and they are never trained on, tokenized or part of the digest or the split. When
        None the prompts come from the conversations among `examples` (the training split, the validation split). `progress(phase, done, total)`, when
        given, hears the build as it goes: "tokenizer" while the tokenizer is fitted, then "tokenizing" with the documents cut so far of all of them.
        When `prepared` is given (a dict from services.get_prepared_shards), token shards and tokenizer are read directly from disk without reading examples."""
        import studio_model as sm
        from studio_core import fingerprint
        from studio_planning import merge_duplicate_lineage
        from studio_spec import module_of
        cfg = sm.validate_config({**config, "steps": settings.rounds * settings.inner_steps})
        if cfg["lora"]:
            raise ValueError("Distributed pretraining trains every weight; LoRA is not supported.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        say = progress or (lambda *_: None)

        if prepared is not None:
            split_hash = prepared["split_hash"]
            digest = prepared.get("digest") or split_hash
            split = prepared.get("split") or {"strategy": "lineage-v1"}
            folder = Path(prepared["folder"])
            if not folder.is_absolute():
                folder = (shard_dir or tokenized_root(directory)) / folder
            saved = studio_shards.load_tokenizer(folder)
            if saved is None:
                raise ValueError(f"Prepared tokenizer at {folder} is missing or unreadable.")
            tokenizer = sm.Tokenizer.restore(saved)
            found = studio_shards.load(folder, prepared["shard_key"], tokenizer.vocab_size)
            if found is None:
                raise ValueError(f"Prepared shards in {folder}/{prepared['shard_key']} are missing or unreadable.")
            train = sm.MappedWindows(*found["train"])
            val = sm.MappedWindows(*found["val"])
            say("cached", len(train) + len(val), len(train) + len(val))
        else:
            unique = {}
            for example in examples:
                key = fingerprint(example)
                if key in unique:
                    merge_duplicate_lineage(unique[key], example)
                else:
                    unique[key] = dict(example)
            examples = list(unique.values())
            if not examples:
                raise ValueError("The dataset is empty.")
            digest = hashlib.sha256("".join(fingerprint(e) for e in examples).encode()).hexdigest()
            training, validation, split = sm.partition_examples(examples, cfg["validation"], random.Random(cfg["seed"]), True)
            split_hash = hashlib.sha256(("".join(fingerprint(e) for e in training) + "|" + "".join(fingerprint(e) for e in validation)).encode()).hexdigest()
            spec = sm.effective_spec(cfg)["tokenizer"]
            folder = studio_shards.tokenizer_folder(shard_dir or tokenized_root(directory), split_hash, spec)  # shared by runs: this data under this tokenizer
            tokenizer = None
            saved = studio_shards.load_tokenizer(folder)
            if saved is not None:
                try:
                    tokenizer = sm.Tokenizer.restore(saved)
                except (ValueError, KeyError, TypeError):
                    tokenizer = None  # unreadable: fitted again below
            if tokenizer is None:
                say("tokenizer", 0, 0)
                module, module_cfg = module_of("tokenizer", spec)
                tokenizer = module.fit(module_cfg, training)
                studio_shards.save_tokenizer(folder, tokenizer.serialize())
            tally = [0, len(training) + len(validation)]  # documents cut so far, of all
            cached = studio_shards.load(folder, sm.shard_key(tokenizer, cfg, split_hash), tokenizer.vocab_size) is not None
            say("cached" if cached else "tokenizing", tally[1] if cached else 0, tally[1])  # cached: the windows are on disk from an earlier run
            train, val, _ = sm.shard_windows(_counted(training, tally, say), _counted(validation, tally, say), tokenizer, cfg, split_hash, folder)
            if not cached:
                say("tokenizing", tally[1], tally[1])
        if not train:
            raise ValueError("No trainable target tokens were found.")
        if settings.target_win_rate is not None and not val:
            raise ValueError("target_win_rate needs a validation split: without one no round has a validation loss, so no round is ever the best "
                             "to compare. Hold some of the text out (validation above 0), or leave target_win_rate unset.")
        model = sm.ScratchTransformer(cfg, tokenizer.vocab_size)
        theta = {name: value.astype(np.float32) for name, value in model.state_arrays().items()}
        state = cls._read_state(directory, cfg, settings, digest)
        polish_prompts = eval_prompts = None
        files = [name for name, on in (("prompts.jsonl", settings.polish_every), ("eval_prompts.jsonl", settings.target_win_rate is not None)) if on]
        # A resume never splits again: it keeps the prompts it wrote (chats deleted or re-marked since must not stop it or change what it is
        # compared on). A prompt file that went missing turns that option off for the rest of the run.
        lost = [name for name in files if conversations is not None and state and not (directory / name).is_file()]
        if conversations is not None and files and not state:
            polish_prompts, eval_prompts = split_prompts(conversations, settings, cfg["seed"])
        if state:
            theta = state["theta"]
            if state["extended"]:  # the run was extended: it goes on with the length and H it had grown to
                settings = replace(settings, **state["extended"])
        elif start_from:
            theta = cls._weights_of(start_from, theta)
        coordinator = cls(run, directory, cfg, tokenizer, train, val, settings, theta, _evaluator(model, val, cfg["batch_size"]), digest, clock, state)
        coordinator.split_hash, coordinator.split = split_hash, split
        coordinator.model_spec = model.spec
        for number, more in enumerate(later, 1):
            coordinator.add_source(cls._windows_of(sm, more, tokenizer, cfg, folder))
        coordinator._load_lessons()
        if coordinator.rules.source >= len(coordinator.spans):
            raise ValueError(f"The run had moved on to data source {coordinator.rules.source + 1}, but only {len(coordinator.spans)} "
                             f"{'is' if len(coordinator.spans) == 1 else 'are'} supplied (later_examples); training on the first would be a different run.")
        if not state:
            grown = None  # an extension made before the first round closed; only for the run it was made on (same rounds and H to start)
            if (directory / "extended.json").is_file():
                grown = json.loads((directory / "extended.json").read_text(encoding="utf-8"))
                if (grown["plan"][0]["rounds"], grown["plan"][0]["h"]) != (settings.rounds, settings.inner_steps):
                    grown = None
            if grown:
                coordinator.settings = replace(settings, rounds=grown["rounds"], inner_steps=grown["inner_steps"])
                coordinator.plan = [dict(entry) for entry in grown["plan"]]
            coordinator._start_files(model)
            if grown:
                from studio_core import atomic_json
                coordinator.record.update({"steps": plan_total(coordinator.plan)})
                atomic_json(directory / "run.json", coordinator.record)
            held = directory / "paused.json"  # a pause before the first round closed
            if held.is_file():
                saved = json.loads(held.read_text(encoding="utf-8"))
                coordinator.paused, coordinator.rules = bool(saved.get("paused")), drules.State.from_json(saved.get("rules"))
        if "prompts.jsonl" in lost:
            coordinator.polish_on = False
            coordinator._event("polish_off", "", coordinator.next_round,
                               "The teacher polish ends here: this run's polish prompts are gone from its folder, and splitting the chats again could "
                               "overlap the prompts it is compared on, so the run goes on without it.")
        elif settings.polish_every:  # after _start_files, which clears a new run's prompts
            coordinator._write_prompts(training, polish_prompts)
        if "eval_prompts.jsonl" in lost:
            coordinator.versus_on = False
            coordinator._event("win_rate_off", "", coordinator.next_round,
                               "The win-rate stop rule ends here: this run's held-out prompts are gone from its folder, and splitting the chats again "
                               "could overlap the polish prompts, so the run goes on without it.")
        elif settings.target_win_rate is not None:
            coordinator._write_eval_prompts(validation, training, eval_prompts)
            if start_from and not state:
                shutil.copyfile(start_from, directory / "start.npz")  # a restart compares with the weights it started from
        if settings.polish_every or settings.target_win_rate is not None:
            coordinator.teacher_settings = teacher_settings(settings)  # prepare replaces it with the workspace's own
        return coordinator

    @staticmethod
    def _weights_of(path, fresh):
        """The weights stored in the checkpoint `path`, for theta to start from. ValueError, saying why, when it is missing or its
        weights are not those of the model `fresh` (the seeded initialization) describes."""
        try:
            with np.load(path, allow_pickle=False) as stored:
                found = {key[len("weight::"):]: stored[key].astype(np.float32) for key in stored.files if key.startswith("weight::")}
        except (OSError, ValueError) as exc:
            raise ValueError(f"The checkpoint to start from could not be read ({type(exc).__name__}: {exc}).") from None
        if set(found) != set(fresh) or any(found[name].shape != fresh[name].shape for name in fresh):
            raise ValueError("The checkpoint to start from is of a different model than this run's design.")
        return found

    @classmethod
    def from_replica(cls, run, directory, replica_dir, shard_dir, term, clock=time.monotonic, on_step_down=None):
        """The coordinator of `term` made from the replica this node keeps (Member._replicate: theta, the outer momentum and diloco.json
        of the latest round, info.json, validation.npz) and the shards its member cached under `shard_dir`: it continues from that
        round, with the history, settings and step time of the record. The round that was open when the old coordinator went is lost,
        and so are the deltas members had sent to it. Its own folder `directory` gets the run's files again, starting from this
        round. ValueError, saying what is missing, when the replica is not whole (no theta yet, validation or a shard not fetched)."""
        import studio_model as sm
        import studio_shards
        from studio_core import atomic_json
        from studio_modules.tokenizer.byte import Tokenizer
        replica, directory = Path(replica_dir), Path(directory)
        try:
            info = json.loads((replica / "info.json").read_text(encoding="utf-8"))
            record = json.loads((replica / STATE_FILE).read_text(encoding="utf-8"))
            theta = decode_arrays((replica / "theta.npz").read_bytes())
            momentum = decode_arrays((replica / "momentum.npz").read_bytes())
            validation = (replica / "validation.npz").read_bytes()
            settings = Settings(**record["settings"])
            vocabulary, count = int(info["vocab_size"]), int(info["shard_count"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"This node holds no whole replica of run {run} ({type(exc).__name__}: {exc}).") from None
        if record.get("run") != run:
            raise ValueError(f"The replica on this node is of run {record.get('run')}, not {run}.")
        parts = []
        for shard in range(count):
            found = studio_shards.load(shard_dir, f"shard{shard}", vocabulary)
            if found is None:
                raise ValueError(f"Shard {shard} of run {run} is not in this node's cache.")
            parts.append(sm.MappedWindows(*found["train"]))
        directory.mkdir(parents=True, exist_ok=True)
        val = install_shard(directory / "windows", "val", validation, vocabulary)
        cfg = info["config"]
        model = sm.ScratchTransformer(cfg, vocabulary, probe=True)
        state = {"term": term, "momentum": momentum, "round": int(record["round"]), "generation": int(record["generation"]),
                 "history": record["history"], "step_time": record.get("step_time"), "state_file": None, "rules": record.get("rules"), "paused": record.get("paused"),
                 "lessons": record.get("lessons"), "plan": record.get("plan")}
        coordinator = cls(run, directory, cfg, Tokenizer.restore(info["tokenizer"]), Windows(parts), val, settings, theta,
                          _evaluator(model, val, cfg["batch_size"]), record["digest"], clock, state, on_step_down)
        coordinator.shard_count, coordinator.parts = count, parts
        sizes = [int(n) for n in record.get("layout") or [count]]
        coordinator.spans = [(sum(sizes[:k]), n) for k, n in enumerate(sizes)]
        coordinator.split, coordinator.split_hash = {"strategy": record.get("split_strategy", "lineage-v1")}, record.get("split_hash", "")
        coordinator.model_spec = model.spec
        coordinator.deal = record.get("shard_deal") or {}
        coordinator._start_files(model)
        coordinator.polish_on = False  # the replica holds no training examples to make prompts from, nor the teacher's rows
        coordinator.versus_on = False  # ... nor the prompts of the head-to-head or the weights the run started from
        for entry in coordinator.lessons:
            if entry["status"] == "dropped":
                continue  # the owner's choice carries over: not dealt, and the list says who left it out
            entry["status"] = "left_out"  # the replica holds no lesson shards, so the lessons the old coordinator dealt are not dealt any more
        if coordinator.history:  # only the latest round's weights came over: they are the best this coordinator can keep
            last, rules = coordinator.history[-1], coordinator.rules
            loss = last.get("val_loss")
            if rules.best_round != last["round"]:
                # The replica holds one round's weights, so that round is the only best this coordinator can claim (run.json and the
                # saved Version name it): relabel it, and restart the streaks the old best's rounds counted against.
                finite = loss is not None and loss == loss and abs(loss) != float("inf")
                rules.best_round, rules.best_loss = (last["round"], float(loss)) if finite else (None, None)
                rules.rounds_since_best, rules.worse_streak, rules.points = 0, 0, []
            if rules.best_round is not None:
                coordinator._write_best()
        last = coordinator.history[-1] if coordinator.history else {}
        with (directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            for row in coordinator.history:
                stream.write(json.dumps({k: row.get(k) for k in ("step", "loss", "val_loss", "tokens_per_second", "lr", "grad_norm")}
                                        | {"round": row["round"], "timeline": None}) + "\n")
        step = coordinator._persist(record.get("members") or [], coordinator.deal)
        coordinator.record.update({"step": step, "val_loss": last.get("val_loss"), "status": "completed" if coordinator.finished else "training"})
        atomic_json(directory / "run.json", coordinator.record)
        coordinator._event("took_over", "", coordinator.next_round,
                           f"Term {term} began here, continuing from round {coordinator.next_round} of term {record.get('term', 1)}.")
        if settings.target_win_rate is not None:
            coordinator._event("win_rate_off", "", coordinator.next_round,
                               "The win-rate stop rule ends here: this node holds neither the prompts of the head-to-head nor the weights the run started from, "
                               "so the run goes on without it.")
        if settings.polish_every:
            coordinator._event("polish_off", "", coordinator.next_round,
                               "The teacher polish ends here: this node holds neither the training examples nor the lessons the old coordinator wrote, "
                               "so the run goes on without them.")
        return coordinator

    @staticmethod
    def _windows_of(sm, examples, tokenizer, cfg, directory):
        """The training windows of one later data source."""
        from studio_core import fingerprint
        examples = list(examples)
        if sm.characters_in(examples) >= sm.STREAM_CHARS:
            key = hashlib.sha256("".join(fingerprint(e) for e in examples).encode()).hexdigest()
            windows = sm.shard_windows(examples, [], tokenizer, cfg, key, directory)[0]
        else:
            windows = sm.samples_from(examples, tokenizer, cfg)
        if not len(windows):
            raise ValueError("A later data source has no trainable target tokens.")
        return windows

    def add_source(self, windows):
        """Append a data source the next_dataset pivot can move on to: its windows are cut into shards that get the ids after the others'."""
        count = max(1, min(self.settings.shards, len(windows)))
        self.later.append((windows, np.linspace(0, len(windows), count + 1).astype(int)))
        self.spans.append((self.shard_count, count))
        self.shard_count += count

    # -- prompts and lessons (teacher polish)
    def _write_prompts(self, training, prompts=None):
        """prompts.jsonl: the prompts the teacher is asked to answer, once per run: `prompts` (the polish share of the teacher's own
        conversations), else the conversations of the training split (never a validation one)."""
        path = self.directory / "prompts.jsonl"
        if path.is_file():
            return
        from studio_rlhf import prompts_from
        prompts = prompts_from(training) if prompts is None else prompts
        if not prompts:
            raise ValueError("polish_every is set, but the training text has no conversations for the teacher to answer: "
                             "use a chat dataset, or turn polish_every off.")
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for prompt in prompts:
                stream.write(json.dumps(prompt, ensure_ascii=False) + "\n")
        temporary.replace(path)

    def _write_eval_prompts(self, validation, training, held_out=None):
        """eval_prompts.jsonl: EVAL_PROMPTS conversations for the head-to-head against the starting weights, from the validation split
        (topped up from the END of the training split when it has fewer: the teacher's polish windows start at its front), once per run.
        Each line says where its prompt came from. The prompts the polish turned into lessons are left out when the comparison runs."""
        path = self.directory / "eval_prompts.jsonl"
        if path.is_file():
            return
        from studio_rlhf import prompts_from
        if held_out is not None:  # the teacher's own conversations: none of these is a polish prompt or a training example
            prompts = [{**p, "from": "held_out"} for p in held_out]
        else:
            prompts = [{**p, "from": "validation"} for p in prompts_from(validation)[:EVAL_PROMPTS]]
            spare = EVAL_PROMPTS - len(prompts)
            prompts += [{**p, "from": "training"} for p in (prompts_from(training)[::-1][:spare] if spare > 0 else [])]
        if not prompts:
            raise ValueError("target_win_rate is set, but the text has no conversations for the teacher to compare the weights on: "
                             "use a chat dataset, or leave target_win_rate unset.")
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for prompt in prompts:
                stream.write(json.dumps(prompt, ensure_ascii=False) + "\n")
        temporary.replace(path)

    def prompt_window(self, polish_number):
        """The prompts of the `polish_number`-th polish (counting from 1): PROMPTS_PER_POLISH of them, a new window each time, wrapping
        round the file."""
        path = self.directory / "prompts.jsonl"
        if not path.is_file():
            return []
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return []
        start = (polish_number - 1) * PROMPTS_PER_POLISH % len(lines)
        return [json.loads(line) for line in (lines[start:] + lines[:start])[:PROMPTS_PER_POLISH]]

    def _lesson_windows(self, rows):
        """The training windows of lesson rows: each teacher rewrite is the conversation plus the better answer."""
        import studio_model as sm
        examples = [{"messages": [*row["messages"], {"role": "assistant", "content": row["chosen"]}]} for row in rows]
        return sm.samples_from(examples, self.tokenizer, self.cfg)

    def _load_lessons(self):
        """Rebuild the windows of the lessons diloco.json lists from lessons.jsonl (a restart). A lesson whose rows are gone is left out."""
        if not self.lessons:
            return
        path = self.directory / "lessons.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []
        for entry in self.lessons:
            if entry["status"] not in ("live", "dropped"):
                continue  # a dropped lesson keeps its windows, so Keep can deal it again
            mine = [row for row in rows if row.get("shard") == entry["shard"]]
            windows = self._lesson_windows(mine) if mine else []
            if len(windows):
                self.lesson_windows[entry["shard"]] = windows
            else:
                entry["status"] = "left_out"

    def live_lessons(self, since=None):
        """The lessons being dealt; with `since`, only those written after round `since` closed or later (they were first trained on in
        a round after that round's weights)."""
        return [entry for entry in self.lessons if entry["status"] == "live" and (since is None or entry["round"] >= since)]

    def lesson_list(self):
        """Every lesson, oldest first, with what the teacher gave (the question, the model's reply, the better answer) as lessons.jsonl
        holds it: [{id, round, rows, status, items, more}]. A lesson is one polish's rewrites (one shard); `items` shows the first
        LESSON_ITEMS_SHOWN of them and `more` counts the rest. `keepable` says Keep can deal it again: it is not dealt now and this node
        holds its rows (a coordinator made from a replica holds none). Only a snapshot of the statuses is taken under the lock; the
        file is read and parsed outside it."""
        with self.lock:
            entries = [dict(entry) for entry in self.lessons]
        shown = {}
        path = self.directory / "lessons.jsonl"
        try:
            lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        except OSError:
            lines = []
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            shown.setdefault(row.get("shard"), []).append(row)
        out = []
        for entry in entries:
            rows = shown.get(entry["shard"], [])
            items = [{"prompt": _prompt_of(row["messages"]), "reply": str(row.get("rejected") or ""), "answer": str(row["chosen"])}
                     for row in rows[:LESSON_ITEMS_SHOWN]]
            out.append({"id": entry["shard"], "round": entry["round"], "rows": entry["rows"], "status": entry["status"],
                        "keepable": entry["status"] != "live" and bool(rows), "items": items, "more": max(0, len(rows) - len(items))})
        return out

    def drop_lesson(self, shard):
        """Leave a lesson out of every round that opens from now on (the round in flight keeps its deal). It stays in lessons.jsonl and
        in memory, so keep_lesson can put it back, and the choice goes to diloco.json: a restart and a successor coordinator keep it.
        KeyError for an id that is no lesson; Stale once the run is over; a lesson that is not being dealt changes nothing. The win-rate
        comparison still never judges on a dropped lesson's prompt: the weights may have trained on it in the rounds before, and
        judging on a prompt the model has seen would flatter it."""
        with self.cond:
            self._touch()
            entry = self._lesson(shard)
            if entry["status"] != "live":
                return
            entry["status"] = "dropped"
            self._lesson_changed("lesson_dropped", f"Lesson {shard} (round {entry['round']}, {entry['rows']} answers) was dropped by you: "
                                                   f"the next rounds leave it out.")

    def keep_lesson(self, shard):
        """Undo drop_lesson, or put back a lesson a step back left out: the next rounds deal it again. Stale once the run is over, or when
        this coordinator no longer holds the lesson's rows (one made from a replica); a lesson already dealt changes nothing."""
        with self.cond:
            self._touch()
            entry = self._lesson(shard)
            if entry["status"] == "live":
                return
            if shard not in self.lesson_windows:
                path = self.directory / "lessons.jsonl"
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []
                mine = [row for row in rows if row.get("shard") == shard]
                windows = self._lesson_windows(mine) if mine else []
                if not len(windows):
                    raise Stale(f"This node no longer holds the answers of lesson {shard}, so it cannot be put back.")
                self.lesson_windows[shard] = windows
            entry["status"] = "live"
            self._lesson_changed("lesson_kept", f"Lesson {shard} (round {entry['round']}, {entry['rows']} answers) was kept by you: "
                                                f"the next rounds deal it again.")

    def _lesson(self, shard):
        if self.finished:
            raise Stale("The run is over; its lessons can no longer change.")
        entry = next((entry for entry in self.lessons if entry["shard"] == shard), None)
        if entry is None:
            raise KeyError(shard)
        return entry

    def _lesson_changed(self, kind, text):
        """Journal a lesson's new status and write the lessons to diloco.json (lock held)."""
        from studio_core import atomic_json
        self._event(kind, "", self.next_round, text)
        self._state = (None, b"")  # a replica that fetched the state before this change must fetch it again
        path = self.directory / "diloco.json"
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            atomic_json(path, {**saved, "lessons": [dict(item) for item in self.lessons]})
        self._remember()
        self.cond.notify_all()

    @staticmethod
    def _read_state(directory, cfg, settings, digest):
        path = directory / "diloco.json"
        if not path.is_file():
            return None
        saved = json.loads(path.read_text(encoding="utf-8"))
        started_with = saved["plan"][0]["h"] if saved.get("plan") else saved["settings"]["inner_steps"]  # an extended run's H has changed since
        if saved["digest"] != digest or started_with != settings.inner_steps or saved["seed"] != cfg["seed"]:
            raise ValueError("This folder holds a different DiLoCo run (other data, seed or inner steps).")
        with np.load(directory / saved["state_file"], allow_pickle=False) as archive:
            theta = {key[len(STATE_PREFIXES[0]):]: archive[key].copy() for key in archive.files if key.startswith(STATE_PREFIXES[0])}
            momentum = {key[len(STATE_PREFIXES[1]):]: archive[key].copy() for key in archive.files if key.startswith(STATE_PREFIXES[1])}
        return {"theta": theta, "momentum": momentum, "state_file": saved["state_file"], "round": saved["round"], "generation": saved["generation"],
                "history": saved["history"], "step_time": saved.get("step_time"), "term": saved.get("term", 1),
                "rules": saved.get("rules"), "finished": saved.get("finished", False), "paused": saved.get("paused", False),
                "lessons": saved.get("lessons", []), "plan": saved.get("plan"),
                "extended": {k: saved["settings"][k] for k in ("rounds", "inner_steps")} if saved.get("plan") else None}

    def _start_files(self, model):
        from studio_core import atomic_json, now
        atomic_json(self.directory / "config.json", self.cfg)
        atomic_json(self.directory / "spec.json", model.spec)
        atomic_json(self.directory / "tokenizer.json", self.tokenizer.serialize())
        (self.directory / "metrics.jsonl").write_text("", encoding="utf-8")
        (self.directory / "prompts.jsonl").unlink(missing_ok=True)  # ... nor prompts: they are made from this run's own training split
        (self.directory / "eval_prompts.jsonl").unlink(missing_ok=True)  # ... nor the head-to-head's prompts or the weights it compares with
        (self.directory / "start.npz").unlink(missing_ok=True)
        (self.directory / "lessons.jsonl").unlink(missing_ok=True)  # a new run has no lessons: the rows of an earlier one in this folder are not its own
        self.record = {"name": self.cfg["name"], "createdAt": now(), "status": "training", "step": 0, "steps": self.cfg["steps"],
                       "parameters": model.parameter_count, "vocabulary": self.tokenizer.vocab_size, "backend": "diloco",
                       "device": "distributed", "rounds": self.settings.rounds, "inner_steps": self.settings.inner_steps}
        atomic_json(self.directory / "run.json", self.record)

    # -- membership
    def info(self):
        """What a member needs to start: the config and tokenizer, and the run's shape."""
        return {"term": self.term, "generation": self.generation, "round": self.next_round, "config": self.cfg, "tokenizer": self.tokenizer.serialize(),
                "inner_steps": self.settings.inner_steps, "rounds": self.settings.rounds, "plan": [dict(entry) for entry in self.plan],
                "total_steps": plan_total(self.plan), "delta_dtype": self.settings.delta_dtype, "delta_sparsity": self.settings.delta_sparsity,
                "shard_count": self.shard_count, "seed": self.cfg["seed"], "vocab_size": self.tokenizer.vocab_size}

    # -- terms
    def check_term(self, term, learning=False):
        """Admit a call that carries `term`. Lower than this coordinator's is StaleTerm (409), and changes nothing. Higher means a
        later coordinator exists: this one steps down (see step_down) and refuses the call. Once stepped down every call is refused.
        A member that does not know the term yet sends 0; only the call that joins (`learning`) is let through with it."""
        if term > self.term:
            self.step_down(term)
        with self.lock:
            if self.superseded:
                raise StaleTerm(f"{TERM_STALE} {term}: the run is at term {self.superseded}, so this node no longer coordinates it.", self.superseded)
            if term < self.term and not (learning and term == 0):
                raise StaleTerm(f"{TERM_STALE} {term}: the run is at term {self.term}.", self.term)

    def step_down(self, term, force=False):
        """Told of a higher term (by a call that carries it, or by whoever watches the run's claims): stop closing rounds. The run's
        state stays as it is on disk for the new coordinator's replicas; this node joins the new coordinator as a member (#361).
        Idempotent; a term that is not higher than the coordinator's own changes nothing, unless `force`: a claim of the same term
        that outranks this coordinator's (studio_failover.beats) settles the tie, and the coordinator that lost defers to it."""
        with self.cond:
            if term <= self.superseded or term < self.term or (term == self.term and not force):
                return
            self.superseded = term
            self._polish_stop.set()  # a polish in flight is for a coordinator that no longer holds the run
            self._event("stepped_down", "", self.next_round, f"Term {term} exists (this node held term {self.term}): no longer coordinating the run.")
            self.cond.notify_all()
        if self.on_step_down:
            self.on_step_down(term)

    def defer(self, term):
        """Lose to a claim of `term` (this coordinator's own or higher) that outranks this node's: step down, and discard the rounds
        this coordinator closed after the claim's round, so the run has one theta. The files in its folder stay; nothing here uses them."""
        self.step_down(term, force=True)
        self.release()

    def _lease(self, term, epoch):
        """The fence a member's lease is compared by: the coordinator's term first, then the scheduler's lease epoch."""
        return (self.term if term is None else term, epoch)

    def refusal(self, node):
        """Why `node` cannot be added to the run now, in a sentence, or "" when it can: the owner checks before it invites a machine
        that asked to join. The member's own join checks the same again."""
        with self.cond:
            if self.finished:
                return FINISHED
            if node in self.banned:
                return f"{node} was stopped by the run's owner."
            if node in self.members or node in self.pending:
                return f"{node} is already in this run."
            if len(self.members) - len(self.leaving & set(self.members)) + len(self.pending) >= self.settings.max_members:  # a leaver's seat is free
                return f"The run already has its {self.settings.max_members} members."
            return ""

    def join(self, node, epoch=0, term=None):
        with self.cond:
            self._touch(node)
            self._advance()
            if self.finished:
                raise Stale(FINISHED)
            if node in self.banned:
                raise Stale(f"{node} was stopped by the run's owner.")
            self.gone.pop(node, None)
            known = self.members.get(node) or self.pending.get(node)
            if known:
                if self._lease(term, epoch) < (known["term"], known["epoch"]):
                    raise Stale(f"Stale lease epoch {epoch} for {node}: the current one is {known['epoch']}.")
                known["term"], known["epoch"] = self.term, epoch
                self.leaving.discard(node)
            else:
                if len(self.members) - len(self.leaving & set(self.members)) + len(self.pending) >= self.settings.max_members:  # a leaver's seat is free
                    raise Stale(f"The run already has its {self.settings.max_members} members.")
                self.sequence += 1
                entry = {"term": self.term, "epoch": epoch, "seq": self.sequence, "missed": 0}
                (self.pending if self.current is not None else self.members)[node] = entry  # a boundary is now, or at the round's end
                start = self.next_round + (1 if self.current is not None else 0)  # the round it first trains in (0-based)
                if start > 0:
                    self.joined_round[node] = start  # it came after the run began
                self._event("joined", node, self.next_round, f"{node} joined" + (f" and starts with round {start + 1}." if self.current is not None else "."))
            self._advance()
            self.cond.notify_all()
            return {**self.info(), "node": node}

    def leave(self, node, generation, epoch, term=None):
        with self.cond:
            self._touch(node)
            entry = self.members.get(node) or self.pending.get(node)
            if entry is None or (entry["term"], entry["epoch"]) != self._lease(term, epoch):
                raise Stale(f"{node} is not a member at lease epoch {epoch}.")
            if node in self.pending:
                del self.pending[node]
                self._event("left", node, self.next_round, f"{node} left before its first round.")
            elif self.current is None:
                del self.members[node]
                self.gone[node] = ("left", "Left the run.")
                self._event("left", node, self.next_round, f"{node} left the run.")
            else:
                self.leaving.add(node)  # leaves at the round boundary; the round stops waiting for it now
            self._advance()
            self.cond.notify_all()

    def lose(self, node):
        """The scheduler's lease on `node`'s slot lapsed: it leaves at the round boundary (the round stops waiting for it now) and, unlike
        `drop`, may join again. Does nothing for a node that is not a member."""
        with self.cond:
            self._touch()
            if node in self.pending:
                del self.pending[node]
                self._event("left", node, self.next_round, f"{node} lost its lease before its first round.")
            elif node in self.members and self.current is None:
                del self.members[node]
                self.gone[node] = ("left", "Its lease lapsed, so the run let it go.")
                self._event("left", node, self.next_round, f"{node} lost its lease and left the run.")
            elif node in self.members:
                self.leaving.add(node)
                self._event("left", node, self.next_round, f"{node} lost its lease; it leaves after round {self.next_round}.")
            self._advance()
            self.cond.notify_all()

    # -- rounds
    def steps_through(self, number):
        """The inner steps of the first `number` rounds."""
        return plan_before(self.plan, number)

    def _settings_for(self, number):
        """The settings as round `number` runs them: H as the plan has it for that round (it changes when the run is extended)."""
        h = plan_h(self.plan, number)
        return self.settings if h == self.settings.inner_steps else replace(self.settings, inner_steps=h)

    def _expected(self, current):
        return set(current.members) - self.leaving

    def _event(self, kind, node, number, text):
        """Append to the bounded event log the live view shows, on this coordinator's own clock (seconds since it started)."""
        self.event_seq += 1
        self.events.append({"seq": self.event_seq, "t": self.clock() - self.started, "kind": kind, "node": node, "round": number, "text": text})
        self._changed()

    def _changed(self):
        """Tell the watchers (`watch`) that what the live view shows of this run changed. Each must be quick: it runs under the lock."""
        for watcher in list(WATCHERS):
            watcher(self)

    def _advance(self):
        drules.accrue(self.rules, self.clock(), self._counting())
        while not self.finished and not self.superseded:  # a coordinator that has seen a higher term closes and opens nothing
            now = self.clock()
            current = self.current
            if current is None:
                if not self._open(now):
                    return
                continue
            expected = self._expected(current)
            complete = expected <= set(current.contributions)
            late = now >= current.deadline + self.settings.grace_seconds
            if not (complete or late):
                return
            if not current.contributions:  # nobody sent anything: count the silence, re-arm the deadline or give the round up
                for node in expected:
                    self._miss(node)
                if expected and expected <= set(self.members):
                    current.deadline = now + current.window
                    return
                self.current = None
                self._boundary()
                continue
            self._close(current, now)

    def _counting(self):
        """Whether the time now passing counts toward max_seconds: enough members are in, and the run is not paused between rounds (a
        round in flight when it was paused still counts until it closes)."""
        return (not self.finished and len(self.members) >= self.settings.min_members and not (self.paused and self.current is None))

    def _miss(self, node):
        entry = self.members.get(node)
        if entry is None:
            return
        entry["missed"] += 1
        self._event("late", node, self.next_round, f"{node} sent nothing for round {self.next_round}; its work counted for nothing.")
        if entry["missed"] >= self.settings.missed_limit:
            del self.members[node]  # it must join again
            self.gone[node] = ("dropped", f"Missed {entry['missed']} rounds in a row, so the run dropped it. It can rejoin when it is back.")
            self._event("dropped", node, self.next_round, f"{node} was dropped after missing {entry['missed']} rounds in a row.")

    def _boundary(self):
        for node in self.leaving:
            self.members.pop(node, None)
            if node not in self.gone:
                self.gone[node] = ("left", "Left the run.")
                self._event("left", node, self.next_round, f"{node} left the run.")
        self.leaving.clear()

    def _open(self, now):
        if self.paused or self.polishing:
            return False
        for node in sorted(self.pending, key=lambda n: self.pending[n]["seq"]):
            if len(self.members) < self.settings.max_members:
                self.members[node] = self.pending.pop(node)
        active = sorted(self.members, key=lambda n: self.members[n]["seq"])
        if len(active) < self.settings.min_members:
            return False
        if active != self.last_members:
            self.generation += 1
            self.last_members = active
        for node in [node for node in self.step_times if node not in active]:
            del self.step_times[node]  # a member that left is not timed any more: when it returns it is dealt a probe round
        settings = self._settings_for(self.next_round)  # H as the plan has it for this round
        train = train_seconds(settings, active, self.step_times)
        steps = timed_steps(settings, active, self.step_times, train) if train else deal_steps(settings, active, self.step_times)
        # a member that overlaps runs the previous round's deal (a round's assignment arrives after its steps), so the deadline covers the larger
        runs = {node: max(steps[node], self.last_steps.get(node, steps[node])) for node in active}
        self.last_steps = dict(steps)
        finishes = [(runs[node] * self.step_times[node], node) for node in active if self.step_times.get(node, 0) > 0]
        if train:  # time-boxed: every member trains for `train` seconds, so the deadline follows that, not the steps
            window = round_seconds(settings, max(train, self.last_train), 1)
        elif finishes:  # the slowest predicted finish, so the deadline is consistent with the steps run
            slowest = max(finishes)[1]
            window = round_seconds(settings, self.step_times[slowest], runs[slowest])
        else:
            window = round_seconds(settings, self.step_time)
        self.last_train = train
        first, count = self.spans[min(self.rules.source, len(self.spans) - 1)]
        ids = [*range(first, first + count), *(entry["shard"] for entry in self.live_lessons())]  # the source's shards, then the lessons
        if self.stages is not None:
            ids += self.stages.replay_ids(self, self.next_round)  # a share of the earlier sources, so the new text does not make it forget them
        self.current = Round(self.next_round, self.generation, active, deal_ids(self.cfg["seed"], self.next_round, active, ids),
                             now, now + window, window, steps=steps, train_seconds=train)
        self._changed()
        return True

    def _close(self, current, now):
        current.open, current.closed = False, now
        sent = current.contributions
        self.stage = "outer_step"  # read without the lock by snapshot() while this holds it (the evaluation below can take seconds)
        try:
            self._finish(current, now, sent)
        finally:
            self.stage = ""

    def _finish(self, current, now, sent):
        late = self.late if self.late_round == current.number - 1 else {}  # changes of the round before that missed its close
        self.late, self.late_round = {}, None
        delta = token_average([(tokens, d) for tokens, _, _, _, d in [*sent.values(), *late.values()]])
        norm = 0.
        if delta is not None:
            outer_lr = self.settings.outer_lr * self.rules.lr_scale  # halve_lr scales the outer step too
            lossy = self.settings.delta_dtype != "float32"
            before = {name: value.copy() for name, value in self.theta.items()} if lossy else None
            outer_step(self.theta, self.momentum, delta, outer_lr, self.settings.outer_momentum)
            if lossy:
                self._publish_update(before, current.number)
            norm = step_size(delta, self.momentum, outer_lr, self.settings.outer_momentum)
            self._event("outer_step", "", current.number, f"Outer step applied for round {current.number}: size {norm:.4g}, "
                                                          f"from {len(sent)} of {len(current.members)} members.")
        self.stage = "validating"
        dropped = [node for node in current.members if node not in sent]
        for node in current.members:
            if node in sent:
                if node in self.members:
                    self.members[node]["missed"] = 0
            elif node not in self.leaving:
                self._miss(node)
        times = {node: seconds / steps for node, (_, steps, seconds, _, _) in sent.items() if steps > 0 and 0 < seconds < math.inf}
        self.step_time = min(times.values()) if times else self.step_time
        for node, seconds in times.items():  # a smoothed per-member speed, which the next rounds are dealt by
            before = self.step_times.get(node)
            self.step_times[node] = seconds if before is None else (1 - STEP_TIME_SMOOTHING) * before + STEP_TIME_SMOOTHING * seconds
        tokens = sum(t for t, _, _, _, _ in sent.values())
        accepted_tokens = tokens + sum(t for t, _, _, _, _ in late.values())
        losses = [(t, loss) for t, _, _, loss, _ in sent.values() if loss is not None and t > 0]
        row = {"round": current.number, "step": plan_before(self.plan, current.number + 1), "generation": current.generation,
               "members": list(current.members), "contributed": list(sent), "dropped": dropped, "tokens": tokens,
               "accepted_tokens": accepted_tokens, "late": list(late),
               "steps": {node: value[1] for node, value in sent.items()}, "dealt": dict(current.steps),
               "loss": sum(t * loss for t, loss in losses) / sum(t for t, _ in losses) if losses else None,
               "val_loss": self.evaluate(self.theta), "seconds": now - current.opened,
               "delta_bytes": sum(current.uploaded.get(node, 0) for node in sent), "delta_dtype": self.settings.delta_dtype, "delta_sparsity": self.settings.delta_sparsity, "deadline_seconds": current.window,
               "theta_bytes": sum(current.downloaded.get(node, self.theta_nbytes) for node in current.members),  # what the members downloaded for the round (theta, or the outer update that led to it; full theta for one that has not yet)
               "node_tokens": {node: value[0] for node, value in sent.items()}, "node_seconds": {node: value[2] for node, value in sent.items()},
               "node_bytes": {node: current.uploaded.get(node, 0) for node in sent},
               "node_rates": {node: current.rates[node] for node in sent if node in current.rates},
               "lr": self.settings.outer_lr * self.rules.lr_scale, "tokens_per_second": tokens / max(1e-9, now - current.opened), "grad_norm": 0.,
               "outer_step_norm": norm}
        arrived = {n: current.marks[n]["delta_received"] - current.opened for n in sent if "delta_received" in current.marks.get(n, {})}
        row["health"] = run_health.record(row, {n: v[3] for n, v in sent.items()}, arrived, self.theta_nbytes)  # what this round measured
        self.history.append(row)
        self.deal = current.deal
        self.closed = (current.number, current.generation, set(current.members), set(sent))
        self.next_round += 1
        self.current = None
        self.finished = self.next_round >= self.settings.rounds
        self._save_start(current.number)
        verdict = self._apply_rules(row, current)
        if verdict["action"] == "stop":
            self.finished = True
        self.timelines.append(self._timeline(current))
        self._boundary()
        self._save(row, current)
        if self.finished:
            self._event("finished", "", current.number, self.rules.stop_text or f"The run finished after {self.next_round} rounds.")
        else:
            self._start_polish(row, current, verdict)
            if not self.polishing:
                self._start_versus(current)
        self.cond.notify_all()

    def _publish_update(self, before, number):
        """Quantize the outer update (theta now - theta before, plus what the last one lost) as the members will download it, and make
        theta the dequantized update added to `before`: the coordinator and every member add the same fp32 numbers to the same fp32
        theta, so every node's theta is bit-identical. What the quantization loses goes into the next update (error feedback)."""
        wanted = {name: self.theta[name] - before[name] for name in before}
        body, self.update_residual = encode_feedback(wanted, self.settings.delta_dtype, self.update_residual, update_from=number)
        self.theta.update(add_update(before, decode_arrays(body)))
        self._update, self._params = (number + 1, body), (None, b"")

    def _theta_replaced(self):
        """Theta was replaced by something other than an outer update (a pivot back to the best round, the teacher's polish): the
        members take it whole."""
        self._params, self._state = (None, b""), (None, b"")
        self._update, self.update_residual = None, {}
        self.late, self.late_round = {}, None  # computed against weights that are gone

    def _apply_rules(self, row, current):
        """Fold the closed round into the stop and pivot rules (studio_diloco_rules): keep the best weights, go back to them and apply the
        next pivot when the loss kept getting worse or stopped improving, or end the run with the rule's reason. Returns the verdict."""
        rules = self.rules
        drules.accrue(rules, self.clock(), len(self.members) >= self.settings.min_members and not self.paused)
        loss = row["val_loss"]
        better = loss is not None and math.isfinite(loss) and (rules.best_loss is None or loss < rules.best_loss)
        since = current.number if better else rules.best_round  # the best round once this one is folded in: the lessons written after it are the ones a drop_lessons pivot leaves out
        lessons = len(self.live_lessons(since)) if since is not None else 0
        if self.stages is not None:
            def would(sources):  # the verdict the rules would give with `sources` data sources, on a copy of their state
                return drules.observe(copy.deepcopy(rules), self.settings, current.number, row["step"], row["val_loss"], sources=sources,
                                      final=self.finished, lessons=lessons)
            self.stages.before_observe(self, row, would)
        verdict = drules.observe(rules, self.settings, current.number, row["step"], row["val_loss"],
                                 sources=len(self.spans) if self.stages is None else self.stages.sources(self),
                                 final=self.finished, lessons=lessons)  # the round `rounds` ends anyway: keep the best, no restore or stop reason
        row["best"] = rules.best_round == current.number
        if verdict["improved"]:
            self._write_best()
        if verdict["action"] == "pivot" and rules.best_round is not None:
            self._restore_best()
            row["pivot"] = verdict["name"]
            if verdict["name"] == "drop_lessons":
                for entry in self.live_lessons(rules.best_round):
                    entry["status"] = "left_out"
                    self.lesson_windows.pop(entry["shard"], None)
            before = f"{verdict['before']:.4g}" if verdict["before"] is not None else "no number"
            why = (f"the validation loss got worse {verdict['streak']} rounds in a row" if verdict["why"] == "worse"
                   else "the validation loss stopped improving")
            self._event("pivot", "", current.number,
                        f"Round {current.number}: {why} ({before}, best {verdict['best']:.4g} at round {verdict['best_round']}). "
                        f"Went back to round {verdict['best_round']}'s weights and {drules.PIVOT_TEXT[verdict['name']]}.")
        if self.stages is not None:
            self.stages.after_observe(self, row, verdict)
        return verdict

    def _write_best(self):
        """best.npz: the best round's weights as a checkpoint, plus the outer momentum it had (so a pivot continues from exactly there)."""
        arrays = self._checkpoint_arrays(plan_before(self.plan, self.rules.best_round + 1))
        arrays.update({STATE_PREFIXES[1] + k: v for k, v in self.momentum.items()})
        self._write_npz(self.directory / "best.npz", arrays)

    def _restore_best(self):
        with np.load(self.directory / "best.npz") as stored:
            for name in self.theta:
                self.theta[name] = np.array(stored["weight::" + name], dtype=np.float32)
                self.momentum[name] = np.array(stored[STATE_PREFIXES[1] + name], dtype=np.float32)
        self._theta_replaced()

    def _view(self, number, node):
        entry = self.members.get(node) or self.pending.get(node)
        base = {"term": self.term, "round": number, "member": entry is not None, "current": self.next_round, "paused": self.paused,
                "polishing": self.polishing, "rounds": self.settings.rounds, "plan": [dict(entry) for entry in self.plan]}
        if self.finished:
            return {**base, "state": "finished"}
        if number < self.next_round:
            return {**base, "state": "closed"}
        current = self.current
        if number == self.next_round and current is not None and node in current.members and node not in self.leaving:
            rank = current.members.index(node)
            return {**base, "state": "open", "generation": current.generation, "rank": rank,
                    "replica": node in replica_nodes(current.members, self.settings.replicas, self.prefer),
                    "deadline_s": max(0., current.deadline - self.clock()), "shards": current.deal[node], "members": len(current.members),
                    "lr_scale": self.rules.lr_scale, "steps": current.steps[node], "train_s": current.train_seconds}
        return {**base, "state": "pending"}

    def round_view(self, number, node="", wait=0.):
        """Where round `number` stands for `node`: open (with its assignment), pending, closed, or finished. With wait, a pending
        round is held until that changes, or for at most LONG_POLL seconds; any other state is answered at once."""
        end = time.monotonic() + min(max(wait, 0.), LONG_POLL)
        with self.cond:
            self._touch(node)
            self._advance()
            first = self._view(number, node)
            while first["state"] == "pending":
                left = end - time.monotonic()
                if left <= 0:
                    return first
                self.cond.wait(min(left, .25))  # a short slice: a deadline passing on the clock wakes nobody else
                self._advance()
                view = self._view(number, node)
                if (view["state"] != first["state"] or view.get("member") != first.get("member") or view["paused"] != first["paused"]
                        or view["polishing"] != first["polishing"] or view["plan"] != first["plan"]):
                    return view
            return first

    def ensure_serving(self):
        """Gone when the run is stopped or finished: a member still asking for weights or shards should leave."""
        if self.stopped or self.finished:
            raise Gone(f"Run {self.run} {'was stopped' if self.stopped else 'is finished'}.")

    def params_bytes(self, number, node="", since=None, count=True):
        """Theta of the open round as an .npz body; `node` (who is asking) is noted on the round's timeline as having taken it. A member
        that holds theta of the round before (`since`) is given the outer update that led to this round's theta instead (a quantized
        body marked with UPDATE_KEY), unless the run sends fp32 or theta was replaced since (a pivot, a polish): then it is theta.
        `count` False is a later part of a ranged download: the bytes are counted once, with the part that starts the body."""
        self.ensure_serving()
        with self.cond:
            self._touch()
            self._advance()
            if self.current is None or self.current.number != number:
                raise Stale(f"Round {number} is not the open round (the open one is {self.current.number if self.current else 'none'}).")
            if since is not None and since == number - 1 and self._update is not None and self._update[0] == number:
                body = self._update[1]
            else:
                if self._params[0] != number:
                    self._params = (number, encode_arrays(self.theta, "float32"))
                body = self._params[1]
            if count and node in self.current.members:
                self.current.marks.setdefault(node, {}).setdefault("theta_taken", self.clock())
                self.current.downloaded[node] = self.current.downloaded.get(node, 0) + len(body)
            return body

    def state_bytes(self, number):
        """What a replica keeps beside theta of round `number` (the latest, `next_round`): the outer momentum, fp32, and the text of
        diloco.json as written when the round before closed, in one .npz. Stale for any other round; LookupError before a round has
        closed (nothing is saved yet, and theta is the seeded initialization)."""
        with self.cond:
            self._touch()
            self._advance()
            if number != self.next_round:
                raise Stale(f"Round {number} is not the latest (it is {self.next_round}).")
            path = self.directory / "diloco.json"
            if not path.is_file():
                raise LookupError("No round has closed yet.")
            if self._state[0] != number:
                self._state = (number, encode_state(self.momentum, path.read_bytes()))
            return self._state[1]

    def final_bytes(self):
        """last.npz of a finished run: its final weights, for the owner when the run finished under this coordinator after a failover
        (#361): the best round's weights (best.npz) when the run kept any, else last.npz. Stale (409) while rounds are left, and for a
        stopped run, which saves nothing (#333)."""
        with self.cond:
            if self.stopped:
                raise Stale(f"Run {self.run} was stopped; it saves no weights.")
            if not self.finished:
                raise Stale(f"Run {self.run} has not finished; {self.settings.rounds - self.next_round} rounds are left.")
            best = self.directory / "best.npz"
            return (best if best.is_file() else self.directory / "last.npz").read_bytes()

    def validation_bytes(self):
        """The validation windows, as a shard is: for a replica, once per run. A replica never trains on them."""
        return shard_blob(self.val, 0, len(self.val))

    def progress(self, number, node, generation, epoch, phase, step=0, loss=None, tokens_per_second=None, term=None):
        """Take a member's report of where it is in its round (no weights): the latest per member is kept for the live view, and a
        training or uploading report also marks the round's timeline and adds a sample for its charts. Stale (409) for a node that
        is not a member or a lease epoch that is not its own, and for a training or uploading report about a round that is closed
        or of another generation, as a delta is. An overlapping member that already sent the open round trains the next one while
        the others finish: its reports for that round only freshen its report's time. Returns False, taking nothing, when a closing
        round holds the lock for a while: a report is never worth holding a request for."""
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {', '.join(PHASES)}.")
        self.beats[node] = self.clock()  # a report refused for a closing round is still a sign of life
        if not self.lock.acquire(timeout=PROGRESS_WAIT):
            return False
        try:
            self._touch()
            entry = self.members.get(node) or self.pending.get(node)
            if entry is None:
                raise Stale(f"{node} is not a member of this run.")
            if self._lease(term, epoch) != (entry["term"], entry["epoch"]):
                raise Stale(f"Stale lease epoch {epoch} for {node}: the current one is {entry['epoch']}.")
            now, current = self.clock(), self.current
            report = self.reports.setdefault(node, {"samples": [], "sample_round": None})
            if phase in WORK_PHASES:
                if (current is not None and current.open and number == current.number + 1 and node in current.contributions
                        and generation == current.generation and "round" in report):
                    report["at"] = now  # ahead (overlap): nothing to mark until its round opens
                    return True
                if current is None or current.number != number or not current.open:
                    raise Stale(f"Round {number} is closed (the open round is {current.number if current else 'none'}).")
                if node not in current.members:
                    raise Stale(f"{node} is not a member of round {number}.")
                if generation != current.generation:
                    raise Stale(f"Stale generation {generation}: round {number} is generation {current.generation}.")
                marks = current.marks.setdefault(node, {})
                marks.setdefault("first_step", now)
                if phase == "uploading":
                    marks.setdefault("delta_sent", now)
                if report["sample_round"] != number:
                    report["samples"], report["sample_round"] = [], number
                report["samples"].append({"step": int(step), "t": now - current.opened, "loss": loss, "tokens_per_second": tokens_per_second})
                del report["samples"][:-SAMPLES_KEPT]
            report.update(round=number, phase=phase, step=int(step), loss=loss, tokens_per_second=tokens_per_second, at=now)
            self._changed()
            return True
        finally:
            self.lock.release()

    def shard_path(self, shard):
        """The shard's .npz body as a file, written the first time it is asked for and served from there after: building it takes
        a copy of the windows through np.savez, which every member (and every replica) of the run would otherwise make again."""
        with self.blob_lock:
            path = self.blobs.get(shard)
            if path is None or not path.is_file():
                body = self.shard_bytes(shard)  # KeyError for a shard there is not
                folder = self.directory / "served_shards"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"shard{shard}.npz"
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(body)
                temporary.replace(path)
                self.blobs[shard] = path
            return path

    def shard_hash(self, shard):
        """The sha256 of the shard's .npz body: a member that holds a shard with this hash need not download it."""
        with self.blob_lock:
            known = self.hashes.get(shard)
        if known is None:
            digest = hashlib.sha256()
            with self.shard_path(shard).open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            known = self.hashes[shard] = digest.hexdigest()
        return known

    def shard_bytes(self, shard):
        if shard in self.lesson_windows:  # a lesson shard: its ids come after the data sources'
            return shard_blob(self.lesson_windows[shard], 0, len(self.lesson_windows[shard]))
        if not 0 <= shard < self.shard_count:
            raise KeyError(shard)
        if self.parts is not None:
            return shard_blob(self.parts[shard], 0, len(self.parts[shard]))
        windows, bounds, first = self.train, self.bounds, 0
        for (low, count), (later, edges) in zip(self.spans[1:], self.later):
            if shard >= low:
                windows, bounds, first = later, edges, low
        shard -= first
        return shard_blob(windows, int(bounds[shard]), int(bounds[shard + 1]))

    def submit(self, number, node, generation, epoch, tokens, steps, delta, seconds=0., loss=None, size=0, term=None, rates=None):
        """Take a member's pseudo-gradient for the open round. Stale is raised (409) for any other round or generation, a
        member that is not in the round, a lease epoch that is not the member's, or a second contribution."""
        with self.cond:
            self._touch(node)
            self._advance()
            if self.superseded:  # the route admitted the term before this coordinator stepped down and released its theta (below, an empty one fails the shape check as a 400)
                raise StaleTerm(f"{TERM_STALE} {self.term if term is None else term}: the run is at term {self.superseded}, so this node no longer coordinates it.",
                                self.superseded)
            current = self.current
            carried = False
            try:
                self._admit(current, number, node, generation, epoch, term)
            except Stale as exc:
                carried = self._carries(current, number, node, generation, epoch, term)
                if not carried:
                    self._event("stale", node, number, f"A change from {node} for round {number} was refused: {exc}")
                    raise
            if set(delta) != set(self.theta) or any(delta[name].shape != self.theta[name].shape for name in delta):
                raise ValueError("The delta does not match the model's parameters.")
            if not all(np.isfinite(value).all() for value in delta.values()):
                raise ValueError("The delta holds a non-finite value.")
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("A contribution's seconds must be finite and not negative.")
            if tokens < 0 or steps < 1:
                raise ValueError("A contribution needs at least one step.")
            if carried:  # too late for its round, not for the next outer step: it counts there, stale by one round and weighted the same
                self.late[node], self.late_round = (int(tokens), int(steps), float(seconds), loss, delta), number
                self.members[node]["missed"] = 0
                self._event("late", node, number, f"The change from {node} for round {number} came in after it closed; it counts in round {number + 1}.")
                return
            current.contributions[node] = (int(tokens), int(steps), float(seconds), loss, delta)
            current.uploaded[node] = int(size)
            if rates:
                current.rates[node] = rates
            marks = current.marks.setdefault(node, {})
            marks["delta_received"] = self.clock()
            marks.setdefault("delta_sent", marks["delta_received"])
            self._changed()
            self._advance()
            self.cond.notify_all()

    def _carries(self, current, number, node, generation, epoch, term):
        """Whether a change refused as stale can still count: it is for the round that closed just before the open one, from a member
        of it that did not contribute, with the round's generation and the member's lease."""
        if self.closed is None or current is None or not current.open or self.closed[0] != number or current.number != number + 1:
            return False
        _, closed_generation, members, contributors = self.closed
        entry = self.members.get(node)
        return (generation == closed_generation and node in members and node not in contributors and node not in self.late and entry is not None
                and self._lease(term, epoch) == (entry["term"], entry["epoch"]))

    def _admit(self, current, number, node, generation, epoch, term=None):
        if current is None or current.number != number or not current.open:
            raise Stale(f"Round {number} is closed (the open round is {current.number if current else 'none'}).")
        if generation != current.generation:
            raise Stale(f"Stale generation {generation}: round {number} is generation {current.generation}.")
        entry = self.members.get(node)
        if node not in current.members or entry is None:
            raise Stale(f"{node} is not a member of round {number}.")
        if self._lease(term, epoch) != (entry["term"], entry["epoch"]):
            raise Stale(f"Stale lease epoch {epoch} for {node}: the current one is {entry['epoch']}.")
        if node in current.contributions:
            raise Stale(f"{node} already contributed to round {number}.")

    # -- the owner's view and controls
    def _timeline(self, found):
        """Round `found` as the swimlane view needs it: offsets in seconds from its opening, per member, of the moments in MARKS."""
        lanes = {}
        for node in found.members:
            sent = found.contributions.get(node)
            lane = {name: (found.marks.get(node, {}).get(name) - found.opened if found.marks.get(node, {}).get(name) is not None else None) for name in MARKS}
            lanes[node] = {**lane, "counted": True if sent else (None if found.open else False),
                           "tokens": sent[0] if sent else 0, "steps": sent[1] if sent else 0, "delta_bytes": found.uploaded.get(node, 0)}
        return {"round": found.number, "open": found.open, "deadline": found.deadline - found.opened, "window": found.window,
                "closed": None if found.closed is None else found.closed - found.opened, "lanes": lanes}

    def _phase(self, current):
        """Where the run is in its cycle: waiting (for members), next_round (members are taking theta; getting_ready in the first round), training, collecting (some
        change is in or on its way), outer_step, validating, or finished / stopped."""
        if self.stopped or self.finished:
            return "stopped" if self.stopped else "finished"
        if self.stage:
            return self.stage
        if self.judging:
            return "judging"
        if self.polishing:
            return "polishing"
        if current is None:
            return "paused" if self.paused else "waiting"
        reports = [self.reports.get(node) for node in current.members]
        working = [r for r in reports if r and r.get("round") == current.number and r.get("phase") in WORK_PHASES]
        if current.contributions or any(r["phase"] == "uploading" for r in working):
            return "collecting"
        if working:
            return "training"
        return "next_round" if self.history else "getting_ready"  # the first round: nothing has happened yet that a step of the cycle could tick

    def snapshot(self):
        """What the Training tab shows, from this process: the run's state and round, each node the run knows, the rounds so far, and
        the live view (phase, deadline, each member's latest report, the recent rounds' timelines and the event log). While a round
        closes (it holds the lock for the validation) the last answer is returned with the closing phase, instead of waiting for it."""
        if not self.lock.acquire(timeout=PROGRESS_WAIT):
            if self._last is None:
                self.lock.acquire()
            else:
                return {**self._last, "phase": self.stage or self._last["phase"]}
        try:
            return self._snapshot()
        finally:
            self.lock.release()

    def _snapshot(self):
        self._advance()
        current, now = self.current, self.clock()
        nodes = {}
        for node, entry in self.members.items():
            behind = entry["missed"]
            nodes[node] = {"state": "behind" if behind else "training", "missed": behind,
                           "sent": current is not None and node in current.contributions,
                           "note": f"Missed {behind} round{'s' if behind != 1 else ''} in a row; the run drops it after {self.settings.missed_limit}." if behind else ""}
        for node in self.pending:
            nodes[node] = {"state": "joining", "missed": 0, "sent": False, "note": "Joins at the start of the next round."}
        for node, (state, note) in self.gone.items():
            nodes[node] = {"state": state, "missed": 0, "sent": False, "note": note}
        for node, found in nodes.items():
            found["joined_round"] = self.joined_round.get(node)
        deal = current.deal if current is not None else self.deal
        for node, found in nodes.items():
            report = self.reports.get(node)
            if report is not None and "round" not in report:
                report = None  # made by a report that was refused (Stale) before it said anything
            live = bool(report) and report["round"] == self.next_round
            found["progress"] = {
                "phase": report["phase"] if report else "waiting", "step": report["step"] if live else 0,
                "steps": current.steps.get(node, plan_h(self.plan, self.next_round)) if current is not None else plan_h(self.plan, self.next_round),
                "loss": report["loss"] if live else None, "tokens_per_second": report["tokens_per_second"] if live else None,
                "reported_ago": now - report["at"] if report else None,
                "samples": [dict(s) for s in report["samples"] if report["sample_round"] == self.next_round and s["loss"] is not None] if report else []}
            found["shards"] = len(deal.get(node, ()))
        timelines = list(self.timelines) + ([self._timeline(current)] if current is not None else [])
        state = "stopped" if self.stopped else "finished" if self.finished else "running" if current is not None else "waiting"
        view = {"state": state, "phase": self._phase(current), "round": self.next_round, "rounds": self.settings.rounds, "generation": self.generation,
                "settings": dict(self.settings.__dict__), "nodes": nodes, "history": [dict(row) for row in self.history],
                "deadline_s": max(0., current.deadline - now) if current is not None else None, "grace_s": self.settings.grace_seconds,
                "window_s": current.window if current is not None else None, "contributed": len(current.contributions) if current is not None else 0,
                "expected": len(self._expected(current)) if current is not None else 0, "theta_bytes": self.theta_nbytes, "now_s": now - self.started,
                "timeline": timelines, "events": [dict(event) for event in self.events],
                "rules": {**self.rules.to_json(), "points": len(self.rules.points)}, "paused": self.paused,
                "lessons": [dict(entry) for entry in self.lessons]}
        self._last = view
        return view

    def drop(self, node):
        """Stop one member: it leaves at once (the round stops waiting for it) and cannot rejoin this run. Stale if it is not a member."""
        with self.cond:
            self._touch()
            if node not in self.members and node not in self.pending:
                raise Stale(f"{node} is not a member of this run.")
            self.banned.add(node)
            self.gone[node] = ("stopped", "Stopped by you.")
            self._event("left", node, self.next_round, f"{node} was stopped by you.")
            if node in self.pending:
                del self.pending[node]
            elif self.current is None:
                del self.members[node]
            else:
                self.leaving.add(node)
            self._advance()
            self.cond.notify_all()

    def stop(self, reason="The run was stopped by you."):
        """End the run now: every member is told it is finished at its next call. The weights of the last finished round stay on disk."""
        with self.cond:
            self._touch()
            if self.finished:
                return
            from studio_core import atomic_json
            self.stopped = self.finished = True
            self.stopped_at = self.clock()
            self._polish_stop.set()
            self._event("finished", "", self.next_round, reason)
            if self.record:
                self.record["status"] = "stopped"
                atomic_json(self.directory / "run.json", self.record)
            self._remember()
            self.cond.notify_all()

    def pause(self):
        """Open no new round until resume(); a round in flight finishes. Kept in diloco.json, so a restart or a successor coordinator
        comes up paused. The time paused between rounds does not count toward max_seconds. Stale once the run is over; a pause of a
        paused run changes nothing."""
        with self.cond:
            self._touch()
            if self.finished:
                raise Stale("The run is over; there is nothing to pause.")
            if self.paused:
                return
            self.paused = True  # first, so the advance below closes a due round but opens no other
            self._advance()
            if self.finished:  # the advance ended the run: there was nothing left to pause
                self.paused = False
                raise Stale("The run is over; there is nothing to pause.")
            drules.accrue(self.rules, self.clock(), self._counting())
            self._event("paused", "", self.next_round, "The run was paused by you. No new round opens until you resume it.")
            self._save_paused()
            self._remember()
            self.cond.notify_all()

    def resume(self):
        """Undo pause(): the next round opens as soon as the members are in. Stale once the run is over."""
        with self.cond:
            self._touch()
            if self.finished:
                raise Stale("The run is over; there is nothing to resume.")
            if not self.paused:
                return
            self._advance()
            self.paused = False
            drules.accrue(self.rules, self.clock(), self._counting())
            self._event("resumed", "", self.next_round, "The run was resumed by you.")
            self._save_paused()
            self._advance()
            self._remember()
            self.cond.notify_all()

    def _save_paused(self):
        """Write the paused flag (and the active seconds it freezes) to diloco.json without closing a round. Before the first round has
        closed there is no diloco.json yet (and no weights worth a restart): only the flag goes to paused.json, which build reads."""
        from studio_core import atomic_json
        self._state = (None, b"")  # a replica that fetched the state before this change must fetch it again
        path = self.directory / "diloco.json"
        if not path.is_file():
            atomic_json(self.directory / "paused.json", {"paused": self.paused, "rules": self.rules.to_json()})
            return
        saved = json.loads(path.read_text(encoding="utf-8"))
        atomic_json(path, {**saved, "paused": self.paused, "rules": self.rules.to_json()})

    def extend(self, rounds=None, inner_steps=None):
        """Make the run longer while it goes: `rounds` is the new total (at least the rounds it has), `inner_steps` the new H. Both apply to the rounds still to come, from round `first` on: the first round whose
        assignment no member can have taken yet (a member that overlaps trains round n+1 on the assignment of round n). The learning-rate
        schedule carries on from where it is: its remaining share is spread over the new remaining steps (see plan_h), so the rate does
        not jump. Returns {"rounds", "inner_steps", "from_round"}. Stale once the run is over; ValueError, saying why, for numbers that
        change nothing or shorten the run (Stop ends it sooner)."""
        with self.cond:
            self._touch()
            self._advance()
            if self.finished:
                raise Stale("The run is over; it can no longer be extended.")
            old_rounds, h = self.settings.rounds, self.settings.inner_steps
            rounds = old_rounds if rounds is None else int(rounds)
            inner_steps = h if inner_steps is None else int(inner_steps)
            if rounds < old_rounds:
                raise ValueError(f"The run has {old_rounds} rounds; extending can only add rounds (Stop ends a run sooner).")
            if not 1 <= inner_steps <= 1_000_000 or rounds > 100_000:
                raise ValueError("A run has at most 100000 rounds, and a round has between 1 and 1000000 steps.")
            first = min(self.next_round + 2, old_rounds)
            if rounds == old_rounds and (inner_steps == h or first >= old_rounds):
                raise ValueError("Nothing changes: give more rounds than the run has"
                                 + ("." if inner_steps == h else ", since no round is left that could run another number of steps."))
            # where the schedule is at `first`, as a share of its frame; at most where the last old round starts (nothing of the schedule
            # is left past the end, and a run extended from there has no decay left to spread)
            share = min(plan_position(self.plan, number) / plan_frame(self.plan, number) for number in (first, old_rounds - 1))
            kept = [entry for entry in self.plan if entry["round"] < first]
            entry = {"round": first, "q": share, "h": inner_steps, "rounds": rounds}
            # the warmup is a count of steps, not a share: carry the position (not the share) over to the new frame, so the rate is
            # continuous when the extension lands inside the warmup, or when a warmup makes the share not the same point of the decay
            old_frame, new_frame = plan_frame(self.plan, first), plan_frame([*kept, entry], first)
            entry["q"] = plan_carry(share * old_frame, old_frame, new_frame, self.cfg.get("warmup") or 0) / new_frame
            self.plan = [*kept, entry]
            self.settings = replace(self.settings, rounds=rounds, inner_steps=inner_steps)
            self._state = (None, b"")  # a replica that fetched the state before this change must fetch it again
            said = f"Extended to {rounds} rounds at round {self.next_round}." if rounds != old_rounds else f"Changed the steps a round to {inner_steps} at round {self.next_round}."
            if inner_steps != h and rounds != old_rounds:
                said += f" From round {first} each runs {inner_steps} steps."
            elif inner_steps != h:
                said += f" It applies from round {first}."
            self._event("extended", "", self.next_round, said)
            self._save_extended()
            self._remember()
            self.cond.notify_all()
            return {"rounds": rounds, "inner_steps": inner_steps, "from_round": first}

    def _save_extended(self):
        """Write the new length, H and plan to diloco.json (and run.json) without closing a round, so a restart or a successor coordinator
        keeps them. Before the first round has closed there is no diloco.json: extended.json holds them instead (see build)."""
        from studio_core import atomic_json
        path = self.directory / "diloco.json"
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            atomic_json(path, {**saved, "settings": self.settings.__dict__, "plan": self.plan})
        else:  # no round has closed: extended.json holds it for build, which reads it like paused.json
            atomic_json(self.directory / "extended.json", {"rounds": self.settings.rounds, "inner_steps": self.settings.inner_steps, "plan": self.plan})
        if self.record:
            self.record.update({"rounds": self.settings.rounds, "inner_steps": self.settings.inner_steps,
                                "steps": plan_total(self.plan)})
            atomic_json(self.directory / "run.json", self.record)

    # -- the teacher's polish
    def _start_polish(self, row, current, verdict):
        """After every polish_every-th closed round (called with the lock held, the round saved): hand last.npz to the teacher on a thread
        of its own, and open no round until it is done. Never after the final round, and not after a round that went back to the best
        weights (the loss measured is not of the weights now held). Does nothing without prompts, on a coordinator made from a replica
        (which holds no data of its own to make them from), or once the run has ended."""
        every = self.settings.polish_every
        if (not every or not self.polish_on or self.superseded or self.stopped or self.next_round % every or verdict["action"] == "pivot"):
            return
        prompts = self.prompt_window(self.next_round // every)
        if not prompts:
            return
        stop = self._polish_stop = threading.Event()
        context = {"number": current.number, "step": row["step"], "old": row["val_loss"], "members": list(current.members), "prompts": prompts, "row": row}
        self.polishing = True
        self._event("polish", "", current.number, f"Round {current.number}: the teacher ({self.settings.teacher}) is polishing the weights. "
                                                 f"No round opens until it is done.")
        try:
            threading.Thread(target=self._polish_run, args=(context, stop), daemon=True, name="diloco-polish").start()
        except Exception as exc:  # no thread to be had: the gate must not stay shut
            self.polishing = False
            row["polish"] = "skipped"
            self._event("polish", "", current.number, f"Round {current.number}: the teacher polish could not start ({exc}). Training goes on without it.")

    def _polish_run(self, context, stop):
        """The polish thread: ask the teacher (self.polisher), score a polished checkpoint on the validation windows, then hand the outcome
        to _polish_done under the lock. Nothing here touches the run's state; no round is open to change it. What failed is said in
        outcome["error"]; the teacher's lessons are kept whichever step after it failed."""
        outcome = {"error": "", "rows": [], "theta": None, "loss": None}
        try:
            base = self.directory / "last.npz"
            try:
                result = self.polisher(self.directory / "teacher", self.teacher_settings, str(base), context["prompts"], self.cfg.get("device", "auto"),
                                       stop, lambda event: None)
            except Exception as exc:  # the teacher is optional: a failed polish is skipped and the run goes on
                outcome["error"] = "the teacher failed: " + (str(exc) or type(exc).__name__)[:300]
            else:
                outcome["rows"] = [dict(found) for found in result.get("lessons") or []]
                path = result.get("checkpoint")
                if self.settings.polish and path and not stop.is_set() and Path(path).resolve() != base.resolve():
                    try:
                        outcome["theta"] = self._weights_of(path, self.theta)
                    except Exception as exc:
                        outcome["error"] = "the polished weights could not be read: " + (str(exc) or type(exc).__name__)[:300]
                    else:
                        try:
                            outcome["loss"] = self.evaluate(outcome["theta"])
                        except Exception as exc:
                            outcome["theta"] = None
                            outcome["error"] = "the polished weights could not be scored: " + (str(exc) or type(exc).__name__)[:300]
        finally:
            for folder in (self.directory / "teacher").glob("rlhf_*"):  # the preference runs have given what they had: their checkpoints are not kept
                shutil.rmtree(folder, ignore_errors=True)
        job = None
        with self.cond:
            try:
                self._polish_done(context, outcome, stop)
                job = self._versus_job(stop)  # the polish may have made a new best
                if job is not None:
                    self._begin_judging(job, context["number"])
            finally:
                if job is None:
                    self.polishing = self.judging = False
                    self._advance()
                    self.cond.notify_all()
        if job is not None:
            self._versus_run(job, stop)

    def _polish_done(self, context, outcome, stop):
        """Take the teacher's work into the run (lock held). The polished weights replace theta only when the teacher's head-to-head gate
        passed (the polisher returned another checkpoint) and their validation loss is no worse than the round's: the round's val_loss is
        then the polished one (the old one is kept as val_loss_before_polish) and every file says so. The better answers become a
        lesson: an extra shard from the next round on. A stopped, released or superseded run takes nothing."""
        if stop.is_set() or self.superseded or self.finished or not self.theta:
            self._remember()
            return
        number, row, old, new = context["number"], context["row"], context["old"], outcome["loss"]
        drules.accrue(self.rules, self.clock(), self._counting())  # the time the teacher took counts toward max_seconds
        said = []
        added = self._add_lessons(number, outcome["rows"]) if self.settings.lessons else 0  # the file first: a lesson never reuses a shard id
        if added:
            row["lessons"] = added
        accepted = False
        if outcome["error"]:
            row["polish"] = "skipped"
            said.append(f"polish was skipped ({outcome['error']}). Training goes on without it")
        elif outcome["theta"] is not None:
            if old is not None and new is not None and math.isfinite(new) and new <= old:
                accepted = True
                self.theta = outcome["theta"]
                self._theta_replaced()
                verdict = drules.revise(self.rules, self.settings, number, context["step"], old, new)
                row.update({"polish": "accepted", "val_loss_before_polish": float(old), "val_loss": float(new), "best": self.rules.best_round == number})
                if verdict["improved"]:
                    self._write_best()
                if verdict["action"] == "stop":
                    self.finished = True
                said.append(f"kept the polished weights (validation loss {new:.4g}, was {old:.4g})")
            else:
                row["polish"] = "rejected"
                if new is not None and math.isfinite(new):
                    row["polish_val_loss"] = float(new)
                said.append("kept the weights it had: the polished ones were no better on the validation windows"
                            + (f" ({new:.4g} against {old:.4g})" if new is not None and old is not None else ""))
        elif self.settings.polish:
            row["polish"] = "rejected"
            said.append("the polished weights did not win the teacher's head-to-head, so they were not kept")
        if not accepted and not self.finished:
            found = drules.stop_reason(self.rules, self.settings)  # the polish may have used up max_seconds
            if found:
                self.rules.stop_rule, self.rules.stop_text = found
                self.finished = True
        self._persist(context["members"], self.deal)  # right after best.npz: the files agree on the weights held
        if accepted:
            self._rewrite_metric(row)
        self._write_record(row)
        if self.settings.lessons and not added and not outcome["error"]:
            said.append("had no lessons to give")
        self._event("polish", "", number, f"Round {number}: the teacher " + ("; ".join(said) or "had nothing to add") + ".")
        if added:
            self._event("lessons", "", number, f"Round {number}: {added} lesson{'s' if added != 1 else ''} join training from round {number + 1}.")
        if self.finished:
            self._event("finished", "", number, self.rules.stop_text or f"The run finished after {self.next_round} rounds.")
        self._remember()

    # -- the head-to-head against the starting weights (target_win_rate)
    def _save_start(self, number):
        """start.npz, once: the weights after the first trained round, which the best weights are compared with (a restart's is its
        restart_from checkpoint, copied in by build). Called with the lock held, the round folded in and before any rule restores weights."""
        if self.settings.target_win_rate is None or not self.versus_on or (self.directory / "start.npz").is_file():
            return
        self._write_npz(self.directory / "start.npz", self._checkpoint_arrays(plan_before(self.plan, self.next_round)))
        self.rules.start_round = number

    def _versus_job(self, stop):
        """What the head-to-head of the best weights needs (lock held), or None when none is due: the rule is set, this coordinator can
        run it, the run goes on, and the best round has not been compared yet. Also not the run's own start (the first round, when
        start.npz was saved from it), which would only meet itself."""
        rules = self.rules
        if (self.settings.target_win_rate is None or not self.versus_on or self.superseded or self.stopped or self.finished or not self.theta
                or stop.is_set() or rules.best_round is None or rules.best_round == rules.win_round or rules.best_round == rules.start_round):
            return None
        prompts_file = self.directory / "eval_prompts.jsonl"
        if not (self.directory / "start.npz").is_file() or not (self.directory / "best.npz").is_file() or not prompts_file.is_file():
            return None
        prompts = [json.loads(line) for line in prompts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        taught = self._lesson_prompts()  # never judge on a conversation the lessons teach
        prompts = [p for p in prompts if json.dumps(p["messages"], sort_keys=True) not in taught]
        if not prompts:
            return None
        return {"round": rules.best_round, "new": str(self.directory / "best.npz"), "old": str(self.directory / "start.npz"), "prompts": prompts}

    def _lesson_prompts(self):
        """The conversations lessons.jsonl answers, as strings to compare the head-to-head's prompts with."""
        path = self.directory / "lessons.jsonl"
        if not path.is_file():
            return set()
        return {json.dumps(json.loads(line)["messages"], sort_keys=True)
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

    def _begin_judging(self, job, number):
        """The held phase is now the head-to-head (lock held): the phase and the journal say so, with where the prompts came from."""
        self.judging = True
        sources = [p.get("from") for p in job["prompts"]]
        said = ", ".join(f"{sources.count(kind)} {label}" for kind, label in (("validation", "from the validation split"), ("training", "from the training split"),
                                                                              ("held_out", "held-out conversations")) if kind in sources)
        logging.getLogger(__name__).info("head-to-head of round %d: %d prompts (%s)", job["round"], len(sources), said)
        self._event("judging", "", number, f"Round {number}: the teacher ({self.settings.teacher}) is comparing round {job['round']}'s weights with "
                                           f"the starting weights on {len(sources)} prompts ({said}). No round opens until it is done "
                                           f"(at most {self.settings.versus_seconds:,.0f} seconds).")

    def _start_versus(self, current):
        """After a round that did not start a polish (called with the lock held, the round saved): when it made a new best, compare it with
        the starting weights on a thread of its own, and open no round until that is done."""
        stop = threading.Event()
        job = self._versus_job(stop)
        if job is None:
            return
        self._polish_stop = stop
        self.polishing = True
        self._begin_judging(job, current.number)
        try:
            threading.Thread(target=self._versus_run, args=(job, stop), daemon=True, name="diloco-versus").start()
        except Exception as exc:  # no thread to be had: the gate must not stay shut
            self.polishing = self.judging = False
            self.rules.win_round = job["round"]
            drules.record_win(self.rules, self.settings, None)
            self._event("win_rate", "", current.number, f"Round {current.number}: the head-to-head with the starting weights could not start ({exc}).")

    def _versus_run(self, job, stop):
        """The head-to-head thread: ask self.versus for the share of replies where the best weights beat the starting ones, then hand the
        outcome to _versus_done under the lock. Nothing here touches the run's state; no round is open to change it."""
        outcome, halt = {"rate": None, "error": ""}, threading.Event()  # halt: told to the comparison when the run stops or the time is up

        def compare():
            try:
                outcome["rate"] = self.versus(self.teacher_settings, job["new"], job["old"], job["prompts"], self.cfg.get("device", "auto"), halt)
            except Exception as exc:  # the teacher is optional: a failed comparison is skipped and the run goes on
                outcome["error"] = (str(exc) or type(exc).__name__)[:300]

        try:
            worker = threading.Thread(target=compare, daemon=True, name="diloco-versus-call")
            worker.start()
            end = time.monotonic() + self.settings.versus_seconds  # real time: the coordinator's clock is injectable
            while worker.is_alive() and time.monotonic() < end and not stop.is_set():
                worker.join(.05)
            timed_out = worker.is_alive() and not stop.is_set()
            if timed_out:
                outcome["rate"], outcome["error"] = None, f"it took longer than the {self.settings.versus_seconds:g} seconds allowed"
        except Exception as exc:  # no thread to be had: the gate must still open
            outcome["rate"], outcome["error"] = None, "the comparison could not start: " + (str(exc) or type(exc).__name__)[:300]
        finally:
            halt.set()  # always: a call still running is left behind and ends at its next check of halt (its models stay loaded until then)
        with self.cond:
            try:
                self._versus_done(job, outcome, stop)
            finally:
                self.polishing = self.judging = False
                self._advance()
                self.cond.notify_all()

    def _versus_done(self, job, outcome, stop):
        """Take the comparison into the run (lock held): the win rate joins the rules' state, and the run ends when it reached the target.
        A stopped, released or superseded run takes nothing."""
        if stop.is_set() or self.superseded or self.finished or not self.theta:
            self._remember()
            return
        rules, number, rate = self.rules, self.history[-1]["round"], outcome["rate"]
        drules.accrue(rules, self.clock(), self._counting())  # the time the teacher took counts toward max_seconds
        rules.win_round = job["round"]  # tried: the next comparison is for the next best, whether or not this one gave a number
        drules.record_win(rules, self.settings, None if outcome["error"] else rate)
        if outcome["error"] or rate is None or not math.isfinite(rate):
            self._event("win_rate", "", number, f"Round {number}: the head-to-head of round {job['round']}'s weights with the starting weights "
                                                f"was skipped ({outcome['error'] or 'the teacher judged no replies'}).")
        else:
            self._event("win_rate", "", number, f"Round {number}: round {job['round']}'s weights win {rate:.0%} of head-to-head replies against "
                                                f"the starting weights (target {self.settings.target_win_rate:.0%}, reached "
                                                f"{rules.win_hits} of {drules.WIN_READINGS} times in a row).")
        found = drules.stop_reason(rules, self.settings)  # the target, or max_seconds used up by the teacher
        if found:
            rules.stop_rule, rules.stop_text = found
            self.finished = True
        self._persist(self.history[-1]["members"], self.deal)
        self._write_record(self.history[-1])
        if self.finished:
            self._event("finished", "", number, rules.stop_text or f"The run finished after {self.next_round} rounds.")
        self._remember()

    def _rewrite_metric(self, row):
        """metrics.jsonl with round `row`'s line carrying its polished val_loss (and the one it replaced)."""
        path = self.directory / "metrics.jsonl"
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("round") == row["round"]:
                entry.update({"val_loss": row["val_loss"], "val_loss_before_polish": row["val_loss_before_polish"]})
            lines.append(json.dumps(entry))
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        temporary.replace(path)

    def _add_lessons(self, number, rows):
        """Keep the teacher's rewrites as one lesson: a new shard (the ids after the data sources' and every lesson's, also those a crash
        left in lessons.jsonl) that the next round deals with the others. Returns how many rows it holds; 0 when they make no training window."""
        rows = [row for row in rows if row.get("messages") and row.get("chosen")]
        windows = self._lesson_windows(rows) if rows else []
        if not len(windows):
            return 0
        path = self.directory / "lessons.jsonl"
        used = [entry["shard"] for entry in self.lessons]
        if path.is_file():
            used += [json.loads(line).get("shard", -1) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        shard = max([self.shard_count - 1, *used]) + 1
        with path.open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps({"shard": shard, "round": number, "messages": row["messages"], "chosen": row["chosen"],
                                         "rejected": row.get("rejected", "")}, ensure_ascii=False) + "\n")
        self.lesson_windows[shard] = windows
        self.lessons.append({"shard": shard, "round": number, "rows": len(rows), "status": "live"})
        return len(rows)

    # -- durability
    def _touch(self, node=""):
        self.touched = time.monotonic()
        if node:
            self.beats[node] = self.clock()

    def heard_ago(self, node):
        """Seconds since `node` last called this coordinator (it joined, reported, waited for a round or sent a delta); since the
        coordinator was made when it has not called yet."""
        return self.clock() - self.beats.get(node, self.started)

    def _remember(self):
        """Have the Training tab's record of this run (studio_distributed) kept with its account as it stands now. It takes the snapshot on
        its own thread: this one holds the lock, and a snapshot advances the clock."""
        import studio_distributed
        if studio_distributed.keeps(self.run):
            threading.Thread(target=studio_distributed.remember, args=(self.run, self), daemon=True).start()

    def _save(self, row, closed):
        """The round's files, in the order that keeps a crash recoverable: the state, last.npz, then diloco.json that names it."""
        self._persist(closed.members, closed.deal)
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({k: row[k] for k in ("step", "loss", "val_loss", "tokens_per_second", "lr", "grad_norm")}
                                    | {"round": row["round"], "timeline": self._timeline(closed)}) + "\n")
        self._event("checkpoint", "", row["round"], f"Round {row['round']} checkpoint written.")
        self._write_record(row)
        self._remember()

    def _write_record(self, row):
        """run.json as it stands after round `row` (a polished round's val_loss is already the polished loss)."""
        from studio_core import atomic_json, now
        self.record.update({"step": plan_before(self.plan, self.next_round),
                            "val_loss": row["val_loss"],
                            "status": "completed" if self.finished else "training",
                            "best_round": self.rules.best_round, "best_val_loss": self.rules.best_loss, "win_rate": self.rules.win_rate})
        if self.rules.stop_rule:
            self.record.update({"stop_rule": self.rules.stop_rule, "stop_reason": self.rules.stop_text})
        if row["loss"] is not None:
            self.record["loss"] = row["loss"]
        if self.finished:
            self.record["finishedAt"] = now()
        atomic_json(self.directory / "run.json", self.record)

    def _persist(self, members, deal):
        """Write the state for `next_round` rounds closed: the state archive, last.npz, then diloco.json that names it (a crash between
        them leaves the round before). Returns the step. `members` and `deal` are the last closed round's."""
        from studio_core import atomic_json
        old = self.state_file
        self.state_file = f"diloco_state_{self.next_round}.npz"
        arrays = {STATE_PREFIXES[0] + k: v for k, v in self.theta.items()}
        arrays.update({STATE_PREFIXES[1] + k: v for k, v in self.momentum.items()})
        self._write_npz(self.directory / self.state_file, arrays)
        step = plan_before(self.plan, self.next_round)
        self._write_npz(self.directory / "last.npz", self._checkpoint_arrays(step))
        self._keep_checkpoint()
        atomic_json(self.directory / "diloco.json", {
            "format": FORMAT, "run": self.run, "term": self.term, "round": self.next_round, "generation": self.generation, "state_file": self.state_file,
            "momentum_file": self.state_file, "digest": self.digest, "seed": self.cfg["seed"], "settings": self.settings.__dict__, "plan": self.plan,
            "members": members, "shard_deal": deal, "step_time": self.step_time, "history": self.history,
            "finished": self.finished, "split_strategy": self.split["strategy"], "split_hash": self.split_hash,
            "rules": self.rules.to_json(), "layout": [count for _, count in self.spans], "paused": self.paused and not self.finished,
            "lessons": self.lessons})
        if old and old != self.state_file:
            (self.directory / old).unlink(missing_ok=True)
        return step

    def _checkpoint_arrays(self, step):
        """The arrays of a checkpoint (.npz) holding theta at `step`."""
        metadata = {"architecture": "groundwork_numpy_decoder", "format_version": 1, "config": self.cfg,
                    "tokenizer": self.tokenizer.serialize(), "step": step, "best": min([r["val_loss"] for r in self.history if r["val_loss"] is not None], default=None),
                    "optimizer_steps": step, "history": [], "data_hash": self.digest, "sampler_state": list(random.Random(self.cfg["seed"]).getstate()),
                    "model_rng": np.random.default_rng(self.cfg["seed"]).bit_generator.state, "device": "cpu",
                    "split_strategy": self.split["strategy"], "split_hash": self.split_hash}
        checkpoint = {"metadata": np.array(json.dumps(metadata))}
        checkpoint.update({"weight::" + k: v for k, v in self.theta.items()})
        return checkpoint

    def _keep_checkpoint(self):
        """With keep_checkpoints = N, also keep last.npz of the latest N rounds as round_<n>.npz, and delete the older ones."""
        keep = self.settings.keep_checkpoints
        if not keep:
            return
        shutil.copyfile(self.directory / "last.npz", self.directory / f"round_{self.next_round}.npz")
        stale = self.next_round - keep
        for path in self.directory.glob("round_*.npz"):
            if path.stem[6:].isdigit() and int(path.stem[6:]) <= stale:
                path.unlink(missing_ok=True)

    @staticmethod
    def _write_npz(path, arrays):
        from studio_core import replace_file
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        replace_file(temporary, path)

    def release(self):
        """Drop the big arrays once the run is over; the files on disk are the result."""
        with self.cond:
            self._polish_stop.set()
            self.theta, self.momentum, self._params, self._state, self._update = {}, {}, (None, b""), (None, b""), None


def _evaluator(model, val, size, limit=None):
    """evaluate(theta) -> cross-entropy over an evenly spread subset of the validation windows (None without any)."""
    import studio_model as sm
    limit = limit or sm.EVAL_WINDOWS

    def evaluate(theta):
        if not len(val):
            return None
        model.load_weights(theta)
        total, count = 0., 0
        chosen = np.linspace(0, len(val) - 1, min(len(val), limit), dtype=int) if len(val) > limit else np.arange(len(val))
        for start in range(0, len(chosen), size):
            x, y = sm.batch_arrays([val[int(i)] for i in chosen[start:start + size]])
            targets = int((y != -100).sum())
            total += model.loss(x, y) * targets
            count += targets
        return total / count if count else None
    return evaluate


# ---- coordinators of this process -------------------------------------------------------------------------------------

COORDINATORS = {}  # run id -> Coordinator: what studio_api.routers.collective serves
WATCHERS = []  # callables taking a Coordinator whose live view changed (studio_remote.Cluster turns them into `run` events)
IDLE_RELEASE = 3600.  # seconds untouched after which a coordinator is forgotten when another starts


def is_coordinating(folder):
    """Whether a live coordinator of this process (not finished or stopped) writes into `folder`: its last.tmp and best.tmp are in use."""
    target = Path(folder).resolve()
    return any(not (c.finished or c.stopped) and c.directory and Path(c.directory).resolve() == target for c in list(COORDINATORS.values()))


_LOCK = threading.Lock()


def prompt_key(prompt):
    """What makes two prompts the same question."""
    return json.dumps(prompt["messages"], sort_keys=True)


def unique_prompts(conversations):
    """The prompts the conversations make (those ending in an assistant turn), the same question counted once: what a teacher can be asked."""
    from studio_rlhf import prompts_from
    unique = {}
    for prompt in prompts_from(conversations):
        unique.setdefault(prompt_key(prompt), prompt)
    return list(unique.values())


def prompts_needed(polish, versus):
    """How many different prompts the teacher's rules need: one to polish on; MIN_EVAL_PROMPTS held out for the win-rate stop, and with
    both on that many again for the polish (the head-to-head never takes more than half)."""
    return (2 * MIN_EVAL_PROMPTS if polish else MIN_EVAL_PROMPTS) if versus else 1


def split_prompts(conversations, settings, seed):
    """(polish prompts, head-to-head prompts) from the teacher's conversations: the different prompts they make, shuffled by the run's
    seed, then dealt so the two never share one. With both rules on the head-to-head gets up to EVAL_PROMPTS of them but never more than
    half; with one on, that rule gets them all (the head-to-head up to EVAL_PROMPTS). ValueError, in a sentence, when there are fewer
    than the rules need (prompts_needed)."""
    prompts = unique_prompts(conversations)
    random.Random(seed).shuffle(prompts)
    polish, versus = bool(settings.polish_every), settings.target_win_rate is not None
    need = prompts_needed(polish, versus)
    if len(prompts) < need:
        what = (f"the win-rate target compares on at least {MIN_EVAL_PROMPTS} held-out prompts" + (f" and the polish needs {MIN_EVAL_PROMPTS} more" if polish else "")
                if versus else "the polish needs a prompt")
        raise ValueError(f"The chosen data makes {len(prompts)} different prompt{'' if len(prompts) == 1 else 's'}, but {what}: "
                         f"add conversations (at least {need} different questions), or turn the teacher off.")
    if polish and versus:
        held = min(EVAL_PROMPTS, len(prompts) // 2)
        return prompts[held:], prompts[:held]
    return (prompts, []) if polish else ([], prompts[:EVAL_PROMPTS])


def teacher_settings(settings, root=None):
    """The Ollama teacher a run polishes with: the workspace's RLHF settings (address, rubric, judge model) with the run's own teacher model."""
    from studio_rlhf import SETTINGS_DEFAULTS, load_settings
    return {**(load_settings(root) if root else json.loads(json.dumps(SETTINGS_DEFAULTS))), "teacher_model": settings.teacher}


def _teacher(found, stop=None):
    """The Ollama teacher for `found`; with `stop`, it asks nothing more once that is set (a request already sent ends by its timeout)."""
    from studio_rlhf import OllamaTeacher
    return OllamaTeacher(found["url"], found["teacher_model"], rubric=found.get("rubric"), judge_model=found.get("judge_model") or None, stop=stop)


PREFLIGHT_TIMEOUT = 10  # seconds the start-up check waits for Ollama


def check_teacher(found):
    """RuntimeError, saying why, when the Ollama teacher cannot be reached or does not have the model the run polishes with."""
    teacher = _teacher(found)
    teacher.timeout = PREFLIGHT_TIMEOUT
    names = teacher.models()
    wanted = found["teacher_model"]
    if wanted not in names and f"{wanted}:latest" not in names:
        raise RuntimeError(f"Ollama at {teacher.url} has no model called {wanted}. Pull it first"
                           + (f", or pick one of: {', '.join(names[:5])}." if names else "."))


def default_polish(directory, found, checkpoint, prompts, device, stop, emit):
    """One preference round of the teacher from `checkpoint`: sample, grade, train, and
    keep the result only if it wins the head-to-head gate. Returns {"checkpoint", "win_rate", "lessons": [{"messages", "chosen", "rejected"}]};
    "checkpoint" is the given one when the gate did not pass. Raises when the teacher cannot be reached."""
    from studio_rlhf import PreferenceStore, auto_rlhf
    teacher = _teacher(found, stop)
    teacher.models()  # RuntimeError when Ollama is not reachable
    Path(directory).mkdir(parents=True, exist_ok=True)
    store = PreferenceStore(directory)
    before = len(store)
    result = auto_rlhf(str(checkpoint), prompts, teacher, store, directory, {"rounds": 1}, {"device": device}, emit, stop, None, teacher.judge_teacher())
    if result.get("aborted"):
        raise RuntimeError(result["aborted"])
    rows = [row for row in store.all()[before:] if row.get("kind") == "rewrite"]
    return {"checkpoint": result["checkpoint"], "win_rate": result["win_rate"], "lessons": [dict(row) for row in rows]}


def default_versus(settings, new, old, prompts, device, stop):
    """The share of head-to-head replies where checkpoint `new` beats checkpoint `old`, as graded by the teacher (a tie counts half);
    None when no reply was judged. Raises when the teacher cannot be reached."""
    from studio_rlhf import compare_checkpoints
    teacher = _teacher(settings, stop)
    teacher.models()  # RuntimeError when Ollama is not reachable
    return compare_checkpoints(new, old, prompts, teacher.judge_teacher(), device=device, stop=stop)["win_rate"]


def watch(watcher):
    """Call `watcher(coordinator)` whenever a coordinator's live view changes; returns the call that stops it."""
    WATCHERS.append(watcher)
    return lambda: watcher in WATCHERS and WATCHERS.remove(watcher)


def register(run, coordinator):
    with _LOCK:
        for key, other in list(COORDINATORS.items()):
            if other.finished or time.monotonic() - other.touched > IDLE_RELEASE:
                import studio_distributed
                studio_distributed.archive(key, other)  # the Training tab keeps what it showed
                other.release()
                del COORDINATORS[key]
        COORDINATORS[run] = coordinator


def prepare(job, services, directory, port, client_for=None):
    """App side, before the owner's job starts its worker: build the coordinator (tokenizing the data once), register it for the
    routes, and tell the worker where it is. A folder with diloco.json continues that run from its last completed round, which is how a
    run resumes after the app restarted; the machines chosen are invited again. `client_for` reaches a peer (default PeerClient)."""
    import studio_distributed
    p = job.params
    say = studio_distributed.preparing(job.id)  # what the run page shows until the first round: written here, read by the request path
    try:
        return _prepare(job, services, directory, port, client_for, p, say)
    finally:
        studio_distributed.prepared(job.id)


def _prepare(job, services, directory, port, client_for, p, say):
    import studio_distributed
    say("reading")
    config = dict(p["config"])
    settings = Settings.from_params(p, steps=config.get("steps"))
    teacher = None
    policy = None
    prepared = None
    if p.get("stages"):  # next_dataset goes to the workspace's unseen text sets, worst first, with replay (studio_stages)
        import studio_stages
        policy, examples = studio_stages.Policy.for_job(p, services)
    else:
        if p.get("datasets") and not p.get("examples"):
            getter = getattr(services, "get_prepared_shards", None)
            if getter is not None:
                prepared = getter(p.get("datasets"), config, shard_dir=tokenized_root(directory))
        examples = None if prepared is not None else (p["examples"] if "examples" in p else services.examples(p))
    conversations = services.conversations(p) if p.get("conversations") else None  # the teacher's prompts: kept apart from what the members train on
    if examples is not None and not examples and p.get("reviewed_only"):
        raise ValueError("Reviewed examples only is on, but no example in the chosen text is marked Good and free to train on.")
    try:
        if settings.polish_every or settings.target_win_rate is not None:
            teacher = teacher_settings(settings, getattr(services, "root", None))
            say("teacher")
            if not p.get("resuming"):  # a new run fails now, not at the first use of the teacher, when it cannot be used; a resumed one meets it then
                try:
                    check_teacher(teacher)
                except RuntimeError as exc:
                    raise ValueError(f"The run {'polishes' if settings.polish_every else 'judges'} with the teacher {settings.teacher}, but it cannot be used: {exc}") from None
        coordinator = Coordinator.build(job.id, directory, config, examples, settings, later=p.get("later_examples") or (),
                                       start_from=p.get("restart_from"), conversations=conversations, progress=say, prepared=prepared)
    except ValueError as exc:
        if p.get("resuming"):
            raise ValueError(f"Ended: this machine restarted at round {p['resuming']['round']}, and the run could not be resumed: {exc}") from None
        raise
    run = studio_distributed.run_of(job)
    if teacher is not None:
        coordinator.teacher_settings = teacher
    coordinator.stages = policy
    coordinator.prefer = (run.handoff,) if run.handoff else ()  # the node chosen to coordinate holds the run from the first round on
    register(job.id, coordinator)
    run.folder = str(directory)
    studio_distributed.persist()
    studio_distributed.resumed(job)
    p.update(run=job.id, coordinator_url=f"http://127.0.0.1:{port}", settings=dict(settings.__dict__))
    say("inviting")
    studio_distributed.invite(job, **({"client_for": client_for} if client_for else {}))  # the peers the person chose: a member job on each, which joins at the coordinator
    return coordinator


# ---- the member -------------------------------------------------------------------------------------------------------

class Member:
    """One node's side of a run. It is the `distributed=` hook of studio_model.train: before the first step it loads theta of
    its round, after H steps (or at the round's deadline) it sends delta and waits for the next round, whose theta it loads.
    With `overlap` it does not wait: the exchange (delta up, next theta down) runs through `spawn(fn)` (a Background thread; tests
    inject their own) while the next round's steps run, and is joined at the end of that round, where its theta is applied."""

    def __init__(self, client, run, node, epoch, shard_dir, stop, clock=time.monotonic, emit=None, wait=LONG_POLL, patience=300., sleep=time.sleep,
                 progress=True, overlap=False, spawn=Background, replica_dir=None, locate=None, pause=None, memory_share=None,
                 cache_dir=None, streams=None, receiver_grants=False, fit=None):
        if streams is not None and hasattr(client, "streams"):
            client.streams = max(1, int(streams))  # parallel streams per transfer (studio_peer.PeerClient); None keeps the client's default
        self.receiver_grants = bool(receiver_grants)
        if hasattr(client, "receiver_grants"):
            client.receiver_grants = self.receiver_grants
            client.receiver_downloads = self.receiver_grants
        self.down_bps = 0.  # how fast the latest weights came down, bytes per second; told to the coordinator with the change
        self.pause, self.memory_share = pause, memory_share  # this node's queue plan: its pause event and its share of memory
        self.fit = fit  # {batch_size, accumulation, offload_optimizer}: what the coordinator sized this member to (studio_fit.fit_members); None: as the design says
        self.client, self.run, self.node, self.epoch, self.shard_dir, self.stop = client, run, node, epoch, Path(shard_dir), stop
        self.cache_dir = Path(cache_dir) if cache_dir else None  # shards by content hash, kept across runs (None: per-run copies only)
        self.replica_dir = Path(replica_dir) if replica_dir else None  # where a replica keeps the latest round (None: it keeps none)
        self.prefetch, self.prefetched, self.shard_locks, self.replica_lock = None, False, {}, threading.Lock()
        self.shard_demands = {}  # required users can promote a shard already being prefetched
        self.replica_epoch, self.replica_guard = 0, threading.RLock()  # a write to replica_dir is void once the epoch it began under is over
        self.clock, self.emit, self.wait, self.patience, self.sleep = clock, emit or (lambda event: None), wait, patience, sleep
        self.relocate = RELOCATE_SECONDS  # how long a coordinator that no longer has the run is given to be replaced before the member leaves
        self.base = f"/collective/{run}"
        self.train, self.val, self.sampler = Windows(), Windows(), random.Random(0)
        self.started, self.finished, self.fenced = False, False, False
        self.round = 0  # the round in hand; read (round_steps, in a progress report) from the join on, before the first round is entered
        self.warm, self.busy, self.timed = False, 0., 0  # stepped(): the first step is warm-up; the time and count of the timed steps of the round
        self.term, self.superseded = 0, 0  # the coordinator's term (0: not learnt yet); superseded: a higher term seen, so nothing more goes to this coordinator
        self.loss = None
        self.theta, self.theta_round, self.residual = {}, None, {}  # theta and the round it belongs to (an update applies to the round before only); what the int8 encoding of the last change lost
        self.train_deadline = None  # a time-boxed round ends when the clock gets here
        self.shards, self.downloaded = {}, 0  # downloaded: bytes of shards fetched so far
        # what it tells the coordinator (POST /progress): the phase, its step in the round and the latest loss and speed
        self.progress, self.phase, self.report_round, self.round_step, self.reported_step = progress, "waiting", None, 0, 0
        self.reported_phase = None
        self.live = {"loss": None, "tokens_per_second": None}
        self.wake, self.halt, self.reporter = threading.Event(), threading.Event(), None
        self.overlap, self.spawn, self.pending, self.entered, self.anchor = overlap, spawn, None, None, None
        self.pending_round, self.exchange_waiting = None, threading.Event()
        # locate(run) -> (node, term, client) of the coordinator now in charge, or None (studio_failover); where: the (node, term) this member
        # last rebound to; ended: this attempt of train() is over but the job is not (the member may rebind); holding: (round, rank) it keeps
        self.locate, self.where, self.ended, self.holding, self.moved = locate, None, threading.Event(), None, False

    # -- the train() hook
    @property
    def plan(self):
        """The run's plan (plan_h) as the newest round this member took says it (the join's, before any): a run extended while it goes
        reaches the member in the rounds it takes."""
        if self.entered is not None and self.entered.plan:
            return self.entered.plan
        return self.info.get("plan") or [{"round": 0, "q": 0., "h": self.info["inner_steps"], "rounds": self.info["rounds"]}]

    @property
    def rounds(self):
        """The run's rounds as the coordinator last said."""
        return plan_rounds(self.plan)

    @property
    def total_steps(self):
        """The length the learning-rate schedule has in the round in hand (the run's, until an extension reaches it)."""
        return plan_frame(self.plan, self.round)

    @property
    def run_steps(self):
        """The inner steps of the whole run as it stands."""
        return plan_total(self.plan)

    @property
    def round_steps(self):
        """The inner steps this member was dealt for the round in hand."""
        return (self.entered.steps if self.entered is not None else 0) or plan_h(self.plan, self.round)

    def rate_step(self, number):
        """The schedule position of step `number`: each round spans H positions of the schedule, and the member's steps (however many it
        was dealt) are spread over them, so every member is at the same point of the schedule when the round ends."""
        plan = self.plan
        offset, h = number - self.first_step, plan_h(plan, self.round)
        return plan_position(plan, self.round, min(h, offset if self.round_steps == h else offset * h / self.round_steps))

    @property
    def rate_scale(self):
        """The factor the run's pivots (halve_lr) put on the learning rate, as the round in hand was opened with."""
        return self.entered.lr_scale if self.entered is not None else 1.

    def stepped(self, seconds):
        """train() tells how long the step just done took (the step itself: not time paused, not the sync). The first step of an
        attempt is warm-up (kernels, allocations) and is not timed."""
        if not self.warm:
            self.warm = True
            return
        self.busy, self.timed = self.busy + seconds, self.timed + 1

    def _work_seconds(self, steps):
        """Seconds of work for `steps` steps, at the pace of the timed steps of the round (without pauses and warm-up); 0 (no
        measurement: the coordinator ignores it) when none was timed."""
        return self.busy / self.timed * steps if self.timed else 0.

    def due(self, step, trained_tokens):
        self.round_step = step - self.first_step  # the reporter thread reads it; nothing here waits on the network
        if self.round_step - self.reported_step >= PROGRESS_STEPS:
            self.wake.set()
        if step <= self.first_step:
            return False
        if self.train_deadline is not None:  # time-boxed: by the clock, within a bound on the steps so the loop never runs out of them
            return self.clock() >= min(self.train_deadline, self.deadline) or step - self.first_step >= plan_h(self.plan, self.round) * MAX_SHARE
        return step - self.first_step >= self.round_steps or self.clock() >= self.deadline

    def sync(self, step, trained_tokens, model):
        if self.overlap:
            return self._sync_overlapped(step, trained_tokens, model)
        if self.started:
            self._send(step, trained_tokens, model)
            if not (self.finished or self.fenced):
                self._enter(self.round + 1)
        else:
            self.started = True
        if self.finished or self.fenced:
            self._phase("waiting")
            self._end_attempt()
            return
        model.load_weights(self.theta)
        self._begin(step, trained_tokens)

    def _begin(self, step, trained_tokens):
        """Start the steps of self.round, from the weights the model now holds."""
        self.first_step, self.first_tokens, self.began, self.deadline = step, trained_tokens, self.clock(), self.clock() + self.deadline_s
        train_s = self.entered.train_s if self.entered is not None else 0.
        self.train_deadline = self.began + train_s if train_s > 0 else None
        self.sampler.seed(sampler_seed(self.info["seed"], self.rank, self.round))
        self.round_step, self.reported_step = 0, 0
        self.busy, self.timed = 0., 0
        if self.round_steps > 2:
            self.warm = False  # the first step of every round is not timed (cold caches after the weights were replaced)
        self.report_round = self.round
        self._phase("training")

    def _sync_overlapped(self, step, trained_tokens, model):
        """The end of a round's steps without waiting for the network. The exchange begun at the end of the last round (its delta up,
        this round's theta down) is joined here, which takes no time when it was faster than the steps. Its theta then replaces
        the one the round started from, and the work done since is kept: w' = theta + (w - w_start) = theta - delta. This round's
        delta goes out on a new exchange, and the next round's steps start from w' at once."""
        if not self.started:
            self.started = True
            self.anchor = self.theta
            model.load_weights(self.theta)
            self._begin(step, trained_tokens)
            return
        entered = self._join() if self.pending is not None else self.entered
        if entered is None or self.finished or self.fenced:
            self._phase("waiting")
            self._end_attempt()
            return
        self._apply(entered)
        if entered.round != self.round:  # this round closed without the member: its work counts for nothing, so start over from the new weights
            self.emit({"type": "info", "text": f"Round {self.round} closed without this member; continuing from round {entered.round}."})
            self.round, self.anchor = entered.round, entered.theta
            model.load_weights(entered.theta)
            self._begin(step, trained_tokens)
            return
        number, weights = self.round, {name: value.astype(np.float32) for name, value in model.state_arrays().items()}
        delta = {name: self.anchor[name] - weights[name] for name in self.anchor}
        params = {"term": self.term, "node": self.node, "generation": entered.generation, "epoch": self.epoch, "tokens": trained_tokens - self.first_tokens,
                  "steps": step - self.first_step}
        if self.timed:  # none timed (a round of a step or two): no seconds, and the coordinator keeps the member's speed as it was
            params["seconds"] = self._work_seconds(step - self.first_step)
        if self.loss is not None:
            params["loss"] = self.loss
        self.pending_round = number
        self.pending = self.spawn(lambda: self._exchange(number, delta, params))
        if number + 1 >= self.rounds:  # the last round: nothing to train after it, so wait for the run's end
            self._phase("waiting")
            self.entered = self._join()
            if self.entered is None or self.finished or self.fenced:
                self._end_attempt()
                return
        renewed = {name: entered.theta[name] - delta[name] for name in delta}
        model.load_weights(renewed)
        self.round, self.anchor = number + 1, renewed
        self._begin(step, trained_tokens)

    def _end_attempt(self):
        """Have train() end: the run finished or the member was fenced out. When it may rebind (it has `locate`) and was fenced, only
        this attempt ends, and the job's own stop stays unset."""
        if self.locate is not None and (self.superseded or self.moved) and not self.finished:
            self.ended.set()
        else:
            self.stop.set()

    def _join(self):
        """The result of the exchange in flight (an Entered, or None when the run finished, the member was fenced out or stop was set)."""
        pending, self.pending = self.pending, None
        number, self.pending_round = self.pending_round, None
        began = self.clock()
        self.exchange_waiting.set()
        try:
            return pending.result()
        finally:
            self.exchange_waiting.clear()
            self._network_event("exchange_join", began, exchange_round=number, exposed_wait_s=max(0., self.clock() - began))

    def _exchange(self, number, delta, params):
        """On the background thread: send round `number`'s delta, then take round number + 1."""
        began = self.clock()
        body = self._encode(delta)
        self._network_event("encode", began, exchange_round=number, bytes=len(body))
        self._post(number, body, params, overlapped=True)
        if self.fenced:
            return None
        return self._take(number + 1, quiet=True)

    # -- telling the coordinator where it is
    def _phase(self, phase):
        """Change the phase it reports and have the reporter say so now."""
        self.phase = phase
        self.wake.set()

    def report(self):
        """Send one progress report (no weights). A report that fails, for any reason, is dropped: it never raises, and never stops
        the training."""
        if not self.progress or self.report_round is None or self.superseded or not self.term:  # no term: not joined to this coordinator yet
            return False
        step, phase = self.round_step, self.phase
        body = {"node": self.node, "generation": getattr(self, "generation", self.info["generation"]), "epoch": self.epoch,
                "round": self.report_round, "phase": phase, "step": step, "loss": self.live["loss"],
                "tokens_per_second": self.live["tokens_per_second"]}
        try:
            answer = self._seen(self.client.post_json(f"{self.base}/progress", body, self._q(), timeout=PROGRESS_TIMEOUT))
            if isinstance(answer, dict):  # the plan: a member training a round learns of an extension now, not at its next view
                self._replica_plan(answer)
        except Exception as exc:  # noqa: BLE001 - a stale 409, an unreachable coordinator, a timeout: the report is only a courtesy
            if hasattr(exc, "detail"):
                self._refused(exc)
            return False
        self.reported_step, self.reported_phase = step, phase
        try:  # the app on this node publishes it as this member's own fact (studio_facts, "progress" and "shards")
            self.emit({"type": "diloco_member", "run": self.run, "node": self.node, "term": self.term, "round": self.report_round,
                       "phase": phase, "step_in_round": step, "round_steps": self.round_steps, "loss": body["loss"],
                       "tokens_per_second": body["tokens_per_second"], "shard_ids": sorted(list(self.shards)), "downloaded_bytes": self.downloaded})
        except Exception:  # noqa: BLE001 - a courtesy, like the report itself
            pass
        return True

    def _report_loop(self):
        last = 0.
        while not self.halt.is_set():
            self.wake.wait(PROGRESS_SECONDS)
            self.wake.clear()
            if self.phase == self.reported_phase:  # a change of phase is said at once; the gap only spaces step reports
                self.halt.wait(max(0., PROGRESS_GAP - (time.monotonic() - last)))
            if self.halt.is_set():
                return
            self.report()
            last = time.monotonic()

    # -- terms
    def follow(self, term):
        """A term above this member's own has been seen (in an answer, or by whoever watches the run's claims, #361): the
        coordinator it talks to is an old one. Nothing more is sent to it, and the member stops (it rejoins the new coordinator)."""
        if term > max(self.term, self.superseded):
            self.superseded, self.fenced = term, True
            self.emit({"type": "info", "text": f"Term {term} exists (this member is at term {self.term}): no longer sending to the coordinator it joined."})
            self.wake.set()

    def _seen(self, answer):
        if isinstance(answer, dict) and isinstance(answer.get("term"), int) and self.term:
            self.follow(answer["term"])
        return answer

    def _refused(self, exc):
        """A 409 for a term that is not the coordinator's: when it names a higher one, follow it."""
        if exc.status != 409 or not exc.detail.startswith(TERM_STALE):
            return
        self.moved = True  # the coordinator it talks to is not the run's coordinator any more
        found = re.search(r"at term (\d+)", exc.detail)
        if found:
            self.follow(int(found.group(1)))

    def _q(self, params=None):
        return {**(params or {}), "term": self.term}

    # -- talking to the coordinator
    def _call(self, call):
        """Run a client call, waiting for the coordinator to come back if it cannot be reached (a restart) for up to `patience`. An
        empty gateway answer counts as not reached: it is `tailscale serve` saying the app behind it is down, as while it restarts.
        Once a higher term has been seen it sends nothing: the call is a 409 without leaving this process."""
        from studio_cluster import gateway_down
        from studio_peer import Cancelled, PeerError
        end = self.clock() + self.patience
        while True:
            if self.superseded:
                raise PeerError(f"{TERM_STALE} {self.term}: the run is at term {self.superseded}.", status=409,
                                detail=f"{TERM_STALE} {self.term}: the run is at term {self.superseded}.")
            try:
                return self._seen(call())
            except Cancelled:
                raise
            except PeerError as exc:
                self._refused(exc)
                if exc.status == 404 and exc.detail.startswith(GONE_RUN):
                    # a coordinator that stepped down is gone from its node, and the run is coordinated elsewhere now; the claim that says where
                    # may still be on its way to this node's store, so it is waited for, but only a moment (`relocate`, a failover poll or two, and
                    # never past this call's own deadline): a coordinator whose app restarted never names a successor
                    found = self._relocated_within(min(end, self.clock() + self.relocate)) if self.locate is not None else None
                    if found is not None:
                        raise Rebound(found) from None
                    raise PeerError(f"The coordinator no longer has run {self.run} (its app restarted), so this member left. "
                                    f"It joins again if it is invited again.", status=404, detail=exc.detail) from None
                unreached = exc.status is None or gateway_down(exc)
                if not unreached or self.clock() >= end or self.stop.is_set():
                    if unreached and not self.stop.is_set():
                        raise PeerError(f"The coordinator of run {self.run} did not answer for {self.patience:.0f} seconds, so this member "
                                        f"left. It joins again if it is invited again.") from None
                    raise
                found = self._relocated()
                if found is not None:
                    raise Rebound(found) from None
                self.sleep(2.)

    def _relocated(self):
        """(node, term, client) of the coordinator now in charge of the run, when it is not the one this member last joined; else None."""
        if self.locate is None:
            return None
        try:
            found = self.locate(self.run)
        except Exception:  # noqa: BLE001 - where the run is now is a question for the store; a failing answer is no answer
            return None
        if not found or (found[0], found[1]) == self.where or found[1] < self.term:
            return None
        return found

    def _relocated_within(self, end):
        """Like _relocated, but waits until `end` (a time on this member's clock) for the claim of the coordinator in charge to reach this
        node's store: a member told of a higher term has seen it before the claim that carries it. A claim that names the coordinator
        this member already talks to (the same node and term) is no move, so a member left behind by a restart waits out `end` and goes."""
        while True:
            found = self._relocated()
            if found is not None or self.clock() >= end or self.stop.is_set():
                return found
            self.sleep(2.)

    def _bound(self, found):
        """_rebind to `found`, and to whichever coordinator is in charge by the time that one is unreachable too, until one holds."""
        while found is not None:
            try:
                self._rebind(found)
                found = None
            except Rebound as bound:
                found = bound.found

    def _rebind(self, found):
        """Leave the coordinator this member was attached to and join the one in charge (a failover or a settled claim, #361): the
        round in flight is dropped, the term is learnt again, and the shards already held are kept (the run's shard cut does not change)."""
        node, term, client = found
        self.emit({"type": "info", "text": f"The run is coordinated on {node} at term {term} now; rejoining there."})
        self.pending = None
        self.client, self.where = client, (node, term)
        if hasattr(client, "receiver_grants"):
            client.receiver_grants = self.receiver_grants
            client.receiver_downloads = self.receiver_grants
        self.term, self.superseded = 0, 0
        self.fenced = self.finished = self.started = self.moved = False
        self.entered = self.anchor = None
        self.ended.clear()
        from studio_peer import PeerError
        try:
            self.join()
        except PeerError as exc:
            if exc.status != 409 or FINISHED not in exc.detail:
                raise
            self.finished = True  # the run ended on the coordinator in charge while this member was still leaving the old one: nothing is left to do
            return
        self._enter(self.info["round"])

    def join(self):
        self.info = self._call(lambda: self.client.post_json(f"{self.base}/join", {"node": self.node, "epoch": self.epoch}, self._q()))
        self.term = self.info["term"]
        from studio_modules.tokenizer.byte import Tokenizer
        self.tokenizer = Tokenizer.restore(self.info["tokenizer"])
        self.report_round = self.info["round"]
        return self.info

    def _enter(self, want):
        """Wait until round `want` (or the current one, when it has moved on) is open for this member, then take its theta and shards."""
        entered = self._take(want)
        if entered is not None:
            self._apply(entered)
            self.round = entered.round

    def _apply(self, entered):
        """Make a taken round the member's own: its theta, place and shards (the sampler's windows swap here, never mid-round)."""
        self.theta, self.generation, self.rank, self.deadline_s, self.entered = entered.theta, entered.generation, entered.rank, entered.deadline_s, entered
        self.theta_round = entered.round
        self.train.set([self.shards[s] for s in entered.shards])

    def _take(self, want, quiet=False):
        """Like _enter, but returns what it took (an Entered) instead of applying it; None when the run finished, the member was fenced
        out or stop was set. `quiet`, for the background exchange, leaves the phase and round the reporter says alone."""
        from studio_peer import PeerError
        asked, noted, noted_polish = want, False, False
        while not self.stop.is_set():
            if not quiet:
                self.report_round = want
                self._phase("waiting")
            try:
                view = self._call(lambda: self.client.get_json(f"{self.base}/rounds/{want}", self._q({"node": self.node, "wait": self.wait}),
                                                               timeout=self.wait + 15))
            except PeerError as exc:
                if exc.status == 409:
                    self.emit({"type": "info", "text": f"Left the run: {exc.detail}"})
                    self.fenced = True
                    return None
                raise
            if view["state"] == "finished":
                self.finished = True
                return None
            self._replica_plan(view)
            if view["state"] == "pending" and view.get("paused"):
                self._replica_paused()
            if view["state"] == "pending" and view.get("paused") and not quiet and not noted:
                noted = True
                self.emit({"type": "info", "text": "The run is paused by its owner. Waiting for it to be resumed."})
            if view["state"] == "pending" and view.get("polishing") and not quiet and not noted_polish:
                noted_polish = True
                self.emit({"type": "info", "text": "The teacher is polishing the weights. The next round opens when it is done."})
            if view["state"] == "closed":
                want = view["current"]
            elif view["state"] == "pending" and not view["member"]:
                self.join()
                want = self.info["round"]
            elif view["state"] == "open":
                if not quiet:
                    self._phase("downloading")
                try:
                    base, base_round = self.theta, self.theta_round  # read once: the training thread replaces them when it applies the round before
                    asking = {"round": want, "node": self.node, **({"since": base_round} if base and base_round is not None else {})}
                    began = self.clock()
                    priority = self.exchange_waiting.is_set if quiet else True
                    body = self._download(f"{self.base}/params", self._q(asking), priority=priority)
                    self.down_bps = len(body) / max(1e-6, self.clock() - began)
                    decoded = self.clock()
                    origin = update_origin(body)
                    if origin is None:
                        theta = decode_arrays(body)
                    elif origin == base_round and base:
                        theta = add_update(base, decode_arrays(body))
                    else:
                        raise ValueError(f"Round {want}'s weights came as an update from round {origin}, but this member holds round {base_round}.")
                    self._network_event("decode_weights", decoded, exchange_round=want, bytes=len(body))
                    replica = bool(view.get("replica") and self.replica_dir is not None)
                    if replica:
                        self._replicate(want, body if origin is None else encode_arrays(theta, "float32"), view["rank"], priority=priority)
                    elif self.replica_dir is not None:
                        self._drop_replica()  # no longer a replica: a stale copy would pass for a round it is not
                    dealt = [int(s) for s in view["shards"]]
                    if not quiet and dealt and round_order(self.info["seed"], view["rank"], want, dealt)[0] not in self.shards:
                        self._phase("loading")  # θ is in; the text it has not held yet comes next
                    shards = self._gather(dealt, want, view["rank"], int(view.get("steps", 0)), priority=priority)
                    if self.stop.is_set() or self.finished:
                        return None
                    if replica:
                        self._start_prefetch()
                except PeerError as exc:
                    if self.stop.is_set():
                        return None
                    if exc.status in (409, 412):  # the round or pinned body changed while downloading
                        want = self.info["round"]
                        continue
                    if exc.status == 410:  # the run ended (stopped, finished) while this member was downloading
                        self.emit({"type": "info", "text": f"The run is over: {exc.detail}"})
                        self.finished = True
                        return None
                    raise
                return Entered(want, theta, view["generation"], view["rank"], view["deadline_s"], shards, want != asked,
                               float(view.get("lr_scale", 1.)), int(view.get("steps", 0)), float(view.get("train_s", 0.)), len(body),
                               int(view.get("rounds") or 0), view.get("plan"))
        return None

    def _network_event(self, stage, began, **fields):
        try:
            self.emit({"type": "diloco_transfer", "run": self.run, "node": self.node, "term": self.term,
                       "compute_round": self.round, "stage": stage, "seconds": max(0., self.clock() - began), **fields})
        except Exception:  # measurement must never change training or admission
            pass

    def _download(self, path, params, priority=True, cancel=None):
        """Required weights/shards and optional replica traffic use the same transport budgets."""
        options = {}
        if getattr(self.client, "interruptible", False):
            options["cancel"] = lambda: (self.stop.is_set() or self.halt.is_set() or bool(self.superseded) or
                                          (cancel is not None and cancel()))
        if getattr(self.client, "supports_admission_priority", False):
            options["priority"] = priority
        began, outcome, size = self.clock(), "failed", 0
        try:
            body = self._call(lambda: self.client.get_bytes(path, params, **options))
            size, outcome = len(body), "received"
            return body
        finally:
            self._network_event("download", began, exchange_round=params.get("round"), path=path, bytes=size,
                                outcome=outcome, required=bool(priority() if callable(priority) else priority))

    def _gather(self, dealt, number, rank, steps, priority=True):
        """The shards round `number` trains on, fetched in order: the dealt ones shuffled by (seed, rank, round), taken until their
        windows cover what the round draws (its steps * batch size * accumulation). The rest are not fetched (a replica gets them in
        the background), so a round's download follows its steps, not the corpus, and the first steps start after the first shard.
        The order and the sizes are data, not what this member holds, so the choice is the same on every run of the deal."""
        cfg = self.info["config"]
        need = (steps or plan_h(self.plan, number)) * int(cfg.get("batch_size", 1)) * int(cfg.get("accumulation", 1))
        order = round_order(self.info["seed"], rank, number, dealt)
        chosen, have = [], 0
        for shard in order:
            if self.stop.is_set() or self.finished:
                break
            self._shard(shard, priority=priority)
            chosen.append(shard)
            have += len(self.shards[shard])
            if have >= need:
                break
        return chosen

    def _shard(self, shard, priority=True, cancel=None):
        owner = object()
        with self.replica_lock:
            self.shard_demands.setdefault(shard, {})[owner] = priority

        def required():
            with self.replica_lock:
                demands = tuple(self.shard_demands.get(shard, {}).values())
            return any(demand() if callable(demand) else demand for demand in demands)

        try:
            self._hold_shard(shard, required, cancel)
        finally:
            with self.replica_lock:
                self.shard_demands[shard].pop(owner)
                if not self.shard_demands[shard]:
                    del self.shard_demands[shard]

    def _hold_shard(self, shard, priority=True, cancel=None):
        """Install one shard unless it is held; the training thread and a replica's prefetch may both want it. With a `cache_dir`, the
        coordinator is asked for the shard's content hash first, and a shard this node already holds under that hash (from an earlier
        run) is used without downloading it."""
        from studio_peer import PeerError
        with self.replica_lock:
            lock = self.shard_locks.setdefault(shard, threading.Lock())
        with lock:
            if shard in self.shards or self.stop.is_set() or self.finished:
                return
            vocabulary, digest = self.info["vocab_size"], None
            if self.cache_dir is not None:
                try:
                    answer = self._call(lambda: self.client.get_json(f"{self.base}/shards/{shard}/hash", self._q()))
                    digest = answer.get("hash") if isinstance(answer, dict) else None
                except PeerError as exc:
                    if exc.status != 404:  # 404: a coordinator that does not name its shards
                        raise
                held = cached_shard(self.cache_dir, digest, vocabulary) if digest else None
                if held is not None:
                    self.shards[shard] = held
                    self._keep_for_failover(digest, shard)
                    return
            blob = self._download(f"{self.base}/shards/{shard}", self._q(), priority=priority, cancel=cancel)
            self.downloaded += len(blob)
            if self.cache_dir is not None:
                digest = hashlib.sha256(blob).hexdigest()
                self.shards[shard] = install_shard(self.cache_dir, cache_name(digest, vocabulary), blob, vocabulary)
                self._keep_for_failover(digest, shard)
            else:
                self.shards[shard] = install_shard(self.shard_dir, shard, blob, vocabulary)

    def _keep_for_failover(self, digest, shard):
        """A replica must hold its shards where a coordinator made from it looks (from_replica): copy the cached folder there."""
        if self.replica_dir is None:
            return
        target = self.shard_dir / f"shard{shard}"
        if not target.exists():
            self.shard_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(Path(self.cache_dir) / f"shard{cache_name(digest, self.info['vocab_size'])}", target)

    # -- a replica
    def _start_prefetch(self):
        """Once the round's own shards are in, have a replica fetch the rest in the background (see _prefetch)."""
        if not self.prefetched and (self.prefetch is None or not self.prefetch.is_alive()):
            self.prefetch = threading.Thread(target=self._prefetch, args=(self.replica_epoch,), name=f"diloco-replica-{self.node}", daemon=True)
            self.prefetch.start()

    def _replicate(self, number, theta, rank=0, priority=False):
        """Keep theta (the bytes just downloaded) of round `number` with the outer momentum and diloco.json that go with it, replacing
        the round before; then, once, start fetching the validation windows and the shards this member lacks in the background. A round
        that moved on while the state downloaded (409) or one before any closed (404) leaves the folder as it was."""
        from studio_peer import Cancelled, PeerError
        epoch = self.replica_epoch
        if number > 0:
            try:
                momentum, record = decode_state(self._download(f"{self.base}/state", self._q({"round": number}), priority=priority,
                                                             cancel=lambda: epoch != self.replica_epoch))
            except Cancelled:
                return
            except PeerError as exc:
                if exc.status not in (404, 409, 412):
                    raise
                return
            if epoch != self.replica_epoch:
                return
            for name, data in (("theta.npz", theta), ("momentum.npz", encode_arrays(momentum)), (STATE_FILE, record),
                               ("info.json", json.dumps(self.info).encode("utf-8"))):  # info.json: what a coordinator made from this needs
                self._write_replica(epoch, name, data)
            self.holding = (number, rank)
            self.emit({"type": "info", "text": f"Holding round {number} as a replica."})
            self._announce()

    def _replica_plan(self, view):
        """Every view of a round carries the run's plan: write it (and the rounds and H it ends in) into the replica's copy of
        diloco.json when it differs, so a run extended after the replica fetched its state still has the extension in a coordinator
        made from the replica."""
        plan = view.get("plan")
        if self.replica_dir is None or self.holding is None or not plan:
            return
        with self.replica_guard:  # read and write as one step: _replicate writes this file too, from another thread
            epoch = self.replica_epoch
            try:
                record = json.loads((self.replica_dir / STATE_FILE).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return
            if record.get("plan") != plan:
                settings = {**record["settings"], "rounds": plan_rounds(plan), "inner_steps": plan[-1]["h"]}
                self._write_replica(epoch, STATE_FILE, json.dumps({**record, "settings": settings, "plan": plan}).encode("utf-8"))

    def _replica_paused(self):
        """The run is paused and no round is open, so the replica's copy of diloco.json (fetched when the round before opened) does not
        say so: record it, so a coordinator made from this replica comes up paused."""
        if self.replica_dir is None or self.holding is None:
            return
        path = self.replica_dir / STATE_FILE
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not record.get("paused"):
            self._write_replica(self.replica_epoch, STATE_FILE, json.dumps({**record, "paused": True}).encode("utf-8"))

    def _write_replica(self, epoch, name, data):
        """Write one file of the replica folder, unless that replica's epoch is over (demoted, or the run ended): checked and written
        under the guard, so a late write cannot bring back a folder _drop_replica removed."""
        with self.replica_guard:
            if epoch != self.replica_epoch:
                return False
            self.replica_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.replica_dir / (name + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(self.replica_dir / name)
            return True

    def _drop_replica(self):
        """End the current replica epoch and delete the folder; writes begun under it are void from now on."""
        with self.replica_guard:
            self.replica_epoch += 1
            self.prefetched = False
            shutil.rmtree(self.replica_dir, ignore_errors=True)
        if self.holding is not None:
            self.holding = None
            self._announce()

    def _announce(self):
        """Say what this member keeps as a replica (the app on this node publishes it as the node's `held` fact, studio_failover): the
        round, its rank, and `ready` once the validation windows and every shard are here too, so it could coordinate the run."""
        number, rank = self.holding or (None, None)
        try:
            self.emit({"type": "diloco_replica", "run": self.run, "node": self.node, "term": self.term, "held": number is not None,
                       "round": number, "rank": rank, "ready": bool(self.prefetched and number is not None)})
        except Exception:  # noqa: BLE001 - a courtesy, like a progress report
            pass

    def _prefetch(self, epoch):
        """The once-per-run part of a replica, off the training thread: the validation windows (kept, never trained on) and every shard
        not held. A failure ends it; the next round that finds this member a replica starts it again."""
        from studio_peer import PeerError
        try:
            if not (self.replica_dir / "validation.npz").is_file():
                if not self._write_replica(epoch, "validation.npz", self._download(f"{self.base}/validation", self._q(), priority=False,
                                                                                         cancel=lambda: epoch != self.replica_epoch)):
                    return
            for shard in range(self.info["shard_count"]):
                if self.stop.is_set() or self.halt.is_set() or self.finished or epoch != self.replica_epoch:
                    return
                self._shard(shard, priority=False, cancel=lambda: epoch != self.replica_epoch)
            with self.replica_guard:
                if epoch != self.replica_epoch:
                    return
                self.prefetched = True
            self.emit({"type": "info", "text": "Replica holds the validation windows and every shard."})
            self._announce()
        except (PeerError, OSError):
            return

    def _send(self, step, trained_tokens, model):
        self.round_step = step - self.first_step
        self._phase("uploading")
        weights = model.state_arrays()
        delta = {name: self.theta[name] - weights[name].astype(np.float32) for name in self.theta}
        encoded = self.clock()
        body = self._encode(delta)
        self._network_event("encode", encoded, exchange_round=self.round, bytes=len(body))
        params = {"term": self.term, "node": self.node, "generation": self.generation, "epoch": self.epoch, "tokens": trained_tokens - self.first_tokens,
                  "steps": step - self.first_step}
        if self.timed:  # none timed (a round of a step or two): no seconds, and the coordinator keeps the member's speed as it was
            params["seconds"] = self._work_seconds(step - self.first_step)
        if self.loss is not None:
            params["loss"] = self.loss
        if self.down_bps > 0:
            params["down_bps"] = round(self.down_bps)
        self._post(self.round, body, params)

    def _encode(self, delta):
        """The change as it goes on the wire. int8 loses the most, so it keeps what it lost and adds it to the next change (error
        feedback); the other precisions send the change as it is."""
        dtype = self.info["delta_dtype"]
        if dtype != "int8":
            return encode_arrays(delta, dtype)
        body, self.residual = encode_feedback(delta, dtype, self.residual, sparsity=self.info.get("delta_sparsity", 0.))
        return body

    def _post(self, number, body, params, overlapped=False):
        from studio_peer import PeerError
        began = self.clock()
        try:
            options = {"timeout": self.wait + 120}
            if getattr(self.client, "supports_admission_priority", False):
                options["priority"] = self.exchange_waiting.is_set if overlapped else True
            if getattr(self.client, "receiver_grants", False):
                options["cancel"] = lambda: self.stop.is_set() or self.fenced or self.superseded
            self._call(lambda: self.client.post_bytes(f"{self.base}/rounds/{number}/delta", body, params, **options))
            upload, trained = self.clock() - began, params.get("seconds", 0.)
            self._network_event("upload", began, exchange_round=number, bytes=len(body), outcome="accepted")
            sparse = f" (top {self.info['delta_sparsity']:.1%})" if self.info.get("delta_sparsity") else ""
            text = (f"in {upload:.2f} s, while the next round trained." if overlapped
                    else f"in {upload:.2f} s ({upload / max(1e-9, trained + upload):.0%} of the member's {trained + upload:.2f} s round).")
            self.emit({"type": "info", "text": f"Round {number}: sent {len(body)} bytes of {self.info['delta_dtype']}{sparse} delta {text}"})
        except PeerError as exc:
            self._network_event("upload", began, exchange_round=number, bytes=len(body), outcome="failed", status=exc.status)
            if exc.status != 409:
                raise
            if "lease epoch" in exc.detail or self.superseded:
                self.emit({"type": "info", "text": f"Fenced: {exc.detail}"})
                self.fenced = True
            else:  # late, or the membership moved on: this round's work is dropped; take the new theta and carry on
                self.emit({"type": "info", "text": f"Round {number} was not accepted ({exc.detail}); continuing from the new weights."})

    def leave(self):
        from studio_peer import PeerError
        if self.superseded:  # the old coordinator is not written to any more
            return
        try:
            self.client.post_json(f"{self.base}/leave", {"node": self.node, "generation": getattr(self, "generation", 0), "epoch": self.epoch}, self._q())
        except (PeerError, AttributeError):
            pass

    # -- the whole member
    def work(self, directory, emit=None):
        """Join, wait for the first round and train until the coordinator says the run is finished or `stop` is set. Deletes the
        shard cache and leaves when it ends. Returns the number of rounds this member took part in."""
        import studio_model
        emit = emit or self.emit
        self.emit = emit
        try:
            try:
                self.join()
            except Rebound as bound:  # the coordinator it was told to join is gone and another is in charge
                self._bound(bound.found)
            if self.finished:  # the coordinator in charge had already finished the run: there is nothing to train (and no round info to read)
                return None
            if self.progress:
                self.reporter = threading.Thread(target=self._report_loop, name=f"diloco-progress-{self.node}", daemon=True)
                self.reporter.start()
            self.shard_dir.mkdir(parents=True, exist_ok=True)
            try:
                self._enter(self.info["round"])
            except Rebound as bound:
                self._bound(bound.found)
            if self.finished or self.fenced or self.stop.is_set():
                return None

            def watch(event):
                if event.get("type") == "metric" and event.get("step"):
                    self.loss = event.get("loss")
                    self.live = {"loss": self.loss, "tokens_per_second": event.get("tokens_per_second")}
                if "total" in event:  # the loop bound below is not the run's length: report the run's own
                    event = {**event, "total": max(self.run_steps, event.get("step") or 0)}
                emit(event)
            cfg = {**self.info["config"], "eval_every": 100000, "sample_every": 0, "steps": studio_model.MAX_STEPS}  # the loop bound, as high as train() lets it go: the run's rounds end it (it can be extended while it goes), and the schedule's length is total_steps
            import studio_memory
            offload = studio_memory.offload_wanted(studio_model.validate_config(cfg), self.memory_share or 1.)  # as a job does: CUDA only, and only when the state cannot fit; train() says so
            if self.fit:  # the coordinator sized this member from its own memory: its rows per step in the pieces that fit, and where its optimizer lives
                cfg["batch_size"] = int(self.fit.get("batch_size") or cfg["batch_size"])
                cfg["accumulation"] = int(self.fit.get("accumulation") or cfg.get("accumulation", 1))
                offload = offload or bool(self.fit.get("offload_optimizer"))
            if self.memory_share or offload:  # an offloading member without a plan still has the whole device to fit
                share = self.memory_share or 1.
                declared = cfg["batch_size"]
                cfg["batch_size"] = studio_memory.batch_cap(cfg, share, offload=offload)
                if cfg["batch_size"] < declared:
                    emit({"type": "info", "text": f"A batch of {declared} does not fit the {share:.0%} of memory this node's plan "
                                                  f"allows: this member trains with batch size {cfg['batch_size']}."})
            attempt = 0
            while True:  # one attempt of train() per coordinator this member is attached to; a failover or a settled claim ends one
                self.first_step, found, self.warm = 0, None, False
                folder = directory if not attempt else Path(f"{directory}_{attempt}")
                shutil.rmtree(folder, ignore_errors=True)  # scratch: an earlier start of this member (the app restarted and resumed the run) left its checkpoint here, which train() refuses to write into
                try:
                    studio_model.train(cfg, [], folder, emit=watch,
                                       stop=Halt(self.stop, self.ended), pause=self.pause, distributed=self, telemetry_every=0, offload_optimizer=offload)
                    if (self.superseded or self.moved) and not self.finished and not self.stop.is_set():
                        found = self._relocated_within(self.clock() + self.patience)
                except Rebound as bound:
                    found = bound.found
                if found is None:
                    break
                self._bound(found)
                if self.finished or self.fenced or self.stop.is_set():
                    break
                attempt += 1
                gc.collect()  # the failed attempt's optimizer (its host-side state too) is freed before the next one allocates
            if not (self.finished or self.fenced):
                if self.stop.is_set():
                    return self.round
                raise RuntimeError("This member used all its steps before the run finished (it missed rounds). Start it again to rejoin.")
            return self.round
        finally:
            self.halt.set()
            self.wake.set()
            if self.pending is not None and hasattr(self.pending, "abandon"):
                self.pending.abandon(self.wait + 20)  # the exchange in flight ends with stop; it must not write the shard cache removed below
                self.pending = None
            self.leave()
            if self.prefetch is not None:
                self.prefetch.join(60)
            if self.replica_dir is not None:
                self._drop_replica()
            self.train.set([])
            self.shards.clear()
            gc.collect()  # a memory map keeps its file open (on Windows, undeletable) until it is collected
            shutil.rmtree(self.shard_dir, ignore_errors=True)


def where_file(root, run):
    """Where the app on this node writes which node coordinates `run` now ({"node", "term", "url"}; Cluster, from the store's claims)."""
    return Path(root) / "data" / "diloco_where" / f"{run}.json"


def file_locator(root, client_for=None):
    """`locate` for a member in a worker process, which has no store: it reads where_file, written by the app on its node. Returns
    (node, term, client) or None when the app has written nothing for the run. `client_for(url)` makes the client (a PeerClient)."""
    if client_for is None:
        from studio_peer import PeerClient as client_for
    clients = {}

    def locate(run):
        try:
            found = json.loads(where_file(root, run).read_text(encoding="utf-8"))
            node, term, url = str(found["node"]), int(found["term"]), str(found["url"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if url not in clients:
            clients[url] = client_for(url)
        return node, term, clients[url]
    return locate


def run_member(client, run, node, epoch, root, directory, emit, stop, **options):
    """Run one member against `client` (a studio_peer.PeerClient or anything with its methods); shards are cached under
    <root>/data/remote_shards/<run>/ and deleted at the end; a replica keeps the latest round under <root>/data/replicas/<run>/ and
    deletes it then too."""
    member = Member(client, run, node, epoch, Path(root) / "data" / "remote_shards" / run, stop, emit=emit,
                    replica_dir=Path(root) / "data" / "replicas" / run, cache_dir=Path(root) / "data" / "shard_cache", **options)
    return member.work(directory, emit)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m studio_diloco")
    sub = parser.add_subparsers(dest="command", required=True)
    member = sub.add_parser("member", help="join a run on another node's API and train until it finishes")
    member.add_argument("--coordinator", required=True, help="the owner node: a tailnet name or a URL, e.g. http://windows-box:8787")
    member.add_argument("--run", required=True, help="the run id: the pretrain_distributed job's id")
    member.add_argument("--node", default="", help="this member's name (default: the hostname)")
    member.add_argument("--epoch", type=int, default=0, help="lease epoch (default 0)")
    member.add_argument("--overlap", action="store_true",
                        help="train the next round while this round's delta uploads and the next weights download (applied a round late)")
    member.add_argument("--streams", type=int, default=None,
                        help="parallel streams per transfer of weights, shards and changes (default: studio_peer.STREAMS; 1 is one stream)")
    member.add_argument("--receiver-grants", action="store_true", help="opt in to experimental shared admission for delta uploads and versioned downloads")
    member.add_argument("--root", default="", help="folder that holds data/remote_shards (default: the workspace root)")
    args = parser.parse_args(argv)
    import socket
    from studio_core import ROOT
    from studio_peer import PeerClient
    stop = threading.Event()
    root = Path(args.root) if args.root else ROOT
    scratch = tempfile.mkdtemp(prefix="diloco_member_")
    try:
        def show(event):
            if event.get("type") in ("metric", "info"):
                print(json.dumps({key: event[key] for key in ("type", "step", "loss", "val_loss", "text") if key in event}), flush=True)
        run_member(PeerClient(args.coordinator), args.run, args.node or socket.gethostname(), args.epoch, root,
                   Path(scratch) / "member", show, stop, overlap=args.overlap, streams=args.streams,
                   receiver_grants=args.receiver_grants)
    except KeyboardInterrupt:
        stop.set()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
