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
import gc
import hashlib
import io
import json
import math
import random
import re
import shutil
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import studio_diloco_rules as drules

FORMAT = 1
TERM_HEADER = "X-Run-Term"  # every answer on /collective/{run} carries the coordinator's term here and in its JSON body
TERM_STALE = "Stale term"  # the start of the 409 message for a term that is not the coordinator's
LONG_POLL = 30.  # seconds a GET /rounds/{r} may wait for a change
STATE_PREFIXES = ("theta::", "momentum::")
DELTA_DTYPES = ("int8", "float16", "float32")
GONE_RUN = "No distributed run"  # the start of the 404 detail of a run the coordinator's app does not hold (routers.collective.coordinator)
STATE_FILE = "diloco.json"  # the entry of a GET /state archive that holds diloco.json's bytes
SCALE_SUFFIX = "::scale"  # an int8 array's float32 scale sits beside it in the archive under this suffix
RUN_ID = r"^[A-Za-z0-9_.-]{1,64}$"
PHASES = ("downloading", "training", "uploading", "waiting")  # what a member says it is doing (POST /progress)
WORK_PHASES = ("training", "uploading")  # the phases that belong to one open round: a report in them for a closed round is stale
MARKS = ("theta_taken", "first_step", "delta_sent", "delta_received")  # a round's timeline per member, in the order they happen
PROGRESS_SECONDS = 2.  # a member reports at least this often while it runs ...
PROGRESS_STEPS = 20  # ... or after this many steps, whichever comes first (never twice within PROGRESS_GAP)
MIN_SHARE = .1  # a member is dealt at least this fraction of H steps (deal_steps) ...
MAX_SHARE = 3  # ... and at most this many times H
PROBE_SHARE = .25  # a member with no measurement yet is dealt this fraction of H: a short first round that times it
STEP_TIME_SMOOTHING = .5  # the weight of the latest round in a member's smoothed seconds per step
PROGRESS_GAP = .5
PROGRESS_TIMEOUT = 5.  # seconds a progress report may take; a slow or failing one is dropped
PROGRESS_WAIT = .25  # seconds a progress report or a snapshot waits for a round that is closing before it answers without it
EVENTS_KEPT = 200  # the coordinator's event log is bounded
TIMELINES_KEPT = 8  # rounds whose timeline stays in memory (every finished round's also goes to metrics.jsonl)
SAMPLES_KEPT = 240  # progress samples kept per member for the open round
PROMPTS_PER_POLISH = 500  # the teacher is given this many training prompts per polish, a new window each time


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


class Stale(RuntimeError):
    """A request from an earlier round, generation or lease epoch, or into a closed or full run: the API answers 409."""


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
    delta_dtype: str = "float16"
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
    target_win_rate: float | None = None  # not available yet (#511): rejected
    teacher: str = ""                  # the teacher model's name (an Ollama model); needed by polish_every
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
        if not 1 <= self.min_members <= self.max_members:
            raise ValueError("min_members must be between 1 and max_members.")
        if self.delta_dtype not in DELTA_DTYPES:
            raise ValueError(f"delta_dtype must be one of {', '.join(DELTA_DTYPES)}.")
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
    The measured members share the work an equal-H round would give them (H each) in proportion to their speed, clamped to
    [MIN_SHARE, MAX_SHARE] of H; so equal speeds (and one member) give H to everyone, and mixed speeds do the same total work in
    the time of the fastest-weighted share instead of waiting for the slowest. A member with no measurement gets a probe round of
    PROBE_SHARE of H, so it is timed without holding the others up. While nobody is measured everyone gets H."""
    h = settings.inner_steps
    measured = {node: step_times[node] for node in members if step_times.get(node, 0) > 0}
    if not measured:
        return {node: h for node in members}
    low, high = max(1, math.ceil(h * MIN_SHARE)), max(1, int(h * MAX_SHARE))
    probe = min(h, max(1, round(h * PROBE_SHARE)))
    speed_total = sum(1. / seconds for seconds in measured.values())
    work = h * len(measured)
    return {node: (min(high, max(low, round(work / measured[node] / speed_total))) if node in measured else probe) for node in members}


def quantize_int8(value):
    """(int8 values, float32 scale) for one tensor: scale = max |x| / 127, values round(x / scale). A non-finite tensor gets a NaN
    scale, so it decodes non-finite and the coordinator refuses it as it would any other."""
    value = np.asarray(value, dtype=np.float32)
    peak = float(np.max(np.abs(value))) if value.size else 0.
    if not np.isfinite(peak):
        return np.zeros(value.shape, np.int8), np.float32("nan")
    scale = np.float32(peak / 127.)
    if scale == 0:
        return np.zeros(value.shape, np.int8), np.float32(0.)
    return np.clip(np.rint(value / scale), -127, 127).astype(np.int8), scale


def encode_arrays(arrays, dtype="float32"):
    """A {name: array} as an uncompressed .npz body; float16 is clipped to its range first; int8 stores each tensor as int8 values
    plus a per-tensor float32 scale (name + "::scale"), a quarter of the fp32 size."""
    buffer = io.BytesIO()
    if dtype == "float16":
        np.savez(buffer, **{name: np.clip(value, -65504., 65504.).astype(np.float16) for name, value in arrays.items()})
    elif dtype == "int8":
        out = {}
        for name, value in arrays.items():
            out[name], out[name + SCALE_SUFFIX] = quantize_int8(value)
        np.savez(buffer, **out)
    else:
        np.savez(buffer, **{name: np.asarray(value, dtype=np.float32) for name, value in arrays.items()})
    return buffer.getvalue()


def decode_arrays(data):
    """The fp32 arrays of an .npz body. ValueError when it is not one."""
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            out = {}
            for name in archive.files:
                if name.endswith(SCALE_SUFFIX) and name[:-len(SCALE_SUFFIX)] in archive.files:
                    continue
                value = archive[name]
                if value.dtype == np.int8:
                    value = value.astype(np.float32) * np.float32(archive[name + SCALE_SUFFIX])
                out[name] = value.astype(np.float32)
            return out
    except Exception as exc:  # noqa: BLE001 - zip and numpy raise many types for a bad body
        raise ValueError(f"The body is not an .npz archive of arrays ({exc}).") from None


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
    contributions: dict = field(default_factory=dict)  # node -> (tokens, steps, seconds, loss, delta)
    uploaded: dict = field(default_factory=dict)  # node -> bytes of its delta body on the wire
    open: bool = True
    marks: dict = field(default_factory=dict)  # node -> {MARKS name: coordinator clock time}
    closed: float | None = None


def step_size(delta, momentum, lr, mu):
    """The L2 norm of the outer step just taken, lr * (delta + mu * v) over every tensor."""
    return float(np.sqrt(sum(float(((lr * (gradient + mu * momentum[name])) ** 2).sum()) for name, gradient in delta.items())))


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
        self.paused = False  # the owner paused the run: no round opens until resume (a round in flight finishes)
        self.polishing = False  # the teacher is polishing theta, on a thread off the lock: like paused, no round opens meanwhile
        self.polisher, self.teacher_settings, self.polish_on = default_polish, {}, True  # polish_on: off on a coordinator made from a replica
        self._polish_stop = threading.Event()  # set by stop, step-down and release: the polish in flight is discarded
        self.lessons, self.lesson_windows = [], {}  # [{shard, round, rows, status}] in shard-id order; shard id -> its windows
        self.parts = None  # a coordinator made from a replica serves the shards it holds one by one (from_replica)
        self._params, self._state = (None, b""), (None, b"")
        self.started, self.theta_nbytes = clock(), sum(value.nbytes for value in self.theta.values())  # the zero of the event log; θ's size
        self.reports, self.events, self.timelines = {}, deque(maxlen=EVENTS_KEPT), deque(maxlen=TIMELINES_KEPT)  # the live view's state
        self.event_seq, self.stage, self._last = 0, "", None  # stage: "outer_step" or "validating" while a round closes (read without the lock)
        self.term, self.superseded, self.on_step_down = 1, 0, on_step_down  # superseded: the higher term this coordinator was told of, else 0
        self.prefer = ()  # nodes that are replicas before the others: the node chosen to coordinate (prepare)
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
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
    def build(cls, run, directory, config, examples, settings, clock=time.monotonic, shard_dir=None, later=(), start_from=None):
        """Tokenize `examples` once the way studio_model.train does, cut the windows, start theta from the config's seeded
        initialization, and continue from diloco.json if the folder has one. `later` are the example lists of the data sources
        the next_dataset pivot moves on to, in order: cut with the first source's tokenizer, never part of the validation. `start_from`, the
        path of a checkpoint (best.npz or last.npz of an earlier run), is where theta starts when the folder holds no run yet (a restart)."""
        import studio_model as sm
        from studio_core import fingerprint
        from studio_planning import merge_duplicate_lineage
        from studio_spec import module_of
        cfg = sm.validate_config({**config, "steps": settings.rounds * settings.inner_steps})
        if cfg["lora"]:
            raise ValueError("Distributed pretraining trains every weight; LoRA is not supported.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
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
        module, module_cfg = module_of("tokenizer", sm.effective_spec(cfg)["tokenizer"])
        tokenizer = module.fit(module_cfg, training)
        if sm.characters_in(training) + sm.characters_in(validation) >= sm.STREAM_CHARS:
            train, val, _ = sm.shard_windows(training, validation, tokenizer, cfg, split_hash, shard_dir or directory / "windows")
        else:
            train, val = sm.samples_from(training, tokenizer, cfg), sm.samples_from(validation, tokenizer, cfg)
        if not train:
            raise ValueError("No trainable target tokens were found.")
        model = sm.ScratchTransformer(cfg, tokenizer.vocab_size)
        theta = {name: value.astype(np.float32) for name, value in model.state_arrays().items()}
        state = cls._read_state(directory, cfg, settings, digest)
        if state:
            theta = state["theta"]
        elif start_from:
            theta = cls._weights_of(start_from, theta)
        coordinator = cls(run, directory, cfg, tokenizer, train, val, settings, theta, _evaluator(model, val, cfg["batch_size"]), digest, clock, state)
        coordinator.split_hash, coordinator.split = split_hash, split
        coordinator.model_spec = model.spec
        for number, more in enumerate(later, 1):
            coordinator.add_source(cls._windows_of(sm, more, tokenizer, cfg, directory / "windows" / f"source{number}"))
        coordinator._load_lessons()
        if coordinator.rules.source >= len(coordinator.spans):
            raise ValueError(f"The run had moved on to data source {coordinator.rules.source + 1}, but only {len(coordinator.spans)} "
                             f"{'is' if len(coordinator.spans) == 1 else 'are'} supplied (later_examples); training on the first would be a different run.")
        if not state:
            coordinator._start_files(model)
            held = directory / "paused.json"  # a pause before the first round closed
            if held.is_file():
                saved = json.loads(held.read_text(encoding="utf-8"))
                coordinator.paused, coordinator.rules = bool(saved.get("paused")), drules.State.from_json(saved.get("rules"))
        if settings.polish_every:  # after _start_files, which clears a new run's prompts
            coordinator._write_prompts(training)
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
        model = sm.ScratchTransformer(cfg, vocabulary)
        state = {"term": term, "momentum": momentum, "round": int(record["round"]), "generation": int(record["generation"]),
                 "history": record["history"], "step_time": record.get("step_time"), "state_file": None, "rules": record.get("rules"), "paused": record.get("paused"),
                 "lessons": record.get("lessons")}
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
        for entry in coordinator.lessons:
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
    def _write_prompts(self, training):
        """prompts.jsonl: the conversations of the training split the teacher is asked to answer (never a validation one), once per run."""
        path = self.directory / "prompts.jsonl"
        if path.is_file():
            return
        from studio_rlhf import prompts_from
        prompts = prompts_from(training)
        if not prompts:
            raise ValueError("polish_every is set, but the training text has no conversations for the teacher to answer: "
                             "use a chat dataset, or turn polish_every off.")
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
            if entry["status"] != "live":
                continue
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

    @staticmethod
    def _read_state(directory, cfg, settings, digest):
        path = directory / "diloco.json"
        if not path.is_file():
            return None
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["digest"] != digest or saved["settings"]["inner_steps"] != settings.inner_steps or saved["seed"] != cfg["seed"]:
            raise ValueError("This folder holds a different DiLoCo run (other data, seed or inner steps).")
        with np.load(directory / saved["state_file"], allow_pickle=False) as archive:
            theta = {key[len(STATE_PREFIXES[0]):]: archive[key].copy() for key in archive.files if key.startswith(STATE_PREFIXES[0])}
            momentum = {key[len(STATE_PREFIXES[1]):]: archive[key].copy() for key in archive.files if key.startswith(STATE_PREFIXES[1])}
        return {"theta": theta, "momentum": momentum, "state_file": saved["state_file"], "round": saved["round"], "generation": saved["generation"],
                "history": saved["history"], "step_time": saved.get("step_time"), "term": saved.get("term", 1),
                "rules": saved.get("rules"), "finished": saved.get("finished", False), "paused": saved.get("paused", False),
                "lessons": saved.get("lessons", [])}

    def _start_files(self, model):
        from studio_core import atomic_json, now
        atomic_json(self.directory / "config.json", self.cfg)
        atomic_json(self.directory / "spec.json", model.spec)
        atomic_json(self.directory / "tokenizer.json", self.tokenizer.serialize())
        (self.directory / "metrics.jsonl").write_text("", encoding="utf-8")
        (self.directory / "prompts.jsonl").unlink(missing_ok=True)  # ... nor prompts: they are made from this run's own training split
        (self.directory / "lessons.jsonl").unlink(missing_ok=True)  # a new run has no lessons: the rows of an earlier one in this folder are not its own
        self.record = {"name": self.cfg["name"], "createdAt": now(), "status": "training", "step": 0, "steps": self.cfg["steps"],
                       "parameters": model.parameter_count, "vocabulary": self.tokenizer.vocab_size, "backend": "diloco",
                       "device": "distributed", "rounds": self.settings.rounds, "inner_steps": self.settings.inner_steps}
        atomic_json(self.directory / "run.json", self.record)

    # -- membership
    def info(self):
        """What a member needs to start: the config and tokenizer, and the run's shape."""
        return {"term": self.term, "generation": self.generation, "round": self.next_round, "config": self.cfg, "tokenizer": self.tokenizer.serialize(),
                "inner_steps": self.settings.inner_steps, "rounds": self.settings.rounds,
                "total_steps": self.settings.rounds * self.settings.inner_steps, "delta_dtype": self.settings.delta_dtype,
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

    def join(self, node, epoch=0, term=None):
        with self.cond:
            self._touch(node)
            self._advance()
            if self.finished:
                raise Stale("The run is finished.")
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
                self._event("joined", node, self.next_round, f"{node} joined" + (" and starts with the next round." if self.current is not None else "."))
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
    def _expected(self, current):
        return set(current.members) - self.leaving

    def _event(self, kind, node, number, text):
        """Append to the bounded event log the live view shows, on this coordinator's own clock (seconds since it started)."""
        self.event_seq += 1
        self.events.append({"seq": self.event_seq, "t": self.clock() - self.started, "kind": kind, "node": node, "round": number, "text": text})

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
        steps = deal_steps(self.settings, active, self.step_times)
        finishes = [(steps[node] * self.step_times[node], node) for node in active if self.step_times.get(node, 0) > 0]
        if finishes:  # the slowest predicted finish, so the deadline is consistent with the steps dealt
            slowest = max(finishes)[1]
            window = round_seconds(self.settings, self.step_times[slowest], steps[slowest])
        else:
            window = round_seconds(self.settings, self.step_time)
        first, count = self.spans[min(self.rules.source, len(self.spans) - 1)]
        ids = [*range(first, first + count), *(entry["shard"] for entry in self.live_lessons())]  # the source's shards, then the lessons
        self.current = Round(self.next_round, self.generation, active, deal_ids(self.cfg["seed"], self.next_round, active, ids),
                             now, now + window, window, steps=steps)
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
        delta = token_average([(tokens, d) for tokens, _, _, _, d in sent.values()])
        norm = 0.
        if delta is not None:
            outer_lr = self.settings.outer_lr * self.rules.lr_scale  # halve_lr scales the outer step too
            outer_step(self.theta, self.momentum, delta, outer_lr, self.settings.outer_momentum)
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
        times = {node: seconds / steps for node, (_, steps, seconds, _, _) in sent.items() if steps > 0 and seconds > 0}
        self.step_time = min(times.values()) if times else self.step_time
        for node, seconds in times.items():  # a smoothed per-member speed, which the next rounds are dealt by
            before = self.step_times.get(node)
            self.step_times[node] = seconds if before is None else (1 - STEP_TIME_SMOOTHING) * before + STEP_TIME_SMOOTHING * seconds
        tokens = sum(t for t, _, _, _, _ in sent.values())
        losses = [(t, loss) for t, _, _, loss, _ in sent.values() if loss is not None and t > 0]
        row = {"round": current.number, "step": (current.number + 1) * self.settings.inner_steps, "generation": current.generation,
               "members": list(current.members), "contributed": list(sent), "dropped": dropped, "tokens": tokens,
               "steps": {node: value[1] for node, value in sent.items()}, "dealt": dict(current.steps),
               "loss": sum(t * loss for t, loss in losses) / sum(t for t, _ in losses) if losses else None,
               "val_loss": self.evaluate(self.theta), "seconds": now - current.opened,
               "delta_bytes": sum(current.uploaded.get(node, 0) for node in sent), "delta_dtype": self.settings.delta_dtype, "deadline_seconds": current.window,
               "theta_bytes": sum(value.nbytes for value in self.theta.values()) * len(current.members),  # what the members downloaded
               "node_tokens": {node: value[0] for node, value in sent.items()}, "node_seconds": {node: value[2] for node, value in sent.items()},
               "node_bytes": {node: current.uploaded.get(node, 0) for node in sent},
               "lr": self.settings.outer_lr * self.rules.lr_scale, "tokens_per_second": tokens / max(1e-9, now - current.opened), "grad_norm": 0.,
               "outer_step_norm": norm}
        self.history.append(row)
        self.deal = current.deal
        self.next_round += 1
        self.current = None
        self.finished = self.next_round >= self.settings.rounds
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
        self.cond.notify_all()

    def _apply_rules(self, row, current):
        """Fold the closed round into the stop and pivot rules (studio_diloco_rules): keep the best weights, go back to them and apply the
        next pivot when the loss kept getting worse or stopped improving, or end the run with the rule's reason. Returns the verdict."""
        rules = self.rules
        drules.accrue(rules, self.clock(), len(self.members) >= self.settings.min_members and not self.paused)
        loss = row["val_loss"]
        better = loss is not None and math.isfinite(loss) and (rules.best_loss is None or loss < rules.best_loss)
        since = current.number if better else rules.best_round  # the best round once this one is folded in: the lessons written after it are the ones a drop_lessons pivot leaves out
        lessons = len(self.live_lessons(since)) if since is not None else 0
        verdict = drules.observe(rules, self.settings, current.number, row["step"], row["val_loss"], sources=len(self.spans),
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
        return verdict

    def _write_best(self):
        """best.npz: the best round's weights as a checkpoint, plus the outer momentum it had (so a pivot continues from exactly there)."""
        arrays = self._checkpoint_arrays((self.rules.best_round + 1) * self.settings.inner_steps)
        arrays.update({STATE_PREFIXES[1] + k: v for k, v in self.momentum.items()})
        self._write_npz(self.directory / "best.npz", arrays)

    def _restore_best(self):
        with np.load(self.directory / "best.npz") as stored:
            for name in self.theta:
                self.theta[name] = np.array(stored["weight::" + name], dtype=np.float32)
                self.momentum[name] = np.array(stored[STATE_PREFIXES[1] + name], dtype=np.float32)
        self._params, self._state = (None, b""), (None, b"")

    def _view(self, number, node):
        entry = self.members.get(node) or self.pending.get(node)
        base = {"term": self.term, "round": number, "member": entry is not None, "current": self.next_round, "paused": self.paused,
                "polishing": self.polishing}
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
                    "lr_scale": self.rules.lr_scale, "steps": current.steps[node]}
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
                        or view["polishing"] != first["polishing"]):
                    return view
            return first

    def params_bytes(self, number, node=""):
        """Theta of the open round as an .npz body; `node` (who is asking) is noted on the round's timeline as having taken it."""
        with self.cond:
            self._touch()
            self._advance()
            if self.current is None or self.current.number != number:
                raise Stale(f"Round {number} is not the open round (the open one is {self.current.number if self.current else 'none'}).")
            if self._params[0] != number:
                self._params = (number, encode_arrays(self.theta, "float32"))
            if node in self.current.members:
                self.current.marks.setdefault(node, {}).setdefault("theta_taken", self.clock())
            return self._params[1]

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
        or of another generation, as a delta is. Returns False, taking nothing, when a closing round holds the lock for a while: a
        report is never worth holding a request for."""
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
            return True
        finally:
            self.lock.release()

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

    def submit(self, number, node, generation, epoch, tokens, steps, delta, seconds=0., loss=None, size=0, term=None):
        """Take a member's pseudo-gradient for the open round. Stale is raised (409) for any other round or generation, a
        member that is not in the round, a lease epoch that is not the member's, or a second contribution."""
        with self.cond:
            self._touch(node)
            self._advance()
            current = self.current
            try:
                self._admit(current, number, node, generation, epoch, term)
            except Stale as exc:
                self._event("stale", node, number, f"A change from {node} for round {number} was refused: {exc}")
                raise
            if set(delta) != set(self.theta) or any(delta[name].shape != self.theta[name].shape for name in delta):
                raise ValueError("The delta does not match the model's parameters.")
            if not all(np.isfinite(value).all() for value in delta.values()):
                raise ValueError("The delta holds a non-finite value.")
            if tokens < 0 or steps < 1:
                raise ValueError("A contribution needs at least one step.")
            current.contributions[node] = (int(tokens), int(steps), float(seconds), loss, delta)
            current.uploaded[node] = int(size)
            marks = current.marks.setdefault(node, {})
            marks["delta_received"] = self.clock()
            marks.setdefault("delta_sent", marks["delta_received"])
            self._advance()
            self.cond.notify_all()

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
        """Where the run is in its cycle: waiting (for members), next_round (members are taking theta), training, collecting (some
        change is in or on its way), outer_step, validating, or finished / stopped."""
        if self.stopped or self.finished:
            return "stopped" if self.stopped else "finished"
        if self.stage:
            return self.stage
        if self.polishing:
            return "polishing"
        if current is None:
            return "paused" if self.paused else "waiting"
        reports = [self.reports.get(node) for node in current.members]
        working = [r for r in reports if r and r.get("round") == current.number and r.get("phase") in WORK_PHASES]
        if current.contributions or any(r["phase"] == "uploading" for r in working):
            return "collecting"
        return "training" if working else "next_round"

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
        deal = current.deal if current is not None else self.deal
        for node, found in nodes.items():
            report = self.reports.get(node)
            if report is not None and "round" not in report:
                report = None  # made by a report that was refused (Stale) before it said anything
            live = bool(report) and report["round"] == self.next_round
            found["progress"] = {
                "phase": report["phase"] if report else "waiting", "step": report["step"] if live else 0,
                "steps": current.steps.get(node, self.settings.inner_steps) if current is not None else self.settings.inner_steps,
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
        with self.cond:
            try:
                self._polish_done(context, outcome, stop)
            finally:
                self.polishing = False
                self._advance()
                self.cond.notify_all()

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
                self._params, self._state = (None, b""), (None, b"")
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
        self.record.update({"step": self.next_round * self.settings.inner_steps,
                            "val_loss": row["val_loss"],
                            "status": "completed" if self.finished else "training",
                            "best_round": self.rules.best_round, "best_val_loss": self.rules.best_loss})
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
        step = self.next_round * self.settings.inner_steps
        self._write_npz(self.directory / "last.npz", self._checkpoint_arrays(step))
        self._keep_checkpoint()
        atomic_json(self.directory / "diloco.json", {
            "format": FORMAT, "run": self.run, "term": self.term, "round": self.next_round, "generation": self.generation, "state_file": self.state_file,
            "momentum_file": self.state_file, "digest": self.digest, "seed": self.cfg["seed"], "settings": self.settings.__dict__,
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
            self.theta, self.momentum, self._params, self._state = {}, {}, (None, b""), (None, b"")


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
IDLE_RELEASE = 3600.  # seconds untouched after which a coordinator is forgotten when another starts


_LOCK = threading.Lock()


def teacher_settings(settings, root=None):
    """The Ollama teacher a run polishes with: the workspace's RLHF settings (address, rubric, judge model) with the run's own teacher model."""
    from studio_rlhf import SETTINGS_DEFAULTS, load_settings
    return {**(load_settings(root) if root else json.loads(json.dumps(SETTINGS_DEFAULTS))), "teacher_model": settings.teacher}


def _teacher(found):
    from studio_rlhf import OllamaTeacher
    return OllamaTeacher(found["url"], found["teacher_model"], rubric=found.get("rubric"), judge_model=found.get("judge_model") or None)


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
    teacher = _teacher(found)
    teacher.models()  # RuntimeError when Ollama is not reachable
    Path(directory).mkdir(parents=True, exist_ok=True)
    store = PreferenceStore(directory)
    before = len(store)
    result = auto_rlhf(str(checkpoint), prompts, teacher, store, directory, {"rounds": 1}, {"device": device}, emit, stop, None, teacher.judge_teacher())
    if result.get("aborted"):
        raise RuntimeError(result["aborted"])
    rows = [row for row in store.all()[before:] if row.get("kind") == "rewrite"]
    return {"checkpoint": result["checkpoint"], "win_rate": result["win_rate"], "lessons": [dict(row) for row in rows]}


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
    config = dict(p["config"])
    settings = Settings.from_params(p, steps=config.get("steps"))
    teacher = None
    examples = p["examples"] if "examples" in p else services.examples(p)
    try:
        if settings.polish_every:
            teacher = teacher_settings(settings, getattr(services, "root", None))
            if not p.get("resuming"):  # a new run fails now, not at the first polish, when the teacher cannot be used; a resumed one meets it then
                try:
                    check_teacher(teacher)
                except RuntimeError as exc:
                    raise ValueError(f"The run polishes with the teacher {settings.teacher}, but it cannot be used: {exc}") from None
        coordinator = Coordinator.build(job.id, directory, config, examples, settings, later=p.get("later_examples") or (),
                                       start_from=p.get("restart_from"))
    except ValueError as exc:
        if p.get("resuming"):
            raise ValueError(f"Ended: this machine restarted at round {p['resuming']['round']}, and the run could not be resumed: {exc}") from None
        raise
    run = studio_distributed.run_of(job)
    if teacher is not None:
        coordinator.teacher_settings = teacher
    coordinator.prefer = (run.handoff,) if run.handoff else ()  # the node chosen to coordinate holds the run from the first round on
    register(job.id, coordinator)
    run.folder = str(directory)
    studio_distributed.persist()
    studio_distributed.resumed(job)
    p.update(run=job.id, coordinator_url=f"http://127.0.0.1:{port}", settings=dict(settings.__dict__))
    studio_distributed.invite(job, **({"client_for": client_for} if client_for else {}))  # the peers the person chose: a member job on each, which joins at the coordinator
    return coordinator


# ---- the member -------------------------------------------------------------------------------------------------------

class Member:
    """One node's side of a run. It is the `distributed=` hook of studio_model.train: before the first step it loads theta of
    its round, after H steps (or at the round's deadline) it sends delta and waits for the next round, whose theta it loads.
    With `overlap` it does not wait: the exchange (delta up, next theta down) runs through `spawn(fn)` (a Background thread; tests
    inject their own) while the next round's steps run, and is joined at the end of that round, where its theta is applied."""

    def __init__(self, client, run, node, epoch, shard_dir, stop, clock=time.monotonic, emit=None, wait=LONG_POLL, patience=300., sleep=time.sleep,
                 progress=True, overlap=False, spawn=Background, replica_dir=None, locate=None, pause=None, memory_share=None):
        self.pause, self.memory_share = pause, memory_share  # this node's queue plan: its pause event and its share of memory
        self.client, self.run, self.node, self.epoch, self.shard_dir, self.stop = client, run, node, epoch, Path(shard_dir), stop
        self.replica_dir = Path(replica_dir) if replica_dir else None  # where a replica keeps the latest round (None: it keeps none)
        self.prefetch, self.prefetched, self.shard_locks, self.replica_lock = None, False, {}, threading.Lock()
        self.replica_epoch, self.replica_guard = 0, threading.Lock()  # a write to replica_dir is void once the epoch it began under is over
        self.clock, self.emit, self.wait, self.patience, self.sleep = clock, emit or (lambda event: None), wait, patience, sleep
        self.base = f"/collective/{run}"
        self.train, self.val, self.sampler = Windows(), Windows(), random.Random(0)
        self.started, self.finished, self.fenced = False, False, False
        self.term, self.superseded = 0, 0  # the coordinator's term (0: not learnt yet); superseded: a higher term seen, so nothing more goes to this coordinator
        self.loss = None
        self.shards = {}
        # what it tells the coordinator (POST /progress): the phase, its step in the round and the latest loss and speed
        self.progress, self.phase, self.report_round, self.round_step, self.reported_step = progress, "waiting", None, 0, 0
        self.reported_phase = None
        self.live = {"loss": None, "tokens_per_second": None}
        self.wake, self.halt, self.reporter = threading.Event(), threading.Event(), None
        self.overlap, self.spawn, self.pending, self.entered, self.anchor = overlap, spawn, None, None, None
        # locate(run) -> (node, term, client) of the coordinator now in charge, or None (studio_failover); where: the (node, term) this member
        # last rebound to; ended: this attempt of train() is over but the job is not (the member may rebind); holding: (round, rank) it keeps
        self.locate, self.where, self.ended, self.holding, self.moved = locate, None, threading.Event(), None, False

    # -- the train() hook
    @property
    def total_steps(self):
        return self.info["total_steps"]

    @property
    def round_steps(self):
        """The inner steps this member was dealt for the round in hand."""
        return (self.entered.steps if self.entered is not None else 0) or self.info["inner_steps"]

    def rate_step(self, number):
        """The schedule position of step `number`: each round spans H positions of the schedule, and the member's steps (however many it
        was dealt) are spread over them, so every member is at the same point of the schedule when the round ends."""
        offset, h = number - self.first_step, self.info["inner_steps"]
        return self.round * h + (offset if self.round_steps == h else offset * h / self.round_steps)

    @property
    def rate_scale(self):
        """The factor the run's pivots (halve_lr) put on the learning rate, as the round in hand was opened with."""
        return self.entered.lr_scale if self.entered is not None else 1.

    def due(self, step, trained_tokens):
        self.round_step = step - self.first_step  # the reporter thread reads it; nothing here waits on the network
        if self.round_step - self.reported_step >= PROGRESS_STEPS:
            self.wake.set()
        return step > self.first_step and (step - self.first_step >= self.round_steps or self.clock() >= self.deadline)

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
        self.sampler.seed(sampler_seed(self.info["seed"], self.rank, self.round))
        self.round_step, self.reported_step = 0, 0
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
                  "steps": step - self.first_step, "seconds": self.clock() - self.began}
        if self.loss is not None:
            params["loss"] = self.loss
        self.pending = self.spawn(lambda: self._exchange(number, delta, params))
        if number + 1 >= self.info["rounds"]:  # the last round: nothing to train after it, so wait for the run's end
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
        return pending.result()

    def _exchange(self, number, delta, params):
        """On the background thread: send round `number`'s delta, then take round number + 1."""
        body = encode_arrays(delta, self.info["delta_dtype"])
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
            self._seen(self.client.post_json(f"{self.base}/progress", body, self._q(), timeout=PROGRESS_TIMEOUT))
        except Exception as exc:  # noqa: BLE001 - a stale 409, an unreachable coordinator, a timeout: the report is only a courtesy
            if hasattr(exc, "detail"):
                self._refused(exc)
            return False
        self.reported_step, self.reported_phase = step, phase
        try:  # the app on this node publishes it as this member's own fact (studio_facts, "progress" and "shards")
            self.emit({"type": "diloco_member", "run": self.run, "node": self.node, "term": self.term, "round": self.report_round,
                       "phase": phase, "step_in_round": step, "round_steps": self.round_steps, "loss": body["loss"],
                       "tokens_per_second": body["tokens_per_second"], "shard_ids": sorted(list(self.shards))})
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
        """Run a client call, waiting for the coordinator to come back if it cannot be reached (a restart) for up to `patience`.
        Once a higher term has been seen it sends nothing: the call is a 409 without leaving this process."""
        from studio_peer import PeerError
        end = self.clock() + self.patience
        while True:
            if self.superseded:
                raise PeerError(f"{TERM_STALE} {self.term}: the run is at term {self.superseded}.", status=409,
                                detail=f"{TERM_STALE} {self.term}: the run is at term {self.superseded}.")
            try:
                return self._seen(call())
            except PeerError as exc:
                self._refused(exc)
                if exc.status == 404 and exc.detail.startswith(GONE_RUN):
                    found = self._relocated()  # a coordinator that stepped down is gone from its node, and the run is coordinated elsewhere now
                    if found is not None:
                        raise Rebound(found) from None
                    raise PeerError(f"The coordinator no longer has run {self.run} (its app restarted), so this member left. "
                                    f"It joins again if it is invited again.", status=404, detail=exc.detail) from None
                if exc.status is not None or self.clock() >= end or self.stop.is_set():
                    if exc.status is None and not self.stop.is_set():
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

    def _relocated_within(self):
        """Like _relocated, but waits up to `patience` for the claim of the coordinator in charge to reach this node's store: a member told
        of a higher term has seen it before the claim that carries it."""
        end = self.clock() + self.patience
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
        self.term, self.superseded = 0, 0
        self.fenced = self.finished = self.started = self.moved = False
        self.entered = self.anchor = None
        self.ended.clear()
        self.join()
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
                    body = self._call(lambda: self.client.get_bytes(f"{self.base}/params", self._q({"round": want, "node": self.node})))
                    theta = decode_arrays(body)
                    if view.get("replica") and self.replica_dir is not None:
                        self._replicate(want, body, view["rank"])
                    elif self.replica_dir is not None:
                        self._drop_replica()  # no longer a replica: a stale copy would pass for a round it is not
                    shards = [int(s) for s in view["shards"]]
                    self._fetch(shards)
                except PeerError as exc:
                    if exc.status == 409:  # the round moved on while this member was downloading
                        want = self.info["round"]
                        continue
                    raise
                return Entered(want, theta, view["generation"], view["rank"], view["deadline_s"], shards, want != asked,
                               float(view.get("lr_scale", 1.)), int(view.get("steps", 0)))
        return None

    def _fetch(self, wanted):
        for shard in wanted:
            self._shard(shard)

    def _shard(self, shard):
        """Download and install one shard unless it is held; the training thread and a replica's prefetch may both want it."""
        with self.replica_lock:
            lock = self.shard_locks.setdefault(shard, threading.Lock())
        with lock:
            if shard not in self.shards:
                blob = self._call(lambda: self.client.get_bytes(f"{self.base}/shards/{shard}", self._q()))
                self.shards[shard] = install_shard(self.shard_dir, shard, blob, self.info["vocab_size"])

    # -- a replica
    def _replicate(self, number, theta, rank=0):
        """Keep theta (the bytes just downloaded) of round `number` with the outer momentum and diloco.json that go with it, replacing
        the round before; then, once, start fetching the validation windows and the shards this member lacks in the background. A round
        that moved on while the state downloaded (409) or one before any closed (404) leaves the folder as it was."""
        from studio_peer import PeerError
        if number > 0:
            try:
                momentum, record = decode_state(self._call(lambda: self.client.get_bytes(f"{self.base}/state", self._q({"round": number}))))
            except PeerError as exc:
                if exc.status not in (404, 409):
                    raise
                return
            epoch = self.replica_epoch
            for name, data in (("theta.npz", theta), ("momentum.npz", encode_arrays(momentum)), (STATE_FILE, record),
                               ("info.json", json.dumps(self.info).encode("utf-8"))):  # info.json: what a coordinator made from this needs
                self._write_replica(epoch, name, data)
            self.holding = (number, rank)
            self.emit({"type": "info", "text": f"Holding round {number} as a replica."})
            self._announce()
        if not self.prefetched and (self.prefetch is None or not self.prefetch.is_alive()):
            self.prefetch = threading.Thread(target=self._prefetch, args=(self.replica_epoch,), name=f"diloco-replica-{self.node}", daemon=True)
            self.prefetch.start()

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
                if not self._write_replica(epoch, "validation.npz", self._call(lambda: self.client.get_bytes(f"{self.base}/validation", self._q()))):
                    return
            for shard in range(self.info["shard_count"]):
                if self.stop.is_set() or self.halt.is_set() or epoch != self.replica_epoch:
                    return
                self._shard(shard)
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
        body = encode_arrays(delta, self.info["delta_dtype"])
        params = {"term": self.term, "node": self.node, "generation": self.generation, "epoch": self.epoch, "tokens": trained_tokens - self.first_tokens,
                  "steps": step - self.first_step, "seconds": self.clock() - self.began}
        if self.loss is not None:
            params["loss"] = self.loss
        self._post(self.round, body, params)

    def _post(self, number, body, params, overlapped=False):
        from studio_peer import PeerError
        began = self.clock()
        try:
            self._call(lambda: self.client.post_bytes(f"{self.base}/rounds/{number}/delta", body, params, timeout=self.wait + 120))
            upload, trained = self.clock() - began, params["seconds"]
            text = (f"in {upload:.2f} s, while the next round trained." if overlapped
                    else f"in {upload:.2f} s ({upload / max(1e-9, trained + upload):.0%} of the member's {trained + upload:.2f} s round).")
            self.emit({"type": "info", "text": f"Round {number}: sent {len(body)} bytes of {self.info['delta_dtype']} delta {text}"})
        except PeerError as exc:
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
                emit(event)
            cfg = {**self.info["config"], "eval_every": 100000, "sample_every": 0, "steps": self.info["total_steps"]}
            if self.memory_share:
                import studio_memory
                declared = cfg["batch_size"]
                cfg["batch_size"] = studio_memory.batch_cap(cfg, self.memory_share)
                if cfg["batch_size"] < declared:
                    emit({"type": "info", "text": f"A batch of {declared} does not fit the {self.memory_share:.0%} of memory this node's plan "
                                                  f"allows: this member trains with batch size {cfg['batch_size']}."})
            attempt = 0
            while True:  # one attempt of train() per coordinator this member is attached to; a failover or a settled claim ends one
                self.first_step, found = 0, None
                try:
                    studio_model.train(cfg, [], directory if not attempt else Path(f"{directory}_{attempt}"), emit=watch,
                                       stop=Halt(self.stop, self.ended), pause=self.pause, distributed=self, telemetry_every=0)
                    if (self.superseded or self.moved) and not self.finished and not self.stop.is_set():
                        found = self._relocated_within()
                except Rebound as bound:
                    found = bound.found
                if found is None:
                    break
                self._bound(found)
                if self.finished or self.fenced or self.stop.is_set():
                    break
                attempt += 1
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
                    replica_dir=Path(root) / "data" / "replicas" / run, **options)
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
                   Path(scratch) / "member", show, stop, overlap=args.overlap)
    except KeyboardInterrupt:
        stop.set()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
