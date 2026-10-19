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
PROGRESS_GAP = .5
PROGRESS_TIMEOUT = 5.  # seconds a progress report may take; a slow or failing one is dropped
PROGRESS_WAIT = .25  # seconds a progress report or a snapshot waits for a round that is closing before it answers without it
EVENTS_KEPT = 200  # the coordinator's event log is bounded
TIMELINES_KEPT = 8  # rounds whose timeline stays in memory (every finished round's also goes to metrics.jsonl)
SAMPLES_KEPT = 240  # progress samples kept per member for the open round


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


def replica_nodes(members, count):
    """The replicas of a round: the `count` lowest-ranked of its members (a member's rank is its place in the round's list)."""
    return list(members)[:max(0, count)]


def deal_shards(seed, round_number, members, shard_count):
    """{member: sorted shard ids}: the shards shuffled by (seed, round, membership) and dealt round-robin, so a rerun with the
    same membership history deals the same, and one member always holds every shard in order."""
    order = list(range(shard_count))
    random.Random(f"{seed}:{round_number}:{','.join(members)}").shuffle(order)
    return {member: sorted(order[index::len(members)]) for index, member in enumerate(members)}


def sampler_seed(seed, rank, round_number):
    return int.from_bytes(hashlib.sha256(f"{seed}:{rank}:{round_number}".encode()).digest()[:8], "big")


def round_seconds(settings, step_time):
    """A round's deadline from the fastest member's measured seconds per step (None: nobody has been timed yet)."""
    if not step_time:
        return settings.first_round_seconds
    return step_time * settings.inner_steps * (1 + settings.slack_ratio) + settings.slack_seconds


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
        self.members, self.pending, self.leaving, self.last_members, self.sequence = {}, {}, set(), [], 0
        self.current, self.finished, self.touched, self.deal = None, False, time.monotonic(), {}
        self.beats = {}  # node -> the clock time of its last call: the heartbeat studio_cluster_watch reads
        self.stopped, self.banned, self.gone = False, set(), {}  # stopped by the owner; nodes the owner removed; node -> (state, sentence) of those out
        self.split, self.split_hash, self.state_file, self.record = {"strategy": "lineage-v1"}, "", None, {}
        self.shard_count = max(1, min(settings.shards, len(train)))
        self.bounds = np.linspace(0, len(train), self.shard_count + 1).astype(int)
        self.parts = None  # a coordinator made from a replica serves the shards it holds one by one (from_replica)
        self._params, self._state = (None, b""), (None, b"")
        self.started, self.theta_nbytes = clock(), sum(value.nbytes for value in self.theta.values())  # the zero of the event log; θ's size
        self.reports, self.events, self.timelines = {}, deque(maxlen=EVENTS_KEPT), deque(maxlen=TIMELINES_KEPT)  # the live view's state
        self.event_seq, self.stage, self._last = 0, "", None  # stage: "outer_step" or "validating" while a round closes (read without the lock)
        self.term, self.superseded, self.on_step_down = 1, 0, on_step_down  # superseded: the higher term this coordinator was told of, else 0
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        if state:
            self.term = state["term"]
            self.momentum = state["momentum"]
            self.next_round, self.generation = state["round"], state["generation"]
            self.history, self.step_time, self.state_file = state["history"], state.get("step_time"), state["state_file"]
            self.finished = self.next_round >= settings.rounds

    # -- building and resuming
    @classmethod
    def build(cls, run, directory, config, examples, settings, clock=time.monotonic, shard_dir=None):
        """Tokenize `examples` once the way studio_model.train does, cut the windows, start theta from the config's seeded
        initialization, and continue from diloco.json if the folder has one."""
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
        coordinator = cls(run, directory, cfg, tokenizer, train, val, settings, theta, _evaluator(model, val, cfg["batch_size"]), digest, clock, state)
        coordinator.split_hash, coordinator.split = split_hash, split
        coordinator.model_spec = model.spec
        if not state:
            coordinator._start_files(model)
        return coordinator

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
                 "history": record["history"], "step_time": record.get("step_time"), "state_file": None}
        coordinator = cls(run, directory, cfg, Tokenizer.restore(info["tokenizer"]), Windows(parts), val, settings, theta,
                          _evaluator(model, val, cfg["batch_size"]), record["digest"], clock, state, on_step_down)
        coordinator.shard_count, coordinator.parts = count, parts
        coordinator.split, coordinator.split_hash = {"strategy": record.get("split_strategy", "lineage-v1")}, record.get("split_hash", "")
        coordinator.model_spec = model.spec
        coordinator.deal = record.get("shard_deal") or {}
        coordinator._start_files(model)
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
        return coordinator

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
                "history": saved["history"], "step_time": saved.get("step_time"), "term": saved.get("term", 1)}

    def _start_files(self, model):
        from studio_core import atomic_json, now
        atomic_json(self.directory / "config.json", self.cfg)
        atomic_json(self.directory / "spec.json", model.spec)
        atomic_json(self.directory / "tokenizer.json", self.tokenizer.serialize())
        (self.directory / "metrics.jsonl").write_text("", encoding="utf-8")
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
        for node in sorted(self.pending, key=lambda n: self.pending[n]["seq"]):
            if len(self.members) < self.settings.max_members:
                self.members[node] = self.pending.pop(node)
        active = sorted(self.members, key=lambda n: self.members[n]["seq"])
        if len(active) < self.settings.min_members:
            return False
        if active != self.last_members:
            self.generation += 1
            self.last_members = active
        window = round_seconds(self.settings, self.step_time)
        self.current = Round(self.next_round, self.generation, active, deal_shards(self.cfg["seed"], self.next_round, active, self.shard_count),
                             now, now + window, window)
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
            outer_step(self.theta, self.momentum, delta, self.settings.outer_lr, self.settings.outer_momentum)
            norm = step_size(delta, self.momentum, self.settings.outer_lr, self.settings.outer_momentum)
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
        times = [seconds / steps for _, steps, seconds, _, _ in sent.values() if steps > 0 and seconds > 0]
        self.step_time = min(times) if times else self.step_time
        tokens = sum(t for t, _, _, _, _ in sent.values())
        losses = [(t, loss) for t, _, _, loss, _ in sent.values() if loss is not None and t > 0]
        row = {"round": current.number, "step": (current.number + 1) * self.settings.inner_steps, "generation": current.generation,
               "members": list(current.members), "contributed": list(sent), "dropped": dropped, "tokens": tokens,
               "steps": {node: value[1] for node, value in sent.items()},
               "loss": sum(t * loss for t, loss in losses) / sum(t for t, _ in losses) if losses else None,
               "val_loss": self.evaluate(self.theta), "seconds": now - current.opened,
               "delta_bytes": sum(current.uploaded.get(node, 0) for node in sent), "delta_dtype": self.settings.delta_dtype, "deadline_seconds": current.window,
               "theta_bytes": sum(value.nbytes for value in self.theta.values()) * len(current.members),  # what the members downloaded
               "node_tokens": {node: value[0] for node, value in sent.items()}, "node_seconds": {node: value[2] for node, value in sent.items()},
               "node_bytes": {node: current.uploaded.get(node, 0) for node in sent},
               "lr": self.settings.outer_lr, "tokens_per_second": tokens / max(1e-9, now - current.opened), "grad_norm": 0.,
               "outer_step_norm": norm}
        self.history.append(row)
        self.deal = current.deal
        self.next_round += 1
        self.current = None
        self.finished = self.next_round >= self.settings.rounds
        self.timelines.append(self._timeline(current))
        self._boundary()
        self._save(row, current)
        if self.finished:
            self._event("finished", "", current.number, f"The run finished after {self.next_round} rounds.")
        self.cond.notify_all()

    def _view(self, number, node):
        entry = self.members.get(node) or self.pending.get(node)
        base = {"term": self.term, "round": number, "member": entry is not None, "current": self.next_round}
        if self.finished:
            return {**base, "state": "finished"}
        if number < self.next_round:
            return {**base, "state": "closed"}
        current = self.current
        if number == self.next_round and current is not None and node in current.members and node not in self.leaving:
            rank = current.members.index(node)
            return {**base, "state": "open", "generation": current.generation, "rank": rank,
                    "replica": node in replica_nodes(current.members, self.settings.replicas),
                    "deadline_s": max(0., current.deadline - self.clock()), "shards": current.deal[node], "members": len(current.members)}
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
                if view["state"] != first["state"] or view.get("member") != first.get("member"):
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
        (#361). Stale (409) while rounds are left, and for a stopped run, which saves nothing (#333)."""
        with self.cond:
            if self.stopped:
                raise Stale(f"Run {self.run} was stopped; it saves no weights.")
            if not self.finished:
                raise Stale(f"Run {self.run} has not finished; {self.settings.rounds - self.next_round} rounds are left.")
            return (self.directory / "last.npz").read_bytes()

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
        if not 0 <= shard < self.shard_count:
            raise KeyError(shard)
        if self.parts is not None:
            return shard_blob(self.parts[shard], 0, len(self.parts[shard]))
        return shard_blob(self.train, int(self.bounds[shard]), int(self.bounds[shard + 1]))

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
        if current is None:
            return "waiting"
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
            live = bool(report) and report["round"] == self.next_round
            found["progress"] = {
                "phase": report["phase"] if report else "waiting", "step": report["step"] if live else 0, "steps": self.settings.inner_steps,
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
                "timeline": timelines, "events": [dict(event) for event in self.events]}
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
            self._event("finished", "", self.next_round, reason)
            if self.record:
                self.record["status"] = "stopped"
                atomic_json(self.directory / "run.json", self.record)
            self._remember()
            self.cond.notify_all()

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
        from studio_core import atomic_json, now
        step = self._persist(closed.members, closed.deal)
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({k: row[k] for k in ("step", "loss", "val_loss", "tokens_per_second", "lr", "grad_norm")}
                                    | {"round": row["round"], "timeline": self._timeline(closed)}) + "\n")
        self._event("checkpoint", "", row["round"], f"Round {row['round']} checkpoint written.")
        self.record.update({"step": step, "val_loss": row["val_loss"], "status": "completed" if self.finished else "training"})
        if row["loss"] is not None:
            self.record["loss"] = row["loss"]
        if self.finished:
            self.record["finishedAt"] = now()
        atomic_json(self.directory / "run.json", self.record)
        self._remember()

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
        metadata = {"architecture": "groundwork_numpy_decoder", "format_version": 1, "config": self.cfg,
                    "tokenizer": self.tokenizer.serialize(), "step": step, "best": min([r["val_loss"] for r in self.history if r["val_loss"] is not None], default=None),
                    "optimizer_steps": step, "history": [], "data_hash": self.digest, "sampler_state": list(random.Random(self.cfg["seed"]).getstate()),
                    "model_rng": np.random.default_rng(self.cfg["seed"]).bit_generator.state, "device": "cpu",
                    "split_strategy": self.split["strategy"], "split_hash": self.split_hash}
        checkpoint = {"metadata": np.array(json.dumps(metadata))}
        checkpoint.update({"weight::" + k: v for k, v in self.theta.items()})
        self._write_npz(self.directory / "last.npz", checkpoint)
        atomic_json(self.directory / "diloco.json", {
            "format": FORMAT, "run": self.run, "term": self.term, "round": self.next_round, "generation": self.generation, "state_file": self.state_file,
            "momentum_file": self.state_file, "digest": self.digest, "seed": self.cfg["seed"], "settings": self.settings.__dict__,
            "members": members, "shard_deal": deal, "step_time": self.step_time, "history": self.history,
            "finished": self.finished, "split_strategy": self.split["strategy"], "split_hash": self.split_hash})
        if old and old != self.state_file:
            (self.directory / old).unlink(missing_ok=True)
        return step

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
    examples = p["examples"] if "examples" in p else services.examples(p)
    try:
        coordinator = Coordinator.build(job.id, directory, config, examples, settings)
    except ValueError as exc:
        if p.get("resuming"):
            raise ValueError(f"Ended: this machine restarted at round {p['resuming']['round']}, and the run could not be resumed: {exc}") from None
        raise
    register(job.id, coordinator)
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

    def rate_step(self, number):
        return self.round * self.info["inner_steps"] + (number - self.first_step)

    def due(self, step, trained_tokens):
        self.round_step = step - self.first_step  # the reporter thread reads it; nothing here waits on the network
        if self.round_step - self.reported_step >= PROGRESS_STEPS:
            self.wake.set()
        return step > self.first_step and (step - self.first_step >= self.info["inner_steps"] or self.clock() >= self.deadline)

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
                       "phase": phase, "step_in_round": step, "round_steps": self.info["inner_steps"], "loss": body["loss"],
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
        asked = want
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
                return Entered(want, theta, view["generation"], view["rank"], view["deadline_s"], shards, want != asked)
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
