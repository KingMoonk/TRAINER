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
"""
from __future__ import annotations

import argparse
import bisect
import gc
import hashlib
import io
import json
import random
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT = 1
LONG_POLL = 30.  # seconds a GET /rounds/{r} may wait for a change
STATE_PREFIXES = ("theta::", "momentum::")
DELTA_DTYPES = ("int8", "float16", "float32")
SCALE_SUFFIX = "::scale"  # an int8 array's float32 scale sits beside it in the archive under this suffix
RUN_ID = r"^[A-Za-z0-9_.-]{1,64}$"


class Stale(RuntimeError):
    """A request from an earlier round, generation or lease epoch, or into a closed or full run: the API answers 409."""


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

    def __post_init__(self):
        if self.inner_steps < 1 or self.rounds < 1 or self.shards < 1:
            raise ValueError("inner_steps, rounds and shards must be at least 1.")
        if not 1 <= self.min_members <= self.max_members:
            raise ValueError("min_members must be between 1 and max_members.")
        if self.delta_dtype not in DELTA_DTYPES:
            raise ValueError(f"delta_dtype must be one of {', '.join(DELTA_DTYPES)}.")
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


class Coordinator:
    """The global state of one run. Thread-safe; every method that can change it first moves the clock's consequences along
    (_advance: close a round that is complete or past its deadline, open the next). `evaluate(theta)` returns the validation loss."""

    def __init__(self, run, directory, cfg, tokenizer, train, val, settings, theta, evaluate, digest="", clock=time.monotonic, state=None):
        self.run, self.directory, self.cfg, self.tokenizer = run, Path(directory), cfg, tokenizer
        self.train, self.val, self.settings, self.evaluate, self.digest, self.clock = train, val, settings, evaluate, digest, clock
        self.theta = {name: np.array(value, dtype=np.float32) for name, value in theta.items()}
        self.momentum = {name: np.zeros_like(value) for name, value in self.theta.items()}
        self.next_round, self.generation, self.history, self.step_time = 0, 0, [], None
        self.members, self.pending, self.leaving, self.last_members, self.sequence = {}, {}, set(), [], 0
        self.current, self.finished, self.touched, self.deal = None, False, time.monotonic(), {}
        self.split, self.split_hash, self.state_file, self.record = {"strategy": "lineage-v1"}, "", None, {}
        self.shard_count = max(1, min(settings.shards, len(train)))
        self.bounds = np.linspace(0, len(train), self.shard_count + 1).astype(int)
        self._params = (None, b"")
        self.cond = threading.Condition(threading.RLock())
        if state:
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
                "history": saved["history"], "step_time": saved.get("step_time")}

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
        return {"generation": self.generation, "round": self.next_round, "config": self.cfg, "tokenizer": self.tokenizer.serialize(),
                "inner_steps": self.settings.inner_steps, "rounds": self.settings.rounds,
                "total_steps": self.settings.rounds * self.settings.inner_steps, "delta_dtype": self.settings.delta_dtype,
                "shard_count": self.shard_count, "seed": self.cfg["seed"], "vocab_size": self.tokenizer.vocab_size}

    def join(self, node, epoch=0):
        with self.cond:
            self._touch()
            self._advance()
            if self.finished:
                raise Stale("The run is finished.")
            known = self.members.get(node) or self.pending.get(node)
            if known:
                if epoch < known["epoch"]:
                    raise Stale(f"Stale lease epoch {epoch} for {node}: the current one is {known['epoch']}.")
                known["epoch"] = epoch
                self.leaving.discard(node)
            else:
                if len(self.members) + len(self.pending) >= self.settings.max_members:
                    raise Stale(f"The run already has its {self.settings.max_members} members.")
                self.sequence += 1
                entry = {"epoch": epoch, "seq": self.sequence, "missed": 0}
                (self.pending if self.current is not None else self.members)[node] = entry  # a boundary is now, or at the round's end
            self._advance()
            self.cond.notify_all()
            return {**self.info(), "node": node}

    def leave(self, node, generation, epoch):
        with self.cond:
            self._touch()
            entry = self.members.get(node) or self.pending.get(node)
            if entry is None or entry["epoch"] != epoch:
                raise Stale(f"{node} is not a member at lease epoch {epoch}.")
            if node in self.pending:
                del self.pending[node]
            elif self.current is None:
                del self.members[node]
            else:
                self.leaving.add(node)  # leaves at the round boundary; the round stops waiting for it now
            self._advance()
            self.cond.notify_all()

    # -- rounds
    def _expected(self, current):
        return set(current.members) - self.leaving

    def _advance(self):
        while not self.finished:
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
        if entry["missed"] >= self.settings.missed_limit:
            del self.members[node]  # it must join again

    def _boundary(self):
        for node in self.leaving:
            self.members.pop(node, None)
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
        current.open = False
        sent = current.contributions
        delta = token_average([(tokens, d) for tokens, _, _, _, d in sent.values()])
        if delta is not None:
            outer_step(self.theta, self.momentum, delta, self.settings.outer_lr, self.settings.outer_momentum)
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
               "lr": self.settings.outer_lr, "tokens_per_second": tokens / max(1e-9, now - current.opened), "grad_norm": 0.}
        self.history.append(row)
        self.deal = current.deal
        self.next_round += 1
        self.current = None
        self.finished = self.next_round >= self.settings.rounds
        self._boundary()
        self._save(row, current)
        self.cond.notify_all()

    def _view(self, number, node):
        entry = self.members.get(node) or self.pending.get(node)
        base = {"round": number, "member": entry is not None, "current": self.next_round}
        if self.finished:
            return {**base, "state": "finished"}
        if number < self.next_round:
            return {**base, "state": "closed"}
        current = self.current
        if number == self.next_round and current is not None and node in current.members and node not in self.leaving:
            return {**base, "state": "open", "generation": current.generation, "rank": current.members.index(node),
                    "deadline_s": max(0., current.deadline - self.clock()), "shards": current.deal[node], "members": len(current.members)}
        return {**base, "state": "pending"}

    def round_view(self, number, node="", wait=0.):
        """Where round `number` stands for `node`: open (with its assignment), pending, closed, or finished. With wait, a pending
        round is held until that changes, or for at most LONG_POLL seconds; any other state is answered at once."""
        end = time.monotonic() + min(max(wait, 0.), LONG_POLL)
        with self.cond:
            self._touch()
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

    def params_bytes(self, number):
        with self.cond:
            self._touch()
            self._advance()
            if self.current is None or self.current.number != number:
                raise Stale(f"Round {number} is not the open round (the open one is {self.current.number if self.current else 'none'}).")
            if self._params[0] != number:
                self._params = (number, encode_arrays(self.theta, "float32"))
            return self._params[1]

    def shard_bytes(self, shard):
        if not 0 <= shard < self.shard_count:
            raise KeyError(shard)
        return shard_blob(self.train, int(self.bounds[shard]), int(self.bounds[shard + 1]))

    def submit(self, number, node, generation, epoch, tokens, steps, delta, seconds=0., loss=None, size=0):
        """Take a member's pseudo-gradient for the open round. Stale is raised (409) for any other round or generation, a
        member that is not in the round, a lease epoch that is not the member's, or a second contribution."""
        with self.cond:
            self._touch()
            self._advance()
            current = self.current
            if current is None or current.number != number or not current.open:
                raise Stale(f"Round {number} is closed (the open round is {current.number if current else 'none'}).")
            if generation != current.generation:
                raise Stale(f"Stale generation {generation}: round {number} is generation {current.generation}.")
            entry = self.members.get(node)
            if node not in current.members or entry is None:
                raise Stale(f"{node} is not a member of round {number}.")
            if epoch != entry["epoch"]:
                raise Stale(f"Stale lease epoch {epoch} for {node}: the current one is {entry['epoch']}.")
            if node in current.contributions:
                raise Stale(f"{node} already contributed to round {number}.")
            if set(delta) != set(self.theta) or any(delta[name].shape != self.theta[name].shape for name in delta):
                raise ValueError("The delta does not match the model's parameters.")
            if not all(np.isfinite(value).all() for value in delta.values()):
                raise ValueError("The delta holds a non-finite value.")
            if tokens < 0 or steps < 1:
                raise ValueError("A contribution needs at least one step.")
            current.contributions[node] = (int(tokens), int(steps), float(seconds), loss, delta)
            current.uploaded[node] = int(size)
            self._advance()
            self.cond.notify_all()

    # -- durability
    def _touch(self):
        self.touched = time.monotonic()

    def _save(self, row, closed):
        """The round's files, in the order that keeps a crash recoverable: the state, last.npz, then diloco.json that names it."""
        from studio_core import atomic_json, now
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
            "format": FORMAT, "run": self.run, "round": self.next_round, "generation": self.generation, "state_file": self.state_file,
            "momentum_file": self.state_file, "digest": self.digest, "seed": self.cfg["seed"], "settings": self.settings.__dict__,
            "members": closed.members, "shard_deal": closed.deal, "step_time": self.step_time, "history": self.history,
            "finished": self.finished})
        if old and old != self.state_file:
            (self.directory / old).unlink(missing_ok=True)
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({k: row[k] for k in ("step", "loss", "val_loss", "tokens_per_second", "lr", "grad_norm")} | {"round": row["round"]}) + "\n")
        self.record.update({"step": step, "val_loss": row["val_loss"], "status": "completed" if self.finished else "training"})
        if row["loss"] is not None:
            self.record["loss"] = row["loss"]
        if self.finished:
            self.record["finishedAt"] = now()
        atomic_json(self.directory / "run.json", self.record)

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
            self.theta, self.momentum, self._params = {}, {}, (None, b"")


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
                other.release()
                del COORDINATORS[key]
        COORDINATORS[run] = coordinator


def prepare(job, services, directory, port):
    """App side, before the owner's job starts its worker: build the coordinator (tokenizing the data once), register it for the
    routes, and tell the worker where it is. A folder with diloco.json continues that run."""
    p = job.params
    config = dict(p["config"])
    settings = Settings.from_params(p, steps=config.get("steps"))
    examples = p["examples"] if "examples" in p else services.examples(p)
    coordinator = Coordinator.build(job.id, directory, config, examples, settings)
    register(job.id, coordinator)
    p.update(run=job.id, coordinator_url=f"http://127.0.0.1:{port}", settings=dict(settings.__dict__))
    return coordinator


# ---- the member -------------------------------------------------------------------------------------------------------

class Member:
    """One node's side of a run. It is the `distributed=` hook of studio_model.train: before the first step it loads theta of
    its round, after H steps (or at the round's deadline) it sends delta and waits for the next round, whose theta it loads."""

    def __init__(self, client, run, node, epoch, shard_dir, stop, clock=time.monotonic, emit=None, wait=LONG_POLL, patience=300., sleep=time.sleep):
        self.client, self.run, self.node, self.epoch, self.shard_dir, self.stop = client, run, node, epoch, Path(shard_dir), stop
        self.clock, self.emit, self.wait, self.patience, self.sleep = clock, emit or (lambda event: None), wait, patience, sleep
        self.base = f"/collective/{run}"
        self.train, self.val, self.sampler = Windows(), Windows(), random.Random(0)
        self.started, self.finished, self.fenced = False, False, False
        self.loss = None
        self.shards = {}

    # -- the train() hook
    @property
    def total_steps(self):
        return self.info["total_steps"]

    def rate_step(self, number):
        return self.round * self.info["inner_steps"] + (number - self.first_step)

    def due(self, step, trained_tokens):
        return step > self.first_step and (step - self.first_step >= self.info["inner_steps"] or self.clock() >= self.deadline)

    def sync(self, step, trained_tokens, model):
        if self.started:
            self._send(step, trained_tokens, model)
            if not (self.finished or self.fenced):
                self._enter(self.round + 1)
        else:
            self.started = True
        if self.finished or self.fenced:
            self.stop.set()
            return
        model.load_weights(self.theta)
        self.first_step, self.first_tokens, self.began, self.deadline = step, trained_tokens, self.clock(), self.clock() + self.deadline_s
        self.sampler.seed(sampler_seed(self.info["seed"], self.rank, self.round))

    # -- talking to the coordinator
    def _call(self, call):
        """Run a client call, waiting for the coordinator to come back if it cannot be reached (a restart) for up to `patience`."""
        from studio_peer import PeerError
        end = self.clock() + self.patience
        while True:
            try:
                return call()
            except PeerError as exc:
                if exc.status is not None or self.clock() >= end or self.stop.is_set():
                    raise
                self.sleep(2.)

    def join(self):
        self.info = self._call(lambda: self.client.post_json(f"{self.base}/join", {"node": self.node, "epoch": self.epoch}))
        from studio_modules.tokenizer.byte import Tokenizer
        self.tokenizer = Tokenizer.restore(self.info["tokenizer"])
        return self.info

    def _enter(self, want):
        """Wait until round `want` (or the current one, when it has moved on) is open for this member, then take its theta and shards."""
        from studio_peer import PeerError
        while not self.stop.is_set():
            try:
                view = self._call(lambda: self.client.get_json(f"{self.base}/rounds/{want}", {"node": self.node, "wait": self.wait},
                                                               timeout=self.wait + 15))
            except PeerError as exc:
                if exc.status == 409:
                    self.emit({"type": "info", "text": f"Left the run: {exc.detail}"})
                    self.fenced = True
                    return
                raise
            if view["state"] == "finished":
                self.finished = True
                return
            if view["state"] == "closed":
                want = view["current"]
            elif view["state"] == "pending" and not view["member"]:
                self.join()
                want = self.info["round"]
            elif view["state"] == "open":
                try:
                    self.theta = decode_arrays(self._call(lambda: self.client.get_bytes(f"{self.base}/params", {"round": want})))
                    self._fetch([int(s) for s in view["shards"]])
                except PeerError as exc:
                    if exc.status == 409:  # the round moved on while this member was downloading
                        want = self.info["round"]
                        continue
                    raise
                self.round, self.generation, self.rank, self.deadline_s = want, view["generation"], view["rank"], view["deadline_s"]
                return

    def _fetch(self, wanted):
        for shard in wanted:
            if shard not in self.shards:
                blob = self._call(lambda shard=shard: self.client.get_bytes(f"{self.base}/shards/{shard}"))
                self.shards[shard] = install_shard(self.shard_dir, shard, blob, self.info["vocab_size"])
        self.train.set([self.shards[s] for s in wanted])

    def _send(self, step, trained_tokens, model):
        from studio_peer import PeerError
        weights = model.state_arrays()
        delta = {name: self.theta[name] - weights[name].astype(np.float32) for name in self.theta}
        body = encode_arrays(delta, self.info["delta_dtype"])
        params = {"node": self.node, "generation": self.generation, "epoch": self.epoch, "tokens": trained_tokens - self.first_tokens,
                  "steps": step - self.first_step, "seconds": self.clock() - self.began}
        if self.loss is not None:
            params["loss"] = self.loss
        began = self.clock()
        try:
            self._call(lambda: self.client.post_bytes(f"{self.base}/rounds/{self.round}/delta", body, params, timeout=self.wait + 120))
            upload, trained = self.clock() - began, params["seconds"]
            self.emit({"type": "info", "text": f"Round {self.round}: sent {len(body)} bytes of {self.info['delta_dtype']} delta in {upload:.2f} s "
                                               f"({upload / max(1e-9, trained + upload):.0%} of the member's {trained + upload:.2f} s round)."})
        except PeerError as exc:
            if exc.status != 409:
                raise
            if "lease epoch" in exc.detail:
                self.emit({"type": "info", "text": f"Fenced: {exc.detail}"})
                self.fenced = True
            else:  # late, or the membership moved on: this round's work is dropped; take the new theta and carry on
                self.emit({"type": "info", "text": f"Round {self.round} was not accepted ({exc.detail}); continuing from the new weights."})

    def leave(self):
        from studio_peer import PeerError
        try:
            self.client.post_json(f"{self.base}/leave", {"node": self.node, "generation": getattr(self, "generation", 0), "epoch": self.epoch})
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
            self.join()
            self.shard_dir.mkdir(parents=True, exist_ok=True)
            self._enter(self.info["round"])
            if self.finished or self.fenced or self.stop.is_set():
                return None

            def watch(event):
                if event.get("type") == "metric" and event.get("step"):
                    self.loss = event.get("loss")
                emit(event)
            cfg = {**self.info["config"], "eval_every": 100000, "sample_every": 0, "steps": self.info["total_steps"]}
            self.first_step = 0
            studio_model.train(cfg, [], directory, emit=watch, stop=self.stop, distributed=self, telemetry_every=0)
            if not (self.finished or self.fenced):
                if self.stop.is_set():
                    return self.round
                raise RuntimeError("This member used all its steps before the run finished (it missed rounds). Start it again to rejoin.")
            return self.round
        finally:
            self.leave()
            self.train.set([])
            self.shards.clear()
            gc.collect()  # a memory map keeps its file open (on Windows, undeletable) until it is collected
            shutil.rmtree(self.shard_dir, ignore_errors=True)


def run_member(client, run, node, epoch, root, directory, emit, stop, **options):
    """Run one member against `client` (a studio_peer.PeerClient or anything with its methods); shards are cached under
    <root>/data/remote_shards/<run>/ and deleted at the end."""
    member = Member(client, run, node, epoch, Path(root) / "data" / "remote_shards" / run, stop, emit=emit, **options)
    return member.work(directory, emit)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m studio_diloco")
    sub = parser.add_subparsers(dest="command", required=True)
    member = sub.add_parser("member", help="join a run on another node's API and train until it finishes")
    member.add_argument("--coordinator", required=True, help="the owner node: a tailnet name or a URL, e.g. http://windows-box:8787")
    member.add_argument("--run", required=True, help="the run id: the pretrain_distributed job's id")
    member.add_argument("--node", default="", help="this member's name (default: the hostname)")
    member.add_argument("--epoch", type=int, default=0, help="lease epoch (default 0)")
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
                   Path(scratch) / "member", show, stop)
    except KeyboardInterrupt:
        stop.set()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
