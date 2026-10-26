"""A language-model training engine on PyTorch (CPU, CUDA or Apple MPS).

The transformer, tokenizer, checkpoint format and sampler are this repository's own; autograd supplies the gradients.

Data-pipeline config keys (stable API, also used by the curriculum trainer):
  tokenizer  "byte" | "character"   - byte is the default: 263 tokens, no unknown characters, so every
                                       dataset shares one vocabulary and the embedding stays tiny.
  context    32..2048                - tokens per window (default 256 so most chat turns fit prompt + answer).
  windowing  "anchored" | "sequential"
      anchored   - one window per assistant answer (long answers continue in further windows), each
                   left-filled with as much of the preceding conversation as fits, so an answer is always
                   trained with its prompt, exactly as at inference. Raw "text" rows use sequential chunks.
      sequential - legacy: independent context-sized chunks cut from token 0 (answers may lose their prompt).
                   Checkpoints written before this key existed resume with "sequential".
  objective  "assistant" | "all"      - which tokens are targets; every target token is trained exactly once.
Use samples_from(examples, tokenizer, cfg) -> WindowStore and batch_arrays(pairs) to feed train(). From STREAM_CHARS characters of
training text on, train() reads the same windows from token shards on disk instead (shard_windows, studio_shards), cut from encoded examples cached per tokenizer by their text, so a run tokenizes only text no earlier run did.
"""
from __future__ import annotations

import array
import bisect
import gc
import hashlib
import inspect
import json
import math
import os
import random
import shutil
import threading
import time
from pathlib import Path

for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_variable, "4")

import numpy as np
import torch

import studio_corpus
import studio_shards
import studio_dataset_snapshot
from studio_arch import parameter_breakdown, validate_spec
from studio_backend import DEVICES
from studio_memory import StateHome
from studio_core import ROOT, atomic_json, fingerprint, jsonl_rows, messages_of, now, replace_file
from studio_compile import CompiledModel, transplant
from studio_estimate import estimate
from studio_modules.mixer.attention import new_cache
from studio_modules.objective.ce import cross_entropy  # noqa: F401  (part of this module's API)
from studio_modules.ops import softmax as _softmax
from studio_modules.optimizer.adamw import AdamW as _AdamW, AdamWConfig
from studio_modules.tokenizer.byte import Tokenizer
from studio_planning import merge_duplicate_lineage
from studio_registry import for_slot, get as get_module
from studio_lora import FILE_PREFIX, PREFIX, attach_lora, lora_settings, merge_arrays, scale_of
from studio_slots import ASSISTANT, BOS, EOS, OFFSET, PAD, SYSTEM, UNK, USER
from studio_spec import config_from_spec, is_model_spec, model_spec, module_of, structure_key, validate_spec as validate_model_spec

DEFAULT_CONFIG = {
    "name": "My first transformer", "tokenizer": "byte", "width": 64,
    "layers": 2, "heads": 4, "context": 256, "expansion": 4, "dropout": 0.1,
    "batch_size": 4, "accumulation": 1, "steps": 300, "learning_rate": 0.001,
    "weight_decay": 0.01, "warmup": 20, "eval_every": 25, "eval_windows": 64, "validation": 0.1,
    "seed": 42, "objective": "assistant", "windowing": "anchored", "device": "auto", "architecture": None,
    "sample_every": 4, "sample_prompt": "The ",  # run page: a greedy sample of the prompt every N evaluation intervals (0 = off)
    "lora": None,  # {"rank", "alpha", "targets"}: train low-rank adapters on a frozen model instead of every weight (studio_lora)
}
# Why these defaults: a character vocabulary over mixed datasets reaches thousands of symbols (hundreds of
# thousands of embedding parameters, up to 8% unknown characters across datasets); bytes cost ~13% more
# tokens but remove both problems. Context 256 lets most question+answer pairs fit one window, and
# anchored windows with the assistant objective train answers only with their prompt in view.
ARCHITECTURE_KEYS = ("tokenizer", "width", "layers", "heads", "context", "expansion", "dropout", "architecture")
WINDOWINGS = ("anchored", "sequential")
MAX_STEPS = 20_000_000  # the most optimizer steps one run trains (a big text for a few passes needs a couple of million); a distributed run counts rounds × inner steps
EVAL_WINDOWS = 400  # legacy checkpoints and the distributed coordinator retain their original evaluation budget
PREFILL_CHUNK = 256  # a reply's first KV-cached prefill runs this many tokens per forward pass, and checks `stop` between passes
STREAM_CHARS = 16_000_000  # training text from this many characters on is tokenized into on-disk shards (about 48 MB of windows)


def validate_config(config):
    cfg = dict(DEFAULT_CONFIG, **{key: value for key, value in config.items() if key in DEFAULT_CONFIG})
    spec = cfg["architecture"]
    if isinstance(spec, str):
        # Form-driven callers may hand back "", "None" or a JSON string instead of a dict.
        spec = None if spec.strip() in ("", "None", "null") else json.loads(spec)
        cfg["architecture"] = spec
    if is_model_spec(spec):
        # A ModelSpec is authoritative for structure and for its optimizer and schedule settings; the display
        # fields are derived from it. Values passed alongside it (a curriculum stage's learning rate) still win.
        derived = config_from_spec(spec)
        spec = derived.pop("architecture")
        cfg.update({key: value for key, value in derived.items()
                    if key in DEFAULT_CONFIG and (key in ARCHITECTURE_KEYS[:6] or key not in config)}, architecture=spec)
    elif spec is not None:
        # A per-block spec is authoritative for structure; layers/heads/expansion are derived for display.
        spec = validate_spec(spec)
        first, embedding = spec["blocks"][0], spec["embedding"]
        cfg.update(architecture=spec, tokenizer=embedding["tokenizer"], width=embedding["width"],
                   context=embedding["context"], layers=len(spec["blocks"]),
                   heads=first["attention"]["heads"], expansion=first["mlp"]["expansion"])
        if "dropout" not in config:
            cfg["dropout"] = embedding["dropout"]
    model_spec_given = is_model_spec(spec)
    structural = [] if spec is not None else [("width", 16, 512), ("layers", 1, 12), ("heads", 1, 16), ("context", 32, 2048), ("expansion", 1, 8)]
    for key, low, high in structural + [("batch_size", 1, 128), ("accumulation", 1, 64),
                                        ("steps", 1, MAX_STEPS), ("warmup", 0, 100000), ("eval_every", 1, 100000), ("eval_windows", 1, 100000),
                                        ("seed", 0, 2147483647), ("sample_every", 0, 100000)]:
        cfg[key] = int(cfg[key])
        if not low <= cfg[key] <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
    for key, low, high in [("dropout", 0, .5), ("validation", 0, .4), ("learning_rate", 1e-7, _learning_rate_limit(spec)), ("weight_decay", 0, 1)]:
        cfg[key] = float(cfg[key])
        if not low <= cfg[key] <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
    if spec is None and cfg["width"] % cfg["heads"]:
        raise ValueError("Embedding width must be divisible by the attention head count.")
    if cfg["tokenizer"] not in ({m.short for m in for_slot("tokenizer")} if model_spec_given else {"byte", "character"}) or cfg["objective"] not in {"all", "assistant"}:
        raise ValueError("Invalid tokenizer or objective.")
    cfg["lora"] = lora_settings(cfg["lora"])
    if cfg["windowing"] not in WINDOWINGS:
        raise ValueError("Windowing must be anchored or sequential.")
    if cfg["device"] not in DEVICES:
        raise ValueError("Device must be auto, cpu, cuda, or mps.")
    return cfg


def _learning_rate_limit(spec):
    """The optimizer module's own upper bound for its peak rate (SGD allows up to 1, AdamW 0.1)."""
    optimizer = spec["train"]["optimizer"]["id"] if is_model_spec(spec) else "adamw"
    try:
        module = get_module("optimizer", optimizer)
    except ValueError:
        return .1
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(module.config)} if module.config else {}
    return fields["lr"].metadata.get("max", .1) if "lr" in fields else .1


def effective_spec(cfg):
    """The ModelSpec a validated config trains (older configs and architectures are converted)."""
    return model_spec(cfg)



def softmax(x, xp=np):
    return _softmax(x, xp)


class ScratchTransformer(CompiledModel):
    """The model a training config describes: its ModelSpec, compiled (see studio_compile).

    Older configs (layers/heads/expansion, or a per-block "groundwork-arch/1" architecture) are converted
    to the equivalent ModelSpec and produce exactly the parameter names of their checkpoints.
    """
    def __init__(self, config, vocabulary, probe=False):
        self.config = validate_config(config)
        cfg = self.config
        super().__init__(model_spec(cfg), vocabulary, seed=cfg["seed"], device=cfg["device"], batch_size=cfg["batch_size"], probe=probe)


class AdamW(_AdamW):
    """AdamW with the engine's default settings, attached to a model (the fine-tuning trainers use this directly)."""
    def __init__(self, model, weight_decay=.01):
        super().__init__(AdamWConfig(weight_decay=weight_decay))
        self.attach(model)


def build_optimizer(model, offload=False):
    """The optimizer and schedule the model's spec names, ready to step it. `model` may be a studio_lora.Trainable view.
    `offload` keeps the optimizer state in host memory (pinned on CUDA) and streams it through the device for each update."""
    train = model.spec["train"]
    optimizer, optimizer_cfg = module_of("optimizer", train["optimizer"])
    schedule, schedule_cfg = module_of("schedule", train["schedule"])
    home = StateHome.for_device(model.backend.device) if offload else None
    return optimizer.build(optimizer_cfg, model.dims, None).attach(model, home), schedule.build(schedule_cfg, model.dims, None)


def vocabulary_guess(cfg, vocab=263):
    """The vocabulary a config will have: fixed tokenizers know it, custom BPE and whole words are capped by their setting, characters are unknown."""
    spec = cfg.get("architecture")
    if is_model_spec(spec):
        entry = spec.get("tokenizer", {})
        try:
            module = get_module("tokenizer", entry.get("id", "byte"))
        except ValueError:
            return vocab
        config = module.parse(entry.get("config"))
        fixed = module.vocab(config)
        if fixed:
            return fixed
        if module.short in ("bpe_custom", "bpe_fast", "word"):
            return config.vocab_size
    return vocab


def parameter_estimate(cfg, vocab=263):
    vocab = vocabulary_guess(cfg, vocab)
    if is_model_spec(cfg.get("architecture")):
        return estimate(validate_model_spec(cfg["architecture"]), vocab)["total"]
    if cfg.get("architecture"):
        return parameter_breakdown(cfg["architecture"], vocab)["total"]
    d, f = cfg["width"], cfg["width"] * cfg["expansion"]
    return vocab * d + cfg["context"] * d + cfg["layers"] * (4 * d * d + 2 * d * f + f + 5 * d) + 2 * d


def _encode_arrays(example, tokenizer, assistant_only=False):
    """Token ids (int32), target mask (bool), role-token positions, their role ids, and assistant spans.

    A span is the half-open token range (content + EOS) of one assistant message; start - 1 holds its role token.
    """
    byte = tokenizer.kind == "byte"

    def content_ids(text):
        if byte:
            return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32) + OFFSET
        return np.asarray(tokenizer.encode(text), dtype=np.int32)

    eos = np.array([EOS], dtype=np.int32)
    parts, flags = [np.array([BOS], dtype=np.int32)], [np.zeros(1, dtype=bool)]
    positions, roles, spans, size = [], [], [], 1
    if example.get("text"):
        content = np.concatenate([content_ids(example["text"]), eos])
        parts.append(content)
        flags.append(np.ones(len(content), dtype=bool))
    else:
        role_ids = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}
        for message in messages_of(example):
            role = message["role"]
            if role not in role_ids:
                continue
            content = np.concatenate([content_ids(message["content"]), eos])
            positions.append(size)
            roles.append(role_ids[role])
            parts += [np.array([role_ids[role]], dtype=np.int32), content]
            flags += [np.full(1, not assistant_only, dtype=bool),
                      np.full(len(content), not assistant_only or role == "assistant", dtype=bool)]
            if role == "assistant":
                spans.append((size + 1, size + 1 + len(content)))
            size += 1 + len(content)
    return np.concatenate(parts), np.concatenate(flags), positions, roles, spans


def encode_example(example, tokenizer, assistant_only=False):
    """Token list and per-token "is a training target" flags for one example (BOS first, role token per turn)."""
    tokens, mask, _, _, _ = _encode_arrays(example, tokenizer, assistant_only)
    return tokens.tolist(), mask.tolist()


class WindowStore:
    """Compact, read-only sequence of training windows.

    All windows live in one flat token buffer (uint16; int32 for huge vocabularies), one flat target-flag
    buffer and an offsets array instead of millions of Python lists. Window i is the segment
    tokens[offsets[i]:offsets[i+1]]: inputs are its first n-1 tokens, targets its last n-1 tokens where the flag
    is set (else -100). Indexing returns an (inputs, targets) int32 pair; slices return a list of pairs, so it
    is a drop-in sequence for batch_arrays().
    """

    def __init__(self, tokens, flags, offsets):
        self.tokens, self.flags, self.offsets = tokens, flags, offsets

    def __len__(self):
        return len(self.offsets) - 1

    def _pair(self, i):
        low, high = int(self.offsets[i]), int(self.offsets[i + 1])
        segment = self.tokens[low:high]
        keep = self.flags[low - i:high - i - 1].astype(bool)
        segment = segment.astype(np.int32)
        return segment[:-1], np.where(keep, segment[1:], -100).astype(np.int32)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._pair(i) for i in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Window index out of range.")
        return self._pair(index)

    def __iter__(self):
        return (self._pair(i) for i in range(len(self)))

    @property
    def nbytes(self):
        return int(self.tokens.nbytes + self.flags.nbytes + self.offsets.nbytes)

    def describe(self):
        """Window count, tokens, target tokens, share of windows starting at the example start, memory."""
        count = len(self)
        starts = (self.tokens[self.offsets[:-1]] == BOS).mean() if count else 0.
        return {"windows": count, "tokens": int(len(self.tokens) - count), "targets": int(self.flags.sum()),
                "from_start": float(starts), "megabytes": self.nbytes / 1024 ** 2}


class _WindowBuilder:
    def __init__(self, vocab_size):
        self.dtype, self.tokens = (np.uint16, array.array("H")) if vocab_size <= 65535 else (np.int32, array.array("i"))
        self.flags, self.offsets = array.array("B"), array.array("q", [0])

    def add(self, segment, flags):
        self.tokens.frombytes(np.ascontiguousarray(segment, dtype=self.dtype).tobytes())
        self.flags.frombytes(np.ascontiguousarray(flags, dtype=np.uint8).tobytes())
        self.offsets.append(len(self.tokens))

    def finish(self):
        return WindowStore(np.frombuffer(self.tokens, dtype=self.dtype), np.frombuffer(self.flags, dtype=np.uint8),
                           np.frombuffer(self.offsets, dtype=np.int64))


def _sequential_windows(builder, tokens, mask, context):
    for start in range(0, len(tokens) - 1, context):
        segment = tokens[start:start + context + 1]
        flags = mask[start + 1:start + len(segment)]
        if flags.any():
            builder.add(segment, flags)


def _anchored_windows(builder, tokens, mask, positions, roles, units, context):
    """One window per piece of every target unit; each ends at its piece and is left-filled with context."""
    full, stride = context + 1, max(1, context // 2)
    for low, high in units:
        c = low
        while c < high:
            # Prompt start still fits: anchor at the example start and take as many tokens as fit; after that slide in
            # stride-sized pieces, spending the rest of the window on the text just before the piece.
            d = min(high, full) if c < full else min(high, c + stride)
            s = max(0, d - full)
            segment = tokens[s:d].copy()
            if s > 0:
                owner = bisect.bisect_right(positions, s) - 1
                if owner >= 0 and positions[owner] != s:
                    segment[0] = roles[owner]  # cut mid-message: the first token becomes that message's role token
            flags = np.zeros(d - s - 1, dtype=bool)
            flags[c - s - 1:] = mask[c:d]
            if flags.any():
                builder.add(segment, flags)
            c = d


def cut_windows(examples, tokenizer, cfg, builder, encode=None):
    """Cut examples into training windows and hand each to builder.add(segment, flags): the cut samples_from() and the shards share.
    `encode(example)` returns what _encode_arrays does; left out, the example is tokenized here (the shards pass their cached pieces)."""
    context = cfg["context"]
    anchored = cfg.get("windowing", "sequential") == "anchored"
    assistant_only = cfg["objective"] == "assistant"
    encode = encode or (lambda example: _encode_arrays(example, tokenizer, assistant_only))
    for example in examples:
        tokens, mask, positions, roles, spans = encode(example)
        if not anchored or not spans:
            _sequential_windows(builder, tokens, mask, context)
            continue
        if assistant_only:
            units = spans
        else:
            ends = [high for _, high in spans]
            units = list(zip([1] + ends[:-1], ends)) + ([(ends[-1], len(tokens))] if ends[-1] < len(tokens) else [])
        _anchored_windows(builder, tokens, mask, positions, roles, units, context)


def samples_from(examples, tokenizer, cfg):
    """Cut examples into training windows: a WindowStore of (inputs, targets) pairs (target -100 = ignored).

    cfg["windowing"] (a missing key means legacy "sequential"), cfg["context"] and cfg["objective"] decide the cut.
    Anchored: each assistant answer is the target of a window that ends with it (long answers continue in
    stride-sized pieces); under objective "all" the text before the first answer is targeted by the first window,
    so every target token is trained exactly once per example. Text rows and answer-less rows are chunked
    sequentially.
    """
    builder = _WindowBuilder(tokenizer.vocab_size)
    cut_windows(examples, tokenizer, cfg, builder)
    return builder.finish()


class MappedWindows(WindowStore):
    """The windows of a token-shard folder (studio_shards): a WindowStore whose buffers are memory maps, so only the windows a
    batch touches are read from disk. Counts come from the folder's index, not from a scan."""

    def __init__(self, tokens, flags, offsets, entry):
        super().__init__(tokens, flags, offsets)
        self.entry = entry

    def close(self):
        """Close the memory maps, so the folder can be removed or rebuilt on Windows (the windows must not be read afterwards)."""
        studio_shards.release_maps(self.tokens, self.flags, self.offsets)

    def describe(self):
        count = self.entry["windows"]
        return {"windows": count, "tokens": self.entry["tokens"] - count, "targets": self.entry["targets"],
                "from_start": self.entry["from_start"] / count if count else 0., "megabytes": self.nbytes / 1024 ** 2}


def characters_in(examples):
    """Characters of text the examples hold: what the size of their token windows follows (a byte token is about one)."""
    return sum(len(e["text"]) if e.get("text") else sum(len(m["content"]) for m in messages_of(e)) for e in examples)


def _tokenizer_hash(tokenizer):
    return hashlib.sha256(json.dumps(tokenizer.serialize(), sort_keys=True).encode()).hexdigest()


def shard_key(tokenizer, cfg, split_hash):
    """The folder name under a shard root for this data split (split_hash), tokenizer and cut."""
    return studio_shards.key_for(split_hash, _tokenizer_hash(tokenizer), cfg["windowing"], cfg["context"], cfg["objective"])


ENCODE_WEIGHT = 20  # progress counts an encoded character this many times a character cut from the cache: a slow tokenizer takes about 20x as long to encode as the cut takes (estimate)


class _ListRows:
    """A list of examples seen as shard_windows reads rows: meta() gives each row's (fingerprint, characters), rows[i] the example."""

    def __init__(self, examples, fingerprint_of):
        self.examples = examples if isinstance(examples, (list, tuple)) else list(examples)  # (a caller may count the rows as they are read)
        self.keys = [fingerprint_of(e) for e in self.examples]
        self.weights = [characters_in([e]) for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, position):
        return self.examples[position]

    def meta(self):
        return zip(self.keys, self.weights)


def shard_windows(training, validation, tokenizer, cfg, split_hash, root, progress=None, fingerprint_of=None):
    """(train windows, validation windows, reused) read from token shards under root, tokenizing into them first when this data,
    tokenizer and windowing have none yet. The key is the data split (split_hash) with the tokenizer and the cut.

    The shards of a split nobody has seen are cut from the TokenCache beside them (tokens-<key> under root): examples are cached
    by fingerprint under the tokenizer, so only the examples no earlier run encoded are tokenized, however the split or the data
    around them changed. The windows are the ones samples_from() cuts. progress(done, total) is told as the work goes (it is
    weighted by characters, encoding counting ENCODE_WEIGHT times); fingerprint_of(example) is the caller's fingerprint when it has one.

    training and validation are lists of examples, or rows of a corpus on disk (studio_corpus.Rows: meta() and indexing, no list):
    then no example is held beyond the one being encoded."""
    say = progress or (lambda done, total: None)
    key = shard_key(tokenizer, cfg, split_hash)
    found = studio_shards.load(root, key, tokenizer.vocab_size)
    reused = found is not None
    if reused:
        say(1, 1)
    else:
        def fill(writers):
            fp = fingerprint_of or fingerprint
            assistant_only = cfg["objective"] == "assistant"
            cache = studio_shards.TokenCache(root, _tokenizer_hash(tokenizer), tokenizer.vocab_size, assistant_only)
            try:
                views = {name: rows if hasattr(rows, "meta") else _ListRows(rows, fp) for name, rows in (("train", training), ("val", validation))}
                total, counted = 0, set()
                for view in views.values():
                    for fingerprint_, weight in view.meta():
                        total += weight
                        if fingerprint_ not in cache and fingerprint_ not in counted:
                            counted.add(fingerprint_)
                            total += weight * ENCODE_WEIGHT
                counted.clear()
                done = [0]

                def advance(weight):
                    done[0] += weight
                    say(done[0], total)
                say(0, total)

                def missing(view):  # the rows no earlier run encoded, read one at a time
                    for position, (fingerprint_, weight) in enumerate(view.meta()):
                        if fingerprint_ not in cache:
                            yield fingerprint_, view[position], weight * ENCODE_WEIGHT
                cache.add((item for view in views.values() for item in missing(view)), lambda e: _encode_arrays(e, tokenizer, assistant_only), advance)
                cache.touch()
                for name, view in views.items():
                    current = [0]

                    def fingerprints_of(view=view, current=current):  # cut_windows is handed each row's fingerprint, and cached() its encoding
                        for fingerprint_, weight in view.meta():
                            current[0] = weight
                            yield fingerprint_

                    def cached(fingerprint_, current=current):
                        arrays = cache.get(fingerprint_)
                        advance(current[0])
                        return arrays
                    cut_windows(fingerprints_of(), tokenizer, cfg, writers[name], cached)
            finally:
                cache.close()
        found = studio_shards.build(root, key, tokenizer.vocab_size, fill)
    return MappedWindows(*found["train"]), MappedWindows(*found["val"]), reused


def batch_arrays(samples):
    """Pad (inputs, targets) pairs - lists or arrays - into int64 matrices; padding targets are -100."""
    width = max(len(item[0]) for item in samples)
    x = np.full((len(samples), width), PAD, dtype=np.int64)
    y = np.full((len(samples), width), -100, dtype=np.int64)
    for i, (inputs, targets) in enumerate(samples):
        x[i, :len(inputs)] = inputs
        y[i, :len(targets)] = targets
    return x, y


def cuda_stage():
    """stage() for BatchPrefetcher on CUDA: copy each batch from pinned host memory on a side stream, without blocking the host.

    Returns (x, y, ready); ready() makes the compute stream wait for this batch's copy only (an event, so a later batch's copy
    is not waited on) and tells the allocator the tensors are used there."""
    stream = torch.cuda.Stream()

    def stage(x, y):
        with torch.cuda.stream(stream):
            x, y = (torch.from_numpy(a).pin_memory().to("cuda", non_blocking=True) for a in (x, y))
            copied = torch.cuda.Event()
            copied.record(stream)

        def ready():
            compute = torch.cuda.current_stream()
            compute.wait_event(copied)
            x.record_stream(compute)
            y.record_stream(compute)
        return x, y, ready
    return stage


def batch_stage(device):
    """The stage BatchPrefetcher gets: only CUDA stages and prefetches; CPU and MPS (unified memory, nothing to pin) take the plain path."""
    return cuda_stage() if device == "cuda" else None


class BatchPrefetcher:
    """Training batches from `draw()`, which reads `rng`; next() returns (x, y, target_tokens).

    Without `stage` (CPU and MPS) next() just draws and returns the arrays. With `stage(x, y) -> (x, y, ready)` the next batch is
    drawn and staged (on CUDA: a pinned, non-blocking copy) before the current one is returned, so its copy overlaps the step.
    The draws come in the same order either way, so a seed gives the same batches. A batch drawn ahead has moved `rng`, so:
    discard() puts those draws back (an OOM changes the shape the next draw reads) and state() is the rng as the consumed
    batches left it, which is what a checkpoint must store."""

    def __init__(self, rng, draw, stage=None):
        self.rng, self.draw, self.stage, self.ahead = rng, draw, stage, None  # ahead: (rng state before the draw, staged batch)

    def _make(self):
        before = self.rng.getstate()
        x, y = self.draw()
        tokens = int((y != -100).sum())
        return before, (*self.stage(x, y), tokens)

    def next(self):
        if self.stage is None:
            x, y = self.draw()
            return x, y, int((y != -100).sum())
        x, y, ready, tokens = (self.ahead or self._make())[1]
        self.ahead = self._make()
        if ready is not None:
            ready()
        return x, y, tokens

    def discard(self):
        if self.ahead:
            self.rng.setstate(self.ahead[0])
            self.ahead = None

    def state(self):
        return self.ahead[0] if self.ahead else self.rng.getstate()


def load_checkpoint(path, weights_only=False, merged=True):
    """The checkpoint's metadata and arrays; weights_only skips the optimizer state (m/v), which inference never reads.

    A LoRA checkpoint (adapters stored under lora::) comes back as one plain merged model unless merged=False, which keeps the
    frozen base and returns the adapters too (as "lora.<weight>.a" / ".b"), the shape a resumed LoRA run needs."""
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("architecture") != "groundwork_numpy_decoder" or metadata.get("format_version") != 1:
            raise ValueError("Choose a Groundwork NumPy checkpoint (.npz).")
        metadata["model"] = {key.removeprefix("weight::"): archive[key].copy() for key in archive.files if key.startswith("weight::")}
        adapters = {key[len(FILE_PREFIX):]: archive[key].copy() for key in archive.files if key.startswith(FILE_PREFIX)}
        if adapters and merged:
            settings = lora_settings(metadata["config"].get("lora"))
            if settings is None:
                raise ValueError("This checkpoint has LoRA adapters but its settings do not say how to apply them.")
            metadata["model"] = merge_arrays(metadata["model"], adapters, scale_of(settings))
        elif adapters:
            metadata["model"].update({PREFIX + key: value for key, value in adapters.items()})
        # optimizer state arrays are stored as adam_<state>::<parameter>, whatever the optimizer (the prefix is historical)
        metadata["optimizer_state"] = {}
        for key in () if weights_only else archive.files:
            if key.startswith("adam_"):
                state, name = key[5:].split("::", 1)
                metadata["optimizer_state"].setdefault(state, {})[name] = archive[key].copy()
        metadata["m"], metadata["v"] = metadata["optimizer_state"].get("m", {}), metadata["optimizer_state"].get("v", {})
    if not metadata.get("history"):
        metadata["history"] = read_metrics(Path(path).parent / "metrics.jsonl")
    return metadata


def checkpoint_device(path):
    """The device kind ("cuda", "mps" or "cpu") that wrote a checkpoint, or None when it does not say or cannot be read.
    Reads only the metadata entry, not the weights."""
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            device = json.loads(str(archive["metadata"].item())).get("device")
    except (OSError, ValueError, KeyError):
        return None
    return device if device in DEVICES and device != "auto" else None


def read_metrics(path):
    """The rows of a run's metrics.jsonl (a partial last line from a crash is skipped). A run in progress is read only past what was read before."""
    return jsonl_rows(path)


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def row_identities(example):
    """The (kind, id) pairs that tie a row to its source row, group and the rows it was made from (what partition_examples keeps together)."""
    provenance = example.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    identities = [("row", str(example["id"]))] if example.get("id") else []
    if example.get("groupId"):
        identities.append(("group", str(example["groupId"])))
    for key, kind in (("source_ids", "row"), ("group_ids", "group")):
        values = provenance.get(key, [])
        if isinstance(values, (list, tuple)):
            identities.extend((kind, str(value)) for value in values if isinstance(value, (str, int)) and value)
    return identities


def partition_indices(count, identities_of, fraction, sampler, grouped=True):
    """(training indices, validation indices, report) for `count` rows; identities_of(index) is that row's row_identities().
    partition_examples is this over a list; a corpus on disk (studio_corpus) calls it with only the identities in memory."""
    indices = list(range(count))
    sampler.shuffle(indices)
    target = min(count - 1, max(1, round(count * fraction))) if fraction and count > 1 else 0
    parents = list(range(count))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    if grouped:
        anchors = {}
        for index in range(count):
            for identity in identities_of(index):
                if identity in anchors:
                    left, right = root(index), root(anchors[identity])
                    parents[max(left, right)] = min(left, right)
                else:
                    anchors[identity] = index
    groups = {}
    for index in indices:
        groups.setdefault(root(index), []).append(index)
    note = ""
    if len(groups) == count:
        validation_ids = set(indices[:target])
    elif target and len(groups) > 1:
        selected_groups = []
        taken = 0
        for key, members in groups.items():
            if len(selected_groups) == len(groups) - 1:
                break
            if abs(target - taken - len(members)) < abs(target - taken):
                selected_groups.append(key)
                taken += len(members)
        if not selected_groups:
            selected_groups = [min(groups, key=lambda key: len(groups[key]))]
        validation_ids = {index for key in selected_groups for index in groups[key]}
        if fraction <= .5 and len(validation_ids) > count / 2:
            validation_ids = set(indices) - validation_ids
        note = "Validation uses whole source groups; its size may differ from the requested fraction."
    else:
        validation_ids = set()
        if target:
            note = "Validation is disabled: all examples share one source lineage. Add independent sources for a held-out evaluation."
    training = [index for index in indices if index not in validation_ids]
    validation = [index for index in indices if index in validation_ids]
    report = {"strategy": "lineage-v1" if grouped else "rows-v1", "groups": len(groups),
              "training_groups": len({root(index) for index in indices if index not in validation_ids}),
              "validation_groups": len({root(index) for index in validation_ids}), "note": note}
    return training, validation, report


def partition_examples(examples, fraction, sampler, grouped=True):
    """Keep source conversations, documents, and their synthetic descendants together.

    A union-find index is linear in rows plus lineage references; it never builds
    pairwise similarity graphs. Ordinary examples retain the original seeded
    row split. Grouped splits approximate the desired fraction with whole groups.
    """
    training, validation, report = partition_indices(len(examples), lambda index: row_identities(examples[index]), fraction, sampler, grouped)
    return [examples[index] for index in training], [examples[index] for index in validation], report


def _piece(tokenizer, index):
    """The text (or bytes) behind one vocabulary id, for matching rows across tokenizers; None for special tokens."""
    if index < OFFSET:
        return ("special", index)
    if tokenizer.kind == "character":
        return tokenizer.characters[index - OFFSET]
    pieces = getattr(tokenizer, "pieces", None)
    if pieces is not None:
        return pieces[index - OFFSET]
    decoder = getattr(tokenizer, "decoder", None)
    if decoder is not None:
        return decoder.get(index - OFFSET)
    return tokenizer.decode([index])


def remap_vocabulary(old, old_tokenizer, tokenizer, fresh):
    """Rows (or columns) of vocabulary-shaped tensors re-indexed by piece when the tokenizer changed.

    `fresh` maps the parameter name to the new model's freshly initialised tensor (NumPy), so pieces the old
    tokenizer never had keep their fresh values. Returns the remapped dict (other tensors unchanged).
    """
    if old_tokenizer.serialize() == tokenizer.serialize():
        return old
    old_rows = {}
    for i in range(old_tokenizer.vocab_size):
        old_rows.setdefault(_piece(old_tokenizer, i), i)
    mapping = [(i, old_rows.get(_piece(tokenizer, i))) for i in range(tokenizer.vocab_size)]
    old_vocab = old_tokenizer.vocab_size
    out = dict(old)
    for name, value in old.items():
        value = np.asarray(value)
        if name not in fresh:
            continue
        target = np.array(fresh[name], copy=True)
        if value.ndim >= 1 and value.shape[0] == old_vocab and target.shape[0] == tokenizer.vocab_size and value.shape[1:] == target.shape[1:]:
            for new_index, old_index in mapping:
                if old_index is not None:
                    target[new_index] = value[old_index]
            out[name] = target
        elif value.ndim >= 1 and value.shape[-1] == old_vocab and target.shape[-1] == tokenizer.vocab_size and value.shape[:-1] == target.shape[:-1]:
            for new_index, old_index in mapping:
                if old_index is not None:
                    target[..., new_index] = value[..., old_index]
            out[name] = target
    return out


def _initialize_from(model, tokenizer, path):
    """Transplant weights from a checkpoint into the model's (possibly different) architecture."""
    data = load_checkpoint(path)
    old_tokenizer = Tokenizer.restore(data["tokenizer"])
    old = remap_vocabulary(data["model"], old_tokenizer, tokenizer, model.state_arrays())
    return transplant(old, model)


def _prepare_reporter(emit, stop, phase, low, high):
    """progress(done, total) that tells the queue a `prepare` event (the phase, and a percent of the whole preparation between low and
    high), once for each percent, and ends the preparation when stop is set. A total of None means the size is not known (percent low)."""
    reported = [-1]

    def progress(done, total):
        if stop.is_set():
            raise InterruptedError("Training preparation cancelled.")
        fraction = 0. if total is None else 1. if done >= total else done / total
        percent = high if fraction >= 1 else int(low + (high - low) * fraction)
        if percent != reported[0]:
            reported[0] = percent
            emit({"type": "prepare", "phase": phase, "percent": percent, "done": int(done), "of": int(total or 0)})
    return progress


def train(config, examples, output_dir, emit=None, stop=None, pause=None, resume=None, init_from=None, telemetry_every=5,
          snapshot_every=0, snapshot_keep=20, token_budget=None, wall_seconds=None, shard_dir=None, stream_chars=STREAM_CHARS,
          schedule_window=None, offload_optimizer=False, distributed=None, rebase=False, snapshot_dir=None):
    """Train a model. schedule_window=(steps_before, plan_total) makes this run one slice of a longer learning-rate schedule:
    step n of this run is step steps_before + n of plan_total, so a staged plan keeps one warmup and one decay.
    Left None (the default) the schedule spans this run alone.
    offload_optimizer keeps the optimizer state in host memory and streams it through the device for each update: slower,
    the same result, and a model whose state does not fit the GPU can train (studio_memory.offload_wanted says when).

    rebase=True (with resume) continues a checkpoint on different data: the weights, tokenizer, optimizer state and step carry over,
    and cfg["steps"] is the number of steps to add. The dataset, split, seed, learning rate and batch size may differ from the
    checkpoint's; the sampler, the best score and its best.npz start fresh.

    examples is a list, or an iterator for a big text (the worker passes one): then the rows are read one at a time into a spool on disk
    (studio_corpus), reading stops once the unique text covers token_budget, and memory holds a few hundred bytes per row, not the text.
    Under stream_chars characters the rows are gathered into a list and everything runs as it does for a list.

    The run folder keeps the deduplicated examples as dataset.jsonl, which a resume rebuilds the split from. The bytes are stored
    once under their hash in snapshot_dir (default: .snapshots in the `runs` folder holding the run; elsewhere none) and the folder holds a hard link to them,
    so a retry, a restart or a resume of the same data writes nothing again (studio_dataset_snapshot.py).

    distributed is the hook studio_diloco's members train through (DiLoCo, docs/distributed.md); every other caller leaves it None.
    It is any object with: `tokenizer`; `train` and `val`, window sequences read at every batch, so it may swap what they
    hold; `sampler`, the random.Random that picks the windows; `total_steps`, the length of the whole schedule;
    `rate_step(n)`, the schedule position of this run's step n; `due(step, trained_tokens)`, asked after every optimizer step;
    and `sync(step, trained_tokens, model)`, called once before the first step and then whenever due() says so. sync may
    replace the weights in model.p in place (model.load_weights); the optimizer state is kept. With it, examples, resume,
    init_from and LoRA are not used: the data is the hook's."""
    emit = emit or (lambda event: None)
    stop, pause = stop or threading.Event(), pause or threading.Event()
    cfg = validate_config(config)
    if token_budget is not None and token_budget < 1:
        raise ValueError("Token budget must be positive.")
    if wall_seconds is not None and wall_seconds <= 0:
        raise ValueError("Wall-clock budget must be positive.")
    if resume and init_from:
        raise ValueError("Resume and initialize-from cannot be combined.")
    if rebase and not resume:
        raise ValueError("Rebase continues a checkpoint, so it needs resume.")
    if cfg["lora"] and not (resume or init_from):
        raise ValueError("LoRA fine-tunes a model that already exists, so start it from a version you have already trained or imported.")
    if schedule_window and schedule_window[0] + cfg["steps"] > schedule_window[1]:
        raise ValueError("The schedule window must cover this run's steps.")
    directory = Path(output_dir)
    if any((directory / name).exists() for name in ("config.json", "dataset.jsonl", "last.npz", "best.npz")):
        raise ValueError("Choose a new output folder to preserve the existing run.")
    directory.mkdir(parents=True, exist_ok=True)
    if distributed is not None:
        if resume or init_from or cfg["lora"]:
            raise ValueError("A distributed member neither resumes, initializes from a checkpoint, nor trains LoRA adapters.")
        tokenizer, train_samples, val_samples, sampler, checkpoint = distributed.tokenizer, distributed.train, distributed.val, distributed.sampler, None
        examples, training, validation, digest, split_hash = [], [], [], "", ""
        split, streamed, reused = {"strategy": "federated", "note": ""}, False, False
        corpus = None
    else:
        corpus = None
        if not isinstance(examples, (list, tuple)):  # an iterator: a big text is read as a stream (studio_corpus), never held whole
            reading = _prepare_reporter(emit, stop, "reading", 0, 40)
            examples, corpus = studio_corpus.collect(
                examples, directory, stream_chars, None if resume and not rebase else studio_corpus.budget_chars(token_budget), stop, reading)
        if corpus is None:
            unique, copied = {}, set()
            for example in examples:
                if stop.is_set():
                    raise InterruptedError("Training preparation cancelled.")
                content_hash = fingerprint(example)
                if content_hash in unique:
                    if content_hash not in copied:  # the caller's row is copied only when a duplicate's lineage is merged into it
                        unique[content_hash] = dict(unique[content_hash])
                        copied.add(content_hash)
                    merge_duplicate_lineage(unique[content_hash], example)
                else:
                    unique[content_hash] = example
            examples = list(unique.values())
            if not examples:
                raise ValueError("The dataset is empty.")
            fingerprints = {id(e): h for h, e in unique.items()}  # each row is fingerprinted once, here
            digest = hashlib.sha256("".join(unique).encode()).hexdigest()
            del unique, copied
        else:
            if not len(corpus):
                corpus.discard()
                raise ValueError("The dataset is empty.")
            digest, fingerprints = corpus.digest.hexdigest(), None
        sampler = random.Random(cfg["seed"])
        checkpoint = load_checkpoint(resume) if resume else None
        if checkpoint and checkpoint["config"].get("lora"):  # a LoRA run resumes from the frozen base plus its adapters, not the merged weights
            checkpoint["model"] = load_checkpoint(resume, weights_only=True, merged=False)["model"]
        # Older checkpoints must resume with their original row-based partition.
        grouped = not checkpoint or rebase or checkpoint.get("split_strategy") == "lineage-v1"
        if corpus is None:
            training, validation, split = partition_examples(examples, cfg["validation"], sampler, grouped)
            split_hash = hashlib.sha256(("".join(fingerprints[id(e)] for e in training) + "|" +
                                         "".join(fingerprints[id(e)] for e in validation)).encode()).hexdigest()
        else:
            training, validation, split, split_hash = corpus.partition(cfg["validation"], sampler, grouped)
            examples = corpus.snapshot_rows()  # the run's dataset.jsonl is written from the spool, one row at a time
        if checkpoint:
            if not rebase and checkpoint["data_hash"] != digest:
                raise ValueError("Resume requires the original dataset snapshot.")
            if not rebase and checkpoint.get("split_hash") and checkpoint["split_hash"] != split_hash:
                raise ValueError("Resume requires the original source lineage and validation split.")
            # Old checkpoints have no "architecture": validate_config derives the equivalent uniform spec.
            previous = validate_config(checkpoint["config"])
            if not rebase:
                # Best loss and history belong to this fixed subset. A resume must not compare a
                # cheaper/new subset against them; pre-budget checkpoints always evaluated 400.
                cfg["eval_windows"] = previous["eval_windows"] if "eval_windows" in checkpoint["config"] else EVAL_WINDOWS
            if "windowing" not in checkpoint["config"]:
                # Checkpoints from before windowing existed were trained with independent sequential chunks.
                previous["windowing"] = cfg["windowing"] = "sequential"
            keys = tuple(k for k in ARCHITECTURE_KEYS if k != "architecture") + (
                "objective", "windowing", "weight_decay", "warmup", "lora",
            ) + (() if rebase else ("seed", "validation", "batch_size", "accumulation", "learning_rate"))
            current_spec, previous_spec = effective_spec(cfg), effective_spec(previous)
            recipe = ("objective", "optimizer", "schedule")
            if rebase:  # the optimizer entry carries the learning rate, which a rebased round may change
                for spec in (current_spec, previous_spec):
                    spec["train"]["optimizer"] = {**spec["train"]["optimizer"], "config": {k: v for k, v in spec["train"]["optimizer"]["config"].items() if k != "lr"}}
            if (any(cfg[k] != previous[k] for k in keys)
                    or structure_key(current_spec) != structure_key(previous_spec)
                    or any(current_spec["train"][key] != previous_spec["train"][key] for key in recipe)):
                raise ValueError("Resume must preserve architecture, tokenizer, split, seed, objective, optimizer, schedule and batch settings.")
            tokenizer = Tokenizer.restore(checkpoint["tokenizer"])
        else:
            module, module_cfg = module_of("tokenizer", effective_spec(cfg)["tokenizer"])
            fitting = _prepare_reporter(emit, stop, "tokenizer", 40, 45) if corpus is not None else None
            if fitting:
                fitting(0, 1)
            tokenizer = module.fit(module_cfg, training, fitting) if fitting and "progress" in inspect.signature(module.fit).parameters else module.fit(module_cfg, training)
        streamed = reused = False
        if corpus is not None or characters_in(training) + characters_in(validation) >= stream_chars:
            # Big data: tokenize once into shards on disk and read windows from there, so the windows take no memory and the next run
            # on the same data and tokenizer starts at once. Under the threshold the windows are built in memory as always.
            preparing = _prepare_reporter(emit, stop, "tokenizing", 45 if corpus is not None else 0, 100)
            train_samples, val_samples, reused = shard_windows(training, validation, tokenizer, cfg, split_hash, shard_dir or ROOT / "data" / "shards",
                                                               preparing, None if corpus is not None else lambda e: fingerprints[id(e)])
            streamed = True
        else:
            train_samples, val_samples = samples_from(training, tokenizer, cfg), samples_from(validation, tokenizer, cfg)
        del fingerprints
    if not train_samples:
        raise ValueError("No trainable target tokens were found.")
    model = ScratchTransformer(cfg, tokenizer.vocab_size, probe=True)  # "auto" times real steps on each device
    transplanted = _initialize_from(model, tokenizer, init_from) if init_from else None
    trainable = attach_lora(model, cfg["lora"]) if cfg["lora"] else model  # LoRA: the base is frozen; the optimizer sees only the adapters
    optimizer, schedule = build_optimizer(trainable, offload=offload_optimizer)
    step, best, history = 0, float("inf"), []
    if checkpoint:
        model.load_weights(checkpoint["model"])
        for state in optimizer.states:
            saved = checkpoint["optimizer_state"].get(state, {})
            for key, target in getattr(optimizer, state).items():
                value = saved.get(key)
                if value is None or tuple(np.shape(value)) != tuple(target.shape) or not np.isfinite(value).all():
                    raise ValueError(f"Checkpoint optimizer state {state} for {key} is missing or invalid.")
                target.copy_(torch.as_tensor(np.asarray(value), dtype=target.dtype, device=target.device))
        previous_best = Path(resume).parent / "best.npz"
        if previous_best.is_file() and not rebase:
            shutil.copy2(previous_best, directory / "best.npz")  # the best so far carries over until this run beats it
        step = checkpoint["step"]
        best = checkpoint["best"] if checkpoint["best"] is not None and not rebase else float("inf")
        optimizer.steps = checkpoint["optimizer_steps"]
        history = checkpoint.get("history", [])
        if rebase:
            cfg["steps"] += step  # the steps to add, on top of the ones already taken
        else:
            sampler.setstate(_tuples(checkpoint["sampler_state"]))
            model.rng.bit_generator.state = checkpoint["model_rng"]
        if cfg["steps"] <= step:
            raise ValueError("Total steps must exceed the checkpoint step.")
    record = {"name": cfg["name"], "createdAt": now(), "status": "training", "step": step, "steps": cfg["steps"],
              "parameters": model.parameter_count, "vocabulary": tokenizer.vocab_size,
              "backend": "torch", "device": model.backend.device,
              "device_name": model.backend.name, "device_probe": model.backend.probe,
              "train_examples": len(training), "validation_examples": len(validation), "resumed_from": str(resume) if resume else None,
              "initialized_from": str(init_from) if init_from else None,
              "best_loss": best if math.isfinite(best) else None, "split": split}
    atomic_json(directory / "config.json", cfg)
    atomic_json(directory / "spec.json", model.spec)
    atomic_json(directory / "tokenizer.json", tokenizer.serialize())
    if distributed is None:
        shared = snapshot_dir if snapshot_dir is not None else studio_dataset_snapshot.folder_for(directory)
        emit({"type": "info", "text": "Saving the dataset snapshot"})
        try:
            snapshot, created, _ = studio_dataset_snapshot.adopt(examples, shared, directory / "dataset.jsonl")
        finally:
            if corpus is not None:
                corpus.discard()
        if shared is not None:
            emit({"type": "info", "text": f"Dataset snapshot {snapshot.stem[:12]} ({snapshot.stat().st_size / 1024 ** 2:,.0f} MB): "
                                          + ("written" if created else "already stored, linked")})
    else:
        (directory / "dataset.jsonl").write_text("\n", encoding="utf-8")
    metrics_file = directory / "metrics.jsonl"
    metrics_file.write_text("".join(json.dumps(row) + "\n" for row in history), encoding="utf-8")
    atomic_json(directory / "run.json", record)
    if model.backend.reason:
        emit({"type": "info", "text": model.backend.reason + " · using CPU"})
    if model.precision_note:
        emit({"type": "info", "text": model.precision_note})
    if split["note"]:
        emit({"type": "info", "text": split["note"]})
    if offload_optimizer:
        emit({"type": "info", "text": "Keeping the optimizer state in host memory and streaming it to the device for each update: "
                                      "slower steps, but the state no longer has to fit the device."})
    emit({"type": "info", "text": f"{model.backend.name} · {model.parameter_count:,} parameters · {tokenizer.vocab_size} tokens · {len(train_samples)} training windows ({cfg['windowing']}, "
                                  f"{train_samples.describe()['from_start']:.0%} start at the prompt, {train_samples.nbytes / 1024 ** 2:.1f} MB"
                                  f"{(' on disk, reused' if reused else ' on disk') if streamed else ''})"})
    if transplanted:
        emit({"type": "info", "text": f"Initialized from {Path(init_from).name}: copied {len(transplanted['copied'])} tensors, "
                                      f"{len(transplanted['fresh'])} freshly initialized."})
    emit({"type": "history", "history": [dict(row) for row in history]})
    emit({"type": "info", "text": f"Evaluation uses up to {cfg['eval_windows']} fixed windows per split at every evaluation."})

    def save(filename):
        metadata = {"architecture": "groundwork_numpy_decoder", "format_version": 1, "config": cfg,
                    "tokenizer": tokenizer.serialize(), "step": step, "best": best if math.isfinite(best) else None,
                    "optimizer_steps": optimizer.steps, "history": [], "data_hash": digest,  # history lives in metrics.jsonl
                    "sampler_state": prefetcher.state(), "model_rng": model.rng.bit_generator.state,
                    "device": model.backend.device, "split_strategy": split["strategy"], "split_hash": split_hash}
        arrays = {"metadata": np.array(json.dumps(metadata))}
        arrays.update({(FILE_PREFIX + key[len(PREFIX):] if key in model.adapter_names else "weight::" + key): model.backend.to_cpu(value)
                       for key, value in model.p.items()})
        for state in optimizer.states:
            arrays.update({f"adam_{state}::{key}": model.backend.to_cpu(value) for key, value in getattr(optimizer, state).items()})
        path = directory / filename
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        replace_file(temporary, path)

    def evaluate(samples):
        """Plain cross-entropy over a fixed, evenly spread subset of the windows (all of them when few)."""
        limit = cfg["eval_windows"]
        total, count = 0., 0
        chosen = np.linspace(0, len(samples) - 1, min(len(samples), limit), dtype=int) if len(samples) > limit else np.arange(len(samples))
        for start in range(0, len(chosen), shape[0]):
            if stop.is_set():
                return None  # a part-way result must never pick the best checkpoint
            x, y = batch_arrays([samples[int(i)] for i in chosen[start:start + shape[0]]])
            targets = int((y != -100).sum())
            total += model.loss(x, y) * targets
            count += targets
        return total / count if count else None

    def report(row):
        history.append(row)
        with metrics_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        record.update({"step": step, "loss": row["loss"]})
        if row.get("val_loss") is not None:
            record["val_loss"] = row["val_loss"]
        try:
            atomic_json(directory / "run.json", record)
        except PermissionError:
            pass  # Progress metadata only; the next report rewrites it. Never stop training for this.
        emit({"type": "metric", **row, "total": cfg["steps"]})

    started = time.monotonic()
    trained_tokens = 0
    budget_reason = None
    detector_state = set()
    rate = cfg["learning_rate"]
    first_step, planned = step, cfg["steps"]
    last_report = 0.
    shape = [cfg["batch_size"], cfg["accumulation"]]  # runtime batch/accumulation: cfg keeps the declared values (resume checks those); an OOM shrinks this
    prefetcher = BatchPrefetcher(sampler, lambda: batch_arrays([train_samples[sampler.randrange(len(train_samples))] for _ in range(shape[0])]),
                                 batch_stage(model.device))  # reads `shape` below, so it sees an OOM's shrink
    try:
        if distributed is not None:
            distributed.sync(step, 0, model)  # the global weights of the first round go in before any step
        if not history and not stop.is_set():
            loss, validation_loss = evaluate(train_samples), evaluate(val_samples)
            if loss is not None:
                report({"step": step, "loss": loss, "val_loss": validation_loss, "tokens_per_second": 0., "lr": cfg["learning_rate"], "grad_norm": 0.})
        while step < planned and not stop.is_set():
            if token_budget is not None and trained_tokens >= token_budget:
                budget_reason = "tokens"
                break
            if wall_seconds is not None and time.monotonic() - started >= wall_seconds:
                budget_reason = "wall-clock"
                break
            while pause.is_set() and not stop.is_set():
                time.sleep(.08)
            if stop.is_set():
                break
            tick = time.monotonic()
            next_step = step + 1
            if step - first_step >= 2 and (wall_seconds is not None or token_budget is not None):
                # A budget would cut the run off wherever it falls, usually in the steady phase. Instead re-plan the total from
                # the measured pace (the run's elapsed time or tokens per step so far) so the schedule's cool-down ends exactly there.
                left = [cfg["steps"] - step]
                if wall_seconds is not None:
                    left.append((wall_seconds - (tick - started)) * (step - first_step) / max(1e-9, tick - started))
                if token_budget is not None:
                    left.append((token_budget - trained_tokens) * (step - first_step) / max(1, trained_tokens))
                planned = step + max(1, int(min(left)))
            snapshot_due = bool(snapshot_every) and (next_step == 1 or next_step % snapshot_every == 0)
            if distributed is not None:
                rate = schedule.rate(distributed.rate_step(next_step), distributed.total_steps, cfg["learning_rate"]) * getattr(distributed, "rate_scale", 1.)
            else:
                rate = (schedule.rate(schedule_window[0] + next_step, schedule_window[1], cfg["learning_rate"]) if schedule_window
                        else schedule.rate(next_step, planned, cfg["learning_rate"]))
            sample_telemetry = bool(telemetry_every) and (next_step == 1 or next_step % telemetry_every == 0)
            while True:
                model.zero_grad()
                loss, token_count = 0., 0
                model.keep_tape, model.last_tape = sample_telemetry, None
                try:
                    for _ in range(shape[1]):
                        x, y, tokens = prefetcher.next()
                        loss += model.loss_and_backward(x, y, 1 / shape[1]) / shape[1]
                        token_count += tokens
                    break
                except RuntimeError as exc:  # torch.cuda.OutOfMemoryError is one; MPS raises a plain RuntimeError
                    if shape[0] == 1 or not (isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()):
                        raise
                model.last_tape = None  # the tape and the failed step's activations die with the exception
                gc.collect()
                if model.device == "cuda":
                    torch.cuda.empty_cache()
                elif model.device == "mps":
                    torch.mps.empty_cache()
                prefetcher.discard()  # a batch drawn ahead has the old size
                shape[:] = [shape[0] // 2, shape[1] * 2]  # ponytail: an odd batch loses a sample per halving; pad instead if it matters
                emit({"type": "info", "kind": "oom", "step": next_step, "batch": shape[0], "accumulation": shape[1],
                      "text": f"Out of memory at step {next_step}: batch size now {shape[0]}, accumulation {shape[1]}."})
            if not math.isfinite(loss):
                raise ValueError("Non-finite loss. Reduce the learning rate.")
            payload = model.telemetry(model.last_tape) if sample_telemetry and model.last_tape is not None else None
            model.keep_tape, model.last_tape = False, None
            before = {key: model.backend.to_cpu(value).copy() for key, value in model.p.items()} if snapshot_due else None
            norm = optimizer.update(trainable, rate)
            step = next_step
            trained_tokens += token_count
            model.backend.synchronize()
            speed = token_count / max(.001, time.monotonic() - tick)
            validation_loss = evaluate(val_samples) if step % cfg["eval_every"] == 0 or step >= planned else None
            if snapshot_due:
                from studio_snapshot import capture, save as save_snapshot
                probe_ids, _ = batch_arrays(train_samples[:min(2, len(train_samples))])
                snapshot = capture(model, probe_ids, step, before, loss=validation_loss)
                active = {(marker['block'], marker['head'], marker['kind']) for marker in snapshot['markers']}
                snapshot['markers'] = [marker for marker in snapshot['markers']
                                       if (marker['block'], marker['head'], marker['kind']) not in detector_state]
                detector_state = active
                path = save_snapshot(directory, snapshot, keep=snapshot_keep)
                emit({"type": "viz_snapshot", "step": step, "path": str(path), "markers": snapshot["markers"]})
            due = step == 1 or step % 5 == 0 or validation_loss is not None or step >= planned or time.monotonic() - last_report >= 1.
            if due:
                last_report = time.monotonic()
                report({"step": step, "loss": loss, "val_loss": validation_loss, "tokens_per_second": speed, "lr": rate, "grad_norm": norm,
                        "trained_tokens": trained_tokens, "elapsed_seconds": time.monotonic() - started})
            if payload:
                emit({"type": "telemetry", "step": step, "total": cfg["steps"], "loss": loss, "lr": rate, **payload})
            if step % cfg["eval_every"] == 0 or step >= planned:
                score = validation_loss if validation_loss is not None else evaluate(train_samples)  # no held-out windows: fixed training windows, not one random batch
                if score is not None and score < best:
                    best = score
                    save("best.npz")
                record["best_loss"] = best
                save("last.npz")
                if cfg["sample_every"] and (step >= planned or step // cfg["eval_every"] % cfg["sample_every"] == 0):
                    text = complete(model, tokenizer, cfg["sample_prompt"])  # what the model writes now, from the weights in memory
                    with (directory / "samples.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"step": step, "text": text, "prompt": cfg["sample_prompt"]}) + "\n")
                    emit({"type": "sample", "step": step, "text": text, "prompt": cfg["sample_prompt"]})
            if distributed is not None and getattr(distributed, "stepped", None):
                distributed.stepped(time.monotonic() - tick)  # the step's own time: the pause wait is before `tick`
            if distributed is not None and distributed.due(step, trained_tokens):
                prefetcher.discard()  # a batch drawn ahead (CUDA) came from this round's windows; sync swaps them and reseeds the sampler
                distributed.sync(step, trained_tokens, model)
        if step < cfg["steps"] and not (budget_reason or stop.is_set()):
            budget_reason = "wall-clock" if wall_seconds is not None else "tokens"  # the re-planned cool-down finished
        if budget_reason and not stop.is_set():
            final_loss, final_validation = evaluate(train_samples), evaluate(val_samples)
            if final_loss is not None:
                report({"step": step, "loss": final_loss, "val_loss": final_validation, "tokens_per_second": 0.,
                        "lr": rate if step else cfg["learning_rate"], "grad_norm": 0., "trained_tokens": trained_tokens,
                        "elapsed_seconds": time.monotonic() - started})
        save("last.npz")
        record.update({"status": "stopped" if stop.is_set() else "completed", "step": step, "duration_seconds": time.monotonic() - started,
                       "trained_tokens": trained_tokens, "budget_reached": budget_reason, "finishedAt": now()})
        atomic_json(directory / "run.json", record)
        emit({"type": "done", "run": record, "directory": str(directory)})
        return record
    except BaseException as exc:
        record.update(status="stopped" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed", error=str(exc), step=step)
        if step:  # keep what the run learned since its last checkpoint, so it can resume; a failing save must not hide `exc`
            try:
                save("last.npz")
                record["checkpoint"] = "last.npz"
            except Exception:
                pass
        atomic_json(directory / "run.json", record)
        raise


def batch_logits(model, windows, sliding, cache, stop=None, chunk=PREFILL_CHUNK):
    """Logits [rows, V] of the position after each window (all the same length), or None when `stop` was set between prefill chunks.
    With a KV cache only the tokens it has not seen are run, `chunk` tokens per pass through the same cache (a long prompt does not
    hold the thread for one long pass, and `stop` is checked between passes); chunk=None runs them in one pass. Once the text outgrows
    the context (`sliding`) every position shifts, so the cache is dropped and the windows re-prefilled. Without a cache the whole
    window is run. Callers pass `chunk` explicitly, as PREFILL_CHUNK read at call time."""
    if cache is None:
        return model.forward(np.array(windows, dtype=np.int64), last_only=True)[:, -1]
    if sliding:
        cache.clear()
        cache["n"] = 0
    new = np.array([window[cache["n"]:] for window in windows], dtype=np.int64)
    size = chunk or new.shape[1]
    for start in range(0, new.shape[1], size):
        if start and stop and stop.is_set():
            return None
        logits = model.forward(new[:, start:start + size], cache=cache, last_only=True)
        if start + size < new.shape[1]:
            logits[0, -1, 0].item()   # a GPU runs ahead of this thread: wait for the chunk, or a stop would find every later chunk already queued
    return logits[:, -1]


def last_logits(model, window, sliding, cache, stop=None, chunk=PREFILL_CHUNK):
    """Logits of the position after `window`, or None when `stop` was set during the prefill; see batch_logits."""
    logits = batch_logits(model, [window], sliding, cache, stop, chunk)
    return None if logits is None else logits[0]


def expand_cache(cache, rows):
    """Copy a batch-1 KV cache to `rows` independent rows (every cached tensor leads with the batch axis)."""
    for key, value in cache.items():
        if key != "n":
            cache[key] = tuple(part.expand(rows, *part.shape[1:]).contiguous() for part in value)


def keep_cache_rows(cache, rows):
    """Drop every row of a batched KV cache except `rows` (indices into the current batch), so a finished answer stops costing compute."""
    for key, value in cache.items():
        if key != "n":
            index = torch.as_tensor(rows, dtype=torch.long, device=value[0].device)
            cache[key] = tuple(part.index_select(0, index) for part in value)


def unsayable(tokenizer):
    """Ids the model is never allowed to say: the markers (PAD, BOS and the role tokens, UNK) and, for GPT-2's word list, its real
    `<|endoftext|>` piece. An imported GPT-2 ties BOS, EOS and that piece to one table row, so all three get the same logit; EOS
    is the one stop token, and the real piece is blocked so the end of text is not counted twice or printed as literal text."""
    ids = [PAD, BOS, USER, SYSTEM, ASSISTANT, UNK]
    end = tokenizer.encoder.get("<|endoftext|>") if tokenizer.kind == "bpe_gpt2" else None
    return ids if end is None else ids + [OFFSET + end]


def complete(model, tokenizer, prompt, max_tokens=48, reduced_cache=False):
    """Raw-text continuation (no chat template) of `prompt` from the model's current weights: greedy, a few dozen pieces.
    reduced_cache: keep the KV cache in half precision (see new_cache); the default float32 cache reproduces the uncached maths."""
    tokens, result = [BOS] + tokenizer.encode(prompt), []
    cache = new_cache(model.device, reduced_cache) if model.can_cache else None
    blocked = unsayable(tokenizer)
    with torch.no_grad():
        for i in range(max_tokens):
            logits = last_logits(model, tokens[-model.dims.n_ctx:], len(tokens) > model.dims.n_ctx, cache, None, PREFILL_CHUNK if i == 0 else None)
            logits[blocked] = -torch.inf
            token = int(logits.argmax())  # only the chosen id leaves the device
            if token == EOS:
                break
            result.append(token)
            tokens.append(token)
    return tokenizer.decode(result)


class LocalModel:
    # Generation's KV cache: True keeps keys and values in bfloat16 (float16 where unsupported), False in float32 (tests, inspection).
    # None: half precision on CUDA only, where chat shares the GPU's memory with jobs; on the CPU and MPS reading the cache back
    # up costs about a tenth of the tokens/s for memory those devices have to spare.
    reduced_kv_cache = None

    def __init__(self, path, seed=42, device="auto", data=None):
        data = data or load_checkpoint(path, weights_only=True)   # `data`: the same checkpoint, already loaded by the caller
        self.config = validate_config(data["config"])
        self.config["device"] = device
        self.tokenizer = Tokenizer.restore(data["tokenizer"])
        self.model = ScratchTransformer(self.config, self.tokenizer.vocab_size)
        self.model.load_weights(data["model"])
        self.rng = np.random.default_rng(seed)
        self.step = data["step"]

    @property
    def device(self):
        return self.model.device

    def to(self, device):
        """Move the weights to "cpu", "cuda" or "mps". Not while a reply is being generated: its cache follows the model's device."""
        self.model.to(device)
        self.config["device"] = device
        return self

    def reduced_cache(self):
        return self.model.device == "cuda" if self.reduced_kv_cache is None else bool(self.reduced_kv_cache)

    def complete(self, prompt, max_tokens=48):
        return complete(self.model, self.tokenizer, prompt, max_tokens, self.reduced_cache())

    def _chat(self, messages):
        """The prompt's token ids, and (position of each message's role token, role id) so a cut window starts with the right role."""
        tokens = [BOS]
        roles = {"user": USER, "assistant": ASSISTANT, "system": SYSTEM}
        starts = []
        for message in messages:
            starts.append((len(tokens), roles.get(message["role"], USER)))
            tokens += [roles.get(message["role"], USER)] + self.tokenizer.encode(message["content"]) + [EOS]
        starts.append((len(tokens), ASSISTANT))
        tokens += [ASSISTANT]
        return tokens, starts

    def _window(self, tokens, starts):
        """The last `context` tokens, cut mid-message the way training cuts."""
        context = self.config["context"]
        if len(tokens) <= context:
            return tokens
        cut = tokens[-context:]
        first = len(tokens) - context
        owner = [role for position, role in starts if position <= first]
        if owner and first not in {position for position, _ in starts}:
            cut = [owner[-1]] + cut[1:]  # cut mid-message: the first token becomes that message's role token, as in training
        return cut

    def generate(self, messages, max_tokens=160, temperature=.8, top_k=40, stop=None, on_text=None, ablate=None):
        """ablate: optional [(block, head), ...] mean-ablated (causal running mean, as the Head lab's) for this reply only.
        The patched pass needs every position, so an ablated reply runs without the KV cache."""
        tokens, starts = self._chat(messages)
        result = []
        context = self.config["context"]
        blocked = unsayable(self.tokenizer)
        ablate = [(int(block), int(head)) for block, head in ablate or ()]
        cache = new_cache(self.model.device, self.reduced_cache()) if self.model.can_cache and not ablate else None  # prefill once, then one token per step; no support (or an ablated reply, which needs every position) -> the whole window each step
        available = self.tokenizer.vocab_size - OFFSET + 1
        k = available if int(top_k) <= 0 else min(int(top_k), available)
        # Sampling runs on the model's device and only the chosen id is copied back. Its stream is seeded from the model's NumPy
        # stream once per reply, so a seed still gives the same replies on a given device.
        generator = torch.Generator(self.model.device).manual_seed(int(self.rng.integers(2 ** 62))) if temperature > 0 else None
        for i in range(max(1, min(4096, int(max_tokens)))):
            if stop and stop.is_set():
                break
            if ablate:
                import studio_inspect
                with torch.no_grad():
                    logits = studio_inspect._ablated_logits(self.model, np.array([self._window(tokens, starts)], dtype=np.int64), ablate, "mean")[0, -1].detach().clone()
            else:
                logits = last_logits(self.model, self._window(tokens, starts), len(tokens) > context, cache, stop, PREFILL_CHUNK if i == 0 else None)
                if logits is None:   # stopped during the prefill
                    break
            logits[blocked] = -torch.inf
            if temperature <= 0:
                token = int(logits.argmax())
            else:
                top, keep = torch.topk(logits / max(.05, float(temperature)), k)
                token = int(keep[torch.multinomial(torch.softmax(top, dim=-1), 1, generator=generator)])
            if token == EOS:
                break
            result.append(token)
            tokens.append(token)
            if on_text and i % 4 == 0:
                on_text(self.tokenizer.decode(result))
        text = self.tokenizer.decode(result)
        if on_text:
            on_text(text)
        return text

    def generate_group(self, messages, n, max_tokens=160, temperature=.8, top_k=40, stop=None):
        """`n` replies to one prompt, as a list of texts: the same replies, for the same seeds, as `n` calls of generate().

        The prompt is run through the model once (batch 1), its KV cache is copied to `n` rows and the rows decode together.
        Each reply stops at its own end token or at the token budget, and a finished row leaves the batch. Each row is sampled with
        its own generator, seeded from `self.rng` in the order generate() would draw them. `temperature` is one number or one per
        reply. A model with no KV cache falls back to calling generate() `n` times. Once the text outgrows the context window every
        row re-reads its own window each step, as generate() does."""
        n = int(n)
        temperatures = list(temperature) if isinstance(temperature, (list, tuple)) else [temperature] * n
        if len(temperatures) != n:
            raise ValueError("Give one temperature, or one per reply.")
        if n < 2 or not self.model.can_cache:
            return [self.generate(messages, max_tokens=max_tokens, temperature=t, top_k=top_k, stop=stop) for t in temperatures]
        if all(t <= 0 for t in temperatures):   # greedy rows are identical: decode one
            return [self.generate(messages, max_tokens=max_tokens, temperature=0, top_k=top_k, stop=stop)] * n
        tokens, starts = self._chat(messages)
        context = self.config["context"]
        blocked = unsayable(self.tokenizer)
        available = self.tokenizer.vocab_size - OFFSET + 1
        k = available if int(top_k) <= 0 else min(int(top_k), available)
        generators = [torch.Generator(self.model.device).manual_seed(int(self.rng.integers(2 ** 62))) if t > 0 else None for t in temperatures]
        scale = torch.tensor([[max(.05, float(t)) if t > 0 else 1.] for t in temperatures], dtype=torch.float32, device=self.model.device)
        rows = [list(tokens) for _ in range(n)]   # each reply's prompt + answer so far
        results = [[] for _ in range(n)]
        active = list(range(n))                   # reply ids still decoding, in batch order
        cache = new_cache(self.model.device, self.reduced_cache())
        for i in range(max(1, min(4096, int(max_tokens)))):
            if stop and stop.is_set():
                break
            sliding = len(rows[active[0]]) > context
            if i == 0:   # one prompt for everyone: run it once
                logits = batch_logits(self.model, [self._window(tokens, starts)], sliding, cache, stop, PREFILL_CHUNK)
                if logits is not None:
                    if not sliding:
                        expand_cache(cache, n)
                    logits = logits.expand(n, -1).clone()
            else:
                logits = batch_logits(self.model, [self._window(rows[r], starts) for r in active], sliding, cache, stop, None)
            if logits is None:   # stopped during the prefill
                break
            logits[:, blocked] = -torch.inf
            top, keep = torch.topk(logits / scale[active], k, dim=-1)
            probabilities = torch.softmax(top, dim=-1)
            picks = [keep[j, torch.multinomial(probabilities[j], 1, generator=generators[r])[0]] if generators[r] is not None
                     else logits[j].argmax() for j, r in enumerate(active)]
            chosen = torch.stack(picks).tolist()   # the only copy back to the host
            alive = []
            for j, (r, token) in enumerate(zip(active, chosen)):
                if token != EOS:
                    results[r].append(token)
                    rows[r].append(token)
                    alive.append(j)
            if len(alive) < len(active):
                active = [active[j] for j in alive]
                if not active:
                    break
                if not sliding:   # a sliding step's cache is rebuilt from the windows on the next one
                    keep_cache_rows(cache, alive)
        return [self.tokenizer.decode(result) for result in results]
