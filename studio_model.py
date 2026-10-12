"""A language-model training engine with handwritten NumPy/CuPy gradients.

The same transformer runs on CPU or CUDA without autograd or pretrained weights.

Data-pipeline config keys (stable API, also used by the curriculum trainer):
  tokenizer  "byte" | "character"   - byte is the default: 263 tokens, no unknown characters, so every
                                       dataset shares one vocabulary and the embedding stays tiny.
  context    32..1024                - tokens per window (default 256 so most chat turns fit prompt + answer).
  windowing  "anchored" | "sequential"
      anchored   - one window per assistant answer (long answers continue in further windows), each
                   left-filled with as much of the preceding conversation as fits, so an answer is always
                   trained with its prompt, exactly as at inference. Raw "text" rows use sequential chunks.
      sequential - legacy: independent context-sized chunks cut from token 0 (answers may lose their prompt).
                   Checkpoints written before this key existed resume with "sequential".
  objective  "assistant" | "all"      - which tokens are targets; every target token is trained exactly once.
Use samples_from(examples, tokenizer, cfg) -> WindowStore and batch_arrays(pairs) to feed train().
"""
from __future__ import annotations

import array
import bisect
import hashlib
import json
import math
import os
import random
import threading
import time
from pathlib import Path

for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_variable, "4")

import numpy as np

from studio_arch import parameter_breakdown, parameter_names, spec_from_config, structure_key, transplant, validate_spec
from studio_core import atomic_json, fingerprint, messages_of, now, replace_file
from studio_backend import get_backend
from studio_planning import merge_duplicate_lineage

DEFAULT_CONFIG = {
    "name": "My first transformer", "tokenizer": "byte", "width": 64,
    "layers": 2, "heads": 4, "context": 256, "expansion": 4, "dropout": 0.1,
    "batch_size": 4, "accumulation": 1, "steps": 300, "learning_rate": 0.001,
    "weight_decay": 0.01, "warmup": 20, "eval_every": 25, "validation": 0.1,
    "seed": 42, "objective": "assistant", "windowing": "anchored", "device": "auto", "architecture": None,
}
# Why these defaults: a character vocabulary over mixed datasets reaches thousands of symbols (hundreds of
# thousands of embedding parameters, up to 8% unknown characters across datasets); bytes cost ~13% more
# tokens but remove both problems. Context 256 lets most question+answer pairs fit one window, and
# anchored windows with the assistant objective train answers only with their prompt in view.
ARCHITECTURE_KEYS = ("tokenizer", "width", "layers", "heads", "context", "expansion", "dropout", "architecture")
PAD, BOS, EOS, USER, ASSISTANT, SYSTEM, UNK = range(7)
OFFSET = 7
WINDOWINGS = ("anchored", "sequential")


def validate_config(config):
    cfg = dict(DEFAULT_CONFIG, **{key: value for key, value in config.items() if key in DEFAULT_CONFIG})
    spec = cfg["architecture"]
    if isinstance(spec, str):
        # Form-driven callers may hand back "", "None" or a JSON string instead of a dict.
        spec = None if spec.strip() in ("", "None", "null") else json.loads(spec)
        cfg["architecture"] = spec
    if spec is not None:
        # A per-block spec is authoritative for structure; layers/heads/expansion are derived for display.
        spec = validate_spec(spec)
        first, embedding = spec["blocks"][0], spec["embedding"]
        cfg.update(architecture=spec, tokenizer=embedding["tokenizer"], width=embedding["width"],
                   context=embedding["context"], layers=len(spec["blocks"]),
                   heads=first["attention"]["heads"], expansion=first["mlp"]["expansion"])
        if "dropout" not in config:
            cfg["dropout"] = embedding["dropout"]
    structural = [] if spec is not None else [("width", 16, 512), ("layers", 1, 12), ("heads", 1, 16), ("context", 32, 1024), ("expansion", 1, 8)]
    for key, low, high in structural + [("batch_size", 1, 128), ("accumulation", 1, 64),
                                        ("steps", 1, 1000000), ("warmup", 0, 100000), ("eval_every", 1, 100000), ("seed", 0, 2147483647)]:
        cfg[key] = int(cfg[key])
        if not low <= cfg[key] <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
    for key, low, high in [("dropout", 0, .5), ("validation", 0, .4), ("learning_rate", 1e-7, .1), ("weight_decay", 0, 1)]:
        cfg[key] = float(cfg[key])
        if not low <= cfg[key] <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
    if spec is None and cfg["width"] % cfg["heads"]:
        raise ValueError("Embedding width must be divisible by the attention head count.")
    if cfg["tokenizer"] not in {"byte", "character"} or cfg["objective"] not in {"all", "assistant"}:
        raise ValueError("Invalid tokenizer or objective.")
    if cfg["windowing"] not in WINDOWINGS:
        raise ValueError("Windowing must be anchored or sequential.")
    if cfg["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda.")
    return cfg


def effective_spec(cfg):
    """The per-block spec a validated config describes (legacy uniform configs are converted)."""
    return cfg.get("architecture") or spec_from_config(cfg)


class Tokenizer:
    def __init__(self, kind="byte", characters=None):
        self.kind = kind
        self.characters = list(characters or [])
        self.lookup = {char: i + OFFSET for i, char in enumerate(self.characters)}
        self.vocab_size = (256 if kind == "byte" else len(self.characters)) + OFFSET

    @classmethod
    def fit(cls, kind, examples):
        if kind == "byte":
            return cls("byte")
        characters = set()
        for example in examples:
            texts = [example["text"]] if example.get("text") else [m["content"] for m in messages_of(example)]
            for text in texts:
                characters.update(text)
            if len(characters) > 16384:
                raise ValueError("Use the byte tokenizer for a vocabulary larger than 16,384 characters.")
        return cls("character", sorted(characters))

    def encode(self, text):
        if self.kind == "byte":
            return [int(b) + OFFSET for b in text.encode("utf-8")]
        return [self.lookup.get(char, UNK) for char in text]

    def decode(self, tokens):
        if self.kind == "byte":
            return bytes(int(t) - OFFSET for t in tokens if OFFSET <= t < self.vocab_size).decode("utf-8", errors="replace")
        return "".join(self.characters[int(t) - OFFSET] for t in tokens if OFFSET <= t < self.vocab_size)

    def serialize(self):
        return {"kind": self.kind, "characters": self.characters}

    @classmethod
    def restore(cls, data):
        if data.get("kind") not in {"byte", "character"}:
            raise ValueError("Invalid tokenizer in checkpoint.")
        return cls(data["kind"], data.get("characters", []))


def softmax(x, xp=np):
    shifted = x - xp.max(x, axis=-1, keepdims=True)
    exp = xp.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def cross_entropy(logits, targets, gradient=False, xp=np):
    flat = logits.reshape(-1, logits.shape[-1])
    labels = xp.asarray(targets).reshape(-1)
    valid = labels != -100
    count = int(valid.sum())
    if count == 0:
        raise ValueError("This batch has no target tokens.")
    shifted = flat - flat.max(axis=-1, keepdims=True)
    exponential = xp.exp(shifted)
    denominator = exponential.sum(axis=-1, keepdims=True)
    rows = xp.flatnonzero(valid)
    loss = float((xp.log(denominator[rows, 0]) - shifted[rows, labels[rows]]).mean())
    if not gradient:
        return loss
    derivative = exponential / denominator
    derivative[rows, labels[rows]] -= 1
    derivative[~valid] = 0
    derivative /= count
    return loss, derivative.reshape(logits.shape)


class ScratchTransformer:
    """Pre-norm causal transformer with explicit forward and reverse passes.

    Every block is described by the architecture spec: optional multi-head attention
    and an optional MLP with 1-4 hidden layers. A legacy uniform config is the
    special case of attention + one GELU layer in every block.
    """
    def __init__(self, config, vocabulary):
        self.config = validate_config(config)
        cfg = self.config
        work = cfg["batch_size"] * cfg["context"] * cfg["width"] * max(1, cfg["layers"])
        self.backend = get_backend(cfg["device"], work=work)
        self.xp = xp = self.backend.xp
        # Keeping the stochastic stream on CPU makes old checkpoints and RNG state
        # portable between devices. Parameters and all learning operations live on xp.
        self.rng = np.random.default_rng(cfg["seed"])
        self.p = {}
        self.spec = effective_spec(cfg)
        self.blocks = [{"attention": b["attention"]["enabled"], "heads": b["attention"]["heads"],
                        "mlp": b["mlp"]["enabled"], "layers": b["mlp"]["layers"], "activation": b["mlp"]["activation"],
                        "id": b["id"]} for b in self.spec["blocks"]]
        self.keep_tape, self.last_tape = False, None
        # parameter_names is ordered like the legacy constructor drew from the RNG: weights in order, biases zero.
        for name, shape in parameter_names(self.spec, vocabulary).items():
            if name.endswith(".w") or name in ("token", "position"):
                self.p[name] = xp.asarray(self.rng.normal(0, .02, shape).astype(np.float32))
            elif name.endswith(".g"):
                self.p[name] = xp.ones(shape, dtype=xp.float32)
            else:
                self.p[name] = xp.zeros(shape, dtype=xp.float32)
        self.g = {key: xp.zeros_like(value) for key, value in self.p.items()}

    @property
    def parameter_count(self):
        return sum(value.size for value in self.p.values())

    def zero_grad(self):
        for grad in self.g.values():
            grad.fill(0)

    def linear(self, x, name):
        output = x @ self.p[name + ".w"]
        if name + ".b" in self.p:
            output = output + self.p[name + ".b"]
        return output

    def linear_back(self, dy, x, name):
        self.g[name + ".w"] += x.reshape(-1, x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        if name + ".b" in self.g:
            self.g[name + ".b"] += dy.sum(axis=tuple(range(dy.ndim - 1)))
        return dy @ self.p[name + ".w"].T

    def norm(self, x, name):
        centered = x - x.mean(axis=-1, keepdims=True)
        inv = 1. / self.xp.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 1e-5)
        normalized = centered * inv
        return normalized * self.p[name + ".g"] + self.p[name + ".b"], (normalized, inv)

    def norm_back(self, dy, cache, name):
        normalized, inv = cache
        axes = tuple(range(dy.ndim - 1))
        self.g[name + ".g"] += (dy * normalized).sum(axis=axes)
        self.g[name + ".b"] += dy.sum(axis=axes)
        dx = dy * self.p[name + ".g"]
        return inv * (dx - dx.mean(axis=-1, keepdims=True) - normalized * (dx * normalized).mean(axis=-1, keepdims=True))

    def dropout(self, x, training):
        probability = self.config["dropout"]
        if training and probability:
            # Same CPU random stream as before; only the boolean keep-mask crosses to the device (1 byte per value, not 4).
            mask = self.xp.asarray(self.rng.random(x.shape) >= probability).astype(x.dtype) / (1 - probability)
            return x * mask, mask
        return x, None

    @staticmethod
    def drop_back(dy, mask):
        return dy * mask if mask is not None else dy

    def activate(self, pre, kind):
        """Activation plus the value its derivative needs (tanh for GELU)."""
        if kind == "relu":
            return self.xp.maximum(pre, 0), None
        tanh = self.xp.tanh(math.sqrt(2 / math.pi) * (pre + .044715 * pre ** 3))
        return .5 * pre * (1 + tanh), tanh

    @staticmethod
    def activate_derivative(pre, tanh, kind):
        if kind == "relu":
            return (pre > 0).astype(pre.dtype)
        return .5 * (1 + tanh) + .5 * pre * (1 - tanh * tanh) * math.sqrt(2 / math.pi) * (1 + 3 * .044715 * pre ** 2)

    def forward(self, ids, training=False, tape=False):
        xp = self.xp
        ids = xp.asarray(ids, dtype=xp.int64)
        batch, length = ids.shape
        cfg = self.config
        if length > cfg["context"]:
            raise ValueError("Sequence exceeds the context length.")
        x = self.p["token"][ids] + self.p["position"][:length]
        x, embedding_mask = self.dropout(x, training)
        block_cache = []
        d = cfg["width"]
        causal = xp.triu(xp.ones((length, length), dtype=bool), 1)
        for i, block in enumerate(self.blocks):
            name = f"block{i}"
            entry = {}
            residual = x
            if block["attention"]:
                h = block["heads"]
                depth = d // h
                normalized1, norm1_cache = self.norm(x, name + ".n1")
                qkv = self.linear(normalized1, name + ".qkv")
                q, k, v = [part.reshape(batch, length, h, depth).transpose(0, 2, 1, 3) for part in xp.split(qkv, 3, axis=-1)]
                scores = (q @ k.swapaxes(-1, -2)) / math.sqrt(depth)
                scores = xp.where(causal, -1e9, scores)
                probabilities = softmax(scores, xp)
                attention, attention_mask = self.dropout(probabilities, training)
                context = (attention @ v).transpose(0, 2, 1, 3).reshape(batch, length, d)
                residual = x + self.linear(context, name + ".proj")
                entry["attention"] = (normalized1, norm1_cache, q, k, v, probabilities, attention, attention_mask, context)
            if block["mlp"]:
                kind = block["activation"]
                normalized2, norm2_cache = self.norm(residual, name + ".n2")
                pre = self.linear(normalized2, name + ".f1")
                act, tanh = self.activate(pre, kind)
                pres, tanhs, acts = [pre], [tanh], [act]
                for j in range(1, block["layers"]):
                    pre = self.linear(act, f"{name}.fh{j}")
                    act, tanh = self.activate(pre, kind)
                    pres.append(pre)
                    tanhs.append(tanh)
                    acts.append(act)
                feedforward, ff_mask = self.dropout(self.linear(act, name + ".f2"), training)
                x = residual + feedforward
                entry["mlp"] = (normalized2, norm2_cache, pres, tanhs, acts, ff_mask)
            else:
                x = residual
            if tape:
                entry["out"] = x
                block_cache.append(entry)
        hidden, final_cache = self.norm(x, "final")
        logits = hidden @ self.p["token"].T
        if tape:
            return logits, (ids, embedding_mask, block_cache, hidden, final_cache)
        return logits

    def loss(self, x, y):
        return cross_entropy(self.forward(x), y, xp=self.xp)

    def loss_and_backward(self, ids, targets, scale=1.0):
        logits, cache = self.forward(ids, training=True, tape=True)
        loss, dlogits = cross_entropy(logits, targets, gradient=True, xp=self.xp)
        dlogits *= scale
        self.backward(dlogits, cache)
        if self.keep_tape:
            self.last_tape = cache
        return loss

    def backward(self, dlogits, cache):
        """Accumulate parameter gradients for any loss expressed as d(loss)/d(logits)."""
        ids, embedding_mask, blocks, hidden, final_cache = cache
        batch, length = ids.shape
        d = self.config["width"]
        self.g["token"] += dlogits.reshape(-1, dlogits.shape[-1]).T @ hidden.reshape(-1, d)
        dx = self.norm_back(dlogits @ self.p["token"], final_cache, "final")
        for i in reversed(range(len(self.blocks))):
            name, block, entry = f"block{i}", self.blocks[i], blocks[i]
            dresidual = dx
            if block["mlp"]:
                kind = block["activation"]
                n2, nc2, pres, tanhs, acts, ff_mask = entry["mlp"]
                dact = self.linear_back(self.drop_back(dx, ff_mask), acts[-1], name + ".f2")
                for j in reversed(range(block["layers"])):
                    dpre = dact * self.activate_derivative(pres[j], tanhs[j], kind)
                    dact = self.linear_back(dpre, n2 if j == 0 else acts[j - 1], name + (".f1" if j == 0 else f".fh{j}"))
                dresidual = dx + self.norm_back(dact, nc2, name + ".n2")
            if block["attention"]:
                h = block["heads"]
                depth = d // h
                n1, nc1, q, k, v, probabilities, attention, attention_mask, context = entry["attention"]
                dcontext = self.linear_back(dresidual, context, name + ".proj").reshape(batch, length, h, depth).transpose(0, 2, 1, 3)
                dv = attention.swapaxes(-1, -2) @ dcontext
                dattention = self.drop_back(dcontext @ v.swapaxes(-1, -2), attention_mask)
                dscores = probabilities * (dattention - (dattention * probabilities).sum(axis=-1, keepdims=True)) / math.sqrt(depth)
                dq = dscores @ k
                dk = dscores.swapaxes(-1, -2) @ q
                dqkv = self.xp.concatenate([value.transpose(0, 2, 1, 3).reshape(batch, length, d) for value in (dq, dk, dv)], axis=-1)
                dn1 = self.linear_back(dqkv, n1, name + ".qkv")
                dx = dresidual + self.norm_back(dn1, nc1, name + ".n1")
            else:
                dx = dresidual
        dx = self.drop_back(dx, embedding_mask)
        self.xp.add.at(self.g["token"], ids.reshape(-1), dx.reshape(-1, d))
        self.g["position"][:length] += dx.sum(axis=0)

    def telemetry(self, tape, sample=2):
        """Cheap per-component statistics from a forward tape plus current gradients (call before the update)."""
        xp, to_cpu = self.xp, self.backend.to_cpu
        names = list(self.g)
        norms = to_cpu(xp.stack([xp.sum(self.g[n] * self.g[n], dtype=xp.float64) for n in names]))
        squared = dict(zip(names, (float(v) for v in norms)))

        def grad_norm(keys):
            return math.sqrt(sum(squared[n] for n in keys))

        def rms(array):
            return float(xp.sqrt((array[:sample] ** 2).mean()))

        ids, _, blocks, _, _ = tape
        embedded = self.p["token"][ids[:sample]] + self.p["position"][:ids.shape[1]]
        result = {"embedding": {"grad_norm": grad_norm(("token", "position")), "rms": rms(embedded)},
                  "blocks": [], "output": {"grad_norm": grad_norm(("final.g", "final.b"))}}
        attention_parts = ("n1", "qkv", "proj")
        for i, (block, entry) in enumerate(zip(self.blocks, blocks)):
            prefix = f"block{i}."
            own = [n for n in squared if n.startswith(prefix)]
            attention = mlp = None
            if block["attention"]:
                probabilities = entry["attention"][5][:sample]
                entropy = -(probabilities * xp.log(probabilities + 1e-12)).sum(axis=-1).mean(axis=(0, 2))
                attention = {"entropy": [float(v) for v in to_cpu(entropy)],
                             "grad_norm": grad_norm([n for n in own if n.split(".")[1] in attention_parts])}
            if block["mlp"]:
                pres, acts = entry["mlp"][2], entry["mlp"][4]
                activity = xp.abs(acts[-1][:sample, :, :48]).mean(axis=(0, 1))
                peak = float(activity.max())
                mlp = {"neurons": [float(v) / peak if peak > 0 else 0. for v in to_cpu(activity)],
                       "active_fraction": float((pres[-1][:sample] > 0).mean()),
                       "grad_norm": grad_norm([n for n in own if n.split(".")[1] not in attention_parts])}
            result["blocks"].append({"id": block["id"], "index": i, "residual_rms": rms(entry["out"]),
                                     "attention": attention, "mlp": mlp})
        return result

    def load_weights(self, weights):
        if set(weights) != set(self.p):
            raise ValueError("Checkpoint parameter names do not match the architecture.")
        for key, value in weights.items():
            value = self.xp.asarray(value)
            if value.shape != self.p[key].shape or not bool(self.xp.isfinite(value).all()):
                raise ValueError(f"Invalid checkpoint parameter: {key}")
            self.p[key][...] = value


class AdamW:
    def __init__(self, model, weight_decay=.01):
        self.xp = model.xp
        self.m = {key: self.xp.zeros_like(value) for key, value in model.p.items()}
        self.v = {key: self.xp.zeros_like(value) for key, value in model.p.items()}
        self.steps = 0
        self.weight_decay = weight_decay

    def update(self, model, learning_rate, clip=1.0):
        norms = [self.xp.sum(grad * grad, dtype=self.xp.float64) for grad in model.g.values()]
        norm = math.sqrt(float(self.xp.stack(norms).sum()))
        if not math.isfinite(norm):
            raise ValueError("Non-finite gradient. Lower the learning rate.")
        multiplier = min(1., clip / (norm + 1e-12))
        self.steps += 1
        for key, parameter in model.p.items():
            gradient = model.g[key] * multiplier
            self.m[key] *= .9
            self.m[key] += .1 * gradient
            self.v[key] *= .999
            self.v[key] += .001 * gradient * gradient
            m = self.m[key] / (1 - .9 ** self.steps)
            v = self.v[key] / (1 - .999 ** self.steps)
            parameter *= 1 - learning_rate * self.weight_decay
            parameter -= learning_rate * m / (self.xp.sqrt(v) + 1e-8)
        return norm


def parameter_estimate(cfg, vocab=263):
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


def samples_from(examples, tokenizer, cfg):
    """Cut examples into training windows: a WindowStore of (inputs, targets) pairs (target -100 = ignored).

    cfg["windowing"] (a missing key means legacy "sequential"), cfg["context"] and cfg["objective"] decide the cut.
    Anchored: each assistant answer is the target of a window that ends with it (long answers continue in
    stride-sized pieces); under objective "all" the text before the first answer is targeted by the first window,
    so every target token is trained exactly once per example. Text rows and answer-less rows are chunked
    sequentially.
    """
    context = cfg["context"]
    anchored = cfg.get("windowing", "sequential") == "anchored"
    assistant_only = cfg["objective"] == "assistant"
    builder = _WindowBuilder(tokenizer.vocab_size)
    for example in examples:
        tokens, mask, positions, roles, spans = _encode_arrays(example, tokenizer, assistant_only)
        if not anchored or not spans:
            _sequential_windows(builder, tokens, mask, context)
            continue
        if assistant_only:
            units = spans
        else:
            ends = [high for _, high in spans]
            units = list(zip([1] + ends[:-1], ends)) + ([(ends[-1], len(tokens))] if ends[-1] < len(tokens) else [])
        _anchored_windows(builder, tokens, mask, positions, roles, units, context)
    return builder.finish()


def batch_arrays(samples):
    """Pad (inputs, targets) pairs - lists or arrays - into int64 matrices; padding targets are -100."""
    width = max(len(item[0]) for item in samples)
    x = np.full((len(samples), width), PAD, dtype=np.int64)
    y = np.full((len(samples), width), -100, dtype=np.int64)
    for i, (inputs, targets) in enumerate(samples):
        x[i, :len(inputs)] = inputs
        y[i, :len(targets)] = targets
    return x, y


def load_checkpoint(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("architecture") != "groundwork_numpy_decoder" or metadata.get("format_version") != 1:
            raise ValueError("Choose a Groundwork NumPy checkpoint (.npz).")
        metadata["model"] = {key.removeprefix("weight::"): archive[key].copy() for key in archive.files if key.startswith("weight::")}
        metadata["m"] = {key.removeprefix("adam_m::"): archive[key].copy() for key in archive.files if key.startswith("adam_m::")}
        metadata["v"] = {key.removeprefix("adam_v::"): archive[key].copy() for key in archive.files if key.startswith("adam_v::")}
    return metadata


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def partition_examples(examples, fraction, sampler, grouped=True):
    """Keep source conversations, documents, and their synthetic descendants together.

    A union-find index is linear in rows plus lineage references; it never builds
    pairwise similarity graphs. Ordinary examples retain the original seeded
    row split. Grouped splits approximate the desired fraction with whole groups.
    """
    indices = list(range(len(examples)))
    sampler.shuffle(indices)
    target = min(len(examples) - 1, max(1, round(len(examples) * fraction))) if fraction and len(examples) > 1 else 0
    parents = list(range(len(examples)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    if grouped:
        anchors = {}
        for index, example in enumerate(examples):
            provenance = example.get("provenance")
            provenance = provenance if isinstance(provenance, dict) else {}
            identities = [("row", str(example["id"]))] if example.get("id") else []
            if example.get("groupId"):
                identities.append(("group", str(example["groupId"])))
            for key, kind in (("source_ids", "row"), ("group_ids", "group")):
                values = provenance.get(key, [])
                if isinstance(values, (list, tuple)):
                    identities.extend((kind, str(value)) for value in values if isinstance(value, (str, int)) and value)
            for identity in identities:
                if identity in anchors:
                    left, right = root(index), root(anchors[identity])
                    parents[max(left, right)] = min(left, right)
                else:
                    anchors[identity] = index
    groups = {}
    for index in indices:
        groups.setdefault(root(index), []).append(index)
    note = ""
    if len(groups) == len(examples):
        validation_ids = set(indices[:target])
    elif target and len(groups) > 1:
        selected_groups = []
        count = 0
        for key, members in groups.items():
            if len(selected_groups) == len(groups) - 1:
                break
            if abs(target - count - len(members)) < abs(target - count):
                selected_groups.append(key)
                count += len(members)
        if not selected_groups:
            selected_groups = [min(groups, key=lambda key: len(groups[key]))]
        validation_ids = {index for key in selected_groups for index in groups[key]}
        if fraction <= .5 and len(validation_ids) > len(examples) / 2:
            validation_ids = set(indices) - validation_ids
        note = "Validation uses whole source groups; its size may differ from the requested fraction."
    else:
        validation_ids = set()
        if target:
            note = "Validation is disabled: all examples share one source lineage. Add independent sources for a held-out evaluation."
    training = [examples[index] for index in indices if index not in validation_ids]
    validation = [examples[index] for index in indices if index in validation_ids]
    report = {"strategy": "lineage-v1" if grouped else "rows-v1", "groups": len(groups),
              "training_groups": len({root(index) for index in indices if index not in validation_ids}),
              "validation_groups": len({root(index) for index in validation_ids}), "note": note}
    return training, validation, report


def _initialize_from(model, tokenizer, path):
    """Transplant weights from a checkpoint into the model's (possibly different) architecture."""
    data = load_checkpoint(path)
    previous = validate_config(data["config"])
    xp = model.xp
    old = {key: xp.asarray(value) for key, value in data["model"].items()}
    old_tokenizer = Tokenizer.restore(data["tokenizer"])
    if "token" in old and old_tokenizer.serialize() != tokenizer.serialize() and old_tokenizer.kind == tokenizer.kind == "character":
        # Different alphabets: keep rows for shared characters and the special tokens, the rest stay fresh.
        rows = model.p["token"].copy()
        old_rows = {char: i + OFFSET for i, char in enumerate(old_tokenizer.characters)}
        for i, char in enumerate(tokenizer.characters):
            if char in old_rows and old["token"].shape[1] == rows.shape[1]:
                rows[i + OFFSET] = old["token"][old_rows[char]]
        if old["token"].shape[1] == rows.shape[1]:
            rows[:OFFSET] = old["token"][:OFFSET]
        old["token"] = rows
    return transplant(effective_spec(previous), old, model.spec, model.p)


def train(config, examples, output_dir, emit=None, stop=None, pause=None, resume=None, init_from=None, telemetry_every=5):
    emit = emit or (lambda event: None)
    stop, pause = stop or threading.Event(), pause or threading.Event()
    cfg = validate_config(config)
    if resume and init_from:
        raise ValueError("Resume and initialize-from cannot be combined.")
    directory = Path(output_dir)
    if any((directory / name).exists() for name in ("config.json", "dataset.jsonl", "last.npz", "best.npz")):
        raise ValueError("Choose a new output folder to preserve the existing run.")
    directory.mkdir(parents=True, exist_ok=True)
    unique = {}
    for example in examples:
        if stop.is_set():
            raise InterruptedError("Training preparation cancelled.")
        content_hash = fingerprint(example)
        if content_hash in unique:
            merge_duplicate_lineage(unique[content_hash], example)
        else:
            unique[content_hash] = dict(example)
    examples = list(unique.values())
    if not examples:
        raise ValueError("The dataset is empty.")
    digest = hashlib.sha256("".join(fingerprint(e) for e in examples).encode()).hexdigest()
    sampler = random.Random(cfg["seed"])
    checkpoint = load_checkpoint(resume) if resume else None
    # Older checkpoints must resume with their original row-based partition.
    grouped = not checkpoint or checkpoint.get("split_strategy") == "lineage-v1"
    training, validation, split = partition_examples(examples, cfg["validation"], sampler, grouped)
    split_hash = hashlib.sha256(("".join(fingerprint(e) for e in training) + "|" +
                                 "".join(fingerprint(e) for e in validation)).encode()).hexdigest()
    if checkpoint:
        if checkpoint["data_hash"] != digest:
            raise ValueError("Resume requires the original dataset snapshot.")
        if checkpoint.get("split_hash") and checkpoint["split_hash"] != split_hash:
            raise ValueError("Resume requires the original source lineage and validation split.")
        # Old checkpoints have no "architecture": validate_config derives the equivalent uniform spec.
        previous = validate_config(checkpoint["config"])
        if "windowing" not in checkpoint["config"]:
            # Checkpoints from before windowing existed were trained with independent sequential chunks.
            previous["windowing"] = cfg["windowing"] = "sequential"
        keys = tuple(k for k in ARCHITECTURE_KEYS if k != "architecture") + ("seed", "validation", "objective", "windowing", "batch_size", "accumulation")
        if (any(cfg[k] != previous[k] for k in keys)
                or structure_key(effective_spec(cfg)) != structure_key(effective_spec(previous))):
            raise ValueError("Resume must preserve architecture, tokenizer, split, seed, objective and batch settings.")
        tokenizer = Tokenizer.restore(checkpoint["tokenizer"])
    else:
        tokenizer = Tokenizer.fit(cfg["tokenizer"], training)
    train_samples = samples_from(training, tokenizer, cfg)
    val_samples = samples_from(validation, tokenizer, cfg)
    if not train_samples:
        raise ValueError("No trainable target tokens were found.")
    model = ScratchTransformer(cfg, tokenizer.vocab_size)
    optimizer = AdamW(model, cfg["weight_decay"])
    step, best, history = 0, float("inf"), []
    transplanted = None
    if init_from:
        transplanted = _initialize_from(model, tokenizer, init_from)
    if checkpoint:
        model.load_weights(checkpoint["model"])
        for key in model.p:
            optimizer.m[key][...] = model.xp.asarray(checkpoint["m"][key])
            optimizer.v[key][...] = model.xp.asarray(checkpoint["v"][key])
        step = checkpoint["step"]
        best = checkpoint["best"] if checkpoint["best"] is not None else float("inf")
        optimizer.steps = checkpoint["optimizer_steps"]
        history = checkpoint.get("history", [])
        sampler.setstate(_tuples(checkpoint["sampler_state"]))
        model.rng.bit_generator.state = checkpoint["model_rng"]
        if cfg["steps"] <= step:
            raise ValueError("Total steps must exceed the checkpoint step.")
    record = {"name": cfg["name"], "createdAt": now(), "status": "training", "step": step, "steps": cfg["steps"],
              "parameters": model.parameter_count, "vocabulary": tokenizer.vocab_size,
              "backend": "cupy" if model.backend.device == "cuda" else "numpy", "device": model.backend.device,
              "device_name": model.backend.name,
              "train_examples": len(training), "validation_examples": len(validation), "resumed_from": str(resume) if resume else None,
              "initialized_from": str(init_from) if init_from else None,
              "best_loss": best if math.isfinite(best) else None, "split": split}
    atomic_json(directory / "config.json", cfg)
    atomic_json(directory / "tokenizer.json", tokenizer.serialize())
    (directory / "dataset.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in examples) + "\n", encoding="utf-8")
    metrics_file = directory / "metrics.jsonl"
    metrics_file.write_text("".join(json.dumps(row) + "\n" for row in history), encoding="utf-8")
    atomic_json(directory / "run.json", record)
    if model.backend.reason:
        emit({"type": "info", "text": model.backend.reason + " · using CPU"})
    if split["note"]:
        emit({"type": "info", "text": split["note"]})
    emit({"type": "info", "text": f"{model.backend.name} · {model.parameter_count:,} parameters · {tokenizer.vocab_size} tokens · {len(train_samples)} training windows ({cfg['windowing']}, "
                                  f"{train_samples.describe()['from_start']:.0%} start at the prompt, {train_samples.nbytes / 1024 ** 2:.1f} MB)"})
    if transplanted:
        emit({"type": "info", "text": f"Initialized from {Path(init_from).name}: copied {len(transplanted['copied'])} tensors, "
                                      f"{len(transplanted['fresh'])} freshly initialized."})
    emit({"type": "history", "history": [dict(row) for row in history]})

    def save(filename):
        metadata = {"architecture": "groundwork_numpy_decoder", "format_version": 1, "config": cfg,
                    "tokenizer": tokenizer.serialize(), "step": step, "best": best if math.isfinite(best) else None,
                    "optimizer_steps": optimizer.steps, "history": history, "data_hash": digest,
                    "sampler_state": sampler.getstate(), "model_rng": model.rng.bit_generator.state,
                    "device": model.backend.device, "split_strategy": split["strategy"], "split_hash": split_hash}
        arrays = {"metadata": np.array(json.dumps(metadata))}
        arrays.update({"weight::" + key: model.backend.to_cpu(value) for key, value in model.p.items()})
        arrays.update({"adam_m::" + key: model.backend.to_cpu(value) for key, value in optimizer.m.items()})
        arrays.update({"adam_v::" + key: model.backend.to_cpu(value) for key, value in optimizer.v.items()})
        path = directory / filename
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        replace_file(temporary, path)

    def evaluate(samples):
        total, count = 0., 0
        for start in range(0, min(len(samples), cfg["batch_size"] * 20), cfg["batch_size"]):
            if stop.is_set():
                break
            x, y = batch_arrays(samples[start:start + cfg["batch_size"]])
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
    try:
        if not history and not stop.is_set():
            loss, validation_loss = evaluate(train_samples), evaluate(val_samples)
            if loss is not None:
                report({"step": step, "loss": loss, "val_loss": validation_loss, "tokens_per_second": 0., "lr": cfg["learning_rate"], "grad_norm": 0.})
        while step < cfg["steps"] and not stop.is_set():
            while pause.is_set() and not stop.is_set():
                time.sleep(.08)
            if stop.is_set():
                break
            tick = time.monotonic()
            next_step = step + 1
            warmup = min(cfg["warmup"], cfg["steps"] - 1)
            if warmup and next_step <= warmup:
                rate = cfg["learning_rate"] * next_step / warmup
            else:
                progress = (next_step - warmup) / max(1, cfg["steps"] - warmup)
                rate = cfg["learning_rate"] * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
            model.zero_grad()
            loss, token_count = 0., 0
            sample_telemetry = bool(telemetry_every) and (next_step == 1 or next_step % telemetry_every == 0)
            model.keep_tape, model.last_tape = sample_telemetry, None
            for _ in range(cfg["accumulation"]):
                batch = [train_samples[sampler.randrange(len(train_samples))] for _ in range(cfg["batch_size"])]
                x, y = batch_arrays(batch)
                loss += model.loss_and_backward(x, y, 1 / cfg["accumulation"]) / cfg["accumulation"]
                token_count += int((y != -100).sum())
            if not math.isfinite(loss):
                raise ValueError("Non-finite loss. Reduce the learning rate.")
            payload = model.telemetry(model.last_tape) if sample_telemetry and model.last_tape is not None else None
            model.keep_tape, model.last_tape = False, None
            norm = optimizer.update(model, rate)
            step = next_step
            model.backend.synchronize()
            speed = token_count / max(.001, time.monotonic() - tick)
            validation_loss = evaluate(val_samples) if step % cfg["eval_every"] == 0 or step == cfg["steps"] else None
            if step == 1 or step % 5 == 0 or validation_loss is not None or step == cfg["steps"]:
                report({"step": step, "loss": loss, "val_loss": validation_loss, "tokens_per_second": speed, "lr": rate, "grad_norm": norm})
            if payload:
                emit({"type": "telemetry", "step": step, "total": cfg["steps"], "loss": loss, "lr": rate, **payload})
            if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
                score = validation_loss if validation_loss is not None else loss
                if score < best:
                    best = score
                    save("best.npz")
                record["best_loss"] = best
                save("last.npz")
        save("last.npz")
        record.update({"status": "stopped" if stop.is_set() else "completed", "step": step, "duration_seconds": time.monotonic() - started, "finishedAt": now()})
        atomic_json(directory / "run.json", record)
        emit({"type": "done", "run": record, "directory": str(directory)})
        return record
    except Exception as exc:
        record.update(status="failed", error=str(exc), step=step)
        atomic_json(directory / "run.json", record)
        raise


class LocalModel:
    def __init__(self, path, seed=42, device="auto"):
        data = load_checkpoint(path)
        self.config = validate_config(data["config"])
        self.config["device"] = device
        self.tokenizer = Tokenizer.restore(data["tokenizer"])
        self.model = ScratchTransformer(self.config, self.tokenizer.vocab_size)
        self.model.load_weights(data["model"])
        self.rng = np.random.default_rng(seed)
        self.step = data["step"]

    def generate(self, messages, max_tokens=160, temperature=.8, top_k=40, stop=None, on_text=None):
        tokens = [BOS]
        roles = {"user": USER, "assistant": ASSISTANT, "system": SYSTEM}
        for message in messages:
            tokens += [roles.get(message["role"], USER)] + self.tokenizer.encode(message["content"]) + [EOS]
        tokens += [ASSISTANT]
        result = []
        for i in range(max(1, min(4096, int(max_tokens)))):
            if stop and stop.is_set():
                break
            # Transfer only the final vocabulary vector for seeded CPU sampling.
            output = self.model.forward(np.array([tokens[-self.config["context"]:]], dtype=np.int64))
            logits = self.model.backend.to_cpu(output[0, -1]).astype(np.float64)
            logits[[PAD, BOS, USER, SYSTEM, ASSISTANT, UNK]] = -np.inf
            if temperature <= 0:
                token = int(logits.argmax())
            else:
                logits /= max(.05, float(temperature))
                available = self.tokenizer.vocab_size - OFFSET + 1
                k = available if int(top_k) <= 0 else min(int(top_k), available)
                keep = np.argpartition(logits, -k)[-k:]
                probabilities = softmax(logits[keep])
                token = int(self.rng.choice(keep, p=probabilities))
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
