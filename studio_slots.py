"""The module contract: typed slots, and the base classes every swappable module builds on.

A model is a ModelSpec (studio_spec) whose slots each name one registered module (studio_registry).
Model-side modules build a Layer with a hand-written forward and backward pass; training-side modules
build an objective, an Optimizer, a schedule, an initializer or a dtype. Nothing here imports a module.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

# id -> (label, tensor signature, group). Order is the order the Sandbox lists them in.
SLOTS = {
    "tokenizer": ("Tokenizer", "text → ids [B, T]", "data"),
    "embedding": ("Embedding", "ids [B, T] → x [B, T, d]", "model"),
    "position": ("Position", "x [B, T, d] → x [B, T, d], or applied inside the token mixer", "model"),
    "norm": ("Norm", "x [B, T, d] → x [B, T, d] · each token alone", "model"),
    "mixer": ("Token mixer", "x [B, T, d] → Δx [B, T, d] · causal: output t may read only inputs ≤ t", "model"),
    "channel": ("Channel mixer", "x [B, T, d] → Δx [B, T, d] · each token alone", "model"),
    "head": ("Output head", "x [B, T, d] → logits [B, T, V]", "model"),
    "objective": ("Objective", "(logits [B, T, V], targets [B, T]) → scalar loss", "training"),
    "optimizer": ("Optimizer", "(params, grads, lr) → updated params", "training"),
    "schedule": ("LR schedule", "step → learning rate", "training"),
    "init": ("Initialization", "(module, config) → initial weights", "training"),
    "precision": ("Precision", "forward/backward dtype policy", "training"),
}
BLOCK_SLOTS = ("norm", "mixer", "channel")  # every block is x + mixer(norm(x)), then x + channel(norm(x))
TRAIN_SLOTS = ("objective", "optimizer", "schedule", "init", "precision")
LAYER_SLOTS = ("embedding", "position", "norm", "mixer", "channel", "head")  # slots whose build() returns a Layer

# Token ids shared by every tokenizer: real vocabulary entries start at OFFSET.
PAD, BOS, EOS, USER, ASSISTANT, SYSTEM, UNK = range(7)
OFFSET = 7

# Parameter roles. The initializer fills by role; optimizers decay only the DECAY roles.
# "weight" matrix · "residual_out" matrix that writes into the residual stream · "embedding" lookup table ·
# "gain" starts at 1 · "bias" starts at 0 · "custom" was given its values by the module and is left alone.
DECAY_ROLES = frozenset({"weight", "residual_out", "embedding"})


@dataclass(frozen=True)
class Dims:
    """What a module may know about the model it is built into."""
    d_model: int
    n_layer: int
    n_ctx: int
    vocab: int
    dropout: float = 0.0
    layer: int = 0              # index of the block being built (0 outside the stack)
    xp: object = np             # array namespace: numpy, or cupy on the GPU. Modules never import numpy for math.
    dtype: object = np.float32

    @property
    def bytes(self):
        return np.dtype(self.dtype).itemsize


@dataclass
class Estimate:
    params: int = 0
    flops_per_token: float = 0.0     # forward pass, one token
    activation_bytes: int = 0        # per token, kept for the backward pass
    active_params: int | None = None  # parameters a single token actually uses (mixture of experts); None = all
    kv_bytes: int = 0                # per token of context kept for generation


@dataclass
class Issue:
    level: str          # "ERROR" blocks training · "WARN" · "INFO"
    text: str
    slot: str = ""
    block: int | None = None


class Ctx:
    """Per-forward-pass context handed to every Layer.forward."""

    def __init__(self, xp=np, training=False, rng=None, dropout=0.0, hooks=None, patches=None, position=None, embedding=None):
        self.xp, self.training, self.rng, self.dropout = xp, training, rng, dropout
        self.capture = hooks        # None: record nothing · True: record every hook · a set of full hook names
        self.patches = patches or {}  # full hook name -> function(value) -> replacement value
        self.captured = {}
        self.block = None           # index of the block being run, or None outside the stack
        self.position = position if position is not None else NO_POSITION
        self.embedding = embedding
        self.aux = 0.0              # auxiliary loss a module adds on top of the objective (MoE balance loss)
        self.scale = 1.0            # set before backward: multiply auxiliary-loss gradients by this
        self._mask = {}

    def mask(self, length):
        """[T, T] bool, True where key position > query position (the future, which must stay hidden)."""
        if length not in self._mask:
            self._mask[length] = self.xp.triu(self.xp.ones((length, length), dtype=bool), 1)
        return self._mask[length]

    def drop(self, x):
        """Inverted dropout while training: (x * mask, mask). Outside training: (x, None)."""
        if self.training and self.dropout:
            # the random stream stays on the CPU so a run is reproducible across devices
            mask = self.xp.asarray(self.rng.random(x.shape) >= self.dropout).astype(x.dtype) / (1 - self.dropout)
            return x * mask, mask
        return x, None

    def wants(self, name):
        full = name if self.block is None else f"block{self.block}.{name}"
        return full in self.patches or self.capture is True or (self.capture is not None and full in self.capture)

    def emit(self, name, value):
        """Expose a named intermediate. Returns the value to keep using (a patch may have replaced it)."""
        if self.capture is None and not self.patches:
            return value
        full = name if self.block is None else f"block{self.block}.{name}"
        if full in self.patches:
            value = self.patches[full](value)
        if self.capture is True or (self.capture is not None and full in self.capture):
            self.captured[full] = value
        return value


class Layer:
    """One built module: named weights plus a hand-written forward and backward pass.

    Rules the compiler relies on:
    - register every weight with add(); read it as self.params[name] each time (never rebind it)
    - forward() stores everything backward() needs in self.cache and nothing else on self
    - backward(dy) returns dx and ADDS (+=) its parameter gradients into self.grads
    """

    def __init__(self, dims):
        self.dims, self.xp, self.dtype = dims, dims.xp, dims.dtype
        self.params, self.grads, self.roles, self.lr_scale = {}, {}, {}, {}
        self.cache = None

    def add(self, name, shape, role="weight", value=None):
        """Register a weight. Without `value` the model's initializer fills it by role."""
        xp = self.xp
        self.params[name] = xp.zeros(shape, dtype=self.dtype) if value is None else xp.asarray(value, dtype=self.dtype).reshape(shape)
        self.grads[name] = xp.zeros(shape, dtype=self.dtype)
        self.roles[name] = role if value is None else "custom"

    def forward(self, x, ctx):
        raise NotImplementedError

    def backward(self, dy):
        raise NotImplementedError

    # Position modules that act inside attention override these; attention-style mixers call them on ctx.position.
    def rotate(self, q, k):
        """q [B, H, T, h], k [B, Hk, T, h] -> the same shapes, position applied."""
        return q, k

    def rotate_back(self, dq, dk):
        return dq, dk

    def score_bias(self, length, heads):
        """Additive bias for attention scores, [heads, T, T] indexed [head, query, key], or None."""
        return None


NO_POSITION = Layer(Dims(1, 1, 1, 1))


class Optimizer:
    """Base for optimizer modules: subclasses name their per-parameter state and write apply()."""
    states = ()  # e.g. ("m", "v"): one array per parameter each, saved in checkpoints

    def __init__(self, cfg=None):
        self.cfg, self.steps = cfg, 0

    def attach(self, model):
        self.xp, self.roles, self.lr_scale = model.xp, model.roles, model.lr_scale
        for name in self.states:
            setattr(self, name, {key: self.xp.zeros_like(value) for key, value in model.p.items()})
        return self

    def decays(self, key):
        return self.roles.get(key) in DECAY_ROLES

    def update(self, model, learning_rate, clip=1.0):
        """Clip the global gradient norm, then step every parameter in place. Returns the pre-clip norm."""
        xp = self.xp
        norm = math.sqrt(float(xp.stack([xp.sum(grad * grad, dtype=xp.float64) for grad in model.g.values()]).sum()))
        if not math.isfinite(norm):
            raise ValueError("Non-finite gradient. Lower the learning rate.")
        multiplier = min(1., clip / (norm + 1e-12))
        self.steps += 1
        for key, parameter in model.p.items():
            rate = learning_rate * self.lr_scale[key] if key in self.lr_scale else learning_rate
            self.apply(key, parameter, model.g[key] * multiplier, rate)
        return norm

    def apply(self, key, parameter, gradient, lr):
        """Update `parameter` in place from its (already clipped) gradient."""
        raise NotImplementedError


class ModuleSpec:
    """Metadata and factory for one module. Subclass, set the class attributes, decorate with @register."""
    id = ""            # "<slot>/<short id>", e.g. "mixer/ssm"
    slot = ""
    name = ""
    blurb = ""
    ref = ""
    config = None      # a dataclass whose fields carry metadata: min, max, choices, help
    provides = frozenset()   # capabilities other modules can require
    requires = frozenset()   # position modules: capabilities every token mixer must provide
    hooks = ()         # (name, symbolic shape) pairs, e.g. ("attn.pattern", "B H T T")
    skip = False       # True for "none" modules: the compiler leaves the step out instead of building a layer
    backward = "Hand-written in NumPy · passes the float64 gradient check"
    glyph3d = None     # {"kind": "beams"|"ribbon"|"fan"|"tower", "hooks": [names]}; None = the slot's default glyph

    @property
    def short(self):
        return self.id.split("/", 1)[1]

    def parse(self, raw=None):
        """Validated config instance from a (possibly partial) JSON dict; raises ValueError."""
        return parse_config(self.config, raw, self.name or self.id)

    def symbols(self, cfg, dims):
        """Values for the module's own symbols in hook shapes (B, T, d and V are always known)."""
        return {}

    def estimate(self, cfg, dims):
        return Estimate()

    def validate(self, cfg, dims):
        return []

    def build(self, cfg, dims, rng):
        raise NotImplementedError


def parse_config(cls, raw, label="config"):
    raw = dict(raw or {})
    if cls is None:
        if raw:
            raise ValueError(f"{label} takes no settings (got {', '.join(sorted(raw))}).")
        return None
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ValueError(f"{label} has no setting named {', '.join(unknown)}.")
    values = {}
    for name, f in fields.items():
        default = f.default
        value = raw.get(name, default)
        where = f"{label} · {name}"
        if isinstance(default, bool):
            if not isinstance(value, (bool, int)) or value not in (0, 1):
                raise ValueError(f"{where} must be true or false.")
            value = bool(value)
        elif isinstance(default, int):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
                raise ValueError(f"{where} must be a whole number.")
            value = int(value)
        elif isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{where} must be a number.")
            value = float(value)
        elif not isinstance(value, str):
            raise ValueError(f"{where} must be text.")
        meta = f.metadata
        if "choices" in meta and value not in meta["choices"]:
            raise ValueError(f"{where} must be one of {', '.join(str(c) for c in meta['choices'])}.")
        if ("min" in meta and value < meta["min"]) or ("max" in meta and value > meta["max"]):
            raise ValueError(f"{where} must be between {meta.get('min', '-∞')} and {meta.get('max', '∞')}.")
        values[name] = value
    return cls(**values)


def config_dict(cfg):
    return dataclasses.asdict(cfg) if cfg is not None else {}
