"""Compile a ModelSpec into a stack of Layers, and run it forward and backward.

The block template is written once here: x + mixer(norm(x)), then x + channel(norm(x)) (or the post-norm
variant). The residual gradient fans out to both branches and is summed back. Everything else is a module.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

import studio_registry as registry
from studio_backend import get_backend
from studio_slots import Ctx, Dims
from studio_spec import block_at, module_of, validate_spec
from studio_validate import errors

TELEMETRY_HOOKS = ("attn.pattern", "mlp.pre", "mlp.act", "resid.post")


@dataclass
class Block:
    id: str
    index: int
    post: bool          # post-norm: norm(x + f(x)) instead of x + f(norm(x))
    norm1: object = None
    mixer: object = None
    norm2: object = None
    channel: object = None


@dataclass
class Tape:
    """What one forward pass leaves behind for its backward pass."""
    ids: object
    ctx: Ctx
    caches: list        # each layer's cache, in the order the layers ran
    masks: list         # dropout masks, in the order they were drawn
    captured: dict


class CompiledModel:
    """A ModelSpec made runnable. `p` and `g` map full parameter names to the layers' own arrays."""

    def __init__(self, spec, vocabulary, seed=42, device="auto", batch_size=4):
        self.spec = spec = validate_spec(spec)
        problems = errors(spec, vocabulary)
        if problems:
            raise ValueError(problems[0])
        d = spec["dims"]
        self.backend = get_backend(device, work=batch_size * d["n_ctx"] * d["d_model"] * max(1, d["n_layer"]))
        self.xp = self.backend.xp
        # Keeping the stochastic stream on CPU makes checkpoints and RNG state portable between devices.
        self.rng = np.random.default_rng(seed)
        precision, precision_cfg = module_of("precision", spec["train"]["precision"])
        self.dims = Dims(d["d_model"], d["n_layer"], d["n_ctx"], int(vocabulary), d["dropout"], 0, self.xp, precision.build(precision_cfg, None, None))
        init, init_cfg = module_of("init", spec["train"]["init"])
        self._init = init.build(init_cfg, self.dims, None)
        objective, objective_cfg = module_of("objective", spec["train"]["objective"])
        self.objective = objective.build(objective_cfg, self.dims, None)
        self.p, self.g, self.roles, self.lr_scale, self.owner, self.group = {}, {}, {}, {}, {}, {}
        self.layers = []
        self.keep_tape, self.last_tape = False, None
        self.embedding = self._make("embedding", spec["embedding"], "")
        self.position = self._make("position", spec["position"], "")
        self.stack = []
        for i in range(d["n_layer"]):
            entry = block_at(spec, i)
            block = Block(entry["id"], i, entry["norm"]["config"].get("placement") == "post")
            mixer_on, channel_on = (not registry.get(slot, entry[slot]["id"]).skip for slot in ("mixer", "channel"))
            if mixer_on:
                block.norm1 = self._make("norm", entry["norm"], f"block{i}.n1.", i)
                block.mixer = self._make("mixer", entry["mixer"], f"block{i}.", i)
            if channel_on:
                block.norm2 = self._make("norm", entry["norm"], f"block{i}.n2.", i)
                block.channel = self._make("channel", entry["channel"], f"block{i}.", i)
            self.stack.append(block)
        self.blocks = [{"id": block.id, "index": block.index} for block in self.stack]
        self.final = self._make("norm", spec["block"]["norm"], "final.")
        self.head = self._make("head", spec["head"], "head.")

    def _make(self, slot, entry, prefix, block=None):
        """Build, initialize and register one layer; None for a "none" module."""
        module, cfg = module_of(slot, entry)
        if module.skip:
            return None
        layer = module.build(cfg, dataclasses.replace(self.dims, layer=block or 0), self.rng)
        layer.module, layer.cfg = module, cfg
        self.layers.append(layer)
        self._init.apply(layer, self.rng, self.dims)
        for name, value in layer.params.items():
            full = prefix + name
            if full in self.p:
                raise ValueError(f"Two modules both name a parameter {full!r}; give one a more specific name.")
            self.p[full], self.g[full], self.roles[full] = value, layer.grads[name], layer.roles[name]
            self.owner[full], self.group[full] = (slot, block), len(self.layers)
            if name in layer.lr_scale:
                self.lr_scale[full] = layer.lr_scale[name]
        return layer

    @property
    def parameter_count(self):
        return sum(value.size for value in self.p.values())

    def zero_grad(self):
        for grad in self.g.values():
            grad.fill(0)

    def forward(self, ids, training=False, tape=False, hooks=None, patches=None):
        """Logits [B, T, V]. tape=True also returns the Tape backward() needs.

        hooks: True, or a set of hook names ("block0.attn.pattern", "logits") to record on the tape.
        patches: {hook name: function(value) -> value} to intervene on a hook that feeds the computation.
        """
        xp = self.xp
        ids = xp.asarray(ids, dtype=xp.int64)
        if ids.shape[1] > self.dims.n_ctx:
            raise ValueError("Sequence exceeds the context length.")
        ctx = Ctx(xp, training, self.rng, self.dims.dropout, hooks, patches, self.position, self.embedding)
        caches, masks = [], []

        def run(layer, x):
            y = layer.forward(x, ctx)
            caches.append(layer.cache)
            return y

        def drop(x):
            x, mask = ctx.drop(x)
            masks.append(mask)
            return x

        x = run(self.embedding, ids)
        if self.position is not None:
            x = run(self.position, x)
        x = ctx.emit("embed", drop(x))
        for block in self.stack:
            ctx.block = block.index
            x = ctx.emit("resid.pre", x)
            if block.mixer is not None:
                if block.post:
                    x = x + run(block.mixer, x)
                    x = run(block.norm1, x) if block.norm1 is not None else x
                else:
                    x = x + run(block.mixer, run(block.norm1, x) if block.norm1 is not None else x)
            x = ctx.emit("resid.mid", x)
            if block.channel is not None:
                if block.post:
                    x = x + drop(run(block.channel, x))
                    x = run(block.norm2, x) if block.norm2 is not None else x
                else:
                    x = x + drop(run(block.channel, run(block.norm2, x) if block.norm2 is not None else x))
            x = ctx.emit("resid.post", x)
        ctx.block = None
        if self.final is not None:
            x = run(self.final, x)
        x = ctx.emit("final", x)
        logits = ctx.emit("logits", run(self.head, x))
        if tape:
            return logits, Tape(ids, ctx, caches, masks, ctx.captured)
        return logits

    def backward(self, dlogits, tape):
        """Accumulate parameter gradients for any loss expressed as d(loss)/d(logits)."""
        caches, masks = list(tape.caches), list(tape.masks)

        def back(layer, dy):
            layer.cache = caches.pop()  # the cache of the forward pass this tape came from, even if others ran since
            return layer.backward(dy)

        def undrop(dy):
            mask = masks.pop()
            return dy * mask if mask is not None else dy

        dx = back(self.head, dlogits)
        if self.final is not None:
            dx = back(self.final, dx)
        for block in reversed(self.stack):
            # each `x + branch(x)` sends the gradient down both paths; the two results are added back together
            if block.channel is not None:
                if block.post:
                    dx = back(block.norm2, dx) if block.norm2 is not None else dx
                    dx = dx + back(block.channel, undrop(dx))
                else:
                    branch = back(block.channel, undrop(dx))
                    dx = dx + (back(block.norm2, branch) if block.norm2 is not None else branch)
            if block.mixer is not None:
                if block.post:
                    dx = back(block.norm1, dx) if block.norm1 is not None else dx
                    dx = dx + back(block.mixer, dx)
                else:
                    branch = back(block.mixer, dx)
                    dx = dx + (back(block.norm1, branch) if block.norm1 is not None else branch)
        dx = undrop(dx)
        if self.position is not None:
            dx = back(self.position, dx)
        back(self.embedding, dx)

    def loss(self, x, y):
        return self.objective(self.forward(x), y, xp=self.xp)

    def loss_and_backward(self, ids, targets, scale=1.0):
        hooks = None
        if self.keep_tape:
            hooks = {f"block{block.index}.{name}" for block in self.stack for name in TELEMETRY_HOOKS}
        logits, tape = self.forward(ids, training=True, tape=True, hooks=hooks)
        loss, dlogits = self.objective(logits, targets, gradient=True, xp=self.xp)
        dlogits *= scale
        tape.ctx.scale = scale
        self.backward(dlogits, tape)
        if self.keep_tape:
            self.last_tape = tape
        return loss + float(tape.ctx.aux)

    def telemetry(self, tape, sample=2):
        """Cheap per-component statistics from a forward tape plus current gradients (call before the update)."""
        xp, to_cpu = self.xp, self.backend.to_cpu
        names = list(self.g)
        norms = to_cpu(xp.stack([xp.sum(self.g[n] * self.g[n], dtype=xp.float64) for n in names]))
        squared = dict(zip(names, (float(v) for v in norms)))

        def grad_norm(slots, block=None):
            return math.sqrt(sum(squared[n] for n in names if self.owner[n][0] in slots and self.owner[n][1] == block))

        def rms(array):
            return float(xp.sqrt((array[:sample] ** 2).mean()))

        got = tape.captured
        result = {"embedding": {"grad_norm": grad_norm(("embedding", "position")), "rms": rms(self._embedded(tape, sample))},
                  "blocks": [], "output": {"grad_norm": math.sqrt(sum(squared[n] for n in names if n.startswith(("final.", "head."))))}}
        for block in self.stack:
            prefix = f"block{block.index}."
            own = [n for n in names if n.startswith(prefix)]
            first = [n for n in own if n.startswith(prefix + "n1.") or (block.mixer is not None and n[len(prefix):] in block.mixer.params)]
            attention = mlp = None
            if prefix + "attn.pattern" in got:
                probabilities = got[prefix + "attn.pattern"][:sample]
                entropy = -(probabilities * xp.log(probabilities + 1e-12)).sum(axis=-1).mean(axis=(0, 2))
                attention = {"entropy": [float(v) for v in to_cpu(entropy)], "grad_norm": math.sqrt(sum(squared[n] for n in first))}
            if prefix + "mlp.act" in got:
                activity = xp.abs(got[prefix + "mlp.act"][:sample, :, :48]).mean(axis=(0, 1))
                peak = float(activity.max())
                mlp = {"neurons": [float(v) / peak if peak > 0 else 0. for v in to_cpu(activity)],
                       "active_fraction": float((got[prefix + "mlp.pre"][:sample] > 0).mean()),
                       "grad_norm": math.sqrt(sum(squared[n] for n in own if n not in first))}
            result["blocks"].append({"id": block.id, "index": block.index, "residual_rms": rms(got[prefix + "resid.post"]),
                                     "attention": attention, "mlp": mlp,
                                     "mixer": {"id": block.mixer.module.short if block.mixer is not None else "none", "grad_norm": math.sqrt(sum(squared[n] for n in first))},
                                     "channel": {"id": block.channel.module.short if block.channel is not None else "none",
                                                 "grad_norm": math.sqrt(sum(squared[n] for n in own if n not in first))}})
        return result

    def _embedded(self, tape, sample):
        ctx = Ctx(self.xp, position=self.position, embedding=self.embedding)
        x = self.embedding.forward(tape.ids[:sample], ctx)
        return self.position.forward(x, ctx) if self.position is not None else x

    def load_weights(self, weights):
        if set(weights) != set(self.p):
            raise ValueError("Checkpoint parameter names do not match the architecture.")
        for key, value in weights.items():
            value = self.xp.asarray(value)
            if value.shape != self.p[key].shape or not bool(self.xp.isfinite(value).all()):
                raise ValueError(f"Invalid checkpoint parameter: {key}")
            self.p[key][...] = value


def transplant(old_params, model):
    """Copy every tensor whose name and shape match into the model; return {"copied": [...], "fresh": [...]}.

    A layer whose shapes changed (a new head count keeps its shapes, a new d_ff does not) stays freshly
    initialized as a whole, so it never runs on half-old, half-new weights.
    """
    broken = {model.group[name] for name, value in model.p.items()
              if name in old_params and tuple(old_params[name].shape) != tuple(value.shape)}
    copied, fresh = [], []
    for name, value in model.p.items():
        if name in old_params and model.group[name] not in broken and tuple(old_params[name].shape) == tuple(value.shape):
            value[...] = model.xp.asarray(old_params[name])
            copied.append(name)
        else:
            fresh.append(name)
    return {"copied": copied, "fresh": fresh}
