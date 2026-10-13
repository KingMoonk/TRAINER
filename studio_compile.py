"""Compile a ModelSpec into a stack of Layers, and run it forward and backward with PyTorch autograd.

The block template is written once here: x + mixer(norm(x)), then x + channel(norm(x)) (or the post-norm
variant). Everything else is a module. Parameters are plain tensors in `p`; a forward pass that needs
gradients swaps in detached leaf views, and backward() accumulates their gradients into the `g` buffers,
so `p` and `g` are stable dicts that consumers (optimizers, telemetry, snapshots) may read and edit freely.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
import torch

import studio_registry as registry
from studio_backend import Backend, get_backend
from studio_slots import Ctx, Dims, as_tensor
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
    logits: object                      # the graph output; backward() differentiates it
    live: dict                          # parameter name -> leaf tensor the graph was built on
    captured: dict
    forward_values: dict = field(default_factory=dict)
    backward_values: dict = field(default_factory=dict)


class CompiledModel:
    """A ModelSpec made runnable. `p` and `g` map full parameter names to the layers' own tensors."""

    def __init__(self, spec, vocabulary, seed=42, device="auto", batch_size=4):
        self.spec = spec = validate_spec(spec)
        problems = errors(spec, vocabulary)
        if problems:
            raise ValueError(problems[0])
        d = spec["dims"]
        self.backend = get_backend(device, work=batch_size * d["n_ctx"] * d["d_model"] * max(1, d["n_layer"]))
        precision, precision_cfg = module_of("precision", spec["train"]["precision"])
        dtype = precision.build(precision_cfg, None, None)
        if self.backend.device == "mps" and dtype == torch.float64:
            self.backend = Backend(torch, "cpu", "PyTorch / CPU", "float64 is not available on Apple GPUs; using the CPU")
        self.xp = torch
        self.device = self.backend.device
        # Keeping the stochastic stream on CPU makes checkpoints and RNG state portable between devices.
        self.rng = np.random.default_rng(seed)
        self.dims = Dims(d["d_model"], d["n_layer"], d["n_ctx"], int(vocabulary), d["dropout"], 0, torch, dtype, self.device)
        init, init_cfg = module_of("init", spec["train"]["init"])
        self._init = init.build(init_cfg, self.dims, None)
        objective, objective_cfg = module_of("objective", spec["train"]["objective"])
        self.objective = objective.build(objective_cfg, self.dims, None)
        self.p, self.g, self.roles, self.lr_scale, self.owner, self.group = {}, {}, {}, {}, {}, {}
        self._slots = {}        # full name -> (layer, local name)
        self.elementwise = set()
        self.layers = []
        self.layer_names = {}
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
        self.layer_names[id(layer)] = prefix + slot
        self._init.apply(layer, self.rng, self.dims)
        for name, value in layer.params.items():
            full = prefix + name
            if full in self.p:
                raise ValueError(f"Two modules both name a parameter {full!r}; give one a more specific name.")
            self.p[full], self.g[full], self.roles[full] = value, layer.grads[name], layer.roles[name]
            self.owner[full], self.group[full] = (slot, block), len(self.layers)
            self._slots[full] = (layer, name)
            if name in layer.lr_scale:
                self.lr_scale[full] = layer.lr_scale[name]
            if name in layer.elementwise:
                self.elementwise.add(full)
        return layer

    @property
    def parameter_count(self):
        return sum(value.numel() for value in self.p.values())

    def zero_grad(self):
        for grad in self.g.values():
            grad.zero_()

    def _ids(self, ids):
        if isinstance(ids, torch.Tensor):
            return ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(np.asarray(ids), dtype=torch.long, device=self.device)

    def forward(self, ids, training=False, tape=False, hooks=None, patches=None, attention_summary=None, trace=False, cache=None):
        """Logits [B, T, V]. tape=True also returns the Tape backward() needs.

        hooks: True, or a set of hook names ("block0.attn.pattern", "logits") to record on the tape.
        cache: {"n": 0} makes this a KV-cached pass (see can_cache): it appends to the cache and continues at index cache["n"].
        patches: {hook name: function(value) -> value} to intervene on a hook that feeds the computation.
        """
        ids = self._ids(ids)
        if ids.shape[1] + (cache["n"] if cache else 0) > self.dims.n_ctx:
            raise ValueError("Sequence exceeds the context length.")
        ctx = Ctx(torch, training, self.rng, self.dims.dropout, hooks, patches, self.position, self.embedding, self.device, cache)
        if attention_summary is not None:
            if training or tape or patches or hooks:
                raise ValueError("Attention summaries require unpatched inference without a tape or dense hooks.")
            ctx.attention_summary = attention_summary
            ctx.summaries = {}
        graph = training or tape
        live = {}
        if graph:
            for full, (layer, name) in self._slots.items():
                live[full] = layer.params[name] = self.p[full].detach().requires_grad_(True)
        try:
            with torch.enable_grad() if graph else torch.no_grad():
                result = self._run(ids, ctx, trace, attention_summary is not None)
        finally:
            for full, (layer, name) in self._slots.items():
                layer.params[name] = self.p[full]
        if attention_summary is not None:
            return result
        logits, values, backward_values = result
        if cache is not None:
            cache["n"] += ids.shape[1]
        if tape:
            return logits, Tape(ids, ctx, logits, live, ctx.captured, values, backward_values)
        return logits.detach() if graph else logits

    @property
    def can_cache(self):
        """True when every mixer and the position module support a KV cache; otherwise generation re-runs the whole window."""
        return all(block.mixer is None or block.mixer.cacheable for block in self.stack) and (self.position is None or self.position.cacheable)

    def _run(self, ids, ctx, trace, summary):
        values, backward_values = {}, {}

        def run(layer, x):
            name = self.layer_names[id(layer)]
            if trace and x.is_floating_point() and x.requires_grad:
                x = x.view_as(x)   # a private alias, so its gradient is what flows through this layer alone
                x.register_hook(lambda g, n=name: backward_values.__setitem__(n, g.detach().clone()))
            y = layer.forward(x, ctx)
            if trace:
                values[name] = y.detach().clone()
            return y

        def drop(x):
            x, _ = ctx.drop(x)
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
            if summary:
                from studio_summary import bin_edges
                edges = bin_edges(ids.shape[1], ctx.attention_summary)
                norms = torch.sqrt((x * x).sum(dim=-1))
                ctx.summaries[f"block{block.index}.resid.norm"] = torch.stack([norms[:, a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
        ctx.block = None
        if self.final is not None:
            x = run(self.final, x)
        x = ctx.emit("final", x)
        logits = ctx.emit("logits", run(self.head, x))
        if summary:
            return logits, ctx.summaries
        return logits, values, backward_values

    def _accumulate(self, outputs, grad_outputs, tape):
        names = [name for name, leaf in tape.live.items()]
        leaves = [tape.live[name] for name in names]
        grads = torch.autograd.grad(outputs, leaves, grad_outputs=grad_outputs, retain_graph=True, allow_unused=True)
        with torch.no_grad():
            for name, grad in zip(names, grads):
                if grad is not None:
                    self.g[name].add_(grad)

    def backward(self, dlogits, tape):
        """Accumulate parameter gradients for any loss expressed as d(loss)/d(logits).

        The auxiliary loss a module added (MoE balance) contributes tape.ctx.scale × its gradient.
        """
        outputs, grad_outputs = [tape.logits], [as_tensor(dlogits, like=tape.logits)]
        aux = tape.ctx.aux
        if isinstance(aux, torch.Tensor) and aux.requires_grad and tape.ctx.scale:
            outputs.append(aux)
            grad_outputs.append(torch.as_tensor(float(tape.ctx.scale), dtype=aux.dtype, device=aux.device))
        self._accumulate(outputs, grad_outputs, tape)

    def backward_loss(self, loss, tape):
        """Accumulate parameter gradients of a scalar loss tensor built from this tape's logits."""
        self._accumulate([loss], [None], tape)

    def loss(self, x, y):
        """Plain next-token cross-entropy (no regularizers) as a float, for evaluation."""
        from studio_modules.objective.ce import cross_entropy
        return float(cross_entropy(self.forward(x), y))

    def loss_and_backward(self, ids, targets, scale=1.0):
        hooks = None
        if self.keep_tape:
            hooks = {f"block{block.index}.{name}" for block in self.stack for name in TELEMETRY_HOOKS}
        logits, tape = self.forward(ids, training=True, tape=True, hooks=hooks)
        loss = self.objective(logits, targets)
        total = (loss + tape.ctx.aux) * scale
        self.backward_loss(total, tape)
        if self.keep_tape:
            self.last_tape = tape
        return float(loss.detach()) + float(torch.as_tensor(tape.ctx.aux).detach())

    @torch.no_grad()
    def telemetry(self, tape, sample=2):
        """Cheap per-component statistics from a forward tape plus current gradients (call before the update)."""
        to_cpu = self.backend.to_cpu
        names = list(self.g)
        norms = torch.stack([(self.g[n].float() * self.g[n].float()).sum() for n in names]).tolist()
        squared = dict(zip(names, norms))

        def grad_norm(slots, block=None):
            return math.sqrt(sum(squared[n] for n in names if self.owner[n][0] in slots and self.owner[n][1] == block))

        def rms(array):
            return float(torch.sqrt((array[:sample].float() ** 2).mean()))

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
                entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum(dim=-1).mean(dim=(0, 2))
                attention = {"entropy": [float(v) for v in to_cpu(entropy)], "grad_norm": math.sqrt(sum(squared[n] for n in first))}
            if prefix + "mlp.act" in got:
                activity = got[prefix + "mlp.act"][:sample, :, :48].abs().mean(dim=(0, 1))
                peak = float(activity.max())
                mlp = {"neurons": [float(v) / peak if peak > 0 else 0. for v in to_cpu(activity)],
                       "active_fraction": float((got[prefix + "mlp.pre"][:sample] > 0).float().mean()),
                       "grad_norm": math.sqrt(sum(squared[n] for n in own if n not in first))}
            result["blocks"].append({"id": block.id, "index": block.index, "residual_rms": rms(got[prefix + "resid.post"]),
                                     "attention": attention, "mlp": mlp,
                                     "mixer": {"id": block.mixer.module.short if block.mixer is not None else "none", "grad_norm": math.sqrt(sum(squared[n] for n in first))},
                                     "channel": {"id": block.channel.module.short if block.channel is not None else "none",
                                                 "grad_norm": math.sqrt(sum(squared[n] for n in own if n not in first))}})
        return result

    def _embedded(self, tape, sample):
        ctx = Ctx(torch, position=self.position, embedding=self.embedding, device=self.device)
        x = self.embedding.forward(tape.ids[:sample], ctx)
        return self.position.forward(x, ctx) if self.position is not None else x

    @torch.no_grad()
    def load_weights(self, weights):
        if set(weights) != set(self.p):
            raise ValueError("Checkpoint parameter names do not match the architecture.")
        for key, value in weights.items():
            value = as_tensor(value, like=self.p[key])
            if tuple(value.shape) != tuple(self.p[key].shape) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid checkpoint parameter: {key}")
            self.p[key].copy_(value)

    def state_arrays(self):
        """Every parameter as its own NumPy copy (never a view of the live tensor), for checkpoints and inspection."""
        return {key: self.backend.to_cpu(value).copy() for key, value in self.p.items()}


@torch.no_grad()
def transplant(old_params, model):
    """Copy every tensor whose name and shape match into the model; return {"copied": [...], "fresh": [...]}.

    A layer whose shapes changed (a new head count keeps its shapes, a new d_ff does not) stays freshly
    initialized as a whole, so it never runs on half-old, half-new weights.
    """
    broken = {model.group[name] for name, value in model.p.items()
              if name in old_params and tuple(np.shape(old_params[name])) != tuple(value.shape)}
    copied, fresh = [], []
    for name, value in model.p.items():
        if name in old_params and model.group[name] not in broken and tuple(np.shape(old_params[name])) == tuple(value.shape):
            value.copy_(as_tensor(old_params[name], like=value))
            copied.append(name)
        else:
            fresh.append(name)
    return {"copied": copied, "fresh": fresh}
