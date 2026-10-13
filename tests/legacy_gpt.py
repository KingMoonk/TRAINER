"""The GPT engine as it was before modules existed, frozen as the reference for tests/test_parity.py.

The compiled default spec must reproduce this model: same seed, same data, same loss curve.
Do not edit; delete it once nobody doubts the modules any more.
"""
import math

import numpy as np

from studio_arch import parameter_names, spec_from_config
from studio_model import validate_config


class _NumpyBackend:
    """The frozen reference always runs on NumPy, whatever the live engine uses."""
    xp, device, name, reason = np, "cpu", "NumPy / CPU", ""

    def to_cpu(self, value):
        return np.asarray(value)

    def synchronize(self):
        pass


def effective_spec(cfg):
    return cfg.get("architecture") or spec_from_config(cfg)


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
        self.backend = _NumpyBackend()
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
            if not key.endswith((".g", ".b")):  # no decay on normalisation gains and biases
                parameter *= 1 - learning_rate * self.weight_decay
            parameter -= learning_rate * m / (self.xp.sqrt(v) + 1e-8)
        return norm
