"""Post-training with preferences: readiness checks, DPO/IPO, and an Ollama teacher.

Pretraining teaches the model to imitate text. Post-training uses comparisons
("this answer is better than that one") to push the model toward preferred
behaviour. Pairs come from a human reviewer or from a local Ollama teacher, and
the policy is optimized with Direct Preference Optimization (DPO, or its
squared-loss cousin IPO): no separate reward model is needed and the frozen
pretrained model anchors the update.

The automated loop is best-of-N: the model writes N answers, the teacher grades
them against an editable rubric, and the best/worst become a preference pair.
A new checkpoint is only kept if it beats the previous one head-to-head.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from studio_core import atomic_json, messages_of, now, replace_file, uid
from studio_model import (ASSISTANT, BOS, EOS, SYSTEM, USER, AdamW, LocalModel, ScratchTransformer, Tokenizer,
                          batch_arrays, load_checkpoint, samples_from)

DPO_DEFAULTS = {"beta": .1, "learning_rate": 1e-4, "steps": 100, "batch_size": 4, "sft_weight": .2,
                "validation": .1, "eval_every": 10, "device": "auto", "seed": 7, "weight_decay": 0.,
                "length_normalize": False, "loss_type": "dpo"}
LOSS_TYPES = ("dpo", "ipo")
AUTO_DEFAULTS = {"rounds": 3, "prompts_per_round": 24, "dpo_steps": 60, "eval_prompts": 8, "candidates": 4,
                 "temperatures": None, "max_tokens": 120, "rewrite_below": 4, "min_margin": 1, "tiebreak_gap": .5,
                 "extra_pairs": False, "both_orders": True, "keep_only_improvements": True, "replay": 200,
                 "gate_samples": 2, "win_margin": .05, "gate_temperature": .4, "tie_eps": .5, "max_failures": 8}
ROLE_TOKENS = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}
MIN_PAIRS_TO_TRAIN = 20

DEFAULT_RUBRIC = [
    {"text": "The reply is correct for what was asked.", "weight": 3},
    {"text": "It does exactly what the instruction says.", "weight": 3},
    {"text": "It is concise: no padding, repetition or rambling.", "weight": 1},
    {"text": "It is readable, well-formed language.", "weight": 1},
    {"text": "It is not gibberish, off-topic or broken text.", "weight": 2},
]
SETTINGS_DEFAULTS = {"url": "http://localhost:11434", "teacher_model": "", "judge_model": "", "rubric": DEFAULT_RUBRIC}


# --------------------------------------------------------------------------- rubric and settings

def normalize_rubric(items):
    """Clean a list of {"text", "weight"} criteria: drop blanks, clamp weights to 0.5-10."""
    rubric = []
    for item in items or []:
        text = str(item.get("text", "") if isinstance(item, dict) else item).strip()
        if not text:
            continue
        try:
            weight = float(item.get("weight", 1)) if isinstance(item, dict) else 1.
        except (TypeError, ValueError):
            weight = 1.
        weight = max(.5, min(10., weight))
        rubric.append({"text": text[:300], "weight": int(weight) if weight == int(weight) else round(weight, 1)})
    return rubric


def rubric_text(rubric):
    rubric = normalize_rubric(rubric) or normalize_rubric(DEFAULT_RUBRIC)
    return "\n".join(f"- {c['text']} (weight {c['weight']})" for c in rubric)


def settings_path(root):
    return Path(root) / "data" / "rlhf_settings.json"


def load_settings(root):
    """Workspace RLHF settings (rubric, teacher/judge models, loop options), merged over defaults."""
    settings = json.loads(json.dumps(SETTINGS_DEFAULTS))
    try:
        stored = json.loads(settings_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = {}
    if isinstance(stored, dict):
        for key, value in stored.items():
            settings[key] = value
    settings["rubric"] = normalize_rubric(settings.get("rubric")) or normalize_rubric(DEFAULT_RUBRIC)
    return settings


def save_settings(root, settings):
    clean = dict(settings)
    clean["rubric"] = normalize_rubric(clean.get("rubric")) or normalize_rubric(DEFAULT_RUBRIC)
    atomic_json(settings_path(root), clean)
    return clean


# --------------------------------------------------------------------------- readiness

def evaluate_checkpoint(path, examples, device="auto", limit=200, stop=None):
    """Held-out style metrics on assistant tokens: loss, perplexity and next-token error rate."""
    data = load_checkpoint(path)
    cfg = dict(data["config"], device=device, objective="assistant", dropout=0)
    tokenizer = Tokenizer.restore(data["tokenizer"])
    model = ScratchTransformer(cfg, tokenizer.vocab_size)
    model.load_weights(data["model"])
    samples = samples_from(examples[:limit], tokenizer, cfg)
    if not samples:
        raise ValueError("These examples contain no assistant tokens to evaluate.")
    total_loss, errors, tokens, stopped = 0., 0, 0, False
    for start in range(0, len(samples), 16):
        if stop is not None and stop.is_set():
            stopped = True
            break
        x, y = batch_arrays(samples[start:start + 16])
        with torch.no_grad():
            logits = model.forward(x)
            labels = torch.as_tensor(y, dtype=torch.long, device=logits.device)
            valid = labels != -100
            safe = torch.where(valid, labels, torch.zeros_like(labels))
            picked = torch.log_softmax(logits, dim=-1).gather(-1, safe[..., None])[..., 0]
            total_loss += float((-picked * valid).sum())
            errors += int(((logits.argmax(dim=-1) != labels) & valid).sum())
            tokens += int(valid.sum())
    loss = total_loss / max(1, tokens)
    unknown = sum(1 for e in examples[:limit] for m in messages_of(e) if m["role"] == "assistant"
                  for t in tokenizer.encode(m["content"]) if t == 6)
    return {"loss": loss, "perplexity": math.exp(min(loss, 50)), "token_error": errors / max(1, tokens),
            "token_accuracy": 1 - errors / max(1, tokens), "tokens": tokens, "examples": min(limit, len(examples)),
            "unknown_tokens": unknown, "step": data["step"], "name": data["config"].get("name", "model"), "stopped": stopped}


# --------------------------------------------------------------------------- preference store

def pair_key(record):
    """Content identity of a preference (prompt + chosen + rejected), ignoring ids and metadata."""
    body = {"m": [[m.get("role"), (m.get("content") or "").strip()] for m in record.get("messages", [])],
            "c": (record.get("chosen") or "").strip(), "r": (record.get("rejected") or "").strip()}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _valid_record(row):
    return (isinstance(row, dict) and isinstance(row.get("messages"), list) and row["messages"]
            and all(isinstance(m, dict) and isinstance(m.get("content"), str) and m.get("role") in ROLE_TOKENS for m in row["messages"])
            and isinstance(row.get("chosen"), str) and isinstance(row.get("rejected"), str)
            and row["chosen"].strip() and row["chosen"].strip() != row["rejected"].strip())


class PreferenceStore:
    """JSONL of comparisons; each line is one (prompt, chosen, rejected) pair.

    New pairs are appended; edits, flips and deletes rewrite the file atomically
    (temp file + replace) so a crash never leaves a half-written store.
    """

    def __init__(self, root):
        self.path = Path(root) / "data" / "preferences.jsonl"
        self.lock = threading.RLock()

    # -- reading
    def _lines(self):
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").split("\n"):  # not splitlines(): it also splits on U+2028 inside a record
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                out.append((row if _valid_record(row) else None, line))
            except ValueError:
                out.append((None, line))  # kept verbatim on rewrites, ignored by readers
        return out

    @staticmethod
    def kind_of(row):
        return "human" if row.get("source") == "human" else "teacher"

    def all(self, sources=None):
        with self.lock:
            rows = [row for row, _ in self._lines() if row is not None]
        return [r for r in rows if sources is None or self.kind_of(r) in sources]

    def get(self, pair_id):
        return next((r for r in self.all() if r.get("id") == pair_id), None)

    def __len__(self):
        return len(self.all())

    # -- writing
    def _write(self, lines):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(line.rstrip("\n") + "\n" for line in lines), encoding="utf-8")
        replace_file(temporary, self.path)

    def add(self, messages, chosen, rejected, source, **meta):
        if not chosen.strip() or chosen.strip() == rejected.strip():
            raise ValueError("A preference needs a non-empty chosen answer that differs from the rejected one.")
        record = {"id": uid("pref"), "createdAt": now(), "messages": [dict(role=m["role"], content=m["content"]) for m in messages],
                  "chosen": chosen, "rejected": rejected, "source": source, **meta}
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a+b") as stream:
                size = stream.seek(0, 2)
                needs_break = size and (stream.seek(-1, 2), stream.read(1))[1] != b"\n"  # a file lacking its final newline would glue two records together
                stream.write((b"\n" if needs_break else b"") + (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
        return record

    def update(self, pair_id, **fields):
        """Change fields of one pair (chosen, rejected, messages, kind, note …). Returns the new record."""
        for protected in ("id", "createdAt"):
            fields.pop(protected, None)
        with self.lock:
            lines, updated = self._lines(), None
            for index, (row, _) in enumerate(lines):
                if row is not None and row.get("id") == pair_id:
                    updated = {**row, **fields, "editedAt": now()}
                    if not _valid_record(updated):
                        raise ValueError("A preference needs a non-empty chosen answer that differs from the rejected one.")
                    lines[index] = (updated, json.dumps(updated, ensure_ascii=False))
                    break
            if updated is None:
                raise KeyError(pair_id)
            self._write([line for _, line in lines])
        return updated

    def flip(self, pair_id):
        """Swap chosen and rejected (you changed your mind)."""
        row = self.get(pair_id)
        if row is None:
            raise KeyError(pair_id)
        return self.update(pair_id, chosen=row["rejected"], rejected=row["chosen"], flipped=not row.get("flipped", False))

    def delete(self, ids):
        ids = {ids} if isinstance(ids, str) else set(ids)
        with self.lock:
            lines = self._lines()
            kept = [(row, line) for row, line in lines if not (row is not None and row.get("id") in ids)]
            removed = len(lines) - len(kept)
            if removed:
                self._write([line for _, line in kept])
        return removed

    def export(self, path, ids=None):
        """Write pairs to a JSONL file others can import. Returns how many were written."""
        rows = [r for r in self.all() if ids is None or r.get("id") in set(ids)]
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        replace_file(temporary, target)
        return len(rows)

    def import_file(self, path):
        """Merge a JSONL export, skipping pairs whose content already exists. Returns counts."""
        added, duplicates, invalid = 0, 0, 0
        with self.lock:
            existing = self._lines()
            known_keys = {pair_key(r) for r, _ in existing if r is not None}
            known_ids = {r.get("id") for r, _ in existing if r is not None}
            fresh = []
            for line in Path(path).read_text(encoding="utf-8").split("\n"):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    invalid += 1
                    continue
                if not _valid_record(row):
                    invalid += 1
                    continue
                key = pair_key(row)
                if key in known_keys:
                    duplicates += 1
                    continue
                known_keys.add(key)
                if not row.get("id") or row["id"] in known_ids:
                    row["id"] = uid("pref")
                known_ids.add(row["id"])
                row.setdefault("createdAt", now())
                row.setdefault("source", "import")
                row["imported"] = True
                fresh.append(json.dumps(row, ensure_ascii=False))
                added += 1
            if fresh:
                self._write([line for _, line in existing] + fresh)
        return {"added": added, "duplicates": duplicates, "invalid": invalid}

    import_pairs = import_file

    def stats(self):
        rows = self.all()
        human = sum(r.get("source") == "human" for r in rows)
        return {"total": len(rows), "human": human, "teacher": len(rows) - human,
                "rewrites": sum(r.get("kind") in {"rewrite", "edit"} for r in rows),
                "by_kind": dict(Counter(r.get("kind", "comparison") for r in rows)),
                "by_source": dict(Counter(r.get("source", "?") for r in rows))}


def prompts_from(examples):
    """Conversation prefixes ending in a user turn, with the dataset answer kept as a reference."""
    prompts = []
    for example in examples:
        turns = messages_of(example)
        if example.get("text") or not turns or turns[-1]["role"] != "assistant":
            continue
        prompts.append({"messages": turns[:-1], "reference": turns[-1]["content"], "source_id": example.get("id")})
    return prompts


# --------------------------------------------------------------------------- DPO / IPO

def _encode_pair(tokenizer, messages, response, context):
    prefix = [BOS]
    for message in messages:
        if message["role"] in ROLE_TOKENS:
            prefix += [ROLE_TOKENS[message["role"]]] + tokenizer.encode(message["content"]) + [EOS]
    prefix.append(ASSISTANT)
    answer = tokenizer.encode(response) + [EOS]
    window = context + 1
    # Keep the answer (what DPO scores) and as much recent prompt context as fits.
    answer = answer[:max(1, window - 8)]
    prefix = prefix[-(window - len(answer)):]
    tokens = prefix + answer
    targets = [-100] * (len(prefix) - 1) + answer
    return tokens[:-1], targets


def _sequence_logps(model, x, y, tape=False):
    """Summed log-probability of the target tokens per sequence; with tape=True the result carries gradients."""
    out = model.forward(x, training=False, tape=tape)
    logits, cache = out if tape else (out, None)
    labels = torch.as_tensor(np.asarray(y), dtype=torch.long, device=logits.device)
    valid = labels != -100
    safe = torch.where(valid, labels, torch.zeros_like(labels))
    token_logp = torch.log_softmax(logits, dim=-1).gather(-1, safe[..., None])[..., 0]
    logps = (token_logp * valid).sum(dim=1)
    return logps, (token_logp, safe, valid, cache)


def token_logps(model, x, y, tape=False):
    """Per-token log-probabilities [B, T] (zero where the target is -100) and the valid mask; gradients when tape=True."""
    logps, (token_logp, _, valid, cache) = _sequence_logps(model, x, y, tape=tape)
    return token_logp * valid, valid, cache


def preference_terms(policy_logp, reference_logp, lengths, count, cfg):
    """Preference loss from summed sequence log-probs (chosen rows first, then rejected).

    Returns (metrics, g) where g is d(loss)/d(policy_logp) per sequence.
      raw = Δ_chosen − Δ_rejected with Δ = (policy − reference) logp, optionally divided by answer length
      dpo: loss = softplus(−β·raw)           ipo: loss = (raw − 1/(2β))²
    plus sft_weight × mean negative per-token log-likelihood of the chosen answers.
    """
    beta = float(cfg["beta"])
    lengths = np.maximum(1., np.asarray(lengths, dtype=np.float64))
    norm = lengths if cfg.get("length_normalize") else np.ones_like(lengths)
    delta = (policy_logp - reference_logp) / norm
    raw = delta[:count] - delta[count:]
    z = beta * raw
    if cfg.get("loss_type", "dpo") == "ipo":
        residual = raw - 1 / (2 * beta)
        per_pair, d_raw = residual ** 2, 2 * residual
    else:
        per_pair, d_raw = np.logaddexp(0, -z), (0.5 * (1 + np.tanh(z / 2)) - 1) * beta
    loss = float(np.mean(per_pair))
    g = np.zeros(2 * count)
    g[:count] = d_raw / (norm[:count] * count)
    g[count:] = -d_raw / (norm[count:] * count)
    sft = float(cfg.get("sft_weight") or 0)
    if sft:
        loss += sft * float(np.mean(-policy_logp[:count] / lengths[:count]))
        g[:count] -= sft / (lengths[:count] * count)
    metrics = {"loss": loss, "reward_accuracy": float(np.mean(z > 0)), "margin": float(np.mean(z)),
               "chosen_reward": float(beta * np.mean(delta[:count])), "rejected_reward": float(beta * np.mean(delta[count:])),
               "chosen_len": float(np.mean(lengths[:count])), "rejected_len": float(np.mean(lengths[count:]))}
    return metrics, g


def preference_batch(policy, reference, rows, count, cfg, backward=False):
    """Score `rows` (count chosen then count rejected, encoded) and optionally accumulate gradients into `policy`."""
    x, y = batch_arrays(rows)
    policy_logp, (_, _, valid, cache) = _sequence_logps(policy, x, y, tape=backward)
    with torch.no_grad():
        reference_logp, _ = _sequence_logps(reference, x, y)
    pl = policy.backend.to_cpu(policy_logp).astype(np.float64)
    rl = reference.backend.to_cpu(reference_logp).astype(np.float64)
    lengths = policy.backend.to_cpu(valid.sum(dim=1)).astype(np.float64)
    result, g = preference_terms(pl, rl, lengths, count, cfg)
    if backward:
        # d(loss)/d(policy_logp) per sequence is g; autograd carries it back through the log-softmax and the model.
        weights = torch.as_tensor(g, dtype=policy_logp.dtype, device=policy_logp.device)
        policy.backward_loss((policy_logp * weights).sum(), cache)
    return result


def _save(path, model, optimizer, cfg, tokenizer, step, best, history, extra):
    metadata = {"architecture": "groundwork_numpy_decoder", "format_version": 1, "config": cfg,
                "tokenizer": tokenizer.serialize(), "step": step, "best": best if best is not None and math.isfinite(best) else None,
                "optimizer_steps": optimizer.steps, "history": history, "data_hash": extra.get("data_hash", ""),
                "sampler_state": None, "model_rng": model.rng.bit_generator.state, "device": model.backend.device,
                "split_strategy": "preferences", "split_hash": "", **extra}
    arrays = {"metadata": np.array(json.dumps(metadata))}
    arrays.update({"weight::" + k: model.backend.to_cpu(v) for k, v in model.p.items()})
    arrays.update({"adam_m::" + k: model.backend.to_cpu(v) for k, v in optimizer.m.items()})
    arrays.update({"adam_v::" + k: model.backend.to_cpu(v) for k, v in optimizer.v.items()})
    temporary = Path(path).with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    replace_file(temporary, path)


def dpo_train(base_checkpoint, pairs, output_dir, config=None, emit=None, stop=None, pause=None, name=None):
    """Optimize a checkpoint on preference pairs with DPO or IPO (+ optional SFT on chosen answers)."""
    cfg = {**DPO_DEFAULTS, **(config or {})}
    emit = emit or (lambda event: None)
    stop, pause = stop or threading.Event(), pause or threading.Event()
    if len(pairs) < 2:
        raise ValueError("Collect at least two preference pairs before post-training.")
    for key, low, high in [("beta", .001, 10), ("learning_rate", 1e-7, .01), ("sft_weight", 0, 10), ("validation", 0, .5)]:
        if not low <= float(cfg[key]) <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
    for key, low, high in [("steps", 1, 100000), ("batch_size", 1, 64), ("eval_every", 1, 100000)]:
        cfg[key] = int(cfg[key])
        if not low <= cfg[key] <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
    if cfg["loss_type"] not in LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {', '.join(LOSS_TYPES)}.")
    cfg["length_normalize"] = bool(cfg["length_normalize"])
    data = load_checkpoint(base_checkpoint)
    model_cfg = dict(data["config"], device=cfg["device"], dropout=0)
    tokenizer = Tokenizer.restore(data["tokenizer"])
    policy = ScratchTransformer(model_cfg, tokenizer.vocab_size)
    policy.load_weights(data["model"])
    reference = ScratchTransformer(model_cfg, tokenizer.vocab_size)
    reference.load_weights(data["model"])
    optimizer = AdamW(policy, cfg["weight_decay"])
    rng = random.Random(cfg["seed"])
    encoded = [(_encode_pair(tokenizer, p["messages"], p["chosen"], model_cfg["context"]),
                _encode_pair(tokenizer, p["messages"], p["rejected"], model_cfg["context"])) for p in pairs]
    order = list(range(len(encoded)))
    rng.shuffle(order)
    held = int(len(order) * cfg["validation"]) if len(order) >= 10 else 0
    val_ids, train_ids = order[:held], order[held:]
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256("".join(p.get("id", "") for p in pairs).encode()).hexdigest()
    base_name = name or f"{data['config'].get('name', 'model')} · {cfg['loss_type'].upper()}"
    record = {"name": base_name, "createdAt": now(), "status": "training", "stage": "dpo", "step": 0, "steps": cfg["steps"],
              "parameters": policy.parameter_count, "vocabulary": tokenizer.vocab_size, "device": policy.backend.device,
              "device_name": policy.backend.name, "base_checkpoint": str(base_checkpoint), "pairs": len(pairs),
              "train_pairs": len(train_ids), "validation_pairs": len(val_ids), "dpo": cfg, "best_loss": None}
    atomic_json(directory / "run.json", record)
    atomic_json(directory / "config.json", model_cfg)
    atomic_json(directory / "tokenizer.json", tokenizer.serialize())
    (directory / "preferences.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    metrics_file = directory / "metrics.jsonl"
    metrics_file.write_text("", encoding="utf-8")
    beta, history = float(cfg["beta"]), []
    flavour = cfg["loss_type"].upper() + (" · length-normalized" if cfg["length_normalize"] else "")
    emit({"type": "info", "text": f"{flavour} · {policy.backend.name} · {len(train_ids)} training pairs · {len(val_ids)} held out · β={beta}"})
    emit({"type": "history", "history": []})

    def stats(ids, backward=False):
        rows = [encoded[i][0] for i in ids] + [encoded[i][1] for i in ids]
        return preference_batch(policy, reference, rows, len(ids), cfg, backward)

    def evaluate():
        if not val_ids:
            return None
        totals, n = {}, 0
        for start in range(0, len(val_ids), cfg["batch_size"]):
            chunk = val_ids[start:start + cfg["batch_size"]]
            result = stats(chunk)
            for key, value in result.items():
                totals[key] = totals.get(key, 0) + value * len(chunk)
            n += len(chunk)
        return {key: value / n for key, value in totals.items()}

    best, step = float("inf"), 0
    extra = {"stage": "dpo", "base_checkpoint": str(base_checkpoint), "data_hash": digest}
    started = time.monotonic()
    try:
        first = evaluate()
        if first:
            row = {"step": 0, "loss": first["loss"], "val_loss": first["loss"], "val_reward_accuracy": first["reward_accuracy"], "lr": 0., "grad_norm": 0., "tokens_per_second": 0.,
                   "val_chosen_len": first["chosen_len"], "val_rejected_len": first["rejected_len"]}
            history.append(row)
            with metrics_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            emit({"type": "metric", **row, "total": cfg["steps"]})
        while step < cfg["steps"] and not stop.is_set():
            while pause.is_set() and not stop.is_set():
                time.sleep(.08)
            if stop.is_set():
                break
            tick = time.monotonic()
            batch = [train_ids[rng.randrange(len(train_ids))] for _ in range(min(cfg["batch_size"], len(train_ids)))]
            policy.zero_grad()
            result = stats(batch, backward=True)
            rate = cfg["learning_rate"] * min(1., (step + 1) / max(1, min(10, cfg["steps"] // 10)))
            norm = optimizer.update(policy, rate)
            step += 1
            policy.backend.synchronize()
            validation = evaluate() if step % cfg["eval_every"] == 0 or step == cfg["steps"] else None
            row = {"step": step, "loss": result["loss"], "val_loss": validation["loss"] if validation else None,
                   "reward_accuracy": result["reward_accuracy"], "margin": result["margin"],
                   "chosen_len": result["chosen_len"], "rejected_len": result["rejected_len"],
                   "val_reward_accuracy": validation["reward_accuracy"] if validation else None,
                   "val_chosen_len": validation["chosen_len"] if validation else None,
                   "val_rejected_len": validation["rejected_len"] if validation else None,
                   "lr": rate, "grad_norm": norm, "tokens_per_second": len(batch) * 2 / max(.001, time.monotonic() - tick)}
            history.append(row)
            with metrics_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            emit({"type": "metric", **row, "total": cfg["steps"]})
            score = validation["loss"] if validation else result["loss"]
            if (validation or not val_ids) and score < best:
                best = score
                _save(directory / "best.npz", policy, optimizer, model_cfg, tokenizer, data["step"] + step, best, history, extra)
            record.update(step=step, loss=result["loss"], best_loss=best if math.isfinite(best) else None)
            if step % 10 == 0:
                try:
                    atomic_json(directory / "run.json", record)
                except PermissionError:
                    pass  # progress metadata only; rewritten on the next update
        _save(directory / "last.npz", policy, optimizer, model_cfg, tokenizer, data["step"] + step, best, history, extra)
        if not (directory / "best.npz").exists():
            _save(directory / "best.npz", policy, optimizer, model_cfg, tokenizer, data["step"] + step, best, history, extra)
        final = evaluate()
        record.update(status="stopped" if stop.is_set() else "completed", step=step, finishedAt=now(),
                      duration_seconds=time.monotonic() - started, final_validation=final,
                      checkpoint=str(directory / "last.npz"))
        atomic_json(directory / "run.json", record)
        emit({"type": "done", "run": record, "directory": str(directory)})
        return {**record, "directory": str(directory)}
    except Exception as exc:
        record.update(status="failed", error=str(exc), step=step)
        atomic_json(directory / "run.json", record)
        raise


# --------------------------------------------------------------------------- Ollama teacher

JUDGE_SYSTEM = """You are a strict, fair grader of a small assistant model that is still learning.
Compare two candidate replies to the same conversation, using these criteria (a higher weight matters more):
{criteria}
Garbled or off-topic text scores low. Length is not a virtue.
Reply with JSON only: {"score_a": 1-10, "score_b": 1-10, "winner": "A" | "B" | "tie", "reason": "one short sentence"}"""
SCORE_SYSTEM = """You grade one reply from a small assistant model that is still learning, using these criteria (a higher weight matters more):
{criteria}
Garbled or off-topic text scores low. Length is not a virtue.
Reply with JSON only: {"score": 1-10, "reason": "one short sentence"}"""
BATCH_SYSTEM = """You grade several numbered replies to the same conversation, written by a small assistant model that is still learning.
Give every reply a score from 1 to 10 using these criteria (a higher weight matters more):
{criteria}
Garbled, repetitive or off-topic text scores low. Length is not a virtue. Use the whole scale so better replies clearly score higher than worse ones.
Reply with JSON only: {"scores": [one number per reply, in the order shown], "reason": "one short sentence"}"""
REWRITE_SYSTEM = """Write the ideal assistant reply to the conversation. Be correct and concise: at most {max_chars} characters.
It will be graded on these criteria:
{criteria}
Reply with JSON only: {"answer": "the reply text"}"""


def _transcript(messages):
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


class TeacherError(RuntimeError):
    """The teacher could not produce a usable reply. `fatal` means retrying is pointless (e.g. Ollama is down)."""

    def __init__(self, message, fatal=False):
        super().__init__(message)
        self.fatal = fatal


def _number(value):
    """A 1-10 grade from a teacher reply; anything else is a malformed reply."""
    if isinstance(value, bool) or value is None:
        raise ValueError("missing score")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"score is not a number: {str(value)[:40]}") from None
    if not math.isfinite(number):
        raise ValueError("score is not finite")
    return max(1., min(10., number))


class OllamaTeacher:
    def __init__(self, url="http://localhost:11434", model="", timeout=240, rubric=None, judge_model=None,
                 retries=2, retry_delay=.4, stop=None):
        self.url, self.model, self.timeout, self.stop = url.rstrip("/"), model, timeout, stop
        self.rubric = normalize_rubric(rubric) or normalize_rubric(DEFAULT_RUBRIC)
        self.judge_model = judge_model or None
        self.retries, self.retry_delay = int(retries), float(retry_delay)
        self.think_supported = True

    def _request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.url + path, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except ValueError:
            raise RuntimeError("Ollama sent a reply that could not be read.") from None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama error {exc.code}: {detail[:300]}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self.url}. Is it running? ({exc.reason})") from None
        except (TimeoutError, ConnectionError, OSError) as exc:
            raise RuntimeError(f"Ollama did not answer in time or dropped the connection ({exc})") from None

    def models(self):
        return sorted(m["name"] for m in self._request("/api/tags").get("models", []) if "embed" not in m["name"])

    def chat(self, system, user, json_mode=True, temperature=0.):
        if not self.model:
            raise ValueError("Choose an Ollama teacher model.")
        payload = {"model": self.model, "stream": False, "options": {"temperature": temperature},
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            payload["format"] = "json"
        if self.think_supported:
            payload["think"] = False
        try:
            reply = self._request("/api/chat", payload)
        except RuntimeError as exc:
            if "think" not in str(exc).lower() or not self.think_supported:
                raise
            self.think_supported = False
            payload.pop("think")
            reply = self._request("/api/chat", payload)
        text = re.sub(r"<think>.*?</think>", "", ((reply.get("message") or {}).get("content") or ""), flags=re.S).strip()
        if not json_mode:
            return text
        try:
            return json.loads(text)
        except ValueError:
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                try:
                    return json.loads(match.group())
                except ValueError:
                    pass
            raise RuntimeError(f"The teacher did not return JSON: {text[:200]}") from None

    # -- robust helpers -------------------------------------------------------------------

    def judge_teacher(self):
        """The object that grades: this teacher, or a sibling using the separate judge model."""
        if not self.judge_model or self.judge_model == self.model:
            return self
        return OllamaTeacher(self.url, self.judge_model, self.timeout, self.rubric, None, self.retries, self.retry_delay, self.stop)

    def _system(self, template, **values):
        text = template.replace("{criteria}", rubric_text(self.rubric))
        for key, value in values.items():
            text = text.replace("{" + key + "}", str(value))
        return text

    def ask(self, system, user, temperature=0., parse=None):
        """chat() with retries. `parse(dict)` validates/extracts; a ValueError from it counts as a malformed reply
        and is retried. Returns the dict (or parse's result) or raises TeacherError."""
        last = "no reply"
        for attempt in range(max(0, self.retries) + 1):
            if self.stop is not None and self.stop.is_set():
                raise TeacherError("Stopped.", fatal=True)
            try:
                result = self.chat(system, user, True, temperature)
                if not isinstance(result, dict):
                    raise ValueError("the teacher returned JSON that is not an object")
                return parse(result) if parse else result
            except Exception as exc:
                last = str(exc) or exc.__class__.__name__
                if last.startswith("Cannot reach Ollama") or "Choose an Ollama" in last:
                    raise TeacherError(last, fatal=True) from None
                if attempt < self.retries:
                    time.sleep(self.retry_delay * (attempt + 1))
                    temperature = max(temperature, .3)  # a slightly different sample often fixes a malformed reply
        raise TeacherError(f"Teacher reply unusable after {self.retries + 1} tries: {last[:200]}")

    @staticmethod
    def _score(value):
        try:
            return max(1., min(10., float(value)))
        except (TypeError, ValueError):
            return 5.

    def judge(self, messages, first, second, reference=None, both_orders=True, rng=None):
        """Return scores for (first, second). Randomized order plus a swapped pass reduce position bias."""
        rng = rng or random.Random()
        hint = f"\n\nA reference answer from the dataset (may be phrased differently):\n{reference}" if reference else ""
        system = self._system(JUDGE_SYSTEM)

        def parse(result):
            return _number(result.get("score_a")), _number(result.get("score_b")), str(result.get("reason", ""))[:300]

        def one(a, b):
            return self.ask(system, f"CONVERSATION\n{_transcript(messages)}{hint}\n\nREPLY A\n{a}\n\nREPLY B\n{b}", parse=parse)

        swap = rng.random() < .5
        a, b, reason = one(second, first) if swap else one(first, second)
        if swap:  # report the reason in terms of the caller's A/B order
            reason = re.sub(r"\b([AB])\b", lambda m: "B" if m.group(1) == "A" else "A", reason)
        scores = [b, a] if swap else [a, b]
        consistent = True
        if both_orders:
            a2, b2, reason2 = one(first, second) if swap else one(second, first)
            second_scores = [a2, b2] if swap else [b2, a2]
            if not swap:  # this pass was asked with the replies swapped
                reason2 = re.sub(r"\b([AB])\b", lambda m: "B" if m.group(1) == "A" else "A", reason2)
            consistent = (scores[0] - scores[1]) * (second_scores[0] - second_scores[1]) >= 0
            scores = [(scores[0] + second_scores[0]) / 2, (scores[1] + second_scores[1]) / 2]
            if not consistent:
                reason = f"Inconsistent across orderings: {reason} / {reason2}"
                scores = [min(scores), min(scores)]
        return {"scores": scores, "reason": reason, "consistent": consistent}

    def score(self, messages, reply, reference=None):
        hint = f"\n\nReference answer from the dataset:\n{reference}" if reference else ""
        return self.ask(self._system(SCORE_SYSTEM), f"CONVERSATION\n{_transcript(messages)}{hint}\n\nREPLY\n{reply}",
                        parse=lambda result: _number(result.get("score")))

    def score_many(self, messages, replies, reference=None, rng=None):
        """Grade N replies, ideally in one JSON call (shuffled to cancel position bias); falls back to one by one."""
        scores = [1.0 if not r.strip() else None for r in replies]
        live = [i for i, s in enumerate(scores) if s is None]
        if len(live) == 1:
            scores[live[0]] = self.score(messages, replies[live[0]], reference)
        elif live:
            order = list(live)
            (rng or random).shuffle(order)
            hint = f"\n\nReference answer from the dataset (may be phrased differently):\n{reference}" if reference else ""
            body = "\n\n".join(f"REPLY {n + 1}\n{replies[i]}" for n, i in enumerate(order))
            def parse(result):
                grades = result.get("scores")
                if isinstance(grades, dict):
                    grades = list(grades.values())
                if not isinstance(grades, list) or len(grades) != len(order):
                    raise ValueError("wrong number of scores")
                return [_number(g.get("score") if isinstance(g, dict) else g) for g in grades]
            try:
                grades = self.ask(self._system(BATCH_SYSTEM), f"CONVERSATION\n{_transcript(messages)}{hint}\n\n{body}", parse=parse)
                for grade, i in zip(grades, order):
                    scores[i] = grade
            except TeacherError as exc:
                if exc.fatal:
                    raise
            for i in live:  # fallback: anything the batch call did not settle is graded individually
                if scores[i] is None:
                    scores[i] = self.score(messages, replies[i], reference)
        return scores

    def rewrite(self, messages, max_chars=240):
        # JSON mode keeps reasoning-style models from leaking their thinking into the answer.
        result = self.ask(self._system(REWRITE_SYSTEM, max_chars=max_chars), _transcript(messages), temperature=.2)
        answer = result.get("answer")
        return (answer.strip()[:max_chars].strip() if isinstance(answer, str) else "")


# --------------------------------------------------------------------------- best-of-N pair building

def temperature_spread(n, requested=None, low=.5, high=1.2):
    """N sampling temperatures spread from focused to adventurous (or interpolated from a user list)."""
    n = max(1, int(n))
    if requested:
        low, high = float(min(requested)), float(max(requested))
        if len(requested) == n:
            return [float(t) for t in requested]
    return [(low + high) / 2] if n == 1 else [round(low + (high - low) * i / (n - 1), 3) for i in range(n)]


def sample_candidates(model, messages, cfg, stop):
    """N distinct replies at a temperature spread (duplicates are re-sampled hotter)."""
    n = max(2, min(8, int(cfg["candidates"])))
    answers = []
    for temperature in temperature_spread(n, cfg.get("temperatures")):
        if stop.is_set():
            return answers
        answers.append(model.generate(messages, max_tokens=cfg["max_tokens"], temperature=temperature, top_k=40, stop=stop))
    seen, unique = set(), []
    for answer in answers:
        if answer.strip() not in seen:
            seen.add(answer.strip())
            unique.append(answer)
    extra = 0
    while len(unique) < min(n, 3) and extra < n and not stop.is_set():
        extra += 1
        answer = model.generate(messages, max_tokens=cfg["max_tokens"], temperature=1.3, top_k=0, stop=stop)
        if answer.strip() not in seen:
            seen.add(answer.strip())
            unique.append(answer)
    return unique


def teacher_pairs(teacher, model, prompt, cfg, rng, stop, judge=None):
    """Best-of-N: sample candidates, grade them, and turn the result into 0+ preference pairs.

    Returns (pairs, info). `info["skip"]` names why no pair was made; info["scores"] holds every
    candidate's grade. Teacher exceptions propagate so the caller can count them per prompt.
    """
    judge = judge or teacher
    messages, reference = prompt["messages"], prompt.get("reference")
    info = {"skip": None, "scores": [], "candidates": [], "reason": "", "rewrite": None}
    candidates = sample_candidates(model, messages, cfg, stop)
    info["candidates"] = candidates
    if stop.is_set():
        info["skip"] = "stopped"
        return [], info
    if len(candidates) < 2:
        info["skip"] = "model gave identical replies"
        return [], info
    scores = judge.score_many(messages, candidates, reference, rng)
    info["scores"] = list(scores)
    ranking = sorted(range(len(candidates)), key=lambda i: (-scores[i], len(candidates[i])))  # ties: shorter wins
    best, worst = ranking[0], ranking[-1]
    if scores[best] - scores[ranking[1]] <= float(cfg["tiebreak_gap"]):
        try:  # top two are neck and neck: settle it head-to-head, in both orders
            verdict = judge.judge(messages, candidates[best], candidates[ranking[1]], reference, True, rng)
            info["reason"] = verdict["reason"]
            if verdict["scores"][1] > verdict["scores"][0] + .25:
                best = ranking[1]
                ranking[0], ranking[1] = ranking[1], ranking[0]
        except TeacherError as exc:
            if exc.fatal:
                raise
    chosen, chosen_score, kind = candidates[best], scores[best], "comparison"
    if chosen_score < cfg["rewrite_below"]:
        try:
            rewritten = teacher.rewrite(messages, max_chars=max(40, int(cfg["max_tokens"]) * 2))
            if rewritten.strip() and rewritten.strip() not in {c.strip() for c in candidates}:
                regrade = judge.score_many(messages, [rewritten, candidates[best]], reference, rng)
                info["rewrite"] = {"text": rewritten, "scores": regrade}
                if regrade[0] > regrade[1]:  # a rewrite is only trusted if it beats the model's own best attempt
                    chosen, chosen_score, kind = rewritten, regrade[0], "rewrite"
        except TeacherError as exc:
            if exc.fatal:
                raise
    if chosen_score - scores[worst] < float(cfg["min_margin"]):
        info["skip"] = "rewrite no better than best reply" if info["rewrite"] is not None and kind == "comparison" else "scores too close"
        return [], info
    if not chosen.strip() or chosen.strip() == candidates[worst].strip():
        info["skip"] = "empty or identical answer"
        return [], info
    base = {"messages": messages, "kind": kind, "reference": reference, "source_id": prompt.get("source_id"),
            "reason": info["reason"], "n_candidates": len(candidates)}
    pairs = [{**base, "chosen": chosen, "rejected": candidates[worst], "scores": [chosen_score, scores[worst]]}]
    if cfg.get("extra_pairs"):
        for i in ranking:
            if i not in (worst, best) and chosen_score - scores[i] >= float(cfg["min_margin"]) and candidates[i].strip() != chosen.strip():
                pairs.append({**base, "chosen": chosen, "rejected": candidates[i], "scores": [chosen_score, scores[i]]})
        if kind == "rewrite" and chosen_score - scores[best] >= float(cfg["min_margin"]):
            pairs.append({**base, "chosen": chosen, "rejected": candidates[best], "scores": [chosen_score, scores[best]]})
    return pairs, info


def teacher_pair(teacher, model, prompt, cfg, rng, stop):
    """Single-pair convenience wrapper over teacher_pairs (kept for older callers)."""
    pairs, info = teacher_pairs(teacher, model, prompt, {**AUTO_DEFAULTS, **cfg}, rng, stop)
    return (pairs[0] if pairs else None), info


# --------------------------------------------------------------------------- arena: checkpoint vs checkpoint

def compare_checkpoints(a, b, prompts, judge, samples=2, max_tokens=120, temperature=.4, both_orders=True,
                        tie_eps=.5, device="auto", stop=None, emit=None, rng=None, examples=12):
    """Head-to-head: do checkpoint `a`'s replies beat checkpoint `b`'s, as graded by `judge`?

    Every prompt is answered `samples` times by each model; identical replies are ties without asking
    the judge, otherwise the judge compares them (both orders by default). wins/losses/ties are from
    `a`'s point of view and win_rate counts a tie as half. Usable standalone (no workspace needed).
    """
    stop = stop or threading.Event()
    emit = emit or (lambda event: None)
    rng = rng or random.Random(17)
    model_a, model_b = LocalModel(a, seed=21, device=device), LocalModel(b, seed=22, device=device)
    wins = ties = losses = errors = 0
    shown, total, done = [], len(prompts) * max(1, int(samples)), 0
    failures = 0
    for prompt in prompts:
        for _ in range(max(1, int(samples))):
            if stop.is_set():
                break
            done += 1
            messages = prompt["messages"]
            reply_a = model_a.generate(messages, max_tokens=max_tokens, temperature=temperature, top_k=20, stop=stop)
            reply_b = model_b.generate(messages, max_tokens=max_tokens, temperature=temperature, top_k=20, stop=stop)
            if stop.is_set():
                break
            scores, reason = None, ""
            if reply_a.strip() == reply_b.strip():
                outcome = "tie"
            else:
                try:
                    verdict = judge.judge(messages, reply_a, reply_b, prompt.get("reference"), both_orders, rng)
                    scores, reason = verdict["scores"], verdict["reason"]
                    gap = scores[0] - scores[1]
                    outcome = "win" if gap > tie_eps else "loss" if gap < -tie_eps else "tie"
                    failures = 0
                except Exception as exc:
                    if stop.is_set():
                        break
                    errors += 1
                    failures += 1
                    if isinstance(exc, TeacherError) and exc.fatal or failures >= 6:
                        raise TeacherError(f"The judge failed repeatedly: {exc}", fatal=True) from None
                    emit({"type": "rlhf_gate", "done": done, "total": total, "error": str(exc)[:160]})
                    continue
            wins, ties, losses = wins + (outcome == "win"), ties + (outcome == "tie"), losses + (outcome == "loss")
            if len(shown) < examples:
                shown.append({"prompt": _transcript(messages), "a": reply_a, "b": reply_b, "outcome": outcome,
                              "scores": scores, "reason": reason, "reference": prompt.get("reference")})
            emit({"type": "rlhf_gate", "done": done, "total": total, "wins": wins, "ties": ties, "losses": losses})
    n = wins + ties + losses
    return {"a": str(a), "b": str(b), "wins": wins, "ties": ties, "losses": losses, "errors": errors, "n": n,
            "win_rate": (wins + .5 * ties) / n if n else None, "examples": shown, "stopped": stop.is_set()}


def mean_teacher_score(teacher, checkpoint, prompts, cfg, stop, device="auto"):
    """Average absolute grade (1-10) the teacher gives a checkpoint on some prompts."""
    model = LocalModel(checkpoint, seed=11, device=device)
    scores = []
    for prompt in prompts:
        if stop.is_set():
            break
        reply = model.generate(prompt["messages"], max_tokens=cfg["max_tokens"], temperature=.3, top_k=20, stop=stop)
        scores.append(teacher.score(prompt["messages"], reply, prompt.get("reference")))
    return sum(scores) / len(scores) if scores else None


# --------------------------------------------------------------------------- automated loop

def _skip_label(exc):
    if isinstance(exc, TeacherError):
        return "teacher unreachable" if exc.fatal else "teacher reply unusable"
    if isinstance(exc, (ValueError, KeyError, TypeError, AttributeError)):
        return "malformed teacher reply"
    return "error: " + exc.__class__.__name__


def auto_rlhf(base_checkpoint, prompts, teacher, store, run_root, config=None, dpo_config=None, emit=None, stop=None,
              pause=None, judge=None):
    """Iterative online DPO: sample N → teacher grades → train → head-to-head gate; keep only real improvements.

    `judge` (default: the teacher itself, or its judge_model sibling) grades candidates and runs the
    win-rate gate, so a different model can mark the homework than the one that wrote the rewrites.
    A bad teacher reply only skips that prompt; the run ends early only if the teacher keeps failing.
    """
    cfg = {**AUTO_DEFAULTS, **(config or {})}
    cfg["candidates"] = max(2, min(8, int(cfg["candidates"])))
    emit = emit or (lambda event: None)
    stop, pause = stop or threading.Event(), pause or threading.Event()
    device = (dpo_config or {}).get("device", "auto")
    judge = judge or teacher.judge_teacher()
    if len(prompts) < cfg["eval_prompts"] + 4:
        raise ValueError("Choose a dataset with more prompts (user → assistant conversations).")
    rng = random.Random(cfg.get("seed", 5))
    pool = list(prompts)
    rng.shuffle(pool)
    eval_prompts, train_pool = pool[:cfg["eval_prompts"]], pool[cfg["eval_prompts"]:]
    queue_ = []
    current = str(base_checkpoint)
    say = lambda text: emit({"type": "rlhf_log", "text": text})
    judged = f" · judge {judge.model}" if judge is not teacher else ""
    say(f"Teacher {teacher.model}{judged} · {cfg['candidates']} replies per prompt · "
        f"a new checkpoint must win ≥ {(.5 + cfg['win_margin']) * 100:.0f}% head-to-head on {len(eval_prompts)} held-out prompts.")
    emit({"type": "rlhf_round", "round": 0, "pairs": 0, "rewrites": 0, "score": None, "win_rate": None, "wins": 0, "ties": 0,
          "losses": 0, "accepted": True, "checkpoint": current, "skipped": {}})
    rounds, collected, skip_total, last_rate, aborted = [], [], Counter(), None, None
    for round_index in range(1, int(cfg["rounds"]) + 1):
        if stop.is_set() or aborted:
            break
        model = LocalModel(current, seed=round_index, device=device)
        fresh, skipped, streak, last_error = [], Counter(), 0, ""
        total = int(cfg["prompts_per_round"])
        for i in range(total):
            while pause.is_set() and not stop.is_set():
                time.sleep(.1)
            if stop.is_set():
                break
            if not queue_:
                queue_ = list(train_pool)
                rng.shuffle(queue_)
            prompt = queue_.pop()
            info = {}
            try:
                pairs, info = teacher_pairs(teacher, model, prompt, cfg, rng, stop, judge)
                streak = 0
                if not pairs and info.get("skip") and info["skip"] != "stopped":
                    skipped[info["skip"]] += 1
                for pair in pairs:
                    fresh.append(store.add(pair["messages"], pair["chosen"], pair["rejected"], f"teacher:{teacher.model}",
                                           kind=pair["kind"], scores=pair["scores"], reason=pair["reason"], checkpoint=current,
                                           round=round_index, reference=pair["reference"], source_id=pair["source_id"],
                                           candidates=pair["n_candidates"]))
            except Exception as exc:  # one bad reply must never kill the run
                if stop.is_set():
                    break
                label = _skip_label(exc)
                skipped[label] += 1
                streak += 1
                last_error = str(exc)[:200]
                info = {"error": last_error}
                if streak >= int(cfg["max_failures"]):
                    aborted = f"The teacher failed {streak} prompts in a row ({label}: {last_error}). Check Ollama and try again."
            emit({"type": "rlhf_progress", "round": round_index, "done": i + 1, "total": total, "pairs": len(fresh),
                  "skipped": sum(skipped.values()), "skip_reasons": dict(skipped), "info": info, "last_error": last_error})
            if aborted:
                break
        collected.extend(fresh)
        skip_total.update(skipped)
        reasons = ", ".join(f"{k} ×{v}" for k, v in skipped.items())
        if aborted:
            say(aborted)
            break
        if stop.is_set() or len(fresh) < 2:
            say(f"Round {round_index}: {len(fresh)} usable pairs" + (f" · skipped: {reasons}" if reasons else "") +
                (" · stopping" if stop.is_set() else " · too few to train, continuing"))
            continue
        replay = [p for p in collected if p not in fresh][-int(cfg["replay"]):] if int(cfg["replay"]) > 0 else []
        say(f"Round {round_index}: {len(fresh)} new pairs" + (f" (skipped: {reasons})" if reasons else "") +
            f" + {len(replay)} replayed · training…")
        directory = Path(run_root) / uid("rlhf")
        try:
            result = dpo_train(current, fresh + replay, directory, {**(dpo_config or {}), "steps": int(cfg["dpo_steps"])},
                               emit, stop, pause, name=f"Auto-RLHF round {round_index}")
        except Exception as exc:
            say(f"Round {round_index}: training failed ({exc}) · keeping the previous checkpoint")
            continue
        candidate = result["checkpoint"]
        say(f"Round {round_index}: comparing the new checkpoint with the current one…")
        try:
            gate = compare_checkpoints(candidate, current, eval_prompts, judge, int(cfg["gate_samples"]), cfg["max_tokens"],
                                       cfg["gate_temperature"], cfg["both_orders"], cfg["tie_eps"], device, stop, emit, rng)
        except Exception as exc:
            gate = {"wins": 0, "ties": 0, "losses": 0, "n": 0, "win_rate": None, "error": str(exc)[:200]}
            say(f"Round {round_index}: comparison failed ({exc})")
        if stop.is_set():
            say(f"Round {round_index}: stopped before the comparison finished · keeping the previous checkpoint")
            break
        rate = gate["win_rate"]
        needed = .5 + float(cfg["win_margin"])
        accepted = (not cfg["keep_only_improvements"]) or (rate is not None and rate >= needed)
        if accepted:
            current, last_rate = candidate, rate
        row = {"type": "rlhf_round", "round": round_index, "pairs": len(fresh),
               "rewrites": sum(p.get("kind") == "rewrite" for p in fresh), "score": rate, "win_rate": rate,
               "wins": gate["wins"], "ties": gate["ties"], "losses": gate["losses"], "n": gate["n"],
               "accepted": accepted, "checkpoint": candidate, "skipped": dict(skipped)}
        rounds.append(row)
        emit(row)
        shown = "n/a" if rate is None else f"{rate * 100:.0f}% (win {gate['wins']} · tie {gate['ties']} · loss {gate['losses']})"
        say(f"Round {round_index}: win rate {shown} vs needed {needed * 100:.0f}% · " + ("kept" if accepted else "rejected — keeping the previous checkpoint"))
    return {"checkpoint": current, "score": last_rate, "win_rate": last_rate, "rounds": rounds, "pairs": len(collected),
            "skipped": dict(skip_total), "aborted": aborted}
