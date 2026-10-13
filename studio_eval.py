"""Evaluation and references: bits per character, uniform and unigram reference lines, noise band, run status.

Documents are plain strings. Model calls go through LocalModel / ScratchTransformer.forward + backend.to_cpu only.
"""
from __future__ import annotations

import json
import math
import random
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from studio_slots import BOS, UNK

LN2 = math.log(2)


def heldout_split(docs, fraction=.1, seed=0):
    """(train, held_back): exact duplicates removed first, then whole documents split by a seeded shuffle."""
    unique = list(dict.fromkeys(docs))
    random.Random(seed).shuffle(unique)
    held = min(len(unique) - 1, max(1, round(len(unique) * fraction))) if len(unique) > 1 else 0
    return unique[held:], unique[:held]


def score_docs(model, docs):
    """Per document (nats, characters). Characters the tokenizer cannot encode (UNK pieces) are scored at ln V.

    model is a checkpoint path or a LocalModel.
    """
    from studio_model import LocalModel
    local = model if isinstance(model, LocalModel) else LocalModel(model, device="cpu")
    tok, context, uniform = local.tokenizer, local.config["context"], math.log(local.tokenizer.vocab_size)
    out = []
    for text in docs:
        ids = np.array([BOS] + tok.encode(text), dtype=np.int64)
        nats = 0.
        for start in range(0, len(ids) - 1, context):
            x, y = ids[start:start + context], ids[start + 1:start + context + 1]
            logits = local.model.backend.to_cpu(local.model.forward(x[None, :len(y)]))[0].astype(np.float64)
            logits -= logits.max(axis=-1, keepdims=True)
            logp = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
            picked = logp[np.arange(len(y)), y]
            unknown = y == UNK
            nats += float(-picked[~unknown].sum()) + uniform * int(unknown.sum())
        out.append((nats, len(text)))
    return out


def _bpc(scores):
    chars = sum(c for _, c in scores)
    return sum(n for n, _ in scores) / chars / LN2 if chars else float("nan")


def bits_per_char(checkpoint, docs):
    """Total nats over total characters, divided by ln 2."""
    return _bpc(score_docs(checkpoint, docs))


def uniform_bpc(tokenizer, docs):
    """The uniform guess: ln V per piece, converted with the corpus characters-per-piece."""
    pieces = sum(len(tokenizer.encode(d)) for d in docs)
    chars = sum(len(d) for d in docs)
    return math.log(tokenizer.vocab_size) * pieces / chars / LN2


def unigram_bpc(tokenizer, docs, eval_docs=None):
    """Counting word frequencies: exact entropy of the piece counts over docs, scored on eval_docs (default docs).

    Unencodable pieces and pieces never seen in docs are scored at the uniform rate.
    """
    counts = Counter(t for d in docs for t in tokenizer.encode(d) if t != UNK)
    total, uniform = sum(counts.values()), math.log(tokenizer.vocab_size)
    nats = chars = 0
    for d in docs if eval_docs is None else eval_docs:
        chars += len(d)
        for t in tokenizer.encode(d):
            nats += math.log(total / counts[t]) if counts.get(t) else uniform
    return nats / chars / LN2


def band_from_scores(scores, resamples=200, seed=0):
    """(lo, hi) 10th-90th percentile of the corpus bpc over bootstrap resamples of documents."""
    rng = np.random.default_rng(seed)
    arr = np.array(scores, dtype=np.float64)
    picks = rng.integers(0, len(arr), (resamples, len(arr)))
    est = arr[picks, 0].sum(1) / arr[picks, 1].sum(1) / LN2
    return float(np.percentile(est, 10)), float(np.percentile(est, 90))


def noise_band(checkpoint, docs, resamples=200, seed=0):
    return band_from_scores(score_docs(checkpoint, docs), resamples, seed)


def smooth(values, alpha=.2):
    """Exponential moving average, seeded with the first value."""
    out, current = [], None
    for v in values:
        current = v if current is None else alpha * v + (1 - alpha) * current
        out.append(current)
    return out


def run_status(history, window=20):
    """learning | flat | memorising from engine history rows ({"loss", "val_loss"}); few points count as learning."""
    rows = [r for r in history if r.get("loss") is not None]
    sm = smooth([r["loss"] for r in rows])
    held = [(i, r["val_loss"]) for i, r in enumerate(rows) if r.get("val_loss") is not None]
    if len(held) >= 3:
        (i0, a), (_, b), (_, c) = held[-3:]
        if a < b < c and sm[-1] < sm[i0]:
            return "memorising"
    if len(rows) < 4:
        return "learning"
    w = min(window, len(rows))
    raw = np.array([r["loss"] for r in rows[-w:]])
    noise = float(np.std(np.diff(raw)))
    return "learning" if sm[-w] - sm[-1] > noise else "flat"


def training_saved(curve_new, curve_old, schedule_new=None, schedule_old=None):
    """{percent, comparable, reason}: share of the old run's compute the new run needed to match its final bpc.

    Curves are lists of (compute_or_tokens, bpc). Comparable needs the same total length and schedule id.
    """
    if not curve_new or not curve_old:
        return {"percent": None, "comparable": False, "reason": "A run has no points yet."}
    if curve_new[-1][0] != curve_old[-1][0]:
        return {"percent": None, "comparable": False, "reason": "The runs have different total lengths."}
    if schedule_new != schedule_old:
        return {"percent": None, "comparable": False, "reason": "The runs use different schedules."}
    target, total = curve_old[-1][1], curve_old[-1][0]
    prev = None
    for x, y in curve_new:
        if y <= target:
            hit = x if prev is None or prev[1] <= target else prev[0] + (x - prev[0]) * (prev[1] - target) / (prev[1] - y)
            return {"percent": 100 * (1 - hit / total), "comparable": True, "reason": "Matched the old final error early."}
        prev = (x, y)
    return {"percent": 0., "comparable": True, "reason": "The new run never matched the old final error."}


def memorise_one_batch(cfg, batch, steps=80, learning_rate=.01):
    """Train a throwaway tiny run on one batch (strings or example dicts). Returns {start, end, ok}."""
    from studio_model import train
    examples = [{"text": b} if isinstance(b, str) else b for b in batch]
    config = dict(cfg, steps=steps, batch_size=len(examples), learning_rate=learning_rate, validation=0,
                  warmup=0, eval_every=steps, device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        train(config, examples, Path(tmp) / "run")
        rows = [json.loads(line) for line in (Path(tmp) / "run" / "metrics.jsonl").read_text().splitlines()]
    start, end = rows[0]["loss"], rows[-1]["loss"]
    return {"start": start, "end": end, "ok": end < .5 * start}


def sanity_checks(source, vocab_size=None, memorised=None):
    """The four quick-test checks as [{id, ok, detail}]; ok None means not run.

    source is a run directory (metrics.jsonl plus a checkpoint) or a history list. memorised is the
    memorise_one_batch result. Losses are nats per piece, as in engine history.
    """
    if isinstance(source, (str, Path)):
        run = Path(source)
        history = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
        if vocab_size is None:
            from studio_model import Tokenizer, load_checkpoint
            ckpt = next(p for p in (run / "last.npz", run / "best.npz") if p.exists())
            vocab_size = Tokenizer.restore(load_checkpoint(ckpt)["tokenizer"]).vocab_size
    else:
        history = list(source)
    losses = [r["loss"] for r in history if r.get("loss") is not None]
    first, last = (losses[0], losses[-1]) if losses else (float("nan"), float("nan"))
    checks = []
    if vocab_size:
        guess = math.log(vocab_size)
        checks.append(("starts_near_uniform", abs(first - guess) <= .1 * guess, f"first {first:.2f} vs ln V {guess:.2f}"))
    else:
        checks.append(("starts_near_uniform", None, "vocabulary size unknown"))
    checks.append(("finite", bool(losses) and all(math.isfinite(v) for v in losses), f"{len(losses)} values"))
    checks.append(("memorises_one_batch", None if memorised is None else memorised["ok"],
                   "not run" if memorised is None else f"{memorised['start']:.2f} -> {memorised['end']:.2f}"))
    checks.append(("error_falls", len(losses) > 1 and last < first, f"{first:.2f} -> {last:.2f}"))
    return [{"id": i, "ok": ok, "detail": d} for i, ok, d in checks]
