"""Evaluation and references: bits per character, uniform and unigram reference lines, paired gain test, run status.

Documents are plain strings. Model calls go through LocalModel / ScratchTransformer.forward + backend.to_cpu only.
"""
from __future__ import annotations

import json
import math
import random
import re
import tempfile
from pathlib import Path

import numpy as np
import torch

from studio_ui_guard import slow
from studio_catalog import CHOICE_ASK
from studio_slots import BOS, OFFSET, PAD, UNK

LN2 = math.log(2)
SCORE_WINDOWS = 512  # windows gathered before a scoring pass
SCORE_ELEMENTS = 4_000_000  # most float64 logits (batch x window x vocabulary) one scoring pass holds
SHOTS = (0, 1, 4)  # worked examples in front of each few-shot question: none, one, a few
CHOICE_LINE = re.compile(r"([A-J])\) (.+)")
SCORE_DEVICE = "cpu"  # score_docs reads a checkpoint path on this device, so held-back scores do not depend on the machine that scores
SCORE_METHOD = "stride-half-2"  # how score_docs reads a document; scores stored under another value are stale
MIN_DOCS = 20  # fewest held-back documents paired_gain will judge on; a percentile bootstrap over fewer undercovers
BOOTSTRAP_ELEMENTS = 2_000_000  # most resampled indices one bootstrap pass holds


def heldout_split(docs, fraction=.1, seed=0):
    """(train, held_back): exact duplicates removed first, then whole documents split by a seeded shuffle."""
    unique = list(dict.fromkeys(docs))
    random.Random(seed).shuffle(unique)
    held = min(len(unique) - 1, max(1, round(len(unique) * fraction))) if len(unique) > 1 else 0
    return unique[held:], unique[:held]


def _ids(tokenizer, text):
    """Token ids of text as an int64 array (tokenizers without encode_array build it from the list)."""
    return tokenizer.encode_array(text) if hasattr(tokenizer, "encode_array") else np.asarray(tokenizer.encode(text), dtype=np.int64)


def score_windows(length, context):
    """(start, stop, skip) per forward pass over a document of `length` ids (BOS first): inputs ids[start:stop], targets
    ids[start + 1:stop + 1], and the first `skip` positions are context only. Each window after the first moves on by
    context // 2 and re-reads the previous half, so every id after BOS is scored exactly once and (past the first
    window) with at least half a window of context."""
    stride, last = max(1, context // 2), length - 1  # targets are ids[1:length]
    out, first = [], 1
    while first <= last:
        stop = min(last, first + (context if first == 1 else stride) - 1)
        start = max(0, stop - context)
        out.append((start, stop, first - start - 1))
        first = stop + 1
    return out


@slow("score documents with a model")
def score_docs(model, docs):
    """Per document (nats, characters). Characters the tokenizer cannot encode (UNK pieces) are scored at ln V.

    model is a checkpoint path or a LocalModel. A document is read in overlapping windows (score_windows): each window
    after the first scores only its new half, with the half before it as context, so no piece is scored cold except the
    first window's. Windows from many documents run through the model together, right-padded to the longest in the
    batch; every mixer is causal, so the padding never reaches a real position. The probability is shared among the real
    pieces only: a document is never made of PAD, BOS, EOS or role tokens, and an imported GPT-2 gives BOS and EOS the same row
    as its end-of-text piece, so counting them would charge every position for the same piece three times.
    """
    from studio_model import LocalModel
    local = model if isinstance(model, LocalModel) else LocalModel(model, device=SCORE_DEVICE)
    tok, context, uniform = local.tokenizer, local.config["context"], math.log(local.tokenizer.vocab_size)
    nats = [0.] * len(docs)
    pending = []  # (document, inputs, targets, skip); scored and cleared once enough have gathered, so memory stays bounded

    def flush():
        pending.sort(key=lambda window: len(window[1]))  # similar lengths together: little padding
        size = max(1, SCORE_ELEMENTS // (context * tok.vocab_size))
        for first in range(0, len(pending), size):
            batch = pending[first:first + size]
            lengths = np.array([len(inputs) for _, inputs, _, _ in batch])
            skips = np.array([skip for _, _, _, skip in batch])
            x, y = np.full((2, len(batch), lengths.max()), PAD, dtype=np.int64)
            for row, (_, inputs, targets, _) in enumerate(batch):
                x[row, :len(inputs)], y[row, :len(targets)] = inputs, targets
            logits = local.model.forward(x).double()
            logits[..., :OFFSET] = -torch.inf
            logp = torch.log_softmax(logits, dim=-1).gather(-1, torch.as_tensor(y)[..., None])[..., 0].cpu().numpy()
            position = np.arange(x.shape[1])[None, :]
            real = (position < lengths[:, None]) & (position >= skips[:, None])
            unknown = real & (y == UNK)
            scored = np.where(real & ~unknown, -logp, 0.).sum(axis=1) + uniform * unknown.sum(axis=1)
            for (doc, _, _, _), value in zip(batch, scored):
                nats[doc] += float(value)
        pending.clear()

    for doc, text in enumerate(docs):
        ids = np.concatenate([[BOS], _ids(tok, text)])
        for start, stop, skip in score_windows(len(ids), context):
            pending.append((doc, ids[start:stop], ids[start + 1:stop + 1], skip))
        if len(pending) >= SCORE_WINDOWS:
            flush()
    flush()
    return [(n, len(text)) for n, text in zip(nats, docs)]


def corpus_bpc(scores):
    """Total nats over total characters of per-document (nats, characters) rows, divided by ln 2."""
    arr = np.asarray(scores, dtype=np.float64).reshape(-1, 2)
    chars = arr[:, 1].sum()
    return float(arr[:, 0].sum() / chars / LN2) if chars else float("nan")


def bits_per_char(checkpoint, docs):
    """Total nats over total characters, divided by ln 2."""
    return corpus_bpc(score_docs(checkpoint, docs))


@slow("encode a whole corpus")
def uniform_bpc(tokenizer, docs):
    """The uniform guess: ln V per piece, converted with the corpus characters-per-piece."""
    pieces = sum(len(tokenizer.encode(d)) for d in docs)
    chars = sum(len(d) for d in docs)
    return math.log(tokenizer.vocab_size) * pieces / chars / LN2


@slow("encode a whole corpus")
def unigram_bpc(tokenizer, docs, eval_docs=None):
    """Counting word frequencies: exact entropy of the piece counts over docs, scored on eval_docs (default docs).

    Unencodable pieces and pieces never seen in docs are scored at the uniform rate.
    """
    counts = np.zeros(tokenizer.vocab_size, dtype=np.int64)
    for first in range(0, len(docs), 1024):  # a thousand documents at a time: the whole corpus is never one array
        counts += np.bincount(np.concatenate([_ids(tokenizer, d) for d in docs[first:first + 1024]]), minlength=len(counts))
    counts[UNK] = 0
    total, uniform = counts.sum(), math.log(tokenizer.vocab_size)
    nats = chars = 0
    for d in docs if eval_docs is None else eval_docs:
        chars += len(d)
        seen = counts[_ids(tokenizer, d)]
        nats += float(np.log(total / seen[seen > 0]).sum()) + uniform * int((seen == 0).sum())
    return nats / chars / LN2


def band_from_scores(scores, resamples=200, seed=0):
    """(lo, hi) 10th-90th percentile of the corpus bpc over bootstrap resamples of documents.

    One model on its own: it says how far that one number moves with the choice of documents. It is not a test of one
    version against another; for that use paired_gain.
    """
    rng = np.random.default_rng(seed)
    arr = np.array(scores, dtype=np.float64)
    picks = rng.integers(0, len(arr), (resamples, len(arr)))
    est = arr[picks, 0].sum(1) / arr[picks, 1].sum(1) / LN2
    return float(np.percentile(est, 10)), float(np.percentile(est, 90))


def noise_band(checkpoint, docs, resamples=200, seed=0):
    return band_from_scores(score_docs(checkpoint, docs), resamples, seed)


def paired_gain(parent, candidate, resamples=1000, seed=0, confidence=.95):
    """Is the candidate better than the parent on the same held-back documents? {diff, upper, better, docs} (plus reason).

    parent and candidate are score_docs rows (nats, characters) for the same documents in the same order. Documents are
    resampled with the same indices for both, so what the two versions share (how hard each document is) cancels, and
    diff is candidate minus parent in corpus bits per character. upper is the upper end of a one-sided `confidence`
    interval of that difference; the candidate is better only when upper is below 0, so a gap that luck in the choice of
    documents could produce reads as not sure. Identical models give diff 0, upper 0 and are never better.

    Repeated looks: the test is meant for one look per pair, at the end of a run on its best checkpoint. Re-asking at
    every checkpoint would let luck win eventually (each look is another chance), so Chat does not. A caller that must
    look k times should ask for confidence 1 - (1 - confidence) / k (Bonferroni).

    Fewer than MIN_DOCS documents is never better: with a handful the bootstrap's one-sided bound is too narrow and "better"
    would fire on luck, so the answer is better False, upper nan and reason "few_docs" (diff is still the measured gap).

    This covers the choice of documents only. Seed variance (a different random start giving a different result) is not in
    it; Compare runs measure that spread.
    """
    p, c = np.asarray(parent, dtype=np.float64).reshape(-1, 2), np.asarray(candidate, dtype=np.float64).reshape(-1, 2)
    if len(p) != len(c) or not np.array_equal(p[:, 1], c[:, 1]):
        raise ValueError("Both versions must be scored on the same documents.")
    n, chars = len(p), p[:, 1]
    if not chars.sum():
        return {"diff": float("nan"), "upper": float("nan"), "better": False, "docs": n}
    delta = c[:, 0] - p[:, 0]
    if n < MIN_DOCS:
        return {"diff": float(delta.sum() / chars.sum() / LN2), "upper": float("nan"), "better": False, "docs": n, "reason": "few_docs"}
    rng = np.random.default_rng(seed)
    diffs, step = np.empty(resamples), max(1, BOOTSTRAP_ELEMENTS // n)
    for first in range(0, resamples, step):
        picks = rng.integers(0, n, (min(step, resamples - first), n))
        diffs[first:first + len(picks)] = delta[picks].sum(1) / chars[picks].sum(1) / LN2
    upper = float(np.percentile(diffs, 100 * confidence))
    return {"diff": float(delta.sum() / chars.sum() / LN2), "upper": upper, "better": upper < 0, "docs": n}


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


def choice_item(example):
    """A lettered multiple-choice problem (studio_catalog._lettered) as {"task", "question", "choices", "answer"}, else None.

    Takes a workspace example (user, rule and params, or metadata) or a post_rows row (prompt, rule, params). The task is
    the instruction in front of CHOICE_ASK, shared by every item of a source; answer is the keyed option's index.
    """
    meta = example.get("metadata") or {}
    if (example.get("rule") or meta.get("rule")) != "choice":
        return None
    head, _, tail = str(example.get("user") or example.get("prompt") or "").strip().rpartition("\n\n")
    found = [CHOICE_LINE.fullmatch(line.strip()) for line in tail.split("\n")]
    if not head or len(found) < 2 or not all(found) or "".join(m[1] for m in found) != "ABCDEFGHIJ"[:len(found)]:
        return None
    letters = [m[1] for m in found]
    key = str((example.get("params") or meta.get("params") or {}).get("answer") or str(example.get("assistant") or "")[:1])
    if key not in letters:
        return None
    task, ask, question = head.partition(CHOICE_ASK.strip())
    return {"task": task.strip() if ask else "", "question": (question if ask else head).strip(),
            "choices": [m[2].strip() for m in found], "answer": letters.index(key)}


def few_shot_prompt(item, examples=()):
    """The task line, each worked example as its question and answer, then the question with the answer left open."""
    parts = [item["task"]] if item.get("task") else []
    parts += [f"{e['question']}\nAnswer: {e['choices'][e['answer']]}" for e in examples]
    return "\n\n".join(parts + [f"{item['question']}\nAnswer:"])


def few_shot_table(items, encode, context, logprob, shots=SHOTS, seed=0, pool=None):
    """Multiple-choice accuracy with 0, 1 and k worked examples in the prompt and no training (GPT-3, Brown et al. 2020).

    encode(text) -> token ids; logprob([(prompt_ids, continuation_ids)]) -> summed log-probability of each continuation.
    Worked examples come from `pool` (default: items) with the same task, never the question itself; one seeded draw per
    item, so the k-shot prompt extends the 1-shot one. The oldest examples drop out until prompt and answer fit `context`.
    The pick is the choice with the highest log-probability per piece; chance is what guessing scores (mean 1 / choices).
    Returns {"items", "chance", "rows": [{"shots", "used" (mean examples that fitted), "correct", "accuracy"}]}.
    """
    items = list(items)
    if not items:
        return {"items": 0, "chance": None, "rows": []}
    by_task = {}
    for p in items if pool is None else pool:
        by_task.setdefault(p["task"], []).append(p)
    rng, most = random.Random(seed), max(shots)
    pairs, plan = [], []  # plan: (shot column, item, examples that fitted, index of its first choice in pairs)
    for item in items:
        others = [p for p in by_task.get(item["task"], []) if p["question"] != item["question"]]
        drawn = rng.sample(others, min(most, len(others)))
        answers = [encode(" " + c) for c in item["choices"]]
        for column, n in enumerate(shots):
            examples = drawn[:n]
            prompt = encode(few_shot_prompt(item, examples))
            while examples and len(prompt) + max(map(len, answers)) > context:
                examples = examples[1:]
                prompt = encode(few_shot_prompt(item, examples))
            plan.append((column, item, len(examples), len(pairs)))
            pairs += [(prompt, a) for a in answers]
    scores = logprob(pairs)
    correct, used = [0] * len(shots), [0] * len(shots)
    for column, item, n, first in plan:
        per_piece = [scores[first + i] / max(1, len(pairs[first + i][1])) for i in range(len(item["choices"]))]
        correct[column] += per_piece.index(max(per_piece)) == item["answer"]
        used[column] += n
    return {"items": len(items), "chance": sum(1 / len(i["choices"]) for i in items) / len(items),
            "rows": [{"shots": n, "used": used[c] / len(items), "correct": correct[c], "accuracy": correct[c] / len(items)}
                     for c, n in enumerate(shots)]}


def _continuation_logprobs(local, pairs):
    """Summed log-probability of each continuation after its prompt (token-id lists), batched like score_docs.
    A window longer than the context keeps its last pieces."""
    context, vocab = local.config["context"], local.tokenizer.vocab_size
    windows = [np.asarray([BOS, *prompt, *answer], dtype=np.int64)[-(context + 1):] for prompt, answer in pairs]
    order = sorted(range(len(pairs)), key=lambda i: len(windows[i]))  # similar lengths together: little padding
    size = max(1, SCORE_ELEMENTS // (context * vocab))
    out = [0.] * len(pairs)
    for first in range(0, len(order), size):
        batch = order[first:first + size]
        x, y = np.full((2, len(batch), max(len(windows[i]) for i in batch) - 1), PAD, dtype=np.int64)
        for row, i in enumerate(batch):
            x[row, :len(windows[i]) - 1], y[row, :len(windows[i]) - 1] = windows[i][:-1], windows[i][1:]
        with torch.no_grad():
            logp = torch.log_softmax(local.model.forward(x).double(), dim=-1).gather(-1, torch.as_tensor(y)[..., None])[..., 0].cpu().numpy()
        for row, i in enumerate(batch):
            end = len(windows[i]) - 1
            out[i] = float(logp[row, max(0, end - len(pairs[i][1])):end].sum())
    return out


@slow("run few-shot checks")
def few_shot(model, items, shots=SHOTS, seed=0, pool=None):
    """few_shot_table scored by a checkpoint path or LocalModel."""
    from studio_model import LocalModel
    local = model if isinstance(model, LocalModel) else LocalModel(model, device="cpu")
    return few_shot_table(items, lambda text: _ids(local.tokenizer, text).tolist(), local.config["context"],
                          lambda pairs: _continuation_logprobs(local, pairs), shots, seed, pool)
