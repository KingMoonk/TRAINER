"""Checkpoint probes and interventions, isolated from the training model and its RNG."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from studio_compile import CompiledModel
from studio_modules.ops import softmax


def probe_model(source):
    """Return a private model and optional tokenizer; never mutate a supplied live model."""
    if isinstance(source, (str, Path)):
        from studio_model import ScratchTransformer, Tokenizer, load_checkpoint
        checkpoint = load_checkpoint(source)
        tokenizer = Tokenizer.restore(checkpoint['tokenizer'])
        config = dict(checkpoint['config'], device='cpu')
        model = ScratchTransformer(config, tokenizer.vocab_size)
        model.load_weights(checkpoint['model'])
        return model, tokenizer
    if not isinstance(source, CompiledModel):
        raise ValueError('Choose a model or checkpoint to inspect.')
    model = CompiledModel(source.spec, source.dims.vocab, device='cpu')
    model.load_weights({key: source.backend.to_cpu(value).copy() for key, value in source.p.items()})
    return model, None


def probe_ids(model, text, tokenizer=None):
    if isinstance(text, str):
        if tokenizer is None:
            raise ValueError('Text probes need a checkpoint tokenizer; pass token ids for a live model.')
        text = tokenizer.encode(text)
    ids = np.asarray(text, dtype=np.int64)
    if ids.ndim == 1:
        ids = ids[None, :]
    if ids.ndim != 2 or not ids.size or ids.shape[1] > model.dims.n_ctx:
        raise ValueError('Probe must contain 1 to context-length tokens per sequence.')
    if np.any(ids < 0) or np.any(ids >= model.dims.vocab):
        raise ValueError('Probe token is outside the model vocabulary.')
    return ids


def forward(source, text, hooks=True):
    model, tokenizer = probe_model(source)
    ids = probe_ids(model, text, tokenizer)
    logits, tape = model.forward(ids, hooks=hooks, tape=True)
    tokens = [[tokenizer.decode([int(token)]) if tokenizer else str(int(token)) for token in row] for row in ids]
    return {'ids': ids.copy(), 'tokens': tokens, 'logits': logits.copy(),
            'hooks': {name: value.copy() for name, value in tape.captured.items()}, 'spec': copy.deepcopy(model.spec)}


def _targets(ids, targets=None):
    if targets is None:
        targets = np.roll(ids, -1, axis=1)
        targets[:, -1] = -100
    targets = np.array(targets, dtype=np.int64, copy=True)
    if targets.shape != ids.shape:
        raise ValueError('Targets must have the same shape as probe ids.')
    return targets


def backward(source, text, pos=None, targets=None):
    model, tokenizer = probe_model(source)
    ids = probe_ids(model, text, tokenizer)
    targets = _targets(ids, targets)
    if pos is not None:
        if not 0 <= pos < ids.shape[1]:
            raise ValueError('Selected position is outside the probe.')
        targets[:, np.arange(ids.shape[1]) != pos] = -100
    logits, tape = model.forward(ids, hooks=True, tape=True)
    loss, derivative = model.objective(logits, targets, gradient=True, xp=np)
    model.zero_grad()
    tape.ctx.scale = 0.  # explain the selected objective, excluding auxiliary router regularization
    model.backward(derivative, tape)
    groups = {}
    for key, gradient in model.g.items():
        slot, block = model.owner[key]
        group = slot if block is None else f'block{block}.{slot}'
        groups[group] = groups.get(group, 0.) + float(np.sum(gradient.astype(np.float64) ** 2))
    probabilities = softmax(logits, np)
    return {'loss': loss, 'targets': targets, 'probabilities': probabilities, 'dlogits': derivative,
            'groups': {key: value ** .5 for key, value in groups.items()},
            'gradients': {key: value.copy() for key, value in model.g.items()}}


def logit_lens(source, text, top_k=5):
    model, tokenizer = probe_model(source)
    ids = probe_ids(model, text, tokenizer)
    _, tape = model.forward(ids, tape=True, hooks=True)
    rows = []
    names = ['embed'] + [f'block{i}.resid.post' for i in range(model.dims.n_layer)]
    from studio_slots import Ctx
    ctx = Ctx(embedding=model.embedding, position=model.position)
    for name in names:
        hidden = tape.captured[name]
        if model.final is not None:
            hidden = model.final.forward(hidden, ctx)
        logits = model.head.forward(hidden, ctx)
        probabilities = softmax(logits, np)
        k = max(1, min(int(top_k), model.dims.vocab))
        top = np.argsort(probabilities, axis=-1)[..., -k:][..., ::-1]
        rows.append({'hook': name, 'probabilities': probabilities, 'top_ids': top,
                     'top_probabilities': np.take_along_axis(probabilities, top, axis=-1)})
    return rows


def head_stats(pattern):
    """Per-head metrics for [B,H,query,key], with masked zeroes excluded from log."""
    pattern = np.asarray(pattern)
    if pattern.ndim != 4 or pattern.shape[-1] != pattern.shape[-2]:
        raise ValueError('Head statistics require a square [B,H,T,T] attention pattern.')
    length = pattern.shape[-1]
    distance = np.arange(length)[:, None] - np.arange(length)[None, :]
    entropy = -(pattern * np.log(np.maximum(pattern, np.finfo(pattern.dtype).tiny))).sum(axis=-1)
    return {'entropy': entropy.mean(axis=(0, 2)),
            'mean_distance': (pattern * distance).sum(axis=-1).mean(axis=(0, 2)),
            'first_token_mass': pattern[..., 0].mean(axis=(0, 2)),
            'previous_token_mass': np.diagonal(pattern[..., 1:, :-1], axis1=-2, axis2=-1).mean(axis=(0, 2)) if length > 1 else np.zeros(pattern.shape[1])}


def ablate(source, text, targets=None, mode='zero', heads=None):
    if mode not in ('zero', 'mean'):
        raise ValueError('Ablation must be zero or mean.')
    model, tokenizer = probe_model(source)
    ids = probe_ids(model, text, tokenizer)
    targets = _targets(ids, targets)
    logits, tape = model.forward(ids, hooks=True, tape=True)
    baseline = model.objective(logits, targets, xp=np)
    selected = heads
    if selected is None:
        selected = [(i, h) for i in range(model.dims.n_layer)
                    if f'block{i}.attn.head_out' in tape.captured
                    for h in range(tape.captured[f'block{i}.attn.head_out'].shape[1])]
    results = []
    for block, head in selected:
        hook = f'block{block}.attn.head_out'
        if hook not in tape.captured or not 0 <= head < tape.captured[hook].shape[1]:
            raise ValueError('Selected block/head has no attention output hook.')
        reference = tape.captured[hook][:, head].mean(axis=(0, 1), keepdims=True)
        def patch(value):
            changed = value.copy()
            changed[:, head] = 0 if mode == 'zero' else reference
            return changed
        changed = model.forward(ids, patches={hook: patch})
        loss = model.objective(changed, targets, xp=np)
        stats = head_stats(tape.captured[f'block{block}.attn.pattern'])
        results.append({'block': block, 'head': head, 'loss': loss, 'delta': loss - baseline,
                        **{key: float(value[head]) for key, value in stats.items()}})
    return {'baseline': baseline, 'mode': mode, 'heads': results}


def detectors(pattern, threshold=.5, ids=None):
    """Descriptive pattern heuristics, not explanations of a head's learned function."""
    stats = head_stats(pattern)
    findings = [{'head': head, 'kind': kind, 'value': float(value), 'threshold': threshold}
            for key, kind in (('first_token_mass', 'first-token focus'), ('previous_token_mass', 'previous-token focus'))
            for head, value in enumerate(stats[key]) if value >= threshold]
    if ids is not None:
        ids = np.asarray(ids)
        if ids.shape != (pattern.shape[0], pattern.shape[2]):
            raise ValueError('Detector ids must match the attention batch and positions.')
        total, eligible = np.zeros(pattern.shape[1]), 0
        for batch, row in enumerate(ids):
            for query in range(1, len(row)):
                keys = np.flatnonzero(row[:query] == row[query]) + 1
                if keys.size:
                    total += pattern[batch, :, query, keys].sum(axis=0)
                    eligible += 1
        if eligible:
            findings.extend({'head': head, 'kind': 'induction-pattern score', 'value': float(value), 'threshold': threshold}
                            for head, value in enumerate(total / eligible) if value >= threshold)
    return findings
