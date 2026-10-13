"""Checkpoint probes and interventions, isolated from the training model and its RNG."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from studio_compile import CompiledModel
from studio_modules.ops import softmax


def to_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


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
    model.load_weights(source.state_arrays())
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
    return {'ids': ids.copy(), 'tokens': tokens, 'logits': to_numpy(logits),
            'hooks': {name: to_numpy(value) for name, value in tape.captured.items()}, 'spec': copy.deepcopy(model.spec)}


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
    loss, derivative = model.objective(logits, targets, gradient=True)
    model.zero_grad()
    tape.ctx.scale = 0.  # explain the selected objective, excluding auxiliary router regularization
    model.backward(derivative, tape)
    groups = {}
    for key, gradient in model.g.items():
        slot, block = model.owner[key]
        group = slot if block is None else f'block{block}.{slot}'
        groups[group] = groups.get(group, 0.) + float((gradient.double() ** 2).sum())
    logits = to_numpy(logits)
    return {'loss': loss, 'targets': targets, 'probabilities': softmax(logits, np), 'dlogits': to_numpy(derivative),
            'groups': {key: value ** .5 for key, value in groups.items()},
            'gradients': {key: to_numpy(value).copy() for key, value in model.g.items()}}


def logit_lens(source, text, top_k=5):
    model, tokenizer = probe_model(source)
    ids = probe_ids(model, text, tokenizer)
    _, tape = model.forward(ids, tape=True, hooks=True)
    rows = []
    names = ['embed'] + [f'block{i}.resid.post' for i in range(model.dims.n_layer)]
    from studio_slots import Ctx
    ctx = Ctx(embedding=model.embedding, position=model.position, device=model.device)
    for name in names:
        hidden = tape.captured[name]
        with torch.no_grad():
            if model.final is not None:
                hidden = model.final.forward(hidden, ctx)
            logits = to_numpy(model.head.forward(hidden, ctx))
        probabilities = softmax(logits, np)
        k = max(1, min(int(top_k), model.dims.vocab))
        top = np.argsort(probabilities, axis=-1)[..., -k:][..., ::-1]
        rows.append({'hook': name, 'probabilities': probabilities, 'top_ids': top,
                     'top_probabilities': np.take_along_axis(probabilities, top, axis=-1)})
    return rows


def head_stats(pattern):
    """Per-head metrics for [B,H,query,key], with masked zeroes excluded from log."""
    pattern = to_numpy(pattern)
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
    baseline = float(model.objective(logits, targets))
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
        # causal running mean over positions ≤ t (and the batch), so no position reads information from the future
        head_out = tape.captured[hook][:, head]                                   # [B, T, h]
        running = head_out.cumsum(dim=1) / torch.arange(1, head_out.shape[1] + 1, device=head_out.device)[None, :, None]
        reference = running.mean(dim=0, keepdim=True)

        def patch(value):
            changed = value.clone()
            changed[:, head] = 0 if mode == 'zero' else reference
            return changed
        changed = model.forward(ids, patches={hook: patch})
        loss = float(model.objective(changed, targets))
        stats = head_stats(to_numpy(tape.captured[f'block{block}.attn.pattern']))
        results.append({'block': block, 'head': head, 'loss': loss, 'delta': loss - baseline,
                        **{key: float(value[head]) for key, value in stats.items()}})
    return {'baseline': baseline, 'mode': mode, 'heads': results}


def detectors(pattern, threshold=.5, ids=None):
    """Descriptive pattern heuristics, not explanations of a head's learned function."""
    pattern = to_numpy(pattern)
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


def _piece(tokenizer, token):
    return tokenizer.decode([int(token)]) if tokenizer else str(int(token))


def trace(source, text, pos=-1):
    """One row per stage of one position's residual stream: size, relative change, cosine with before, logit-lens top.

    Rows: embedding, then per block a mixer and a channel step, then output. The embedding has no 'before', so its
    rel_change and cosine are None. Early lens guesses are approximate (only the output row is the real prediction).
    """
    model, tokenizer = probe_model(source)
    ids = probe_ids(model, text, tokenizer)[:1]
    _, tape = model.forward(ids, hooks=True, tape=True)
    hooks = tape.captured
    pos = range(ids.shape[1])[pos]
    from studio_slots import Ctx
    ctx = Ctx(embedding=model.embedding, position=model.position, device=model.device)

    def stream(name):
        return hooks[name][0, pos].cpu().double()

    def lens(hidden):
        with torch.no_grad():
            hidden = hidden.float()[None, None]
            if model.final is not None:
                hidden = model.final.forward(hidden, ctx)
            return softmax(to_numpy(model.head.forward(hidden, ctx))[0, 0], np)

    def row(stage, kind, before, after, probabilities, approximate=True):
        after_norm = float(after.norm())
        top = int(probabilities.argmax())
        out = {'stage': stage, 'kind': kind, 'norm': after_norm, 'rel_change': None, 'cosine': None,
               'top_id': top, 'top_token': _piece(tokenizer, top), 'top_p': float(probabilities[top]),
               'approximate': approximate}
        if before is not None:
            out['rel_change'] = float((after - before).norm() / max(float(before.norm()), 1e-12))
            out['cosine'] = float(np.clip((before @ after) / max(float(before.norm()) * after_norm, 1e-12), -1, 1))
        return out

    x = stream('embed')
    rows = [row('embedding', 'embedding', None, x, lens(x))]
    for i in range(model.dims.n_layer):
        pre, mid, post = (stream(f'block{i}.resid.{part}') for part in ('pre', 'mid', 'post'))
        rows.append(row(f'block {i + 1} mixer', 'mixer', pre, mid, lens(mid)))
        rows.append(row(f'block {i + 1} channel', 'channel', mid, post, lens(post)))
    final = stream('final')
    probabilities = softmax(to_numpy(hooks['logits'])[0, pos], np)
    rows.append(row('output', 'output', stream(f'block{model.dims.n_layer - 1}.resid.post') if model.dims.n_layer else x,
                    final, probabilities, approximate=False))
    return {'tokens': [_piece(tokenizer, t) for t in ids[0]], 'pos': pos, 'rows': rows}


def prepared_text(tokenizer, n=4, seed=0):
    """BOS + n distinct random pieces + the same n again, so copying patterns separate from the rest."""
    from studio_slots import BOS, OFFSET, UNK
    import string
    pool = sorted({t for t in tokenizer.encode(string.ascii_lowercase) if t != UNK}) or list(range(OFFSET, tokenizer.vocab_size))
    pieces = np.random.default_rng(seed).choice(pool, size=min(n, len(pool)), replace=False).tolist()
    return [BOS] + pieces + pieces


def head_tests(pattern, ids):
    """Three descriptive tests per head on [B,H,T,T] attention (rows = queries). ids: [T] or [B,T] of BOS + A + A.

    previous: mean share on the previous piece; first: mean share on the first piece (an idle-head sign);
    induction: on the repeated half, share on the piece after the earlier copy of the current piece.
    """
    pattern = to_numpy(pattern)
    ids = np.asarray(ids)
    ids = ids[None] if ids.ndim == 1 else ids
    if pattern.ndim != 4 or ids.shape != (pattern.shape[0], pattern.shape[2]):
        raise ValueError('head_tests needs [B,H,T,T] attention and matching ids.')
    length = ids.shape[1]
    previous = np.diagonal(pattern[..., 1:, :-1], axis1=-2, axis2=-1).mean(axis=(0, 2)) if length > 1 else np.zeros(pattern.shape[1])
    first = pattern[..., 1:, 0].mean(axis=(0, 2)) if length > 1 else np.zeros(pattern.shape[1])
    total, count = np.zeros(pattern.shape[1]), 0
    for b, row in enumerate(ids):
        for q in range((length - 1) // 2 + 1, length):
            keys = np.flatnonzero(row[:q - 1] == row[q]) + 1  # key after an earlier copy of the current piece
            if keys.size:
                total += pattern[b, :, q, keys].sum(axis=-1)
                count += 1
    return {'previous': previous, 'first': first, 'induction': total / count if count else np.zeros(pattern.shape[1])}


def rank_heads(source, ids, mode='mean'):
    """Each head ablated alone on a fixed batch: one entry per (block, head), rise = loss rise in nats per piece.

    mode 'mean' is the causal running mean over positions <= t (see ablate); 'zero' is harsher.
    """
    result = ablate(source, ids, mode=mode)
    return [{'block': r['block'], 'head': r['head'], 'rise': r['delta'], 'loss': r['loss'], 'baseline': result['baseline']}
            for r in result['heads']]
