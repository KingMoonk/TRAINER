"""Versioned numeric visualization snapshots and exact, isolated optimizer-step replay."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from studio_inspect import detectors, probe_ids, probe_model

SCHEMA = 1


def capture(model, ids, step, before=None, loss=None, trace=None):
    """Capture a fixed probe on a private model; caller's live caches and RNG stay untouched."""
    probe, _ = probe_model(model)
    ids = probe_ids(probe, ids)
    ids = ids[:, :32]  # fixed, bounded probe; whole-model views use chunked summaries separately
    logits, tape = probe.forward(ids, hooks=True, tape=True)
    arrays = {f'hook::{key}': value.copy() for key, value in tape.captured.items()}
    arrays['ids'], arrays['logits'] = ids.copy(), logits.copy()
    norms, ratios, markers, lens = {}, {}, [], {}
    from studio_modules.ops import softmax
    from studio_slots import Ctx
    ctx = Ctx(embedding=probe.embedding, position=probe.position)
    for block in range(probe.dims.n_layer):
        hidden = tape.captured[f'block{block}.resid.post']
        if probe.final is not None:
            hidden = probe.final.forward(hidden, ctx)
        probabilities = softmax(probe.head.forward(hidden, ctx), np)
        top = np.argsort(probabilities, axis=-1)[..., -5:][..., ::-1]
        lens[str(block)] = {'ids': top[0].tolist(), 'probabilities': np.take_along_axis(probabilities, top, axis=-1)[0].tolist()}
    for key, gradient in model.g.items():
        norms[key] = float(np.linalg.norm(model.backend.to_cpu(gradient).astype(np.float64)))
        if before is not None:
            weight = model.backend.to_cpu(model.p[key])
            old = np.asarray(before[key])
            ratios[key] = float(np.linalg.norm(weight - old) / max(np.linalg.norm(old), 1e-12))
            arrays[f'update::{key}'] = (weight - old).copy()
        arrays[f'gradient::{key}'] = model.backend.to_cpu(gradient).copy()
        arrays[f'weight::{key}'] = model.backend.to_cpu(model.p[key]).copy()
    for key, value in tape.captured.items():
        if key.endswith('attn.pattern'):
            markers.extend(dict(marker, block=int(key.split('.')[0][5:])) for marker in detectors(value, ids=ids))
    if trace is not None:
        arrays.update({f'forward::{key}': model.backend.to_cpu(value).copy() for key, value in trace.forward_values.items()})
        arrays.update({f'backward::{key}': model.backend.to_cpu(value).copy() for key, value in trace.backward_values.items()})
    return {'schema': SCHEMA, 'step': int(step), 'loss': loss, 'spec': copy.deepcopy(model.spec),
            'gradient_norms': norms, 'update_ratios': ratios, 'markers': markers, 'lens': lens, 'arrays': arrays}


def save(directory, snapshot, keep=None):
    folder = Path(directory) / 'viz'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"step_{snapshot['step']:08d}.npz"
    metadata = {key: value for key, value in snapshot.items() if key != 'arrays'}
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, metadata=np.array(json.dumps(metadata)), **snapshot['arrays'])
    temporary.replace(path)
    if keep:
        for old in sorted(folder.glob('step_*.npz'))[:-keep]:
            old.unlink()
    return path


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if metadata.get('schema') != SCHEMA:
            raise ValueError('Unsupported visualization snapshot schema.')
        metadata['arrays'] = {key: archive[key].copy() for key in archive.files if key != 'metadata'}
    return metadata


def one_step(source, ids, targets, learning_rate=.001, optimizer=None):
    """Run one real step on a private model and return its exact layer outputs/returned gradients.

    A provided optimizer is cloned with all state. A checkpoint restores its saved optimizer/RNG.
    Nothing advances the original training job.
    """
    from studio_model import build_optimizer, load_checkpoint
    if isinstance(source, (str, Path)) and optimizer is not None:
        raise ValueError('A checkpoint restores its saved optimizer; do not also pass a live optimizer.')
    model, tokenizer = probe_model(source)
    ids = probe_ids(model, ids, tokenizer)
    targets = np.asarray(targets, dtype=np.int64)
    if targets.shape != ids.shape:
        raise ValueError('Targets must match the probe shape.')
    if optimizer is not None:
        from studio_spec import module_of
        module, cfg = module_of('optimizer', model.spec['train']['optimizer'])
        own = module.build(cfg, model.dims, None).attach(model)
        own.steps = optimizer.steps
        for state in own.states:
            for key in model.p:
                getattr(own, state)[key][...] = source.backend.to_cpu(getattr(optimizer, state)[key])
    else:
        own, _ = build_optimizer(model)
    if isinstance(source, (str, Path)):
        checkpoint = load_checkpoint(source)
        own.steps = checkpoint.get('optimizer_steps', checkpoint['step'])
        for state in own.states:
            for key in model.p:
                saved = checkpoint.get('optimizer_state', {}).get(state, {})
                if key not in saved:
                    raise ValueError(f'Checkpoint is missing optimizer state {state} for {key}.')
                getattr(own, state)[key][...] = saved[key]
        model.rng.bit_generator.state = checkpoint['model_rng']
    elif hasattr(source, 'rng'):
        model.rng.bit_generator.state = copy.deepcopy(source.rng.bit_generator.state)
    before = {key: value.copy() for key, value in model.p.items()}
    model.zero_grad()
    logits, tape = model.forward(ids, training=True, tape=True, hooks=True, trace=True)
    loss, derivative = model.objective(logits, targets, gradient=True, xp=np)
    model.backward(derivative, tape)
    loss += float(tape.ctx.aux)
    own.update(model, learning_rate)
    snapshot = capture(model, ids, own.steps, before, loss=loss, trace=tape)
    snapshot['exact_step'] = True
    return snapshot
