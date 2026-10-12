"""Bounded visualization summaries. Attention arithmetic uses query chunks, never a T×T array."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def bin_edges(length, n_bins=16):
    if length < 1 or not 1 <= n_bins <= 32:
        raise ValueError('Choose a positive length and 1 to 32 bins.')
    count = min(length, n_bins)
    # Symmetric exponential spacing resolves the beginning and newest tokens more finely.
    positions = np.linspace(-1, 1, count + 1)
    curved = np.sign(positions) * np.log1p(np.abs(positions) * np.expm1(3)) / 3
    desired = np.rint((curved + 1) * length / 2).astype(int)
    edges = [0]
    for i in range(1, count):
        edges.append(max(edges[-1] + 1, min(int(desired[i]), length - (count - i))))
    return np.array(edges + [length], dtype=int)


def chunked_attention(q, k, v, position, n_bins=16, chunk_size=32, xp=np):
    """Exact causal softmax output and mean query-to-key-bin mass [H,K,K].

    q/k/v are [B,H,T,h]. Working scores are at most [B,H,min(chunk_size,T),T].
    Position modules may provide score_bias_rows to avoid allocating a full bias matrix.
    """
    length = q.shape[-2]
    if k.shape[-2] != length or v.shape[-2] != length or chunk_size < 1:
        raise ValueError('Attention summary requires equal sequence lengths and positive chunk size.')
    edges = bin_edges(length, n_bins)
    labels = np.repeat(np.arange(len(edges) - 1), np.diff(edges))
    mass = xp.zeros((q.shape[1], len(edges) - 1, len(edges) - 1), dtype=q.dtype)
    output = xp.empty_like(q)
    keys = xp.arange(length)
    for start in range(0, length, chunk_size):
        end = min(length, start + chunk_size)
        scores = q[..., start:end, :] @ k.swapaxes(-1, -2) / q.shape[-1] ** .5
        if hasattr(position, 'score_bias_rows'):
            bias = position.score_bias_rows(start, end, length, q.shape[1])
        else:
            # Existing none/learned/RoPE have no bias. ALiBi must implement the row interface.
            if getattr(getattr(position, 'module', None), 'short', None) == 'alibi':
                raise ValueError('ALiBi summaries require score_bias_rows.')
            bias = None
        if bias is not None:
            scores += bias
        scores = xp.where(keys[None, :] > xp.arange(start, end)[:, None], -1e9, scores)
        scores -= scores.max(axis=-1, keepdims=True)
        weights = xp.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        output[..., start:end, :] = weights @ v
        for query_bin in np.unique(labels[start:end]):
            selected = xp.asarray(labels[start:end] == query_bin)
            for key_bin, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                mass[:, query_bin, key_bin] += weights[:, :, selected, left:right].sum(axis=(0, 2, 3))
    mass /= xp.asarray(np.diff(edges))[None, :, None] * q.shape[0]
    return output, mass


def binned_attention(source, text, n_bins=16):
    from studio_inspect import probe_ids, probe_model
    model, tokenizer = probe_model(source)
    ids = probe_ids(model, text, tokenizer)
    logits, summaries = model.forward(ids, attention_summary=n_bins)
    return {'edges': bin_edges(ids.shape[1], n_bins),
            'attention': {key: value for key, value in summaries.items() if key.endswith('attn.pattern')},
            'stream_norms': {key: value for key, value in summaries.items() if key.endswith('resid.norm')}, 'logits': logits}


def run_paths(run_ids, max_points=2000):
    if not 1 <= max_points <= 2000:
        raise ValueError('Run paths allow 1 to 2,000 points.')
    paths = []
    for run in run_ids:
        directory = Path(run)
        metrics = directory / 'metrics.jsonl'
        points = []
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                loss = value.get('val_loss')
                if loss is not None and np.isfinite(loss):
                    points.append((value['step'], loss))
        if len(points) > max_points:
            points = [points[i] for i in np.linspace(0, len(points) - 1, max_points, dtype=int)]
        markers = []
        for path in sorted((directory / 'viz').glob('*.npz')):
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive['metadata']))
            markers.extend(dict(marker, step=metadata['step']) for marker in metadata.get('markers', []))
        paths.append({'run': str(directory), 'points': points, 'markers': markers})
    return paths
