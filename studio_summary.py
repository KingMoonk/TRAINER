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


def chunked_attention(q, k, v, position, n_bins=16, chunk_size=32, xp=None):
    """Exact causal softmax output and mean query-to-key-bin mass [H,K,K].

    q/k/v are [B,H,T,h] tensors. Working scores are at most [B,H,min(chunk_size,T),T].
    Position modules may provide score_bias_rows to avoid allocating a full bias matrix.
    """
    import torch
    length = q.shape[-2]
    if k.shape[-2] != length or v.shape[-2] != length or chunk_size < 1:
        raise ValueError('Attention summary requires equal sequence lengths and positive chunk size.')
    edges = bin_edges(length, n_bins)
    labels = np.repeat(np.arange(len(edges) - 1), np.diff(edges))
    mass = torch.zeros((q.shape[1], len(edges) - 1, len(edges) - 1), dtype=q.dtype, device=q.device)
    output = torch.empty_like(q)
    keys = torch.arange(length, device=q.device)
    for start in range(0, length, chunk_size):
        end = min(length, start + chunk_size)
        scores = q[..., start:end, :] @ k.transpose(-1, -2) / q.shape[-1] ** .5
        if hasattr(position, 'score_bias_rows'):
            bias = position.score_bias_rows(start, end, length, q.shape[1])
        else:
            # Existing none/learned/RoPE have no bias. ALiBi must implement the row interface.
            if getattr(getattr(position, 'module', None), 'short', None) == 'alibi':
                raise ValueError('ALiBi summaries require score_bias_rows.')
            bias = None
        if bias is not None:
            scores = scores + bias
        scores = scores.masked_fill(keys[None, :] > torch.arange(start, end, device=q.device)[:, None], -1e9)
        weights = torch.softmax(scores, dim=-1)
        output[..., start:end, :] = weights @ v
        for query_bin in np.unique(labels[start:end]):
            selected = torch.as_tensor(labels[start:end] == query_bin, device=q.device)
            for key_bin, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                mass[:, query_bin, key_bin] += weights[:, :, selected, left:right].sum(dim=(0, 2, 3))
    mass = mass / (torch.as_tensor(np.diff(edges), dtype=q.dtype, device=q.device)[None, :, None] * q.shape[0])
    return output, mass


def binned_attention(source, text, n_bins=16):
    from studio_inspect import probe_ids, probe_model
    model, tokenizer = probe_model(source)
    ids = probe_ids(model, text, tokenizer)
    logits, summaries = model.forward(ids, attention_summary=n_bins)
    to_cpu = model.backend.to_cpu
    return {'edges': bin_edges(ids.shape[1], n_bins),
            'attention': {key: to_cpu(value) for key, value in summaries.items() if key.endswith('attn.pattern')},
            'stream_norms': {key: to_cpu(value) for key, value in summaries.items() if key.endswith('resid.norm')}, 'logits': to_cpu(logits)}


SUMMARY_BINS = 32
SUMMARY_FILE = 'summary.npz'
SUMMARY_SCHEMA = 'groundwork-summary/1'


def summary_path(checkpoint):
    """Where a checkpoint's whole-context summary lives: beside the run's training snapshots (viz/step_*.npz)."""
    return Path(checkpoint).parent / 'viz' / SUMMARY_FILE


def _stamp(checkpoint):
    stat = Path(checkpoint).stat()
    return {'file': Path(checkpoint).name, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def whole_context(checkpoint, documents, n_bins=SUMMARY_BINS):
    """One inference pass over a full n_ctx window of `documents` (strings, taken in order): per attention block the mean
    query-bin to key-bin mass [heads, bins, bins], per block the residual-stream size per bin, and the bin edges 0..n_ctx. Blocks
    without attention have stream sizes only. Records the checkpoint it came from; save_summary stores it."""
    from studio_inspect import probe_model
    model, tokenizer = probe_model(checkpoint)
    n_ctx = model.dims.n_ctx
    ids = []
    for text in documents:
        ids.extend(tokenizer.encode(text[:8 * n_ctx]))  # a long document needs no more than its start; a short one continues in the next
        if len(ids) >= n_ctx:
            break
    if not ids:
        raise ValueError('This version has no held-back text to read, so its whole-context summary cannot be made yet.')
    if len(ids) < n_ctx:
        raise ValueError(f"The held-back text is shorter than the model's {n_ctx} positions, so its whole-context summary cannot be made yet.")
    bins = min(n_bins, n_ctx)
    _logits, found = model.forward(np.asarray(ids[:n_ctx], dtype=np.int64)[None, :], attention_summary=bins)
    to_cpu = model.backend.to_cpu
    with np.load(checkpoint, allow_pickle=False) as archive:  # the step is in the metadata; the weights are already loaded
        step = int(json.loads(str(archive['metadata'].item())).get('step', 0))
    return {'edges': bin_edges(n_ctx, bins).tolist(), 'n_ctx': n_ctx, 'step': step, 'checkpoint': _stamp(checkpoint),
            'attention': {key: to_cpu(value) for key, value in found.items() if key.endswith('attn.pattern')},
            'stream_norms': {key: to_cpu(value) for key, value in found.items() if key.endswith('resid.norm')}}


def save_summary(checkpoint, summary):
    path = summary_path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'schema': SUMMARY_SCHEMA, 'step': summary['step'], 'n_ctx': summary['n_ctx'], 'edges': summary['edges'],
                'checkpoint': summary['checkpoint']}  # 'step' too: run_paths reads it from every viz/*.npz
    arrays = {f'attention::{key}': value for key, value in summary['attention'].items()}
    arrays.update({f'stream_norms::{key}': value for key, value in summary['stream_norms'].items()})
    from studio_core import replace_file
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, metadata=np.array(json.dumps(metadata)), **arrays)
    replace_file(temporary, path)
    return path


def summary_state(checkpoint):
    """(state, summary) of a checkpoint's saved summary: no_checkpoint (none given, or the file is gone), none (never made or
    unreadable), stale (made from another checkpoint file or changed since) or ready."""
    if not checkpoint or not Path(checkpoint).is_file():
        return 'no_checkpoint', None
    try:
        with np.load(summary_path(checkpoint), allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata']))
            if metadata.get('schema') != SUMMARY_SCHEMA:
                return 'none', None
            summary = dict(metadata, attention={}, stream_norms={})
            for key in archive.files:
                kind, _, hook = key.partition('::')
                if hook and kind in ('attention', 'stream_norms'):
                    summary[kind][hook] = archive[key].copy()
    except (OSError, ValueError, KeyError):
        return 'none', None
    return ('ready' if summary['checkpoint'] == _stamp(checkpoint) else 'stale'), summary


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
