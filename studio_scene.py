"""Renderer-independent, bounded scenes for the five network zoom levels."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np

import studio_registry as registry
from studio_spec import block_at, upgrade, validate_spec
from studio_summary import bin_edges

LEVELS = ('Run', 'Model', 'Window', 'Head', 'Weights')
MODES = ('Build', 'Train', 'Ride a token')
HEAD_STAGES = ('scores', 'scale', 'mask', 'pattern', 'values')  # the forward pass of one head, in order
TILE_KINDS = ('weight', 'gradient', 'update')
LIMITS = {'Run': 2000, 'Model': 32, 'Window': 32, 'Head': 32, 'Weights': 16}


@dataclass
class Selection:
    run: str | None = None
    layer: int = 0
    token: int = 0
    head: int = 0
    weight: str | None = None
    row: int = 0
    column: int = 0


class SceneModel:
    def __init__(self, spec, selection=None):
        self.selection = selection or Selection()
        self.level, self.mode = 'Model', 'Build'
        self.reduce_motion, self.show_numbers = False, True
        self.gates = []
        self.snapshot = None
        self.paths, self.summaries, self.stream_norms = [], {}, {}
        self.summary_edges = None
        self.update_spec(spec)

    def update_spec(self, spec):
        self.spec = upgrade(spec)
        old = {gate['index']: gate for gate in self.gates}
        gates, changed = [], []
        for index in range(self.spec['dims']['n_layer']):
            entry = copy.deepcopy(block_at(self.spec, index))
            prior = old.get(index)
            if prior and prior['entry'] == entry and prior['width'] == self.spec['dims']['d_model']:
                gates.append(prior)
                continue
            glyphs = {}
            for slot in ('mixer', 'channel'):
                module = registry.get(slot, entry[slot]['id'])
                glyphs[slot] = copy.deepcopy(module.glyph3d or {'kind': 'tower' if slot == 'channel' else 'gate', 'hooks': []})
            gates.append({'index': index, 'id': entry['id'], 'entry': entry,
                          'width': self.spec['dims']['d_model'], 'glyphs': glyphs})
            changed.append(index)
        changed.extend(index for index in old if index >= len(gates))
        self.gates = gates
        self.selection.layer = min(self.selection.layer, len(gates) - 1)
        self.selection.token = min(self.selection.token, self.spec['dims']['n_ctx'] - 1)
        heads = block_at(self.spec, self.selection.layer)['mixer']['config'].get('heads', 1)
        self.selection.head = min(self.selection.head, heads - 1)
        return changed

    def choose(self, **values):
        for key, value in values.items():
            if not hasattr(self.selection, key):
                raise ValueError(f'Unknown selection field {key}.')
            if key in ('layer', 'token', 'head', 'row', 'column') and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError('Selection coordinates must be nonnegative integers.')
            setattr(self.selection, key, value)
        self.update_spec(self.spec)

    def load_snapshot(self, snapshot):
        self.snapshot = snapshot
        if snapshot.get('spec'):
            self.update_spec(snapshot['spec'])

    def swap(self, index, slot, entry):
        from studio_spec import set_block
        from studio_validate import errors
        if slot not in ('norm', 'mixer', 'channel') or not 0 <= index < len(self.gates):
            raise ValueError('Choose a valid block slot.')
        spec = copy.deepcopy(self.spec)
        set_block(spec, index, slot, entry)
        spec = validate_spec(spec)
        problems = errors(spec)
        if problems:
            raise ValueError(problems[0])
        return self.update_spec(spec)

    def frame(self, level=None):
        level = level or self.level
        if level not in LEVELS or self.mode not in MODES:
            raise ValueError('Unknown scene level or mode.')
        arrays = self.snapshot['arrays'] if self.snapshot else {}
        result = {'level': level, 'mode': self.mode, 'selection': asdict(self.selection),
                  'reduce_motion': self.reduce_motion, 'show_numbers': self.show_numbers,
                  'step': self.snapshot.get('step') if self.snapshot else None,
                  'loss': self.snapshot.get('loss') if self.snapshot else None}
        if level == 'Run':
            result['paths'] = [{**path, 'points': path['points'][:2000]} for path in self.paths]
            result['markers'] = self.snapshot.get('markers', []) if self.snapshot else []
        elif level == 'Model':
            bins = min(32, self.spec['dims']['n_ctx'])
            result['edges'] = list(self.summary_edges) if self.summary_edges is not None else bin_edges(self.spec['dims']['n_ctx'], bins).tolist()
            result['gates'] = self.gates
            bins = min(32, self.spec['dims']['n_ctx'])
            result['attention'] = {key: np.asarray(value)[:, :bins, :bins].tolist() for key, value in self.summaries.items()}
            result['stream_norms'] = {key: np.asarray(value)[:32].tolist() for key, value in self.stream_norms.items()}
        elif level in ('Window', 'Head'):
            ids = arrays.get('ids')
            length = ids.shape[1] if ids is not None else min(32, self.spec['dims']['n_ctx'])
            token = min(self.selection.token, length - 1)
            start = max(0, min(token - 16, max(0, length - 32)))
            end = min(length, start + 32)
            result['start'], result['end'] = start, end
            result['tokens'] = ids[0, start:end].tolist() if ids is not None else list(range(start, end))
            result['gates'] = self.gates
            result['patterns'] = {}
            result['streams'] = {}
            for key, value in arrays.items():
                if key.startswith('hook::') and key.endswith('attn.pattern'):
                    result['patterns'][key[6:]] = value[0, :, start:end, start:end].tolist()
                if key.startswith('hook::') and key.endswith('resid.post'):
                    result['streams'][key[6:]] = np.linalg.norm(value[0, start:end], axis=-1).tolist()
            if level == 'Head':
                block = self.selection.layer
                # logits: pre-softmax scores as the engine applies them (score scale, position bias), when it exposes them
                for name in ('scores', 'logits', 'pattern'):
                    value = arrays.get(f'hook::block{block}.attn.{name}')
                    result[name] = value[0, min(self.selection.head, value.shape[1] - 1), start:end, start:end].tolist() if value is not None else []
                # rows = queries, columns = keys, the same orientation as the Window patterns; keys after the query are masked
                result['mask'] = (np.arange(start, end)[None, :] > np.arange(start, end)[:, None]).tolist()
                output = arrays.get(f'hook::block{block}.attn.head_out')
                result['values'] = output[0, min(self.selection.head, output.shape[1] - 1), start:end, :32].tolist() if output is not None else []
        elif level == 'Weights':
            names = sorted(key[8:] for key in arrays if key.startswith('weight::') and arrays[key].ndim == 2)
            name = self.selection.weight if self.selection.weight in names else (names[0] if names else None)
            result['weight_name'], result['available'] = name, names
            for kind in ('weight', 'gradient', 'update'):
                value = arrays.get(f'{kind}::{name}')
                if value is not None:
                    row = min(self.selection.row, max(0, value.shape[0] - 16))
                    col = min(self.selection.column, max(0, value.shape[1] - 16))
                    result[kind] = value[row:row + 16, col:col + 16].tolist()
                    result['shape'], result['origin'] = tuple(value.shape), (row, col)
                else:
                    result[kind] = []
        return result

    def replay_values(self, direction='forward'):
        if direction not in ('forward', 'backward'):
            raise ValueError('Choose forward or backward replay.')
        if not self.snapshot:
            return []
        prefix = direction + '::'
        return [(key[len(prefix):], value.copy()) for key, value in self.snapshot['arrays'].items() if key.startswith(prefix)]


def next_stage(current, delta=0, stage=None):
    """The head stage after stepping `delta` (clamped to the ends of the forward pass), or `stage` itself when it is a real one."""
    if stage in HEAD_STAGES:
        return stage
    return HEAD_STAGES[max(0, min(len(HEAD_STAGES) - 1, HEAD_STAGES.index(current) + int(delta)))]


def clamp_index(index, count):
    """A snapshot index inside 0..count-1 (-1 when there are none)."""
    return max(0, min(count - 1, int(index))) if count else -1


def stats_line(stats):
    """The one-line training readout from a 'stats' message: step, loss, learning rate, speed (missing values left out)."""
    parts = []
    if stats.get('step') is not None:
        parts.append(f"step {stats['step']}" + (f"/{stats['total']}" if stats.get('total') else ''))
    if stats.get('loss') is not None:
        parts.append(f"loss {stats['loss']:.4f}")
    if stats.get('lr') is not None:
        parts.append(f"lr {stats['lr']:.2e}")
    if stats.get('tokens_per_second'):
        parts.append(f"{stats['tokens_per_second']:,.0f} tok/s")
    return ' | '.join(parts)
