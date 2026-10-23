"""The 3D viewer's scene as plain data for a browser renderer (react-three-fiber), and the viewer's controls as plain functions.

No Panda3D, no Tk. `scene_payload` is the whole scene to draw from scratch, `scene_frame` only what changed since the
client's last cursor, `scene_binary` the same arrays packed as float32 for typed arrays. Geometry mirrors
The removed Panda3D viewer used the same coordinates (x right, y into the screen, z up), the same limits, the same colour per meaning.

Shape of a payload (a frame has the same keys, only for what changed, plus `removed_groups`):
    view      level, mode, head stage, tile kind, toggles, which controls show, camera target and distance
    selection the selected block, token, head, matrix and tile origin
    stats     step, loss, the training line, the HUD line, the viewer's note
    design    the blocks: id, label, mixer and channel entries, glyph kinds
    playback  snapshot position, Play/Previous/Next state, the replay pulse (see `pulse_pose`)
    groups    {key: group}: `gate:<i>` per block at Model and Window, `matrix` at Head and Weights. A group has an
              `origin`, `boxes` and `lines` (one entry per colour kind, columns as {shape, data}) and `labels`
A box is centred on its position, `size` is its full extent. Colours are kind ids; `palette` maps each to a colour.

Thermal look (`ViewState.thermal`, the browser's default): the Head, Weights and Model levels carry `tiles` in their groups instead of
bars. A tile is a real matrix to draw as a heat-map plane facing the camera: `values` (rows x columns, at most TILE_CAP each way,
block-averaged when the matrix is larger; `shape` is the true size), `range` (the ramp's low and high, from the whole matrix), its
`origin` (centre) and `size` (width, height) in the group, and optionally a `mask` (1 = hidden cell) and `selected`. A tile keeps its
`id` from frame to frame, so a client updates it in place. Connectors are `connector` lines (arrow shafts and heads).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from studio_scene import HEAD_STAGES, MODES, TILE_KINDS, SceneModel, clamp_index, next_stage, stats_line
from studio_spec import block_at

SCHEMA = 1
VIEWER_LEVELS = ('Model', 'Window', 'Head', 'Weights')  # the viewer offers these four; Run needs loss paths the studio no longer sends
GATE_SPACING, LANE_PITCH, MATRIX_PITCH = 3., .38, .4  # between blocks (y), between tokens (x), between matrix cells
BIN_START, BIN_SPAN = -5.5, 11.  # the Model level's bins run along x from here, this far
TILE_CAP = 32  # a tile carries at most this many rows and columns of values, so a payload stays small however large the matrix
STACK_HEADS = 16  # a layer's head stack draws at most this many query heads
STACK_X, STACK_WIDTH = 7.2, 7.  # where the Model level's head stacks start (beside a block, which spans x -6..6), and how wide they may be
POSTER_TOKENS = 16  # token labels per tile axis: more are thinned to every n-th

# the theme colours the scene uses
THEME = {'bg': '#0e1014', 'panel': '#15181e', 'panel2': '#1b1f27', 'border': '#2a2f3a', 'text': '#e6e8ec', 'text2': '#c3c8d1',
         'violet': '#9f8cff', 'amber': '#e5b567', 'blue': '#6cb4ee', 'pink': '#e58fb0', 'violet_bg': '#1a1730'}
# kind id -> (theme colour, alpha): one colour per meaning
KINDS = {'activation': ('blue', .85), 'mixer': ('pink', .85), 'channel': ('violet', .7), 'token': ('amber', 1.),
         'lane': ('border', 1.), 'gate': ('border', 1.), 'positive': ('amber', .95), 'negative': ('blue', .95),
         'masked': ('panel2', 1.), 'picked': ('text', 1.), 'update': ('violet', .8),  # update: colour is per box
         'connector': ('text', .9)}
PALETTE = {kind: {'color': THEME[name], 'alpha': alpha} for kind, (name, alpha) in KINDS.items()}
PALETTE['background'] = {'color': THEME['bg'], 'alpha': 1.}
PALETTE['note'] = {'color': '#e0ebff', 'alpha': 1.}  # label_at: token numbers and the empty-matrix note
PALETTE['label'] = {'color': THEME['text2'], 'alpha': 1., 'frame': THEME['panel'], 'frame_alpha': .9}
PALETTE['label_selected'] = {'color': THEME['violet'], 'alpha': 1., 'frame': THEME['violet_bg'], 'frame_alpha': .9}
PALETTE['tile_title'] = {'color': THEME['text'], 'alpha': 1.}  # a tile's name (W_Q, QK^T, A)
PALETTE['tile_dims'] = {'color': THEME['text2'], 'alpha': 1.}  # a tile's shape (16 x 64)
PALETTE['tile_axis'] = {'color': THEME['text2'], 'alpha': 1.}  # a token under a tile's rows or columns


@dataclass
class ViewState:
    """What the Panda3D viewer keeps outside SceneModel: the head step, tile kind, picked cell, replay and the HUD note."""
    head_stage: str = 'pattern'
    tile_kind: str = 'weight'
    cell: list = field(default_factory=lambda: [0, 0])  # the picked weight, relative to the visible 16x16 tile
    playing: bool = False
    snapshot_paths: list = field(default_factory=list)  # server-side; the payload carries only the count and position
    snapshot_index: int = -1
    probing: bool = False
    thermal: bool = False  # heat-map tiles in place of bars (see the module doc)
    token_text: list | None = None  # the snapshot's first sequence as text, one entry per token, for labels (None: the ids)
    linked: bool = True  # False = a demo with no studio behind it: no snapshot buttons
    auto_orbit: bool = False  # the demo's slow camera orbit
    stats: dict = field(default_factory=dict)
    note: str = ''


# --- arrays: numpy inside, {shape, data} (JSON) or {shape, name} (a slice of the binary buffer) outside ------------------

class _Array:
    __slots__ = ('data',)

    def __init__(self, values):
        self.data = np.ascontiguousarray(np.nan_to_num(np.asarray(values, dtype='<f4'), nan=0., posinf=0., neginf=0.))  # inline and binary agree


def _encode(node, mode, path=''):
    """A part as JSON-safe values. mode 'inline' puts the numbers in; 'binary' names the array inside the buffer."""
    if isinstance(node, _Array):
        if mode == 'binary':
            return {'shape': list(node.data.shape), 'name': path}
        return {'shape': list(node.data.shape), 'data': [float(f'{v:.6g}') for v in node.data.ravel().tolist()]}
    if isinstance(node, dict):
        return {str(key): _encode(value, mode, f'{path}/{key}') for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_encode(value, mode, f'{path}/{i}') for i, value in enumerate(node)]
    if isinstance(node, np.ndarray):
        return _encode(node.tolist(), mode, path)
    if isinstance(node, np.generic):
        node = node.item()
    if isinstance(node, float) and not math.isfinite(node):
        return None
    return node


def _arrays(node, path=''):
    if isinstance(node, _Array):
        yield path, node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _arrays(value, f'{path}/{key}')
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            yield from _arrays(value, f'{path}/{i}')


def _digest(node, path):
    """Changes exactly when the part's JSON or any of its numbers change."""
    digest = hashlib.blake2b(json.dumps(_encode(node, 'binary', path), sort_keys=True, separators=(',', ':')).encode(), digest_size=8)
    for _, array in _arrays(node, path):
        digest.update(array.data.tobytes())
    return digest.hexdigest()


def _fit(matrix, cap=TILE_CAP):
    """`matrix` (rows x columns) shrunk to at most `cap` rows and columns by averaging neighbouring blocks."""
    out = np.asarray(matrix, dtype=float)
    for axis in (0, 1):
        n = out.shape[axis]
        if n > cap:
            edges = np.linspace(0, n, cap + 1).astype(int)
            out = np.add.reduceat(out, edges[:-1], axis=axis) / np.diff(edges).reshape((-1, 1) if axis == 0 else (1, -1))
    return out


# --- one keyed piece of geometry ----------------------------------------------------------------------------------------

class _Group:
    def __init__(self, origin=(0., 0., 0.), **meta):
        self.origin, self.meta = origin, meta
        self._boxes, self._lines, self._labels, self._tiles = {}, {}, [], []

    def box(self, kind, position, size, alpha=None, **extra):
        """extra columns (value, color, cell) group boxes with the same columns into one entry."""
        alpha = KINDS[kind][1] if alpha is None else alpha
        self._boxes.setdefault((kind, tuple(extra)), []).append((position, size, alpha, extra))

    def line(self, kind, points, width=1.):
        self._lines.setdefault((kind, len(points)), []).append((points, width))

    def label(self, text, position, kind='label', selected=False, size=.28):
        self._labels.append({'text': str(text), 'position': [float(v) for v in position], 'kind': kind, 'selected': selected, 'size': size})

    def tile(self, tid, values, origin, size, *, label='', shape=None, mask=None, selected=False, low=None, high=None):
        """A heat-map plane (facing -y) centred on `origin`, `size` = (width, height). `values` None draws it empty (no data yet)."""
        entry = {'id': tid, 'label': label, 'origin': [float(v) for v in origin], 'size': [float(v) for v in size], 'selected': bool(selected)}
        if values is None:
            entry.update(shape=list(shape or (0, 0)), range=[0., 1.])
        else:
            whole = np.nan_to_num(np.asarray(values, dtype=float), nan=0., posinf=0., neginf=0.)
            fitted = _fit(whole)
            entry.update(shape=list(shape or whole.shape), values=_Array(fitted),
                         range=[float(whole.min() if low is None else low), float(whole.max() if high is None else high)])
            if mask is not None:
                entry['mask'] = _Array(_fit(np.asarray(mask, dtype=float)) > .5)
        self._tiles.append(entry)

    def arrow(self, start, end, head=.28):
        """A thin connector from `start` to `end` with an open arrowhead at `end` (in the xz plane the tiles stand in)."""
        dx, dz = end[0] - start[0], end[2] - start[2]
        length = math.hypot(dx, dz) or 1.
        ux, uz = dx / length, dz / length
        self.line('connector', [tuple(start), tuple(end)], .8)
        wing = lambda sign: (end[0] - ux * head - sign * uz * head * .5, end[1], end[2] - uz * head + sign * ux * head * .5)  # noqa: E731
        self.line('connector', [wing(1), tuple(end), wing(-1)], .8)

    def encode(self):
        boxes = []
        for (kind, names), rows in self._boxes.items():
            entry = {'kind': kind, 'count': len(rows), 'position': _Array([r[0] for r in rows]), 'size': _Array([r[1] for r in rows]),
                     'alpha': _Array([r[2] for r in rows])}
            entry.update({name: _Array([r[3][name] for r in rows]) for name in names})
            boxes.append(entry)
        lines = [{'kind': kind, 'count': len(rows), 'points': _Array([r[0] for r in rows]), 'width': _Array([r[1] for r in rows])}
                 for (kind, _), rows in self._lines.items()]
        out = {'origin': [float(v) for v in self.origin], **self.meta, 'boxes': boxes, 'lines': lines, 'labels': self._labels}
        if self._tiles:
            out['tiles'] = self._tiles
        return out


# --- the controls' decisions ---

def block_matrices(names, layer):
    """The matrices the menu offers: the selected block's, plus the ones outside any block (token, position embeddings)."""
    return [n for n in names if not n.startswith('block') or n.startswith(f'block{layer}.')]


def pick_matrix(names, layer, current):
    """The matrix to show after the block or the snapshot changes: the current one if still offered, else the block's first."""
    offered = block_matrices(names, layer)
    if current in offered:
        return current
    return next((n for n in offered if n.startswith('block')), offered[0] if offered else None)


def weight_names(model):
    arrays = model.snapshot['arrays'] if model.snapshot else {}
    return sorted(key[8:] for key in arrays if key.startswith('weight::') and arrays[key].ndim == 2)


def view_rows(level, mode, matrices):
    """Which control rows show: zoom and mode always; the block's mixer in Build; the head step at Head; the tile kind and matrix at Weights."""
    rows = ['level', 'mode']
    if mode == 'Build':
        rows.append('mixer')
    if level == 'Head':
        rows.append('stage')
    if level == 'Weights':
        rows.append('tile')
        if matrices:
            rows.append('matrix')
    return rows


def playback_state(count, index, playing):
    if not count:
        return {'show': False, 'play': 'Play', 'position': 'No snapshots loaded.', 'previous': False, 'next': False}
    return {'show': True, 'play': 'Pause' if playing else 'Play', 'position': f'Snapshot {index + 1} of {count}',
            'previous': index > 0, 'next': index + 1 < count}


def snapshot_buttons(linked, probing):
    if not linked:
        return {'show': False, 'probe': 'Run one real step', 'probe_enabled': False,
                'note': 'Demo model. Open the viewer from the studio to load your snapshots.'}
    return {'show': True, 'probe': 'Running one step...' if probing else 'Run one real step', 'probe_enabled': not probing, 'note': ''}


def toggle_names(level, snapshots, auto_orbit):
    shown = []
    if snapshots or auto_orbit:
        shown.append('motion')
    if level != 'Model':
        shown.append('numbers')
    return shown


def matrix_legend(level):
    legend = 'amber = positive, blue = negative, taller = larger'
    if level == 'Head':
        return f'rows = queries, columns = keys | {legend} | flat dark = masked (a later key)'
    return f'{legend} | white = the picked cell'


def empty_matrix_note(level, has_snapshot, layer, mixer, has_heads):
    if not has_snapshot:
        return 'Load snapshots, or run one real step, to see these values.'
    if level == 'Head' and not has_heads:
        return f'Block {layer + 1} mixes tokens with {mixer}, which has no attention heads. Tab moves to the next block.'
    return f'This snapshot has no {level.lower()} values for block {layer + 1}.'


def camera_view(level, blocks, rows=16):
    """(target (x, y, z), distance) that frames a level: the column of blocks at Model and Window, the matrix at Head and Weights."""
    if level in ('Head', 'Weights'):
        return (0., -(max(1, rows) - 1) * .2, 0.), max(8., rows * .75)
    return (0., -max(1, blocks - 1) * 1.5, 1.), max(22., blocks * 3.5)


def pass_seconds(blocks):
    """How long Play shows one snapshot: the pulse runs forward through every block, then back, two blocks a second."""
    return max(1, blocks)


def pulse_pose(blocks, elapsed, reduce_motion=False):
    """(block, depth, forward) of the replay pulse `elapsed` seconds into a snapshot. depth is the y distance in block spacings."""
    phase = (elapsed * 2) % (2 * blocks)
    forward = phase < blocks
    block = min(blocks - 1, int(phase if forward else 2 * blocks - phase - 1e-6))
    return block, (block if reduce_motion else (phase if forward else 2 * blocks - phase)), forward


# --- the scene ----------------------------------------------------------------------------------------------------------

def _gate_group(model, frame, gate, view=None):
    """One block at Model or Window, in coordinates local to the block."""
    index, selected = gate['index'], gate['index'] == model.selection.layer
    entry = gate['entry']
    group = _Group((0., -index * GATE_SPACING, 0.), index=index, selected=selected)
    group.box('gate', (0., 0., 1.5), (12., .36, 3.), alpha=.65 if selected else .3)
    if model.snapshot:
        ratios = [value for key, value in model.snapshot.get('update_ratios', {}).items() if key.startswith(f'block{index}.')]
        brightness = min(1., max(ratios, default=0.) * 20)
        group.box('update', (0., 0., .05), (12., .42, .08), alpha=.8, color=(brightness, .25 + brightness * .5, .3, .8), value=brightness)
    group.label(f"{index + 1}  {entry['mixer']['id']} / {entry['channel']['id']}", (0., 0., 3.5),
                kind='label_selected' if selected else 'label', selected=selected)
    if frame['level'] == 'Model':
        bins = len(frame['edges']) - 1
        norms = frame['stream_norms'].get(f'block{index}.resid.norm')
        peak = max((max(value, default=0.) for value in frame['stream_norms'].values()), default=0.)  # one scale for every block
        for b in range(bins):
            x = BIN_START + BIN_SPAN * b / max(1, bins - 1)
            if norms is not None and b < len(norms) and peak > 0:  # the stream's size in this bin of the context, raw in `value`
                group.box('activation', (x, 0., 1.5), (.18, .6, min(2., max(.1, 2. * norms[b] / peak))), value=float(norms[b]))
            else:
                group.box('activation', (x, 0., 1.5), (.18, .6, .8))
        for hook, mass in frame['attention'].items():
            if not hook.startswith(f'block{index}.'):
                continue
            mean = np.asarray(mass).mean(axis=0)
            for query in range(mean.shape[0]):
                for key in np.argsort(mean[query])[-4:]:
                    weight = float(mean[query, key])
                    if weight < .05:
                        continue
                    x1, x2 = (BIN_START + BIN_SPAN * i / max(1, bins - 1) for i in (query, key))
                    group.line('mixer', [(x1, 0., 2.), ((x1 + x2) / 2, 0., 2.5 + abs(x1 - x2) / 3), (x2, 0., 2.)], 1 + weight * 3)
        if view is not None and view.thermal:
            _head_stack(group, model, view, frame, gate)
        return group
    count = len(frame['tokens'])
    pattern = frame['patterns'].get(f'block{index}.attn.pattern')
    pattern = np.asarray(pattern)[min(model.selection.head, len(pattern) - 1)] if pattern is not None else None
    norms = frame['streams'].get(f'block{index}.resid.post', [])
    glyph = gate['glyphs']['mixer']['kind']
    for lane in range(count):
        x = (lane - (count - 1) / 2) * LANE_PITCH
        group.line('lane', [(x, 1.5, .1), (x, -1.5, .1)])
        magnitude = min(2., norms[lane]) if norms else .25
        group.box('token' if lane + frame['start'] == model.selection.token else 'activation',
                  (x, 0., magnitude / 2), (.16, .28, max(.1, magnitude)))
        if glyph == 'beams' and pattern is not None:
            for key in np.argsort(pattern[lane])[-4:]:
                weight = float(pattern[lane, key])
                if key > lane or weight < .05:
                    continue
                key_x = (key - (count - 1) / 2) * LANE_PITCH
                group.line('mixer', [(x, 0., .4), ((x + key_x) / 2, .5, 1.), (key_x, 1., .4)], 1 + weight * 4)
        elif glyph == 'ribbon':
            group.line('mixer', [(x, 1.3, .6), (x, -1.3, .6)], 3.)
        elif glyph == 'fan':
            for offset in range(min(entry['mixer']['config'].get('kernel_size', 4), lane + 1)):
                group.line('mixer', [(x, 0., .6), (x - offset * LANE_PITCH, 1., .1)])
        if entry['channel']['id'] != 'none':
            group.box('channel', (x, -.7, 1.), (.12, .2, .8))
        if model.show_numbers and index == 0:
            group.label(frame['tokens'][lane], (x, 1.5, .2), kind='note', size=.14)
    return group


def _matrix_values(model, frame, view):
    """The matrix the Head or Weights level draws and the mask over it, or (empty, None)."""
    if frame['level'] == 'Weights':
        return np.asarray(frame.get(view.tile_kind, []), dtype=float), None
    matrix = np.asarray(frame.get(view.head_stage, []), dtype=float)
    mask = np.asarray(frame['mask'], dtype=bool)
    if view.head_stage in ('scale', 'mask'):
        matrix = np.asarray(frame.get('logits') or [], dtype=float)  # what softmax sees: score scale and position bias included
        if not matrix.size:
            matrix = np.asarray(frame['scores'], dtype=float)
            if matrix.size:  # snapshots without attn.logits: plain q.k / sqrt(head_dim)
                heads = block_at(model.spec, model.selection.layer)['mixer']['config'].get('heads', 1)
                matrix = matrix / math.sqrt(model.spec['dims']['d_model'] / heads)
        if matrix.size and view.head_stage == 'mask':
            matrix = np.where(mask, 0, matrix)
    return matrix, mask


def _matrix_group(model, frame, view):
    level = frame['level']
    matrix, mask = _matrix_values(model, frame, view)
    matrix = np.nan_to_num(matrix, nan=0., posinf=0., neginf=0.)  # a diverged run's weights still draw
    group = _Group(level=level, kind=view.tile_kind if level == 'Weights' else view.head_stage,
                   weight_name=frame.get('weight_name'), shape=list(frame['shape']) if frame.get('shape') else None,
                   origin_cell=list(frame['origin']) if frame.get('origin') else [0, 0], pitch=MATRIX_PITCH, caption='')
    if not matrix.size or matrix.ndim != 2:
        gate = model.gates[model.selection.layer]
        group.label(empty_matrix_note(level, bool(model.snapshot), model.selection.layer, gate['entry']['mixer']['id'],
                                      gate['glyphs']['mixer']['kind'] == 'beams'), (0., 0., 2.), kind='note')
        return group, 16
    rows, columns = matrix.shape
    maximum = max(float(np.max(np.abs(matrix))), 1e-12)
    use_mask = level == 'Head' and view.head_stage != 'values' and mask is not None and mask.shape == matrix.shape
    for row in range(rows):
        for column in range(columns):
            value = float(matrix[row, column])
            masked = bool(use_mask and mask[row, column])
            kind = 'masked' if masked else 'positive' if value > 0 else 'negative'
            if level == 'Weights' and [row, column] == view.cell:
                kind = 'picked'
            height = .015 if masked else .03 + abs(value) / maximum * 1.5
            group.box(kind, ((column - columns / 2) * MATRIX_PITCH, -row * MATRIX_PITCH, height / 2), (.36, .36, height),
                      value=value, cell=(row, column))
    group.meta['caption'] = matrix_legend(level)
    if model.show_numbers:
        numbers = f'{rows}x{columns} | range {matrix.min():.3g} to {matrix.max():.3g}'
        if level == 'Weights':
            cell = (min(view.cell[0], rows - 1), min(view.cell[1], columns - 1))
            numbers += f' | cell {cell[0]},{cell[1]} = {float(matrix[cell]):.4g} (I J K L move)'
        group.meta['caption'] += f'\n{numbers}'
    return group, rows

# --- the thermal look: real matrices as heat-map tiles --------------------------------------------------------------------

POSTER_HEIGHT = 3.  # a tile of tokens is this tall (scene units)
QKV_ROWS = (4.2, 0., -4.2)  # the z of the Q, K and V rows of a head's poster
VARIANTS = ((lambda heads, shared: shared == heads, 'MHA'), (lambda heads, shared: shared == 1, 'MQA'), (lambda heads, shared: True, 'GQA'))


def token_names(view, ids, start, count):
    """The window's tokens as readable labels: the text when `view` has it, else the ids. A space shows as a dot, a break as a return sign."""
    text = view.token_text or []
    names = []
    for i in range(start, start + count):
        raw = text[i] if i < len(text) else None
        if raw is None:
            raw = str(int(ids[i])) if i < len(ids) else str(i)
        elif raw == '':
            raw = f'#{int(ids[i])}' if i < len(ids) else '#'
        names.append(raw.replace(' ', '\u00b7').replace('\n', '\u21b5').replace('\t', '\u2192')[:6])
    return names


def _axis_labels(group, tile, names, rows=False, columns=False):
    """Token labels beside a tile's rows and above its columns, thinned to POSTER_TOKENS each."""
    (x, _, z), (w, h) = tile['origin'], tile['size']
    count = len(names)
    step = max(1, math.ceil(count / POSTER_TOKENS))
    for i in range(0, count, step):
        if rows:
            group.label(names[i], (x - w / 2 - .08, 0., z + h / 2 - (i + .5) * h / count), kind='tile_axis', size=.14)
        if columns:
            group.label(names[i], (x - w / 2 + (i + .5) * w / count, 0., z + h / 2 + .14), kind='tile_axis', size=.14)


def _caption(group, tile, name, shape_text):
    (x, _, z), (_, h) = tile['origin'], tile['size']
    group.label(name, (x, 0., z + h / 2 + .38), kind='tile_title', size=.3)
    group.label(shape_text, (x, 0., z - h / 2 - .3), kind='tile_dims', size=.16)


def _place(group, tid, values, centre, size, *, name, shape, mask=None, selected=False, low=None, high=None, names=None, rows=False, columns=False):
    """One tile with its name above, its shape below and the token labels asked for."""
    group.tile(tid, values, centre, size, label=name, shape=shape, mask=mask, selected=selected, low=low, high=high)
    tile = {'origin': centre, 'size': size}
    _caption(group, tile, name, ' \u00d7 '.join(str(n) for n in shape))
    if names is not None:
        _axis_labels(group, tile, names, rows, columns)
    return tile


def _edge(tile, side):
    """The middle of a tile's left, right, top or bottom edge, nudged off it so an arrow does not touch."""
    (x, y, z), (w, h) = tile['origin'], tile['size']
    return {'left': (x - w / 2 - .12, y, z), 'right': (x + w / 2 + .12, y, z), 'top': (x, y, z + h / 2 + .12), 'bottom': (x, y, z - h / 2 - .12)}[side]


def _head_weights(arrays, block, mixer_id, head, kv_group, width, heads, kv_heads):
    """(W_Q, W_K, W_V) columns of one head, each (d_model x head size), or Nones when the snapshot does not hold them."""
    depth = max(1, width // max(1, heads))
    prefix = f'weight::block{block}.'
    if mixer_id.endswith('gqa'):
        sources = ((f'{prefix}gqa.q.w', head), (f'{prefix}gqa.k.w', kv_group), (f'{prefix}gqa.v.w', kv_group))
        picks = [(arrays.get(name), index) for name, index in sources]
    else:
        fused = arrays.get(f'{prefix}qkv.w')
        picks = [(fused, part * heads + head) for part in range(3)] if fused is not None else [(None, 0)] * 3
    out = []
    for matrix, index in picks:
        out.append(None if matrix is None or matrix.ndim != 2 or matrix.shape[1] < (index + 1) * depth else matrix[:, index * depth:(index + 1) * depth])
    return out


def _head_poster(group, model, view, frame):
    """The Head level as the real matrices of one head, left to right: X, W_Q W_K W_V, Q K V, QK^T, A (the causal mask visible), head out."""
    arrays = model.snapshot['arrays'] if model.snapshot else {}
    block, selection = model.selection.layer, model.selection
    start, end = frame['start'], frame['end']
    count = end - start
    q = arrays.get(f'hook::block{block}.attn.q')
    if q is None or q.ndim != 4:
        return False
    heads = q.shape[1]
    head = min(selection.head, heads - 1)
    k, v = arrays.get(f'hook::block{block}.attn.k'), arrays.get(f'hook::block{block}.attn.v')
    kv_heads = k.shape[1] if k is not None else heads
    kv_group = min(kv_heads - 1, head // max(1, heads // kv_heads))
    x = arrays.get(f'hook::block{block}.resid.pre')
    ids = arrays['ids'][0].tolist() if 'ids' in arrays else list(range(count))
    names = token_names(view, ids, start, count)
    mixer_id = model.gates[block]['entry']['mixer']['id']
    width = model.spec['dims']['d_model']
    weights = _head_weights(arrays, block, mixer_id, head, kv_group, width, heads, kv_heads)
    window = slice(start, end)
    depth = q.shape[3]
    stage = view.head_stage
    mask = np.arange(start, end)[None, :] > np.arange(start, end)[:, None]
    tall = POSTER_HEIGHT
    tiles = {}

    def put(tid, values, centre, size, name, shape, **extra):
        tiles[tid] = _place(group, tid, values, centre, size, name=name, shape=shape, **extra)

    put('x', x[0, window] if x is not None else None, (-15.5, 0., 0.), (3.6, tall), 'X', (count, width), names=names, rows=True)
    for row, (tid, label, source, matrix, kind) in zip(QKV_ROWS, (('q', 'Q', q, weights[0], 'wq'), ('k', 'K', k, weights[1], 'wk'), ('v', 'V', v, weights[2], 'wv'))):
        part = None if source is None else source[0, head if tid == 'q' else kv_group, window]
        if matrix is not None:
            put(kind, matrix, (-10., 0., row), (1.4, tall), 'W_' + label, matrix.shape)
            group.arrow(_edge(tiles['x'], 'right') if row == 0 else (_edge(tiles['x'], 'right')[0], 0., row * .45), _edge(tiles[kind], 'left'))
            group.arrow(_edge(tiles[kind], 'right'), (-6.6, 0., row))
        else:
            group.arrow((_edge(tiles['x'], 'right')[0], 0., row * .45), (-6.6, 0., row))
        put(tid, part, (-5.5, 0., row), (1.8, tall), label + ('' if tid == 'q' or kv_heads == heads else f' (shared by {heads // kv_heads})'),
            (count, depth))
    scores = arrays.get(f'hook::block{block}.attn.scores')
    pattern = arrays.get(f'hook::block{block}.attn.pattern')
    out = arrays.get(f'hook::block{block}.attn.head_out')
    middle = (QKV_ROWS[0] + QKV_ROWS[1]) / 2
    sources = [tiles['q'], tiles['k']]
    here = None
    if scores is not None:
        here = (.8, 0., middle)
        put('scores', scores[0, head, window, window], here, (tall, tall), 'QK\u1d40', (count, count), mask=mask,
            selected=stage in ('scores', 'scale', 'mask'), names=names, rows=True, columns=True)
        for source, side in zip(sources, ('top', 'bottom')):
            group.arrow(_edge(source, 'right'), _edge(tiles['scores'], 'left')[:2] + (tiles['scores']['origin'][2] + (tall * .3 if side == 'top' else -tall * .3),))
    if pattern is not None:
        centre = (5.6, 0., middle)
        put('pattern', pattern[0, head, window, window], centre, (tall, tall), 'A = softmax (masked)', (count, count), mask=mask,
            selected=stage == 'pattern', names=names, rows=True, columns=True, low=0.)
        if 'scores' in tiles:
            group.arrow(_edge(tiles['scores'], 'right'), _edge(tiles['pattern'], 'left'))
        else:
            for source, side in zip(sources, (.3, -.3)):
                group.arrow(_edge(source, 'right'), (_edge(tiles['pattern'], 'left')[0], 0., middle + tall * side))
        here = centre
    if out is not None:
        lower = (here[2] if here else middle) + QKV_ROWS[2]
        put('out', out[0, head, window], (11.2, 0., lower / 2), (1.8, tall), 'A\u00b7V (head out)', (count, depth), selected=stage == 'values', names=names, rows=True)
        if 'pattern' in tiles:
            group.arrow(_edge(tiles['pattern'], 'right'), _edge(tiles['out'], 'left')[:2] + (tiles['out']['origin'][2] + tall * .3,))
        group.arrow(_edge(tiles['v'], 'right'), (_edge(tiles['out'], 'left')[0], 0., tiles['out']['origin'][2] - tall * .3))
    return True


def _weights_tile(group, model, view, frame):
    """The Weights level: the visible 16 x 16 piece of one matrix as a heat map, the picked cell framed."""
    tile = np.nan_to_num(np.asarray(frame.get(view.tile_kind, []), dtype=float), nan=0., posinf=0., neginf=0.)
    rows, columns = tile.shape
    size = 6.
    where = frame['origin'] if frame.get('origin') else (0, 0)
    group.tile('weights', tile, (0., 0., 0.), (size, size), label=frame.get('weight_name') or '', shape=frame.get('shape') or tile.shape)
    shape = frame.get('shape') or tile.shape
    group.label(f"{frame.get('weight_name')}  {shape[0]} \u00d7 {shape[1]}", (0., 0., size / 2 + .45), kind='tile_title', size=.3)
    group.label(f'rows {where[0]}-{where[0] + rows - 1}, columns {where[1]}-{where[1] + columns - 1}', (0., 0., -size / 2 - .3), kind='tile_dims', size=.16)
    row, column = min(view.cell[0], rows - 1), min(view.cell[1], columns - 1)
    x, z = (column + .5) * size / columns - size / 2, size / 2 - (row + .5) * size / rows
    dx, dz = size / columns / 2, size / rows / 2
    group.line('connector', [(x - dx, -.02, z - dz), (x + dx, -.02, z - dz), (x + dx, -.02, z + dz), (x - dx, -.02, z + dz), (x - dx, -.02, z - dz)], 2.)


def _thermal_matrix_group(model, frame, view):
    """_matrix_group's counterpart for the thermal look: same meta (caption, weight name, shape), tiles instead of bars."""
    level = frame['level']
    group = _Group(level=level, kind=view.tile_kind if level == 'Weights' else view.head_stage,
                   weight_name=frame.get('weight_name'), shape=list(frame['shape']) if frame.get('shape') else None,
                   origin_cell=list(frame['origin']) if frame.get('origin') else [0, 0], pitch=MATRIX_PITCH, caption='')
    drawn = False
    if level == 'Weights':
        tile = np.asarray(frame.get(view.tile_kind, []), dtype=float)
        if tile.ndim == 2 and tile.size:
            _weights_tile(group, model, view, frame)
            drawn = True
    elif model.snapshot:
        drawn = _head_poster(group, model, view, frame)
    if not drawn:
        gate = model.gates[model.selection.layer]
        group.label(empty_matrix_note(level, bool(model.snapshot), model.selection.layer, gate['entry']['mixer']['id'],
                                      gate['glyphs']['mixer']['kind'] == 'beams'), (0., 0., 2.), kind='note')
        return group, 16
    group.meta['caption'] = thermal_legend(level)
    if model.show_numbers and level == 'Weights':
        tile = np.asarray(frame.get(view.tile_kind), dtype=float)
        cell = (min(view.cell[0], tile.shape[0] - 1), min(view.cell[1], tile.shape[1] - 1))
        group.meta['caption'] += f'\n{tile.shape[0]}x{tile.shape[1]} | range {tile.min():.3g} to {tile.max():.3g} | cell {cell[0]},{cell[1]} = {float(tile[cell]):.4g} (I J K L move)'
    return group, 16


def thermal_legend(level):
    ramp = 'colour = value, dark purple low to yellow high'
    if level == 'Head':
        return f'{ramp} | rows = queries, columns = keys | dark = masked (a later key)'
    return f'{ramp} | the white frame is the picked cell'


def thermal_camera(level, blocks, target, distance):
    """The camera for the tiles: the poster is about 30 wide, a weight tile 6, the Model level has the head stacks to the right."""
    if level == 'Head':
        return (-2.5, 0., 0.), 32.
    if level == 'Weights':
        return (0., 0., 0.), 11.
    return (target[0] + 4., target[1], target[2]), max(distance, 26.)


def head_variant(heads, shared):
    """MHA (every head has its own keys and values), MQA (all share one set) or GQA (groups of heads share a set)."""
    return next(name for test, name in VARIANTS if test(heads, shared))


def _head_stack(group, model, view, frame, gate):
    """A layer's heads at the Model level, beside its block: one tile per query head (that head's attention), and under them the
    key/value slabs, each as wide as the heads that share it (MHA: one per head; GQA: one per group; MQA: one for all)."""
    index = gate['index']
    config = gate['entry']['mixer']['config']
    if gate['glyphs']['mixer']['kind'] != 'beams' or int(config.get('heads', 0)) < 1:
        return
    heads = int(config['heads'])
    shared = int(config.get('kv_heads', heads))
    shared = shared if 1 <= shared <= heads and heads % shared == 0 else heads
    per = heads // shared  # heads sharing one key/value set
    shown = min(heads, STACK_HEADS)
    pitch = min(1., STACK_WIDTH / shown)
    arrays = model.snapshot['arrays'] if model.snapshot else {}
    pattern = frame['attention'].get(f'block{index}.attn.pattern')
    pattern = np.asarray(pattern) if pattern is not None else arrays.get(f'hook::block{index}.attn.pattern')
    pattern = pattern[0] if pattern is not None and pattern.ndim == 4 else pattern
    keys, values = arrays.get(f'hook::block{index}.attn.k'), arrays.get(f'hook::block{index}.attn.v')
    label_x = STACK_X + shown * pitch / 2
    for h in range(shown):
        x = STACK_X + (h + .5) * pitch
        view_of = pattern[h] if pattern is not None and h < len(pattern) else None
        group.tile(f'head:{h}', view_of, (x, 0., 2.3), (pitch * .86, pitch * .86), label=f'head {h + 1}', shape=view_of.shape if view_of is not None else None,
                   mask=np.arange(view_of.shape[1])[None, :] > np.arange(view_of.shape[0])[:, None] if view_of is not None else None,
                   selected=gate['index'] == model.selection.layer and h == model.selection.head, low=0.)
    for g in range(math.ceil(shown / per)):
        first, last = g * per, min(shown, (g + 1) * per)
        centre = STACK_X + (first + last) / 2 * pitch
        wide = (last - first) * pitch * .86
        for z, kind, source in ((1.15, 'k', keys), (.55, 'v', values)):
            part = source[0, g] if source is not None and g < source.shape[1] else None
            group.tile(f'{kind}:{g}', part, (centre, 0., z), (wide, .48), label=f'{kind.upper()} {g + 1}', shape=part.shape if part is not None else None)
        for h in range(first, last):
            x = STACK_X + (h + .5) * pitch
            group.arrow((x, 0., 2.3 - pitch * .43 - .04), (x, 0., 1.15 + .24 + .06), head=.12)
    variant = head_variant(heads, shared)
    group.label(f'{variant}: {heads} heads, {shared} key/value set' + ('s' if shared != 1 else ''), (label_x, 0., 3.35), kind='tile_title', size=.22)


def _info(model, frame):
    info = f"{model.spec['name']} | {frame['level']} | {frame['mode']} | block {model.selection.layer + 1}"
    if frame['step'] is not None:
        info += f" | step {frame['step']}"
    if frame['loss'] is not None:
        info += f" | loss {frame['loss']:.4f}"
    if model.mode == 'Ride a token' and model.snapshot:
        lens = model.snapshot.get('lens', {}).get(str(model.selection.layer))
        if lens:
            token = min(model.selection.token, len(lens['ids']) - 1)
            info += f" | predicted id {lens['ids'][token][0]} ({lens['probabilities'][token][0]:.3f})"
    return info


def _pulse(model, view, frame):
    """What the renderer needs to animate the replay pulse. Each frame: pose = `pulse_pose(blocks, elapsed, reduce_motion)`,
    position (token_x, -depth * spacing, z), colour `colors[forward or backward]`, edge length `sizes[direction][block]`
    (`idle_size` where that is null). Play moves to the next snapshot after `playback.pass_seconds` (the `advance` action)."""
    blocks = len(model.gates)
    pulse = {'animate': bool(view.playing and model.snapshot and frame['level'] in ('Model', 'Window')), 'blocks': blocks,
             'drawn': frame['level'] in ('Model', 'Window'), 'spacing': GATE_SPACING, 'speed': 2., 'z': 1., 'follow': model.mode == 'Ride a token', 'token_x': 0.,
             'colors': {'forward': [1., 1., 1., 1.], 'backward': [1., .45, .08, 1.]}, 'idle_size': [.3, .4, .3], 'sizes': {}}
    if frame['level'] == 'Window':
        pulse['token_x'] = (min(frame['end'] - 1, model.selection.token) - frame['start'] - (len(frame['tokens']) - 1) / 2) * LANE_PITCH
    if model.snapshot and frame['level'] in ('Model', 'Window'):
        for direction in ('forward', 'backward'):
            records = model.replay_values(direction)
            sizes = []
            for layer in range(blocks):
                values = [value for name, value in records if name.startswith(f'block{layer}.')]
                sizes.append(2 * (.1 + min(.4, max(float(np.linalg.norm(v) / max(1, v.size) ** .5) for v in values))) if values else None)
            pulse['sizes'][direction] = sizes
    return pulse


def _parts(model, view):
    """{cursor key: part}: every piece of the current scene that can change on its own."""
    frame = model.frame()
    level, selection = frame['level'], model.selection
    groups, rows = {}, 16
    if level in ('Model', 'Window'):
        groups = {f"groups/gate:{gate['index']}": _gate_group(model, frame, gate, view).encode() for gate in frame['gates']}
    elif level in ('Head', 'Weights'):
        group, rows = (_thermal_matrix_group if view.thermal else _matrix_group)(model, frame, view)
        groups = {'groups/matrix': group.encode()}
    names = block_matrices(weight_names(model), selection.layer)
    snap = snapshot_buttons(view.linked, view.probing)
    target, distance = camera_view(level, len(model.gates), rows)
    if view.thermal:
        target, distance = thermal_camera(level, len(model.gates), target, distance)
    return {
        'view': {'level': level, 'mode': model.mode, 'head_stage': view.head_stage, 'tile_kind': view.tile_kind,
                 'thermal': view.thermal, 'reduce_motion': model.reduce_motion, 'show_numbers': model.show_numbers, 'levels': list(VIEWER_LEVELS),
                 'modes': list(MODES), 'head_stages': list(HEAD_STAGES), 'tile_kinds': list(TILE_KINDS),
                 'controls': {'rows': view_rows(level, model.mode, names), 'matrices': names, 'snapshots': snap,
                              'toggles': toggle_names(level, view.snapshot_paths, view.auto_orbit)}},
        'camera': {'target': list(target), 'distance': distance, 'level': level},
        'selection': {**{key: getattr(selection, key) for key in ('run', 'layer', 'token', 'head', 'weight', 'row', 'column')},
                      'block_id': model.gates[selection.layer]['id'], 'cell': list(view.cell)},
        'stats': {'step': frame['step'], 'loss': frame['loss'], 'info': _info(model, frame), 'training': dict(view.stats),
                  'training_line': stats_line(view.stats), 'note': view.note},
        'design': {'name': model.spec['name'], 'd_model': model.spec['dims']['d_model'], 'n_layer': model.spec['dims']['n_layer'],
                   'n_ctx': model.spec['dims']['n_ctx'],
                   'gates': [{'index': g['index'], 'id': g['id'], 'label': g['entry'].get('label', f"Block {g['index'] + 1}"),
                              'mixer': g['entry']['mixer'], 'channel': g['entry']['channel'], 'glyphs': g['glyphs']}
                             for g in model.gates]},
        'playback': {**playback_state(len(view.snapshot_paths), view.snapshot_index, view.playing),
                     'count': len(view.snapshot_paths), 'index': view.snapshot_index, 'playing': view.playing,
                     'pass_seconds': pass_seconds(len(model.gates)), 'pulse': _pulse(model, view, frame)},
        **groups}


def _assemble(parts, keys, mode):
    out, groups = {}, {}
    for key in keys:
        (groups.__setitem__(key[len('groups/'):], _encode(parts[key], mode, key)) if key.startswith('groups/')
         else out.__setitem__(key, _encode(parts[key], mode, key)))
    return out, groups


def scene_payload(model: SceneModel, view: ViewState | None = None, arrays: str = 'inline') -> dict:
    """The whole scene, JSON-serializable. arrays='binary' leaves each array as {shape, name}, its data in `scene_binary`."""
    if arrays not in ('inline', 'binary'):
        raise ValueError("arrays is 'inline' or 'binary'.")
    parts = _parts(model, view or ViewState())
    out, groups = _assemble(parts, parts, arrays)
    return {'schema': SCHEMA, 'full': True, 'up': 'z', 'palette': copy.deepcopy(PALETTE), **out, 'groups': groups, 'removed_groups': [],
            'cursor': {key: _digest(part, key) for key, part in parts.items()}}


def _check_since(since):
    if since and not isinstance(since, dict):
        raise ValueError('since is the cursor of a previous payload or frame.')


def scene_frame(model: SceneModel, since: dict | None, view: ViewState | None = None, arrays: str = 'inline') -> dict:
    """What changed since `since`, the `cursor` of the client's last payload or frame (None or empty: the whole payload).
    Apply it by replacing each section it carries, replacing the groups it carries and dropping `removed_groups`.
    Each call reads the model once, so the caller serializes calls with whatever changes the model (and with
    `scene_binary` for the same frame: it must see the same state)."""
    _check_since(since)
    if not since:
        return scene_payload(model, view, arrays)
    parts = _parts(model, view or ViewState())
    cursor = {key: _digest(part, key) for key, part in parts.items()}
    out, groups = _assemble(parts, [key for key in parts if since.get(key) != cursor[key]], arrays)
    return {'schema': SCHEMA, 'full': False, **out, 'groups': groups, 'cursor': cursor,
            'removed_groups': [key[len('groups/'):] for key in since if key.startswith('groups/') and key not in parts]}


def scene_binary(model: SceneModel, view: ViewState | None = None, since: dict | None = None) -> tuple[bytes, dict]:
    """(little-endian float32 bytes, {name: {offset, shape, dtype}}) for every array of `scene_payload(arrays='binary')`, or
    of the `scene_frame(since=...)` for the same cursor. offset is in bytes; name is the one in the payload's {shape, name}."""
    _check_since(since)
    parts = _parts(model, view or ViewState())
    manifest, chunks, offset = {}, [], 0
    for key, part in parts.items():
        if since and since.get(key) == _digest(part, key):
            continue
        for name, array in _arrays(part, key):
            chunks.append(array.data.tobytes())
            manifest[name] = {'offset': offset, 'shape': list(array.data.shape), 'dtype': 'float32'}
            offset += array.data.nbytes
    return b''.join(chunks), manifest


# --- the viewer's actions as SceneModel / ViewState calls ---

def _load_snapshot(path):
    from studio_snapshot import load
    return load(path)


def set_level(model, level):
    if level not in VIEWER_LEVELS:
        raise ValueError(f'Choose a zoom level: {", ".join(VIEWER_LEVELS)}.')
    model.level = level


def set_mode(model, mode):
    if mode not in MODES:
        raise ValueError(f'Choose a mode: {", ".join(MODES)}.')
    model.mode = mode


def step_stage(view, delta=0, stage=None):
    """Walk the head's forward pass: delta +1 / -1, or jump to a named stage."""
    view.head_stage = next_stage(view.head_stage, delta, stage)


def show_tile(view, what):
    if what not in TILE_KINDS:
        raise ValueError(f'Choose a tile kind: {", ".join(TILE_KINDS)}.')
    view.tile_kind = what


def follow_block_matrix(model):
    chosen = pick_matrix(weight_names(model), model.selection.layer, model.selection.weight)
    if chosen != model.selection.weight:
        model.choose(weight=chosen)


def select(model, **values):
    """A selection chosen from outside (the studio's page, a click): applied, and the matrix follows the block."""
    model.choose(**values)
    follow_block_matrix(model)


def select_matrix(model, name):
    model.choose(weight=name)


def swap_mixer(model, view, module):
    """A token mixer picked for the selected block (Build mode). Linked: returns the `module_swap` message to forward to the
    studio, which saves a new version and answers with a `spec` action. A demo swaps here. None when nothing is to be sent."""
    layer = model.selection.layer
    if module == model.gates[layer]['entry']['mixer']['id']:
        return None
    message = {'type': 'module_swap', 'block_id': model.gates[layer]['id'], 'slot': 'mixer', 'entry': {'id': module}}
    if view.linked:
        view.note = f'Saving block {layer + 1} with {module} as a new version...'
        return message
    try:
        model.swap(layer, 'mixer', message['entry'])
    except ValueError as error:
        view.note = str(error)
    return None


def next_block(model):
    select(model, layer=(model.selection.layer + 1) % len(model.gates))


def select_token(model, delta):
    model.choose(token=max(0, min(model.spec['dims']['n_ctx'] - 1, model.selection.token + delta)))


def select_head(model, delta):
    heads = block_at(model.spec, model.selection.layer)['mixer']['config'].get('heads', 1)
    model.choose(head=(model.selection.head + delta) % heads)


def pan_tiles(model, row, column):
    model.choose(row=max(0, model.selection.row + row), column=max(0, model.selection.column + column))


def move_cell(model, view, row, column):
    tile = np.asarray(model.frame('Weights').get('weight', []))
    if tile.ndim == 2:
        view.cell = [max(0, min(tile.shape[0] - 1, view.cell[0] + row)), max(0, min(tile.shape[1] - 1, view.cell[1] + column))]


def toggle_motion(model):
    model.reduce_motion = not model.reduce_motion


def toggle_numbers(model):
    model.show_numbers = not model.show_numbers


def set_spec(model, spec, selected=None):
    """The studio sent a new design; `selected` is a block id to select. Returns the changed block indexes."""
    changed = model.update_spec(spec)
    ids = [gate['id'] for gate in model.gates]
    if selected in ids:
        model.choose(layer=ids.index(selected))
    return changed


def set_summary(model, message):
    """The studio's attention summary: {attention: {hook: [heads][bins][bins]}, stream_norms: {hook: [..]}, edges: [..]}."""
    summaries, norms = message.get('attention', model.summaries), message.get('stream_norms', model.stream_norms)
    edges = message.get('edges', model.summary_edges)
    try:
        for value in summaries.values():
            if np.asarray(value, dtype=float).ndim != 3:
                raise ValueError
        for value in norms.values():
            if np.asarray(value, dtype=float).ndim != 1:
                raise ValueError
        if edges is not None and np.asarray(edges, dtype=float).ndim != 1:
            raise ValueError
    except (AttributeError, TypeError, ValueError):
        raise ValueError('A summary message carries attention (hook -> heads x bins x bins), stream_norms (hook -> bins) and edges (a list).') from None
    model.paths, model.summaries, model.stream_norms, model.summary_edges = message.get('paths', model.paths), summaries, norms, edges


def set_stats(view, message):
    """The training line: step, total, loss, lr and tokens_per_second, each a number or None."""
    stats = {key: value for key, value in message.items() if key != 'type'}
    if any(value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))) for value in stats.values()):
        raise ValueError('Stats values are numbers.')
    view.stats = stats


def read_snapshot(model, path, load=None):
    model.load_snapshot((load or _load_snapshot)(path))
    follow_block_matrix(model)


def load_replay(model, view, paths, load=None):
    """The studio's list of this version's snapshots: show the first."""
    view.snapshot_paths, view.snapshot_index, view.playing = list(paths), -1, False
    view.note = (f'Loaded {len(view.snapshot_paths)} snapshots. Play them, or step with Previous and Next.'
                 if view.snapshot_paths else 'This version has no training snapshots yet.')
    scrub(model, view, 1, load)


def _why(error):
    return getattr(error, 'strerror', None) or str(error)  # an OSError's own text leaves out the file's path


def scrub_to(model, view, index, load=None):
    if view.snapshot_paths:
        chosen = clamp_index(index, len(view.snapshot_paths))
        try:
            read_snapshot(model, view.snapshot_paths[chosen], load)
        except Exception as error:  # noqa: BLE001 - the position stays on what is shown, and the note says why
            view.note, view.playing = f'Could not load the snapshot: {_why(error)}', False
            return
        view.snapshot_index = chosen


def scrub(model, view, delta, load=None):
    scrub_to(model, view, view.snapshot_index + delta, load)


def add_snapshot(model, view, path, load=None, probed=False):
    """A new snapshot arrived (training saved one, or `probed`: the answer to Run one real step). It becomes the current one.
    Paths of pruned files are forgotten so scrubbing never opens a missing file; a bad file is said in the note."""
    if probed:
        view.probing, view.note = False, 'Ran one real step. Press Play to watch it pass through the blocks.'
    try:
        read_snapshot(model, path, load)
        view.snapshot_paths = [p for p in view.snapshot_paths if p != path and os.path.exists(p)] + [path]
        view.snapshot_index = len(view.snapshot_paths) - 1
    except Exception as error:  # noqa: BLE001 - a truncated or foreign file is reported, and the scene carries on
        view.note = f'Could not load the snapshot: {_why(error)}'


def toggle_play(model, view, load=None):
    if not view.snapshot_paths:
        view.note = 'Nothing to play yet: load snapshots first.'
        return
    if not view.playing and view.snapshot_index + 1 >= len(view.snapshot_paths):
        scrub_to(model, view, 0, load)  # at the end, Play starts over
    view.playing = not view.playing


def advance_playback(model, view, load=None):
    """The pulse went forward and back (`playback.pass_seconds` later): the next snapshot, or stop after the last.
    Does nothing once paused, so a late or repeated request never skips a snapshot."""
    if not view.playing:
        return
    if view.snapshot_index + 1 < len(view.snapshot_paths):
        scrub(model, view, 1, load)
    else:
        view.playing = False


def begin_probe(view):
    """Run one real step was pressed. False when one is already running or there is no studio to run it; otherwise True: the
    caller forwards the request, and the studio answers with add_snapshot(probed=True) or `note_reply`."""
    if view.probing or not view.linked:
        return False
    view.probing, view.note = True, "Running one real training step on this version's checkpoint..."
    return True


def note_reply(view, kind, message):
    """The studio's 'notice' (the step still runs) or 'error' (it stopped) for a button here."""
    view.probing = view.probing and kind == 'notice'
    view.note = message


def _scene_message(model, view, a, load=None):
    if a.get('selected'):
        select(model, **a['selected'])
    if a.get('mode') in MODES:
        model.mode = a['mode']
    if a.get('level') in VIEWER_LEVELS:
        model.level = a['level']


# action type -> handler(model, view, action, load); the types are the viewer's own messages plus its keys and menus
ACTIONS = {
    'scene': _scene_message,
    'level': lambda m, v, a, load: set_level(m, a['value']),
    'mode': lambda m, v, a, load: set_mode(m, a['value']),
    'selection': lambda m, v, a, load: select(m, **a['selection']),
    'step': lambda m, v, a, load: step_stage(v, a.get('delta', 0), a.get('stage')),
    'show': lambda m, v, a, load: show_tile(v, a['what']),
    'matrix': lambda m, v, a, load: select_matrix(m, a['name']),
    'swap': lambda m, v, a, load: swap_mixer(m, v, a['module']),
    'next_block': lambda m, v, a, load: next_block(m),
    'token': lambda m, v, a, load: select_token(m, a['delta']),
    'head': lambda m, v, a, load: select_head(m, a['delta']),
    'pan': lambda m, v, a, load: pan_tiles(m, a.get('row', 0), a.get('column', 0)),
    'cell': lambda m, v, a, load: move_cell(m, v, a.get('row', 0), a.get('column', 0)),
    'motion': lambda m, v, a, load: toggle_motion(m),
    'numbers': lambda m, v, a, load: toggle_numbers(m),
    'play': lambda m, v, a, load: toggle_play(m, v, load),
    'advance': lambda m, v, a, load: advance_playback(m, v, load),
    'scrub': lambda m, v, a, load: scrub_to(m, v, a['index'], load) if 'index' in a else scrub(m, v, a['delta'], load),
    'replay': lambda m, v, a, load: load_replay(m, v, a.get('paths', []), load),
    'viz_snapshot': lambda m, v, a, load: add_snapshot(m, v, a['path'], load),
    'snapshot': lambda m, v, a, load: add_snapshot(m, v, a['path'], load, probed=True),
    'probe_step': lambda m, v, a, load: begin_probe(v),
    'notice': lambda m, v, a, load: note_reply(v, 'notice', a.get('message', '')),
    'error': lambda m, v, a, load: note_reply(v, 'error', a.get('message', '')),
    'stats': lambda m, v, a, load: set_stats(v, a),
    'summary': lambda m, v, a, load: set_summary(m, a),
    'spec': lambda m, v, a, load: set_spec(m, a['spec'], a.get('selected')),
}


def apply_action(model, view, action, load=None):
    """Do what a viewer message or control did, by its `type` (see ACTIONS). Returns the handler's answer: for the two that
    ask the studio, what to forward: `probe_step` True (False: not now), `swap` the `module_swap` message (or None).
    ValueError for an unknown type or a bad value, and nothing else escapes for bad client input."""
    handler = ACTIONS.get(action.get('type')) if isinstance(action, dict) else None
    if handler is None:
        raise ValueError(f"Unknown viewer action {action.get('type') if isinstance(action, dict) else action!r}.")
    try:
        return handler(model, view, action, load)
    except KeyError as error:
        raise ValueError(f'The {action["type"]} action needs {error}.') from None
    except (TypeError, AttributeError, IndexError):  # a value of the wrong kind from a client
        raise ValueError(f'The {action["type"]} action has a value of the wrong kind.') from None
