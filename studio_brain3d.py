"""Panda3D renderer for bounded network scenes, hosted in its own process."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from panda3d.core import (Filename, Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
                          GeomVertexWriter, LineSegs, TextNode, TransparencyAttrib, Vec3,
                          loadPrcFileData)

from studio_scene import LEVELS, MODES, SceneModel
from studio_spec import block_at, default_spec, to_legacy_view, upgrade


def sample_spec(n=6):
    return {'format': 'groundwork-arch/1', 'name': 'Demo model',
            'embedding': {'tokenizer': 'byte', 'width': 64, 'context': 256, 'dropout': .1},
            'blocks': [{'id': f'blk_{i:06x}', 'label': f'Block {i + 1}',
                        'attention': {'enabled': True, 'heads': 8},
                        'mlp': {'enabled': True, 'expansion': 4, 'layers': 1, 'activation': 'gelu'}} for i in range(n)],
            'training': {}}


def normalize_spec(spec):
    return to_legacy_view(upgrade(spec))


def valid_heads(width, current, direction):
    values = [h for h in range(1, 17) if width % h == 0]
    candidates = [h for h in values if h > current] if direction > 0 else [h for h in values if h < current]
    return (min(candidates) if direction > 0 else max(candidates)) if candidates else None


def synth_telemetry(spec, step, total, t):
    """Legacy demo fixture; real visualization is supplied by snapshots."""
    return {'step': step, 'total': total, 'loss': 2 + math.exp(-t),
            'blocks': [{'id': b['id'], 'index': i, 'residual_rms': 1., 'attention': None, 'mlp': None}
                       for i, b in enumerate(normalize_spec(spec)['blocks'])]}


def _cube():
    data = GeomVertexData('cube', GeomVertexFormat.getV3n3(), Geom.UHStatic)
    vertex, normal = GeomVertexWriter(data, 'vertex'), GeomVertexWriter(data, 'normal')
    faces = [((1, 0, 0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
             ((-1, 0, 0), [(-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)]),
             ((0, 1, 0), [(-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)]),
             ((0, -1, 0), [(1, -1, -1), (-1, -1, -1), (-1, -1, 1), (1, -1, 1)]),
             ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
             ((0, 0, -1), [(-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)])]
    primitive = GeomTriangles(Geom.UHStatic)
    for face, (direction, vertices) in enumerate(faces):
        for point in vertices:
            vertex.addData3(*point)
            normal.addData3(*direction)
        i = face * 4
        primitive.addVertices(i, i + 1, i + 2)
        primitive.addVertices(i, i + 2, i + 3)
    geom = Geom(data)
    geom.addPrimitive(primitive)
    node = GeomNode('cube')
    node.addGeom(geom)
    from panda3d.core import NodePath
    return NodePath(node)


class BrainApp:
    def __new__(cls, *args, **kwargs):
        from direct.showbase.ShowBase import ShowBase
        class Application(_SceneRenderer, ShowBase):
            def __init__(self, link=None, demo=True):
                ShowBase.__init__(self)
                self.initialize_scene(link, demo)
        return Application(*args, **kwargs)


class _SceneRenderer:
    def initialize_scene(self, link, demo):
        from direct.gui.DirectGui import DirectButton, DirectCheckButton, DirectOptionMenu, DirectLabel
        self.link, self.demo = link, demo
        self.disableMouse()
        self.setBackgroundColor(.025, .04, .065)
        self.scene = SceneModel(sample_spec(6) if demo else default_spec())
        self.ids, self.sel_id = [], None
        self.cam_h, self.cam_p, self.cam_d = 25., 30., 28.
        self.cam_t = self.render.attachNewNode('target')
        self.cam_t.setPos(0, -8, 1)
        self.auto_orbit, self.playing = demo, False
        self.last_mouse, self.dragging = None, False
        self.keys = set()
        self.geometry = self.render.attachNewNode('scene')
        self.cube = _cube()
        self.gate_nodes = {}
        self.weight_kind, self.head_stage = 'pattern', 'pattern'
        self.snapshot_paths, self.snapshot_index = [], -1
        self.animation_elapsed = 0.
        self.hud = DirectLabel(text='', scale=.043, pos=(-.05, 0, .86), frameColor=(0, 0, 0, 0), text_fg=(.85, .92, 1, 1))
        self.level_menu = DirectOptionMenu(items=list(LEVELS), initialitem=1, scale=.04, pos=(-1.28, 0, .9), command=self.set_level)
        self.mode_menu = DirectOptionMenu(items=list(MODES), initialitem=0, scale=.04, pos=(-1.28, 0, .77), command=self.set_mode)
        DirectCheckButton(text='Reduce motion', scale=.04, pos=(-1.12, 0, -.8), command=self.motion)
        DirectCheckButton(text='Show numbers', indicatorValue=True, scale=.04, pos=(-.5, 0, -.8), command=self.numbers)
        DirectButton(text='Play / pause', scale=.04, pos=(.3, 0, -.8), command=self.toggle_play)
        DirectButton(text='Previous', scale=.04, pos=(.75, 0, -.8), command=lambda: self.scrub(-1))
        DirectButton(text='Next', scale=.04, pos=(1.08, 0, -.8), command=lambda: self.scrub(1))
        DirectButton(text='Load replay', scale=.04, pos=(1.08, 0, .84), command=lambda: self.send({'type': 'load_replay'}))
        DirectButton(text='1 real step', scale=.04, pos=(1.08, 0, .7), command=lambda: self.send({'type': 'probe_step'}))
        self.tile_menu = DirectOptionMenu(items=['weight', 'gradient', 'update'], scale=.04, pos=(-1.28, 0, .64), command=self.set_weight_kind)
        self.step_menu = DirectOptionMenu(items=['scores', 'scale', 'mask', 'pattern', 'values'], initialitem=3,
                                          scale=.04, pos=(-1.28, 0, .51), command=self.set_head_stage)
        import studio_registry as registry
        self.module_menu = DirectOptionMenu(items=[m.short for m in registry.for_slot('mixer')], scale=.04,
                                            pos=(-1.28, 0, .38), command=self.swap_mixer)
        self.matrix_menu = DirectOptionMenu(items=['Choose matrix'], scale=.035, pos=(-1.28, 0, .25), command=self.select_matrix)
        for key in 'wasdqe':
            self.accept(key, self.key, [key, True])
            self.accept(key + '-up', self.key, [key, False])
        self.accept('mouse1', self.drag, [True])
        self.accept('mouse1-up', self.drag, [False])
        self.accept('wheel_up', self.zoom, [.8])
        self.accept('wheel_down', self.zoom, [1.25])
        self.accept('tab', self.next_block)
        self.accept('r', self.set_mode, ['Ride a token'])
        self.accept('b', self.set_mode, ['Build'])
        self.accept('space', self.toggle_play)
        self.accept('escape', self.set_level, ['Model'])
        self.accept('arrow_left', self.scrub, [-1])
        self.accept('arrow_right', self.scrub, [1])
        self.accept('arrow_up', self.select_token, [1])
        self.accept('arrow_down', self.select_token, [-1])
        self.accept('page_up', self.select_head, [1])
        self.accept('page_down', self.select_head, [-1])
        self.accept('[', self.pan_tiles, [-16, 0])
        self.accept(']', self.pan_tiles, [16, 0])
        self.accept(',', self.pan_tiles, [0, -16])
        self.accept('.', self.pan_tiles, [0, 16])
        self.accept('-', self.edit_weight, [-.01])
        self.accept('+', self.edit_weight, [.01])
        self.accept('window-event', self.window_event)
        self.set_spec(self.scene.spec, fit=True)
        self.taskMgr.add(self.tick, 'scene-events')

    def window_event(self, window):
        if window and not window.getProperties().getOpen():
            self.close()

    def send(self, message):
        if self.link:
            self.link.send(message)

    def close(self):
        self.send({'type': 'closed'})
        self.userExit()

    def key(self, key, pressed):
        self.keys.add(key) if pressed else self.keys.discard(key)

    def drag(self, value):
        self.dragging, self.last_mouse = value, None

    def motion(self, value):
        self.scene.reduce_motion = bool(value)

    def numbers(self, value):
        self.scene.show_numbers = bool(value)
        self.rebuild()

    def set_level(self, value):
        self.scene.level = value
        self.rebuild()

    def set_mode(self, value):
        self.scene.mode = value
        self.rebuild()

    def set_weight_kind(self, value):
        self.weight_kind = value
        if hasattr(self, 'geometry'):
            self.rebuild()

    def set_head_stage(self, value):
        self.head_stage = value
        if hasattr(self, 'geometry'):
            self.rebuild()

    def select_matrix(self, value):
        if value == 'Choose matrix' or not hasattr(self, 'scene'):
            return
        self.scene.choose(weight=value)
        self.rebuild()

    def zoom(self, factor):
        self.cam_d = max(2., min(160., self.cam_d * factor))
        level = 'Run' if self.cam_d > 70 else 'Model' if self.cam_d > 22 else 'Window' if self.cam_d > 10 else 'Head' if self.cam_d > 4 else 'Weights'
        if self.scene.level != level:
            self.set_level(level)

    def next_block(self):
        self.scene.choose(layer=(self.scene.selection.layer + 1) % len(self.scene.gates))
        self.sel_id = self.ids[self.scene.selection.layer]
        self.send({'type': 'select', 'block_id': self.sel_id})
        self.rebuild()

    def select_token(self, delta):
        self.scene.choose(token=max(0, min(self.scene.spec['dims']['n_ctx'] - 1, self.scene.selection.token + delta)))
        self.send({'type': 'selection', 'selection': {'token': self.scene.selection.token}})
        self.rebuild()

    def select_head(self, delta):
        heads = self._selected_block()['mixer']['config'].get('heads', 1)
        self.scene.choose(head=(self.scene.selection.head + delta) % heads)
        self.send({'type': 'selection', 'selection': {'head': self.scene.selection.head}})
        self.rebuild()

    def set_spec(self, spec, selected=None, keep_selection=True, fit=False):
        count = len(self.ids)
        changed = self.scene.update_spec(spec)
        self.ids = [gate['id'] for gate in self.scene.gates]
        if selected in self.ids:
            self.scene.choose(layer=self.ids.index(selected))
            self.sel_id = selected
        elif not keep_selection or self.sel_id not in self.ids:
            self.sel_id = self.ids[self.scene.selection.layer]
        if fit or count != len(self.ids):
            self.cam_t.setY(-max(1, len(self.ids) - 1) * 1.5)
            self.cam_d = max(22., len(self.ids) * 3.5)
        self.rebuild(changed=changed)

    def _selected_block(self):
        return block_at(self.scene.spec, self.scene.selection.layer)

    def cmd_step(self, name, direction):
        block = self._selected_block()
        heads = block['mixer']['config'].get('heads', 1)
        value = valid_heads(self.scene.spec['dims']['d_model'], heads, direction) if name == 'heads' else None
        if value is None:
            return
        config = dict(block['mixer']['config'], heads=value)
        self.send({'type': 'module_swap', 'block': self.scene.selection.layer, 'block_id': block['id'],
                   'slot': 'mixer', 'entry': {'id': block['mixer']['id'], 'config': config}})
        if self.demo:
            self.scene.swap(self.scene.selection.layer, 'mixer', {'id': block['mixer']['id'], 'config': config})
            self.rebuild()

    def cmd_toggle(self, part):
        block = self._selected_block()
        slot = 'channel' if part == 'mlp' else 'mixer'
        enabled = block[slot]['id'] == 'none'
        entry = ({'id': 'mlp_gelu'} if part == 'mlp' else {'id': 'attention', 'config': {'heads': 4}}) if enabled else {'id': 'none'}
        self.send({'type': 'module_swap', 'block': self.scene.selection.layer, 'block_id': block['id'],
                   'slot': slot, 'entry': entry})

    def cmd_add(self):
        self.send({'type': 'add_block', 'after': self.sel_id})

    def cmd_remove(self):
        self.send({'type': 'remove_block', 'block_id': self.sel_id})

    def swap_mixer(self, module):
        if not hasattr(self, 'ids') or not self.ids:
            return
        bid = self.ids[self.scene.selection.layer]
        self.send({'type': 'module_swap', 'block': self.scene.selection.layer, 'block_id': bid,
                   'slot': 'mixer', 'entry': {'id': module}})
        if self.demo:
            try:
                changed = self.scene.swap(self.scene.selection.layer, 'mixer', {'id': module})
                self.rebuild(changed)
            except ValueError as error:
                self.hud['text'] = str(error)

    def pan_tiles(self, row, column):
        self.scene.choose(row=max(0, self.scene.selection.row + row), column=max(0, self.scene.selection.column + column))
        self.rebuild()

    def edit_weight(self, delta):
        self.send({'type': 'edit_weight', 'weight': self.scene.selection.weight,
                   'row': self.scene.selection.row, 'column': self.scene.selection.column, 'delta': delta})

    def toggle_play(self):
        self.playing = not self.playing

    def scrub(self, delta):
        if not self.snapshot_paths:
            return
        self.snapshot_index = max(0, min(len(self.snapshot_paths) - 1, self.snapshot_index + delta))
        self.read_snapshot(self.snapshot_paths[self.snapshot_index])

    def read_snapshot(self, path):
        from studio_snapshot import load
        self.scene.load_snapshot(load(path))
        self.animation_elapsed = 0.
        names = self.scene.frame('Weights')['available']
        if names:
            self.matrix_menu['items'] = names
            self.matrix_menu.set(0)
            self.scene.choose(weight=names[0])
        self.rebuild()

    def cube_at(self, parent, position, scale, color):
        node = parent.attachNewNode('instance')
        self.cube.instanceTo(node)
        node.setPos(*position)
        node.setScale(*scale)
        node.setColor(*color)
        node.setTransparency(TransparencyAttrib.MAlpha)
        return node

    def label_at(self, parent, text, position, size=.28):
        node = TextNode('label')
        node.setText(str(text))
        node.setTextColor(.88, .92, 1, 1)
        node.setAlign(TextNode.ACenter)
        path = parent.attachNewNode(node)
        path.setPos(*position)
        path.setScale(size)
        path.setBillboardPointEye()
        return path

    def lines(self, parent, points, color=(.3, .8, .9, .8), thickness=1.):
        if len(points) < 2:
            return
        lines = LineSegs()
        lines.setColor(*color)
        lines.setThickness(thickness)
        lines.moveTo(*points[0])
        for point in points[1:]:
            lines.drawTo(*point)
        return parent.attachNewNode(lines.create())

    def draw_gate(self, gate, frame):
        index = gate['index']
        root = self.geometry.attachNewNode(f'gate-{index}')
        root.setY(-index * 3)
        self.gate_nodes[index] = root
        selected = index == self.scene.selection.layer
        self.cube_at(root, (0, 0, 1.5), (6, .18, 1.5), (.12, .28, .38, .65 if selected else .3))
        if self.scene.snapshot:
            ratios = [value for key, value in self.scene.snapshot.get('update_ratios', {}).items() if key.startswith(f'block{index}.')]
            brightness = min(1., max(ratios, default=0.) * 20)
            self.cube_at(root, (0, 0, .05), (6, .21, .04), (brightness, .25 + brightness * .5, .3, .8))
        title = f"{index + 1} · {gate['entry']['mixer']['id']} / {gate['entry']['channel']['id']}"
        self.label_at(root, title, (0, 0, 3.5), .35)
        if frame['level'] == 'Model':
            bins = len(frame['edges']) - 1
            for bin_index in range(bins):
                self.cube_at(root, (-5.5 + 11 * bin_index / max(1, bins - 1), 0, 1.5), (.09, .3, .4), (.3, .75, .85, .65))
            for hook, mass in frame['attention'].items():
                if not hook.startswith(f'block{index}.'):
                    continue
                mean = np.asarray(mass).mean(axis=0)
                for query in range(mean.shape[0]):
                    for key in np.argsort(mean[query])[-4:]:
                        weight = mean[query, key]
                        if weight < .05:
                            continue
                        x1, x2 = -5.5 + 11 * query / max(1, bins - 1), -5.5 + 11 * key / max(1, bins - 1)
                        self.lines(root, [(x1, 0, 2), ((x1 + x2) / 2, 0, 2.5 + abs(x1 - x2) / 3), (x2, 0, 2)], thickness=1 + float(weight) * 3)
            return
        count = len(frame['tokens'])
        pattern = frame['patterns'].get(f'block{index}.attn.pattern')
        pattern = np.asarray(pattern)[min(self.scene.selection.head, len(pattern) - 1)] if pattern is not None else None
        norms = frame['streams'].get(f'block{index}.resid.post', [])
        glyph = gate['glyphs']['mixer']['kind']
        for lane in range(count):
            x = (lane - (count - 1) / 2) * .38
            self.lines(root, [(x, 1.5, .1), (x, -1.5, .1)], color=(.4, .6, .7, .6))
            magnitude = min(2., norms[lane]) if norms else .25
            color = (1., .7, .25, 1) if lane + frame['start'] == self.scene.selection.token else (.25, .75, .85, .8)
            self.cube_at(root, (x, 0, magnitude / 2), (.08, .14, max(.05, magnitude / 2)), color)
            if glyph == 'beams' and pattern is not None:
                for key in np.argsort(pattern[lane])[-4:]:
                    weight = pattern[lane, key]
                    if key > lane or weight < .05:
                        continue
                    key_x = (key - (count - 1) / 2) * .38
                    self.lines(root, [(x, 0, .4), ((x + key_x) / 2, .5, 1), (key_x, 1, .4)], thickness=1 + float(weight) * 4)
            elif glyph == 'ribbon':
                self.lines(root, [(x, 1.3, .6), (x, -1.3, .6)], thickness=3)
            elif glyph == 'fan':
                kernel = gate['entry']['mixer']['config'].get('kernel_size', 4)
                for offset in range(min(kernel, lane + 1)):
                    self.lines(root, [(x, 0, .6), (x - offset * .38, 1, .1)])
            if gate['entry']['channel']['id'] != 'none':
                self.cube_at(root, (x, -.7, 1), (.06, .1, .4), (.6, .45, .9, .7))
            if self.scene.show_numbers and index == 0:
                self.label_at(root, frame['tokens'][lane], (x, 1.5, .2), .14)

    def rebuild(self, changed=None):
        if not hasattr(self, 'hud'):
            return
        frame = self.scene.frame()
        incremental = changed is not None and frame['level'] in ('Model', 'Window') and getattr(self, 'drawn_level', None) == frame['level']
        if incremental:
            for index in changed:
                if index in self.gate_nodes:
                    self.gate_nodes.pop(index).removeNode()
            for gate in frame['gates']:
                if gate['index'] in changed:
                    self.draw_gate(gate, frame)
        else:
            self.geometry.removeNode()
            self.geometry = self.render.attachNewNode('scene')
            self.gate_nodes = {}
            self.pulse = None
            level = frame['level']
            if level in ('Model', 'Window'):
                for gate in frame['gates']:
                    self.draw_gate(gate, frame)
                self.pulse = self.cube_at(self.geometry, (0, 0, 1), (.15, .2, .15), (1, 1, 1, 1))
            elif level == 'Run':
                for index, path in enumerate(frame['paths']):
                    points = path['points']
                    if points:
                        max_step = max(1, max(point[0] for point in points))
                        self.lines(self.geometry, [(index * 3, -step / max_step * 30, loss * 3) for step, loss in points], thickness=2)
                        self.label_at(self.geometry, Path(path['run']).name, (index * 3, 1, 1))
                        for marker in path.get('markers', [])[:2000]:
                            y = -marker['step'] / max_step * 30
                            self.cube_at(self.geometry, (index * 3, y, 1), (.12, .12, .12), (1, .65, .2, 1))
                            self.label_at(self.geometry, marker['kind'], (index * 3, y, 1.4), .16)
                if not frame['paths']:
                    self.label_at(self.geometry, 'Load a run to inspect its measured loss path.', (0, 0, 2))
            elif level in ('Head', 'Weights'):
                matrix = np.asarray(frame.get(self.weight_kind if level == 'Weights' else self.head_stage, []))
                if level == 'Head' and self.head_stage in ('scale', 'mask'):
                    matrix = np.asarray(frame['scores'])
                    if matrix.size:
                        heads = self._selected_block()['mixer']['config'].get('heads', 1)
                        matrix = matrix / math.sqrt(self.scene.spec['dims']['d_model'] / heads)
                        if self.head_stage == 'mask':
                            matrix = np.where(frame['mask'], 0, matrix)
                if matrix.size:
                    maximum = max(float(np.max(np.abs(matrix))), 1e-12)
                    for row in range(matrix.shape[0]):
                        for column in range(matrix.shape[1]):
                            value = float(matrix[row, column])
                            x, y = (column - matrix.shape[1] / 2) * .4, -row * .4
                            masked = level == 'Head' and self.head_stage != 'values' and frame['mask'][row][column]
                            color = (.08, .09, .12, 1) if masked else (1, .6, .2, .9) if value > 0 else (.2, .6, 1, .9)
                            height = .015 if masked else .03 + abs(value) / maximum * 1.5
                            self.cube_at(self.geometry, (x, y, height / 2), (.18, .18, height / 2), color)
                    if self.scene.show_numbers:
                        self.label_at(self.geometry, f'{matrix.shape[0]}×{matrix.shape[1]} · range {matrix.min():.3g} … {matrix.max():.3g}', (0, 1, 1))
                else:
                    self.label_at(self.geometry, 'Load a snapshot with the selected hooks or matrix.', (0, 0, 2))
            self.drawn_level = level
        info = f"{self.scene.spec['name']} · {frame['level']} · {frame['mode']} · block {self.scene.selection.layer + 1}"
        if frame['step'] is not None:
            info += f" · step {frame['step']}"
        if frame['loss'] is not None:
            info += f" · loss {frame['loss']:.4f}"
        if self.scene.mode == 'Ride a token' and self.scene.snapshot:
            lens = self.scene.snapshot.get('lens', {}).get(str(self.scene.selection.layer))
            if lens:
                token = min(self.scene.selection.token, len(lens['ids']) - 1)
                info += f" · predicted id {lens['ids'][token][0]} ({lens['probabilities'][token][0]:.3f})"
        self.hud['text'] = info

    def tick(self, task):
        from panda3d.core import ClockObject
        dt = min(.1, ClockObject.getGlobalClock().getDt())
        if self.playing and getattr(self, 'pulse', None) is not None and self.scene.snapshot:
            self.animation_elapsed += dt
            count = len(self.scene.gates)
            phase = (self.animation_elapsed * 2) % (2 * count)
            forward = phase < count
            layer = min(count - 1, int(phase if forward else 2 * count - phase - 1e-6))
            depth = layer if self.scene.reduce_motion else (phase if forward else 2 * count - phase)
            token_x = 0.
            frame = self.scene.frame('Window')
            if self.scene.level == 'Window':
                token_x = (min(frame['end'] - 1, self.scene.selection.token) - frame['start'] - (len(frame['tokens']) - 1) / 2) * .38
            self.pulse.setPos(token_x, -depth * 3, 1)
            self.pulse.setColor(1, 1 if forward else .45, 1 if forward else .08, 1)
            records = self.scene.replay_values('forward' if forward else 'backward')
            values = [value for name, value in records if name.startswith(f'block{layer}.')]
            if values:
                magnitude = max(float(np.linalg.norm(value) / max(1, value.size) ** .5) for value in values)
                self.pulse.setScale(.1 + min(.4, magnitude))
            if self.scene.mode == 'Ride a token':
                self.cam_t.setPos(token_x, -depth * 3, 1)
        if self.link:
            for message in self.link.poll():
                kind = message.get('type')
                if kind == 'quit':
                    self.close()
                elif kind == 'spec':
                    self.set_spec(message['spec'], message.get('selected'))
                elif kind in ('viz_snapshot', 'snapshot'):
                    try:
                        self.read_snapshot(message['path'])
                        if message['path'] not in self.snapshot_paths:
                            self.snapshot_paths.append(message['path'])
                        self.snapshot_index = len(self.snapshot_paths) - 1
                    except (OSError, ValueError, KeyError) as error:
                        self.hud['text'] = str(error)
                elif kind == 'replay':
                    self.snapshot_paths = message.get('paths', [])
                    self.snapshot_index = -1
                    self.scrub(1)
                elif kind == 'summary':
                    self.scene.paths = message.get('paths', self.scene.paths)
                    self.scene.summaries = message.get('attention', self.scene.summaries)
                    self.scene.stream_norms = message.get('stream_norms', self.scene.stream_norms)
                    self.scene.summary_edges = message.get('edges', self.scene.summary_edges)
                    self.rebuild()
                elif kind == 'scene':
                    self.scene.level = message.get('level', self.scene.level)
                    self.scene.mode = message.get('mode', self.scene.mode)
                    if message.get('selected'):
                        self.scene.choose(**message['selected'])
                    self.rebuild()
                elif kind == 'selection':
                    self.scene.choose(**message['selection'])
                    self.rebuild()
                elif kind == 'error':
                    self.hud['text'] = message.get('message', 'Viewer operation failed.')
        if self.dragging and self.mouseWatcherNode.hasMouse():
            mouse = self.mouseWatcherNode.getMouse()
            if self.last_mouse:
                self.cam_h -= (mouse.x - self.last_mouse[0]) * 120
                self.cam_p = max(-80, min(80, self.cam_p + (mouse.y - self.last_mouse[1]) * 80))
            self.last_mouse = (mouse.x, mouse.y)
        if self.auto_orbit and not self.scene.reduce_motion:
            self.cam_h += dt * 5
        self.cam_t.setX(self.cam_t.getX() + (('d' in self.keys) - ('a' in self.keys)) * dt * 5)
        self.cam_t.setY(self.cam_t.getY() + (('w' in self.keys) - ('s' in self.keys)) * dt * 5)
        self.cam_t.setZ(self.cam_t.getZ() + (('e' in self.keys) - ('q' in self.keys)) * dt * 5)
        h, p = math.radians(self.cam_h), math.radians(self.cam_p)
        target = self.cam_t.getPos()
        self.camera.setPos(target + Vec3(self.cam_d * math.sin(h) * math.cos(p), self.cam_d * math.cos(h) * math.cos(p), self.cam_d * math.sin(p)))
        self.camera.lookAt(target)
        if self.playing and task.time - getattr(self, 'last_play', 0.) > 1:
            self.last_play = task.time
            if self.snapshot_index + 1 < len(self.snapshot_paths):
                self.scrub(1)
            else:
                self.playing = False
        return task.cont


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int)
    parser.add_argument('--authkey')
    parser.add_argument('--offscreen', action='store_true')
    parser.add_argument('--selftest', type=int)
    parser.add_argument('--screenshot')
    args = parser.parse_args(argv)
    loadPrcFileData('', 'audio-library-name null\nsync-video 0\nwin-size 1280 900')
    if args.offscreen or args.selftest:
        loadPrcFileData('', 'window-type offscreen\nframebuffer-hardware 0')
    link = None
    if args.port and args.authkey:
        from studio_vizbridge import ViewerLink
        link = ViewerLink(args.port, args.authkey)
    app = BrainApp(link=link, demo=link is None)
    if args.selftest:
        app.set_spec(sample_spec(24), fit=True)
        started = time.monotonic()
        for _ in range(args.selftest):
            app.taskMgr.step()
        if args.screenshot:
            app.win.saveScreenshot(Filename.fromOsSpecific(str(Path(args.screenshot).resolve())))
        print(f'fps(24 blocks): {args.selftest / max(1e-6, time.monotonic() - started):.1f}')
        app.destroy()
    else:
        app.run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
