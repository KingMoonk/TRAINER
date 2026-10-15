"""Panda3D renderer for bounded network scenes, hosted in its own process."""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch  # noqa: F401 - before panda3d: on Windows torch's DLLs fail to initialise (WinError 1114) once panda3d's are loaded
from panda3d.core import (Filename, Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
                          GeomVertexWriter, LineSegs, TextNode, TransparencyAttrib, Vec3,
                          loadPrcFileData)

import studio_viewer_ui as ui
from studio_scene import HEAD_STAGES, MODES, TILE_KINDS, SceneModel, clamp_index, next_stage, stats_line
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
        self.link, self.demo = link, demo
        self.disableMouse()
        self.setBackgroundColor(*ui.rgba('bg'))
        self.scene = SceneModel(sample_spec(6) if demo else default_spec())
        self.scene.level = 'Model'
        self.ids, self.sel_id = [], None
        self.cam_h, self.cam_p, self.cam_d = 25., 30., 28.
        self.cam_t = self.render.attachNewNode('target')
        self.cam_t.setPos(0, -8, 1)
        self.auto_orbit, self.playing, self.probing = demo, False, False
        self.last_mouse, self.dragging = None, False
        self.keys = set()
        self.geometry = self.render.attachNewNode('scene')
        self.cube = _cube()
        from panda3d.core import AmbientLight, DirectionalLight
        ambient, sun = AmbientLight('ambient'), DirectionalLight('sun')
        ambient.setColor((.55, .55, .6, 1))
        sun.setColor((.55, .55, .5, 1))
        sun_path = self.render.attachNewNode(sun)
        sun_path.setHpr(35, -55, 0)
        for light in (self.render.attachNewNode(ambient), sun_path):  # lit faces shade apart, so a bar reads as a bar
            self.cube.setLight(light)
        self.gate_nodes, self.gate_labels = {}, {}
        self.weight_kind, self.head_stage = 'weight', 'pattern'
        self.cell = [0, 0]  # the highlighted weight, relative to the visible 16x16 tile
        self.matrix_rows = 16
        self.link_lost = False
        self.snapshot_paths, self.snapshot_index = [], -1
        self.stats, self.info, self.note, self.caption = {}, '', '', ''
        self.animation_elapsed = 0.
        self.build_controls()
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
        self.accept('i', self.move_cell, [-1, 0])
        self.accept('k', self.move_cell, [1, 0])
        self.accept('j', self.move_cell, [0, -1])
        self.accept('l', self.move_cell, [0, 1])
        self.accept('window-event', self.window_event)
        self.set_spec(self.scene.spec, fit=True)
        self.taskMgr.add(self.tick, 'scene-events')

    # controls: flat panels in the app's colours, anchored to the window's corners so a resize keeps them in place ---
    def label(self, parent, text='', scale=ui.SMALL, color='muted', align=TextNode.ALeft, **kw):
        from direct.gui.DirectGui import DirectLabel
        kw.setdefault('frameColor', (0, 0, 0, 0))
        return DirectLabel(parent=parent, text=text, scale=scale, text_fg=ui.rgba(color), text_align=align, **kw)

    def panel(self, parent, pos):
        from direct.gui import DirectGuiGlobals as DGG
        from direct.gui.DirectGui import DirectFrame
        return DirectFrame(parent=parent, pos=pos, relief=DGG.FLAT, frameColor=ui.rgba('panel', .94), frameSize=(0, ui.PANEL_W, -.1, 0))

    def button(self, parent, text, command, width=ui.CONTROL_W, primary=False):
        from direct.gui import DirectGuiGlobals as DGG
        from direct.gui.DirectGui import DirectButton
        ready, pressed, hover, disabled = ui.text_colors(primary)
        return DirectButton(parent=parent, text=text, command=command, scale=ui.TEXT, relief=DGG.FLAT, pressEffect=0,
                            frameColor=ui.button_colors(primary), frameSize=ui.button_frame(width),
                            text0_fg=ready, text1_fg=pressed, text2_fg=hover, text3_fg=disabled)

    def menu(self, parent, items, index, command):
        from direct.gui import DirectGuiGlobals as DGG
        from direct.gui.DirectGui import DirectOptionMenu
        return DirectOptionMenu(parent=parent, items=list(items), initialitem=index, command=command, scale=ui.TEXT,
                                relief=DGG.FLAT, pressEffect=0, frameSize=ui.menu_frame(), frameColor=ui.button_colors(),
                                text_fg=ui.rgba('text'), highlightColor=ui.rgba('border'),
                                popupMarker_relief=DGG.FLAT, popupMarker_frameColor=ui.rgba('violet'),
                                popupMenu_relief=DGG.FLAT, popupMenu_frameColor=ui.rgba('panel2'),
                                item_relief=DGG.FLAT, item_frameColor=ui.rgba('panel2'), item_text_fg=ui.rgba('text'),
                                item_pressEffect=0)

    def build_controls(self):
        import studio_registry as registry
        self.mixers = [m.short for m in registry.for_slot('mixer')]
        self.hud = self.label(self.a2dTopCenter, scale=ui.TEXT, color='text', align=TextNode.ACenter, pos=(0, 0, -.08),
                              text_wordwrap=34, frameColor=ui.rgba('panel', .85), pad=(.5, .35))
        # left: what the scene shows, one labelled menu per row; a row is hidden where it would do nothing
        self.view_panel = self.panel(self.a2dTopLeft, (.04, 0, -.04))
        menus = {'level': (ui.LEVELS, ui.LEVELS.index('Model'), self.set_level),
                 'mode': (MODES, 0, self.set_mode),
                 'mixer': (self.mixers, 0, self.swap_mixer),
                 'stage': (HEAD_STAGES, HEAD_STAGES.index('pattern'), self.set_head_stage),
                 'tile': (TILE_KINDS, 0, self.set_weight_kind),
                 'matrix': (['none'], 0, self.select_matrix)}
        self.rows = {name: [self.label(self.view_panel), self.menu(self.view_panel, items, index, command)]
                     for name, (items, index, command) in menus.items()}
        self.level_menu, self.mode_menu, self.module_menu = (self.rows[n][1] for n in ('level', 'mode', 'mixer'))
        self.step_menu, self.tile_menu = self.rows['stage'][1], self.rows['tile'][1]
        # right: the two requests answered by the studio (a demo viewer has no studio, so it says so instead)
        self.snap_panel = self.panel(self.a2dTopRight, (-.04 - ui.PANEL_W, 0, -.04))
        self.load_button = self.button(self.snap_panel, 'Load snapshots', self.load_snapshots, primary=True)
        self.probe_button = self.button(self.snap_panel, 'Run one real step', self.probe_step)
        self.snap_note = self.label(self.snap_panel, text_wordwrap=ui.CONTROL_W / ui.SMALL, pos=(ui.PAD, 0, -ui.PAD - ui.SMALL))
        # bottom: the replay of loaded snapshots
        self.play_panel = self.panel(self.a2dBottomCenter, (-.3, 0, .2))
        self.position = self.label(self.play_panel, align=TextNode.ACenter, color='text2', pos=(.3, 0, -ui.PAD - ui.SMALL))
        self.previous_button = self.button(self.play_panel, 'Previous', lambda: self.scrub(-1), width=.17)
        self.play_button = self.button(self.play_panel, 'Play', self.toggle_play, width=.17, primary=True)
        self.next_button = self.button(self.play_panel, 'Next', lambda: self.scrub(1), width=.17)
        for x, b in zip((.115, .3, .485), (self.previous_button, self.play_button, self.next_button)):
            b.setPos(x, 0, -ui.PAD - ui.SMALL - .025 - 1.1 * ui.TEXT)
        # bottom left: on/off switches, shown only where they change something
        self.toggle_panel = self.panel(self.a2dBottomLeft, (.04, 0, .04))
        self.toggles = {'motion': self.button(self.toggle_panel, '', self.toggle_motion),
                        'numbers': self.button(self.toggle_panel, '', self.toggle_numbers)}
        self.keys_hint = self.label(self.a2dBottomRight, text=ui.KEYS, align=TextNode.ARight, color='dim', text_wordwrap=22)
        self.keys_hint.setPos(-.04, 0, .04 - self.keys_hint.getBounds()[2] * ui.SMALL)  # its last line sits just above the edge

    def sync_controls(self):
        """Make every control show the scene's real state: the menus' choices (set without running their commands),
        which rows and buttons exist, the labels of Play and the switches, and the panels' sizes."""
        from direct.gui import DirectGuiGlobals as DGG
        if not hasattr(self, 'rows'):
            return
        layer = self.scene.selection.layer
        names = self.matrix_names()
        if names and list(self.rows['matrix'][1]['items']) != names:  # a new list: a fresh menu keeps its styled items
            self.rows['matrix'][1].destroy()
            self.rows['matrix'][1] = self.menu(self.view_panel, names, 0, self.select_matrix)
        mixer = self.scene.gates[layer]['entry']['mixer']['id'] if self.scene.gates else None
        for name, value in (('level', self.scene.level), ('mode', self.scene.mode), ('mixer', mixer),
                            ('stage', self.head_stage), ('tile', self.weight_kind), ('matrix', self.scene.selection.weight)):
            widget = self.rows[name][1]
            if value in widget['items'] and widget.get() != value:
                widget.set(widget['items'].index(value), fCommand=0)
        shown = ui.view_rows(self.scene.level, self.scene.mode, names)
        places, bottom = ui.view_layout(len(shown))
        text = ui.captions(layer)
        for name, (caption, widget) in self.rows.items():
            if name not in shown:
                caption.hide()
                widget.hide()
                continue
            caption_z, control_z = places[shown.index(name)]
            caption['text'] = text[name]
            caption.setPos(ui.PAD, 0, caption_z)
            widget.setPos(ui.PAD + .4 * ui.TEXT, 0, control_z)
            caption.show()
            widget.show()
        self.view_panel['frameSize'] = (0, ui.PANEL_W, bottom, 0)
        snap = ui.snapshot_buttons(self.link is not None, self.probing)
        if snap['show']:
            self.snap_note.hide()
            places, bottom = ui.stack_layout(2)
            for z, b in zip(places, (self.load_button, self.probe_button)):
                b.setPos(ui.PANEL_W / 2, 0, z)
                b.show()
            self.probe_button['text'] = snap['probe']
            self.probe_button['state'] = DGG.NORMAL if snap['probe_enabled'] else DGG.DISABLED
        else:
            self.load_button.hide()
            self.probe_button.hide()
            self.snap_note['text'] = snap['note']
            self.snap_note.resetFrameSize()
            self.snap_note.show()
            bottom = self.snap_note.getBounds()[2] * ui.SMALL + self.snap_note.getZ() - ui.PAD
        self.snap_panel['frameSize'] = (0, ui.PANEL_W, bottom, 0)
        play = ui.playback(len(self.snapshot_paths), self.snapshot_index, self.playing)
        self.position['text'] = play['position']
        for b, on in ((self.previous_button, play['previous']), (self.next_button, play['next']), (self.play_button, True)):
            b.show() if play['show'] else b.hide()
            b['state'] = DGG.NORMAL if on else DGG.DISABLED
        self.play_button['text'] = play['play']
        self.play_panel['frameSize'] = (0, .6, -(2 * ui.PAD + ui.SMALL + (.025 + ui.CONTROL_H if play['show'] else 0)), 0)
        self.play_panel.setZ(.04 - self.play_panel['frameSize'][2])
        shown = ui.toggles(self.scene.level, self.snapshot_paths, self.auto_orbit)
        places, bottom = ui.stack_layout(len(shown))
        for name, b in self.toggles.items():
            b.show() if name in shown else b.hide()
        for z, name in zip(places, shown):
            self.toggles[name].setPos(ui.PANEL_W / 2, 0, z)
        self.toggles['motion']['text'] = ui.toggle_text('Reduce motion', self.scene.reduce_motion)
        self.toggles['numbers']['text'] = ui.toggle_text('Show numbers', self.scene.show_numbers)
        self.toggle_panel.show() if shown else self.toggle_panel.hide()
        self.toggle_panel['frameSize'] = (0, ui.PANEL_W, bottom, 0)
        self.toggle_panel.setZ(.04 - bottom)

    def matrix_names(self):
        """The 2D weight matrices the loaded snapshot has that the matrix menu offers (the selected block's)."""
        arrays = self.scene.snapshot['arrays'] if self.scene.snapshot else {}
        names = sorted(key[8:] for key in arrays if key.startswith('weight::') and arrays[key].ndim == 2)
        return ui.block_matrices(names, self.scene.selection.layer)

    def follow_block_matrix(self):
        arrays = self.scene.snapshot['arrays'] if self.scene.snapshot else {}
        names = sorted(key[8:] for key in arrays if key.startswith('weight::') and arrays[key].ndim == 2)
        chosen = ui.pick_matrix(names, self.scene.selection.layer, self.scene.selection.weight)
        if chosen != self.scene.selection.weight:
            self.scene.choose(weight=chosen)

    def window_event(self, window):
        if window and not window.getProperties().getOpen():
            self.close()
            return
        super().windowEvent(window)  # ShowBase's own handler: refits the corner anchors and the lens after a resize

    def send(self, message):
        if self.link:
            self.link.send(message)

    def tell_scene(self):
        """The studio's page mirrors the zoom and mode chosen here."""
        self.send({'type': 'scene', 'level': self.scene.level, 'mode': self.scene.mode})

    def close(self):
        self.send({'type': 'closed'})
        self.userExit()

    def key(self, key, pressed):
        self.keys.add(key) if pressed else self.keys.discard(key)

    def drag(self, value):
        self.dragging, self.last_mouse = value, None

    def toggle_motion(self):
        self.scene.reduce_motion = not self.scene.reduce_motion
        self.sync_controls()

    def toggle_numbers(self):
        self.scene.show_numbers = not self.scene.show_numbers
        self.rebuild()

    def set_level(self, value, tell=True):
        changed = value != self.scene.level
        self.scene.level = value
        self.rebuild()
        if changed:
            self.frame_camera()
        if tell:
            self.tell_scene()

    def set_mode(self, value, tell=True):
        self.scene.mode = value
        self.rebuild()
        if tell:
            self.tell_scene()

    def set_weight_kind(self, value):
        self.weight_kind = value
        self.rebuild()

    def set_head_stage(self, value):
        self.head_stage = value
        self.rebuild()

    def select_matrix(self, value):
        self.scene.choose(weight=value)
        self.rebuild()

    def frame_camera(self):
        """Point the camera at what the level draws, keeping the user's orbit angle and pitch."""
        target, distance = ui.camera_view(self.scene.level, len(self.ids), self.matrix_rows)
        self.cam_t.setPos(*target)
        self.cam_d = distance

    def zoom(self, factor):
        self.cam_d = max(2., min(160., self.cam_d * factor))

    def next_block(self):
        self.scene.choose(layer=(self.scene.selection.layer + 1) % len(self.scene.gates))
        self.sel_id = self.ids[self.scene.selection.layer]
        self.follow_block_matrix()
        self.send({'type': 'select', 'block_id': self.sel_id, 'index': self.scene.selection.layer})
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
        count, layer = len(self.ids), self.scene.selection.layer
        changed = self.scene.update_spec(spec)
        self.ids = [gate['id'] for gate in self.scene.gates]
        if selected in self.ids:
            self.scene.choose(layer=self.ids.index(selected))
            self.sel_id = selected
        elif not keep_selection or self.sel_id not in self.ids:
            self.sel_id = self.ids[self.scene.selection.layer]
        if fit or count != len(self.ids):
            self.frame_camera()
        if layer != self.scene.selection.layer:  # both gates' highlight changes
            changed = sorted(set(changed) | {layer, self.scene.selection.layer})
        self.rebuild(changed=changed)

    def _selected_block(self):
        return block_at(self.scene.spec, self.scene.selection.layer)

    def swap_mixer(self, module):
        """Put another token mixer in the selected block. The studio saves it as a new version and sends the design
        back; until then (or if it refuses) the menu shows the block's real mixer."""
        layer = self.scene.selection.layer
        if not self.ids or module == self.scene.gates[layer]['entry']['mixer']['id']:
            return
        self.send({'type': 'module_swap', 'block_id': self.ids[layer], 'slot': 'mixer', 'entry': {'id': module}})
        if self.demo:
            try:
                self.rebuild(self.scene.swap(layer, 'mixer', {'id': module}))
            except ValueError as error:
                self.note = str(error)
                self.rebuild()
        else:
            self.note = f'Saving block {layer + 1} with {module} as a new version...'
            self.rebuild()

    def load_snapshots(self):
        self.note = 'Asking the studio for this version\'s training snapshots...'
        self.show_hud()
        self.send({'type': 'load_replay'})

    def probe_step(self):
        if self.probing:
            return
        self.probing = True
        self.note = 'Running one real training step on this version\'s checkpoint...'
        self.sync_controls()
        self.show_hud()
        self.send({'type': 'probe_step'})

    def pan_tiles(self, row, column):
        self.scene.choose(row=max(0, self.scene.selection.row + row), column=max(0, self.scene.selection.column + column))
        self.rebuild()

    def move_cell(self, row, column):
        tile = np.asarray(self.scene.frame('Weights').get('weight', []))
        if tile.ndim != 2:
            return
        self.cell = [max(0, min(tile.shape[0] - 1, self.cell[0] + row)), max(0, min(tile.shape[1] - 1, self.cell[1] + column))]
        self.rebuild()

    def toggle_play(self):
        if not self.snapshot_paths:
            self.note = 'Nothing to play yet: load snapshots first.'
            self.show_hud()
            return
        if not self.playing and self.snapshot_index + 1 >= len(self.snapshot_paths):
            self.scrub_to(0)  # at the end, Play starts over
        self.playing = not self.playing
        self.sync_controls()

    def scrub(self, delta):
        self.scrub_to(self.snapshot_index + delta)

    def scrub_to(self, index):
        if not self.snapshot_paths:
            return
        self.snapshot_index = clamp_index(index, len(self.snapshot_paths))
        self.read_snapshot(self.snapshot_paths[self.snapshot_index])
        self.send({'type': 'replay_state', 'index': self.snapshot_index, 'count': len(self.snapshot_paths)})

    def show_hud(self):
        line = stats_line(self.stats)
        self.hud['text'] = (self.info + (f' | training: {line}' if line else '') + (f'\n{self.caption}' if self.caption else '')
                            + (f'\n{self.note}' if self.note else ''))
        self.hud.resetFrameSize()  # DirectGUI sizes a frame once: refit the backdrop to the new text

    def read_snapshot(self, path):
        from studio_snapshot import load
        self.scene.load_snapshot(load(path))
        ids = [gate['id'] for gate in self.scene.gates]
        if ids != self.ids:  # the snapshot carries its own design: keep the block list in step with it
            self.ids = ids
            self.sel_id = ids[self.scene.selection.layer]
            self.frame_camera()
        self.animation_elapsed = 0.
        self.follow_block_matrix()
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

    def lines(self, parent, points, color=ui.SCENE['mixer'], thickness=1.):
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
        self.cube_at(root, (0, 0, 1.5), (6, .18, 1.5), ui.SCENE['gate'][:3] + (.65 if selected else .3,))
        if self.scene.snapshot:
            ratios = [value for key, value in self.scene.snapshot.get('update_ratios', {}).items() if key.startswith(f'block{index}.')]
            brightness = min(1., max(ratios, default=0.) * 20)
            self.cube_at(root, (0, 0, .05), (6, .21, .04), (brightness, .25 + brightness * .5, .3, .8))
        self.gate_labels[index] = self.label(self.aspect2d, f"{index + 1}  {gate['entry']['mixer']['id']} / {gate['entry']['channel']['id']}",
                                             color='violet' if selected else 'text2', align=TextNode.ACenter, pad=(.4, .25),
                                             frameColor=ui.rgba('violet_bg' if selected else 'panel', .9))
        if frame['level'] == 'Model':
            bins = len(frame['edges']) - 1
            for bin_index in range(bins):
                self.cube_at(root, (-5.5 + 11 * bin_index / max(1, bins - 1), 0, 1.5), (.09, .3, .4), ui.SCENE['activation'])
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
            self.lines(root, [(x, 1.5, .1), (x, -1.5, .1)], color=ui.SCENE['lane'])
            magnitude = min(2., norms[lane]) if norms else .25
            color = ui.SCENE['token'] if lane + frame['start'] == self.scene.selection.token else ui.SCENE['activation']
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
                self.cube_at(root, (x, -.7, 1), (.06, .1, .4), ui.SCENE['channel'])
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
                if index in self.gate_labels:
                    self.gate_labels.pop(index).destroy()
            for gate in frame['gates']:
                if gate['index'] in changed:
                    self.draw_gate(gate, frame)
        else:
            self.geometry.removeNode()
            self.geometry = self.render.attachNewNode('scene')
            self.gate_nodes = {}
            for chip in self.gate_labels.values():
                chip.destroy()
            self.gate_labels = {}
            self.pulse = None
            self.caption = ''
            level = frame['level']
            if level in ('Model', 'Window'):
                for gate in frame['gates']:
                    self.draw_gate(gate, frame)
                self.pulse = self.cube_at(self.geometry, (0, 0, 1), (.15, .2, .15), (1, 1, 1, 1))
            elif level in ('Head', 'Weights'):
                matrix = np.asarray(frame.get(self.weight_kind if level == 'Weights' else self.head_stage, []))
                if level == 'Head' and self.head_stage in ('scale', 'mask'):
                    matrix = np.asarray(frame.get('logits') or [])  # what softmax sees: score scale and position bias included
                    if not matrix.size:
                        matrix = np.asarray(frame['scores'])
                        if matrix.size:  # snapshots without attn.logits: plain q·k/√head_dim
                            heads = self._selected_block()['mixer']['config'].get('heads', 1)
                            matrix = matrix / math.sqrt(self.scene.spec['dims']['d_model'] / heads)
                    if matrix.size and self.head_stage == 'mask':
                        matrix = np.where(frame['mask'], 0, matrix)
                self.matrix_rows = matrix.shape[0] if matrix.ndim == 2 else 16
                if matrix.size:
                    maximum = max(float(np.max(np.abs(matrix))), 1e-12)
                    for row in range(matrix.shape[0]):
                        for column in range(matrix.shape[1]):
                            value = float(matrix[row, column])
                            x, y = (column - matrix.shape[1] / 2) * .4, -row * .4
                            masked = level == 'Head' and self.head_stage != 'values' and frame['mask'][row][column]
                            color = ui.SCENE['masked'] if masked else ui.SCENE['positive'] if value > 0 else ui.SCENE['negative']
                            if level == 'Weights' and [row, column] == self.cell:
                                color = ui.SCENE['picked']
                            height = .015 if masked else .03 + abs(value) / maximum * 1.5
                            self.cube_at(self.geometry, (x, y, height / 2), (.18, .18, height / 2), color)
                    self.caption = ui.matrix_legend(level)
                    if self.scene.show_numbers:
                        numbers = f'{matrix.shape[0]}x{matrix.shape[1]} | range {matrix.min():.3g} to {matrix.max():.3g}'
                        if level == 'Weights':
                            cell = (min(self.cell[0], matrix.shape[0] - 1), min(self.cell[1], matrix.shape[1] - 1))
                            numbers += f' | cell {cell[0]},{cell[1]} = {float(matrix[cell]):.4g} (I J K L move)'
                        self.caption += f'\n{numbers}'
                else:
                    gate = self.scene.gates[self.scene.selection.layer]
                    note = ui.empty_matrix_note(level, bool(self.scene.snapshot), self.scene.selection.layer, gate['entry']['mixer']['id'],
                                                gate['glyphs']['mixer']['kind'] == 'beams')
                    self.label_at(self.geometry, note, (0, 0, 2))
            self.drawn_level = level
        info = f"{self.scene.spec['name']} | {frame['level']} | {frame['mode']} | block {self.scene.selection.layer + 1}"
        if frame['step'] is not None:
            info += f" | step {frame['step']}"
        if frame['loss'] is not None:
            info += f" | loss {frame['loss']:.4f}"
        if self.scene.mode == 'Ride a token' and self.scene.snapshot:
            lens = self.scene.snapshot.get('lens', {}).get(str(self.scene.selection.layer))
            if lens:
                token = min(self.scene.selection.token, len(lens['ids']) - 1)
                info += f" | predicted id {lens['ids'][token][0]} ({lens['probabilities'][token][0]:.3f})"
        self.info = info
        self.sync_controls()
        self.show_hud()
        self.place_block_labels()

    def screen_point(self, point):
        """Where a point of the 3D scene appears on screen, in aspect2d units; None when it is off screen."""
        from panda3d.core import Point2, Point3
        projected = Point2()
        if not self.camLens.project(self.cam.getRelativePoint(self.render, Point3(*point)), projected):
            return None
        on_screen = self.aspect2d.getRelativePoint(self.render2d, Point3(projected.x, 0, projected.y))
        return on_screen.x, on_screen.z

    def place_block_labels(self):
        """Each block's label floats above its gate, drawn flat on the screen so it stays readable from any angle, and
        moves with its gate as the camera turns; where gates line up one behind the other, the farther labels hide
        (the selected block's label always shows) instead of jumping up to dodge."""
        if not self.gate_labels:
            return
        camera = self.camera.getPos(self.render)
        selected = self.scene.selection.layer
        order = sorted(self.gate_labels, key=lambda i: (i != selected, (camera - Vec3(0, -i * 3, 3.5)).length()))
        items = []
        for index in order:
            spot = self.screen_point((0, -index * 3, 3.5))
            chip = self.gate_labels[index]
            items.append(None if spot is None else (spot[0], spot[1], chip.getWidth() * ui.SMALL))
        height = max(chip.getHeight() for chip in self.gate_labels.values()) * ui.SMALL
        for index, item, shown in zip(order, items, ui.place_labels(items, height)):
            chip = self.gate_labels[index]
            if shown:
                chip.show()
                chip.setPos(item[0], 0, item[1])
            else:
                chip.hide()

    def handle_message(self, message):
        kind = message.get('type')
        if kind == 'quit':
            self.close()
        elif kind == 'spec':
            self.set_spec(message['spec'], message.get('selected'))
        elif kind in ('viz_snapshot', 'snapshot'):
            if kind == 'snapshot':  # the answer to Run one real step, loadable or not
                self.probing = False
                self.note = 'Ran one real step. Press Play to watch it pass through the blocks.'
            try:
                self.read_snapshot(message['path'])
                # training keeps only the newest snapshots on disk; forget the pruned ones so scrubbing never opens a missing file
                self.snapshot_paths = [p for p in self.snapshot_paths if p != message['path'] and os.path.exists(p)] + [message['path']]
                self.snapshot_index = len(self.snapshot_paths) - 1
                self.send({'type': 'replay_state', 'index': self.snapshot_index, 'count': len(self.snapshot_paths)})
            except Exception as error:  # noqa: BLE001 - a truncated or foreign file is reported, and the viewer carries on
                self.note = f'Could not load the snapshot: {error}'
                self.show_hud()
            self.sync_controls()
        elif kind == 'step':  # walk the head's forward pass: delta +1 / -1, or jump to a named stage
            stage = next_stage(self.head_stage, message.get('delta', 0), message.get('stage'))
            self.set_head_stage(stage)
        elif kind == 'scrub':
            self.scrub_to(message['index'])
        elif kind == 'show':  # weight tile kind at the Weights level
            if message.get('what') in TILE_KINDS:
                self.set_weight_kind(message['what'])
        elif kind == 'stats':
            self.stats = message
            self.show_hud()
        elif kind == 'replay':
            self.snapshot_paths = list(message.get('paths', []))
            self.snapshot_index = -1
            self.playing = False
            self.note = (f'Loaded {len(self.snapshot_paths)} snapshots. Play them, or step with Previous and Next.'
                         if self.snapshot_paths else 'This version has no training snapshots yet.')
            self.scrub(1)
            self.sync_controls()
            self.show_hud()
        elif kind == 'summary':
            self.scene.paths = message.get('paths', self.scene.paths)
            self.scene.summaries = message.get('attention', self.scene.summaries)
            self.scene.stream_norms = message.get('stream_norms', self.scene.stream_norms)
            self.scene.summary_edges = message.get('edges', self.scene.summary_edges)
            self.rebuild()
        elif kind == 'scene':  # the studio's page chose these: apply them without telling it back
            if message.get('selected'):
                self.scene.choose(**message['selected'])
                self.follow_block_matrix()
            if message.get('mode') in MODES:
                self.scene.mode = message['mode']
            self.set_level(message['level'] if message.get('level') in ui.LEVELS else self.scene.level, tell=False)
        elif kind == 'selection':
            self.scene.choose(**message['selection'])
            self.follow_block_matrix()
            self.rebuild()
        elif kind in ('notice', 'error'):  # the studio's answer to a button here, said in the HUD
            self.probing = self.probing and kind == 'notice'
            self.note = message.get('message', '')
            self.rebuild()  # also puts the mixer menu back on the block's real mixer after a refused swap

    def tick(self, task):
        from panda3d.core import ClockObject
        dt = min(.1, ClockObject.getGlobalClock().getDt())
        if self.playing:
            self.animation_elapsed += dt
        if self.playing and getattr(self, 'pulse', None) is not None and self.scene.snapshot:
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
        if self.link and not self.link.connected and not self.link_lost:
            self.link_lost = True  # the studio is gone: say so, then leave instead of lingering as an orphan
            self.note = 'studio disconnected'
            self.show_hud()
            self.taskMgr.doMethodLater(1.5, lambda _task: self.userExit(), 'studio-disconnected')
        if self.link:
            for message in self.link.poll():
                try:
                    self.handle_message(message)
                except Exception as error:  # one bad message must not take the viewer down
                    self.note = f"Ignored a {message.get('type') if isinstance(message, dict) else 'malformed'} message | {error}"
                    self.show_hud()
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
        self.place_block_labels()
        if self.playing and self.animation_elapsed >= ui.pass_seconds(len(self.scene.gates)):  # the pulse went forward and back
            if self.snapshot_index + 1 < len(self.snapshot_paths):
                self.scrub(1)
            else:
                self.playing = False
                self.sync_controls()
        return task.cont


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int)
    parser.add_argument('--offscreen', action='store_true')
    parser.add_argument('--selftest', type=int)
    parser.add_argument('--screenshot')
    args, _ = parser.parse_known_args(argv)
    loadPrcFileData('', 'audio-library-name null\nsync-video 0\nwin-size 1280 900')
    if args.offscreen or args.selftest:
        loadPrcFileData('', 'window-type offscreen\nframebuffer-hardware 0')
    link = None
    if args.port:
        from studio_vizbridge import ViewerLink, parse_viewer_args
        port, authkey = parse_viewer_args(argv)  # the key arrives on stdin, never on the command line
        link = ViewerLink(port, authkey) if port and authkey else None
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
