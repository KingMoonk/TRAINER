"""The 3D viewer's scene as JSON and float32 bytes for a browser renderer (studio_scene_json): payloads, frames, the binary
manifest and the viewer's controls as plain functions. Window-free; the fixtures are the ones tests/test_scene.py uses."""
import copy
import itertools
import json
import unittest

import numpy as np

import studio_scene_json as sj
import studio_viewer_ui as ui
from studio_compile import CompiledModel
from studio_scene import MODES, SceneModel
from studio_snapshot import one_step
from studio_spec import default_spec, validate_spec

ENVELOPE = {'schema', 'full', 'groups', 'cursor', 'removed_groups'}


def make_spec(layers=3):
    spec = default_spec()
    spec['dims'].update(d_model=16, n_layer=layers, n_ctx=256, dropout=0.)
    return validate_spec(spec)


def sections(frame):
    """The parts of a frame that are not its envelope."""
    return set(frame) - ENVELOPE


def apply_frame(client, frame):
    """What the browser does with a frame: replace each section and group it carries, drop what it names, keep the cursor."""
    client = copy.deepcopy(client)
    client.update({key: frame[key] for key in sections(frame)})
    client['groups'] = {key: group for key, group in client['groups'].items() if key not in frame['removed_groups']}
    client['groups'].update(frame['groups'])
    client['cursor'] = frame['cursor']
    return client


def settled(payload):
    """The payload as a client gets it: through JSON."""
    return json.loads(json.dumps(payload, allow_nan=False))


def refs(node):
    """Every {shape, data} or {shape, name} array in a payload."""
    if isinstance(node, dict):
        if 'shape' in node and set(node) in ({'shape', 'data'}, {'shape', 'name'}):
            yield node
        else:
            for value in node.values():
                yield from refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from refs(value)


def decode(node, data, manifest):
    """A binary payload with its arrays read back from the bytes, in the inline form."""
    if isinstance(node, dict):
        if set(node) == {'shape', 'name'}:
            entry = manifest[node['name']]
            count = int(np.prod(entry['shape']))
            return {'shape': entry['shape'], 'data': np.frombuffer(data, '<f4', count, entry['offset']).tolist()}
        return {key: decode(value, data, manifest) for key, value in node.items()}
    if isinstance(node, list):
        return [decode(value, data, manifest) for value in node]
    return node


def entry(items, kind):
    found = [item for item in items if item['kind'] == kind]
    return found[0] if found else None


class Scenes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = make_spec()
        cls.snapshot = one_step(CompiledModel(cls.spec, 9, device='cpu'), [[1, 2, 3, 4, 5, 6]], [[2, 3, 4, 5, 6, 7]])

    def scene(self, level='Model', snapshot=True):
        scene = SceneModel(copy.deepcopy(self.spec))
        if snapshot:
            scene.load_snapshot(copy.deepcopy(self.snapshot))
        scene.level = level
        return scene

    def load(self, step=None):
        later = copy.deepcopy(self.snapshot)
        if step is not None:
            later['step'] = step
        return lambda path: later


class Payload(Scenes):
    def test_every_level_is_json_with_and_without_a_snapshot(self):
        for snapshot, level in itertools.product((False, True), sj.VIEWER_LEVELS):
            payload = sj.scene_payload(self.scene(level, snapshot))
            self.assertEqual(settled(payload)['view']['level'], level)
            self.assertEqual((payload['schema'], payload['full'], payload['up']), (sj.SCHEMA, True, 'z'))
            for array in refs(payload):
                self.assertEqual(len(array['data']), int(np.prod(array['shape'])), (level, snapshot))

    def test_arguments_are_checked(self):
        with self.assertRaises(ValueError):
            sj.scene_payload(self.scene(), arrays='csv')
        with self.assertRaises(ValueError):
            sj.scene_frame(self.scene(), ['not', 'a', 'cursor'])

    def test_design_and_stats_follow_the_model(self):
        scene = self.scene('Window')
        payload = sj.scene_payload(scene, sj.ViewState(stats={'step': 5, 'total': 10, 'loss': 1.23456}))
        self.assertEqual(payload['design']['n_layer'], len(scene.gates))
        self.assertEqual([g['id'] for g in payload['design']['gates']], [g['id'] for g in scene.gates])
        self.assertEqual((payload['stats']['step'], payload['stats']['loss']), (scene.snapshot['step'], scene.snapshot['loss']))
        self.assertEqual(payload['stats']['training_line'], 'step 5/10 | loss 1.2346')
        self.assertTrue(payload['stats']['info'].startswith(f"{scene.spec['name']} | Window | Build | block 1"))
        self.assertEqual(payload['selection']['layer'], 0)
        self.assertEqual(payload['selection']['block_id'], scene.gates[0]['id'])
        self.assertEqual(payload['view']['levels'], list(ui.LEVELS))

    def test_model_level_has_a_group_per_block_and_a_box_per_bin(self):
        scene = self.scene('Model')
        frame, payload = scene.frame(), sj.scene_payload(scene)
        self.assertEqual(sorted(payload['groups']), [f'gate:{i}' for i in range(len(scene.gates))])
        for index in range(len(scene.gates)):
            group = payload['groups'][f'gate:{index}']
            self.assertEqual(group['origin'], [0., -index * 3., 0.])  # studio_brain3d.draw_gate: root.setY(-index * 3)
            self.assertEqual(group['selected'], index == 0)
            bins = entry(group['boxes'], 'activation')
            self.assertEqual(bins['count'], len(frame['edges']) - 1)
            self.assertEqual((bins['position']['shape'], bins['size']['shape']), ([bins['count'], 3], [bins['count'], 3]))
            self.assertEqual(entry(group['boxes'], 'gate')['size']['data'], [12., .36, 3.])  # the viewer's cube has half-extent 1
            self.assertEqual(group['labels'][0]['text'], f"{index + 1}  {scene.gates[index]['entry']['mixer']['id']} / "
                                                         f"{scene.gates[index]['entry']['channel']['id']}")
        self.assertEqual(payload['groups']['gate:0']['labels'][0]['kind'], 'label_selected')
        self.assertEqual(payload['groups']['gate:1']['labels'][0]['kind'], 'label')

    def test_model_level_draws_attention_arcs_from_the_summary(self):
        scene = self.scene('Model', snapshot=False)
        bins = 8
        scene.summary_edges = np.linspace(0, 64, bins + 1)
        mass = np.zeros((2, bins, bins))
        mass[:, np.arange(bins), np.arange(bins)] = .9  # every bin attends to itself: one arc per bin
        scene.summaries = {'block1.attn.pattern': mass}
        payload = sj.scene_payload(scene)
        self.assertIsNone(entry(payload['groups']['gate:0']['lines'], 'mixer'))
        arcs = entry(payload['groups']['gate:1']['lines'], 'mixer')
        self.assertEqual((arcs['count'], arcs['points']['shape']), (bins, [bins, 3, 3]))
        self.assertEqual(entry(payload['groups']['gate:1']['boxes'], 'activation')['count'], bins)

    def test_window_level_has_a_lane_and_a_bar_per_token(self):
        scene = self.scene('Window')
        frame, payload = scene.frame(), sj.scene_payload(scene)
        count = len(frame['tokens'])
        for index, gate in enumerate(scene.gates):
            group = payload['groups'][f'gate:{index}']
            lanes = entry(group['lines'], 'lane')
            self.assertEqual((lanes['count'], lanes['points']['shape']), (count, [count, 2, 3]))
            bars = sum(item['count'] for item in group['boxes'] if item['kind'] in ('activation', 'token'))
            self.assertEqual(bars, count)
            self.assertEqual(entry(group['boxes'], 'token')['count'], 1)  # the selected token, amber
            if gate['glyphs']['mixer']['kind'] == 'beams':
                self.assertIsNotNone(entry(group['lines'], 'mixer'))
            if gate['entry']['channel']['id'] != 'none':
                self.assertEqual(entry(group['boxes'], 'channel')['count'], count)
        tokens = [label for label in payload['groups']['gate:0']['labels'] if label['kind'] == 'note']
        self.assertEqual([label['text'] for label in tokens], [str(token) for token in frame['tokens']])
        self.assertFalse([label for label in payload['groups']['gate:1']['labels'] if label['kind'] == 'note'])  # numbers on block 1 only

    def test_numbers_off_drops_the_token_labels(self):
        scene = self.scene('Window')
        sj.toggle_numbers(scene)
        labels = sj.scene_payload(scene)['groups']['gate:0']['labels']
        self.assertEqual([label['kind'] for label in labels], ['label_selected'])

    def test_head_level_has_a_bar_per_cell(self):
        scene = self.scene('Head')
        frame, payload = scene.frame(), sj.scene_payload(scene)
        group = payload['groups']['matrix']
        pattern = np.asarray(frame['pattern'])
        self.assertEqual(sum(item['count'] for item in group['boxes']), pattern.size)
        self.assertEqual((group['level'], group['kind']), ('Head', 'pattern'))
        masked = entry(group['boxes'], 'masked')
        self.assertEqual(masked['count'], int(np.sum(frame['mask'])))
        for item in group['boxes']:
            for (row, column), value in zip(np.reshape(item['cell']['data'], (-1, 2)).astype(int), item['value']['data']):
                self.assertAlmostEqual(value, pattern[row, column], places=5)
                self.assertEqual(item['kind'] == 'masked', bool(frame['mask'][row][column]))
        self.assertIn('rows = queries', group['caption'])

    def test_matrix_bars_stand_where_the_viewer_puts_them(self):
        scene = self.scene('Weights')
        group = sj.scene_payload(scene)['groups']['matrix']
        rows = columns = 16
        for item in group['boxes']:
            positions, sizes = np.reshape(item['position']['data'], (-1, 3)), np.reshape(item['size']['data'], (-1, 3))
            for (row, column), position, size in zip(np.reshape(item['cell']['data'], (-1, 2)), positions, sizes):
                np.testing.assert_allclose(position, ((column - columns / 2) * .4, -row * .4, size[2] / 2), atol=1e-5)
                np.testing.assert_allclose(size[:2], (.36, .36), atol=1e-6)
        self.assertEqual(sum(item['count'] for item in group['boxes']), rows * columns)

    def test_weights_level_matches_the_tile(self):
        scene = self.scene('Weights')
        frame = scene.frame()
        view = sj.ViewState()
        group = sj.scene_payload(scene, view)['groups']['matrix']
        self.assertEqual((group['level'], group['kind'], group['weight_name']), ('Weights', 'weight', frame['weight_name']))
        self.assertEqual((group['shape'], group['origin_cell']), (list(frame['shape']), list(frame['origin'])))
        picked = entry(group['boxes'], 'picked')
        self.assertEqual((picked['count'], picked['cell']['data']), (1, [0., 0.]))
        sj.move_cell(scene, view, 1, 2)
        sj.show_tile(view, 'gradient')
        group = sj.scene_payload(scene, view)['groups']['matrix']
        self.assertEqual((view.cell, group['kind']), ([1, 2], 'gradient'))
        self.assertEqual(entry(group['boxes'], 'picked')['cell']['data'], [1., 2.])
        sj.move_cell(scene, view, 99, 99)
        self.assertEqual(view.cell, [15, 15])
        self.assertIn('cell 15,15 =', sj.scene_payload(scene, view)['groups']['matrix']['caption'])

    def test_a_level_with_nothing_to_draw_says_why(self):
        for level, snapshot in (('Head', False), ('Weights', False)):
            group = sj.scene_payload(self.scene(level, snapshot))['groups']['matrix']
            self.assertEqual(group['boxes'], [])
            self.assertEqual(group['labels'][0]['text'], 'Load snapshots, or run one real step, to see these values.')

    def test_nan_weights_still_make_strict_json(self):
        snapshot = copy.deepcopy(self.snapshot)
        key = next(key for key, value in snapshot['arrays'].items() if key.startswith('weight::') and value.ndim == 2)
        snapshot['arrays'][key][0, 0] = np.nan
        snapshot['arrays'][key][0, 1] = np.inf
        scene = self.scene('Weights', snapshot=False)
        scene.load_snapshot(snapshot)
        scene.choose(weight=key[8:])
        payload = sj.scene_payload(scene)
        json.dumps(payload, allow_nan=False)
        self.assertEqual(sum(item['count'] for item in payload['groups']['matrix']['boxes']), np.asarray(scene.frame()['weight']).size)

    def test_nan_in_attention_reads_the_same_inline_and_binary(self):
        snapshot = copy.deepcopy(self.snapshot)
        key = next(key for key in snapshot['arrays'] if key.startswith('hook::') and key.endswith('attn.pattern'))
        snapshot['arrays'][key][0, 0, 0, 0] = np.nan
        scene = self.scene('Window', snapshot=False)
        scene.load_snapshot(snapshot)
        inline = sj.scene_payload(scene)
        json.dumps(inline, allow_nan=False)
        data, manifest = sj.scene_binary(scene)
        self.assertTrue(np.isfinite(np.frombuffer(data, '<f4')).all())
        for left, right in zip(refs(decode(sj.scene_payload(scene, arrays='binary'), data, manifest)), refs(inline)):
            np.testing.assert_allclose(left['data'], right['data'], rtol=1e-5, atol=1e-6)

    def test_matrix_bar_heights_follow_the_viewers_formula(self):
        for level, kind in (('Weights', 'weight'), ('Head', 'pattern')):
            scene = self.scene(level)
            frame = scene.frame()
            matrix = np.asarray(frame[kind])
            maximum = np.max(np.abs(matrix))
            for item in sj.scene_payload(scene)['groups']['matrix']['boxes']:
                for (row, column), size in zip(np.reshape(item['cell']['data'], (-1, 2)).astype(int), np.reshape(item['size']['data'], (-1, 3))):
                    expected = .015 if item['kind'] == 'masked' else .03 + abs(matrix[row, column]) / maximum * 1.5
                    self.assertAlmostEqual(size[2], expected, places=5, msg=(level, row, column))
            self.assertEqual(sj.scene_payload(scene)['groups']['matrix']['pitch'], .4)

    def test_labels_and_the_palette_are_the_viewers(self):
        note = sj.PALETTE['note']['color']
        for got, wanted in zip(ui.rgba(note), (.88, .92, 1., 1.)):  # studio_brain3d.label_at's text colour
            self.assertAlmostEqual(got, wanted, delta=.005)
        payload = sj.scene_payload(self.scene('Window'))
        payload['palette']['note']['color'] = '#000000'
        self.assertEqual(sj.PALETTE['note']['color'], note)  # a payload owns its palette

    def test_camera_frames_the_level_like_the_viewer(self):
        for level in sj.VIEWER_LEVELS:
            scene = self.scene(level)
            rows = len(scene.frame()['pattern']) if level == 'Head' else 16
            target, distance = ui.camera_view(level, len(scene.gates), rows)
            self.assertEqual(sj.scene_payload(scene)['camera'], {'target': list(target), 'distance': distance, 'level': level})


class Frames(Scenes):
    def test_nothing_changed_is_an_empty_frame(self):
        scene, view = self.scene('Window'), sj.ViewState()
        first = sj.scene_payload(scene, view)
        frame = sj.scene_frame(scene, first['cursor'], view)
        self.assertEqual((frame['full'], sections(frame), frame['groups'], frame['removed_groups']), (False, set(), {}, []))
        self.assertEqual(frame['cursor'], first['cursor'])
        self.assertEqual(sj.scene_payload(scene, view)['cursor'], first['cursor'])
        self.assertEqual(sj.scene_payload(scene, view, 'binary')['cursor'], first['cursor'])  # one cursor for both encodings

    def test_no_cursor_is_a_whole_payload(self):
        scene = self.scene('Head')
        for since in (None, {}):
            frame = sj.scene_frame(scene, since)
            self.assertTrue(frame['full'])
            self.assertEqual(settled(frame), settled(sj.scene_payload(scene)))

    def test_a_stage_change_shows_in_the_next_frame(self):
        scene, view = self.scene('Head'), sj.ViewState()
        first = sj.scene_payload(scene, view)
        self.assertEqual(first['view']['head_stage'], 'pattern')
        sj.step_stage(view, 1)
        frame = sj.scene_frame(scene, first['cursor'], view)
        self.assertEqual((frame['full'], sections(frame), sorted(frame['groups'])), (False, {'view'}, ['matrix']))  # not 'camera': the orbit stays
        self.assertEqual((frame['view']['head_stage'], frame['groups']['matrix']['kind']), ('values', 'values'))
        values = np.asarray(scene.frame()['values'])
        self.assertEqual(sum(item['count'] for item in frame['groups']['matrix']['boxes']), values.size)
        sj.step_stage(view, -2)
        frame = sj.scene_frame(scene, frame['cursor'], view)
        self.assertEqual((frame['view']['head_stage'], frame['groups']['matrix']['kind']), ('mask', 'mask'))
        self.assertEqual(entry(frame['groups']['matrix']['boxes'], 'masked')['count'], int(np.sum(scene.frame()['mask'])))
        sj.step_stage(view, stage='scores')
        sj.step_stage(view, -1)  # clamped at the first step
        self.assertEqual(view.head_stage, 'scores')
        self.assertEqual(sj.scene_frame(scene, frame['cursor'], view)['view']['head_stage'], 'scores')

    def test_a_step_shows_in_the_next_frame(self):
        scene, view = self.scene('Model', snapshot=False), sj.ViewState()
        first = sj.scene_payload(scene, view)
        self.assertEqual((first['stats']['step'], first['playback']['show']), (None, False))
        sj.add_snapshot(scene, view, 'step7.npz', load=self.load(step=7))
        frame = sj.scene_frame(scene, first['cursor'], view)
        self.assertEqual((frame['stats']['step'], frame['stats']['loss']), (7, self.snapshot['loss']))
        self.assertIn('step 7', frame['stats']['info'])
        self.assertEqual((frame['playback']['position'], frame['playback']['show']), ('Snapshot 1 of 1', True))
        self.assertEqual(sorted(frame['groups']), ['gate:0', 'gate:1', 'gate:2'])  # every block gained its update bar
        self.assertIsNotNone(entry(frame['groups']['gate:0']['boxes'], 'update')['value'])
        again = sj.scene_frame(scene, frame['cursor'], view)
        self.assertEqual(sections(again), set())
        sj.add_snapshot(scene, view, 'step9.npz', load=self.load(step=9))
        self.assertEqual(sj.scene_frame(scene, frame['cursor'], view)['stats']['step'], 9)

    def test_selecting_a_block_redraws_only_the_blocks_that_change(self):
        scene, view = self.scene('Window'), sj.ViewState()
        first = sj.scene_payload(scene, view)
        sj.select(scene, layer=1)
        frame = sj.scene_frame(scene, first['cursor'], view)
        self.assertEqual(sorted(frame['groups']), ['gate:0', 'gate:1'])  # the one deselected and the one selected; block 3 is untouched
        self.assertTrue(frame['groups']['gate:1']['selected'])
        self.assertEqual(frame['selection']['layer'], 1)
        self.assertNotIn('design', sections(frame))
        sj.select_token(scene, 2)
        later = sj.scene_frame(scene, frame['cursor'], view)
        self.assertEqual(sorted(later['groups']), ['gate:0', 'gate:1', 'gate:2'])  # the amber token moves in every block
        self.assertEqual(later['selection']['token'], 2)

    def test_dropped_groups_are_named(self):
        scene, view = self.scene('Window'), sj.ViewState()
        first = sj.scene_payload(scene, view)
        sj.set_level(scene, 'Head')
        frame = sj.scene_frame(scene, first['cursor'], view)
        self.assertEqual((sorted(frame['groups']), frame['removed_groups']), (['matrix'], ['gate:0', 'gate:1', 'gate:2']))
        sj.set_level(scene, 'Window')
        first = sj.scene_payload(scene, view)
        spec = copy.deepcopy(scene.spec)
        spec['dims']['n_layer'] = 2
        sj.set_spec(scene, validate_spec(spec))
        frame = sj.scene_frame(scene, first['cursor'], view)
        self.assertEqual((frame['removed_groups'], frame['design']['n_layer']), (['gate:2'], 2))

    def test_a_client_that_applies_every_frame_equals_a_fresh_payload(self):
        scene, view = self.scene('Model', snapshot=False), sj.ViewState()
        client = sj.scene_payload(scene, view)
        load = self.load(step=3)
        script = [
            {'type': 'level', 'value': 'Window'}, {'type': 'selection', 'selection': {'layer': 1}}, {'type': 'token', 'delta': 2},
            {'type': 'mode', 'value': 'Train'}, {'type': 'numbers'}, {'type': 'motion'},
            {'type': 'viz_snapshot', 'path': 'a.npz'}, {'type': 'stats', 'step': 3, 'total': 10, 'loss': 1.5, 'lr': 1e-3},
            {'type': 'level', 'value': 'Head'}, {'type': 'step', 'delta': 1}, {'type': 'head', 'delta': 1}, {'type': 'next_block'},
            {'type': 'level', 'value': 'Weights'}, {'type': 'show', 'what': 'gradient'}, {'type': 'cell', 'row': 1, 'column': 3},
            {'type': 'pan', 'row': 4}, {'type': 'matrix', 'name': sj.weight_names(self.scene())[1]},
            {'type': 'replay', 'paths': ['a.npz', 'b.npz', 'c.npz']}, {'type': 'play'}, {'type': 'advance'},
            {'type': 'level', 'value': 'Model'}, {'type': 'level', 'value': 'Window'}, {'type': 'probe_step'}, {'type': 'error', 'message': 'no'},
        ]
        for action in script:
            sj.apply_action(scene, view, action, load)
            frame = sj.scene_frame(scene, client['cursor'], view)
            client = apply_frame(client, frame)
            self.assertEqual(settled(client), settled(sj.scene_payload(scene, view)), action)

    def test_frames_stay_small_when_one_thing_changes(self):
        scene, view = self.scene('Weights'), sj.ViewState()
        first = sj.scene_payload(scene, view)
        sj.apply_action(scene, view, {'type': 'stats', 'step': 4, 'loss': 2.})
        frame = sj.scene_frame(scene, first['cursor'], view)
        self.assertEqual((sections(frame), frame['groups']), ({'stats'}, {}))
        self.assertLess(len(json.dumps(frame)), len(json.dumps(first)) / 10)


class Binary(Scenes):
    def test_manifest_covers_the_buffer(self):
        for level in sj.VIEWER_LEVELS:
            scene = self.scene(level)
            data, manifest = sj.scene_binary(scene)
            payload = sj.scene_payload(scene, arrays='binary')
            named = list(refs(payload))
            self.assertTrue(named)
            self.assertEqual({array['name'] for array in named}, set(manifest), level)
            end = 0
            for name, item in sorted(manifest.items(), key=lambda pair: pair[1]['offset']):
                self.assertEqual((item['offset'], item['dtype']), (end, 'float32'), name)  # no gap, no overlap
                end += 4 * int(np.prod(item['shape']))
            self.assertEqual(end, len(data), level)
            for array in named:
                self.assertEqual(array['shape'], manifest[array['name']]['shape'])

    def test_binary_numbers_are_the_inline_numbers(self):
        for level in sj.VIEWER_LEVELS:
            scene = self.scene(level)
            data, manifest = sj.scene_binary(scene)
            fromdata = decode(sj.scene_payload(scene, arrays='binary'), data, manifest)
            inline = sj.scene_payload(scene)
            self.assertEqual(settled(fromdata).keys(), settled(inline).keys())
            for left, right in zip(refs(fromdata), refs(inline)):
                self.assertEqual(left['shape'], right['shape'])
                np.testing.assert_allclose(left['data'], right['data'], rtol=1e-5, atol=1e-6)
            self.assertEqual(len(list(refs(fromdata))), len(list(refs(inline))))

    def test_binary_is_smaller_than_the_json_it_replaces(self):
        scene = self.scene('Weights')
        data, _ = sj.scene_binary(scene)
        self.assertLess(len(data), len(json.dumps(sj.scene_payload(scene))))

    def test_binary_frames_ship_only_what_changed(self):
        scene, view = self.scene('Head'), sj.ViewState()
        first = sj.scene_payload(scene, view, 'binary')
        whole, _ = sj.scene_binary(scene, view)
        sj.step_stage(view, 1)
        data, manifest = sj.scene_binary(scene, view, first['cursor'])
        frame = sj.scene_frame(scene, first['cursor'], view, 'binary')
        self.assertEqual({array['name'] for array in refs(frame)}, set(manifest))
        self.assertTrue(manifest)
        self.assertEqual(sum(4 * int(np.prod(item['shape'])) for item in manifest.values()), len(data))
        self.assertEqual(sj.scene_binary(scene, view, frame['cursor']), (b'', {}))
        self.assertGreater(len(whole), 0)


class Actions(Scenes):
    def test_unknown_and_malformed_actions_are_refused(self):
        scene, view = self.scene(), sj.ViewState()
        for action in ({'type': 'fly'}, {}, 'level', None):
            with self.assertRaises(ValueError):
                sj.apply_action(scene, view, action)
        with self.assertRaises(ValueError) as caught:
            sj.apply_action(scene, view, {'type': 'level'})
        self.assertIn('level', str(caught.exception))
        for action in ({'type': 'level', 'value': 'Run'}, {'type': 'mode', 'value': 'Fly'}, {'type': 'show', 'what': 'bias'}):
            with self.assertRaises(ValueError):
                sj.apply_action(scene, view, action)
        self.assertEqual((scene.level, scene.mode, view.tile_kind), ('Model', 'Build', 'weight'))

    def test_level_mode_and_scene_message(self):
        scene, view = self.scene(), sj.ViewState()
        for level in sj.VIEWER_LEVELS:
            sj.apply_action(scene, view, {'type': 'level', 'value': level})
            self.assertEqual(scene.level, level)
        for mode in MODES:
            sj.apply_action(scene, view, {'type': 'mode', 'value': mode})
            self.assertEqual(scene.mode, mode)
        sj.apply_action(scene, view, {'type': 'scene', 'level': 'Head', 'mode': 'Train', 'selected': {'layer': 2}})  # inside3d's message
        self.assertEqual((scene.level, scene.mode, scene.selection.layer), ('Head', 'Train', 2))
        sj.apply_action(scene, view, {'type': 'scene', 'level': 'Run', 'mode': 'Fly'})  # what the viewer cannot draw is left alone
        self.assertEqual((scene.level, scene.mode), ('Head', 'Train'))

    def test_selection_moves_stay_in_range(self):
        scene, view = self.scene(), sj.ViewState()
        sj.next_block(scene)
        sj.next_block(scene)
        sj.next_block(scene)
        self.assertEqual(scene.selection.layer, 0)  # wraps
        sj.select_token(scene, -5)
        self.assertEqual(scene.selection.token, 0)
        sj.select_token(scene, 999)
        self.assertEqual(scene.selection.token, scene.spec['dims']['n_ctx'] - 1)
        sj.select_head(scene, -1)
        self.assertGreaterEqual(scene.selection.head, 0)
        sj.pan_tiles(scene, -3, 4)
        self.assertEqual((scene.selection.row, scene.selection.column), (0, 4))
        sj.toggle_motion(scene)
        self.assertTrue(scene.reduce_motion)
        sj.apply_action(scene, view, {'type': 'selection', 'selection': {'token': 3}})
        self.assertEqual(scene.selection.token, 3)
        with self.assertRaises(ValueError):
            sj.apply_action(scene, view, {'type': 'selection', 'selection': {'layer': -1}})

    def test_the_matrix_follows_the_block(self):
        scene = self.scene('Weights')
        names = sj.weight_names(scene)
        sj.read_snapshot(scene, 'a.npz', self.load())
        self.assertEqual(scene.selection.weight, sj.pick_matrix(names, 0, None))
        self.assertTrue(scene.selection.weight.startswith('block0.'))
        sj.select(scene, layer=1)
        self.assertIn(scene.selection.weight, sj.block_matrices(names, 1))
        self.assertTrue(scene.selection.weight.startswith('block1.'))
        payload = sj.scene_payload(scene)
        self.assertEqual(payload['view']['controls']['matrices'], sj.block_matrices(names, 1))
        self.assertEqual(payload['groups']['matrix']['weight_name'], scene.selection.weight)

    def test_playback_steps_through_the_snapshots(self):
        scene, view = self.scene('Model', snapshot=False), sj.ViewState()
        sj.toggle_play(scene, view)
        self.assertFalse(view.playing)
        self.assertEqual(view.note, 'Nothing to play yet: load snapshots first.')
        sj.load_replay(scene, view, [], self.load())
        self.assertEqual(view.note, 'This version has no training snapshots yet.')
        sj.load_replay(scene, view, ['a.npz', 'b.npz', 'c.npz'], self.load())
        playback = sj.scene_payload(scene, view)['playback']
        self.assertEqual((playback['position'], playback['previous'], playback['next'], playback['play']), ('Snapshot 1 of 3', False, True, 'Play'))
        sj.toggle_play(scene, view)
        self.assertEqual(sj.scene_payload(scene, view)['playback']['play'], 'Pause')
        sj.advance_playback(scene, view, self.load())
        sj.advance_playback(scene, view, self.load())
        self.assertEqual((view.snapshot_index, view.playing), (2, True))
        sj.advance_playback(scene, view, self.load())  # after the last: stop
        self.assertEqual((view.snapshot_index, view.playing), (2, False))
        sj.toggle_play(scene, view, self.load())  # at the end, Play starts over
        self.assertEqual((view.snapshot_index, view.playing), (0, True))
        sj.scrub(scene, view, -5, self.load())
        sj.scrub(scene, view, 9, self.load())
        self.assertEqual(view.snapshot_index, 2)
        sj.apply_action(scene, view, {'type': 'scrub', 'index': 1}, self.load())
        self.assertEqual(view.snapshot_index, 1)

    def test_a_bad_snapshot_is_said_and_the_scene_carries_on(self):
        scene, view = self.scene('Model'), sj.ViewState()

        def broken(path):
            raise OSError('truncated')

        sj.add_snapshot(scene, view, 'bad.npz', broken)
        self.assertEqual((view.snapshot_paths, view.note), ([], 'Could not load the snapshot: truncated'))
        self.assertIsNotNone(scene.snapshot)
        self.assertEqual(sj.scene_payload(scene, view)['stats']['note'], view.note)

    def test_bad_values_from_a_client_are_value_errors(self):
        scene, view = self.scene('Window'), sj.ViewState()
        bad = [{'type': 'token', 'delta': 'a'}, {'type': 'selection', 'selection': 5}, {'type': 'replay', 'paths': 5},
               {'type': 'summary', 'attention': 5}, {'type': 'summary', 'attention': {'block0.attn.pattern': [1, 2]}},
               {'type': 'summary', 'edges': 'x'}, {'type': 'stats', 'step': 'x'}, {'type': 'cell', 'row': 'a'},
               {'type': 'spec', 'spec': 5}, {'type': 'scrub', 'delta': None}]
        for action in bad:
            with self.assertRaises(ValueError, msg=action):
                sj.apply_action(scene, view, action, self.load())
        self.assertEqual(sj.scene_payload(scene, view)['view']['level'], 'Window')  # nothing was half applied
        self.assertEqual((view.stats, scene.summaries), ({}, {}))

    def test_a_late_advance_after_pause_skips_nothing(self):
        scene, view = self.scene('Model', snapshot=False), sj.ViewState()
        sj.load_replay(scene, view, ['a.npz', 'b.npz'], self.load())
        sj.apply_action(scene, view, {'type': 'advance'}, self.load())
        self.assertEqual((view.snapshot_index, view.playing), (0, False))
        sj.toggle_play(scene, view)
        sj.apply_action(scene, view, {'type': 'advance'}, self.load())
        self.assertEqual(view.snapshot_index, 1)

    def test_scrubbing_to_a_bad_file_stays_put(self):
        scene, view = self.scene('Model', snapshot=False), sj.ViewState()
        sj.load_replay(scene, view, ['a.npz', 'gone.npz'], self.load())
        sj.toggle_play(scene, view)

        def missing(path):
            raise FileNotFoundError(2, 'No such file or directory', '/home/someone/runs/gone.npz')

        sj.scrub(scene, view, 1, missing)
        self.assertEqual((view.snapshot_index, view.playing), (0, False))  # the position says what is shown
        self.assertEqual(view.note, 'Could not load the snapshot: No such file or directory')  # not the server's path
        self.assertEqual(sj.scene_payload(scene, view)['playback']['position'], 'Snapshot 1 of 2')

    def test_a_mixer_swap_is_forwarded_or_made_here(self):
        scene, view = self.scene('Window', snapshot=False), sj.ViewState()
        current = scene.gates[0]['entry']['mixer']['id']
        self.assertIsNone(sj.apply_action(scene, view, {'type': 'swap', 'module': current}))
        message = sj.apply_action(scene, view, {'type': 'swap', 'module': 'gru'})
        self.assertEqual(message, {'type': 'module_swap', 'block_id': scene.gates[0]['id'], 'slot': 'mixer', 'entry': {'id': 'gru'}})
        self.assertEqual(scene.gates[0]['entry']['mixer']['id'], current)  # the studio saves it and sends the design back
        self.assertIn('Saving block 1 with gru', view.note)
        demo = sj.ViewState(linked=False)
        first = sj.scene_payload(scene, demo)
        self.assertIsNone(sj.apply_action(scene, demo, {'type': 'swap', 'module': 'gru'}))
        self.assertEqual(scene.gates[0]['entry']['mixer']['id'], 'gru')
        frame = sj.scene_frame(scene, first['cursor'], demo)
        self.assertEqual((sorted(frame['groups']), frame['design']['gates'][0]['mixer']['id']), (['gate:0'], 'gru'))
        sj.apply_action(scene, demo, {'type': 'swap', 'module': 'not-a-mixer'})
        self.assertTrue(demo.note)  # a refusal is said, and the block keeps its mixer
        self.assertEqual(scene.gates[0]['entry']['mixer']['id'], 'gru')

    def test_the_binary_since_is_checked_like_the_frames(self):
        with self.assertRaises(ValueError):
            sj.scene_binary(self.scene(), since=['x'])

    def test_run_one_real_step(self):
        scene, view = self.scene('Model', snapshot=False), sj.ViewState()
        self.assertTrue(sj.apply_action(scene, view, {'type': 'probe_step'}))
        self.assertFalse(sj.apply_action(scene, view, {'type': 'probe_step'}))  # one at a time
        self.assertFalse(sj.begin_probe(sj.ViewState(linked=False)))  # a demo has no studio to run it
        self.assertEqual(sj.scene_payload(scene, view)['view']['controls']['snapshots'], ui.snapshot_buttons(True, True))
        sj.apply_action(scene, view, {'type': 'notice', 'message': 'Still running.'})
        self.assertTrue(view.probing)
        sj.apply_action(scene, view, {'type': 'snapshot', 'path': 'probe.npz'}, self.load(step=1))
        self.assertFalse(view.probing)
        self.assertEqual(view.snapshot_paths, ['probe.npz'])
        self.assertIn('Ran one real step', view.note)
        sj.begin_probe(view)
        sj.apply_action(scene, view, {'type': 'error', 'message': 'No checkpoint.'})
        self.assertEqual((view.probing, view.note), (False, 'No checkpoint.'))

    def test_stats_summary_and_spec_messages(self):
        scene, view = self.scene('Model', snapshot=False), sj.ViewState()
        sj.apply_action(scene, view, {'type': 'stats', 'step': 2, 'total': 4, 'loss': .5})
        self.assertEqual(view.stats, {'step': 2, 'total': 4, 'loss': .5})  # the message type is not a stat
        sj.apply_action(scene, view, {'type': 'summary', 'edges': [0, 32, 64], 'attention': {'block0.attn.pattern': np.ones((1, 2, 2)) / 2}})
        self.assertEqual(scene.summary_edges, [0, 32, 64])
        arcs = entry(sj.scene_payload(scene, view)['groups']['gate:0']['lines'], 'mixer')
        self.assertEqual(arcs['count'], 4)  # two queries, each to two keys
        spec = copy.deepcopy(scene.spec)
        spec['dims']['n_layer'] = 2
        sj.apply_action(scene, view, {'type': 'spec', 'spec': validate_spec(spec), 'selected': scene.gates[1]['id']})
        self.assertEqual((len(scene.gates), scene.selection.layer), (2, 1))

    def test_the_replay_pulse_runs_forward_then_back(self):
        self.assertEqual(sj.pulse_pose(3, 0), (0, 0., True))
        self.assertEqual(sj.pulse_pose(3, .5), (1, 1., True))
        self.assertEqual(sj.pulse_pose(3, 1.5), (2, 3., False))  # turned around at the last block
        self.assertEqual(sj.pulse_pose(3, 2.5), (0, 1., False))
        self.assertEqual(sj.pulse_pose(3, 3), (0, 0., True))  # one pass is pass_seconds, then it starts over
        self.assertEqual(sj.pulse_pose(3, 1.25), (2, 2.5, True))
        self.assertEqual(sj.pulse_pose(3, 1.25, reduce_motion=True), (2, 2, True))  # reduced motion hops from block to block
        self.assertEqual(sj.pass_seconds(3), 3)

    def test_the_pulse_is_described_for_the_renderer(self):
        scene, view = self.scene('Window'), sj.ViewState(snapshot_paths=['a.npz'], snapshot_index=0)
        pulse = sj.scene_payload(scene, view)['playback']['pulse']
        self.assertEqual((pulse['animate'], pulse['blocks'], pulse['spacing']), (False, 3, 3))
        self.assertEqual((pulse['speed'], pulse['z'], pulse['drawn']), (2., 1., True))  # tick: phase = elapsed * 2, setPos(.., 1)
        self.assertEqual((pulse['colors']['forward'], pulse['colors']['backward']), ([1., 1., 1., 1.], [1., .45, .08, 1.]))
        self.assertEqual(pulse['idle_size'], [.3, .4, .3])  # cube_at(..., (.15, .2, .15)): half-extents
        self.assertAlmostEqual(pulse['token_x'], (0 - 0 - (len(scene.frame()['tokens']) - 1) / 2) * .38)
        view.playing = True
        self.assertTrue(sj.scene_payload(scene, view)['playback']['pulse']['animate'])
        for direction in ('forward', 'backward'):
            sizes = pulse['sizes'][direction]
            self.assertEqual(len(sizes), 3)
            self.assertTrue(all(size is None or .2 <= size <= 1. for size in sizes))
        self.assertTrue(any(size is not None for size in pulse['sizes']['forward']))
        scene.level = 'Head'
        self.assertFalse(sj.scene_payload(scene, view)['playback']['pulse']['animate'])  # only the block levels have a pulse
        self.assertFalse(sj.scene_payload(scene, view)['playback']['pulse']['drawn'])  # and a client can tell paused from undrawable
        self.assertEqual(sj.scene_payload(scene, view)['playback']['pulse']['sizes'], {})
        scene.level, scene.mode = 'Model', 'Ride a token'
        pulse = sj.scene_payload(scene, view)['playback']['pulse']
        self.assertEqual((pulse['follow'], pulse['token_x']), (True, 0.))


class ViewerParity(Scenes):
    def test_palette_is_the_viewers(self):
        self.assertEqual(sj.VIEWER_LEVELS, ui.LEVELS)
        for name, value in sj.THEME.items():
            self.assertEqual(value, ui.C[name], name)
        self.assertEqual(set(ui.SCENE) - set(sj.KINDS), set())
        for kind, colour in ui.SCENE.items():
            self.assertEqual(ui.rgba(sj.PALETTE[kind]['color'], sj.PALETTE[kind]['alpha']), colour, kind)
        for kind in sj.KINDS:
            self.assertIn(kind, sj.PALETTE)

    def test_every_colour_the_scene_uses_is_in_the_palette(self):
        for level, snapshot in itertools.product(sj.VIEWER_LEVELS, (False, True)):
            payload = sj.scene_payload(self.scene(level, snapshot))
            for group in payload['groups'].values():
                for item in group['boxes'] + group['lines']:
                    self.assertIn(item['kind'], payload['palette'])
                for label in group['labels']:
                    self.assertIn(label['kind'], payload['palette'])

    def test_control_decisions_equal_the_viewers(self):
        for level, mode, matrices in itertools.product(ui.LEVELS, MODES, ([], ['x'])):
            self.assertEqual(sj.view_rows(level, mode, matrices), ui.view_rows(level, mode, matrices))
        for count, index, playing in ((0, -1, False), (3, 0, False), (3, 1, True), (3, 2, True)):
            self.assertEqual(sj.playback_state(count, index, playing), ui.playback(count, index, playing))
        for linked, probing in itertools.product((False, True), repeat=2):
            self.assertEqual(sj.snapshot_buttons(linked, probing), ui.snapshot_buttons(linked, probing))
        for level, snapshots, orbit in itertools.product(ui.LEVELS, ([], ['a']), (False, True)):
            self.assertEqual(sj.toggle_names(level, snapshots, orbit), ui.toggles(level, snapshots, orbit))
        for level in ui.LEVELS:
            self.assertEqual(sj.matrix_legend(level), ui.matrix_legend(level))
            self.assertEqual(sj.camera_view(level, 6, 16), ui.camera_view(level, 6, 16))
            for snapshot, heads in itertools.product((False, True), repeat=2):
                self.assertEqual(sj.empty_matrix_note(level, snapshot, 1, 'gru', heads), ui.empty_matrix_note(level, snapshot, 1, 'gru', heads))
        names = ['block0.f1.w', 'block0.qkv.w', 'block1.f1.w', 'block10.f1.w', 'position', 'token']
        for layer, current in itertools.product((0, 1, 10, 3), (None, 'token', 'block0.qkv.w', 'block1.f1.w')):
            self.assertEqual(sj.block_matrices(names, layer), ui.block_matrices(names, layer))
            self.assertEqual(sj.pick_matrix(names, layer, current), ui.pick_matrix(names, layer, current))
        self.assertIsNone(sj.pick_matrix([], 0, None))
        for blocks in (0, 1, 12):
            self.assertEqual(sj.pass_seconds(blocks), ui.pass_seconds(blocks))

    def test_the_controls_in_the_payload_follow_the_level(self):
        for level, mode in itertools.product(sj.VIEWER_LEVELS, MODES):
            scene = self.scene(level)
            scene.mode = mode
            controls = sj.scene_payload(scene, sj.ViewState(snapshot_paths=['a.npz']))['view']['controls']
            self.assertEqual(controls['rows'], ui.view_rows(level, mode, controls['matrices']))
            self.assertEqual(controls['toggles'], ui.toggles(level, ['a.npz'], False))
        self.assertEqual(sj.scene_payload(self.scene('Weights', False))['view']['controls']['rows'], ['level', 'mode', 'mixer', 'tile'])


if __name__ == '__main__':
    unittest.main()
