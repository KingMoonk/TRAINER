import copy
import unittest

import numpy as np

from studio_scene import SceneModel, Selection
from studio_snapshot import one_step
from studio_compile import CompiledModel
from studio_spec import default_spec, validate_spec


class Scenes(unittest.TestCase):
    def spec(self):
        spec = default_spec()
        spec['dims'].update(d_model=16, n_layer=3, n_ctx=256, dropout=0.)
        return validate_spec(spec)

    def test_gate_identity_and_glyphs_after_single_swap(self):
        scene = SceneModel(self.spec())
        gates = list(scene.gates)
        changed = scene.swap(1, 'mixer', {'id': 'gru'})
        self.assertEqual(changed, [1])
        self.assertIs(scene.gates[0], gates[0])
        self.assertIs(scene.gates[2], gates[2])
        self.assertEqual(scene.gates[1]['glyphs']['mixer']['kind'], 'ribbon')

    def test_zoom_retains_selection_and_bounded_geometry(self):
        scene = SceneModel(self.spec(), Selection(run='run1', layer=1, token=200, head=2))
        selected = copy.deepcopy(scene.selection)
        scene.paths = [{'run': 'run1', 'points': list(zip(range(5000), [1.] * 5000))}]
        self.assertEqual(len(scene.frame('Run')['paths'][0]['points']), 2000)
        self.assertLessEqual(len(scene.frame('Model')['edges']), 33)
        self.assertLessEqual(len(scene.frame('Window')['tokens']), 32)
        self.assertEqual(scene.selection, selected)

    def test_snapshot_replay_exact_and_head_axis_orientation(self):
        spec = self.spec()
        spec['dims']['n_layer'] = 1
        model = CompiledModel(spec, 9, device='cpu')
        snapshot = one_step(model, [[1, 2, 3]], [[2, 3, 4]])
        scene = SceneModel(spec)
        scene.load_snapshot(snapshot)
        head = scene.frame('Head')
        np.testing.assert_array_equal(head['pattern'], snapshot['arrays']['hook::block0.attn.pattern'][0, 0].T)
        self.assertTrue(head['mask'][2][0])
        for key, value in scene.replay_values():
            np.testing.assert_array_equal(value, snapshot['arrays']['forward::' + key])
        first = scene.frame('Weights')
        self.assertLessEqual(len(first['weight']), 16)
        self.assertLessEqual(len(first['weight'][0]), 16)
        np.testing.assert_array_equal(first['weight'], scene.frame('Weights')['weight'])
