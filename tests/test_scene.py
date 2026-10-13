import copy
import unittest

import numpy as np

from studio_scene import SceneModel, Selection
from studio_snapshot import one_step
from studio_compile import CompiledModel
from studio_spec import default_spec, validate_spec
from studio_summary import bin_edges


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
        np.testing.assert_array_equal(head['pattern'], snapshot['arrays']['hook::block0.attn.pattern'][0, 0])  # rows = queries
        np.testing.assert_array_equal(head['pattern'], scene.frame('Window')['patterns']['block0.attn.pattern'][0])
        self.assertTrue(head['mask'][0][2])  # query 0 cannot see key 2
        self.assertFalse(head['mask'][2][0])
        for key, value in scene.replay_values():
            np.testing.assert_array_equal(value, snapshot['arrays']['forward::' + key])
        first = scene.frame('Weights')
        self.assertLessEqual(len(first['weight']), 16)
        self.assertLessEqual(len(first['weight'][0]), 16)
        np.testing.assert_array_equal(first['weight'], scene.frame('Weights')['weight'])

    def test_summary_edges_travel_with_short_sequence(self):
        """Summary edges must match the sequence length used to create the summary, not n_ctx."""
        spec = self.spec()
        spec['dims']['n_ctx'] = 256  # Large context
        spec['dims']['n_layer'] = 1

        model = CompiledModel(spec, 19, device='cpu')

        # Create a short sequence (much shorter than n_ctx=256)
        short_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.int64)  # 10 tokens

        # Create summaries using attention_summary with 16 bins
        logits_summary, summaries = model.forward(short_ids, attention_summary=16)

        # The edges should be computed from the actual sequence length (10 tokens)
        actual_edges = bin_edges(short_ids.shape[1], 16)

        # Create a scene and set the summary with edges (proper usage)
        scene = SceneModel(spec)
        scene.summaries = {key: value for key, value in summaries.items() if key.endswith('attn.pattern')}
        scene.stream_norms = {key: value for key, value in summaries.items() if key.endswith('resid.norm')}
        # Set summary_edges with the actual edges used to create the summary
        scene.summary_edges = actual_edges

        # Get the frame for Model level
        frame = scene.frame('Model')
        frame_edges = np.array(frame['edges'])

        # After the fix, frame edges should match the actual edges from the summary
        np.testing.assert_array_equal(frame_edges, actual_edges,
                                      err_msg="Frame edges should match the edges used to create the summary")
