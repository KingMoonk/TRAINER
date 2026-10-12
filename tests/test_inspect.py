import copy
import unittest

import numpy as np

import studio_inspect as inspect
from studio_compile import CompiledModel
from studio_spec import default_spec, validate_spec
from studio_summary import bin_edges, binned_attention


class Inspection(unittest.TestCase):
    def setUp(self):
        spec = default_spec()
        spec['dims'].update(d_model=16, n_layer=2, n_ctx=32, dropout=.1)
        self.model = CompiledModel(validate_spec(spec), 12, device='cpu')
        self.ids = np.array([[1, 2, 3, 1, 4]])

    def test_probes_do_not_change_live_model(self):
        state = copy.deepcopy(self.model.rng.bit_generator.state)
        logits, tape = self.model.forward(self.ids, training=True, tape=True)
        caches = [layer.cache for layer in self.model.layers]
        expected = {key: value.copy() for key, value in self.model.p.items()}
        state = copy.deepcopy(self.model.rng.bit_generator.state)
        inspect.forward(self.model, self.ids)
        inspect.backward(self.model, self.ids, pos=1)
        inspect.ablate(self.model, self.ids)
        inspect.logit_lens(self.model, self.ids)
        self.assertEqual(state, self.model.rng.bit_generator.state)
        for key, value in expected.items():
            np.testing.assert_array_equal(self.model.p[key], value)
            self.assertFalse(np.any(self.model.g[key]))
        for layer, cache in zip(self.model.layers, caches):
            self.assertIs(layer.cache, cache)
        self.model.backward(np.ones_like(logits), tape)

    def test_selected_loss_uses_one_target(self):
        got = inspect.backward(self.model, self.ids, pos=2)
        probabilities = got['probabilities'][0, 2]
        self.assertAlmostEqual(got['loss'], -float(np.log(probabilities[1])), places=6)
        target = np.zeros_like(probabilities)
        target[1] = 1
        np.testing.assert_allclose(got['dlogits'][0, 2], probabilities - target)
        self.assertEqual(np.count_nonzero(got['targets'] != -100), 1)

    def test_logit_lens_last_layer_matches_final_prediction(self):
        got = inspect.logit_lens(self.model, self.ids)[-1]['probabilities']
        logits = self.model.forward(self.ids)
        from studio_modules.ops import softmax
        np.testing.assert_allclose(got, softmax(logits, np), atol=1e-7)

    def test_ablation_matches_direct_patch(self):
        result = inspect.ablate(self.model, self.ids, heads=[(0, 1)])
        def patch(value):
            value = value.copy()
            value[:, 1] = 0
            return value
        logits = self.model.forward(self.ids, patches={'block0.attn.head_out': patch})
        targets = np.roll(self.ids, -1, axis=1)
        targets[:, -1] = -100
        self.assertAlmostEqual(result['heads'][0]['loss'], self.model.objective(logits, targets), places=7)

    def test_binned_attention_matches_dense_and_preserves_output(self):
        dense = inspect.forward(self.model, self.ids)
        got = binned_attention(self.model, self.ids, 3)
        np.testing.assert_allclose(got['logits'], dense['logits'], atol=1e-7)
        edges = got['edges']
        for name, mass in got['attention'].items():
            pattern = dense['hooks'][name]
            expected = np.empty_like(mass)
            for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
                for j, (c, d) in enumerate(zip(edges[:-1], edges[1:])):
                    expected[:, i, j] = pattern[:, :, a:b, c:d].sum(axis=-1).mean(axis=(0, 2))
            np.testing.assert_allclose(mass, expected, atol=1e-7)
            np.testing.assert_allclose(mass.sum(axis=-1), 1, atol=1e-7)

    def test_bins_and_validation(self):
        edges = bin_edges(256, 16)
        self.assertTrue(np.all(np.diff(edges) > 0))
        self.assertLess(edges[1] - edges[0], edges[8] - edges[7])
        np.testing.assert_array_equal(bin_edges(1), [0, 1])
        for ids in ([], [[100]], [[1] * 33]):
            with self.assertRaises(ValueError):
                inspect.forward(self.model, ids)
