import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import studio_landscape as landscape
from studio_compile import CompiledModel
from studio_spec import default_spec, validate_spec
from studio_ia.services import FixtureServices
from tests.torchcheck import to_np


def tiny_model():
    spec = default_spec()
    spec['dims'].update(d_model=16, n_layer=1, n_ctx=32, dropout=0.)
    return CompiledModel(validate_spec(spec), 12, device='cpu')


def weights_of(model):
    return {key: to_np(value).copy() for key, value in model.p.items()}


class Directions(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.weights = weights_of(self.model)
        self.roles = self.model.roles

    def test_every_filter_matches_the_weights_norm(self):
        d1, d2 = landscape.random_directions(self.weights, self.roles, seed=3)
        checked = 0
        for direction in (d1, d2):
            for name, weight in self.weights.items():
                axis = landscape.filter_axis(self.roles.get(name), weight.ndim)
                if axis is None:
                    self.assertFalse(direction[name].any(), name)
                    continue
                want = np.linalg.norm(weight, axis=tuple(i for i in range(weight.ndim) if i != axis))
                got = np.linalg.norm(direction[name], axis=tuple(i for i in range(weight.ndim) if i != axis))
                np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6, err_msg=name)
                checked += 1
        self.assertGreater(checked, 4)

    def test_same_seed_same_directions_and_pinned_norms_hold(self):
        a = landscape.random_directions(self.weights, self.roles, seed=1)
        b = landscape.random_directions(self.weights, self.roles, seed=1)
        c = landscape.random_directions(self.weights, self.roles, seed=2)
        key = next(k for k, v in self.weights.items() if v.ndim == 2)
        np.testing.assert_array_equal(a[0][key], b[0][key])
        self.assertFalse(np.allclose(a[0][key], c[0][key]))
        norms = landscape.filter_norms(self.weights, self.roles)
        moved = {k: v * 1.5 for k, v in self.weights.items()}
        pinned = landscape.random_directions(moved, self.roles, seed=1, norms=norms)
        np.testing.assert_allclose(pinned[0][key], a[0][key], rtol=1e-5, atol=1e-7)

    def test_pca_puts_the_farthest_snapshot_at_one_and_recovers_coordinates(self):
        rng = np.random.default_rng(0)
        center = {'w': rng.standard_normal((6, 5))}
        u, v = rng.standard_normal((6, 5)), rng.standard_normal((6, 5))
        points = [(1., 0.), (0., 1.), (-1., .5), (.5, -1.)]
        others = [{'w': center['w'] + 2 * a * u + 3 * b * v} for a, b in points]
        d1, d2 = landscape.pca_directions(center, others)
        for other in others:
            a, b = landscape.coordinates(center, other, d1, d2)
            self.assertLessEqual(max(abs(a), abs(b)), 1 + 1e-5)
        self.assertAlmostEqual(max(abs(landscape.coordinates(center, o, d1, d2)[0]) for o in others), 1., places=4)

    def test_pca_needs_a_plane(self):
        center = {'w': np.zeros((3, 3))}
        with self.assertRaisesRegex(ValueError, 'at least 3'):
            landscape.pca_directions(center, [{'w': np.ones((3, 3))}])
        line = [{'w': np.full((3, 3), t)} for t in (1., 2., 3.)]
        with self.assertRaisesRegex(ValueError, 'one line'):
            landscape.pca_directions(center, line)


class Grid(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.weights = weights_of(self.model)
        self.ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]])
        self.targets = np.array([[2, 3, 4, 5, 6, 7, 8, -100], [3, 4, 5, 6, 7, 8, 9, -100]])

    def test_centre_is_the_models_loss_and_the_model_is_restored(self):
        want = landscape.evaluate(self.model, self.ids, self.targets)
        out = landscape.compute(self.model, self.ids, self.targets, self.weights, [], grid=5, span=.5)
        n = len(out['axis'])
        self.assertEqual((n, len(out['loss'])), (5, 5))
        self.assertAlmostEqual(out['loss'][n // 2][n // 2], want, places=5)
        self.assertAlmostEqual(out['current']['loss'], want, places=5)
        self.assertEqual(out['axis'][n // 2], 0.)
        self.assertEqual(out['evaluations'], 25)
        for key, value in self.weights.items():
            np.testing.assert_array_equal(to_np(self.model.p[key]), value)
        self.assertLessEqual(out['minimum']['loss'], want + 1e-9)

    def test_trajectory_points_carry_their_own_loss(self):
        other = {k: v + .01 for k, v in self.weights.items()}
        out = landscape.compute(self.model, self.ids, self.targets, self.weights, [(7, other)], grid=5)
        self.assertEqual(out['trajectory'][0]['step'], 7)
        self.model.load_weights(other)
        self.assertAlmostEqual(out['trajectory'][0]['loss'], landscape.evaluate(self.model, self.ids, self.targets), places=5)

    def test_stop_cancels_and_restores(self):
        class Set:
            def is_set(self):
                return True
        with self.assertRaises(InterruptedError):
            landscape.compute(self.model, self.ids, self.targets, self.weights, [], grid=5, stop=Set())
        for key, value in self.weights.items():
            np.testing.assert_array_equal(to_np(self.model.p[key]), value)


class Size(unittest.TestCase):
    def test_grid_is_odd_and_clamped(self):
        self.assertEqual([landscape.grid_size(n) for n in (1, 6, 21, 99)], [5, 7, 21, 41])

    def test_too_big_is_a_sentence(self):
        landscape.check_size(10_000, 21)
        with self.assertRaisesRegex(ValueError, 'too much work'):
            landscape.check_size(50_000_000, 41)
        with self.assertRaisesRegex(ValueError, 'more than'):
            landscape.check_size(landscape.MAX_PARAMETERS + 1, 5)


class WorkspaceLandscape(unittest.TestCase):
    """WorkspaceServices.inspect_landscape over the fixture workspace's real trained run, with viz snapshots written beside it."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.svc = FixtureServices(cls.tmp.name)
        cls.version = cls.svc.versions()[0]
        run = Path(cls.version.runs[0])
        step, weights = landscape.read_weights(run / "best.npz")
        (run / "viz").mkdir()
        rng = np.random.default_rng(0)
        for index in range(4):
            moved = {k: v + 0.02 * (index + 1) * rng.standard_normal(v.shape).astype(v.dtype) for k, v in weights.items()}
            np.savez(run / "viz" / f"step_{index + 1:08d}.npz", metadata=json.dumps({"step": index + 1}),
                     **{f"weight::{k}": v for k, v in moved.items()})

    @classmethod
    def tearDownClass(cls):
        cls.svc.close()
        cls.tmp.cleanup()

    def test_centres_on_the_best_checkpoint_and_measures_the_held_back_loss(self):
        out = self.svc.inspect_landscape(self.version.number, grid=5)
        n = out["grid"]
        self.assertEqual(out["center"]["name"], "best.npz")
        self.assertEqual(out["loss"][n // 2][n // 2], out["current"]["loss"])
        self.assertEqual(len(out["trajectory"]), 4)
        self.assertGreater(out["batch"]["tokens"], 0)
        self.assertEqual(out["cost"]["evaluations"], 25 + 4)
        again = self.svc.inspect_landscape(self.version.number, grid=5)
        self.assertIs(again, out)  # the same request is answered from the cache

    def test_pca_directions_put_the_snapshots_in_the_plane(self):
        out = self.svc.inspect_landscape(self.version.number, mode="pca", grid=5)
        self.assertEqual(out["unit"], "trajectory")
        self.assertTrue(all(abs(p["x"]) <= 1.01 and abs(p["y"]) <= 1.01 for p in out["trajectory"]))

    def test_a_live_frame_is_small_random_and_keeps_its_directions(self):
        first = self.svc.inspect_landscape(self.version.number, mode="pca", grid=41, live=True, new_anchor=True)
        second = self.svc.inspect_landscape(self.version.number, grid=41, live=True)
        self.assertEqual((first["mode"], first["grid"]), ("random", 11))
        self.assertEqual(first["loss"], second["loss"])  # same weights, same directions, same batch
        self.assertEqual(first["center"]["name"], "step_00000004.npz")

    def test_a_snapshot_can_be_the_centre(self):
        out = self.svc.inspect_landscape(self.version.number, snapshot=1, grid=5)
        self.assertEqual(out["center"]["name"], "step_00000002.npz")
        with self.assertRaisesRegex(ValueError, "Choose a snapshot"):
            self.svc.inspect_landscape(self.version.number, snapshot=9, grid=5)


if __name__ == '__main__':
    unittest.main()
