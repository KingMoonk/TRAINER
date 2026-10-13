"""Limits, numerical references and isolation contracts for visualization services."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from studio_compile import CompiledModel
from studio_model import build_optimizer
from studio_scene import SceneModel
from studio_spec import default_spec, validate_spec
from studio_summary import bin_edges, chunked_attention
import studio_snapshot
from tests.torchcheck import to_np


def small_spec(position="none", mixer="attention", n_ctx=128):
    spec = default_spec()
    spec["dims"].update(d_model=16, n_layer=2, n_ctx=n_ctx, dropout=0)
    spec["position"] = {"id": position}
    spec["block"]["mixer"] = {"id": mixer, "config": {"heads": 4}}
    if mixer == "gqa":
        spec["block"]["mixer"]["config"]["kv_heads"] = 2
    return validate_spec(spec)


def dense_bin_mass(pattern, edges):
    result = np.zeros((pattern.shape[1], len(edges) - 1, len(edges) - 1))
    for qi, (q0, q1) in enumerate(zip(edges[:-1], edges[1:])):
        for ki, (k0, k1) in enumerate(zip(edges[:-1], edges[1:])):
            result[:, qi, ki] = pattern[:, :, q0:q1, k0:k1].sum(axis=-1).mean(axis=(0, 2))
    return result


class TrackedTensor(torch.Tensor):
    """Records the shape of every matrix product, so a test can see what attention actually materializes."""
    products = []

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        value = super().__torch_function__(func, types, args, kwargs or {})
        if "matmul" in getattr(func, "__name__", ""):
            cls.products.append(tuple(value.shape))
        return value


class VisualizationLimitTests(unittest.TestCase):
    def test_chunked_attention_never_forms_a_full_context_square(self):
        batch, heads, length, depth, chunk = 1, 2, 129, 8, 16
        rng = np.random.default_rng(8)
        q, k, v = (torch.as_tensor(rng.normal(size=(batch, heads, length, depth))).as_subclass(TrackedTensor) for _ in range(3))
        TrackedTensor.products = []
        output, mass = chunked_attention(q, k, v, object(), n_bins=7, chunk_size=chunk)
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(mass.shape, (heads, 7, 7))
        self.assertTrue(TrackedTensor.products)
        self.assertTrue(all(shape[-2] <= chunk for shape in TrackedTensor.products))
        self.assertTrue(any(shape[-1] == length for shape in TrackedTensor.products))
        self.assertFalse(any(shape[-2:] == (length, length) for shape in TrackedTensor.products))

    def test_alibi_gqa_binned_summary_matches_dense_attention_and_output(self):
        spec = small_spec(position="alibi", mixer="gqa")
        model = CompiledModel(spec, 19, device="cpu")
        ids = np.random.default_rng(13).integers(0, 19, size=(2, 67))
        dense_logits, tape = model.forward(ids, hooks=True, tape=True)
        summary_logits, summaries = model.forward(ids, attention_summary=9)
        edges = bin_edges(ids.shape[1], 9)
        np.testing.assert_allclose(to_np(summary_logits), to_np(dense_logits), rtol=1e-4, atol=1e-6)
        name = "block0.attn.pattern"
        mass = to_np(summaries[name])
        np.testing.assert_allclose(mass, dense_bin_mass(to_np(tape.captured[name]), edges), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(mass.sum(axis=-1), 1, atol=1e-5)

    def test_bins_refine_both_oldest_and_newest_context_ends(self):
        edges = bin_edges(4096, 32)
        widths = np.diff(edges)
        self.assertEqual(edges[0], 0)
        self.assertEqual(edges[-1], 4096)
        self.assertLess(widths[0], widths[len(widths) // 2])
        self.assertLess(widths[-1], widths[len(widths) // 2])
        self.assertTrue(np.all(widths > 0))

    def test_snapshot_roundtrip_version_and_no_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            original = {"schema": studio_snapshot.SCHEMA, "step": 4, "loss": .25,
                        "arrays": {"ids": np.array([[1, 2]], dtype=np.int64),
                                   "logits": np.arange(6, dtype=np.float32).reshape(1, 2, 3)}}
            path = studio_snapshot.save(directory, original)
            loaded = studio_snapshot.load(path)
            np.testing.assert_array_equal(loaded["arrays"]["ids"], original["arrays"]["ids"])
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(json.loads(str(archive["metadata"]))["schema"], studio_snapshot.SCHEMA)
            bad_version = Path(directory) / "bad_version.npz"
            np.savez(bad_version, metadata=np.array(json.dumps({"schema": 999})))
            with self.assertRaisesRegex(ValueError, "Unsupported visualization snapshot schema"):
                studio_snapshot.load(bad_version)
            unsafe = Path(directory) / "unsafe.npz"
            np.savez(unsafe, metadata=np.array(json.dumps({"schema": studio_snapshot.SCHEMA})),
                     payload=np.array([{"would": "require pickle"}], dtype=object))
            with self.assertRaises(ValueError):
                studio_snapshot.load(unsafe)

    def test_one_step_replay_preserves_live_model_optimizer_and_rng(self):
        model = CompiledModel(small_spec(), 17, device="cpu")
        optimizer, _ = build_optimizer(model)
        ids = np.array([[1, 3, 5, 2, 8]], dtype=np.int64)
        targets = np.array([[3, 5, 2, 8, -100]], dtype=np.int64)
        model.forward(ids, training=True, tape=True)
        params = {key: to_np(value).copy() for key, value in model.p.items()}
        grads = {key: to_np(value).copy() for key, value in model.g.items()}
        opt_state = {state: {key: to_np(value).copy() for key, value in getattr(optimizer, state).items()}
                     for state in optimizer.states}
        tensors = dict(model.p)
        rng_state = json.dumps(model.rng.bit_generator.state, sort_keys=True)
        result = studio_snapshot.one_step(model, ids, targets, learning_rate=.002, optimizer=optimizer)
        self.assertTrue(result["exact_step"])
        self.assertEqual(result["step"], 1)
        self.assertTrue(any(key.startswith("forward::") for key in result["arrays"]))
        self.assertTrue(any(key.startswith("backward::") for key in result["arrays"]))
        for key, value in params.items():
            np.testing.assert_array_equal(to_np(model.p[key]), value)
            self.assertIs(model.p[key], tensors[key])
            np.testing.assert_array_equal(to_np(model.g[key]), grads[key])
        for state, values in opt_state.items():
            for key, value in values.items():
                np.testing.assert_array_equal(to_np(getattr(optimizer, state)[key]), value)
        for full, (layer, name) in model._slots.items():
            self.assertIs(layer.params[name], model.p[full])
        self.assertEqual(json.dumps(model.rng.bit_generator.state, sort_keys=True), rng_state)

    def test_scene_limits_selection_and_gate_identity(self):
        scene = SceneModel(small_spec(n_ctx=128))
        scene.choose(layer=1, token=80, head=2)
        retained_gate = scene.gates[1]
        changed = scene.swap(0, "channel", {"id": "none"})
        self.assertEqual(changed, [0])
        self.assertIs(scene.gates[1], retained_gate)
        self.assertEqual((scene.selection.layer, scene.selection.token, scene.selection.head), (1, 80, 2))
        scene.paths = [{"run": "long", "points": list(range(2200))}]
        self.assertEqual(len(scene.frame("Run")["paths"][0]["points"]), 2000)
        scene.load_snapshot({"schema": 1, "step": 1, "loss": None, "arrays": {
            "ids": np.zeros((1, 128), dtype=np.int64),
            "weight::large": np.zeros((24, 37)), "gradient::large": np.ones((24, 37)),
            "update::large": np.full((24, 37), 2.)}})
        window = scene.frame("Window")
        self.assertLessEqual(window["end"] - window["start"], 32)
        weights = scene.frame("Weights")
        self.assertEqual(np.asarray(weights["weight"]).shape, (16, 16))


if __name__ == "__main__":
    unittest.main()
