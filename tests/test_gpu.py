"""GPU checks run on whichever GPU torch can use (CUDA or Apple MPS) and skip cleanly elsewhere."""
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from studio_backend import get_backend, gpu_status
from studio_core import normalize
from tests.torchcheck import to_np
from studio_model import AdamW, DEFAULT_CONFIG, LocalModel, ScratchTransformer, load_checkpoint, train


def tiny_config(device, **values):
    config = dict(DEFAULT_CONFIG, device=device, width=32, layers=1, heads=2,
                  context=32, expansion=2, dropout=0., batch_size=4, steps=45,
                  learning_rate=.004, warmup=0, validation=0., eval_every=15,
                  objective="assistant")
    return dict(config, **values)


class BackendSelectionChecks(unittest.TestCase):
    def test_auto_falls_back_but_explicit_cuda_reports_failure(self):
        unavailable = {"available": False, "kind": "", "name": "", "reason": "CUDA unavailable for this test", "memory_bytes": 0}
        with patch("studio_backend.gpu_status", return_value=unavailable):
            backend = get_backend("auto")
            self.assertEqual(backend.device, "cpu")
            self.assertIs(backend.xp, torch)
            self.assertIn("CUDA unavailable", backend.reason)
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                get_backend("cuda")

    def test_explicit_cpu_does_not_probe_cuda(self):
        with patch("studio_backend.gpu_status", side_effect=AssertionError("CPU must not need CUDA")):
            backend = get_backend("cpu")
            self.assertEqual(backend.device, "cpu")
            values = np.array([1., 2.], dtype=np.float32)
            np.testing.assert_array_equal(backend.to_cpu(values), values)
            np.testing.assert_array_equal(backend.to_cpu(torch.as_tensor(values)), values)
            backend.synchronize()

    def test_concurrent_first_calls_probe_one_at_a_time(self):
        import studio_backend
        inside, most, gate = [0], [0], threading.Event()

        def probe():
            inside[0] += 1
            most[0] = max(most[0], inside[0])
            gate.wait(.05)  # long enough for every other thread to arrive while this one probes
            inside[0] -= 1
            return {"available": False, "kind": "", "name": "", "reason": "probe for this test", "memory_bytes": 0}

        studio_backend.gpu_status.cache_clear()
        try:
            with patch("studio_backend._probe_gpu", side_effect=probe), ThreadPoolExecutor(8) as pool:
                results = list(pool.map(lambda _: studio_backend.gpu_status(), range(8)))
        finally:
            studio_backend.gpu_status.cache_clear()
        self.assertEqual(most[0], 1)
        self.assertTrue(all(result["reason"] == "probe for this test" for result in results))


class GpuIntegrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        status = gpu_status()
        if not status["available"]:
            raise unittest.SkipTest(status.get("reason") or "No GPU is available")
        cls.kind = status["kind"]

    def test_forward_gradients_and_adamw_match_cpu(self):
        cpu = ScratchTransformer(tiny_config("cpu"), 16)
        gpu = ScratchTransformer(tiny_config(self.kind), 16)
        self.assertEqual(gpu.backend.device, self.kind)
        self.assertIsInstance(gpu.p["token"], torch.Tensor)
        self.assertEqual(gpu.p["token"].device.type, self.kind)
        for key in cpu.p:
            np.testing.assert_array_equal(to_np(cpu.p[key]), gpu.backend.to_cpu(gpu.p[key]), err_msg=key)
        x = np.array([[1, 3, 8, 2, 4, 9], [1, 3, 10, 2, 4, 11]])
        y = np.array([[-100, -100, -100, -100, 9, 2], [-100, -100, -100, -100, 11, 2]])
        np.testing.assert_allclose(to_np(cpu.forward(x)), gpu.backend.to_cpu(gpu.forward(x)), rtol=1e-3, atol=1e-5)
        cpu_loss, gpu_loss = cpu.loss_and_backward(x, y), gpu.loss_and_backward(x, y)
        self.assertAlmostEqual(cpu_loss, gpu_loss, delta=1e-4)
        for key in cpu.g:
            np.testing.assert_allclose(to_np(cpu.g[key]), gpu.backend.to_cpu(gpu.g[key]), rtol=1e-3, atol=1e-5, err_msg=key)
        cpu_optimizer, gpu_optimizer = AdamW(cpu), AdamW(gpu)
        self.assertEqual(gpu_optimizer.m["token"].device.type, self.kind)
        cpu_norm = cpu_optimizer.update(cpu, .001)
        gpu_norm = gpu_optimizer.update(gpu, .001)
        self.assertAlmostEqual(cpu_norm, gpu_norm, delta=1e-4)
        for key in cpu.p:
            np.testing.assert_allclose(to_np(cpu.p[key]), gpu.backend.to_cpu(gpu.p[key]), rtol=1e-3, atol=1e-5, err_msg=key)

        # Greedy generation from one checkpoint loaded on both devices.
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "trained" / "last.npz"
            train(tiny_config("cpu"), [normalize({"user": "a", "assistant": "b"})], checkpoint.parent)
            cpu_local, gpu_local = LocalModel(checkpoint, device="cpu"), LocalModel(checkpoint, device=self.kind)
            tokens = np.array([[1, 3, 8, 2, 4, 9]])
            cpu_logits = cpu_local.model.backend.to_cpu(cpu_local.model.forward(tokens))
            gpu_logits = gpu_local.model.backend.to_cpu(gpu_local.model.forward(tokens))
            np.testing.assert_allclose(cpu_logits, gpu_logits, rtol=0, atol=1e-3)
            np.testing.assert_array_equal(cpu_logits.argmax(axis=-1), gpu_logits.argmax(axis=-1))
            prompt = [{"role": "user", "content": "a"}]
            self.assertEqual(gpu_local.generate(prompt, max_tokens=8, temperature=0),
                             cpu_local.generate(prompt, max_tokens=8, temperature=0))

    def test_gpu_learning_inference_and_portable_legacy_resume(self):
        examples = [normalize({"user": "a", "assistant": "b"})]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu_directory = root / "gpu-trained"
            config = tiny_config(self.kind)
            train(config, examples, gpu_directory)
            checkpoint_path = gpu_directory / "last.npz"
            trained = load_checkpoint(checkpoint_path)
            self.assertEqual(trained["step"], 45)
            self.assertLess(trained["history"][-1]["loss"], trained["history"][0]["loss"] * .5)
            for group in ("model", "m", "v"):
                self.assertTrue(all(isinstance(value, np.ndarray) for value in trained[group].values()))
            prompt = [{"role": "user", "content": "a"}]
            for device in ("cpu", self.kind):
                local = LocalModel(checkpoint_path, device=device)
                self.assertEqual(local.model.backend.device, device)
            cpu_reply = LocalModel(checkpoint_path, device="cpu").generate(prompt, max_tokens=5, temperature=0)

            # Desktop loading and generation use distinct short-lived workers.
            # Keeping both executors alive guarantees separate OS threads here.
            with ThreadPoolExecutor(max_workers=1) as loader, ThreadPoolExecutor(max_workers=1) as generator:
                load_thread, threaded_model = loader.submit(
                    lambda: (threading.get_ident(), LocalModel(checkpoint_path, device=self.kind))
                ).result(timeout=30)
                generation_thread, reply = generator.submit(
                    lambda: (threading.get_ident(), threaded_model.generate(prompt, max_tokens=5, temperature=0))
                ).result(timeout=30)
                self.assertNotEqual(load_thread, generation_thread)
                self.assertEqual(reply, cpu_reply)

            # A GPU checkpoint can resume on the CPU.
            cpu_directory = root / "cpu-resumed"
            train(tiny_config("cpu", steps=50), examples, cpu_directory, resume=checkpoint_path)
            cpu_path = cpu_directory / "last.npz"
            cpu_saved = load_checkpoint(cpu_path)
            self.assertEqual(cpu_saved["step"], 50)
            self.assertEqual(cpu_saved["optimizer_steps"], 50)
            self.assertFalse(np.array_equal(trained["model"]["token"], cpu_saved["model"]["token"]))

            # Original CPU-only files have no device field. Preserve their version,
            # parameter keys and optimizer state while checking the migration path.
            with np.load(cpu_path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            metadata = json.loads(str(arrays["metadata"].item()))
            metadata["config"].pop("device", None)
            arrays["metadata"] = np.array(json.dumps(metadata))
            legacy_path = root / "legacy-cpu.npz"
            np.savez_compressed(legacy_path, **arrays)
            legacy_local = LocalModel(legacy_path, device=self.kind)
            self.assertEqual(legacy_local.generate(prompt, max_tokens=5, temperature=0),
                             LocalModel(legacy_path, device="cpu").generate(prompt, max_tokens=5, temperature=0))

            # Resume the same CPU state on both devices to verify restored Adam
            # moments affect the next steps consistently, beyond merely loading it.
            for device in ("cpu", self.kind):
                train(tiny_config(device, steps=55), examples, root / f"legacy-{device}", resume=legacy_path)
            cpu_result = load_checkpoint(root / "legacy-cpu" / "last.npz")
            gpu_result = load_checkpoint(root / f"legacy-{self.kind}" / "last.npz")
            self.assertEqual(gpu_result["step"], 55)
            self.assertEqual(gpu_result["optimizer_steps"], 55)
            for key in cpu_result["model"]:
                np.testing.assert_allclose(cpu_result["model"][key], gpu_result["model"][key], rtol=1e-3, atol=1e-5, err_msg=key)


if __name__ == "__main__":
    unittest.main()
