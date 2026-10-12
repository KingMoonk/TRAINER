"""GPU checks run on a working CUDA installation and skip cleanly elsewhere."""
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from studio_backend import get_backend, gpu_status
from studio_core import normalize
from studio_model import AdamW, DEFAULT_CONFIG, LocalModel, ScratchTransformer, load_checkpoint, train


def tiny_config(device, **values):
    config = dict(DEFAULT_CONFIG, device=device, width=32, layers=1, heads=2,
                  context=32, expansion=2, dropout=0., batch_size=4, steps=45,
                  learning_rate=.004, warmup=0, validation=0., eval_every=15,
                  objective="assistant")
    return dict(config, **values)


class BackendSelectionChecks(unittest.TestCase):
    def test_auto_falls_back_but_explicit_cuda_reports_failure(self):
        unavailable = {"available": False, "name": "", "reason": "CUDA unavailable for this test"}
        with patch("studio_backend.gpu_status", return_value=unavailable):
            backend = get_backend("auto")
            self.assertEqual(backend.device, "cpu")
            self.assertIs(backend.xp, np)
            self.assertIn("CUDA unavailable", backend.reason)
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                get_backend("cuda")

    def test_explicit_cpu_does_not_probe_cuda(self):
        with patch("studio_backend.gpu_status", side_effect=AssertionError("CPU must not need CUDA")):
            backend = get_backend("cpu")
            self.assertEqual(backend.device, "cpu")
            values = np.array([1., 2.], dtype=np.float32)
            np.testing.assert_array_equal(backend.to_cpu(values), values)
            backend.synchronize()


class CudaIntegrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        status = gpu_status()
        if not status["available"]:
            raise unittest.SkipTest(status.get("reason") or "CUDA is unavailable")

    def test_forward_gradients_and_adamw_match_cpu(self):
        cpu = ScratchTransformer(tiny_config("cpu"), 16)
        gpu = ScratchTransformer(tiny_config("cuda"), 16)
        self.assertEqual(gpu.backend.device, "cuda")
        self.assertIsInstance(gpu.p["token"], gpu.xp.ndarray)
        self.assertNotIsInstance(gpu.p["token"], np.ndarray)
        for key in cpu.p:
            np.testing.assert_array_equal(cpu.p[key], gpu.backend.to_cpu(gpu.p[key]), err_msg=key)
        x = np.array([[1, 3, 8, 2, 4, 9], [1, 3, 10, 2, 4, 11]])
        y = np.array([[-100, -100, -100, -100, 9, 2], [-100, -100, -100, -100, 11, 2]])
        np.testing.assert_allclose(cpu.forward(x), gpu.backend.to_cpu(gpu.forward(x)), rtol=2e-5, atol=3e-6)
        cpu_loss, gpu_loss = cpu.loss_and_backward(x, y), gpu.loss_and_backward(x, y)
        self.assertAlmostEqual(cpu_loss, gpu_loss, places=5)
        for key in cpu.g:
            np.testing.assert_allclose(cpu.g[key], gpu.backend.to_cpu(gpu.g[key]), rtol=4e-4, atol=3e-6, err_msg=key)
        cpu_optimizer, gpu_optimizer = AdamW(cpu), AdamW(gpu)
        self.assertIsInstance(gpu_optimizer.m["token"], gpu.xp.ndarray)
        cpu_norm = cpu_optimizer.update(cpu, .001)
        gpu_norm = gpu_optimizer.update(gpu, .001)
        self.assertAlmostEqual(cpu_norm, gpu_norm, places=5)
        for key in cpu.p:
            np.testing.assert_allclose(cpu.p[key], gpu.backend.to_cpu(gpu.p[key]), rtol=5e-4, atol=2e-6, err_msg=key)

    def test_gpu_learning_inference_and_portable_legacy_resume(self):
        examples = [normalize({"user": "a", "assistant": "b"})]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu_directory = root / "gpu-trained"
            config = tiny_config("cuda")
            train(config, examples, gpu_directory)
            checkpoint_path = gpu_directory / "last.npz"
            trained = load_checkpoint(checkpoint_path)
            self.assertEqual(trained["step"], 45)
            self.assertLess(trained["history"][-1]["loss"], trained["history"][0]["loss"] * .5)
            for group in ("model", "m", "v"):
                self.assertTrue(all(isinstance(value, np.ndarray) for value in trained[group].values()))
            prompt = [{"role": "user", "content": "a"}]
            for device in ("cpu", "cuda"):
                local = LocalModel(checkpoint_path, device=device)
                self.assertEqual(local.model.backend.device, device)
                self.assertEqual(local.generate(prompt, max_tokens=5, temperature=0), "b")

            # Desktop loading and generation use distinct short-lived workers.
            # Keeping both executors alive guarantees separate OS threads here.
            with ThreadPoolExecutor(max_workers=1) as loader, ThreadPoolExecutor(max_workers=1) as generator:
                load_thread, threaded_model = loader.submit(
                    lambda: (threading.get_ident(), LocalModel(checkpoint_path, device="cuda"))
                ).result(timeout=30)
                generation_thread, reply = generator.submit(
                    lambda: (threading.get_ident(), threaded_model.generate(prompt, max_tokens=5, temperature=0))
                ).result(timeout=30)
                self.assertNotEqual(load_thread, generation_thread)
                self.assertEqual(reply, "b")

            # A GPU checkpoint can resume with the existing NumPy optimizer on CPU.
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
            legacy_local = LocalModel(legacy_path, device="cuda")
            self.assertEqual(legacy_local.generate(prompt, max_tokens=5, temperature=0), "b")

            # Resume the same CPU state on both devices to verify restored Adam
            # moments affect the next steps consistently, beyond merely loading it.
            for device in ("cpu", "cuda"):
                train(tiny_config(device, steps=55), examples, root / f"legacy-{device}", resume=legacy_path)
            cpu_result = load_checkpoint(root / "legacy-cpu" / "last.npz")
            gpu_result = load_checkpoint(root / "legacy-cuda" / "last.npz")
            self.assertEqual(gpu_result["step"], 55)
            self.assertEqual(gpu_result["optimizer_steps"], 55)
            for key in cpu_result["model"]:
                np.testing.assert_allclose(cpu_result["model"][key], gpu_result["model"][key], rtol=3e-4, atol=3e-6, err_msg=key)


if __name__ == "__main__":
    unittest.main()
