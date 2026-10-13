"""Every registered layer gives the same outputs, hooks and gradients on the GPU (CUDA or Apple MPS) as on the CPU."""
import dataclasses
import unittest

import numpy as np
import torch

import studio_registry as registry
from studio_backend import gpu_status
from studio_slots import Ctx, LAYER_SLOTS
from tests.test_module_contract import fixture
from tests.torchcheck import fill, layer_backward, to_np


class ModuleBackends(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        status = gpu_status()
        if not status["available"]:
            raise unittest.SkipTest(status.get("reason") or "No GPU is available")
        cls.kind = status["kind"]
        # Apple GPUs have no float64; compare in float32 there.
        cls.dtype = torch.float32 if cls.kind == "mps" else torch.float64
        cls.tol = 1e-4 if cls.kind == "mps" else 1e-5

    def test_forward_hooks_and_gradients_match_cpu(self):
        tol, kind = self.tol, self.kind
        for module in registry.every():
            if module.slot not in LAYER_SLOTS or module.skip:
                continue
            with self.subTest(module=module.id):
                cfg, cpu, embedding, x, dy = fixture(module)
                dims = dataclasses.replace(cpu.dims, dtype=self.dtype, device=kind)
                gpu = module.build(cfg, dims, np.random.default_rng(11))
                for key, value in cpu.params.items():
                    fill(gpu.params[key], to_np(value))
                gpu_embedding = None
                if embedding is not None:
                    gpu_embedding = registry.get('embedding', 'learned').build(None, dims, None)
                    for key, value in embedding.params.items():
                        fill(gpu_embedding.params[key], to_np(value))
                ctx_cpu = Ctx(hooks=True, embedding=embedding)
                ctx_gpu = Ctx(hooks=True, embedding=gpu_embedding, device=kind)
                x_gpu = torch.as_tensor(x, device=kind, dtype=None if x.dtype.kind == "i" else self.dtype)
                with torch.no_grad():
                    out_cpu, out_gpu = cpu.forward(torch.as_tensor(x), ctx_cpu), gpu.forward(x_gpu, ctx_gpu)
                np.testing.assert_allclose(to_np(out_gpu), to_np(out_cpu), atol=tol, rtol=tol)
                self.assertEqual(set(ctx_gpu.captured), set(ctx_cpu.captured))
                for key, value in ctx_cpu.captured.items():
                    np.testing.assert_allclose(to_np(ctx_gpu.captured[key]), to_np(value), atol=tol, rtol=tol, err_msg=key)
                dx_cpu = layer_backward(cpu, x, dy, embedding=embedding)
                dx_gpu = self.gpu_backward(gpu, x_gpu, dy, gpu_embedding)
                self.assertEqual(dx_cpu is None, dx_gpu is None)
                if dx_cpu is not None:
                    np.testing.assert_allclose(to_np(dx_gpu), to_np(dx_cpu), atol=tol, rtol=tol)
                for key, value in cpu.grads.items():
                    np.testing.assert_allclose(to_np(gpu.grads[key]), to_np(value), atol=tol, rtol=tol, err_msg=key)
                if embedding is not None:
                    for key, value in embedding.grads.items():
                        np.testing.assert_allclose(to_np(gpu_embedding.grads[key]), to_np(value), atol=tol, rtol=tol, err_msg=key)

    def gpu_backward(self, layer, x, dy, embedding=None):
        """tests.torchcheck.layer_backward for a layer living on the GPU: adds parameter gradients, returns dx."""
        layers = [layer] + ([embedding] if embedding is not None else [])
        ctx = Ctx(hooks=True, embedding=embedding, device=self.kind)
        saved = {(id(owner), name): value for owner in layers for name, value in owner.params.items()}
        for owner in layers:
            for name, value in owner.params.items():
                owner.params[name] = value.detach().requires_grad_(True)
        if x.is_floating_point():
            x = x.detach().clone().requires_grad_(True)
        try:
            with torch.enable_grad():
                y = layer.forward(x, ctx)
                total = (y * torch.as_tensor(np.asarray(dy), dtype=y.dtype, device=y.device)).sum()
                inputs = [owner.params[name] for owner in layers for name in owner.params]
                if x.requires_grad:
                    inputs.append(x)
                grads = torch.autograd.grad(total, inputs, allow_unused=True)
        finally:
            for owner in layers:
                for name in owner.params:
                    owner.params[name] = saved[(id(owner), name)]
        index = 0
        with torch.no_grad():
            for owner in layers:
                for name in owner.params:
                    if grads[index] is not None:
                        owner.grads[name].add_(grads[index])
                    index += 1
        return grads[-1].detach() if x.requires_grad else None
