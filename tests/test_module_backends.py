import dataclasses
import unittest

import numpy as np

import studio_registry as registry
from studio_backend import get_backend
from studio_slots import Ctx, LAYER_SLOTS
from tests.test_module_contract import fixture


class ModuleBackends(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.backend = get_backend('cuda')
        except (RuntimeError, ValueError, ImportError) as error:
            raise unittest.SkipTest(str(error)) from error

    def test_forward_hooks_and_gradients_match_numpy(self):
        xp = self.backend.xp
        for module in registry.every():
            if module.slot not in LAYER_SLOTS or module.skip:
                continue
            with self.subTest(module=module.id):
                cfg, cpu, embedding, x, dy = fixture(module)
                dims = dataclasses.replace(cpu.dims, xp=xp, dtype=xp.float64)
                gpu = module.build(cfg, dims, np.random.default_rng(11))
                for key, value in cpu.params.items():
                    gpu.params[key][...] = xp.asarray(value)
                gpu_embedding = None
                if embedding is not None:
                    gpu_embedding = registry.get('embedding', 'learned').build(None, dims, None)
                    for key, value in embedding.params.items():
                        gpu_embedding.params[key][...] = xp.asarray(value)
                ctx_cpu, ctx_gpu = Ctx(np, hooks=True, embedding=embedding), Ctx(xp, hooks=True, embedding=gpu_embedding)
                out_cpu, out_gpu = cpu.forward(x, ctx_cpu), gpu.forward(xp.asarray(x), ctx_gpu)
                np.testing.assert_allclose(self.backend.to_cpu(out_gpu), out_cpu, atol=1e-5, rtol=1e-5)
                for key, value in ctx_cpu.captured.items():
                    np.testing.assert_allclose(self.backend.to_cpu(ctx_gpu.captured[key]), value, atol=1e-5, rtol=1e-5)
                dx_cpu, dx_gpu = cpu.backward(dy), gpu.backward(xp.asarray(dy))
                if dx_cpu is not None:
                    np.testing.assert_allclose(self.backend.to_cpu(dx_gpu), dx_cpu, atol=1e-5, rtol=1e-5)
                for key, value in cpu.grads.items():
                    np.testing.assert_allclose(self.backend.to_cpu(gpu.grads[key]), value, atol=1e-5, rtol=1e-5)
