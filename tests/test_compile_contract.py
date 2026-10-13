import copy
import unittest

import numpy as np
import torch

from studio_compile import CompiledModel
from studio_spec import default_spec, validate_spec
from tests.torchcheck import to_np


class CompilerContract(unittest.TestCase):
    def model(self, post=False):
        spec = default_spec()
        spec['dims'].update(d_model=16, n_layer=1, n_ctx=32, dropout=0.)
        spec['train']['precision'] = {'id': 'fp64'}
        spec['block']['norm']['config']['placement'] = 'post' if post else 'pre'
        return CompiledModel(validate_spec(spec), 11, device='cpu')

    def test_two_forward_tapes_preserve_backward_caches(self):
        for post in (False, True):
            with self.subTest(post=post):
                model = self.model(post)
                x = np.array([[1, 2, 1]])
                logits, tape = model.forward(x, tape=True)
                dy = np.random.default_rng(7).normal(size=tuple(logits.shape))
                model.backward(dy, tape)
                expected = {k: to_np(v).copy() for k, v in model.g.items()}
                model.zero_grad()
                model.forward(np.array([[4, 5]]))
                model.telemetry(tape) if tape.captured else None
                model.backward(dy, tape)
                for key, value in expected.items():
                    np.testing.assert_array_equal(to_np(model.g[key]), value, err_msg=key)
                model.backward(dy, tape)
                for key, value in expected.items():
                    np.testing.assert_allclose(to_np(model.g[key]), 2 * value, atol=1e-14, err_msg=key)

    def test_intermediate_patches_change_the_computation(self):
        model = self.model()
        x = np.array([[1, 2, 3]])
        baseline = to_np(model.forward(x))
        for hook in ('attn.q', 'attn.k', 'attn.v', 'attn.head_out', 'mlp.pre', 'mlp.act'):
            with self.subTest(hook=hook):
                changed = to_np(model.forward(x, patches={f'block0.{hook}': lambda v: np.zeros(tuple(v.shape))}))
                self.assertGreater(np.max(np.abs(baseline - changed)), 1e-12)

    def test_post_norm_compiled_parameter_gradients(self):
        model = self.model(post=True)
        x = np.array([[1, 2, 1]])
        logits, tape = model.forward(x, tape=True)
        dy = np.random.default_rng(4).normal(size=tuple(logits.shape))
        model.backward(dy, tape)
        rng = np.random.default_rng(3)
        for key, parameter in model.p.items():
            for flat_index in rng.choice(parameter.numel(), min(3, parameter.numel()), replace=False):
                index = np.unravel_index(flat_index, tuple(parameter.shape))
                old = float(parameter[index])
                with torch.no_grad():
                    parameter[index] = old + 1e-5
                upper = np.sum(to_np(model.forward(x)) * dy)
                with torch.no_grad():
                    parameter[index] = old - 1e-5
                lower = np.sum(to_np(model.forward(x)) * dy)
                with torch.no_grad():
                    parameter[index] = old
                self.assertAlmostEqual(float(model.g[key][index]), (upper - lower) / 2e-5, delta=2e-6, msg=key)

    def test_model_causality_and_rng_unchanged_in_inference(self):
        model = self.model()
        rng = copy.deepcopy(model.rng.bit_generator.state)
        first = to_np(model.forward(np.array([[1, 2, 3, 4]])))
        second = to_np(model.forward(np.array([[1, 2, 8, 9]])))
        np.testing.assert_array_equal(first[:, :2], second[:, :2])
        self.assertEqual(rng, model.rng.bit_generator.state)

    def test_gpu_backend_matches_cpu_when_available(self):
        from studio_backend import gpu_status
        status = gpu_status()
        if not status['available']:
            self.skipTest(status['reason'])
        spec = default_spec()
        spec['dims'].update(d_model=16, n_layer=1, n_ctx=32, dropout=0.)
        cpu = CompiledModel(validate_spec(spec), 11, device='cpu')
        gpu = CompiledModel(validate_spec(spec), 11, device=status['kind'])
        gpu.load_weights(cpu.state_arrays())
        x = np.array([[1, 2, 3, 4]])
        np.testing.assert_allclose(gpu.backend.to_cpu(gpu.forward(x)), to_np(cpu.forward(x)), rtol=1e-3, atol=1e-4)
