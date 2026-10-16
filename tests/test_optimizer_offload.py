"""Issue #238: the optimizer state lives in host memory and streams through the device for each update. The math is the same, so
the result is too; the CPU runs it with host = device = cpu through the StateHome seam (nothing is ever pinned here)."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import studio_jobs
import studio_memory
from studio_memory import StateHome, batch_cap, host_bytes, offload_wanted, training_bytes
from studio_model import build_optimizer, effective_spec, load_checkpoint, train, validate_config
from studio_registry import get as get_module
from tests.test_memory import CONFIG as MEMORY_CONFIG, GB

EXAMPLES = [{"text": f"Counting up from {number}: {number}, {number + 1}, {number + 2}."} for number in range(8)]
CONFIG = {"width": 16, "layers": 1, "heads": 2, "context": 32, "expansion": 2, "dropout": 0., "batch_size": 2, "accumulation": 2,
          "steps": 4, "learning_rate": .003, "weight_decay": .01, "warmup": 0, "validation": 0, "eval_every": 4, "seed": 19,
          "objective": "all", "device": "cpu"}


def run(root, name, offload, steps=4, resume=None, optimizer="adamw", events=None):
    cfg = dict(CONFIG, steps=steps)
    if optimizer != "adamw":
        spec = copy.deepcopy(effective_spec(validate_config(cfg)))
        spec["train"]["optimizer"] = {"id": optimizer, "config": {}}
        cfg["architecture"] = spec
    out = Path(root) / name
    train(cfg, EXAMPLES, out, resume=resume, offload_optimizer=offload, emit=events.append if events is not None else None)
    return load_checkpoint(out / "last.npz"), out / "last.npz"


def assert_same_run(case, a, b):
    case.assertEqual(a["optimizer_steps"], b["optimizer_steps"])
    for key, value in a["model"].items():
        np.testing.assert_allclose(b["model"][key], value, rtol=1e-5, atol=1e-6, err_msg=key)
    for state in ("m", "v"):
        for key, value in a["optimizer_state"][state].items():
            np.testing.assert_allclose(b["optimizer_state"][state][key], value, rtol=1e-5, atol=1e-6, err_msg=f"{state}::{key}")


class OffloadedRunChecks(unittest.TestCase):
    def test_weights_and_state_after_a_short_run_equal_the_run_without_offload(self):
        with tempfile.TemporaryDirectory() as root:
            events = []
            plain, _ = run(root, "plain", False)
            offloaded, _ = run(root, "offloaded", True, events=events)
            assert_same_run(self, plain, offloaded)
            self.assertGreater(np.abs(offloaded["optimizer_state"]["v"]["token"]).max(), 0)  # the state really advanced
            self.assertTrue(any(e["type"] == "info" and "host memory" in e["text"] for e in events))

    def test_no_offload_says_nothing_about_it(self):
        with tempfile.TemporaryDirectory() as root:
            events = []
            run(root, "plain", False, steps=1, events=events)
            self.assertFalse(any("host memory" in e.get("text", "") for e in events))

    def test_every_optimizer_gives_the_same_result_with_its_state_offloaded(self):
        for name in ("lion", "sgd", "muon"):
            with self.subTest(optimizer=name), tempfile.TemporaryDirectory() as root:
                plain, _ = run(root, "plain", False, steps=2, optimizer=name)
                offloaded, _ = run(root, "offloaded", True, steps=2, optimizer=name)
                for key, value in plain["model"].items():
                    np.testing.assert_allclose(offloaded["model"][key], value, rtol=1e-5, atol=1e-6, err_msg=key)

    def test_resume_reproduces_the_run_with_offload_on_off_or_switched(self):
        with tempfile.TemporaryDirectory() as root:
            for first in (False, True):
                _, saved = run(root, f"half_{first}", first, steps=2)
                resumed = {second: run(root, f"resumed_{first}_{second}", second, steps=4, resume=saved)[0] for second in (False, True)}
                with self.subTest(first=first):
                    self.assertEqual(resumed[False]["step"], 4)
                    assert_same_run(self, resumed[False], resumed[True])
            # the first half itself is the same checkpoint either way, so a resume may switch in both directions
            a, b = (load_checkpoint(Path(root) / f"half_{first}" / "last.npz") for first in (False, True))
            assert_same_run(self, a, b)

    def test_state_is_held_by_the_home_and_only_the_current_parameter_is_resident(self):
        homed = []

        class Spy(StateHome):
            def fetch(self, stored, device):
                homed.append(stored)
                return super().fetch(stored, device)

        class Model:
            p = {"a": torch.ones(3), "b": torch.ones(2)}
            g = {"a": torch.full((3,), .5), "b": torch.full((2,), .5)}
            roles = {"a": "weight", "b": "weight"}
            lr_scale = {}

        optimizer = get_module("optimizer", "adamw").build(get_module("optimizer", "adamw").parse(), None, None).attach(Model(), Spy())
        hosts = {name: dict(getattr(optimizer, name)) for name in optimizer.states}
        seen = []
        original = optimizer.apply
        optimizer.apply = lambda key, *args: (seen.append({n: getattr(optimizer, n)[key] for n in optimizer.states}), original(key, *args))
        optimizer.update(Model(), .1)
        self.assertEqual(len(homed), 4)  # m and v of both parameters, each fetched once
        for key, resident in zip(("a", "b"), seen):
            for name, tensor in resident.items():
                self.assertIsNot(tensor, hosts[name][key])  # apply() saw a device copy...
        for name, tensors in hosts.items():
            for key, host in tensors.items():
                self.assertIs(getattr(optimizer, name)[key], host)  # ...and the host tensor is what is kept
                self.assertGreater(float(host.abs().sum()), 0)  # with the updated values copied back

    def test_a_pinned_home_is_never_asked_for_on_the_cpu(self):
        self.assertFalse(StateHome.for_device("cpu").pin)
        self.assertFalse(StateHome.for_device("mps").pin)
        self.assertTrue(StateHome.for_device("cuda").pin)
        with mock.patch.object(torch.Tensor, "pin_memory", side_effect=AssertionError("no pinning without CUDA")):
            optimizer, _ = build_optimizer(_model(), offload=True)
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in optimizer.m.values()))


def _model():
    from studio_model import ScratchTransformer
    return ScratchTransformer(validate_config(CONFIG), 263)


class EstimateChecks(unittest.TestCase):
    def setUp(self):
        self.cfg = validate_config(MEMORY_CONFIG)

    def test_the_moments_move_from_the_device_to_the_host(self):
        fixed, per_sample = training_bytes(self.cfg, vocab=263)
        lighter, same = training_bytes(self.cfg, vocab=263, offload=True)
        moments = host_bytes(self.cfg, vocab=263)
        self.assertEqual(same, per_sample)
        self.assertEqual(host_bytes(self.cfg, vocab=263, offload=False), 0)
        from studio_estimate import estimate
        total = estimate(effective_spec(self.cfg), 263)["total"]
        self.assertEqual(moments, total * 4 * studio_memory.STATE_COPIES)
        self.assertLess(lighter, fixed)
        # weights and gradients stay; the device also holds the moments of the one parameter being updated
        self.assertGreaterEqual(lighter, fixed - moments)
        self.assertLess(lighter, fixed - moments + moments // 2)

    def test_offload_lets_a_bigger_batch_fit(self):
        gpu = training_bytes(self.cfg, offload=True)[0] + 2 * training_bytes(self.cfg)[1]
        self.assertLess(batch_cap(MEMORY_CONFIG, 1., device="cuda", gpu=gpu), 2)
        self.assertEqual(batch_cap(MEMORY_CONFIG, 1., device="cuda", gpu=gpu, offload=True), 2)


class AutomaticChoiceChecks(unittest.TestCase):
    def setUp(self):
        self.cfg = validate_config(MEMORY_CONFIG)
        self.fixed, self.per_sample = training_bytes(self.cfg)

    def test_cuda_offloads_only_when_even_one_sample_does_not_fit(self):
        tight = self.fixed + self.per_sample - 1
        self.assertTrue(offload_wanted(self.cfg, 1., device="cuda", gpu=tight))
        self.assertFalse(offload_wanted(self.cfg, 1., device="cuda", gpu=tight + 1))
        self.assertFalse(offload_wanted(self.cfg, 1., device="cuda", gpu=64 * GB))

    def test_the_share_counts(self):
        gpu = (self.fixed + self.per_sample) * 2
        self.assertFalse(offload_wanted(self.cfg, 1., device="cuda", gpu=gpu))
        self.assertTrue(offload_wanted(self.cfg, .4, device="cuda", gpu=gpu))

    def test_cpu_and_mps_never_offload(self):
        for device in ("cpu", "mps"):
            self.assertFalse(offload_wanted(self.cfg, 1., device=device, ram=1, gpu=1), device)

    def test_the_device_is_the_one_the_config_lands_on(self):
        with mock.patch("studio_memory.device_of", return_value="cuda"):
            self.assertTrue(offload_wanted(self.cfg, 1., gpu=1))
        with mock.patch("studio_memory.device_of", return_value="cpu"):
            self.assertFalse(offload_wanted(self.cfg, 1., gpu=1))

    def test_a_pretrain_job_that_does_not_fit_trains_with_offload_and_says_so(self):
        job = mock.Mock(params={"config": dict(MEMORY_CONFIG), "examples": [{"text": "one two"}], "memory_share": 1.})
        events = []
        with mock.patch("studio_jobs.output_dir", return_value="out"), mock.patch("studio_model.train") as trained, \
                mock.patch("studio_memory.device_of", return_value="cuda"), \
                mock.patch("studio_memory.budget", return_value=self.fixed // 2):
            studio_jobs.pretrain(job, mock.Mock(run_dir="r"), events.append, None, None)
        self.assertTrue(trained.call_args.kwargs["offload_optimizer"])
        with mock.patch("studio_jobs.output_dir", return_value="out"), mock.patch("studio_model.train") as trained, \
                mock.patch("studio_memory.device_of", return_value="cpu"):
            studio_jobs.pretrain(job, mock.Mock(run_dir="r"), events.append, None, None)
        self.assertFalse(trained.call_args.kwargs["offload_optimizer"])


@unittest.skipUnless(torch.cuda.is_available(), "pinned memory and the device copies need a CUDA GPU")
class CudaChecks(unittest.TestCase):
    def test_state_is_pinned_on_the_host_and_the_update_matches_the_device_state(self):
        model = _model()
        model = type(model)(validate_config(dict(CONFIG, device="cuda")), 263)
        offloaded, _ = build_optimizer(model, offload=True)
        plain, _ = build_optimizer(model)
        self.assertTrue(all(t.is_pinned() and t.device.type == "cpu" for t in offloaded.m.values()))
        self.assertTrue(all(t.device.type == "cuda" for t in plain.m.values()))
        before = {k: v.clone() for k, v in model.p.items()}
        for key, grad in model.g.items():
            grad.normal_()
        offloaded.update(model, .01)
        after = {k: v.clone() for k, v in model.p.items()}
        for key, value in before.items():
            model.p[key].copy_(value)
        plain.update(model, .01)
        for key, value in after.items():
            torch.testing.assert_close(model.p[key], value, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(offloaded.m[key].cuda(), plain.m[key], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
