"""Issue #168: a job's memory budget and the batch that fits it (studio_memory), from the model's estimate and the device."""
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import studio_backend
import studio_jobs
from studio_estimate import estimate
from studio_memory import ADAMW_COPIES, batch_cap, budget, host_bytes, training_bytes
from studio_model import effective_spec, validate_config

GB = 2 ** 30
CONFIG = {"width": 32, "layers": 2, "heads": 2, "context": 64, "expansion": 2, "dropout": 0., "batch_size": 8,
          "steps": 5, "learning_rate": .003, "warmup": 1, "validation": 0, "eval_every": 5, "seed": 5, "device": "cpu"}


def ram_for(cfg, samples, share=1.):
    """RAM on which `share` of the memory fits the fixed cost plus exactly `samples` samples."""
    fixed, per_sample = training_bytes(validate_config(cfg))
    return int((fixed + samples * per_sample) / share)


class BudgetChecks(unittest.TestCase):
    def test_cuda_uses_the_gpu_memory_and_mps_what_metal_may_address(self):
        self.assertEqual(budget("cuda", .5, ram=64 * GB, gpu=8 * GB), 4 * GB)
        self.assertEqual(budget("mps", .75, ram=36 * GB, gpu=28 * GB), 21 * GB)

    def test_cpu_and_an_unknown_gpu_use_the_ram(self):
        self.assertEqual(budget("cpu", .5, ram=64 * GB, gpu=8 * GB), 32 * GB)
        self.assertEqual(budget("cuda", 1., ram=64 * GB, gpu=0), 64 * GB)

    def test_gpu_memory_comes_from_gpu_status_and_ram_from_the_machine(self):
        with mock.patch("studio_backend.gpu_status", return_value={"memory_bytes": 10 * GB}):
            self.assertEqual(budget("cuda", .5), 5 * GB)
        with mock.patch("studio_memory.physical_memory", return_value=16 * GB):
            self.assertEqual(budget("cpu", .5), 8 * GB)


class TrainingBytesChecks(unittest.TestCase):
    def test_fixed_cost_is_four_float32_copies_of_the_parameters_and_per_sample_one_context(self):
        cfg = validate_config(CONFIG)
        fixed, per_sample = training_bytes(cfg, vocab=263)
        result = estimate(effective_spec(cfg), 263)
        self.assertEqual(fixed, result["total"] * 4 * ADAMW_COPIES)
        self.assertEqual(per_sample, (result["activation_bytes_per_token"] + 2 * 263 * 4) * 64)
        self.assertGreater(per_sample, 0)

    def test_bf16_mixed_precision_keeps_float32_weights_but_halves_the_activations(self):
        spec = {"schema": 1, "dims": {"d_model": 32, "n_layer": 2, "n_ctx": 64, "dropout": 0},
                "block": {"mixer": {"id": "attention", "config": {"heads": 2}}}, "train": {"precision": "bf16"}}
        fp32 = validate_config(dict(CONFIG, architecture=dict(spec, train={"precision": "fp32"})))
        bf16 = validate_config(dict(CONFIG, architecture=spec))
        self.assertEqual(training_bytes(fp32, vocab=263)[0], training_bytes(bf16, vocab=263)[0])
        self.assertLess(training_bytes(bf16, vocab=263)[1], training_bytes(fp32, vocab=263)[1])


class LoraBytesChecks(unittest.TestCase):
    """LoRA trains only its adapters, so AdamW's state (and a gradient for each) is charged on them and not on the whole model."""

    def test_the_estimate_is_below_a_full_fine_tune_by_the_optimizer_state_difference(self):
        from studio_memory import lora_params
        plain = validate_config(CONFIG)
        lora = validate_config(dict(CONFIG, lora={"rank": 4}))
        spec = effective_spec(lora)
        total = estimate(spec, 263)["total"]
        adapters = lora_params(spec, 263, lora["lora"])
        self.assertGreater(adapters, 0)
        self.assertLess(adapters, total)
        full, light = training_bytes(plain, vocab=263), training_bytes(lora, vocab=263)
        self.assertEqual(light[1], full[1])  # the activations are the same: the backward pass still runs through the frozen base
        # weights and the base's gradient buffers stay; AdamW's two moments and the trained gradient shrink to the adapters
        self.assertEqual(full[0] - light[0], 4 * (2 * total - ADAMW_COPIES * adapters))
        self.assertEqual(light[0], 4 * (2 * total + ADAMW_COPIES * adapters))
        self.assertLess(light[0], full[0])

    def test_adapters_follow_the_rank_and_the_targets(self):
        from studio_memory import lora_params
        spec = effective_spec(validate_config(CONFIG))
        small, big = lora_params(spec, 263, {"rank": 2}), lora_params(spec, 263, {"rank": 8})
        self.assertGreater(big, small)
        only_proj = lora_params(spec, 263, {"rank": 4, "targets": ["proj"]})
        both = lora_params(spec, 263, {"rank": 4})
        self.assertLess(only_proj, both)
        # 2 layers x (qkv 32 -> 96, proj 32 -> 32) at rank 4, plus the 7 special-token rows
        self.assertEqual(both, 2 * 4 * ((32 + 96) + (32 + 32)) + 7 * 32)

    def test_offloaded_lora_moves_only_the_adapters_moments(self):
        lora = validate_config(dict(CONFIG, lora={"rank": 4}))
        plain = validate_config(CONFIG)
        self.assertLess(host_bytes(lora, vocab=263), host_bytes(plain, vocab=263))
        self.assertLessEqual(training_bytes(lora, 263, True)[0], training_bytes(lora, 263)[0])

    def test_a_layer_that_cannot_be_sized_falls_back_to_charging_every_weight(self):
        lora = validate_config(dict(CONFIG, lora={"rank": 4}))
        with mock.patch("studio_memory.lora_params", return_value=None):
            self.assertEqual(training_bytes(lora, vocab=263), training_bytes(validate_config(CONFIG), vocab=263))


class BatchCapChecks(unittest.TestCase):
    def test_never_above_the_declared_batch_and_never_below_one(self):
        self.assertEqual(batch_cap(CONFIG, 1., device="cpu", ram=1024 * GB), 8)
        self.assertEqual(batch_cap(CONFIG, 1., device="cpu", ram=1), 1)

    def test_the_cap_is_what_fits_after_the_fixed_cost(self):
        self.assertEqual(batch_cap(CONFIG, 1., device="cpu", ram=ram_for(CONFIG, 3)), 3)
        self.assertEqual(batch_cap(CONFIG, .5, device="cpu", ram=ram_for(CONFIG, 3, share=.5)), 3)
        self.assertEqual(batch_cap(CONFIG, 1., device="cuda", ram=1, gpu=ram_for(CONFIG, 5)), 5)

    def test_a_wider_model_a_longer_context_or_a_smaller_share_lowers_the_cap(self):
        ram = ram_for(CONFIG, 4)
        self.assertEqual(batch_cap(CONFIG, 1., device="cpu", ram=ram), 4)
        self.assertLess(batch_cap(dict(CONFIG, width=128), 1., device="cpu", ram=ram), 4)
        self.assertLess(batch_cap(dict(CONFIG, context=256), 1., device="cpu", ram=ram), 4)
        self.assertLess(batch_cap(CONFIG, .5, device="cpu", ram=ram), 4)

    def test_a_small_auto_model_is_budgeted_on_the_cpu_without_probing_the_gpu(self):
        with mock.patch("studio_backend.gpu_status", side_effect=AssertionError("a small model never probes the GPU")), \
                mock.patch("studio_probe.any_stored", return_value=False):  # no probe on this machine, whatever another test left in the cache
            self.assertEqual(batch_cap(dict(CONFIG, device="auto"), 1., ram=ram_for(CONFIG, 2)), 2)

    def test_a_large_auto_model_is_budgeted_on_the_gpu_it_will_use(self):
        large = dict(CONFIG, device="auto", width=512, layers=4, context=512, batch_size=16)
        with mock.patch("studio_backend.gpu_status", return_value={"available": True, "kind": "cuda", "name": "fake", "reason": "",
                                                                  "memory_bytes": ram_for(large, 6)}):
            self.assertEqual(batch_cap(large, 1., ram=1), 6)


class PretrainJobChecks(unittest.TestCase):
    def run_pretrain(self, share, ram):
        job = SimpleNamespace(params={"config": dict(CONFIG), "examples": [{"text": "one two"}], "memory_share": share, "output_dir": "out"})
        events, seen = [], {}

        def fake_train(cfg, examples, out, **kw):
            seen["cfg"] = cfg
            return {"status": "completed"}

        with mock.patch("studio_memory.physical_memory", return_value=ram), mock.patch("studio_model.train", side_effect=fake_train):
            studio_jobs.pretrain(job, SimpleNamespace(run_dir="runs"), events.append, None, None)
        return seen["cfg"]["batch_size"], [e["text"] for e in events if e["type"] == "info"]

    def test_a_batch_that_does_not_fit_is_lowered_and_said(self):
        batch, infos = self.run_pretrain(.5, ram_for(CONFIG, 3, share=.5))
        self.assertEqual(batch, 3)
        self.assertEqual(infos, ["A batch of 8 does not fit the 50% of memory the plan allows this job: training with batch size 3."])

    def test_a_batch_that_fits_is_kept_quietly(self):
        self.assertEqual(self.run_pretrain(1., 1024 * GB), (8, []))


class GpuStatusChecks(unittest.TestCase):
    def test_mps_reports_the_memory_metal_may_address(self):
        studio_backend.gpu_status.cache_clear()
        try:
            with mock.patch("torch.cuda.is_available", return_value=False), mock.patch("torch.backends.mps.is_available", return_value=True), \
                    mock.patch("torch.mps.recommended_max_memory", return_value=7 * GB), \
                    mock.patch("torch.ones", side_effect=lambda shape, dtype=None, device=None: torch.full(shape, 1., dtype=dtype)):
                status = studio_backend.gpu_status()
            self.assertEqual((status["available"], status["kind"], status["memory_bytes"]), (True, "mps", 7 * GB))
        finally:
            studio_backend.gpu_status.cache_clear()


if __name__ == "__main__":
    unittest.main()
