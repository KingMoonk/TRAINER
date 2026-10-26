"""studio_fit: a solo run scaled to what the machine can train, from an injected hardware profile (no GPU, no window)."""
import os
import tempfile
import unittest
from unittest import mock

import studio_fit as F
import studio_memory
from studio_ia import pretraining_plan as plan
from studio_model import validate_config
from studio_spec import default_spec, validate_spec

G = 2 ** 30
PIECES = 4_000_000_000


def design(width=1024, layers=24, context=2048):
    spec = default_spec()
    spec["dims"].update(d_model=width, n_layer=layers, n_ctx=context)
    spec["block"]["mixer"]["config"]["heads"] = 16
    return validate_spec(spec)


def config(rows=16, **dims):
    return plan.pretrain_config(design(**dims), 1_000_000, rows, .05)


def machine(vram=8, free=7, ram=24, disk=300, **kw):
    return F.Profile(device="cuda", gpu_name="Fake 8 GB", vram_total=int(vram * G), vram_free=int(free * G), ram_total=32 * G,
                     ram_free=int(ram * G), disk_free=int(disk * G), cpu_cores=16, **kw)


class SoloFitTests(unittest.TestCase):
    def test_a_run_that_fits_is_unchanged(self):
        cfg = config(width=256, layers=4, context=256)
        found = F.fit_solo(cfg, machine(), pieces=50_000_000)
        self.assertEqual((found.notes, found.changed, found.refusal), ([], False, ""))
        self.assertEqual(found.config, cfg)
        self.assertFalse(found.offload)

    def test_the_8gb_machine_is_scaled_in_order_and_every_change_is_listed(self):
        cfg = config()
        found = F.fit_solo(cfg, machine(), pieces=PIECES)
        self.assertTrue(found.changed)
        kinds = [n.kind for n in found.notes]
        self.assertEqual(kinds, sorted(kinds, key=F.ORDER.index))
        self.assertIn("batch", kinds)
        self.assertIn("context", kinds)
        self.assertNotIn("model", kinds)  # only as far as needed: the design stays
        self.assertEqual(found.config["architecture"]["dims"]["d_model"], cfg["architecture"]["dims"]["d_model"])
        self.assertLess(found.config["context"], cfg["context"])
        self.assertLess(found.config["batch_size"], cfg["batch_size"])
        self.assertEqual(found.config["batch_size"] * found.config.get("accumulation", 1), cfg["batch_size"] * cfg.get("accumulation", 1))
        self.assertTrue(all(n.text for n in found.notes))
        sizer = F._Sizer(machine(), "cuda", 1., 256)
        self.assertGreaterEqual(sizer.largest_batch(validate_config(found.config), found.offload), 1)

    def test_a_tight_fit_can_be_kept(self):
        cfg = config(width=1024, layers=24, context=1024)
        sizer = F._Sizer(machine(), "cuda", 1., 256)
        need = sizer.bytes_for(validate_config(cfg), False)
        free = need / 1.05 / G  # a little under what the declared batch needs
        scaled = F.fit_solo(cfg, machine(vram=free + 1, free=free), pieces=PIECES)
        kept = F.fit_solo(cfg, machine(vram=free + 1, free=free), pieces=PIECES, keep_mine=True)
        self.assertTrue(scaled.can_keep_mine)
        self.assertTrue(kept.kept_mine)
        self.assertEqual(kept.config["batch_size"], cfg["batch_size"])
        self.assertTrue(all(n.advisory for n in kept.notes))

    def test_a_run_that_cannot_fit_is_refused_naming_the_resource(self):
        cfg = config()
        found = F.fit_solo(cfg, machine(vram=2, free=.3), pieces=PIECES, fixed_architecture=True)
        self.assertIn("GPU memory", found.refusal)
        self.assertEqual(found.blocker["resource"], "vram")
        self.assertGreater(found.blocker["needed"], found.blocker["available"])

    def test_no_gpu_for_a_gpu_run_is_refused(self):
        cfg = dict(config(), device="cuda")
        cpu = F.Profile(device="cpu", ram_total=32 * G, ram_free=16 * G, disk_free=100 * G)
        self.assertIn("no CUDA GPU", F.fit_solo(cfg, cpu, pieces=PIECES).refusal)

    def test_token_budget_is_capped_from_speed_and_window(self):
        cfg = config(width=256, layers=4, context=256)
        found = F.fit_solo(cfg, machine(), pieces=PIECES, token_budget=PIECES, speed=10_000., window_hours=10.)
        self.assertEqual([n.kind for n in found.notes], ["tokens"])
        self.assertEqual(found.token_budget, 10 * 3600 * 10_000)
        self.assertEqual(found.wall_seconds, 10 * 3600)
        self.assertTrue(found.can_keep_mine)
        self.assertLessEqual(found.hours, 10.001)

    def test_override_keeps_the_figure_when_the_run_fits_but_is_slow(self):
        cfg = config(width=256, layers=4, context=256)
        found = F.fit_solo(cfg, machine(), pieces=PIECES, token_budget=PIECES, speed=10_000., window_hours=10., keep_mine=True)
        self.assertEqual(found.token_budget, PIECES)
        self.assertIsNone(found.wall_seconds)
        self.assertTrue(found.kept_mine)
        self.assertEqual([(n.kind, n.advisory) for n in found.notes], [("slow", True)])

    def test_disk_caps_the_text_read(self):
        cfg = config(width=256, layers=4, context=256)
        found = F.fit_solo(cfg, machine(disk=20), pieces=PIECES, token_budget=PIECES)
        note = next(n for n in found.notes if n.reason == "disk")
        self.assertIn("disk", note.text)
        self.assertLess(found.token_budget, PIECES)

    def test_no_disk_for_even_the_smallest_run_is_refused(self):
        cfg = config(width=256, layers=4, context=256)
        found = F.fit_solo(cfg, machine(disk=.5), pieces=PIECES, token_budget=PIECES)
        self.assertEqual(found.blocker["resource"], "disk")
        self.assertIn("disk space", found.refusal)

    def test_unknown_figures_skip_their_check(self):
        cfg = config(width=256, layers=4, context=256)
        blind = F.Profile(device="cuda", vram_total=8 * G, vram_free=7 * G)
        self.assertEqual(F.fit_solo(cfg, blind, pieces=PIECES).notes, [])


class MembersTests(unittest.TestCase):
    def test_members_are_sized_each_by_their_own_memory(self):
        cfg = config(rows=8, width=1024, layers=24, context=1024)
        small, big = machine(vram=6, free=5), machine(vram=80, free=78)
        found = F.fit_members(cfg, [{"name": "laptop", "profile": small, "speed": 8000.}, {"name": "tower", "profile": big, "speed": 30000.}],
                              pieces=PIECES, token_budget=PIECES, window_hours=10.)
        laptop, tower = found["members"]
        self.assertTrue(laptop["fits"] and tower["fits"])
        self.assertEqual((tower["batch_size"], tower["accumulation"]), (cfg["batch_size"], cfg.get("accumulation", 1)))
        self.assertLess(laptop["batch_size"], cfg["batch_size"])
        self.assertEqual(laptop["batch_size"] * laptop["accumulation"], cfg["batch_size"] * cfg.get("accumulation", 1))
        self.assertEqual(found["token_budget"], int(10 * 3600 * 38000.))  # the combined measured speed
        self.assertEqual(found["flagged"], [])

    def test_a_member_that_cannot_fit_is_flagged_and_the_others_are_not_scaled_for_it(self):
        cfg = config(rows=8, width=1024, layers=24, context=1024)
        found = F.fit_members(cfg, [{"name": "tiny", "profile": machine(vram=1, free=.2)}, {"name": "tower", "profile": machine(vram=80, free=78)}])
        self.assertEqual(found["flagged"], ["tiny"])
        self.assertIn("tiny cannot take part", found["members"][0]["refusal"])
        self.assertEqual(found["members"][1]["batch_size"], cfg["batch_size"])


class RefitTests(unittest.TestCase):
    def params(self, **kw):
        return {"config": config(rows=8, context=1024), "mode": "from_scratch", **kw}

    def test_refit_scales_a_fresh_job_from_what_was_declared(self):
        found = F.refit(self.params(), machine(), 1.)
        self.assertTrue(found.changed)
        again = F.applied(self.params(), found)
        self.assertEqual(again[F.RECORD]["declared"]["config"], self.params()["config"])
        self.assertTrue(again[F.RECORD]["scaled"])
        self.assertEqual(again["config"], found.config)
        # fitted again on a bigger machine, from the declared settings and not from the scaled ones
        roomy = F.refit(again, machine(vram=80, free=78), 1.)
        self.assertFalse(roomy.changed)
        self.assertEqual(F.applied(again, roomy)["config"], self.params()["config"])

    def test_a_resume_or_a_restart_is_not_scaled(self):
        self.assertIsNone(F.refit(self.params(resume="x"), machine(), 1.))
        self.assertIsNone(F.refit(self.params(continue_from="x"), machine(), 1.))
        with tempfile.TemporaryDirectory() as out:
            open(os.path.join(out, "last.npz"), "wb").close()
            self.assertIsNone(F.refit(self.params(output_dir=out), machine(), 1.))

    def test_a_catch_up_keeps_its_design(self):
        found = F.refit(self.params(mode="catch_up", init_from="best.npz"), machine(vram=2, free=.3), 1.)
        self.assertEqual(found.blocker["resource"], "vram")

    def test_any_init_from_keeps_its_design_whatever_the_mode(self):
        for mode in ("catch_up", "from_scratch", "grow", None):
            params = self.params(init_from="best.npz")
            params["mode"] = mode
            found = F.refit(params, machine(vram=2, free=.3), 1.)
            self.assertEqual(found.blocker["resource"], "vram", mode)
            self.assertIn("the model it continues from", found.refusal, mode)
        self.assertNotIn("the model it continues from", F.refit(self.params(), machine(vram=2, free=.3), 1.).refusal)  # no weights to keep: it may shrink

    def test_a_lora_job_is_sized_by_its_adapters_and_keeps_its_design(self):
        plain = validate_config(config(rows=1, context=128))
        lora = validate_config(dict(config(rows=1, context=128), lora={"rank": 8}))
        fixed_full, per_sample = studio_memory.training_bytes(plain, 256)
        fixed_lora = studio_memory.training_bytes(lora, 256)[0]
        self.assertLess(fixed_lora, fixed_full)
        # a card with room for the LoRA run's weights and a few rows, but not for the whole model with AdamW's state
        card = (fixed_lora + 4 * per_sample) / 0.9
        self.assertLess(card, fixed_full)
        sized = studio_memory.training_bytes(lora, 256)
        self.assertGreater(card, sized[0] + 2 * sized[1])
        profile = F.Profile(device="cuda", vram_total=int(card), vram_free=int(card), ram_total=64 * G, ram_free=48 * G, disk_free=500 * G, cpu_cores=8)
        kept = F.fit_solo(lora, profile, fixed_architecture=True, vocab=256)
        self.assertFalse(kept.refusal)
        self.assertEqual(kept.config["architecture"], lora["architecture"])
        self.assertFalse(kept.offload)  # the adapters' state stays on the card
        sizer = F._Sizer(profile, "cuda", 1., 256)
        self.assertGreaterEqual(sizer.largest_batch(lora, False), 1)
        self.assertEqual(sizer.largest_batch(plain, False), 0)  # the same design in full needs the optimizer in RAM
        self.assertTrue(F.fit_solo(plain, profile, fixed_architecture=True, vocab=256).offload)

    def test_free_memory_that_dropped_changes_the_fit(self):
        params = self.params()
        before = F.refit(params, machine(vram=80, free=78), 1.)
        after = F.refit(params, machine(vram=80, free=5), 1.)
        self.assertFalse(before.changed)
        self.assertTrue(after.changed)


class ProfileTests(unittest.TestCase):
    def test_round_trip(self):
        p = machine(tokens_per_second=1234.)
        self.assertEqual(F.Profile.from_dict(p.to_dict()), p)

    def test_cpu_profile_from_an_injected_status(self):
        got = F.hardware_profile(status={"available": False, "kind": "cpu"})
        self.assertEqual((got.device, got.vram_total, got.vram_free), ("cpu", 0, None))
        self.assertGreater(got.disk_free or 1, 0)

    def test_cuda_status_reads_the_free_memory_fresh(self):
        torch = mock.MagicMock()
        torch.cuda.mem_get_info.return_value = (3 * G, 8 * G)
        with mock.patch.dict("sys.modules", {"torch": torch}):
            got = F.hardware_profile(status={"available": True, "kind": "cuda", "name": "X", "memory_bytes": 8 * G})
        self.assertEqual((got.device, got.vram_total, got.vram_free), ("cuda", 8 * G, 3 * G))

    def test_available_memory_is_a_number_or_unknown(self):
        found = studio_memory.available_memory()
        self.assertTrue(found is None or 0 < found <= studio_memory.total_memory() * 2)


@unittest.skipUnless(os.environ.get("TRAINER_CUDA_SMOKE"), "set TRAINER_CUDA_SMOKE=1 on a machine with a CUDA GPU")
class CudaSmokeTest(unittest.TestCase):
    def test_a_real_profile_has_a_gpu(self):
        got = F.hardware_profile()
        self.assertEqual(got.device, "cuda")
        self.assertGreater(got.vram_free, 0)


if __name__ == "__main__":
    unittest.main()
