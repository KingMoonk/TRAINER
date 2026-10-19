"""Placing a job by measured speed: nodes report probes (studio_probe.reported), `place`/`decide` prefer the fastest fitting free node,
`device_of` and the plan use a cached probe. Fake nodes, fake status and a temporary cache file: no GPU, no window."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import studio_cluster as sc
import studio_memory
import studio_probe
import studio_scheduler as sched
from studio_cluster import NodeInfo
from studio_ia import design
from studio_model import validate_config
from tests.fake_cluster import Clock

GB = 2 ** 30
SHAPE = "abc123"
GPU = {"available": True, "kind": "cuda", "name": "Fake GPU", "reason": "", "memory_bytes": 8 * GB}
CONFIG = {"width": 16, "context": 32, "layers": 1, "heads": 2, "batch_size": 2}


def speed(device, seconds, age=10., tokens=256):
    return {"device": device, "seconds": {device: seconds}, "tokens": tokens, "age_s": age}


def node(address, device="cuda", seconds=None, shape=SHAPE, busy=False, picked=None, memory=16 * GB):
    speeds = {shape: speed(picked or device, seconds)} if seconds else {}
    return NodeInfo(address=address, hostname=address, device=device, memory_bytes=memory, busy=busy, dtypes=sc.DTYPES[device], speeds=speeds)


def up(*addresses):
    health = sched.Health(Clock())
    for a in addresses:
        health.observe(a, True)
    return health


class PlaceBySpeed(unittest.TestCase):
    def test_the_fastest_measured_node_wins_over_list_order(self):
        nodes = [node("local", seconds=2.), node("a", seconds=1.), node("b", seconds=.5)]
        chosen, why = sc.place_why(sc.defaults_for("pretrain"), nodes, shape=SHAPE)
        self.assertEqual(chosen.address, "b")
        self.assertIn("4.0x faster", why)

    def test_busy_nodes_are_not_candidates_however_fast(self):
        nodes = [node("local", seconds=2.), node("b", seconds=.1, busy=True)]
        self.assertEqual(sc.place(sc.defaults_for("pretrain"), nodes, shape=SHAPE).address, "local")

    def test_an_unmeasured_node_means_the_old_rule_and_no_claim(self):
        nodes = [node("local", seconds=2.), node("b")]
        chosen, why = sc.place_why(sc.defaults_for("pretrain"), nodes, shape=SHAPE)
        self.assertEqual((chosen.address, why), ("local", ""))

    def test_no_shape_is_the_old_rule(self):
        nodes = [node("local", seconds=2.), node("b", seconds=.5)]
        self.assertEqual(sc.place(sc.defaults_for("pretrain"), nodes).address, "local")

    def test_a_shape_only_another_design_measured_is_unmeasured(self):
        nodes = [node("local", seconds=2., shape="other"), node("b", seconds=.5, shape="other")]
        self.assertEqual(sc.place(sc.defaults_for("pretrain"), nodes, shape=SHAPE).address, "local")

    def test_near_ties_keep_the_earlier_node(self):
        nodes = [node("local", seconds=1.02), node("b", seconds=1.)]
        chosen, why = sc.place_why(sc.defaults_for("pretrain"), nodes, shape=SHAPE)
        self.assertEqual((chosen.address, why), ("local", ""))

    def test_a_gpu_machine_whose_probe_chose_its_cpu_does_not_meet_a_cuda_preference(self):
        nodes = [node("local", device="cuda", seconds=.2, picked="cpu"), node("b", device="cuda", seconds=.9)]
        self.assertEqual(sc.place(sc.defaults_for("pretrain"), nodes, shape=SHAPE).address, "b")

    def test_a_pin_is_never_overridden(self):
        nodes = [node("local", seconds=2.), node("b", seconds=.5)]
        self.assertEqual(sc.place(sc.defaults_for("pretrain"), nodes, pinned="local", shape=SHAPE).address, "local")

    def test_decide_goes_remote_to_the_fastest_up_peer_and_says_why(self):
        nodes = [node("local", seconds=2.), node("b", seconds=.5), node("c", seconds=.25)]
        placement = sched.decide("pretrain", nodes, up("b", "c"), shape=SHAPE)
        self.assertEqual((placement.where, placement.node), ("remote", "c"))
        self.assertIn("measured", placement.basis)

    def test_decide_ignores_the_speed_of_a_peer_that_is_down(self):
        nodes = [node("local", seconds=2.), node("b", seconds=.1)]
        self.assertEqual(sched.decide("pretrain", nodes, sched.Health(Clock()), shape=SHAPE).where, "local")


class Reports(unittest.TestCase):
    def test_the_payload_round_trips_and_bad_entries_are_dropped(self):
        info = node("local", seconds=.5)
        self.assertEqual(sc.info_from_payload("x", info.payload()).seconds_for(SHAPE), ("cuda", .5))
        body = info.payload()
        body["speeds"].update(bad={"device": "cuda"}, old=speed("cuda", 1., age=studio_probe.MAX_AGE_S + 1), zero=speed("cuda", 0.))
        self.assertEqual(set(sc.info_from_payload("x", body).speeds), {SHAPE})

    def test_a_peer_without_the_field_has_no_speeds(self):
        body = node("local").payload()
        del body["speeds"]
        self.assertEqual(sc.info_from_payload("x", body).speeds, {})

    def test_the_job_shape_matches_the_probe_key_of_its_config(self):
        self.assertEqual(sched.job_shape("pretrain", {"config": dict(CONFIG)}), studio_memory.shape_of(validate_config(CONFIG)))
        self.assertIsNone(sched.job_shape("eval", {"config": {}}))
        self.assertIsNone(sched.job_shape("pretrain", {"config": {"width": "x"}}))


class StoredProbes(unittest.TestCase):
    def setUp(self):
        studio_probe.forget()
        self.addCleanup(studio_probe.forget)
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "probe.json"

    def store(self, machine, at, shape="s1"):
        hit = {"version": studio_probe.VERSION, "device": "cuda", "seconds": {"cpu": 2., "cuda": 1.}, "tokens": 256, "measured_at": at}
        studio_probe._save(self.path, {f"{shape}|{machine}": hit})

    def test_reported_has_this_machines_fresh_probes_only(self):
        now = time.time()
        self.store(studio_probe.machine_key(GPU), now - 60)
        got = studio_probe.reported(GPU, self.path, now)
        self.assertEqual(got["s1"]["device"], "cuda")
        self.assertAlmostEqual(got["s1"]["age_s"], 60)

    def test_a_probe_from_another_torch_or_gpu_is_not_reported(self):
        self.store(studio_probe.machine_key(GPU).replace("Fake GPU", "Other GPU"), time.time())
        self.assertEqual(studio_probe.reported(GPU, self.path), {})

    def test_a_probe_older_than_the_limit_is_not_reported(self):
        now = time.time()
        self.store(studio_probe.machine_key(GPU), now - studio_probe.MAX_AGE_S - 1)
        self.assertEqual(studio_probe.reported(GPU, self.path, now), {})

    def test_no_gpu_reports_nothing(self):
        self.assertEqual(studio_probe.reported({"available": False}, self.path), {})


def result(device, cpu=1., gpu=.1):
    return {"version": 1, "device": device, "seconds": {"cpu": cpu, "cuda": gpu}, "tokens": 32,
            "tokens_per_second": {"cpu": 32 / cpu, "cuda": 32 / gpu}, "speedup": 10., "failed": {}, "larger_batch": None,
            "reason": "GPU 10.0x faster, using it."}


class DeviceAndPlan(unittest.TestCase):
    def test_device_of_follows_a_cached_probe_over_the_size_rule(self):
        cfg = validate_config(dict(CONFIG, device="auto"))
        with patch("studio_backend.gpu_status", return_value=GPU), patch("studio_probe.cached", return_value=result("cuda")), patch("studio_probe.any_stored", return_value=True):
            self.assertEqual(studio_memory.device_of(cfg), "cuda")  # the size rule says cpu at this size
        with patch("studio_backend.gpu_status", return_value=GPU), patch("studio_probe.cached", return_value=result("cpu", .1, 1.)), patch("studio_probe.any_stored", return_value=True):
            self.assertEqual(studio_memory.device_of(cfg), "cpu")

    def test_device_of_without_a_probe_is_the_size_rule(self):
        with patch("studio_backend.gpu_status", return_value=GPU), patch("studio_probe.cached", return_value=None):
            self.assertEqual(studio_memory.device_of(validate_config(dict(CONFIG, device="auto"))), "cpu")

    def test_the_plan_names_the_measured_device_and_its_step(self):
        spec = design.preset_specs(False)[0][1]
        cost = design.preset_cost(spec, ram=64 * GB, gpu="cuda", probe=result("cuda"))
        self.assertTrue(cost["measured"])
        self.assertEqual(cost["device"], "cuda")
        self.assertTrue(any("Measured on this computer" in line and "GPU is faster" in line for line in cost["lines"]))
        self.assertFalse(any("estimate" in line for line in cost["lines"]))

    def test_the_plan_says_estimate_when_a_gpu_is_there_but_nothing_was_timed(self):
        spec = design.preset_specs(False)[0][1]
        cost = design.preset_cost(spec, ram=64 * GB, gpu="cuda")
        self.assertFalse(cost["measured"])
        self.assertTrue(any("estimate" in line for line in cost["lines"]))


if __name__ == "__main__":
    unittest.main()
