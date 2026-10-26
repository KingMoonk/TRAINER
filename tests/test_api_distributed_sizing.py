"""A distributed run sized per member (studio_fit.fit_members): each machine's profile comes from its own GET /node, the coordinator sizes
batch, accumulation and optimizer placement per member, names a member that cannot fit, and cuts the rounds to the window. Injected
profiles and a faked survey; no GPU, no window, no network."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import studio_cluster
import studio_distributed as runs
import studio_diloco
import studio_fit
from studio_api import app as api_app
from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.services import FakeServices
from studio_spec import default_spec
from tests import diloco_support as support
from tests.test_api_distributed import FakeClient, clients, owner_job

G = 2 ** 30


def machine(vram, free, speed=None):
    return studio_fit.Profile(device="cuda", gpu_name=f"{vram} GB", vram_total=int(vram * G), vram_free=int(free * G), ram_total=64 * G,
                              ram_free=48 * G, disk_free=500 * G, cpu_cores=16, tokens_per_second=speed)


def peer(address, profile, **kw):
    return studio_cluster.NodeInfo(address=address, hostname=address, version="1.0.0", device="cuda", device_name="RTX",
                                   profile=profile.to_dict() if profile is not None else None, **kw)


def big_design():
    spec = default_spec(device="cpu")
    spec["train"]["precision"] = {"id": "bf16"}  # default_spec() would pick it by the host's GPU, and the memory per row with it (#746)
    spec["dims"].update(d_model=1024, n_layer=24, n_ctx=1024)
    spec["block"]["mixer"]["config"]["heads"] = 16
    return spec


class SizingTests(unittest.TestCase):
    PEERS = {}

    @staticmethod
    def estimate(cfg, vocab=None, offload=False):
        """(fixed, per_sample) bytes of a step: 1 GB + 1 GB a row, or with the optimizer in RAM 0.5 GB + 0.25 GB a row. The real estimate
        differs by runner (CPU or GPU); the sizing rules under test do not."""
        return (int(.5 * G), int(.25 * G)) if offload else (G, G)

    def setUp(self):
        self.enterContext(mock.patch("studio_memory.training_bytes", self.estimate))
        self.services = FakeServices(step_seconds=.005)
        self.services.save_version(big_design(), "big")
        self.services.pause()  # the job stays in Up next to be read; nobody is invited
        self.services.hardware = machine(80, 78, speed=30000.)
        with mock.patch.object(api_app, "WEB_OUT", Path(tempfile.gettempdir()) / "no-such-web-out"):
            self.client = self.enterContext(TestClient(create_app(self.services, open=True, hosts=ANY)))
        self.addCleanup(runs.RUNS.clear)
        self.addCleanup(studio_diloco.COORDINATORS.clear)
        self.peers = {"laptop": peer("laptop", machine(6, 5, speed=8000.)), "tiny": peer("tiny", machine(1, .2, speed=500.)),
                      "old": peer("old", None)}
        self.services.set_setting("nodes", list(self.peers))
        self.enterContext(mock.patch("studio_api.routers.distributed.app_version", return_value="1.0.0"))
        self.enterContext(mock.patch.object(studio_cluster, "survey", lambda state, own, *a, **k: [own, *(self.peers[x] for x in state["studioConfig"]["nodes"])]))

    def body(self, **more):
        return {"dataset_ids": ["Your files"], "rounds": 3, "inner_steps": 2, **more}

    def sizing(self, **more):
        found = self.client.post("/distributed/sizing", json=self.body(**more))
        self.assertEqual(found.status_code, 200, found.text)
        return found.json()

    def start(self, **more):
        return self.client.post("/distributed/runs", json=self.body(**more))

    def job(self, run):
        return next(j for j in self.services.queue().up_next if j.id == run["id"])

    def test_two_members_of_different_capacity_each_get_their_own_batch(self):
        found = self.sizing(machines=["laptop"])
        tower, laptop = found["members"]
        self.assertEqual((tower["address"], laptop["address"]), ("local", "laptop"))
        self.assertTrue(tower["fits"] and laptop["fits"])
        self.assertEqual((tower["batch_size"], tower["accumulation"], tower["offload"]), (16, 1, False))
        self.assertLess(laptop["batch_size"], 16)
        self.assertGreater(laptop["accumulation"], 1)
        self.assertEqual(laptop["batch_size"] * laptop["accumulation"], 16)  # the same rows per optimizer step
        self.assertTrue(any("laptop" in note for note in laptop["notes"]))
        self.assertEqual(tower["notes"], [])
        self.assertEqual(found["flagged"], [])
        self.assertEqual(found["tokens_per_second"], 38000.)

    def test_a_member_that_cannot_fit_is_named_with_the_resource_and_figure(self):
        found = self.sizing(machines=["laptop", "tiny"])
        tiny = next(m for m in found["members"] if m["name"] == "tiny")
        self.assertEqual((found["flagged"], tiny["fits"]), (["tiny"], False))
        self.assertIn("tiny cannot take part", tiny["refusal"])
        self.assertRegex(tiny["refusal"], r"\d+(\.\d+)? (MB|GB)")
        self.assertEqual(found["tokens_per_second"], 38000.)  # the speed of the members that fit; tiny's 500 is not in it

    def test_a_peer_that_reports_no_profile_trains_as_the_design_says(self):
        found = self.sizing(machines=["old"])
        old = next(m for m in found["members"] if m["name"] == "old")
        self.assertEqual((old["sized"], old["fits"], old["batch_size"]), (False, True, None))

    def test_without_a_readable_profile_here_nothing_is_sized(self):
        self.services.hardware = None
        self.assertIsNone(self.client.post("/distributed/sizing", json=self.body(machines=["laptop"])).json())
        run = self.start(machines=["laptop"])
        self.assertEqual(run.status_code, 201, run.text)
        self.assertNotIn("member_fit", self.job(run.json()).params)

    def test_a_start_with_a_member_that_cannot_fit_is_refused_until_the_person_confirms(self):
        refused = self.start(machines=["laptop", "tiny"])
        self.assertEqual(refused.status_code, 409)
        self.assertIn("tiny cannot take part", refused.json()["detail"])
        self.assertEqual(list(self.services.queue().up_next), [])
        started = self.start(machines=["laptop", "tiny"], leave_out_unfit=True)
        self.assertEqual(started.status_code, 201, started.text)
        params = self.job(started.json()).params
        self.assertEqual([m["address"] for m in params["machines"]], ["local", "laptop"])
        self.assertEqual([row["node"] for row in params["left_out"]], ["tiny"])
        self.assertIn("cannot take part", params["left_out"][0]["reason"])

    def test_each_member_s_sizing_is_kept_on_the_job_and_told_in_its_note(self):
        started = self.start(machines=["laptop"])
        job = self.job(started.json())
        local, laptop = job.params["member_fit"][job.params["owner"]], job.params["member_fit"]["laptop"]
        self.assertEqual((local["batch_size"], local["accumulation"], local["offload_optimizer"]), (16, 1, False))
        self.assertLess(laptop["batch_size"], 16)
        self.assertIn("Sized per machine:", job.note)
        self.assertIn("laptop", job.note)

    def test_a_member_whose_optimizer_must_move_to_ram_is_told_to(self):
        self.peers["laptop"] = peer("laptop", machine(1.7, 1.2, speed=8000.))
        laptop = self.sizing(machines=["laptop"])["members"][1]
        self.assertEqual((laptop["fits"], laptop["offload"], laptop["batch_size"], laptop["accumulation"]), (True, True, 2, 8))
        self.assertTrue(any("optimizer" in note for note in laptop["notes"]))

    def test_the_rounds_are_cut_to_what_the_members_can_train_in_the_window(self):
        found = self.sizing(machines=["laptop"], rounds=10_000, inner_steps=1000, window_hours=1)
        self.assertTrue(found["rounds_cut"])
        self.assertLess(found["rounds"], 100_000)
        self.assertEqual(found["token_budget"], int(3600 * 38000.))
        self.assertIn("Token budget", found["note"])
        started = self.start(machines=["laptop"], rounds=10_000, inner_steps=1000, window_hours=1)
        self.assertEqual(started.status_code, 201, started.text)
        self.assertEqual(self.job(started.json()).params["rounds"], found["rounds"])

    def test_a_run_that_fits_the_window_keeps_its_rounds(self):
        found = self.sizing(machines=["laptop"])
        self.assertEqual((found["rounds"], found["rounds_cut"], found["token_budget"]), (3, False, None))


class MemberSizingReachesTheMember(unittest.TestCase):
    def setUp(self):
        FakeClient.calls = []
        self.addCleanup(runs.RUNS.clear)

    def test_an_invited_member_gets_its_own_sizing(self):
        sizing = {"box": {"batch_size": 4, "accumulation": 4, "offload_optimizer": False}, "mac": {"batch_size": 16, "accumulation": 1, "offload_optimizer": False}}
        machines = [{"node": "mac", "address": "local", "device": "mps", "device_name": "M3"},
                    {"node": "box", "address": "box", "device": "cuda", "device_name": "RTX"}]
        runs.invite(owner_job(machines, member_fit=sizing), clients())
        self.assertEqual(FakeClient.calls[0][3]["params"]["fit"], sizing["box"])

    def test_a_member_without_one_is_invited_as_before(self):
        machines = [{"node": "mac", "address": "local", "device": "mps", "device_name": "M3"},
                    {"node": "box", "address": "box", "device": "cuda", "device_name": "RTX"}]
        runs.invite(owner_job(machines), clients())
        self.assertNotIn("fit", FakeClient.calls[0][3]["params"])


class MemberTrainsWithItsSizing(unittest.TestCase):
    def test_a_member_trains_with_the_batch_accumulation_and_offload_the_coordinator_sized_it_to(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            folder = Path(tmp)
            coordinator = support.build(folder / "run", "run", inner_steps=2, rounds=2, min_members=1)
            studio_diloco.COORDINATORS["run"] = coordinator
            self.addCleanup(studio_diloco.COORDINATORS.clear)
            client = self.enterContext(support.TermClient(support.app()))
            seen = {}

            def train(cfg, examples, out, **options):
                seen.update(cfg=cfg, offload=options["offload_optimizer"])

            fit = {"batch_size": 2, "accumulation": 8, "offload_optimizer": True}
            with mock.patch("studio_model.train", train), self.assertRaises(RuntimeError):  # the stand-in trains no steps, so the member ends unfinished
                studio_diloco.run_member(support.Peer(client), "run", "m", 0, folder / "m", folder / "out", lambda event: None,
                                         threading.Event(), wait=.2, fit=fit)
        self.assertEqual((seen["cfg"]["batch_size"], seen["cfg"]["accumulation"], seen["offload"]), (2, 8, True))


class NodeProfileTests(unittest.TestCase):
    def test_a_peer_s_profile_round_trips_through_its_node_payload(self):
        profile = machine(8, 7, speed=1234.)
        body = studio_cluster.own_info(status={"available": True, "kind": "cuda", "name": "GPU", "memory_bytes": 8 * G}, profile=profile).payload()
        info = studio_cluster.info_from_payload("box", body)
        self.assertEqual(studio_fit.Profile.from_dict(info.profile), profile)

    def test_a_missing_or_malformed_profile_is_none(self):
        base = {"hostname": "box", "device": "cuda"}
        for bad in (None, "x", {"device": "tpu"}, {"device": "cuda", "vram_free": "lots"}):
            self.assertIsNone(studio_cluster.info_from_payload("box", {**base, "profile": bad}).profile)

    def test_the_node_route_reports_this_machine_s_profile(self):
        services = FakeServices()
        services.hardware = machine(8, 7)
        with mock.patch.object(api_app, "WEB_OUT", Path(tempfile.gettempdir()) / "no-such-web-out"):
            client = TestClient(create_app(services, open=True, hosts=ANY))
        body = client.get("/node").json()
        self.assertEqual(body["profile"]["vram_free"], 7 * G)
        FakeServices().hardware = None
        self.assertIsNone(TestClient(create_app(FakeServices(), open=True, hosts=ANY)).get("/node").json()["profile"])


class NoSoloScalingTests(unittest.TestCase):
    def test_the_queue_never_refits_a_distributed_run_or_its_members(self):
        import studio_queue
        from studio_ia.services import Job
        queue = studio_queue.Runner.__new__(studio_queue.Runner)
        queue.profile = lambda: self.fail("a distributed job was fitted as a solo run")
        for kind in ("pretrain_distributed", "diloco_member"):
            job = Job(id="j", kind=kind, params={"config": {"batch_size": 16}})
            self.assertEqual(queue._fitted(job, 1.), (job.params, job.note, ""))


if __name__ == "__main__":
    unittest.main()
