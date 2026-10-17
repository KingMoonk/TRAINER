"""studio_ia.fake_distributed: the simulated run `studio_api --fake` shows, advanced on a fake clock, and read the way the Training tab reads it."""
import unittest

import studio_distributed as runs
from studio_diloco import Stale
from studio_ia import fake_distributed as fake
from studio_ia.fake_distributed import FakeCoordinator
from tests.test_diloco import Clock


def machines():
    return [runs.Machine("mac", "local", device="mps", device_name="Apple M2"), runs.Machine("box", "box:8787", device="cuda", device_name="RTX 4090")]


class Simulation(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        run = runs.Run("fake1", "label", 1, 0., {"rounds": 3, "inner_steps": 100}, "mac", machines())
        self.coordinator = FakeCoordinator(run, {"rounds": 3, "inner_steps": 100, "delta_dtype": "float16"}, clock=self.clock, unit=1.)
        self.coordinator.start()

    def at(self, beat):
        self.clock.now = beat
        return self.coordinator.snapshot()

    def test_two_members_one_cuda_one_mps_pass_through_the_phases_of_a_round_at_their_own_speed(self):
        early = self.at(.2)["nodes"]
        self.assertEqual((early["mac"]["progress"]["phase"], early["box"]["progress"]["phase"]), ("downloading", "downloading"))
        middle = self.at(2.)
        self.assertEqual((middle["nodes"]["mac"]["progress"]["phase"], middle["nodes"]["box"]["progress"]["phase"]), ("training", "training"))
        self.assertGreater(middle["nodes"]["box"]["progress"]["step"], middle["nodes"]["mac"]["progress"]["step"])  # the CUDA machine is ahead
        self.assertGreater(middle["nodes"]["box"]["progress"]["tokens_per_second"], middle["nodes"]["mac"]["progress"]["tokens_per_second"])
        self.assertEqual(middle["phase"], "training")
        later = self.at(3.)
        self.assertEqual(later["nodes"]["box"]["progress"]["phase"], "uploading")
        self.assertEqual(later["phase"], "collecting")
        waiting = self.at(4.)
        self.assertEqual(waiting["nodes"]["box"]["sent"], True)
        self.assertEqual(waiting["nodes"]["mac"]["sent"], False)

    def test_a_round_closes_with_an_outer_step_and_a_check_then_the_next_opens(self):
        self.assertEqual(self.at(4.5)["phase"], "collecting")
        self.assertEqual(self.at(5.2)["phase"], "outer_step")
        self.assertEqual(self.at(5.7)["phase"], "validating")
        done = self.at(6.01)
        self.assertEqual((done["round"], done["phase"], len(done["history"])), (1, "next_round", 1))
        row = done["history"][0]
        self.assertEqual((row["contributed"], row["dropped"], row["delta_dtype"]), (["mac", "box"], [], "float16"))
        self.assertEqual([e["kind"] for e in done["events"]], ["joined", "joined", "outer_step", "checkpoint"])
        line = done["timeline"][0]
        self.assertEqual((line["round"], line["open"]), (0, False))
        self.assertTrue(all(lane["counted"] for lane in line["lanes"].values()))
        self.assertLess(line["lanes"]["box"]["delta_received"], line["lanes"]["mac"]["delta_received"])
        self.assertEqual(done["timeline"][-1]["round"], 1)  # the open round follows

    def test_the_run_finishes_after_its_rounds(self):
        end = self.at(100.)
        self.assertEqual((end["state"], end["phase"], end["round"]), ("finished", "finished", 3))
        self.assertEqual(end["events"][-1]["kind"], "finished")
        self.assertEqual(self.at(500.)["now_s"], end["now_s"])  # the clock stops with the run

    def test_stopping_a_member_mid_round_leaves_the_round_to_the_others_and_counts_it_for_nothing(self):
        self.at(2.)
        self.coordinator.drop("mac")
        view = self.at(10.)
        self.assertEqual(view["nodes"]["mac"]["state"], "stopped")
        first = view["history"][0]
        self.assertEqual((first["contributed"], first["dropped"]), (["box"], ["mac"]))
        self.assertFalse(view["timeline"][0]["lanes"]["mac"]["counted"])
        self.assertIsNone(view["timeline"][0]["lanes"]["mac"]["delta_received"])
        self.assertIn("left", [e["kind"] for e in view["events"]])
        with self.assertRaises(Stale):
            self.coordinator.drop("mac")
        with self.assertRaises(Stale):
            self.coordinator.drop("nobody")

    def test_stopping_every_member_leaves_the_run_waiting(self):
        self.at(2.)
        self.coordinator.drop("mac")
        self.coordinator.drop("box")
        view = self.at(60.)
        self.assertEqual((view["state"], view["phase"], view["round"]), ("waiting", "waiting", 0))

    def test_stop_freezes_the_run(self):
        self.at(2.)
        self.coordinator.stop()
        view = self.at(50.)
        self.assertEqual((view["state"], view["phase"], view["round"]), ("stopped", "stopped", 0))
        self.assertEqual(view["events"][-1]["kind"], "finished")

    def test_the_deadline_counts_down_to_the_window(self):
        first, second = self.at(1.)["deadline_s"], self.at(2.)["deadline_s"]
        self.assertAlmostEqual(first - second, 1.)
        self.assertAlmostEqual(first, fake.SLACK * 5. - 1.)

    def test_the_view_is_what_the_tab_reads(self):
        self.at(2.)
        run = runs.view(self.coordinator.run, self.coordinator, None)
        self.assertEqual(run["phase"], "training")
        mac, box = run["members"]
        self.assertEqual((mac["device"], box["device"]), ("mps", "cuda"))
        self.assertEqual(box["phase"], "training")
        self.assertGreater(len(box["samples"]), 1)
        self.assertEqual(run["timeline"][-1]["lanes"][0]["node"], "mac")
        self.assertEqual(run["theta_bytes"], fake.BYTES)

    def test_a_snapshot_stays_small(self):
        view = self.at(2.)
        self.assertLessEqual(len(view["nodes"]["box"]["progress"]["samples"]), 240)


if __name__ == "__main__":
    unittest.main()
