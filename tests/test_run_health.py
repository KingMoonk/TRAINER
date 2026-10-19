"""studio_run_health: what a distributed round measures, and the verdict on whether the cluster helps. Window-free."""
import tempfile
import unittest
from pathlib import Path

import studio_distributed as runs
import studio_run_health as health
from studio_ia.fake_distributed import FakeCoordinator
from tests import diloco_support as support
from tests.test_diloco import Clock


def make_row(number, compute, arrive, seconds=10., val_loss=2., tokens_per_s=None):
    """A history row of a round where each node in `compute` (seconds) trained at `tokens_per_s[node]` and its change arrived at `arrive[node]`."""
    rates = tokens_per_s or {node: 1000. for node in compute}
    tokens = {node: int(rates[node] * compute[node]) for node in compute}
    row = {"round": number, "seconds": seconds, "contributed": list(compute), "node_seconds": dict(compute), "node_tokens": tokens,
           "steps": {node: 100 for node in compute}, "tokens": sum(tokens.values()), "val_loss": val_loss, "loss": val_loss + .1, "lr": .7,
           "outer_step_norm": .05, "delta_bytes": 2_000_000 * len(compute), "theta_bytes": 8_000_000 * len(compute), "delta_dtype": "float16",
           "node_bytes": {node: 2_000_000 for node in compute}}
    row["health"] = health.record(row, {node: val_loss + .1 for node in compute}, arrive, 4_000_000)
    return row


def rounds(count, compute, arrive, losses=None, **kw):
    return [make_row(i, compute, arrive, val_loss=(losses[i] if losses else 2. - .1 * i), **kw) for i in range(count)]


class Recording(unittest.TestCase):
    def test_a_fake_two_member_run_records_every_metric_each_round(self):
        clock = Clock()
        run = runs.Run("h1", "label", 1, 0., {"rounds": 2, "inner_steps": 100},
                       "mac", [runs.Machine("mac", "local", device="mps", device_name="M2"), runs.Machine("box", "box:8787", device="cuda", device_name="4090")])
        coordinator = FakeCoordinator(run, {"rounds": 2, "inner_steps": 100, "delta_dtype": "float16"}, clock=clock, unit=1.)
        coordinator.start()
        clock.now = 100.
        snapshot = coordinator.snapshot()
        record = snapshot["history"][0]["health"]
        self.assertEqual(set(record["members"]), {"mac", "box"})
        for member in record["members"].values():
            self.assertTrue(all(member[key] is not None for key in ("loss", "steps", "tokens", "compute_s", "tokens_per_s", "wait_s", "sync_s")))
            self.assertAlmostEqual(member["compute_s"] + member["wait_s"] + member["sync_s"], record["seconds"])
        self.assertEqual(record["slowest"], "mac")  # the mps machine's change comes in last, so the cuda one waited for it
        self.assertEqual(record["members"]["mac"]["wait_s"], 0.)
        self.assertGreater(record["members"]["box"]["wait_s"], 0.)
        self.assertEqual((record["bytes_up"], record["compression"], record["dtype"]), (2_000_000 * 2, 2., "float16"))
        self.assertTrue(record["val_loss"] and record["lr"] is not None and record["update_norm"])
        found = health.summarize(snapshot["history"])
        self.assertEqual(found["baseline_node"], "box")
        self.assertEqual(found["verdict"]["state"], health.NOT_HELPING)  # the cuda machine alone is faster than the pair
        self.assertIn("box alone", found["verdict"]["text"])
        self.assertEqual(runs.view(run, coordinator, None)["health"]["verdict"], found["verdict"])

    def test_the_real_coordinator_records_each_members_wait_from_when_its_change_arrived(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            clock = Clock()
            c = support.build(Path(tmp), clock=clock, inner_steps=2, rounds=2, min_members=2, first_round_seconds=100., grace_seconds=10.)
            c.join("a")
            c.join("b")
            for node, when, seconds, tokens in (("a", 3., 2., 400), ("b", 8., 6., 600)):
                clock.now = when
                view = c.round_view(0, node)
                c.submit(0, node, view["generation"], 0, tokens, 2, support.constant_delta(c), seconds, 2.5, 1000)
            record = c.history[0]["health"]
            self.assertEqual((record["members"]["a"]["wait_s"], record["members"]["b"]["wait_s"], record["slowest"]), (5., 0., "b"))
            self.assertEqual((record["members"]["a"]["tokens_per_s"], record["members"]["a"]["loss"], record["bytes_up"]), (200., 2.5, 2000))
            self.assertEqual(record["seconds"], 8.)
            self.assertAlmostEqual(record["members"]["a"]["compute_s"] + 5. + record["members"]["a"]["sync_s"], 8.)

    def test_a_member_without_a_recorded_arrival_has_no_wait(self):
        record = make_row(0, {"a": 4., "b": 6.}, {"a": 5.})["health"]
        self.assertIsNone(record["members"]["b"]["wait_s"])
        self.assertEqual(record["members"]["a"]["wait_s"], 0.)


class Verdict(unittest.TestCase):
    def test_working_when_the_pair_is_faster_than_the_fastest_alone_and_loss_falls(self):
        # two equal machines at 1000 tokens/s each, 8 s of compute in a 10 s round: 1600 tokens/s against 1000 alone
        found = health.summarize(rounds(4, {"a": 8., "b": 8.}, {"a": 9., "b": 9.}))
        self.assertEqual(found["verdict"]["state"], health.WORKING)
        self.assertAlmostEqual(found["speedup"], 1.6)
        self.assertEqual(found["baseline_tokens_per_s"], 1000.)
        self.assertIn("1.6× faster than", found["verdict"]["text"])
        self.assertIn("loss falling", found["verdict"]["text"])
        self.assertAlmostEqual(found["comm_share"], .2)

    def test_not_helping_when_the_fast_member_waits_for_the_slow_one(self):
        rows = [make_row(i, {"beast": 3., "mac": 9.}, {"beast": 3.5, "mac": 9.5}, seconds=10., tokens_per_s={"beast": 4000., "mac": 500.},
                         val_loss=2. - .1 * i) for i in range(4)]
        found = health.summarize(rows)  # beast's change is in 6 s before mac's: 60% of every round
        self.assertEqual(found["verdict"]["state"], health.NOT_HELPING)
        self.assertEqual(found["verdict"]["text"], "Not helping: beast waits 60% of each round for mac.")
        self.assertEqual((found["wait_node"], found["straggler"]), ("beast", "mac"))

    def test_not_helping_when_communication_eats_the_gain(self):
        found = health.summarize(rounds(3, {"a": 4., "b": 4.}, {"a": 9., "b": 9.}))  # 4 s of compute in 10 s: 800 tokens/s against 1000
        self.assertEqual(found["verdict"]["state"], health.NOT_HELPING)
        self.assertIn("communication takes 60% of each round", found["verdict"]["text"])

    def test_diverging_when_the_outer_loss_rises_three_rounds_running(self):
        found = health.summarize(rounds(5, {"a": 8., "b": 8.}, {"a": 9., "b": 9.}, losses=[2., 1.9, 2., 2.2, 2.6]))
        self.assertEqual(found["verdict"], {"state": "diverging", "text": "Diverging: the loss rose 3 rounds running."})
        steady = health.summarize(rounds(5, {"a": 8., "b": 8.}, {"a": 9., "b": 9.}, losses=[2., 1.9, 2., 2.2, 2.1]))
        self.assertEqual(steady["verdict"]["state"], health.WORKING)

    def test_not_enough_data_instead_of_a_guess(self):
        self.assertEqual(health.summarize([])["verdict"]["state"], health.NOT_ENOUGH)
        one = health.summarize(rounds(6, {"a": 8.}, {"a": 8.}))  # a single member has nothing to be compared with
        self.assertEqual((one["verdict"]["state"], one["speedup"]), (health.NOT_ENOUGH, None))
        early = health.summarize(rounds(1, {"a": 8., "b": 8.}, {"a": 9., "b": 9.}))
        self.assertEqual((early["verdict"]["state"], early["speedup"]), (health.NOT_ENOUGH, None))
        self.assertIn("1 so far", early["verdict"]["text"])

    def test_rows_from_before_health_was_recorded_are_skipped(self):
        found = health.summarize([{"round": 0, "seconds": 1.}])
        self.assertEqual((found["rounds"], found["verdict"]["state"]), ([], health.NOT_ENOUGH))


if __name__ == "__main__":
    unittest.main()
