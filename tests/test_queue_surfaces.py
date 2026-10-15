"""Sidebar areas and the queue outlook on every "add to the queue" surface (window-free)."""
import unittest
from datetime import datetime

from studio_ia.pages.builder import queued_outlook
from studio_ia.pages.post_training import round_outlook
from studio_ia.services import FakeServices, Job, QueueState, if_added
from studio_ia.shell import AREA_HOME, AREAS, ROUTES, UTILITY_AREAS
from studio_queue import Gate

NIGHT = Gate("outside hours", datetime(2099, 1, 1, 22))


def fake(**state):
    f = FakeServices()
    f.set_queue(QueueState(**state))
    return f


class Areas(unittest.TestCase):
    def test_queue_and_resources_are_rows_of_their_own(self):
        names = dict(AREAS)
        self.assertEqual((names["queue"], AREA_HOME["queue"]), ("Queue", "queue"))
        self.assertEqual((names["resources"], AREA_HOME["resources"]), ("Resources", "your_data"))
        self.assertLess([k for k, _ in AREAS].index("chat"), [k for k, _ in AREAS].index("queue"))
        self.assertEqual([ROUTES[r].area for r in ("queue", "your_data", "make_data", "storage")], ["queue", "resources", "resources", "resources"])
        self.assertEqual(ROUTES["storage"].title, "Storage")
        self.assertNotIn("This computer", names.values())

    def test_every_route_lights_a_row_and_every_row_opens_a_page_in_its_area(self):
        keys = [k for k, _ in AREAS]
        self.assertEqual(sorted(keys), sorted(AREA_HOME))
        for route, spec in ROUTES.items():
            self.assertIn(spec.area, keys, route)
        for key, home in AREA_HOME.items():
            self.assertEqual(ROUTES[home].area, key)
        self.assertTrue(set(UTILITY_AREAS) <= set(keys))

    def test_pretraining_keeps_its_own_routes(self):
        self.assertEqual((ROUTES["pretraining"].area, ROUTES["training_run"].area), ("pretraining", "pretraining"))


class OutlookText(unittest.TestCase):
    def test_if_added_reads_as_a_sentence(self):
        self.assertEqual(if_added("Starts now"), "If you add it: starts now")
        self.assertEqual(if_added("Paused by you"), "If you add it: paused by you")

    def test_builder_job_outlook_follows_the_queue(self):
        mine, other = Job("mine", "pretrain", "Version 2"), Job("other", "quicktest", "Quick test", eta_s=600.)
        self.assertEqual(queued_outlook(fake(up_next=[mine]), "mine"), "Starts now")
        self.assertEqual(queued_outlook(fake(up_next=[other, mine]), "mine"), "Starts after Quick test (about 10 min)")
        self.assertEqual(queued_outlook(fake(job=mine), "mine"), "Running now")
        self.assertEqual(queued_outlook(fake(up_next=[mine], paused=True), "mine"), "Paused by you")
        self.assertEqual(queued_outlook(fake(up_next=[mine], gate=Gate("on battery")), "mine"), "Waiting: on battery")
        self.assertEqual(queued_outlook(fake(), "mine"), "")  # it left the queue
        waits = queued_outlook(fake(up_next=[mine], gate=NIGHT, plan={"hours": "nights"}), "mine")
        self.assertTrue(waits.startswith("Starts ") and waits.endswith("at 22:00 (plan: nights only)"), waits)

    def test_post_training_round_before_and_after_adding(self):
        self.assertEqual(round_outlook(fake(), 0, "idle"), "If you add it: starts now")
        self.assertEqual(round_outlook(fake(paused=True), 0, "idle"), "If you add it: paused by you")
        running = Job("r", "pretrain", "Version 3")
        self.assertEqual(round_outlook(fake(job=running), 0, "idle"), "If you add it: starts after Version 3")
        queued = Job("q", "post_round", "Round 2", params={"round": 2})
        self.assertEqual(round_outlook(fake(job=running, up_next=[queued]), 1, "queued"), "Starts after Version 3")
        self.assertEqual(round_outlook(fake(up_next=[queued]), 1, "queued"), "Starts now")

    def test_move_to_the_front_starts_when_the_head_of_the_queue_would(self):
        running = Job("r", "pretrain", "Version 3", eta_s=None)
        head = Job("h", "quicktest", "First")
        held = Job("x", "quicktest", "Held", status="held")
        self.assertEqual(round_outlook(fake(job=running, up_next=[held, head]), 2, "idle"), "If you add it: starts after Version 3")
        self.assertEqual(round_outlook(fake(up_next=[held, head]), 2, "idle"), "If you add it: starts now")


if __name__ == "__main__":
    unittest.main()
