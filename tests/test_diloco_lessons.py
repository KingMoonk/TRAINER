"""Reviewing the teacher's lessons in a DiLoCo run (#513): a real coordinator behind the real routes, a faked teacher in place of Ollama. A
dropped lesson is left out of the rounds that open after it; Keep deals it again; the choice survives a restart (nothing here contacts a
teacher or the network)."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import studio_distributed as runs
import studio_diloco
import studio_peer
from studio_api import app as api_app
from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_api.routers import distributed as router
from studio_api.routers.distributed import Run
from studio_ia.services import FakeServices
from tests import diloco_support as support
from tests.test_diloco_polish import Clock, Teacher

RUN = "run1"


class LessonReview(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "run"
        self.clock = Clock()
        self.addCleanup(runs.RUNS.clear)
        self.addCleanup(studio_diloco.COORDINATORS.clear)
        with mock.patch.object(api_app, "WEB_OUT", Path(tempfile.gettempdir()) / "no-such-web-out"):
            self.client = self.enterContext(TestClient(create_app(FakeServices(), open=True, hosts=ANY)))
        self.url = f"/distributed/runs/{RUN}/lessons"

    def coordinator(self):
        c = support.build(self.dir, run=RUN, clock=self.clock, inner_steps=2, rounds=8, polish_every=2, teacher="t")
        c.polisher = Teacher(win=False)
        losses = iter([5., 4., 4.5, 4.4, 4.3, 4.2, 4.1, 4.])
        c.evaluate = lambda theta: next(losses)
        runs.RUNS[RUN] = runs.Run(RUN, "lessons", None, time.time(), {}, "here")
        studio_diloco.COORDINATORS[RUN] = c
        return c

    def play(self, c, rounds):
        c.join("a")
        for _ in range(rounds):
            self.send(c)

    def send(self, c):
        number = c.next_round
        view = c.round_view(number, "a")
        self.assertEqual(view["state"], "open", view)
        self.clock.now += 1.
        c.submit(number, "a", view["generation"], 0, 10, 2, support.constant_delta(c, .01), 1.)
        end = time.monotonic() + 10
        while c.polishing and time.monotonic() < end:
            time.sleep(.01)
        self.assertFalse(c.polishing, "the polish did not finish")

    def dealt(self, c):
        """Every shard id the open round deals to any member."""
        return sorted({shard for ids in c.current.deal.values() for shard in ids})

    def statuses(self, lessons):
        return {lesson["id"]: lesson["status"] for lesson in lessons}

    def test_a_dropped_lesson_is_left_out_of_the_next_round_and_keep_puts_it_back(self):
        c = self.coordinator()
        self.play(c, 4)  # two polishes: lessons 4 (after round 1) and 5 (after round 3)
        self.assertEqual(self.dealt(c), [0, 1, 2, 3, 4, 5])

        listed = self.client.get(self.url)
        self.assertEqual(listed.status_code, 200, listed.text)
        lessons = listed.json()
        self.assertEqual([(lesson["id"], lesson["round"], lesson["rows"], lesson["status"], lesson["more"]) for lesson in lessons],
                         [(4, 1, 3, "live", 0), (5, 3, 3, "live", 0)])
        self.assertEqual(lessons[0]["items"][0], {"prompt": "What is 1 plus 1?", "reply": "no idea", "answer": "2 exactly."})

        dropped = self.client.post(f"{self.url}/4/drop")
        self.assertEqual(dropped.status_code, 200, dropped.text)
        self.assertEqual(self.statuses(dropped.json()), {4: "dropped", 5: "live"})
        self.assertEqual(self.dealt(c), [0, 1, 2, 3, 4, 5])  # the round in flight keeps its deal
        self.assertEqual(self.client.post(f"{self.url}/4/drop").status_code, 200)  # a second drop changes nothing
        self.send(c)
        self.assertEqual(self.dealt(c), [0, 1, 2, 3, 5])
        self.assertNotIn(4, c.round_view(c.next_round, "a")["shards"])
        self.assertTrue(c.shard_bytes(4))  # a member still fetching the round in flight can read it

        events = [e["kind"] for e in c.snapshot()["events"]]
        self.assertEqual(events.count("lesson_dropped"), 1)
        run = self.client.get(f"/distributed/runs/{RUN}")
        self.assertEqual(run.status_code, 200, run.text)
        Run.model_validate(run.json())
        self.assertEqual((run.json()["lessons"], run.json()["lessons_left_out"]), (3, 3))
        self.assertIn("lesson_dropped", [e["kind"] for e in run.json()["events"]])

        kept = self.client.post(f"{self.url}/4/keep")
        self.assertEqual(self.statuses(kept.json()), {4: "live", 5: "live"})
        self.assertEqual(self.client.post(f"{self.url}/4/keep").status_code, 200)
        self.assertEqual(self.dealt(c), [0, 1, 2, 3, 5])  # the round in flight is not changed by a Keep either
        self.send(c)  # a third polish follows this round: lesson 6 is new, lesson 4 is back
        self.assertEqual(self.dealt(c), [0, 1, 2, 3, 4, 5, 6])
        self.assertIn("lesson_kept", [e["kind"] for e in c.snapshot()["events"]])
        Run.model_validate(self.client.get(f"/distributed/runs/{RUN}").json())

    def test_the_choice_survives_a_restart(self):
        c = self.coordinator()
        self.play(c, 4)
        self.client.post(f"{self.url}/5/drop")
        self.assertEqual(support.read_json(self.dir / "diloco.json")["lessons"][1]["status"], "dropped")
        again = support.build(self.dir, run=RUN, clock=Clock(), inner_steps=2, rounds=8, polish_every=2, teacher="t")
        self.assertEqual(self.statuses(again.lesson_list()), {4: "live", 5: "dropped"})
        again.join("a")
        self.assertEqual(sorted(again.round_view(again.next_round, "a")["shards"]), [0, 1, 2, 3, 4])
        again.keep_lesson(5)  # its windows were rebuilt from lessons.jsonl
        self.assertEqual(sorted(again.live_lessons(), key=lambda e: e["shard"]), [e for e in again.lessons])

    def test_a_dropped_lessons_prompt_is_still_not_judged_on(self):
        c = self.coordinator()
        self.play(c, 2)
        before = c._lesson_prompts()
        self.client.post(f"{self.url}/4/drop")
        self.assertEqual(c._lesson_prompts(), before)
        self.assertTrue(before)

    def test_a_lesson_a_step_back_left_out_can_be_kept(self):
        c = self.coordinator()
        self.play(c, 2)
        c.lessons[0]["status"] = "left_out"
        c.lesson_windows.pop(4)
        self.assertTrue(self.client.get(self.url).json()[0]["keepable"])  # left out, and its rows are on this node
        self.assertEqual(self.statuses(self.client.post(f"{self.url}/4/keep").json()), {4: "live"})
        self.assertIn(4, c.lesson_windows)
        self.assertFalse(self.client.get(self.url).json()[0]["keepable"])  # dealt now
        c.lessons[0]["status"] = "left_out"
        (c.directory / "lessons.jsonl").write_text("", encoding="utf-8")
        self.assertFalse(self.client.get(self.url).json()[0]["keepable"])  # no rows here, as after a failover: Keep cannot work

    def test_the_errors_say_what_is_wrong(self):
        self.assertEqual(self.client.get("/distributed/runs/nope/lessons").status_code, 404)
        self.assertEqual(self.client.post("/distributed/runs/nope/lessons/4/drop").status_code, 404)
        c = self.coordinator()
        self.play(c, 2)
        missing = self.client.post(f"{self.url}/99/drop")
        self.assertEqual((missing.status_code, missing.json()["detail"]), (404, f"Run {RUN} has no lesson 99."))
        c.stop()
        over = self.client.post(f"{self.url}/4/drop")
        self.assertEqual((over.status_code, over.json()["detail"]), (409, "The run is over; its lessons can no longer change."))
        studio_diloco.COORDINATORS.clear()
        idle = self.client.get(self.url)
        self.assertEqual((idle.status_code, idle.json()["detail"]), (409, "The run is not going."))

    def test_a_forwarded_choice_keeps_the_owners_refusal(self):
        class Owner:
            def __init__(self, status):
                self.status = status

            def post_json(self, path, body=None):
                if self.status is None:
                    raise studio_peer.PeerError("box did not answer", status=None)
                raise studio_peer.PeerError(f"box answered {self.status}", status=self.status, detail=f"owner said {self.status}")

        class Cluster:
            status = 404

            def address_of(self, node):
                return "box"

            def client_for(self, address, **options):
                return Owner(self.status)

        cluster = Cluster()
        row = {"id": RUN, "origin": "box"}
        url = f"{self.url}/99/drop"
        with mock.patch.object(router, "_remote", return_value=row), mock.patch.object(router, "_cluster", return_value=cluster):
            for status, expected in ((404, 404), (409, 409), (500, 502), (None, 502)):
                cluster.status = status
                answer = self.client.post(url)
                self.assertEqual(answer.status_code, expected, (status, answer.json()))
            cluster.status = 404
            self.assertEqual(self.client.post(url).json()["detail"], "owner said 404")


if __name__ == "__main__":
    unittest.main()
