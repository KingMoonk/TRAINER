"""A cluster run started with reviewed_only trains only on the examples marked Good: the real route over a real workspace, then the real
coordinator the queue builds from the job it stored."""
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import studio_cluster
import studio_diloco
import studio_distributed as runs
import studio_queue
from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_core import Workspace
from studio_ia.services import WorkspaceServices
from studio_spec import default_spec

GOOD = [f"good page {n}: " + "the quick brown fox jumps over the lazy dog. " * 3 for n in range(30)]
PENDING = [f"pending page {n}: " + "a slow green turtle ~ walks under the old stone bridge. " * 3 for n in range(30)]  # "~" is in no good page

real_own_info = studio_cluster.own_info


def probed(**kwargs):
    """own_info with the GPU and memory probes answered, so the test never imports torch."""
    return real_own_info(**kwargs, status={"device": "cpu", "name": "CPU"}, memory=16 * 2 ** 30)


class ReviewedOnly(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.addCleanup(runs.RUNS.clear)
        self.addCleanup(studio_diloco.COORDINATORS.clear)
        self.enterContext(mock.patch.object(studio_cluster, "own_info", probed))
        self.workspace = Workspace(self.root / "ws")
        self.services = WorkspaceServices(self.workspace)
        self.addCleanup(self.services.close)
        self.services.save_version(default_spec(), "first")
        self.set = self.workspace.create_dataset("Pages")
        self.workspace.add(self.set, [{"text": t, "quality": "Good"} for t in GOOD] + [{"text": t, "quality": "Review"} for t in PENDING])
        self.queued = []

        def enqueue(job):  # the job the route stores, kept without a worker to run it
            queued = studio_queue.Job(**asdict(job))
            self.queued.append(queued)
            self.services.runner.jobs.append(queued)

        self.enterContext(mock.patch.object(self.services, "enqueue", enqueue))
        self.enterContext(mock.patch.object(self.services, "start_queue"))
        app = create_app(self.services, open=True, hosts=ANY)
        self.addCleanup(app.state.jobs.close)
        self.addCleanup(app.state.broadcaster.close)
        self.client = self.enterContext(TestClient(app))

    def start(self, **body):
        return self.client.post("/distributed/runs", json={"dataset_ids": [self.set], "rounds": 2, "inner_steps": 2, **body})

    def coordinator_of(self, job):
        """The coordinator the queue builds from the stored job, and the text of every window it trains and validates on."""
        coordinator = studio_diloco.prepare(job, self.services, self.root / "run", 8787)
        self.addCleanup(coordinator.release)
        text = "".join(coordinator.tokenizer.decode(inputs) for windows in (coordinator.train, coordinator.val) for inputs, _ in windows)
        return coordinator, text

    def test_the_run_says_it_trains_on_reviewed_examples_only(self):
        for body, expected in (({}, False), ({"reviewed_only": False}, False), ({"reviewed_only": True}, True)):
            with self.subTest(body=body):
                response = self.start(**body)
                self.assertEqual(response.status_code, 201, response.text)
                self.assertIs(response.json()["reviewed_only"], expected)
                self.assertIs(self.client.get(f"/distributed/runs/{response.json()['id']}").json()["reviewed_only"], expected)
        self.assertNotIn("reviewed_only", self.queued[0].params)
        self.assertIs(self.queued[2].params["reviewed_only"], True)

    def test_only_good_examples_are_in_the_coordinators_training_and_validation_data(self):
        self.assertEqual(self.start(reviewed_only=True).status_code, 201)
        coordinator, text = self.coordinator_of(self.queued[0])
        self.assertIn("good page", text)
        self.assertNotIn("~", text)
        self.assertNotIn("pending", text)
        self.assertGreater(len(coordinator.val), 0)  # the validation split is cut from the Good examples too

    def test_without_the_option_every_text_is_used(self):
        self.assertEqual(self.start().status_code, 201)
        _, text = self.coordinator_of(self.queued[0])
        self.assertIn("good page", text)
        self.assertIn("~", text)

    def test_a_good_example_held_back_for_scoring_does_not_count_as_text_to_train_on(self):
        lone = self.workspace.create_dataset("One good")
        held = "the only good page, kept back to score every version on"
        self.workspace.add(lone, [{"text": held, "quality": "Good"}] + [{"text": t, "quality": "Review"} for t in PENDING])
        path = self.services.root / "data" / "heldout.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([{"id": "heldout_0", "text": held}]), encoding="utf-8")
        response = self.start(dataset_ids=[lone], reviewed_only=True)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.queued, [])

    def test_one_good_example_among_many_unreviewed_is_either_refused_or_trainable(self):
        few = self.workspace.create_dataset("Almost none")
        self.workspace.add(few, [{"text": GOOD[0], "quality": "Good"}] + [{"text": t, "quality": "Review"} for t in PENDING])
        response = self.start(dataset_ids=[few], reviewed_only=True)
        if response.status_code == 201:  # accepted, so the coordinator must be able to build from it
            _, text = self.coordinator_of(self.queued[-1])
            self.assertNotIn("~", text)
        else:
            self.assertEqual(response.status_code, 400)

    def test_a_start_with_nothing_marked_good_is_a_400_with_a_sentence(self):
        empty = self.workspace.create_dataset("Unreviewed")
        self.workspace.add(empty, [{"text": t, "quality": "Review"} for t in PENDING])
        response = self.start(dataset_ids=[empty], reviewed_only=True)
        self.assertEqual(response.status_code, 400)
        self.assertIn("marked Good", response.json()["detail"])
        self.assertEqual(self.queued, [])
        self.assertEqual(self.start(dataset_ids=[empty]).status_code, 201)  # not asking for reviewed only: the same set is fine
        self.assertEqual(self.start(dataset_ids=[empty, self.set], reviewed_only=True).status_code, 201)  # one set with Good text is enough


if __name__ == "__main__":
    unittest.main()
