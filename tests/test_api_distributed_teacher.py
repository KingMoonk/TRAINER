"""A cluster run started with a teacher (polish, win-rate target): the real route over a real workspace, then the real coordinator the queue
builds from the job it stored, with the teacher check and the teacher itself replaced (nothing here contacts Ollama or the network)."""
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
from studio_ia.services import FakeServices, WorkspaceServices
from studio_spec import default_spec

TEXT = [f"page {n}: " + "the quick brown fox jumps over the lazy dog. " * 3 for n in range(30)]
GOOD_CHATS = [{"user": f"What is {n} plus {n}?", "assistant": f"{n + n}.", "quality": "Good"} for n in range(24)]
PENDING_CHATS = [{"user": f"Name the colour of thing {n}.", "assistant": f"It is teal {n}.", "quality": "Review"} for n in range(24)]

real_own_info = studio_cluster.own_info


def probed(**kwargs):
    """own_info with the GPU and memory probes answered, so the test never imports torch."""
    return real_own_info(**kwargs, status={"device": "cpu", "name": "CPU"}, memory=16 * 2 ** 30)


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.addCleanup(runs.RUNS.clear)
        self.addCleanup(studio_diloco.COORDINATORS.clear)
        self.enterContext(mock.patch.object(studio_cluster, "own_info", probed))
        self.queued = []

    def client_of(self, services):
        def enqueue(job):  # the job the route stores, kept without a worker to run it
            queued = studio_queue.Job(**asdict(job))
            self.queued.append(queued)
            services.runner.jobs.append(queued)

        if hasattr(services, "runner"):
            self.enterContext(mock.patch.object(services, "enqueue", enqueue))
        self.enterContext(mock.patch.object(services, "start_queue"))
        app = create_app(services, open=True, hosts=ANY)
        self.addCleanup(app.state.jobs.close)
        self.addCleanup(app.state.broadcaster.close)
        return self.enterContext(TestClient(app))


class Reaches(Base):
    """Over a real workspace: what the route stores and what the coordinator is built from."""

    def setUp(self):
        super().setUp()
        self.workspace = Workspace(self.root / "ws")
        self.services = WorkspaceServices(self.workspace)
        self.addCleanup(self.services.close)
        self.services.save_version(default_spec(), "first")
        self.set = self.workspace.create_dataset("Pages and chats")
        self.workspace.add(self.set, [{"text": t, "quality": "Good"} for t in TEXT] + GOOD_CHATS + PENDING_CHATS)
        self.client = self.client_of(self.services)

    def start(self, **body):
        return self.client.post("/distributed/runs", json={"dataset_ids": [self.set], "rounds": 3, "inner_steps": 2, **body})

    def coordinator_of(self, job):
        with mock.patch.object(studio_diloco, "check_teacher") as check:  # the model is "there"; no Ollama is asked
            coordinator = studio_diloco.prepare(job, self.services, self.root / "run", 8787)
        self.addCleanup(coordinator.release)
        self.checked = check
        return coordinator

    def test_a_run_without_the_options_is_as_before(self):
        response = self.start()
        self.assertEqual(response.status_code, 201, response.text)
        params = self.queued[0].params
        for key in ("teacher", "polish_every", "target_win_rate", "conversations"):
            self.assertNotIn(key, params)
        self.assertEqual((params["settings"]["teacher"], params["settings"]["polish_every"], params["settings"]["target_win_rate"]), ("", 0, None))
        coordinator = self.coordinator_of(self.queued[0])
        self.assertFalse((coordinator.directory / "prompts.jsonl").exists())
        self.assertFalse((coordinator.directory / "eval_prompts.jsonl").exists())
        self.assertEqual([e for e in self.services.examples(params) if not e.get("text")], [])  # text only, as today

    def test_each_option_reaches_the_coordinators_settings(self):
        for body, expected in (({"teacher": "judge:1b", "polish_every": 1}, ("judge:1b", 1, None)),
                               ({"teacher": "judge:1b", "target_win_rate": .6}, ("judge:1b", 0, .6)),
                               ({"teacher": " judge:1b ", "polish_every": 2, "target_win_rate": 1}, ("judge:1b", 2, 1.))):
            with self.subTest(body=body):
                self.queued.clear()
                response = self.start(**body)
                self.assertEqual(response.status_code, 201, response.text)
                params = self.queued[0].params
                self.assertIs(params["conversations"], True)
                self.assertEqual((params["settings"]["teacher"], params["settings"]["polish_every"], params["settings"]["target_win_rate"]), expected)
                coordinator = self.coordinator_of(self.queued[0])
                settings = coordinator.settings
                self.assertEqual((settings.teacher, settings.polish_every, settings.target_win_rate), expected)
                self.checked.assert_called_once()
                self.assertEqual(self.checked.call_args.args[0]["teacher_model"], "judge:1b")
                studio_diloco.COORDINATORS.clear()

    def test_the_coordinator_gets_the_conversations_with_the_text(self):
        self.assertEqual(self.start(teacher="judge:1b", polish_every=1, target_win_rate=.5).status_code, 201)
        coordinator = self.coordinator_of(self.queued[0])
        polish = [line for line in coordinator.prompt_window(1)]
        self.assertGreater(len(polish), 0)
        self.assertTrue(all(p["messages"][-1]["role"] == "user" for p in polish))
        evals = (coordinator.directory / "eval_prompts.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertGreater(len(evals), 0)
        text = "".join(coordinator.tokenizer.decode(inputs) for windows in (coordinator.train, coordinator.val) for inputs, _ in windows)
        self.assertIn("quick brown fox", text)  # the text the members pretrain on is still there

    def test_a_reviewed_only_run_takes_the_good_conversations_only(self):
        self.assertEqual(self.start(teacher="judge:1b", polish_every=1, reviewed_only=True).status_code, 201)
        coordinator = self.coordinator_of(self.queued[0])
        prompts = [p["messages"][-1]["content"] for p in coordinator.prompt_window(1)]
        self.assertGreater(len(prompts), 0)
        self.assertTrue(all("plus" in p for p in prompts), prompts)
        rows = self.services.examples(self.queued[0].params)
        self.assertEqual({e["quality"] for e in rows}, {"Good"})
        self.assertEqual(len([e for e in rows if not e.get("text")]), len(GOOD_CHATS))

    def test_a_start_with_no_conversations_is_a_400(self):
        text_only = self.workspace.create_dataset("Plain")
        self.workspace.add(text_only, [{"text": t, "quality": "Good"} for t in TEXT])
        response = self.start(dataset_ids=[text_only], teacher="judge:1b", target_win_rate=.5)
        self.assertEqual(response.status_code, 400)
        self.assertIn("conversations", response.json()["detail"])
        self.assertEqual(self.start(dataset_ids=[text_only]).status_code, 201)  # without the options the same set is fine

    def test_a_reviewed_only_start_with_no_good_conversation_is_a_400(self):
        unreviewed = self.workspace.create_dataset("Unreviewed chats")
        self.workspace.add(unreviewed, [{"text": t, "quality": "Good"} for t in TEXT] + PENDING_CHATS)
        response = self.start(dataset_ids=[unreviewed], teacher="judge:1b", polish_every=1, reviewed_only=True)
        self.assertEqual(response.status_code, 400)
        self.assertIn("marked Good", response.json()["detail"])
        self.assertEqual(self.start(dataset_ids=[unreviewed], teacher="judge:1b", polish_every=1).status_code, 201)  # not reviewed only: they count

    def test_unusable_combinations_are_a_400_with_a_sentence(self):
        for body, word in (({"target_win_rate": .5}, "need a teacher"),
                           ({"polish_every": 1}, "need a teacher"),
                           ({"teacher": "judge:1b"}, "does nothing"),
                           ({"teacher": "  ", "polish_every": 1}, "need a teacher"),
                           ({"teacher": "judge:1b", "polish_every": 3}, "below rounds")):
            with self.subTest(body=body):
                response = self.start(**body)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn(word, response.json()["detail"])
        self.assertEqual(self.queued, [])

    def test_out_of_bounds_numbers_are_refused(self):
        for body in ({"polish_every": -1}, {"polish_every": 1.5}, {"target_win_rate": 0}, {"target_win_rate": 1.2}, {"target_win_rate": -.1}):
            with self.subTest(body=body):
                self.assertEqual(self.start(teacher="judge:1b", **body).status_code, 422)
        self.assertEqual(self.queued, [])


class OnTheFakeStudio(Base):
    """The route alone, over the fake services the web tests use."""

    def test_a_fake_studio_without_conversations_refuses(self):
        services = FakeServices()
        services.save_version(default_spec(), "first")
        services.conversations_in_text = False
        client = self.client_of(services)
        set_id = services.data("text").sets[0]["id"]
        response = client.post("/distributed/runs", json={"dataset_ids": [set_id], "teacher": "judge:1b", "polish_every": 1, "rounds": 3})
        self.assertEqual(response.status_code, 400, response.text)
        services.conversations_in_text = True
        response = client.post("/distributed/runs", json={"dataset_ids": [set_id], "teacher": "judge:1b", "polish_every": 1, "rounds": 3})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(services.queue().up_next[0].params["teacher"], "judge:1b")


if __name__ == "__main__":
    unittest.main()
