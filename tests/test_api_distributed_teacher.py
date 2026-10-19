"""A cluster run started with a teacher (polish, win-rate target): the real route over a real workspace, then the real coordinator the queue
builds from the job it stored, with the teacher check and the teacher itself replaced (nothing here contacts Ollama or the network)."""
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

    def coordinator_of(self, job, folder="run"):
        with mock.patch.object(studio_diloco, "check_teacher") as check:  # the model is "there"; no Ollama is asked
            coordinator = studio_diloco.prepare(job, self.services, self.root / folder, 8787)
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

    def prompts_of(self, coordinator, name):
        lines = (coordinator.directory / name).read_text(encoding="utf-8").splitlines()
        return [json.dumps(json.loads(line)["messages"], sort_keys=True) for line in lines if line.strip()]

    def test_the_coordinator_gets_the_conversations_apart_from_the_text(self):
        self.assertEqual(self.start(teacher="judge:1b", polish_every=1, target_win_rate=.5).status_code, 201)
        self.assertEqual(self.start().status_code, 201)
        teaching, plain = self.coordinator_of(self.queued[0]), None
        studio_diloco.COORDINATORS.clear()
        plain = studio_diloco.prepare(self.queued[1], self.services, self.root / "plain", 8787)
        self.addCleanup(plain.release)
        # what the members train on is what a run without a teacher trains on: the text, the same split, the same windows
        self.assertEqual([e for e in self.services.examples(self.queued[0].params)], [e for e in self.services.examples(self.queued[1].params)])
        self.assertTrue(all(e.get("text") for e in self.services.examples(self.queued[0].params)))
        self.assertEqual(teaching.split_hash, plain.split_hash)
        self.assertEqual(teaching.tokenizer.serialize(), plain.tokenizer.serialize())
        self.assertEqual([len(teaching.train), len(teaching.val)], [len(plain.train), len(plain.val)])
        polish, versus = self.prompts_of(teaching, "prompts.jsonl"), self.prompts_of(teaching, "eval_prompts.jsonl")
        self.assertGreater(len(polish), 0)
        self.assertEqual(len(versus), studio_diloco.EVAL_PROMPTS)
        self.assertEqual(set(polish) & set(versus), set())  # a held-out prompt is never a polish prompt
        self.assertEqual(len(polish) + len(versus), len(GOOD_CHATS) + len(PENDING_CHATS))
        text = "".join(teaching.tokenizer.decode(inputs) for windows in (teaching.train, teaching.val) for inputs, _ in windows)
        self.assertIn("quick brown fox", text)
        self.assertNotIn("plus", text)  # no conversation was trained on

    def test_one_option_alone_uses_all_the_conversations_for_itself(self):
        self.assertEqual(self.start(teacher="judge:1b", polish_every=1).status_code, 201)
        coordinator = self.coordinator_of(self.queued[0])
        self.assertEqual(len(self.prompts_of(coordinator, "prompts.jsonl")), len(GOOD_CHATS) + len(PENDING_CHATS))
        self.assertFalse((coordinator.directory / "eval_prompts.jsonl").exists())
        studio_diloco.COORDINATORS.clear()
        self.assertEqual(self.start(teacher="judge:1b", target_win_rate=.5).status_code, 201)
        other = self.coordinator_of(self.queued[1], "run2")
        self.assertEqual(len(self.prompts_of(other, "eval_prompts.jsonl")), studio_diloco.EVAL_PROMPTS)
        self.assertFalse((other.directory / "prompts.jsonl").exists())

    def test_a_reviewed_only_run_takes_the_good_conversations_only(self):
        self.assertEqual(self.start(teacher="judge:1b", polish_every=1, reviewed_only=True).status_code, 201)
        coordinator = self.coordinator_of(self.queued[0])
        prompts = self.prompts_of(coordinator, "prompts.jsonl")
        self.assertEqual(len(prompts), len(GOOD_CHATS))
        self.assertTrue(all("plus" in p for p in prompts), prompts)

    def test_the_prompts_are_the_same_every_time_for_one_seed(self):
        chats = GOOD_CHATS + PENDING_CHATS
        settings = studio_diloco.Settings(rounds=4, polish_every=1, target_win_rate=.5, teacher="t")
        first = studio_diloco.split_prompts(chats, settings, 7)
        self.assertEqual(first, studio_diloco.split_prompts(list(chats), settings, 7))
        self.assertNotEqual(first, studio_diloco.split_prompts(chats, settings, 8))

    def few_chats(self, count, name="Few chats"):
        few = self.workspace.create_dataset(name)
        self.workspace.add(few, [{"text": t} for t in TEXT] + GOOD_CHATS[:count])
        return few

    def test_a_lone_conversation_can_only_polish(self):
        lone = self.few_chats(1)
        for body in ({"polish_every": 1, "target_win_rate": .5}, {"target_win_rate": .5}):  # nothing to hold out
            with self.subTest(body=body):
                response = self.start(dataset_ids=[lone], teacher="judge:1b", **body)
                self.assertEqual(response.status_code, 400)
                self.assertIn("at least", response.json()["detail"])
        self.assertEqual(self.queued, [])
        self.assertEqual(self.start(dataset_ids=[lone], teacher="judge:1b", polish_every=1).status_code, 201)  # polish alone may use fewer

    def test_a_win_rate_needs_a_few_held_out_prompts(self):
        least = studio_diloco.MIN_EVAL_PROMPTS
        self.assertEqual(self.start(dataset_ids=[self.few_chats(least - 1)], teacher="judge:1b", target_win_rate=.5).status_code, 400)
        self.assertEqual(self.start(dataset_ids=[self.few_chats(least, "Enough")], teacher="judge:1b", target_win_rate=.5).status_code, 201)
        both = self.start(dataset_ids=[self.few_chats(2 * least - 1, "Not enough for both")], teacher="judge:1b", polish_every=1, target_win_rate=.5)
        self.assertEqual(both.status_code, 400)
        self.assertIn(f"at least {2 * least}", both.json()["detail"])
        self.assertEqual(self.start(dataset_ids=[self.few_chats(2 * least, "Both")], teacher="judge:1b", polish_every=1, target_win_rate=.5).status_code, 201)
        # the coordinator holds the same line for a job that got past the route (data changed since, a direct caller)
        settings = studio_diloco.Settings(rounds=4, polish_every=1, target_win_rate=.5, teacher="t")
        with self.assertRaisesRegex(ValueError, "different prompts"):
            studio_diloco.split_prompts(GOOD_CHATS[:2 * least - 1], settings, 1)
        polish, versus = studio_diloco.split_prompts(GOOD_CHATS[:2 * least], settings, 1)
        self.assertEqual((len(polish), len(versus)), (least, least))

    def test_the_same_question_asked_twice_counts_once(self):
        twice = self.workspace.create_dataset("Twice")
        asked = [{"user": "Same question?", "assistant": f"Answer {n}.", "quality": "Good"} for n in range(6)]
        self.workspace.add(twice, [{"text": t} for t in TEXT] + asked)
        self.assertTrue(self.services.has_conversations([twice]))
        self.assertFalse(self.services.has_conversations([twice], at_least=2))
        response = self.start(dataset_ids=[twice], teacher="judge:1b", polish_every=1, target_win_rate=.5)
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.start(dataset_ids=[twice], teacher="judge:1b", polish_every=1).status_code, 201)  # one prompt will do to polish
        self.assertEqual(len(studio_diloco.unique_prompts(asked)), 1)

    def test_a_resume_keeps_its_prompts_when_the_conversations_are_gone(self):
        self.assertEqual(self.start(teacher="judge:1b", polish_every=1, target_win_rate=.5).status_code, 201)
        first = self.coordinator_of(self.queued[0])
        before = {name: (first.directory / name).read_text(encoding="utf-8") for name in ("prompts.jsonl", "eval_prompts.jsonl")}
        first._persist([], {})  # a round has closed: the state a restart reads
        first.release()
        studio_diloco.COORDINATORS.clear()
        # the chats are deleted (or no longer marked Good) since: the job's data now makes no prompt at all
        with mock.patch.object(self.services, "conversations", return_value=[]), mock.patch.object(studio_diloco, "check_teacher"):
            again = studio_diloco.prepare(self.queued[0], self.services, self.root / "run", 8787)
        self.addCleanup(again.release)
        for name, text in before.items():
            self.assertEqual((again.directory / name).read_text(encoding="utf-8"), text)
        studio_diloco.COORDINATORS.clear()
        # a new run in the same data still needs them
        with mock.patch.object(self.services, "conversations", return_value=[]), mock.patch.object(studio_diloco, "check_teacher"):
            with self.assertRaisesRegex(ValueError, "different prompts"):
                studio_diloco.prepare(self.queued[0], self.services, self.root / "fresh", 8787)

    def test_a_set_of_conversations_only_can_be_the_source_of_the_prompts(self):
        chats = self.workspace.create_dataset("Only chats")
        self.workspace.add(chats, GOOD_CHATS)
        text = self.workspace.create_dataset("Plain")
        self.workspace.add(text, [{"text": t} for t in TEXT])
        self.assertEqual(self.start(dataset_ids=[chats], teacher="judge:1b", polish_every=1).status_code, 400)  # no text at all
        self.assertEqual(self.start(dataset_ids=[chats]).status_code, 400)  # and without a teacher it is not a text set
        self.assertEqual(self.start(dataset_ids=[text, chats], teacher="judge:1b", polish_every=1).status_code, 201)
        coordinator = self.coordinator_of(self.queued[0])
        self.assertEqual(len(self.prompts_of(coordinator, "prompts.jsonl")), len(GOOD_CHATS))
        self.assertTrue(all(e.get("text") for e in self.services.examples(self.queued[0].params)))

    def test_a_conversation_that_cannot_make_a_prompt_does_not_count(self):
        ends_on_user = self.workspace.create_dataset("Unanswered")
        self.workspace.add(ends_on_user, [{"text": t} for t in TEXT] + [{"messages": [{"role": "user", "content": f"Hello {n}?"}]} for n in range(5)])
        self.assertFalse(self.services.has_conversations([ends_on_user]))
        response = self.start(dataset_ids=[ends_on_user], teacher="judge:1b", polish_every=1)
        self.assertEqual(response.status_code, 400, response.text)

    def test_a_coordinator_that_is_not_on_the_machines_tab_is_told_so_before_the_teacher_is_(self):
        response = self.start(coordinator="gpu-box", machines=["gpu-box"], teacher="judge:1b", polish_every=1)
        self.assertEqual(response.status_code, 400)
        self.assertIn("not on the Machines tab", response.json()["detail"])

    def test_the_chat_sets_are_listed_for_the_form(self):
        chats = self.workspace.create_dataset("Only chats")
        self.workspace.add(chats, GOOD_CHATS[:3])
        listed = {row["id"]: row for row in self.client.get("/distributed/chat_sets").json()}
        self.assertEqual(listed[chats], {"id": chats, "name": "Only chats", "count": 3})
        self.assertEqual(listed[self.set]["count"], len(GOOD_CHATS) + len(PENDING_CHATS))
        text = self.workspace.create_dataset("Plain")
        self.workspace.add(text, [{"text": t} for t in TEXT])
        self.assertNotIn(text, [row["id"] for row in self.client.get("/distributed/chat_sets").json()])

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
