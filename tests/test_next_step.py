"""One next-step model (ordered states), and the three guards that read it: Compare's queue button, Head lab tiles, Your data's collection card."""
import unittest
from datetime import datetime

from studio_ia.next_step import NextStep, empty_line, next_step
from studio_ia.pages.compare import LOOKS, queue_block
from studio_ia.pages.head_lab import NO_PATTERN, PATTERN_WORDS, head_pattern
from studio_ia.pages.your_data import collection_shown
from studio_ia.services import FakeServices, Job, QueueState, Version
from studio_ia.shell import NEXT_EVENTS
from studio_queue import Gate
import studio_data_kinds as dk


def step(has_text=True, versions=(), **queue):
    return next_step(has_text, list(versions), QueueState(**queue))


SAVED = Version(1, status="saved")
DRAFT = Version(2, status="draft")
TRAINED = Version(1, status="trained", best_checkpoint="v1.npz")
RUNNING = Job("r", "pretrain", "Version 1")


class Order(unittest.TestCase):
    def test_no_text_comes_first_even_with_everything_else_done(self):
        self.assertEqual(step(False).route, "your_data")
        self.assertEqual(step(False, [TRAINED]).title, "Add some text")

    def test_no_version(self):
        self.assertEqual(step(True), NextStep("builder", "Design your first model", step(True).line))

    def test_unsaved_draft(self):
        got = step(True, [SAVED, DRAFT])
        self.assertEqual((got.route, got.title), ("builder", "Save it"))
        self.assertIn("Version 2", got.line)

    def test_saved_but_never_trained_and_nothing_queued(self):
        got = step(True, [SAVED])
        self.assertEqual((got.route, got.title), ("builder", "Queue the training"))
        self.assertIn("Version 1", got.line)

    def test_training_now(self):
        for versions in ([SAVED], [TRAINED]):  # a running job wins over "trained"
            got = step(True, versions, job=RUNNING)
            self.assertEqual((got.route, got.title), ("queue", "Training now"))
            self.assertIn("Version 1", got.line)

    def test_a_paused_run_says_so(self):
        self.assertEqual(step(True, [SAVED], job=RUNNING, paused=True).title, "Training is paused")

    def test_trained(self):
        got = step(True, [TRAINED, Version(2, status="saved")])
        self.assertEqual((got.route, got.title), ("chat", "Talk to it"))
        self.assertIn("Version 1", got.line)

    def test_a_queued_job_means_training_is_already_taken_care_of(self):
        got = step(True, [SAVED], up_next=[Job("a", "pretrain")])
        self.assertEqual((got.route, got.title), ("queue", "Waiting to start"))
        self.assertEqual(next_step(True, [SAVED], QueueState(up_next=[Job("a", "pretrain")]), "Starts tonight at 22:00").line,
                         "Starts tonight at 22:00.")

    def test_paused_and_held_are_states_of_their_own(self):
        paused = step(True, [SAVED], paused=True, up_next=[Job("a", "pretrain")])
        self.assertEqual((paused.route, paused.title), ("queue", "Resume the queue"))
        self.assertEqual(step(True, [SAVED], paused=True).title, "Queue the training")  # nothing to resume yet
        held = step(True, [SAVED], up_next=[Job("h", "pretrain", status="held")])
        self.assertEqual((held.route, held.title), ("queue", "Release a held job"))
        mixed = step(True, [SAVED], up_next=[Job("h", "pretrain", status="held"), Job("n", "pretrain")])
        self.assertEqual(mixed.title, "Waiting to start")

    def test_a_trained_version_is_not_hidden_by_a_paused_queue(self):
        self.assertEqual(step(True, [TRAINED], paused=True, up_next=[Job("a", "pretrain")]).route, "chat")


class WithServices(unittest.TestCase):
    def fake(self, **state):
        f = FakeServices()
        f.set_queue(QueueState(**state))
        return f

    def test_the_fake_workspace_has_text_and_no_version(self):
        self.assertEqual(self.fake().next_step().title, "Design your first model")

    def test_saving_and_queueing_moves_it_on(self):
        f = self.fake()
        f.save_version({}, "")
        self.assertEqual(f.next_step().title, "Queue the training")
        f.enqueue(Job("a", "pretrain", "Version 1"))
        got = f.next_step()
        self.assertEqual((got.title, got.line), ("Waiting to start", "Starts now."))

    def test_the_waiting_line_is_the_queue_outlook(self):
        f = self.fake(gate=Gate("outside hours", datetime(2099, 1, 1, 22)), plan={"hours": "nights"}, up_next=[Job("a", "pretrain")])
        f.save_version({}, "")
        self.assertIn("22:00", f.next_step().line)

    def test_an_empty_text_set_asks_for_text(self):
        f = self.fake()
        f._rows()["text"].clear()
        self.assertEqual(f.next_step().title, "Add some text")

    def test_empty_line_is_what_to_do_unless_the_answer_is_chat(self):
        f = self.fake()
        self.assertEqual(empty_line(f), f.next_step().line)
        f._versions = [TRAINED]
        self.assertEqual(empty_line(f), "")

    def test_empty_line_is_blank_when_the_read_would_be_long(self):
        f = self.fake()
        f.ready = lambda work: False
        self.assertEqual(empty_line(f), "")

    def test_the_row_listens_to_everything_that_changes_the_answer(self):
        self.assertTrue({"queue", "versions", "data", "job"} <= set(NEXT_EVENTS))
        f, seen = FakeServices(), []
        f.subscribe(lambda kind, _payload: seen.append(kind))
        f.add_data("text", {"text": "a new line of prose"})
        self.assertIn("data", seen)


class CompareGuard(unittest.TestCase):
    PLAN = {"variants": []}

    def test_off_until_there_is_a_variant_and_a_guess(self):
        self.assertEqual(queue_block(self.PLAN, "", [], None), LOOKS[1])
        self.assertEqual(queue_block(self.PLAN, "", [], "better"), LOOKS[1])
        self.assertEqual(queue_block(self.PLAN, "", [{"name": "v"}], None), LOOKS[0])
        self.assertEqual(queue_block(self.PLAN, "", [{"name": "v"}], ""), LOOKS[0])

    def test_live_with_both(self):
        self.assertEqual(queue_block(self.PLAN, "", [{"name": "v"}], "better"), "")

    def test_an_unbuildable_plan_says_why(self):
        self.assertEqual(queue_block(None, "Runs must differ.", [{"name": "v"}], "better"), "Runs must differ.")


class HeadLabTiles(unittest.TestCase):
    STATS = {"previous": [.9, .1, .1], "first": [.1, .8, .1], "induction": [.1, .1, .7]}

    def test_every_tile_has_its_pattern_without_pressing_anything(self):
        self.assertEqual([head_pattern(self.STATS, h) for h in range(3)],
                         [PATTERN_WORDS["previous"], PATTERN_WORDS["first"], PATTERN_WORDS["induction"]])

    def test_a_head_below_the_threshold_is_called_that_not_blank(self):
        weak = {"previous": [.2], "first": [.3], "induction": [.1]}
        self.assertEqual(head_pattern(weak, 0), NO_PATTERN)

    def test_no_placeholder_text_remains(self):
        import inspect
        from studio_ia.pages import head_lab
        self.assertNotIn("no pattern yet", inspect.getsource(head_lab))


class CollectionCard(unittest.TestCase):
    def test_shows_only_while_the_open_kind_has_no_sets(self):
        self.assertTrue(collection_shown(dk.DataSummary("text")))
        self.assertFalse(collection_shown(dk.DataSummary("text", sets=[{"id": "d", "name": "Your files", "count": 2}])))


if __name__ == "__main__":
    unittest.main()
