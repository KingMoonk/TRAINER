"""One next-step model (ordered states), and the three guards that read it: Compare's queue button, Head lab tiles, Your data's collection card."""
import unittest

from studio_ia.next_step import NextStep, next_step
from studio_ia.services import Job, QueueState, Version


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


if __name__ == "__main__":
    unittest.main()
