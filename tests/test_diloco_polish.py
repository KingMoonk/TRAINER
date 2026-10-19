"""Teacher polish and lessons inside a DiLoCo run (#508): tiny runs on an injected clock, with a fake polisher in place of Ollama (nothing
here contacts a teacher or the network)."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import studio_diloco
from tests import diloco_support as support

ROWS = [{"messages": [{"role": "user", "content": f"What is {n} plus {n}?"}], "chosen": f"{n + n} exactly.", "rejected": "no idea"} for n in range(1, 4)]


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class Teacher:
    """A stand-in for default_polish. `win`: return a polished checkpoint (the base weights plus `shift`); `rows`: the lessons it gives.
    `gate`, when set, holds the call until the test releases it."""

    def __init__(self, win=True, rows=ROWS, shift=.5, error=None, gate=None):
        self.win, self.rows, self.shift, self.error, self.gate = win, rows, shift, error, gate
        self.calls, self.started = [], threading.Event()

    def __call__(self, directory, found, checkpoint, prompts, device, stop, emit):
        self.calls.append({"directory": Path(directory), "found": found, "checkpoint": checkpoint, "prompts": prompts, "device": device})
        self.started.set()
        if self.gate is not None:
            self.gate.wait(10)
        if self.error:
            raise RuntimeError(self.error)
        out = checkpoint
        if self.win:
            with np.load(checkpoint, allow_pickle=False) as stored:
                arrays = {key: stored[key] + self.shift if key.startswith("weight::") else stored[key] for key in stored.files}
            out = str(Path(directory) / f"polished{len(self.calls)}.npz")
            Path(directory).mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as stream:
                np.savez(stream, **arrays)
        return {"checkpoint": out, "win_rate": .7 if self.win else .4, "lessons": [dict(row) for row in self.rows]}


class Polishing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "run"
        self.clock = Clock()

    def coordinator(self, teacher, losses, **changes):
        """A run polished every 2 rounds; `losses` are the validation losses in the order they are asked for (each round's, and a polished
        checkpoint's after the round it follows)."""
        c = support.build(self.dir, clock=self.clock, **{"inner_steps": 2, "rounds": 8, "polish_every": 2, "teacher": "t", **changes})
        c.polisher = teacher
        queue = iter(losses)
        c.evaluate = lambda theta: next(queue)
        return c

    def send(self, c, node="a"):
        number = c.next_round
        view = c.round_view(number, node)
        self.assertEqual(view["state"], "open", view)
        self.clock.now += 1.
        c.submit(number, node, view["generation"], 0, 10, 2, support.constant_delta(c, .01), 1.)

    def settle(self, c):
        """Wait for a polish in flight to be taken in."""
        end = time.monotonic() + 10
        while c.polishing and time.monotonic() < end:
            time.sleep(.01)
        self.assertFalse(c.polishing, "the polish did not finish")

    def play(self, c, rounds, node="a"):
        c.join(node)
        for _ in range(rounds):
            if c.finished:
                break
            self.send(c, node)
            self.settle(c)

    def events(self, c, kind):
        return [e["text"] for e in c.snapshot()["events"] if e["kind"] == kind]

    def record(self):
        return support.read_json(self.dir / "diloco.json")

    # -- when it happens
    def test_off_by_default_no_polish_no_prompts(self):
        c = support.build(self.dir, clock=self.clock, inner_steps=2, rounds=4)
        self.assertFalse((self.dir / "prompts.jsonl").exists())
        self.play(c, 4)
        self.assertEqual((c.finished, c.polishing, c.lessons), (True, False, []))
        self.assertEqual(self.events(c, "polish"), [])

    def test_prompts_come_from_the_training_conversations_only(self):
        c = self.coordinator(Teacher(), [5.])
        lines = (self.dir / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertTrue(lines)
        prompts = [json.loads(line) for line in lines]
        self.assertTrue(all(p["messages"] and p["messages"][-1]["role"] == "user" for p in prompts))
        validation = {text for text in (e["messages"][0]["content"] for e in support.EXAMPLES if "messages" in e)} - {p["messages"][0]["content"] for p in prompts}
        self.assertTrue(validation, "the validation split's conversations are not among the prompts")
        self.assertEqual(c.prompt_window(1), prompts[:studio_diloco.PROMPTS_PER_POLISH])

    def test_it_runs_after_every_nth_closed_round_and_never_after_the_last(self):
        teacher = Teacher(win=False, rows=[])
        c = self.coordinator(teacher, [5., 4., 4.5, 4.4, 4.3, 4.2], rounds=6)
        self.play(c, 6)
        self.assertEqual((c.next_round, c.finished, len(teacher.calls)), (6, True, 2))  # after rounds 2 and 4 closed; round 6 is the last
        self.assertEqual(teacher.calls[0]["checkpoint"], str(self.dir / "last.npz"))
        self.assertEqual(teacher.calls[0]["directory"], self.dir / "teacher")
        self.assertEqual(teacher.calls[0]["found"]["teacher_model"], "t")
        self.assertNotEqual(teacher.calls[0]["prompts"], [])

    def test_no_round_opens_while_the_teacher_works(self):
        gate = threading.Event()
        teacher = Teacher(win=False, rows=[], gate=gate)
        c = self.coordinator(teacher, [5., 4., 3., 2.])
        c.join("a")
        self.send(c)
        self.send(c)
        self.assertTrue(teacher.started.wait(10))
        view = c.round_view(2, "a")
        self.assertEqual((view["state"], view["polishing"], view["member"]), ("pending", True, True))
        snap = c.snapshot()
        self.assertEqual((snap["phase"], snap["state"], snap["round"]), ("polishing", "waiting", 2))
        self.assertIsNone(c.current)
        gate.set()
        self.settle(c)
        self.assertEqual(c.round_view(2, "a")["state"], "open")
        self.assertFalse(c.round_view(2, "a")["polishing"])

    def test_the_waiting_member_is_woken_when_the_polish_ends(self):
        gate = threading.Event()
        c = self.coordinator(Teacher(win=False, rows=[], gate=gate), [5., 4., 3.])
        c.join("a")
        self.send(c)
        self.send(c)
        answer = []
        waiter = threading.Thread(target=lambda: answer.append(c.round_view(2, "a", wait=8)))
        waiter.start()
        gate.set()
        waiter.join(10)
        self.assertEqual(answer[0]["state"], "open")

    # -- the polished weights
    def test_polished_weights_that_are_no_worse_replace_theta_and_become_the_best(self):
        c = self.coordinator(Teacher(win=True, rows=[]), [5., 4., 3.5], lessons=False)
        self.play(c, 2)
        row = c.history[1]
        self.assertEqual((row["polish"], row["val_loss"], row["val_loss_before_polish"], row["best"]), ("accepted", 3.5, 4., True))
        self.assertNotIn("polish_val_loss", row)
        self.assertEqual((c.rules.best_round, c.rules.best_loss, c.rules.rounds_since_best), (1, 3.5, 0))
        with np.load(self.dir / "last.npz") as last, np.load(self.dir / "best.npz") as best:
            for name, value in c.theta.items():
                np.testing.assert_array_equal(last["weight::" + name], value)
                np.testing.assert_array_equal(best["weight::" + name], value)
        self.assertEqual(self.record()["history"][1]["val_loss"], 3.5)
        metrics = [json.loads(line) for line in (self.dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(m["round"], m["val_loss"]) for m in metrics], [(0, 5.), (1, 3.5)])
        self.assertEqual(metrics[1]["val_loss_before_polish"], 4.)
        with np.load(self.dir / "last.npz", allow_pickle=False) as last:
            self.assertEqual(json.loads(str(last["metadata"]))["best"], 3.5)
        run = support.read_json(self.dir / "run.json")
        self.assertEqual((run["val_loss"], run["best_val_loss"], run["best_round"]), (3.5, 3.5, 1))
        self.assertIn("kept the polished weights", self.events(c, "polish")[-1])

    def test_the_polished_weights_are_the_round_weights_plus_what_the_teacher_changed(self):
        c = self.coordinator(Teacher(win=True, rows=[], shift=.5), [5., 4., 3.5], lessons=False)
        c.join("a")
        self.send(c)
        self.send(c)
        before = {name: value.copy() for name, value in c.theta.items()}
        self.settle(c)
        for name, value in c.theta.items():
            np.testing.assert_allclose(value, before[name] + .5, atol=1e-6)
        self.assertEqual(c.round_view(2, "a")["state"], "open")

    def test_polished_weights_that_validate_worse_are_dropped(self):
        c = self.coordinator(Teacher(win=True, rows=[]), [5., 4., 4.5], lessons=False)
        c.join("a")
        self.send(c)
        self.send(c)
        before = {name: value.copy() for name, value in c.theta.items()}
        self.settle(c)
        for name, value in c.theta.items():
            np.testing.assert_array_equal(value, before[name])
        row = c.history[1]
        self.assertEqual((row["polish"], row["val_loss"], row["polish_val_loss"]), ("rejected", 4., 4.5))
        self.assertNotIn("val_loss_before_polish", row)
        self.assertEqual((c.rules.best_round, c.rules.best_loss), (1, 4.))
        self.assertIn("no better on the validation windows", self.events(c, "polish")[-1])

    def test_a_polish_that_lost_the_head_to_head_is_not_even_scored(self):
        c = self.coordinator(Teacher(win=False, rows=[]), [5., 4.], lessons=False)
        self.play(c, 2)
        self.assertEqual(c.history[1]["polish"], "rejected")
        self.assertIn("did not win the teacher's head-to-head", self.events(c, "polish")[-1])

    def test_with_polish_off_only_lessons_are_kept(self):
        teacher = Teacher(win=True)
        c = self.coordinator(teacher, [5., 4.], polish=False)
        before = {name: value.copy() for name, value in c.theta.items()}
        self.play(c, 2)
        self.assertEqual(c.history[1].get("lessons"), 3)
        self.assertNotIn("polish", c.history[1])
        for name in before:
            self.assertFalse(np.array_equal(c.theta[name], before[name]))  # trained, not shifted by the teacher's .5
        self.assertEqual(len(c.lessons), 1)

    def test_a_polish_that_reaches_the_target_ends_the_run(self):
        c = self.coordinator(Teacher(win=True, rows=[]), [5., 4., 3.], lessons=False, target_val_loss=3.5)
        self.play(c, 2)
        self.assertTrue(c.finished)
        self.assertEqual(c.rules.stop_rule, "target_val_loss")
        self.assertEqual(support.read_json(self.dir / "run.json")["status"], "completed")
        self.assertEqual(c.round_view(2, "a")["state"], "finished")

    def test_a_polish_that_improves_a_later_round_moves_the_best_there(self):
        c = self.coordinator(Teacher(win=True, rows=[]), [3., 5., 4.], lessons=False)  # round 0 best at 3; the polish of round 1 reaches 4: still behind
        self.play(c, 2)
        self.assertEqual((c.rules.best_round, c.rules.best_loss), (0, 3.))
        self.assertEqual(c.history[1]["polish"], "accepted")
        self.assertFalse(c.history[1]["best"])
        self.assertEqual(c.rules.rounds_since_best, 1)

    # -- lessons
    def test_lessons_become_extra_shards_dealt_from_the_next_round(self):
        c = self.coordinator(Teacher(win=False), [5., 4.])
        self.assertEqual(c.shard_count, 4)
        self.play(c, 2)
        self.assertEqual(c.lessons, [{"shard": 4, "round": 1, "rows": 3, "status": "live"}])
        self.assertEqual(c.shard_count, 4)  # what members prefetch is unchanged
        view = c.round_view(2, "a")
        self.assertEqual(sorted(view["shards"]), [0, 1, 2, 3, 4])
        blob = c.shard_bytes(4)
        self.assertTrue(blob)
        lines = [json.loads(line) for line in (self.dir / "lessons.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(row["shard"], row["round"], row["chosen"]) for row in lines], [(4, 1, "2 exactly."), (4, 1, "4 exactly."), (4, 1, "6 exactly.")])
        self.assertEqual(c.history[1]["lessons"], 3)
        self.assertEqual(self.record()["lessons"], c.lessons)
        with self.assertRaises(KeyError):
            c.shard_bytes(5)

    def test_each_polish_adds_its_own_lesson_shard(self):
        c = self.coordinator(Teacher(win=False), [5., 4., 4.5, 4.4], rounds=6)
        self.play(c, 4)
        self.assertEqual([(e["shard"], e["round"]) for e in c.lessons], [(4, 1), (5, 3)])
        self.assertEqual(sorted(c.round_view(4, "a")["shards"]), [0, 1, 2, 3, 4, 5])

    def test_lessons_off_deals_only_the_data(self):
        c = self.coordinator(Teacher(win=False), [5., 4.], lessons=False)
        self.play(c, 2)
        self.assertEqual(c.lessons, [])
        self.assertEqual(sorted(c.round_view(2, "a")["shards"]), [0, 1, 2, 3])

    def test_lessons_survive_a_restart(self):
        c = self.coordinator(Teacher(win=False), [5., 4.])
        self.play(c, 2)
        again = support.build(self.dir, clock=Clock(), inner_steps=2, rounds=8, polish_every=2, teacher="t")
        self.assertEqual(again.lessons, c.lessons)
        self.assertEqual(again.next_round, 2)
        self.assertEqual(again.shard_bytes(4), c.shard_bytes(4))
        again.join("a")
        self.assertEqual(sorted(again.round_view(2, "a")["shards"]), [0, 1, 2, 3, 4])

    def test_a_fresh_run_in_the_folder_forgets_old_lessons(self):
        c = self.coordinator(Teacher(win=False), [5., 4.])
        self.play(c, 2)
        (self.dir / "prompts.jsonl").write_text('{"stale": true}\n', encoding="utf-8")
        (self.dir / "diloco.json").unlink()
        fresh = support.build(self.dir, clock=Clock(), inner_steps=2, rounds=8, polish_every=2, teacher="t")
        self.assertEqual((fresh.lessons, fresh.lesson_windows), ([], {}))
        self.assertFalse((self.dir / "lessons.jsonl").exists())
        self.assertNotIn("stale", (self.dir / "prompts.jsonl").read_text(encoding="utf-8"))

    def test_rows_that_make_no_training_window_add_no_lesson(self):
        bad = [{"messages": [], "chosen": "x"}, {"messages": [{"role": "user", "content": "hi"}], "chosen": ""}]
        c = self.coordinator(Teacher(win=False, rows=bad), [5., 4.])
        self.play(c, 2)
        self.assertEqual(c.lessons, [])
        self.assertIn("had no lessons to give", self.events(c, "polish")[-1])

    # -- the drop_lessons pivot
    def test_drop_lessons_goes_back_to_the_best_and_leaves_out_the_lessons_written_since(self):
        c = self.coordinator(Teacher(win=False), [5., 4., 4.5], worse_limit=1, pivots=("drop_lessons",))
        c.join("a")
        self.send(c)
        self.send(c)  # best at round 1, then a polish writes a lesson
        self.settle(c)
        with np.load(self.dir / "best.npz") as best:
            weights = {name: best["weight::" + name].copy() for name in c.theta}
        self.send(c)  # round 2 is worse
        self.settle(c)
        row = c.history[2]
        self.assertEqual(row.get("pivot"), "drop_lessons")
        self.assertEqual([e["status"] for e in c.lessons], ["left_out"])
        for name, value in c.theta.items():
            np.testing.assert_array_equal(value, weights[name])
        self.assertEqual(c.live_lessons(), [])
        self.assertEqual(sorted(c.round_view(3, "a")["shards"]), [0, 1, 2, 3])
        with self.assertRaises(KeyError):
            c.shard_bytes(4)
        self.assertIn("left out the teacher's lessons", self.events(c, "pivot")[-1])
        again = support.build(self.dir, clock=Clock(), inner_steps=2, rounds=8, polish_every=2, teacher="t", worse_limit=1, pivots=("drop_lessons",))
        self.assertEqual([e["status"] for e in again.lessons], ["left_out"])
        self.assertEqual(again.lesson_windows, {})

    def test_drop_lessons_is_skipped_when_the_best_round_is_newer_than_every_lesson(self):
        c = self.coordinator(Teacher(win=False), [5., 4., 3., 3.5, 3.6], rounds=8, worse_limit=1, pivots=("drop_lessons", "halve_lr"))
        self.play(c, 4)  # the lesson of round 1; round 2 is the best (3.); round 3 is worse: the lesson predates the best, so halve_lr
        self.assertEqual(c.history[3].get("pivot"), "halve_lr")
        self.assertEqual([e["status"] for e in c.lessons], ["live"])

    # -- failures and endings
    def test_a_teacher_error_skips_the_polish_and_the_run_goes_on(self):
        c = self.coordinator(Teacher(error="Cannot reach Ollama at http://localhost:11434."), [5., 4., 3.])
        self.play(c, 3)
        self.assertEqual(c.lessons, [])
        self.assertEqual(c.next_round, 3)
        self.assertIn("skipped (the teacher failed: Cannot reach Ollama", self.events(c, "polish")[-1])
        self.assertEqual(c.history[1]["polish"], "skipped")

    def test_stopping_during_a_polish_discards_it(self):
        gate = threading.Event()
        teacher = Teacher(win=True, gate=gate)
        c = self.coordinator(teacher, [5., 4., 1.])
        c.join("a")
        self.send(c)
        self.send(c)
        before = {name: value.copy() for name, value in c.theta.items()}
        self.assertTrue(teacher.started.wait(10))
        c.stop()
        gate.set()
        self.settle(c)
        self.assertEqual((c.stopped, c.lessons), (True, []))
        for name, value in c.theta.items():
            np.testing.assert_array_equal(value, before[name])
        self.assertEqual(c.snapshot()["phase"], "stopped")

    def test_a_coordinator_made_from_a_replica_does_not_polish(self):
        teacher = Teacher(win=False)
        c = self.coordinator(teacher, [5., 4., 3.])
        c.polish_on = False
        self.play(c, 2)
        self.assertEqual(teacher.calls, [])

    def test_polish_is_not_started_after_a_round_that_went_back_to_the_best(self):
        teacher = Teacher(win=False, rows=[])
        c = self.coordinator(teacher, [5., 4., 6., 7.], rounds=8, worse_limit=1, pivots=("halve_lr",), polish_every=1)
        self.play(c, 3)
        self.assertEqual(c.history[2].get("pivot"), "halve_lr")
        self.assertEqual(len(teacher.calls), 2)  # after rounds 0 and 1; not after the pivot

    def test_snapshot_lists_the_lessons_and_the_options(self):
        c = self.coordinator(Teacher(win=False), [5., 4.])
        self.play(c, 2)
        snap = c.snapshot()
        self.assertEqual(snap["lessons"], c.lessons)
        self.assertEqual((snap["settings"]["polish_every"], snap["settings"]["polish"], snap["settings"]["lessons"]), (2, True, True))


    # -- how the rounds and files record it
    def test_the_lessons_have_an_event_of_their_own(self):
        c = self.coordinator(Teacher(win=False), [5., 4.])
        self.play(c, 2)
        self.assertEqual(self.events(c, "lessons"), ["Round 1: 3 lessons join training from round 2."])

    def test_a_lesson_never_reuses_a_shard_id_a_crash_left_in_the_file(self):
        c = self.coordinator(Teacher(win=False), [5., 4.])
        with (self.dir / "lessons.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"shard": 4, "round": 1, "messages": [], "chosen": "x"}) + "\n")  # written, but never named in diloco.json
        self.play(c, 2)
        self.assertEqual([e["shard"] for e in c.lessons], [5])

    def test_a_polish_that_used_up_max_seconds_ends_the_run_there(self):
        gate = threading.Event()
        c = self.coordinator(Teacher(win=False, rows=[], gate=gate), [5., 4.], max_seconds=50.)
        c.join("a")
        self.send(c)
        self.send(c)
        self.clock.now += 100.  # the teacher takes its time
        gate.set()
        self.settle(c)
        self.assertEqual((c.finished, c.rules.stop_rule), (True, "max_seconds"))
        self.assertEqual(support.read_json(self.dir / "run.json")["status"], "completed")

    def test_the_preference_runs_of_the_teacher_are_deleted_once_read(self):
        def polish(directory, found, checkpoint, prompts, device, stop, emit):
            folder = Path(directory) / "rlhf_abc123"
            folder.mkdir(parents=True)
            with np.load(checkpoint, allow_pickle=False) as stored:
                arrays = {key: stored[key] + .5 if key.startswith("weight::") else stored[key] for key in stored.files}
            with open(folder / "last.npz", "wb") as stream:
                np.savez(stream, **arrays)
            return {"checkpoint": str(folder / "last.npz"), "win_rate": .7, "lessons": []}
        c = self.coordinator(polish, [5., 4., 3.5], lessons=False)
        self.play(c, 2)
        self.assertEqual(c.history[1]["polish"], "accepted")
        self.assertEqual(list((self.dir / "teacher").glob("rlhf_*")), [])

    def test_a_polish_thread_that_cannot_start_leaves_the_gate_open(self):
        real = threading.Thread

        def thread(*args, **kwargs):
            if kwargs.get("name") == "diloco-polish":
                raise RuntimeError("can't start new thread")
            return real(*args, **kwargs)
        c = self.coordinator(Teacher(win=False), [5., 4.])
        with mock.patch.object(studio_diloco.threading, "Thread", side_effect=thread):
            self.play(c, 2)
        self.assertFalse(c.polishing)
        self.assertEqual(c.history[1]["polish"], "skipped")
        self.assertIn("could not start", self.events(c, "polish")[-1])
        self.assertEqual(c.round_view(2, "a")["state"], "open")

    # -- when the work after the teacher fails, the lessons are still kept
    def test_scoring_the_polished_weights_can_fail_and_the_lessons_stay(self):
        c = self.coordinator(Teacher(win=True), [5., 4.])
        calls = []

        def evaluate(theta):
            calls.append(1)
            if len(calls) > 2:
                raise RuntimeError("out of memory")
            return 5. if len(calls) == 1 else 4.
        c.evaluate = evaluate
        self.play(c, 2)
        row = c.history[1]
        self.assertEqual((row["polish"], row["lessons"], row["val_loss"]), ("skipped", 3, 4.))
        self.assertEqual(len(c.lessons), 1)
        self.assertIn("could not be scored: out of memory", self.events(c, "polish")[-1])
        self.assertEqual(c.round_view(2, "a")["state"], "open")

    def test_unreadable_polished_weights_skip_the_polish_and_keep_the_lessons(self):
        def polish(directory, found, checkpoint, prompts, device, stop, emit):
            return {"checkpoint": str(Path(directory) / "missing.npz"), "win_rate": .7, "lessons": [dict(row) for row in ROWS]}
        c = self.coordinator(polish, [5., 4.])
        self.play(c, 2)
        self.assertEqual((c.history[1]["polish"], c.history[1]["lessons"]), ("skipped", 3))
        self.assertIn("could not be read", self.events(c, "polish")[-1])

    # -- a polish meets a pause, a successor, a release
    def test_a_pause_during_a_polish_holds_the_next_round_after_it(self):
        gate = threading.Event()
        teacher = Teacher(win=False, rows=[], gate=gate)
        c = self.coordinator(teacher, [5., 4., 3.])
        c.join("a")
        self.send(c)
        self.send(c)
        self.assertTrue(teacher.started.wait(10))
        c.pause()
        gate.set()
        self.settle(c)
        view = c.round_view(2, "a")
        self.assertEqual((view["state"], view["paused"], view["polishing"]), ("pending", True, False))
        self.assertEqual(c.snapshot()["phase"], "paused")
        c.resume()
        self.assertEqual(c.round_view(2, "a")["state"], "open")

    def test_a_successor_term_during_a_polish_discards_it(self):
        gate = threading.Event()
        teacher = Teacher(win=True, gate=gate)
        c = self.coordinator(teacher, [5., 4., 1.])
        c.join("a")
        self.send(c)
        self.send(c)
        before = {name: value.copy() for name, value in c.theta.items()}
        self.assertTrue(teacher.started.wait(10))
        c.step_down(c.term + 1)
        gate.set()
        self.settle(c)
        self.assertEqual(c.lessons, [])
        self.assertNotIn("polish", c.history[1])
        for name, value in c.theta.items():
            np.testing.assert_array_equal(value, before[name])

    def test_releasing_the_run_during_a_polish_discards_it(self):
        gate = threading.Event()
        teacher = Teacher(win=True, gate=gate)
        c = self.coordinator(teacher, [5., 4., 1.])
        c.join("a")
        self.send(c)
        self.send(c)
        self.assertTrue(teacher.started.wait(10))
        c.release()
        gate.set()
        self.settle(c)
        self.assertEqual((c.theta, c.lessons), ({}, []))
        self.assertNotIn("polish", c.history[1])

    # -- it survives the coordinator
    def test_a_rebuilt_coordinator_holds_the_polished_weights_the_best_and_the_lessons(self):
        c = self.coordinator(Teacher(win=True), [5., 4., 3.5])
        self.play(c, 2)
        self.assertEqual(c.history[1]["polish"], "accepted")
        again = support.build(self.dir, clock=Clock(), inner_steps=2, rounds=8, polish_every=2, teacher="t")
        for name, value in c.theta.items():
            np.testing.assert_array_equal(again.theta[name], value)
        self.assertEqual((again.rules.best_round, again.rules.best_loss), (1, 3.5))
        self.assertEqual((again.history[1]["val_loss"], again.history[1]["polish"]), (3.5, "accepted"))
        self.assertEqual(again.lessons, c.lessons)
        self.assertEqual(again.next_round, 2)

    def test_a_coordinator_made_from_a_replica_polishes_no_more_and_leaves_the_lessons_out(self):
        c = self.coordinator(Teacher(win=True), [5., 4., 3.5])
        self.play(c, 2)
        replica, shards = Path(self.tmp.name) / "replica", Path(self.tmp.name) / "shards"
        replica.mkdir()
        (replica / "theta.npz").write_bytes(studio_diloco.encode_arrays(c.theta))
        (replica / "momentum.npz").write_bytes(studio_diloco.encode_arrays(c.momentum))
        (replica / "diloco.json").write_bytes((self.dir / "diloco.json").read_bytes())
        (replica / "info.json").write_text(json.dumps(c.info()), encoding="utf-8")
        (replica / "validation.npz").write_bytes(c.validation_bytes())
        for shard in range(c.shard_count):
            studio_diloco.install_shard(shards, shard, c.shard_bytes(shard), c.tokenizer.vocab_size)
        taken = studio_diloco.Coordinator.from_replica("run1", Path(self.tmp.name) / "taken", replica, shards, 2, clock=Clock())
        self.assertFalse(taken.polish_on)
        self.assertEqual([e["status"] for e in taken.lessons], ["left_out"])
        self.assertEqual([e["kind"] for e in taken.snapshot()["events"] if e["kind"] == "polish_off"], ["polish_off"])
        self.assertEqual((taken.rules.best_round, taken.rules.best_loss), (1, 3.5))  # relabelled from the polished loss the replica holds
        self.assertEqual(taken.record["val_loss"], 3.5)
        taken.join("a")
        self.assertEqual(sorted(taken.round_view(2, "a")["shards"]), [0, 1, 2, 3])


class Preflight(unittest.TestCase):
    def job(self, **params):
        return SimpleNamespace(id="job1", params={"config": dict(support.CONFIG), "examples": support.EXAMPLES, "inner_steps": 2, "rounds": 4, **params})

    def test_an_unreachable_teacher_fails_the_job_before_anything_is_built(self):
        job = self.job(polish_every=2, teacher="t")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, \
                mock.patch.object(studio_diloco, "check_teacher", side_effect=RuntimeError("Cannot reach Ollama at http://localhost:11434.")) as check, \
                mock.patch.object(studio_diloco.Coordinator, "build") as build:
            with self.assertRaisesRegex(ValueError, "polishes with the teacher t.*Cannot reach Ollama"):
                studio_diloco.prepare(job, SimpleNamespace(), tmp, 8000)
        check.assert_called_once()
        self.assertEqual(check.call_args.args[0]["teacher_model"], "t")
        build.assert_not_called()

    def test_the_default_polisher_stops_when_ollama_is_unreachable(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, \
                mock.patch("studio_rlhf.OllamaTeacher.models", side_effect=RuntimeError("Cannot reach Ollama")):
            found = studio_diloco.teacher_settings(support.settings(polish_every=2, teacher="t"))
            with self.assertRaisesRegex(RuntimeError, "Cannot reach Ollama"):
                studio_diloco.default_polish(Path(tmp), found, "x.npz", [], "cpu", threading.Event(), lambda event: None)
            with self.assertRaisesRegex(RuntimeError, "Cannot reach Ollama"):
                studio_diloco.check_teacher(found)

    def test_a_resumed_run_does_not_check_the_teacher_again(self):
        job = self.job(polish_every=2, teacher="t", resuming={"round": 1})
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, \
                mock.patch.object(studio_diloco, "check_teacher") as check, \
                mock.patch.object(studio_diloco.Coordinator, "build", side_effect=ValueError("different data")):
            with self.assertRaisesRegex(ValueError, "Ended: this machine restarted at round 1"):
                studio_diloco.prepare(job, SimpleNamespace(), tmp, 8000)
        check.assert_not_called()

    def test_the_check_wants_the_model_and_waits_only_briefly(self):
        found = studio_diloco.teacher_settings(support.settings(polish_every=2, teacher="t"))
        timeouts = []

        def models(teacher):
            timeouts.append(teacher.timeout)
            return ["other:latest"]
        with mock.patch("studio_rlhf.OllamaTeacher.models", models):
            with self.assertRaisesRegex(RuntimeError, "has no model called t.*other:latest"):
                studio_diloco.check_teacher(found)
        self.assertEqual(timeouts, [studio_diloco.PREFLIGHT_TIMEOUT])
        with mock.patch("studio_rlhf.OllamaTeacher.models", lambda teacher: ["t:latest"]):
            studio_diloco.check_teacher(found)

    def test_a_dataset_without_conversations_fails_the_build_in_a_sentence(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            with self.assertRaisesRegex(ValueError, "no conversations for the teacher"):
                studio_diloco.Coordinator.build("run1", tmp, support.CONFIG, support.EXAMPLES[:40], support.settings(polish_every=2, teacher="t"))


if __name__ == "__main__":
    unittest.main()
