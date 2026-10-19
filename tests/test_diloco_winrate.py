"""The head-to-head win-rate stop rule of a DiLoCo run (#511): tiny runs on an injected clock, with a fake `versus` in place of the teacher
(nothing here contacts Ollama or the network)."""
import json
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import studio_diloco
import studio_diloco_rules as rules
import studio_distributed as runs
from studio_api.routers.distributed import Run
from tests import diloco_support as support


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class Versus:
    """A stand-in for default_versus: answers `rates` in turn (an Exception in the list is raised). `gate` holds the call; `hang` holds it
    until it is told to halt (the way a teacher that does not answer would)."""

    def __init__(self, rates, gate=None, hang=False):
        self.rates, self.gate, self.hang, self.calls, self.started, self.halts = list(rates), gate, hang, [], threading.Event(), []

    def __call__(self, settings, new, old, prompts, device, stop):
        self.calls.append({"settings": settings, "new": new, "old": old, "prompts": prompts, "device": device})
        self.halts.append(stop)
        self.started.set()
        if self.hang:
            stop.wait(10)
            return .99
        if self.gate is not None:
            self.gate.wait(10)
        rate = self.rates[len(self.calls) - 1]
        if isinstance(rate, Exception):
            raise rate
        return rate


def polisher(shift):
    """A fake default_polish whose gate always passes: the base weights plus `shift`."""
    def polish(directory, found, checkpoint, prompts, device, stop, emit):
        with np.load(checkpoint, allow_pickle=False) as stored:
            arrays = {key: stored[key] + shift if key.startswith("weight::") else stored[key] for key in stored.files}
        Path(directory).mkdir(parents=True, exist_ok=True)
        out = str(Path(directory) / "polished.npz")
        with open(out, "wb") as stream:
            np.savez(stream, **arrays)
        return {"checkpoint": out, "win_rate": .7, "lessons": []}
    return polish


class WinRate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "run"
        self.clock = Clock()

    def coordinator(self, versus, losses, **changes):
        c = support.build(self.dir, clock=self.clock, **{"inner_steps": 2, "rounds": 8, "teacher": "t", "target_win_rate": .5, **changes})
        c.versus = versus
        queue = iter(losses)
        c.seen = {}

        def evaluate(theta):
            c.seen[c.next_round] = {name: value.copy() for name, value in theta.items()}
            return next(queue)
        c.evaluate = evaluate
        return c

    def settle(self, c):
        end = time.monotonic() + 10
        while c.polishing and time.monotonic() < end:
            time.sleep(.01)
        self.assertFalse(c.polishing, "the head-to-head did not finish")

    def send(self, c, node="a"):
        number = c.next_round
        view = c.round_view(number, node)
        self.assertEqual(view["state"], "open", view)
        self.clock.now += 1.
        c.submit(number, node, view["generation"], 0, 10, 2, support.constant_delta(c, .01), 1.)

    def play(self, c, rounds, node="a"):
        c.join(node)
        for _ in range(rounds):
            if c.finished:
                break
            self.send(c, node)
            self.settle(c)

    def events(self, c, kind):
        return [e["text"] for e in c.snapshot()["events"] if e["kind"] == kind]

    # -- the rule
    def test_a_new_best_is_compared_and_the_run_stops_once_the_target_is_reached(self):
        versus = Versus([.3, .6, .7])
        c = self.coordinator(versus, [5., 4., 3., 2.5])
        self.play(c, 8)
        self.assertEqual((c.next_round, c.finished), (4, True))  # round 0 is the start itself; rounds 1, 2 and 3 were compared
        self.assertEqual(len(versus.calls), 3)
        self.assertEqual(versus.calls[0]["new"], str(self.dir / "best.npz"))
        self.assertEqual(versus.calls[0]["old"], str(self.dir / "start.npz"))
        self.assertEqual(versus.calls[0]["settings"]["teacher_model"], "t")
        self.assertTrue(0 < len(versus.calls[0]["prompts"]) <= studio_diloco.EVAL_PROMPTS)
        self.assertEqual((c.rules.win_rate, c.rules.win_round, c.rules.stop_rule), (.7, 3, "win_rate"))
        record = support.read_json(self.dir / "run.json")
        self.assertEqual((record["status"], record["stop_rule"], record["win_rate"]), ("completed", "win_rate", .7))
        self.assertIn("won 70%", record["stop_reason"])
        self.assertIn("target 50%", record["stop_reason"])
        self.assertIn("2 comparisons in a row", record["stop_reason"])
        self.assertEqual(self.events(c, "finished")[-1], record["stop_reason"])
        self.assertEqual(len(self.events(c, "win_rate")), 3)
        self.assertIn("30%", self.events(c, "win_rate")[0])
        self.assertEqual(c.round_view(4, "a")["state"], "finished")

    def test_one_reading_at_the_target_is_not_enough_and_a_miss_starts_the_count_again(self):
        versus = Versus([.6, .4, .6, .7])
        c = self.coordinator(versus, [5., 4., 3., 2., 1.5, 1.2], rounds=8)
        self.play(c, 8)
        self.assertEqual((len(versus.calls), c.next_round, c.rules.stop_rule), (4, 5, "win_rate"))  # .6 once, then .4, then .6 and .7
        self.assertEqual(c.rules.win_hits, 2)

    def test_a_comparison_that_gave_no_number_also_starts_the_count_again(self):
        versus = Versus([.6, RuntimeError("Ollama is down"), .6, .6])
        c = self.coordinator(versus, [5., 4., 3., 2., 1.5, 1.2], rounds=8)
        self.play(c, 8)
        self.assertEqual((len(versus.calls), c.rules.stop_rule), (4, "win_rate"))
        self.assertEqual(c.rules.win_hits, 2)

    def test_the_rate_is_kept_in_diloco_json_and_the_snapshot(self):
        c = self.coordinator(Versus([.3]), [5., 4., 4.5], rounds=3)
        self.play(c, 3)
        saved = support.read_json(self.dir / "diloco.json")["rules"]
        self.assertEqual((saved["win_rate"], saved["win_round"], saved["start_round"], saved["win_hits"]), (.3, 1, 0, 0))
        self.assertEqual(c.snapshot()["rules"]["win_rate"], .3)
        again = studio_diloco.Coordinator.build("run1", self.dir, support.CONFIG, support.EXAMPLES, c.settings, clock=self.clock)
        self.assertEqual(again.rules.win_rate, .3)

    def test_the_hits_survive_a_restart_of_the_app(self):
        c = self.coordinator(Versus([.9]), [5., 4., 3.5], rounds=6)
        self.play(c, 2)
        self.assertEqual(support.read_json(self.dir / "diloco.json")["rules"]["win_hits"], 1)
        again = studio_diloco.Coordinator.build("run1", self.dir, support.CONFIG, support.EXAMPLES, c.settings, clock=self.clock)
        self.assertEqual(again.rules.win_hits, 1)

    def test_a_round_that_is_not_a_new_best_is_not_compared(self):
        versus = Versus([.3])
        c = self.coordinator(versus, [5., 4., 4.5, 4.6, 4.7], rounds=5)
        self.play(c, 5)
        self.assertEqual((c.finished, len(versus.calls)), (True, 1))

    def test_the_final_round_is_not_compared(self):
        versus = Versus([.3])
        c = self.coordinator(versus, [5., 4., 3.], rounds=3)
        self.play(c, 3)
        self.assertEqual(len(versus.calls), 1)  # round 1; round 2 ends the run anyway

    def test_start_npz_holds_the_weights_after_the_first_round(self):
        c = self.coordinator(Versus([]), [5., 4.], rounds=2)
        self.play(c, 1)
        with np.load(self.dir / "start.npz", allow_pickle=False) as stored:
            for name, value in c.seen[0].items():
                np.testing.assert_array_equal(stored["weight::" + name], value)
        before = (self.dir / "start.npz").read_bytes()
        self.play(c, 1)
        self.assertEqual((self.dir / "start.npz").read_bytes(), before)

    def test_no_round_opens_while_the_teacher_judges(self):
        gate = threading.Event()
        versus = Versus([.3], gate=gate)
        c = self.coordinator(versus, [5., 4., 3.])
        c.join("a")
        self.send(c)
        self.send(c)
        self.assertTrue(versus.started.wait(10))
        view = c.round_view(2, "a")
        self.assertEqual((view["state"], view["polishing"]), ("pending", True))
        self.assertEqual(c.snapshot()["phase"], "judging")  # not "polishing": no polish is running
        self.assertEqual(len(self.events(c, "judging")), 1)
        self.assertIn("validation split", self.events(c, "judging")[0])
        self.assertIsNone(c.current)
        gate.set()
        self.settle(c)
        self.assertEqual(c.round_view(2, "a")["state"], "open")
        self.assertNotEqual(c.snapshot()["phase"], "judging")

    def test_a_failed_comparison_is_skipped_and_not_retried_for_the_same_best(self):
        versus = Versus([RuntimeError("Ollama is down")])
        c = self.coordinator(versus, [5., 4., 4.5, 4.6], rounds=4)
        self.play(c, 4)
        self.assertEqual((c.finished, len(versus.calls), c.rules.win_rate), (True, 1, None))
        self.assertIn("Ollama is down", self.events(c, "win_rate")[0])
        self.assertEqual(c.rules.stop_rule, "")

    def test_a_rate_under_the_target_lets_the_run_go_on_to_its_rounds(self):
        c = self.coordinator(Versus([.2, .3, .4]), [5., 4., 3., 2.], rounds=4)
        self.play(c, 4)
        self.assertEqual((c.next_round, c.rules.stop_rule, c.rules.win_rate), (4, "", .3))  # round 3 is the last: not compared

    def test_a_polish_that_makes_a_new_best_is_what_is_compared(self):
        versus = Versus([.9, .9])
        c = self.coordinator(versus, [5., 4., 3.5, 3.], polish_every=2, rounds=6)
        c.polisher = polisher(.5)
        self.play(c, 2)
        self.assertEqual(len(versus.calls), 1)  # the polish came first, the comparison once, for the best that came of it
        self.assertEqual((c.rules.best_round, c.finished, c.rules.win_hits), (1, False, 1))
        with np.load(versus.calls[0]["new"], allow_pickle=False) as stored:
            for name, value in c.theta.items():
                np.testing.assert_array_equal(stored["weight::" + name], value)
        self.assertEqual(len(self.events(c, "judging")), 1)
        self.play(c, 1)
        self.assertEqual((c.finished, c.rules.stop_rule), (True, "win_rate"))

    def test_stopping_the_run_discards_a_comparison_in_flight(self):
        gate = threading.Event()
        versus = Versus([.9], gate=gate)
        c = self.coordinator(versus, [5., 4.])
        c.join("a")
        self.send(c)
        self.send(c)
        self.assertTrue(versus.started.wait(10))
        c.stop()
        self.assertTrue(versus.halts[0].wait(10))  # the comparison is told to halt, whether or not it has returned
        gate.set()
        self.settle(c)
        self.assertIsNone(c.rules.win_rate)
        self.assertEqual(c.rules.stop_rule, "")

    def test_a_comparison_that_takes_too_long_is_skipped_and_the_gate_opens(self):
        versus = Versus([], hang=True)
        c = self.coordinator(versus, [5., 4., 3.], versus_seconds=.3)
        c.join("a")
        self.send(c)
        begun = time.monotonic()
        self.send(c)
        self.settle(c)
        self.assertLess(time.monotonic() - begun, 5)
        self.assertTrue(versus.halts[0].is_set())
        self.assertEqual((c.rules.win_round, c.rules.win_rate, c.rules.win_hits, c.finished), (1, None, 0, False))
        self.assertIn("took longer than the 0.3 seconds allowed", self.events(c, "win_rate")[0])
        self.assertEqual(c.round_view(2, "a")["state"], "open")  # the rounds go on

    def test_a_comparison_whose_worker_cannot_start_still_opens_the_gate(self):
        real = threading.Thread

        class NoRoom(real):
            def start(self):
                if self.name == "diloco-versus-call":
                    raise RuntimeError("can't start new thread")
                super().start()
        versus = Versus([.9])
        c = self.coordinator(versus, [5., 4., 3.])
        c.join("a")
        with mock.patch.object(studio_diloco.threading, "Thread", NoRoom):
            self.send(c)
            self.send(c)
            self.settle(c)
        self.assertEqual((versus.calls, c.judging, c.rules.win_round, c.rules.win_hits), ([], False, 1, 0))
        self.assertIn("could not start", self.events(c, "win_rate")[0])
        self.assertEqual(c.round_view(2, "a")["state"], "open")

    def test_the_time_allowed_is_a_positive_number_of_seconds(self):
        self.assertEqual(support.settings().versus_seconds, 600.)
        for bad in (0, -1, True, float("nan"), "soon"):
            with self.assertRaises(ValueError, msg=str(bad)):
                support.settings(versus_seconds=bad)

    # -- the prompts
    def test_the_judged_prompts_are_not_the_ones_the_lessons_teach(self):
        versus = Versus([.3])
        c = self.coordinator(versus, [5., 4.], rounds=4)
        prompts = [json.loads(line) for line in (self.dir / "eval_prompts.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertGreater(len(prompts), 1)
        taught = {"messages": prompts[0]["messages"], "chosen": "x", "rejected": "y", "shard": 4}
        (self.dir / "lessons.jsonl").write_text(json.dumps(taught) + "\n", encoding="utf-8")
        self.play(c, 2)
        self.assertEqual([p["messages"] for p in versus.calls[0]["prompts"]], [p["messages"] for p in prompts[1:]])

    def test_a_top_up_comes_from_the_end_of_the_training_prompts_and_says_so(self):
        self.coordinator(Versus([]), [5.], polish_every=2, rounds=6)
        window = [json.loads(line) for line in (self.dir / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
        prompts = [json.loads(line) for line in (self.dir / "eval_prompts.jsonl").read_text(encoding="utf-8").splitlines()]
        trained = [p for p in prompts if p["from"] == "training"]
        self.assertEqual({p["from"] for p in prompts} - {"validation", "training"}, set())
        self.assertTrue(trained and len(trained) < len(window))
        self.assertEqual([p["messages"] for p in trained], [q["messages"] for q in window[::-1][:len(trained)]])  # polish windows start at the front

    def test_a_run_without_a_validation_split_is_refused(self):
        with self.assertRaisesRegex(ValueError, "target_win_rate needs a validation split"):
            studio_diloco.Coordinator.build("run1", self.dir, {**support.CONFIG, "validation": 0}, support.EXAMPLES,
                                            support.settings(teacher="t", target_win_rate=.5))

    # -- the start
    def test_a_restart_compares_with_the_checkpoint_it_started_from(self):
        first = self.coordinator(Versus([]), [5., 4.], rounds=2)
        self.play(first, 2)
        start = self.dir / "best.npz"
        self.assertTrue(start.is_file())
        folder = Path(self.tmp.name) / "again"
        versus = Versus([.8, .8])
        c = studio_diloco.Coordinator.build("run2", folder, support.CONFIG, support.EXAMPLES,
                                                    support.settings(inner_steps=2, rounds=4, teacher="t", target_win_rate=.5),
                                                    clock=self.clock, start_from=str(start))
        c.versus = versus
        queue = iter([3., 2.])
        c.evaluate = lambda theta: next(queue)
        self.assertEqual((folder / "start.npz").read_bytes(), start.read_bytes())
        self.play(c, 4)
        self.assertEqual(len(versus.calls), 2)  # round 0 is already a new best, compared with the checkpoint the run restarted from; round 1 too
        self.assertEqual((versus.calls[0]["old"], c.rules.stop_rule, c.rules.start_round), (str(folder / "start.npz"), "win_rate", None))

    def test_the_prompts_are_conversations_from_the_validation_split(self):
        self.coordinator(Versus([]), [5.])
        lines = (self.dir / "eval_prompts.jsonl").read_text(encoding="utf-8").splitlines()
        prompts = [json.loads(line) for line in lines]
        self.assertTrue(0 < len(prompts) <= studio_diloco.EVAL_PROMPTS)
        self.assertTrue(all(p["messages"] and p["messages"][-1]["role"] == "user" for p in prompts))

    def test_a_dataset_without_conversations_is_refused(self):
        texts = [{"text": f"Document {n}: " + "the quick brown fox. " * 8} for n in range(40)]
        with self.assertRaisesRegex(ValueError, "target_win_rate.*conversations"):
            studio_diloco.Coordinator.build("run1", self.dir, support.CONFIG, texts, support.settings(teacher="t", target_win_rate=.5))

    def test_off_by_default_no_prompts_no_start(self):
        c = support.build(self.dir, clock=self.clock, inner_steps=2, rounds=3)
        self.play(c, 3)
        self.assertFalse((self.dir / "eval_prompts.jsonl").exists() or (self.dir / "start.npz").exists())
        self.assertEqual(self.events(c, "win_rate"), [])

    def test_a_coordinator_made_from_a_replica_leaves_the_start_alone(self):
        c = self.coordinator(Versus([]), [5., 4.], rounds=4)
        self.play(c, 1)
        c.versus_on = False  # what from_replica sets
        c.rules.start_round = None
        (self.dir / "start.npz").unlink()
        c._save_start(3)
        self.assertEqual((c.rules.start_round, (self.dir / "start.npz").exists()), (None, False))

    # -- what the API says
    def test_the_run_reads_through_the_api_model_after_a_comparison_and_when_the_rule_is_off(self):
        self.addCleanup(runs.RUNS.clear)
        job = types.SimpleNamespace(id="run1", label="Version 1", created_at=100., status="running", error="", kind="pretrain_distributed",
                                    params={"version": 1, "settings": {"rounds": 8, "inner_steps": 2}, "owner": "a", "machines": [
                                        {"node": "a", "address": "local", "device": "cpu", "device_name": ""}]})
        run = runs.run_of(job)
        gate = threading.Event()
        c = self.coordinator(Versus([.3], gate=gate), [5., 4., 3.])
        c.join("a")
        self.send(c)
        self.send(c)
        shown = Run.model_validate(runs.view(run, c, job))  # while it judges
        self.assertEqual(shown.phase, "judging")
        gate.set()
        self.settle(c)
        shown = Run.model_validate(runs.view(run, c, job))
        self.assertEqual(shown.win_rate, .3)
        self.assertEqual({"judging", "win_rate"} - {e.kind for e in shown.events}, set())
        c._event("win_rate_off", "", c.next_round, "The win-rate stop rule ends here.")
        self.assertIn("win_rate_off", {e.kind for e in Run.model_validate(runs.view(run, c, job)).events})

    # -- settings
    def test_target_win_rate_needs_a_teacher_and_a_share(self):
        with self.assertRaisesRegex(ValueError, "target_win_rate needs a teacher"):
            support.settings(target_win_rate=.6)
        for bad in (0, -.1, 1.5, True, float("nan"), "half"):
            with self.assertRaises(ValueError, msg=str(bad)):
                support.settings(target_win_rate=bad, teacher="t")
        for good in (.0001, .6, 1):
            self.assertEqual(support.settings(target_win_rate=good, teacher="t").target_win_rate, good)


class Rule(unittest.TestCase):
    def test_stop_reason_names_the_rule_once_the_rate_reaches_the_target(self):
        s = support.settings(target_win_rate=.6, teacher="t")
        state = rules.State()
        self.assertIsNone(rules.stop_reason(state, s))
        state.win_rate = .9  # a reading is not a stop: it takes WIN_READINGS in a row
        self.assertIsNone(rules.stop_reason(state, s))
        state.win_hits = rules.WIN_READINGS
        found = rules.stop_reason(state, s)
        self.assertEqual(found[0], "win_rate")
        self.assertIn("90%", found[1])

    def test_record_win_counts_readings_in_a_row_and_a_miss_or_a_gap_resets_them(self):
        s, state = support.settings(target_win_rate=.6, teacher="t"), rules.State()
        rules.record_win(state, s, .6)
        self.assertEqual((state.win_rate, state.win_hits, rules.stop_reason(state, s)), (.6, 1, None))
        rules.record_win(state, s, .59)
        self.assertEqual((state.win_rate, state.win_hits), (.59, 0))
        rules.record_win(state, s, .7)
        rules.record_win(state, s, None)
        self.assertEqual((state.win_rate, state.win_hits), (.7, 0))  # no number: the last reading stays, the streak ends
        rules.record_win(state, s, .7)
        rules.record_win(state, s, 1.)
        self.assertEqual(rules.stop_reason(state, s)[0], "win_rate")

    def test_a_rate_without_a_target_stops_nothing(self):
        state = rules.State(win_rate=1., win_hits=5)
        self.assertIsNone(rules.stop_reason(state, support.settings()))

    def test_the_state_round_trips(self):
        state = rules.State(win_rate=.4, win_round=3, start_round=0, win_hits=1)
        again = rules.State.from_json(json.loads(json.dumps(state.to_json())))
        self.assertEqual((again.win_rate, again.win_round, again.start_round, again.win_hits), (.4, 3, 0, 1))


if __name__ == "__main__":
    unittest.main()
