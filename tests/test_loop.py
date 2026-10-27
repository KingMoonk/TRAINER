"""The automated RLHF loop (studio_loop): kept and rolled-back rounds by the gate, the journal, resuming after a restart, the audit
half, the judge going away. The judge is a scripted fake; Ollama is never called."""
import hashlib
import json
import shutil
import tempfile
import threading
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import studio_jobs
import studio_loop
import studio_rlhf
from studio_model import train
from studio_rlhf import TeacherError

DOCS = [f"held back document number {i} about harbors and lanterns" for i in range(10)]
HELD = [{"prompt": f"{a} + {b} =", "rule": "small_addition", "params": {}} for a in range(1, 5) for b in range(1, 4)]
PROBLEMS = [{"prompt": f"Copy: {w}", "rule": "copy", "params": {"text": w}} for w in ("harbor", "mirror", "pebble", "window", "violin")]
CONVERSATIONS = [{"user": f"say {w}", "assistant": w} for w in ("yes", "no", "cat", "dog", "sun", "red", "sky", "big", "ant", "bee", "elk", "fox")]
ROWS = {"conversations": CONVERSATIONS, "problems": PROBLEMS, "heldout_docs": DOCS, "heldout_problems": HELD}
GATE_DOCS, AUDIT_DOCS = studio_loop.split_halves(DOCS)
GATE_PROBLEMS, AUDIT_PROBLEMS = studio_loop.split_halves(HELD)
CONFIG = {"max_rounds": 3, "patience": 5, "prompts_per_round": 6, "candidates": 3, "replay": 0, "gate_prompts": 2, "dpo_steps": 2}


class FakeJudge:
    """Grades by a fixed hash of the reply; never talks to a network. `calls` counts the grading requests."""
    model = "fake-judge"

    def __init__(self, fail_after=None, fatal=False):
        self.calls, self.fail_after, self.fatal = 0, fail_after, fatal

    def _maybe_fail(self):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise TeacherError("the fake judge is down", fatal=self.fatal)

    def score_many(self, messages, replies, reference=None, rng=None):
        self._maybe_fail()
        return [1. + int(hashlib.sha1(r.encode()).hexdigest()[:2], 16) % 10 for r in replies]

    def judge(self, messages, first, second, reference=None, both_orders=True, rng=None):
        self._maybe_fail()
        return {"scores": [len(first) % 7, len(second) % 7], "reason": "fake", "consistent": True}

    def rewrite(self, messages, max_chars=240):
        return "the ideal reply"

    def score(self, messages, reply, reference=None):
        return self.score_many(messages, [reply])[0]


class ScriptedModel:
    """A model whose replies are always new: 'answer 0', 'answer 1'..."""

    def __init__(self):
        self.count = 0

    def generate(self, messages, **kwargs):
        self.count += 1
        return f"answer {self.count}"

    def generate_group(self, messages, n, **kwargs):
        return [self.generate(messages) for _ in range(n)]


def fake_trainer(source_hook=None):
    """Stands in for dpo_train: 'trains' by copying the parent's file into dpo/last.npz."""
    def trainer(base, pairs, output_dir, config=None, emit=None, stop=None, pause=None, name=None):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base, out / "last.npz")
        if source_hook:
            source_hook(out)
        return {"checkpoint": str(out / "last.npz"), "step": config["steps"], "pairs": len(pairs), "final_validation": {"reward_accuracy": .75, "chosen_len": 9., "rejected_len": 8.}}
    return trainer


def scripted_measure(table):
    """measure(path, docs, problems) from {round number: {"bpc", "pass_k", "audit_pass_k"}}; the starting model is round 0."""
    def measure(path, docs, problems):
        round_number = int(path.split("round_")[1][:4]) if "round_" in path else 0
        row = table.get(round_number, table[0])
        audit = problems == AUDIT_PROBLEMS
        return {"bpc": row["bpc"], "pass_k": row.get("audit_pass_k", row["pass_k"]) if audit else row["pass_k"], "pass_n": 10}
    return measure


def win_rates(rates):
    def versus(candidate, current, prompts, judge):
        round_number = int(candidate.split("round_")[1][:4])
        return {"win_rate": rates[round_number], "wins": 1, "ties": 0, "losses": 0, "n": 4, "examples": [{"a": "x" * 10, "b": "y" * 10}]}
    return versus


class LoopCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "loop"
        self.start = Path(self.tmp.name) / "start.npz"
        self.start.write_text("weights", encoding="utf-8")
        self.addCleanup(self.tmp.cleanup)

    def run_loop(self, measure, versus=None, judge="fake", config=None, stop=None, **kw):
        return studio_loop.run_loop(self.dir, str(self.start), ROWS, FakeJudge() if judge == "fake" else judge, {**CONFIG, **(config or {})},
                                    stop=stop, measure=measure, band=lambda ckpt, docs: None, trainer=kw.pop("trainer", fake_trainer()),
                                    versus=versus, collector=kw.pop("collector", partial(studio_loop.collect, model=ScriptedModel())), **kw)

    def journal(self):
        return studio_loop.read_jsonl(studio_loop.paths(self.dir)["journal"])


class Rounds(LoopCase):
    def test_three_rounds_kept_and_rolled_back_as_the_gate_says(self):
        table = {0: {"bpc": 2.0, "pass_k": 5}, 1: {"bpc": 1.9, "pass_k": 7},   # round 1: better on both
                 2: {"bpc": 2.5, "pass_k": 7},                                  # round 2: held-back error rose
                 3: {"bpc": 1.85, "pass_k": 8}}                                 # round 3: better again, against round 1
        result = self.run_loop(scripted_measure(table), win_rates({1: .8, 2: .9, 3: .8}))
        journal = self.journal()
        self.assertEqual([j["status"] for j in journal], ["kept", "rolled_back", "kept"])
        self.assertEqual([j["round"] for j in journal], [1, 2, 3])
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["kept"], 2)
        self.assertTrue(result["current"].replace("\\", "/").endswith("round_0003/dpo/last.npz"))
        rolled = journal[1]
        self.assertIn("held-back", rolled["reason"].lower())
        self.assertFalse(next(c for c in rolled["checks"] if c["name"].startswith("Held-back text"))["ok"])
        self.assertEqual(rolled["rolled_back_to"], journal[0]["checkpoint"])   # round 2 started from round 1's kept checkpoint
        self.assertEqual(journal[2]["parent"], journal[0]["checkpoint"])
        for entry in journal:   # every round records its evidence
            for key in ("pairs", "checks", "before", "after", "judge", "audit", "dpo", "seconds", "started", "finished"):
                self.assertIn(key, entry)
        self.assertEqual(journal[0]["pairs"]["total"], sum(journal[0]["pairs"][k] for k in ("verified", "comparison", "rewrite")))
        self.assertFalse((self.dir / "round_0002" / "dpo").exists())   # a rolled-back round's weights are gone, its evidence stays
        self.assertTrue((self.dir / "round_0002" / "collect.jsonl").exists())

    def test_nothing_improved_is_rolled_back_even_when_nothing_got_worse(self):
        table = {0: {"bpc": 2.0, "pass_k": 5}, 1: {"bpc": 2.0, "pass_k": 5}}
        self.run_loop(scripted_measure(table), win_rates({1: .5}), config={"max_rounds": 1})
        entry = self.journal()[0]
        self.assertEqual(entry["status"], "rolled_back")
        self.assertFalse(next(c for c in entry["checks"] if c["name"] == "Something improved")["ok"])

    def test_the_judge_alone_cannot_keep_a_round_that_got_longer(self):
        table = {0: {"bpc": 2.0, "pass_k": 5}, 1: {"bpc": 2.0, "pass_k": 5}}
        longer = lambda c, cur, prompts, judge: {"win_rate": .95, "wins": 4, "ties": 0, "losses": 0, "n": 4, "examples": [{"a": "x" * 40, "b": "y" * 10}]}
        self.run_loop(scripted_measure(table), longer, config={"max_rounds": 1})
        entry = self.journal()[0]
        self.assertEqual(entry["status"], "rolled_back")
        self.assertIn("longer", entry["reason"])

    def test_the_drift_budget_stops_small_losses_adding_up(self):
        # the starting model had 2.0; the budget is 2%
        kept, checks, _ = studio_loop.decide({"bpc": 2.028, "pass_k": 6, "pass_n": 10}, {"bpc": 2.028, "pass_k": 7, "pass_n": 10},
                                             {"bpc": 2.0, "pass_k": 5, "pass_n": 10}, (1.9, 2.1), None, None, None, studio_loop.clean_config({}))
        self.assertTrue(kept)
        kept, checks, _ = studio_loop.decide({"bpc": 2.05, "pass_k": 6, "pass_n": 10}, {"bpc": 2.05, "pass_k": 7, "pass_n": 10},
                                             {"bpc": 2.0, "pass_k": 5, "pass_n": 10}, (1.9, 2.1), None, None, None, studio_loop.clean_config({}))
        self.assertFalse(kept)
        self.assertFalse(next(c for c in checks if c["name"].startswith("Drift from the starting model: held-back"))["ok"])

    def test_stop_rule_patience(self):
        table = {0: {"bpc": 2.0, "pass_k": 5}, **{n: {"bpc": 3.0, "pass_k": 1} for n in range(1, 10)}}
        result = self.run_loop(scripted_measure(table), win_rates({n: .9 for n in range(1, 10)}), config={"max_rounds": 9, "patience": 2})
        self.assertEqual(len(self.journal()), 2)
        self.assertEqual(result["status"], "finished")
        self.assertIn("stopped improving", result["stop_reason"])

    def test_the_audit_half_vetoes_and_repeated_flags_stop_the_loop(self):
        # passes the gate half every time; the audit half falls well below the starting model
        table = {0: {"bpc": 2.0, "pass_k": 5, "audit_pass_k": 5}, **{n: {"bpc": 1.9, "pass_k": 6 + n, "audit_pass_k": 1} for n in range(1, 10)}}
        result = self.run_loop(scripted_measure(table), win_rates({n: .9 for n in range(1, 10)}), config={"max_rounds": 9, "audit_flags": 2})
        journal = self.journal()
        self.assertEqual([j["status"] for j in journal], ["rolled_back", "rolled_back"])
        self.assertTrue(all(j["audit_flag"] for j in journal))
        self.assertIn("judge exploitation", result["stop_reason"])
        self.assertEqual(result["current"], str(self.start))   # nothing flagged was ever kept

    def test_checked_problems_alone_run_without_a_judge(self):
        table = {0: {"bpc": 2.0, "pass_k": 5}, 1: {"bpc": 2.0, "pass_k": 6}}
        rows = dict(ROWS, conversations=[])
        studio_loop.run_loop(self.dir, str(self.start), rows, None, {**CONFIG, "max_rounds": 1}, measure=scripted_measure(table),
                             band=lambda c, d: None, trainer=fake_trainer(), collector=partial(studio_loop.collect, model=PassingModel()))
        entry = self.journal()[0]
        self.assertEqual(entry["status"], "kept")
        self.assertEqual(entry["pairs"]["verified"], entry["pairs"]["total"])
        self.assertIsNone(entry["judge"])

    def test_too_few_pairs_is_not_a_failure_of_the_run(self):
        table = {0: {"bpc": 2.0, "pass_k": 5}}
        silent = lambda spec, judge, *a, **k: {"pairs": [], "skipped": {"scores too close": 6}, "errors": 0}
        self.run_loop(scripted_measure(table), config={"max_rounds": 1}, collector=silent)
        entry = self.journal()[0]
        self.assertEqual(entry["status"], "too_few_pairs")
        self.assertEqual(entry["pairs"]["skipped"], {"scores too close": 6})

    def test_training_that_fails_rolls_back_and_the_loop_goes_on(self):
        def broken(*args, **kwargs):
            raise RuntimeError("out of memory")
        self.run_loop(scripted_measure({0: {"bpc": 2.0, "pass_k": 5}}), config={"max_rounds": 2}, trainer=broken)
        self.assertEqual([j["status"] for j in self.journal()], ["rolled_back", "rolled_back"])
        self.assertIn("out of memory", self.journal()[0]["error"])


class PassingModel:
    """Half of the answers to 'Copy: x' are right."""

    def __init__(self):
        self.n = 0

    def generate_group(self, messages, n, **kwargs):
        word = messages[-1]["content"].split(": ", 1)[1]
        return [word if i % 2 == 0 else f"{word}!?{self.n}" for i in range(n)]


class Resume(LoopCase):
    def test_a_restart_carries_on_from_the_last_kept_round(self):
        table = {0: {"bpc": 2.0, "pass_k": 5}, 1: {"bpc": 1.9, "pass_k": 7}, 2: {"bpc": 1.8, "pass_k": 8}}
        stop = threading.Event()
        calls = []

        def measure(path, docs, problems):
            calls.append(path)
            if "round_0002" in path:   # the app goes away in the middle of round 2's gate
                stop.set()
            return scripted_measure(table)(path, docs, problems)
        self.run_loop(measure, win_rates({1: .9, 2: .9}), stop=stop, config={"max_rounds": 2})
        state = studio_loop.load_state(self.dir)
        self.assertEqual(state["status"], "stopped")
        self.assertEqual(state["next_round"], 2)
        self.assertEqual([j["status"] for j in self.journal()], ["kept"])
        kept_one = state["current"]
        self.assertTrue(kept_one.replace("\\", "/").endswith("round_0001/dpo/last.npz"))

        calls.clear()
        again = self.run_loop(lambda *a: (calls.append(a[0]), scripted_measure(table)(*a))[1], win_rates({1: .9, 2: .9}), config={"max_rounds": 2})
        self.assertEqual([j["round"] for j in self.journal()], [1, 2])   # round 1 is not done again
        self.assertEqual(self.journal()[1]["parent"], kept_one)
        self.assertEqual(self.journal()[1]["status"], "kept")
        self.assertNotIn(str(self.start), calls)                          # the starting model was measured once, before
        self.assertEqual(again["id"], state["id"])
        self.assertEqual(again["anchor"], str(self.start))

    def test_a_restart_in_the_middle_of_judging_does_not_pay_for_it_twice(self):
        stop = threading.Event()

        class Stopping(FakeJudge):
            def score_many(self, *args, **kwargs):
                result = super().score_many(*args, **kwargs)
                if self.calls == 2:
                    stop.set()
                return result
        first = Stopping()
        table = {0: {"bpc": 2.0, "pass_k": 5}, 1: {"bpc": 1.9, "pass_k": 7}}
        self.run_loop(scripted_measure(table), win_rates({1: .9}), judge=first, stop=stop, config={"max_rounds": 1, "prompts_per_round": 6})
        stored = studio_loop.read_jsonl(self.dir / "round_0001" / "collect.jsonl")
        self.assertGreater(len(stored), 0)
        self.assertLess(len(stored), 6)
        second = FakeJudge()
        self.run_loop(scripted_measure(table), win_rates({1: .9}), judge=second, config={"max_rounds": 1, "prompts_per_round": 6})
        keys = [r["key"] for r in studio_loop.read_jsonl(self.dir / "round_0001" / "collect.jsonl")]
        self.assertEqual(len(keys), len(set(keys)))                       # no prompt judged twice
        self.assertLess(second.calls, 6 + 6)                              # the stored prompts were not asked again
        self.assertLess(second.calls, first.calls + second.calls)

    def test_a_loop_that_finished_does_not_start_over(self):
        table = {0: {"bpc": 2.0, "pass_k": 5}, 1: {"bpc": 1.9, "pass_k": 7}}
        self.run_loop(scripted_measure(table), win_rates({1: .9}), config={"max_rounds": 1})
        self.run_loop(scripted_measure(table), win_rates({1: .9}), config={"max_rounds": 1})
        self.assertEqual(len(self.journal()), 1)
        self.assertEqual(studio_loop.load_state(self.dir)["status"], "finished")


class Judge(LoopCase):
    def test_an_unreachable_judge_is_waited_for_then_the_loop_stops_resumably(self):
        waits = []
        judge = FakeJudge(fail_after=0, fatal=True)
        result = self.run_loop(scripted_measure({0: {"bpc": 2.0, "pass_k": 5}}), judge=judge,
                               config={"judge_wait_s": 100, "max_rounds": 5}, wait=waits.append,
                               collector=partial(studio_loop.collect, model=ScriptedModel()))
        self.assertTrue(waits)
        self.assertEqual(result["status"], "stopped")
        self.assertIn("unreachable", result["stop_reason"])
        self.assertEqual(result["round"], 1)   # still on round 1: Start carries on from there

    def test_refuses_to_start_without_held_back_data_or_something_to_learn_from(self):
        with self.assertRaisesRegex(studio_loop.LoopError, "held-back"):
            studio_loop.run_loop(self.dir, str(self.start), dict(ROWS, heldout_docs=[], heldout_problems=[]), FakeJudge(), CONFIG)
        with self.assertRaisesRegex(studio_loop.LoopError, "Ollama judge"):
            studio_loop.run_loop(self.dir / "b", str(self.start), dict(ROWS, problems=[]), None, CONFIG)

    def test_gate_prompts_are_never_used_to_make_pairs(self):
        seen = []
        real = studio_loop.collect

        def spy(spec, judge, *a, **k):
            seen.extend(item["prompt"]["messages"][-1]["content"] for item in spec["items"] if "prompt" in item)
            return real(spec, judge, *a, model=ScriptedModel(), **k)
        self.run_loop(scripted_measure({0: {"bpc": 2.0, "pass_k": 5}, 1: {"bpc": 1.9, "pass_k": 7}}), win_rates({1: .9}),
                      config={"max_rounds": 1, "prompts_per_round": 50}, collector=spy)
        held_back = sorted(studio_rlhf.prompts_from(CONVERSATIONS), key=lambda p: studio_loop._key(p["messages"]))[:2]
        for prompt in held_back:
            self.assertNotIn(prompt["messages"][-1]["content"], seen)
        self.assertGreater(len(seen), 0)


class Adoption(unittest.TestCase):
    def test_the_loops_best_is_checked_on_all_the_held_back_data(self):
        state = {"anchor": "a", "current": "round_0001/x", "kept": 1}
        better = lambda path, docs, problems: {"bpc": 1.9 if "round" in path else 2.0, "pass_k": 7 if "round" in path else 5, "pass_n": 10}
        result = studio_loop.adoption(state, {"heldout_docs": DOCS, "heldout_problems": HELD}, measure=better, band=lambda c, d: None)
        self.assertTrue(result["ok"])
        self.assertEqual((result["round"], result["checkpoint"]), (5, "round_0001/x"))
        worse = lambda path, docs, problems: {"bpc": 2.5 if "round" in path else 2.0, "pass_k": 5, "pass_n": 10}
        self.assertFalse(studio_loop.adoption(state, {"heldout_docs": DOCS, "heldout_problems": HELD}, measure=worse, band=lambda c, d: None)["ok"])

    def test_nothing_kept_means_nothing_to_adopt(self):
        state = {"anchor": "a", "current": "a", "kept": 0}
        result = studio_loop.adoption(state, {}, measure=lambda *a: self.fail("nothing to measure"), band=lambda c, d: None)
        self.assertFalse(result["ok"])
        self.assertIn("nothing to adopt", result["verdict"])


class Pairs(unittest.TestCase):
    def test_rule_pairs_pick_the_median_length_pass_against_a_fail(self):
        import random

        class Model:
            def generate_group(self, messages, n, **kwargs):
                return ["harbor", "harbor ", "harbor     ", "wrong", "x"][:n]
        pairs, info = studio_loop.rule_pair(Model(), PROBLEMS[0], studio_loop.clean_config({"candidates": 5}), random.Random(1), threading.Event())
        self.assertEqual(pairs[0]["chosen"], "harbor ")
        self.assertIn(pairs[0]["rejected"], ("wrong", "x"))
        self.assertEqual(pairs[0]["kind"], "verified")

    def test_no_pair_when_every_answer_passes_or_fails(self):
        import random
        for answers, skip in ((["harbor", "harbor "], "every answer passed"), (["a", "b"], "no answer passed")):
            class Model:
                def generate_group(self, messages, n, **kwargs):
                    return answers
            pairs, info = studio_loop.rule_pair(Model(), PROBLEMS[0], studio_loop.clean_config({"candidates": 2}), random.Random(1), threading.Event())
            self.assertEqual(pairs, [])
            self.assertEqual(info["skip"], skip)

    def test_the_split_is_stable_and_disjoint(self):
        a, b = studio_loop.split_halves(DOCS)
        self.assertEqual(sorted(a + b), sorted(DOCS))
        self.assertEqual(studio_loop.split_halves(list(reversed(DOCS))), (a, b))


class RealPipeline(unittest.TestCase):
    """The real preference round, measurement and checkpoint files on a tiny model (the judge and the sampled replies are scripted)."""

    def test_one_round_with_the_real_trainer_and_measure(self):
        tiny = dict(device="cpu", width=32, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=4, steps=20, learning_rate=.004,
                    warmup=0, validation=0, eval_every=10, objective="assistant")
        with tempfile.TemporaryDirectory() as tmp:
            train(tiny, [{"user": f"say {w}", "assistant": w} for w in ("yes", "no", "cat", "dog")] * 4, Path(tmp) / "base")
            checkpoint = str(Path(tmp) / "base" / "last.npz")
            result = studio_loop.run_loop(Path(tmp) / "loop", checkpoint, dict(ROWS, heldout_docs=[f"say yes {i}" for i in range(6)]), FakeJudge(),
                                          {**CONFIG, "max_rounds": 1, "dpo_steps": 4, "gate_prompts": 0, "device": "cpu"},
                                          versus=None, collector=partial(studio_loop.collect, model=ScriptedModel()))
            entry = studio_loop.read_jsonl(Path(tmp) / "loop" / "journal.jsonl")[0]
            self.assertIn(entry["status"], ("kept", "rolled_back"))
            self.assertTrue(json.dumps(entry))   # plain JSON all the way
            self.assertEqual(result["round"], 2)
            self.assertIn("bpc", entry["after"])


class Job(unittest.TestCase):
    def test_the_job_kind_is_registered_and_runs_the_loop(self):
        self.assertIn("rlhf_loop", studio_jobs.HANDLERS)
        with tempfile.TemporaryDirectory() as tmp:
            start = Path(tmp) / "start.npz"
            start.write_text("weights", encoding="utf-8")
            job = SimpleNamespace(params={"output_dir": str(Path(tmp) / "loop"), "checkpoint": str(start), "version": 1, "config": {**CONFIG, "max_rounds": 1},
                                          "rows": dict(ROWS, conversations=[]), "loop_id": "loop-x"})
            context = SimpleNamespace(root=tmp, run_dir=tmp, post_rows=lambda kind: [])
            events = []
            from unittest.mock import patch
            with patch("studio_loop.collect", lambda spec, judge, *a, **k: {"pairs": [], "skipped": {}, "errors": 0}), \
                 patch("studio_rl.measure", lambda ckpt, docs, problems: {"bpc": 2.0, "pass_k": 3, "pass_n": 10}), \
                 patch("studio_eval.noise_band", lambda ckpt, docs: None):
                studio_jobs.HANDLERS["rlhf_loop"](job, context, events.append, threading.Event(), threading.Event())
            state = studio_loop.load_state(Path(tmp) / "loop")
            self.assertEqual(state["id"], "loop-x")
            self.assertTrue(any(e["type"] == "loop_round" for e in events))


if __name__ == "__main__":
    unittest.main()
