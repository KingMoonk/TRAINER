"""Decision logic of the training pages, window-free."""
import tempfile
import unittest
from types import SimpleNamespace

import studio_gate
from studio_core import Workspace
from studio_data_kinds import add_problems
from studio_ia.pages.chat import answering_version, stream_reply, verdict_view
from studio_ia.pages.post_training import blocker, step_line
from studio_ia.pages.pretraining import can_continue, forced_mode, pretrain_config, steps_for, word_list_changed
from studio_ia.pages.queue import size_text, storage_state, total_hours
from studio_ia.pages.training_run import history_of, phase_at
from studio_ia.services import FakeServices, QueueState, Job, WorkspaceServices, layout_rows
from studio_model import validate_config
from studio_spec import default_spec

BYTE, BPE = {"id": "byte", "config": {}}, {"id": "bpe_gpt2", "config": {}}


def version(tokenizer, checkpoint=None):
    return SimpleNamespace(spec={"tokenizer": tokenizer}, best_checkpoint=checkpoint)


class PretrainingMode(unittest.TestCase):
    def test_same_word_list_continues(self):
        v, parent = version(BYTE), version(BYTE, "p.npz")
        self.assertTrue(can_continue(v, parent))
        self.assertEqual(forced_mode(v, parent, "catch_up"), "catch_up")

    def test_word_list_change_forces_scratch(self):
        v, parent = version(BPE), version(BYTE, "p.npz")
        self.assertTrue(word_list_changed(v, parent))
        self.assertEqual(forced_mode(v, parent, "catch_up"), "from_scratch")

    def test_untrained_or_missing_parent_forces_scratch(self):
        self.assertEqual(forced_mode(version(BYTE), version(BYTE), "catch_up"), "from_scratch")
        self.assertEqual(forced_mode(version(BYTE), None, "catch_up"), "from_scratch")

    def test_scratch_is_always_allowed(self):
        self.assertEqual(forced_mode(version(BYTE), version(BYTE, "p.npz"), "from_scratch"), "from_scratch")


class ChatStreaming(unittest.TestCase):
    def run_pair(self):
        out = {0: [], 1: []}

        def chat(num, messages, on_text, stop):
            text = ""
            for w in (f"v{num} ", "says ", "hi"):
                text += w
                on_text(text)
            return text
        for i, num in enumerate((2, 1)):
            stream_reply(chat, num, [], None, out[i].append)
        return out

    def test_compare_both_panes_get_cumulative_text_then_end(self):
        out = self.run_pair()
        self.assertEqual(out[0][-2:], ["v2 says hi", None])
        self.assertEqual(out[1][-2:], ["v1 says hi", None])

    def test_final_text_used_when_nothing_streamed_and_errors_shown(self):
        got = []
        stream_reply(lambda *a: "whole", 1, [], None, got.append)
        self.assertEqual(got, ["whole", None])
        got = []
        stream_reply(lambda *a: 1 / 0, 1, [], None, got.append)
        self.assertTrue(got[0].startswith("[error:") and got[-1] is None)


class PageLogic(unittest.TestCase):
    def test_steps_and_config(self):
        self.assertEqual(steps_for(10_000, 4096, 2), 5)
        self.assertIsNone(steps_for(None, 4096, 2))
        cfg = validate_config(pretrain_config(default_spec(), 50, 8, .05))
        self.assertEqual((cfg["steps"], cfg["batch_size"], cfg["validation"], cfg["objective"]), (50, 8, .05, "all"))

    def test_phase_and_history(self):
        phases = [(0, 10, "warm-up"), (10, 80, "steady"), (80, 100, "cool-down")]
        self.assertEqual([phase_at(phases, k) for k in (0, 10, 11, 80, 81, 100)], ["warm-up", "warm-up", "steady", "steady", "cool-down", "cool-down"])
        self.assertEqual(phase_at([], 5), "")
        self.assertEqual(history_of([(0, 3.), (5, 2.)], [(5, 2.5)]), [{"loss": 3., "val_loss": None}, {"loss": 2., "val_loss": 2.5}])

    def test_queue_total_hours(self):
        self.assertIsNone(total_hours(QueueState()))
        job = Job("a", "pretrain")
        self.assertIsNone(total_hours(QueueState(job=job, eta_s=None)))
        self.assertEqual(total_hours(QueueState(job=job, eta_s=3600, up_next=[Job("b", "compare", eta_s=1800)])), 1.5)

    def test_post_blockers_and_step_line(self):
        self.assertIn("Pretraining", blocker(0, {"conversations": 5}, checkpoint=False))
        self.assertIsNone(blocker(0, {"conversations": 5}))
        self.assertIn("conversations", blocker(0, {}))
        self.assertIn("2 pairs", blocker(3, {"pairs": 1}))
        self.assertIsNone(blocker(3, {"pairs": 2}))
        self.assertIn("earlier version", blocker(2, {}, 0, False))
        self.assertIn("correction", blocker(2, {}, 1, True))
        self.assertEqual(step_line(QueueState(step=5, total=20, eta_s=600)), "step 5 of 20 · about 10 min left")
        self.assertEqual(step_line(QueueState()), "step [N] of [N] · about [N] min left")

    def test_chat_follows_best_unless_picked(self):
        self.assertEqual(answering_version([1, 2, 3], 2, None), 2)
        self.assertEqual(answering_version([1, 2, 3], 2, 3), 3)
        self.assertEqual(answering_version([1, 2, 3], 2, None, follows=False, current=3), 3)
        self.assertIsNone(answering_version([], None, None))

    def test_verdict_rule(self):
        v = lambda n, status: SimpleNamespace(number=n, status=status)  # noqa: E731
        line, _, ok = verdict_view(v(1, "trained"), v(2, "training"))
        self.assertFalse(ok)
        self.assertIn("hasn’t finished", line.replace("'", "’"))
        done = {"comparable": True, "percent": 40.}
        line, _, ok = verdict_view(v(1, "trained"), v(2, "trained"), done)
        self.assertTrue(ok)
        self.assertIn("60%", line)
        line, _, ok = verdict_view(v(1, "trained"), v(2, "trained"), {"comparable": False, "reason": "The runs have different total lengths."})
        self.assertFalse(ok)
        self.assertTrue(line.endswith("different total lengths."))

    def test_layout_rows_marks_document_ends(self):
        rows = layout_rows([40, 100], 64, 3, cells=8)
        self.assertEqual(len(rows[0]), 8)
        self.assertIn(-1, rows[0])
        self.assertTrue(set(rows[0]) <= {-1, 0, 1})


class StorageCard(unittest.TestCase):
    class Fake(FakeServices):
        def storage(self, days=None, data=False):
            self.asked = (days, data)
            return (3, 2_500_000)

    def test_choice_maps_to_days_and_data_only_with_all_history(self):
        f = self.Fake()
        self.assertEqual(storage_state(f, "Older than 30 days", True)[:3], (30, False, False))
        self.assertEqual(f.asked, (30, False))
        days, data, box, line, ok = storage_state(f, "All history", True)
        self.assertEqual((days, data, box, ok), (None, True, True, True))
        self.assertEqual(line, "This frees 2.5 MB in 3 items")

    def test_nothing_to_delete_disables_the_button(self):
        days, data, box, line, ok = storage_state(FakeServices(), "Older than 7 days")
        self.assertEqual((days, ok, line), (7, False, "Nothing to delete"))

    def test_one_day_choice(self):
        f = self.Fake()
        self.assertEqual(storage_state(f, "Older than 1 day")[0], 1)

    def test_sizes(self):
        self.assertEqual((size_text(2_500_000), size_text(3_200_000_000)), ("2.5 MB", "3.2 GB"))


class GateRules(unittest.TestCase):
    def test_rise_inside_the_noise_band_passes_beyond_it_fails(self):
        before, band = {"bpc": 2.0, "pass_k": 3, "pass_n": 5}, (1.95, 2.1)
        self.assertTrue(studio_gate.passes(before, {"bpc": 2.05, "pass_k": 3, "pass_n": 5}, band)[0])
        self.assertFalse(studio_gate.passes(before, {"bpc": 2.2, "pass_k": 3, "pass_n": 5}, band)[0])
        self.assertFalse(studio_gate.passes(before, {"bpc": 2.05, "pass_k": 3, "pass_n": 5})[0])
        self.assertFalse(studio_gate.passes(before, {"bpc": 1.9, "pass_k": 2, "pass_n": 5}, band)[0])  # pass rate fell

    def test_chat_moves_only_below_the_band(self):
        self.assertTrue(studio_gate.moves_chat(1.0, None, None))
        self.assertFalse(studio_gate.moves_chat(1.97, 2.0, (1.95, 2.1)))
        self.assertTrue(studio_gate.moves_chat(1.9, 2.0, (1.95, 2.1)))
        self.assertFalse(studio_gate.moves_chat(None, 2.0, None))


class WorkspaceRows(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Workspace(self.tmp.name)
        self.svc = WorkspaceServices(self.ws)
        self.addCleanup(self.svc.close)
        self.text = self.ws.create_dataset("stories")
        self.ws.add(self.text, [{"text": f"story number {i} " * 5} for i in range(20)])
        self.conv = self.ws.create_dataset("talk")
        self.ws.add(self.conv, [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(30)])
        self.prob = self.ws.create_dataset("problems")
        add_problems(self.ws, self.prob, _problems())

    def test_held_back_text_is_fixed_and_never_trained_on(self):
        train = self.svc.examples({"datasets": [self.text], "heldout_fraction": .2})
        held = set(self.svc._heldout())
        self.assertTrue(held)
        self.assertFalse(held & {e["text"] for e in train})
        self.assertEqual(len(train) + len(held), 20)
        again = self.svc.examples({"datasets": [self.text], "heldout_fraction": .5})
        self.assertEqual(set(self.svc._heldout()), held)  # fixed on first use
        self.assertEqual(len(again), len(train))

    def test_post_rows_keep_held_back_apart(self):
        train, held = self.svc.post_rows("problems"), self.svc.post_rows("heldout_problems")
        self.assertEqual(len(train) + len(held), 6)
        self.assertTrue(all(p["rule"] and p["prompt"] for p in train + held))
        convs = self.svc.post_rows("conversations")
        self.assertTrue(0 < len(convs) <= 30)

    def test_corrections_and_pairs_are_saved_for_rounds_3_and_4(self):
        messages = [{"role": "user", "content": "Largest planet?"}]
        self.svc.teach_correction(None, messages, "Jupiter")
        self.assertEqual(self.svc.post_rows("corrections")[0]["correction"], "Jupiter")
        self.svc.teach_correction(None, messages, "Jupiter is the largest.", whole_answer=True)
        self.assertGreater(self.svc.data("conversations").count, 30)
        self.svc.rate_pair(messages, {"version": 2, "reply": "Jupiter"}, {"version": 1, "reply": "Mars"})
        self.assertEqual(len(self.svc.post_rows("pairs")), 1)
        with self.assertRaises(ValueError):
            self.svc.rate_pair(messages, {"version": 2, "reply": "same"}, {"version": 1, "reply": "same"})

    def test_only_a_passed_gate_moves_the_checkpoint(self):
        self.svc.save_version(default_spec(), "first design")
        result = {"ok": False, "checks": [], "verdict": "v", "before": {"bpc": 2.}, "after": {"bpc": 2.5}, "checkpoint": "/x/last.npz"}
        self.svc.record_round(1, 1, result)
        self.assertIsNone(self.svc.versions()[0].best_checkpoint)
        state = self.svc.post_training(1)
        self.assertEqual((state["rounds"][1]["status"], state["rounds"][1]["note"]), ("done", "kept, not live"))
        self.svc.record_round(1, 1, {**result, "ok": True, "after": {"bpc": 1.5, "pass_k": 2, "pass_n": 4}})
        v = self.svc.versions()[0]
        self.assertTrue(v.best_checkpoint.endswith("last.npz"))
        self.assertEqual((v.scorecard["bpc"], v.scorecard["answers"], v.scorecard["problems"]), (1.5, True, [2, 4]))
        self.assertEqual(self.svc.post_training(1)["rounds"][2]["status"], "idle")

    def test_verdict_needs_finished_runs(self):
        self.svc.save_version(default_spec(), "a")
        self.svc.save_version(default_spec(), "b")
        res = self.svc.verdict(1, 2)
        self.assertFalse(res["comparable"])
        self.assertIsNone(res["percent"])


def _problems():
    import random

    from studio_rules import generate
    return generate("copy", random.Random(1), 6)


if __name__ == "__main__":
    unittest.main()
