"""Decision logic of the training pages, window-free."""
import tempfile
import unittest
from types import SimpleNamespace

import studio_gate
from studio_core import Workspace
from studio_data_kinds import add_problems
from studio_ia.pages.chat import answering_version, stream_reply, verdict_view
from studio_ia.pages.post_training import PostTrainingPage, blocker, live_view, step_line
from studio_ia.pages.pretraining import collection_view, enough_text, held_out, ratio_text, source_action, source_status, can_continue, forced_mode, pretrain_config, steps_for, word_list_changed
from studio_ia.pages.queue import total_hours
from studio_ia.pages.storage import size_text, storage_state
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
        with self.assertLogs("studio_ia.services", "ERROR"):
            stream_reply(lambda *a: 1 / 0, 1, [], None, got.append)
        self.assertEqual(got, ["Something went wrong: ZeroDivisionError. The log has the details.", None])


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
        self.assertEqual(step_line(QueueState()), "starting")
        self.assertEqual(step_line(QueueState(step=40, total=1200, eta_s=600)), "step 40 of 1,200 · about 10 min left")

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


class RecommendedCollection(unittest.TestCase):
    @staticmethod
    def fake_fetch(url, path, stop=None, emit=None, maximum=None, range_bytes=None):
        from pathlib import Path
        from studio_catalog import file_sha256
        payload = b"a first story.<|endoftext|>a second story.<|endoftext|>" if Path(path).suffix == ".txt" else b"attribution"
        Path(path).write_bytes(payload)
        return {"url": url, "filename": Path(path).name, "bytes": len(payload), "sha256": file_sha256(path)}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.svc = WorkspaceServices(Workspace(self.tmp.name))
        self.addCleanup(self.svc.close)
        self.events = []
        self.svc.subscribe(lambda kind, payload: self.events.append((kind, payload)))

    def states(self):
        return {r["id"]: r["state"] for r in self.svc.collection()}

    def test_row_labels(self):
        self.assertEqual(source_action({"state": "not downloaded"})[1], "download")
        self.assertEqual(source_action({"state": "downloaded"})[1], "import")
        self.assertEqual(source_action({"state": "downloading"}), (None, None))
        self.assertEqual(source_action({"state": "imported"}), (None, None))
        self.assertEqual(source_status({"state": "downloading"}, "Downloading x · 1.0 MB"), "Downloading x · 1.0 MB")
        self.assertIn("Could not finish: no network.", source_status({"state": "not downloaded", "error": "no network"}))

    def test_collection_view_ignores_progress_but_sees_state_changes(self):
        row = {"id": "a", "name": "A", "size": "1 MB", "records": "5", "state": "downloading", "completed": 1, "total": 9, "error": ""}
        before = collection_view([row])
        row["completed"] = 5  # progress alone does not rebuild the card
        self.assertEqual(collection_view([row]), before)
        row["state"] = "downloaded"
        self.assertNotEqual(collection_view([row]), before)
        row["error"] = "no network"
        self.assertNotEqual(collection_view([row]), collection_view([{**row, "error": ""}]))
        self.assertNotEqual(collection_view([row]), collection_view([]))

    def test_download_then_import_into_the_text_sets(self):
        from unittest.mock import patch
        self.assertEqual(set(self.states().values()), {"not downloaded"})
        with patch("studio_catalog._fetch", side_effect=self.fake_fetch):
            self.assertTrue(self.svc.download_sources(["tinystories"]))
            self.svc._worker.join(10)
            self.assertEqual(self.states()["tinystories"], "downloaded")
            self.assertIn(("download_done", {"id": "tinystories"}), self.events)
            self.assertFalse(self.svc.import_source("wikitext2"))  # not downloaded
            self.assertTrue(self.svc.import_source("tinystories"))
            self.svc._worker.join(10)
        self.assertEqual(self.states()["tinystories"], "imported")
        self.assertEqual(self.svc.data("text").count, 2)

    def test_text_downloads_and_adds_in_one_step(self):
        from unittest.mock import patch
        with patch("studio_catalog._fetch", side_effect=self.fake_fetch):
            self.assertTrue(self.svc.download_sources(["tinystories"], install=True))
            self.svc._worker.join(10)
        self.assertEqual(self.states()["tinystories"], "imported")
        self.assertEqual(self.svc.data("text").count, 2)

    def test_validation_and_test_splits_are_held_out(self):
        self.assertTrue(held_out("catalog_tinystories_valid"))
        self.assertFalse(held_out("catalog_tinystories"))
        self.assertFalse(held_out("d1"))  # the person's own set

    def test_collections_by_kind_and_problem_install(self):
        from pathlib import Path
        from unittest.mock import patch
        import studio_catalog as cat
        stage = {e["id"]: e["stage"] for e in cat.CATALOG}
        self.assertEqual({stage[i] for i in self.svc.collection_ids("text")}, {"language"})
        self.assertEqual({stage[i] for i in self.svc.collection_ids("problems")}, {"problems"})
        self.assertNotIn("winogrande_check", self.svc.collection_ids("problems"))
        self.assertTrue({"ultrachat_a", "gsm8k", "oasst1"} <= set(self.svc.collection_ids("conversations")))

        def fetch(url, path, stop=None, emit=None, maximum=None, range_bytes=None):
            if Path(path).name == "train.jsonl":
                Path(path).write_text('{"question": "What is 2 + 3?", "answer": "2 + 3 = 5\\n#### 5"}\n', encoding="utf-8")
                return {"url": url, "filename": "train.jsonl", "bytes": 1, "sha256": cat.file_sha256(path)}
            return self.fake_fetch(url, path, stop, emit, maximum, range_bytes)
        with patch("studio_catalog._fetch", side_effect=fetch):
            self.assertTrue(self.svc.download_sources(["gsm8k_check"], kind="problems", install=True))
            self.svc._worker.join(10)
        self.assertEqual({r["id"]: r["state"] for r in self.svc.collection("problems")}["gsm8k_check"], "imported")
        self.assertEqual(self.svc.data("problems").count, 1)
        rows = self.svc.post_rows("problems") + self.svc.post_rows("heldout_problems")
        self.assertEqual([(r["rule"], r["params"]) for r in rows], [("final_number", {"answer": "5"})])

    def test_failure_is_reported_and_retryable(self):
        from unittest.mock import patch
        import urllib.error
        offline = "Could not reach the download server. Check the connection and try again."
        with patch("studio_catalog._fetch", side_effect=urllib.error.URLError("no network")):
            self.svc.download_sources(["tinystories"])
            self.svc._worker.join(10)
        row = next(r for r in self.svc.collection() if r["id"] == "tinystories")
        self.assertEqual((row["state"], row["error"]), ("not downloaded", offline))
        self.assertIn(("download_failed", {"id": "tinystories", "error": offline}), self.events)
        with patch("studio_catalog._fetch", side_effect=self.fake_fetch):
            self.assertTrue(self.svc.download_sources(["tinystories"]))  # nothing left running
            self.svc._worker.join(10)
        self.assertEqual(self.states()["tinystories"], "downloaded")


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

    def test_waiting_jobs_from_an_earlier_session_start_again(self):
        from unittest import mock
        import studio_queue
        with mock.patch.object(studio_queue.Runner, "start") as start:
            self.svc.start_queue()
            start.assert_not_called()  # nothing waiting: the first enqueue starts it
            self.svc.runner.jobs.append(studio_queue.Job("quicktest", "left over", status="held"))
            self.svc.start_queue()
            start.assert_not_called()  # held jobs do not count
            self.svc.runner.jobs.append(studio_queue.Job("quicktest", "left over"))
            self.svc.start_queue()
            start.assert_called_once()

    def test_preview_is_cached_on_disk_and_never_computed_by_the_cached_read(self):
        self.svc.save_version(default_spec(), "first design")
        self.assertIsNone(self.svc.cached_preview([self.text], rows=6))  # unknown yet: no work done to find out
        preview = self.svc.pretrain_preview([self.text], rows=6)
        self.assertGreater(preview["pieces"], 0)
        again = WorkspaceServices(Workspace(self.tmp.name))  # a new session, as after a restart
        self.addCleanup(again.close)
        self.assertEqual(again.cached_preview([self.text], rows=6)["pieces"], preview["pieces"])
        self.ws.add(self.text, [{"text": "a new story changes the stored data"}])
        self.assertIsNone(self.svc.cached_preview([self.text], rows=6))

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


class PostTrainingLive(unittest.TestCase):
    """Live events must not rebuild the page for every one of them, nor when the open round shows nothing new."""

    def page(self):
        page = PostTrainingPage.__new__(PostTrainingPage)  # no window: only the event logic is used
        page.selected, page.group, page.heat, page.moved, page.live_gate = 1, None, None, None, {}
        page.gates, page._ticking, page._pending, page._step, page.services = {}, False, False, None, FakeServices()
        page.timers, page.redraws, page.reloads = [], [], []
        page.after = lambda ms, fn: page.timers.append((ms, fn))
        page.refresh = lambda: (page.redraws.append(1), setattr(page, "_drawn", live_view(page.selected, page.group, page.heat, page.moved, {**page.gates, **page.live_gate})))
        page.reload = lambda: page.reloads.append(1)
        page._drawn = live_view(1, None, None, None, {})
        return page

    def test_a_burst_draws_its_first_at_once_and_the_rest_once(self):
        page = self.page()
        for n in range(6):
            page.on_event("train", {"type": "group", "prompt": "p", "rule": "sort", "mean": .5, "answers": [], "n": n})
        self.assertEqual((len(page.timers), page.redraws), (1, [1]))
        page.timers[0][1]()
        self.assertEqual((page.redraws, page.reloads), ([1, 1], []))
        page.timers[1][1]()  # the burst is over: the lane goes quiet
        self.assertEqual((len(page.timers), page.redraws), (2, [1, 1]))

    def test_nothing_new_for_the_open_round_does_not_redraw(self):
        page = self.page()
        page.on_event("train", {"type": "heat", "prompt": "p", "words": ["a"], "surprise": [.1]})  # round 3's visual; round 2 is open
        page.timers[0][1]()
        self.assertEqual(page.redraws, [])

    def test_the_same_group_again_does_not_redraw_but_a_gate_does_at_once(self):
        page = self.page()
        group = {"type": "group", "prompt": "p", "rule": "sort", "mean": .5, "answers": []}
        page.on_event("train", group)
        page.on_event("train", dict(group))
        page.timers[0][1]()
        self.assertEqual(page.redraws, [1])
        page.on_event("train", {"type": "gate", "round": 2, "ok": True, "checks": []})
        self.assertEqual(page.redraws, [1, 1])

    def test_queue_job_and_version_events_still_reload(self):
        page = self.page()
        for kind in ("queue", "job", "versions"):
            page.on_event(kind, {})
        self.assertEqual((len(page.reloads), page.timers), (3, []))


if __name__ == "__main__":
    unittest.main()


class EnoughTextTest(unittest.TestCase):
    def verdict(self, ratio):
        return enough_text(ratio)[1]

    def test_headline_follows_the_rule_of_thumb(self):
        self.assertEqual(self.verdict(2), "little")
        self.assertEqual(self.verdict(15.9), "little")
        self.assertEqual(self.verdict(16), "about_right")
        self.assertEqual(self.verdict(20), "about_right")
        self.assertEqual(self.verdict(30), "about_right")
        self.assertEqual(self.verdict(30.5), "plenty")
        self.assertEqual(self.verdict(549.54), "plenty")

    def test_far_above_the_rule_says_plenty_not_about_as_much(self):
        headline, _ = enough_text(549.54)
        self.assertIn("plenty", headline)
        self.assertIn("27", headline)
        self.assertNotIn("about as much", headline)
        self.assertIn("about 10.0", enough_text(2)[0])

    def test_ratio_reads_plainly(self):
        self.assertEqual(ratio_text(2.0), "2.00")
        self.assertEqual(ratio_text(27.46), "27.5")
        self.assertEqual(ratio_text(549.54), "about 550")
        self.assertEqual(ratio_text(12345.6), "about 12,000")
