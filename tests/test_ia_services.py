"""WorkspaceServices over the fixture workspace (no display)."""
import math
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from studio_ia.services import FixtureServices, WorkspaceServices


class ServicesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.svc = FixtureServices(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.svc.close()
        cls.tmp.cleanup()

    def test_versions(self):
        self.assertEqual([v.number for v in self.svc.versions()], [1, 2])
        best = self.svc.best_version()
        self.assertTrue(best.best_checkpoint)
        self.assertEqual(self.svc.current_version().number, 2)

    def test_run_curve(self):
        run = self.svc.versions()[0].runs[0]
        c = self.svc.run_curve(run)
        self.assertTrue(c["steps"] and c["smooth"] and c["heldout"] and c["lr"])
        self.assertEqual(len(c["steps"]), len(c["smooth"]))
        self.assertTrue(all(math.isfinite(y) for _, y in c["steps"] + c["smooth"] + c["heldout"]))
        self.assertEqual([p[2] for p in c["phases"]], ["warm-up", "steady", "cool-down"])

    def test_references(self):
        r = self.svc.references(self.svc.best_version())
        self.assertTrue(all(math.isfinite(x) for x in (r["uniform_bpc"], r["unigram_bpc"], r["best_so_far_bpc"])))
        self.assertNotIn("noise_band", r)
        self.assertGreater(r["uniform_bpc"], r["unigram_bpc"])

    def test_chat_follows_a_version_only_when_the_paired_test_says_better(self):
        import studio_eval
        store, cards = self.svc.store, [dict(v.scorecard) for v in self.svc.versions()]
        self.addCleanup(lambda: [store._update(n, scorecard=c) for n, c in enumerate(cards, 1)])
        self.addCleanup(self.svc._cache.clear)   # the mocked verdicts must not outlive the test
        store._update(1, scorecard={"bpc": 3., "bpc_method": studio_eval.SCORE_METHOD})
        store._update(2, scorecard={"bpc": 2., "bpc_method": studio_eval.SCORE_METHOD})   # lower, whatever the test says
        for verdict, chat in ((False, 1), (True, 2)):
            with mock.patch("studio_eval.paired_gain", return_value={"diff": -.1, "upper": -.05 if verdict else .01, "better": verdict}):
                self.svc._cache.clear()
                self.assertEqual(self.svc.best_version().number, chat)
                self.assertEqual(self.svc.gate_result(2)["better"], verdict)
                self.assertEqual(self.svc.gate_result(2)["vs"], 1)
        self.assertIsNone(self.svc.gate_result(1))   # the first version has nothing before it
        self.assertGreater(self.svc.checkpoint_step(1), 0)

    def test_the_real_paired_test_reads_both_versions_on_the_same_documents(self):
        store, cards = self.svc.store, [dict(v.scorecard) for v in self.svc.versions()]
        self.addCleanup(lambda: [store._update(n, scorecard=c) for n, c in enumerate(cards, 1)])
        for n in (1, 2):
            store._update(n, scorecard={"bpc": 3.})   # an old-method number: weighing scores both checkpoints again
        verdict = self.svc.gate_result(2)   # the fixture holds back 5 documents: too few to judge on
        self.assertEqual(set(verdict), {"diff", "upper", "better", "docs", "reason", "vs"})
        self.assertEqual((verdict["better"], verdict["reason"], verdict["docs"], verdict["vs"]), (False, "few_docs", 5, 1))
        self.assertTrue(math.isfinite(verdict["diff"]))

    def test_data(self):
        self.assertEqual(self.svc.data("problems").count, 10)

    def test_export_version_writes_the_best_checkpoint_in_both_formats_and_matches_the_engine(self):
        import studio_export
        from tests.test_export import engine_logits, gpt2_reference
        version = self.svc.best_version()
        with tempfile.TemporaryDirectory() as out:
            report = self.svc.export_version(version.number, "hf", str(Path(out) / "model"))
            self.assertTrue((Path(out) / "model" / "model.safetensors").exists())
            self.assertIn("config.json", report["files"])
            self.svc.export_version(version.number, "gguf", str(Path(out) / "model.gguf"))
            self.assertEqual((Path(out) / "model.gguf").read_bytes()[:4], b"GGUF")
            # the app's default design has no projection biases; it still exports as GPT-2 and computes the same function
            tensors, _meta = studio_export.read_safetensors(Path(out) / "model" / "model.safetensors")
            ids = [1, 3, 10, 40, 77, 120, 200, 9, 262, 4]
            spec = studio_export.read_checkpoint(version.best_checkpoint)[0]
            expected = engine_logits(version.best_checkpoint, ids)
            got = gpt2_reference(tensors, ids, spec["dims"]["n_layer"], spec["block"]["mixer"]["config"]["heads"])
            self.assertLess(abs(got - expected).max(), 2e-4)
            with self.assertRaises(ValueError):
                self.svc.export_version(version.number, "onnx", str(Path(out) / "x"))

    def test_chat(self):
        start = time.time()
        text = self.svc.chat(None, [{"role": "user", "content": "Hi"}], lambda _t: None, threading.Event())
        self.assertIsInstance(text, str)
        self.assertLess(time.time() - start, 10)

    def test_queue_events(self):
        from studio_ia.services import Job
        seen = []
        unsub = self.svc.subscribe(lambda k, p: seen.append(k))
        self.svc.pause()  # keep the runner from picking the job up
        job = self.svc.enqueue(Job("x", "quicktest", "t"))
        self.assertEqual([j.id for j in self.svc.queue().up_next], [job.id])
        self.svc.remove(job.id)
        self.assertEqual(self.svc.queue().up_next, [])
        self.assertIn("queue", seen)
        unsub()

    def test_viewer_snapshots_are_the_current_versions_training_snapshots(self):
        version = self.svc.current_version()
        made = []
        for run in version.runs:
            viz = Path(self.svc._abs(run)) / "viz"
            viz.mkdir(parents=True, exist_ok=True)
            for step in (20, 10):
                made.append(viz / f"step_{step:08d}.npz")
                made[-1].write_bytes(b"")
        self.addCleanup(lambda: [path.unlink() for path in made])
        paths = self.svc.viewer_snapshots()
        self.assertEqual(sorted(paths), sorted(str(path) for path in made))
        self.assertTrue(paths[0].endswith("step_00000010.npz"))  # each run's snapshots in step order

    def test_viewer_probe_step_saves_one_real_snapshot(self):
        from studio_snapshot import load
        path = self.svc.viewer_probe_step("The cat sat on the mat because it was tired.")
        self.assertTrue(Path(path).exists())
        self.assertTrue(path.startswith(str(self.svc.workspace.run_dir)))
        self.assertNotIn(path, self.svc.viewer_snapshots())  # a probe is not a training snapshot
        self.assertTrue(any(key.startswith("weight::") for key in load(path)["arrays"]))
        with self.assertRaises(ValueError):
            self.svc.viewer_probe_step("a")

    def test_inspect(self):
        out = self.svc.inspect_forward(None, "hello")
        self.assertEqual(out["logits"].ndim, 3)
        self.assertTrue(self.svc.inspect_logit_lens(None, "hello"))


class WordListsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.svc = FixtureServices(self.tmp.name)
        self.addCleanup(self.svc.close)
        gpt2 = mock.patch.object(WorkspaceServices, "_gpt2", return_value=None)  # nothing is downloaded in a test
        gpt2.start()
        self.addCleanup(gpt2.stop)

    def test_fitted_on_the_stored_text_and_kept_until_it_changes(self):
        from studio_modules.tokenizer.word import WordTokenizer
        empty = self.svc.word_lists()
        self.assertEqual(empty["tokenizers"][1:], (None, None, None))
        self.assertIs(self.svc.word_lists(), empty)
        self.assertTrue(self.svc.add_data("text", {"text": "A kingfisher flashed past the old mill at dawn."}))
        lists = self.svc.word_lists()
        self.assertIsInstance(lists["tokenizers"][3], WordTokenizer)
        self.assertIn(" kingfisher", lists["tokenizers"][3].lookup)
        self.assertEqual(lists["sentence"], "A kingfisher flashed past the old mill at dawn.")


class BandTest(unittest.TestCase):
    """Scoring the held-back text stays off the Tk thread: ready() says no for it, it is kept on disk, and one scored version needs none."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.svc = FixtureServices(self.tmp.name)
        self.addCleanup(self.svc.close)
        self.scores = Path(self.tmp.name) / "data" / "held_back_scores"
        self.svc.store._update(1, scorecard={"bpc": 3.})
        self.svc.store._update(2, scorecard={"bpc": 2.})

    def test_one_scored_version_reads_no_held_back_text(self):
        self.svc.store._update(2, scorecard={"bpc": None})
        with mock.patch.object(self.svc, "_heldout", side_effect=AssertionError("parsed")):
            self.assertEqual(self.svc.best_version().number, 1)
        self.assertFalse(self.scores.exists())

    def test_with_enough_held_back_documents_the_paired_test_judges(self):
        import json

        import studio_eval
        path = Path(self.tmp.name) / "data" / "heldout.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(rows * 6), encoding="utf-8")   # 30 documents
        verdict = self.svc.gate_result(2)
        self.assertEqual(verdict["docs"], 30)
        self.assertNotIn("reason", verdict)
        self.assertTrue(math.isfinite(verdict["upper"]))
        self.assertEqual(verdict["better"], verdict["upper"] < 0)
        self.assertGreaterEqual(verdict["docs"], studio_eval.MIN_DOCS)

    def test_missing_scores_go_to_the_worker_and_are_kept_on_disk(self):
        self.assertFalse(self.svc.ready(self.svc.best_version))  # two scored versions, nothing scored yet
        self.svc.best_version()
        self.assertEqual(len(list(self.scores.glob("*.npy"))), 2)  # one file per checkpoint
        self.assertTrue(self.svc.ready(self.svc.best_version))
        again = WorkspaceServices(self.svc.workspace)  # a new launch: the scores come from disk, the pair is only compared
        self.addCleanup(again.close)
        with mock.patch("studio_eval.score_docs", side_effect=AssertionError("scored again")):
            again.best_version()
            self.assertEqual(again.best_version().number, self.svc.best_version().number)
            self.assertEqual(again.gate_result(2)["diff"], self.svc.gate_result(2)["diff"])

    def test_a_changed_checkpoint_or_held_back_file_is_scored_again(self):
        self.svc.best_version()
        ckpt = self.svc.versions()[0].best_checkpoint
        first = self.svc._held_scores(ckpt).copy()
        heldout = Path(self.tmp.name) / "data" / "heldout.json"
        heldout.write_text(heldout.read_text(encoding="utf-8") + " ", encoding="utf-8")
        self.assertFalse(self.svc.ready(lambda: self.svc._held_scores(ckpt)))
        self.assertTrue((self.svc._held_scores(ckpt) == first).all())  # same text, same scores
        self.assertEqual(len(list(self.scores.glob("*.npy"))), 2)  # the stale file of that checkpoint is gone, not piled up

    def test_scores_made_by_another_scoring_method_are_not_reused_and_the_scorecard_is_refreshed(self):
        import studio_eval
        self.svc.best_version()
        ckpt = self.svc.versions()[0].best_checkpoint
        before = len(list(self.scores.glob("*.npy")))
        with mock.patch.object(studio_eval, "SCORE_METHOD", "old-method"):
            self.assertFalse(self.svc.ready(lambda: self.svc._held_scores(ckpt)))  # the files on disk belong to the current method
        card = self.svc.store.get(1)["scorecard"]
        self.assertEqual(card["bpc_method"], studio_eval.SCORE_METHOD)  # the 3.0 above was replaced by a fresh measurement
        self.assertNotEqual(card["bpc"], 3.)
        self.assertEqual(len(list(self.scores.glob("*.npy"))), before)

    def test_every_scored_version_is_measured_by_the_current_method_even_when_the_pair_cannot_be_weighed(self):
        import studio_eval
        store, cards = self.svc.store, [dict(v.scorecard) for v in self.svc.versions()]
        self.addCleanup(lambda: [store._update(n, scorecard=c) for n, c in enumerate(cards, 1)])
        self.addCleanup(self.svc._cache.clear)
        for n in (1, 2):
            store._update(n, scorecard={"bpc": 9., "bpc_method": "old-method"})
        self.svc._cache.clear()
        with mock.patch.object(self.svc, "_gate", return_value=None):  # the paired test gave no answer for this pair
            self.svc.best_version()
        for n in (1, 2):
            card = store.get(n)["scorecard"]
            self.assertEqual(card["bpc_method"], studio_eval.SCORE_METHOD)
            self.assertNotEqual(card["bpc"], 9.)

    def test_a_cache_hit_does_not_wait_for_a_running_read(self):
        started, release = threading.Event(), threading.Event()
        self.svc._cached("fast", lambda: 1)
        slow = threading.Thread(target=lambda: self.svc._cached("slow", lambda: (started.set(), release.wait(5))))
        slow.start()
        self.addCleanup(slow.join)
        self.addCleanup(release.set)
        self.assertTrue(started.wait(5))
        got = []
        reader = threading.Thread(target=lambda: got.append(self.svc._cached("fast", lambda: 2)))
        reader.start()
        reader.join(2)
        self.assertEqual(got, [1])


if __name__ == "__main__":
    unittest.main()
