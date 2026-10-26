"""Issue #735: chat and score the newest checkpoint of a run that is still training, through a copy, without touching the version."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from studio_core import Workspace
from studio_ia.services import FakeServices, WorkspaceServices
from studio_queue import Job
from studio_versions import NoCheckpoint


def save_checkpoint(path, step):
    np.savez_compressed(path, metadata=np.array(json.dumps({"step": step})), weight=np.zeros(8))


class LiveCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.svc = WorkspaceServices(Workspace(Path(self.tmp.name) / "ws"))
        self.addCleanup(self.svc.close)
        self.run = Path(self.tmp.name) / "run"
        self.run.mkdir()
        row = self.svc.store.save_version({"dims": {"n_ctx": 64}}, "first", None)
        self.number = row["number"]
        self.svc.store.set_status(self.number, "queued")
        self.svc.store.set_status(self.number, "training")
        self.job = Job("pretrain", params={"version": self.number, "output_dir": str(self.run)})
        self.job.status = "running"
        self.svc.runner.jobs.append(self.job)
        self.addCleanup(lambda: self.svc.runner.jobs.remove(self.job) if self.job in self.svc.runner.jobs else None)

    def version(self):
        return self.svc._as_version(self.number)

    def test_before_the_first_save_chat_says_so_instead_of_failing(self):
        self.assertEqual(self.svc.live_checkpoints(), [])
        with self.assertRaises(NoCheckpoint) as caught:
            self.svc.chat(self.number, [], lambda text: None, None)
        self.assertIn("has not saved a checkpoint yet", str(caught.exception))

    def test_chat_answers_from_a_copy_of_the_newest_save(self):
        save_checkpoint(self.run / "last.npz", 40)
        save_checkpoint(self.run / "best.npz", 30)
        [live] = self.svc.live_checkpoints()
        self.assertEqual((live["version"], live["step"], live["file"]), (self.number, 30, "best.npz"))  # best, else last
        loaded = []

        class Model:
            device, model = "cpu", mock.Mock(p={})

            def __init__(self, path, device, data=None):
                loaded.append(Path(path))

            def generate(self, *args, **kwargs):
                return "hi"
        with mock.patch("studio_model.LocalModel", Model), mock.patch("studio_model.load_checkpoint", return_value={}), \
                mock.patch("studio_memory.weight_bytes", return_value=1):
            self.assertEqual(self.svc.chat(self.number, [], None, None), "hi")
        self.assertEqual(loaded[0].parent, self.svc.root / "data" / "live_checkpoints")  # a copy, never the run's own file
        self.assertNotEqual(loaded[0], self.run / "best.npz")
        self.assertIsNone(self.version().best_checkpoint)
        self.assertEqual(self.version().status, "training")

    def test_a_file_caught_half_written_is_refused_with_a_sentence(self):
        save_checkpoint(self.run / "best.npz", 30)
        whole = (self.run / "best.npz").read_bytes()
        (self.run / "best.npz").write_bytes(whole[:len(whole) // 2])
        with self.assertRaises(RuntimeError) as caught:
            self.svc.chat(self.number, [], None, None)
        self.assertIn("Try again in a moment", str(caught.exception))
        self.assertEqual(list((self.svc.root / "data" / "live_checkpoints").glob("*")), [])  # no broken copy is left to be loaded
        self.assertEqual(self.svc.live_checkpoints(), [])  # and it is not offered while unreadable

    def test_a_newer_save_replaces_the_copy(self):
        save_checkpoint(self.run / "last.npz", 10)
        first = self.svc._live_snapshot(self.number)
        save_checkpoint(self.run / "last.npz", 20)
        second = self.svc._live_snapshot(self.number)
        self.assertNotEqual(first, second)
        self.assertEqual(self.svc._live_meta(second)["step"], 20)
        self.assertFalse(first.exists())

    def test_score_now_returns_bits_per_character_and_stores_nothing(self):
        save_checkpoint(self.run / "last.npz", 12)
        before = self.svc.store.get(self.number)
        with mock.patch.object(self.svc, "_heldout", return_value=["abc", "def"]), \
                mock.patch("studio_eval.score_docs", return_value=[(2.0, 4.0), (1.0, 2.0)]) as scored:
            result = self.svc.score_live(self.number)
        self.assertEqual((result["version"], result["step"], result["documents"]), (self.number, 12, 2))
        self.assertAlmostEqual(result["bpc"], 3 / 6 / np.log(2))
        self.assertNotEqual(Path(scored.call_args.args[0]), self.run / "last.npz")
        self.assertEqual(self.svc.store.get(self.number), before)  # best_checkpoint, scorecard and status unchanged
        self.assertFalse((self.svc.root / "data" / "held_back_scores").exists())

    def test_a_stop_ends_score_now_early_without_a_result(self):
        import studio_eval
        save_checkpoint(self.run / "last.npz", 12)
        stop = threading.Event()
        stop.set()
        with mock.patch.object(self.svc, "_heldout", return_value=["abc"]),                 mock.patch("studio_eval.score_docs", side_effect=studio_eval.Stopped("stopped")) as scored:
            self.assertIsNone(self.svc.score_live(self.number, stop=stop))
        self.assertIs(scored.call_args.kwargs["stop"], stop)

    def test_score_now_without_held_back_text_says_so(self):
        save_checkpoint(self.run / "last.npz", 12)
        with mock.patch.object(self.svc, "_heldout", return_value=[]), self.assertRaises(ValueError):
            self.svc.score_live(self.number)

    def test_a_version_with_final_weights_is_not_live(self):
        save_checkpoint(self.run / "last.npz", 12)
        self.svc.store.set_best_checkpoint(self.number, self.run / "last.npz")
        self.assertEqual(self.svc.live_checkpoints(), [])
        self.assertEqual(self.svc._checkpoint(self.number, live=True), self.version().best_checkpoint)

    def chat_once(self, loaded):
        class Model:
            device, model = "cpu", mock.Mock(p={})

            def __init__(self, path, device, data=None):
                loaded.append(str(path))

            def generate(self, *args, **kwargs):
                return "hi"
        with mock.patch("studio_model.LocalModel", Model), mock.patch("studio_model.load_checkpoint", return_value={}),                 mock.patch("studio_memory.weight_bytes", return_value=1):
            self.svc.chat(self.number, [], None, None)

    def test_the_cache_keeps_one_live_model_per_version(self):
        loaded = []
        save_checkpoint(self.run / "last.npz", 10)
        self.chat_once(loaded)
        save_checkpoint(self.run / "last.npz", 20)
        os_utime = (self.run / "last.npz").stat().st_mtime_ns + 1_000_000_000
        import os
        os.utime(self.run / "last.npz", ns=(os_utime, os_utime))  # a distinct mtime, so a distinct snapshot
        self.chat_once(loaded)
        self.assertEqual(len(loaded), 2)
        self.assertNotEqual(loaded[0], loaded[1])
        self.assertEqual(list(self.svc._models), [loaded[1]])  # the older snapshot's model is gone
        self.assertEqual(list(self.svc._using), [loaded[1]])

    def test_the_end_of_the_run_drops_the_cached_model_too(self):
        loaded = []
        save_checkpoint(self.run / "last.npz", 10)
        self.chat_once(loaded)
        self.assertEqual(len(self.svc._models), 1)
        self.svc._drop_live(self.number)
        self.assertEqual(self.svc._models, {})
        self.assertEqual(self.svc._using, {})

    def test_the_end_of_the_run_drops_the_copy(self):
        save_checkpoint(self.run / "last.npz", 12)
        self.svc._live_snapshot(self.number)
        self.svc._drop_live(self.number)
        self.assertEqual(list((self.svc.root / "data" / "live_checkpoints").glob("*.npz")), [])


class FakeLive(unittest.TestCase):
    def test_fake_services_serve_the_same_shape(self):
        fake = FakeServices()
        self.assertEqual(fake.live_checkpoints(), [])
        fake.live = [{"version": 1, "step": 5, "total": 50, "saved_at": 1.0, "file": "best.npz"}]
        fake.heldout_docs = ["a"]
        self.assertEqual(fake.score_live(1)["step"], 5)
        with self.assertRaises(NoCheckpoint):
            fake.score_live(2)


if __name__ == "__main__":
    unittest.main()
