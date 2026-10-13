"""WorkspaceServices over the fixture workspace (no display)."""
import math
import tempfile
import threading
import time
import unittest

from studio_ia.services import FixtureServices


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
        lo, hi = r["noise_band"]
        self.assertTrue(all(math.isfinite(x) for x in (r["uniform_bpc"], r["unigram_bpc"], r["best_so_far_bpc"], lo, hi)))
        self.assertGreater(r["uniform_bpc"], r["unigram_bpc"])

    def test_chat_follows_a_version_only_outside_the_noise_band(self):
        store, cards = self.svc.store, [dict(v.scorecard) for v in self.svc.versions()]
        self.addCleanup(lambda: [store._update(n, scorecard=c) for n, c in enumerate(cards, 1)])
        lo, _hi = self.svc.references(1)["noise_band"]
        store._update(1, scorecard={"bpc": lo + 1e-6})
        store._update(2, scorecard={"bpc": lo + 1e-9})   # lower, but inside version 1's band
        self.assertEqual(self.svc.best_version().number, 1)
        store._update(2, scorecard={"bpc": lo - 1e-6})   # below the band
        self.assertEqual(self.svc.best_version().number, 2)
        self.assertGreater(self.svc.checkpoint_step(1), 0)

    def test_data(self):
        self.assertEqual(self.svc.data("problems").count, 10)

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

    def test_inspect(self):
        out = self.svc.inspect_forward(None, "hello")
        self.assertEqual(out["logits"].ndim, 3)
        self.assertTrue(self.svc.inspect_logit_lens(None, "hello"))


if __name__ == "__main__":
    unittest.main()
