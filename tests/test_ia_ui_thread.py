"""Slow work stays off the Tk thread: the @slow tripwire, the stall log, and every read the shell makes there, with cold caches (no display)."""
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import studio_backend
import studio_jobs
import studio_ui_guard as guard
from studio_ia.services import FixtureServices, WorkspaceServices
from studio_ui_guard import OnUIThread, StallWatch, slow


class Bound(unittest.TestCase):
    """This test thread stands in for the Tk thread, with the tripwire raising."""

    def setUp(self):
        patch = mock.patch.dict(os.environ, {guard.STRICT: "1"})
        patch.start()
        self.addCleanup(patch.stop)
        guard.bind()
        self.addCleanup(guard.unbind)
        guard._seen.clear()


class Tripwire(Bound):
    def test_a_slow_primitive_raises_on_the_ui_thread_under_strict(self):
        @slow("weigh a version")
        def weigh(x):
            return x

        with self.assertRaisesRegex(OnUIThread, "weigh a version ran on the UI thread"):
            weigh(1)

    def test_it_runs_anywhere_else(self):
        @slow("weigh a version")
        def weigh(x):
            return x

        out = []
        worker = threading.Thread(target=lambda: out.append(weigh(2)))
        worker.start()
        worker.join()
        guard.unbind()
        self.assertEqual((out, weigh(3)), ([2], 3))  # a worker, and a thread nobody bound

    def test_outside_strict_it_logs_the_stack_once_per_call_site(self):
        @slow("weigh a version")
        def weigh():
            return "done"

        with mock.patch.dict(os.environ, {guard.STRICT: "0"}), self.assertLogs("groundwork.ui", "WARNING") as logged:
            self.assertEqual(weigh(), "done")  # logged, not raised
        self.assertIn("weigh a version ran on the UI thread", logged.output[0])
        self.assertIn("test_ia_ui_thread.py", logged.output[0])

    def test_one_call_site_is_logged_once(self):
        @slow("weigh a version")
        def weigh():
            return None

        with mock.patch.dict(os.environ, {guard.STRICT: "0"}), self.assertLogs("groundwork.ui", "WARNING") as logged:
            for _ in range(4):
                weigh()
        self.assertEqual(len(logged.records), 1)

    def test_when_says_which_calls_are_long(self):
        @slow("scan a store", when=lambda size: size > 10)
        def scan(size):
            return size

        self.assertEqual(scan(3), 3)
        with self.assertRaises(OnUIThread):
            scan(11)

    def test_a_generator_is_checked_by_the_thread_that_reads_it(self):
        @slow("read rows")
        def rows():
            yield 1

        made = rows()  # made here, read on a worker: fine
        out = []
        worker = threading.Thread(target=lambda: out.extend(made))
        worker.start()
        worker.join()
        self.assertEqual(out, [1])
        with self.assertRaises(OnUIThread):
            list(rows())

    def test_closing_unbinds_only_its_own_thread(self):
        guard.unbind(guard.ui_thread() + 1)
        self.assertTrue(guard.on_ui_thread())
        guard.unbind(guard.ui_thread())
        self.assertFalse(guard.on_ui_thread())

    def test_the_known_slow_primitives_are_marked(self):
        import studio_archdb
        import studio_compile
        import studio_eval
        import studio_model
        for fn in (studio_eval.score_docs, studio_model.load_checkpoint, studio_model.train, studio_compile.CompiledModel.forward,
                   studio_archdb._hash_file):
            with self.subTest(fn=fn.__qualname__), self.assertRaises(OnUIThread):
                fn()  # no arguments: the tripwire comes before anything is read


class StallLog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "logs" / "ui-stalls.log"
        self.now = 0.0
        self.watch = StallWatch(self.path, threading.get_ident(), clock=lambda: self.now)

    def text(self):
        return self.path.read_text(encoding="utf-8") if self.path.exists() else ""

    def test_a_loop_that_keeps_beating_writes_nothing(self):
        for _ in range(20):
            self.now += .1
            self.watch.beat()
            self.assertFalse(self.watch.check())
        self.assertEqual(self.text(), "")

    def test_a_late_beat_writes_the_stack_once_then_how_long_it_lasted(self):
        self.now = .25
        self.assertFalse(self.watch.check())  # under the limit
        self.now = .4
        self.assertTrue(self.watch.check())
        self.assertIn("has not answered for 400 ms", self.text())
        self.assertIn("test_a_late_beat_writes_the_stack", self.text())  # this thread's frames stand in for the Tk thread's
        for t in (.5, 1.0, 5.0):
            self.now = t
            self.assertFalse(self.watch.check())  # still the same stall
        self.assertEqual(self.text().count("has not answered"), 1)
        self.now = 6.0
        self.watch.beat()
        self.assertFalse(self.watch.check())
        self.assertIn("the stall lasted about 6000 ms", self.text())
        self.now = 6.1
        self.assertFalse(self.watch.check())
        self.now = 7.0  # a second stall is written afresh
        self.assertTrue(self.watch.check())
        self.assertEqual(self.text().count("has not answered"), 2)

    def test_the_collector_is_held_before_the_stack_is_taken(self):
        held = []
        self.watch.hold = lambda: held.append(1)
        self.now = 1.0
        self.watch.check()
        self.watch.check()
        self.assertEqual(held, [1])

    def test_a_gone_thread_and_an_unwritable_log_do_not_stop_the_watch(self):
        watch = StallWatch(Path(self.tmp.name) / "x" / "y.log", -1, clock=lambda: self.now, frames=lambda: {})
        (Path(self.tmp.name) / "x").write_text("a file where the folder should be")
        self.now = 1.0
        self.assertTrue(watch.check())

    def test_the_shell_runs_it_unless_tests_turn_it_off(self):
        with mock.patch.dict(os.environ, {guard.STALLS: "0"}):
            self.assertFalse(guard.stalls_enabled())
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(guard.STALLS, None)
            self.assertTrue(guard.stalls_enabled())


class ThreadBudget(unittest.TestCase):
    def test_background_work_leaves_a_core_for_the_window(self):
        self.assertEqual([studio_backend.background_threads(n) for n in (1, 2, 8)], [1, 1, 7])

    def test_cap_threads_sets_torch_once_for_the_process(self):
        with mock.patch("torch.set_num_threads") as cap, mock.patch("os.cpu_count", return_value=8):
            studio_backend.cap_threads()
        cap.assert_called_once_with(7)

    def test_every_job_handler_caps_before_it_starts(self):
        calls = []
        with mock.patch.object(studio_backend, "cap_threads", side_effect=lambda: calls.append("cap")):
            handler = studio_jobs._capped(lambda *a, **k: calls.append("run") or "out")
            self.assertEqual((handler(1, 2), calls), ("out", ["cap", "run"]))
        self.assertTrue(all(getattr(h, "__wrapped__", None) for h in studio_jobs.HANDLERS.values()))


class TkThreadReads(Bound):
    """The reads the shell makes on its thread, over a copy of the fixture workspace with every cache cold."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.source = Path(cls.tmp.name) / "source"
        FixtureServices(cls.source).close()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        super().setUp()
        self.root = Path(self.tmp.name) / self.id().rsplit(".", 1)[-1]
        shutil.copytree(self.source, self.root)
        for cold in ("data/held_back_scores", "data/summaries.sqlite", "data/summaries.sqlite-wal", "data/summaries.sqlite-shm"):
            target = self.root / cold
            shutil.rmtree(target, ignore_errors=True) if target.is_dir() else target.unlink(missing_ok=True)
        self.svc = FixtureServices(self.root)
        self.addCleanup(self.svc.close)
        self.svc.store._update(1, scorecard={"bpc": 3.})  # two scored versions: the paired gate has to weigh them against each other
        self.svc.store._update(2, scorecard={"bpc": 2.})
        self.emitted = []
        self.done = threading.Event()

        def emit(kind, payload=None):
            self.emitted.append(kind)
            if kind == "versions":
                self.done.set()
        patch = mock.patch.object(self.svc, "emit", side_effect=emit)
        patch.start()
        self.addCleanup(patch.stop)
        self.svc.bind_ui(mock.Mock())  # this thread is now the Tk thread
        self.addCleanup(lambda: guard.unbind(self.svc._ui_thread))
        self.assertTrue(guard.on_ui_thread())

    def shell_read(self, work):
        """What Shell.slow does: in place when ready() says it is quick, otherwise on a worker."""
        if self.svc.ready(work):
            return work()
        out, err = [], []

        def run():
            try:
                out.append(work())
            except BaseException as e:  # noqa: BLE001 - handed back to the test thread
                err.append(e)
        worker = threading.Thread(target=run)
        worker.start()
        worker.join(60)
        if err:
            raise err[0]
        return out[0]

    def test_the_strip_and_the_launch_reads_are_quick_with_nothing_cached(self):
        svc = self.svc
        self.assertEqual([v.number for v in svc.versions()], [1, 2])
        self.assertIsNotNone(svc.best_version())  # the pair is not scored: the stored numbers decide, a worker scores them
        self.assertIsNotNone(svc.current_version())
        svc.queue()
        svc.queue_outlook()
        svc.next_step_if_ready()
        svc.checkpoint_step(svc.current_version())
        svc.run_facts(2)
        self.assertTrue(self.done.wait(60))
        self.assertIn("versions", self.emitted)
        self.assertIsNotNone(svc.gate_result(2))  # what the event reads: in place, scored by the worker

    def test_the_queue_page_reads_a_runs_phases_without_loading_its_tokenizer(self):
        run = Path(self.svc.versions()[-1].best_checkpoint).parent
        phases = self.svc.run_phases(run)
        self.assertEqual([name for _, _, name in phases], ["warm-up", "steady", "cool-down"])
        with self.assertRaises(OnUIThread):
            self.svc.run_curve(run)  # what the page asked before: bits per character need the tokenizer and the held-back text
        self.assertEqual(phases, self.shell_read(lambda: self.svc.run_curve(run))["phases"])

    def test_a_cold_run_page_read_goes_to_the_worker_not_the_tk_thread(self):
        v = self.svc.current_version()
        run = Path(v.best_checkpoint).parent
        for read in (lambda: self.svc.references(v), lambda: self.svc.run_curve(run), lambda: self.svc.gate_result(v.number)):
            with self.subTest():
                self.shell_read(read)

    def test_every_pages_preload_follows_the_same_rule(self):
        from studio_ia.pages.passive import PassivePage
        from studio_ia.pages.post_training import PostTrainingPage
        from studio_ia.pages.pretraining import PretrainingPage
        from studio_ia.pages.your_data import YourDataPage
        for cls in (PassivePage, PostTrainingPage, PretrainingPage, YourDataPage):
            with self.subTest(page=cls.__name__):
                page = object.__new__(cls)
                page.services, page.summaries = self.svc, {}
                self.shell_read(page.preload)
                self.shell_read(self.svc.passive)

    def test_the_data_reads_of_a_large_store_wait_for_a_worker(self):
        ws = self.svc.workspace
        for dataset in ws.state["datasets"]:
            dataset["recordCount"] = 10 ** 6
        self.assertFalse(self.svc.ready(lambda: self.svc.data("text")))
        with self.assertRaises(OnUIThread):
            self.svc.data("text")  # asked in place anyway: the tripwire names it
        self.shell_read(lambda: self.svc.data("text"))


class WorkspaceServicesBind(unittest.TestCase):
    def test_closing_the_services_releases_the_ui_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = FixtureServices(tmp)
            svc.bind_ui(mock.Mock())
            self.assertTrue(guard.on_ui_thread())
            svc.close()
            self.assertFalse(guard.on_ui_thread())
        self.assertIsInstance(svc, WorkspaceServices)


if __name__ == "__main__":
    unittest.main()
