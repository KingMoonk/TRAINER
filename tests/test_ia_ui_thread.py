"""Slow work stays off the Tk thread: the @slow tripwire, the stall log, and every read the shell makes there, with cold caches (no display)."""
import os
import tempfile
import threading
import unittest
from unittest import mock

import studio_backend
import studio_jobs
import studio_ui_guard as guard
from studio_ia.services import FixtureServices, WorkspaceServices
from studio_ui_guard import OnUIThread, slow


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


class ThreadBudget(unittest.TestCase):
    def test_background_work_leaves_a_core_for_the_window(self):
        self.assertEqual([studio_backend.background_threads(n) for n in (1, 2, 8)], [1, 1, 7])

    def test_cap_threads_sets_torch_once_for_the_process(self):
        with mock.patch("torch.set_num_threads") as cap, mock.patch("os.cpu_count", return_value=8):
            studio_backend.cap_threads()
        cap.assert_called_once_with(7)

    def test_every_job_runs_in_a_worker_with_a_core_left_for_the_window(self):
        import studio_worker
        for kind, handler in studio_jobs.HANDLERS.items():  # no wrapper: the worker finds each handler by its module and name
            self.assertEqual(studio_worker.handler_ref(handler)["module"], "studio_jobs")
            self.assertIs(getattr(studio_jobs, studio_worker.handler_ref(handler)["name"]), handler)
        with mock.patch("os.cpu_count", return_value=8):
            self.assertEqual(studio_worker.thread_env({})["OMP_NUM_THREADS"], "7")  # the worker starts with torch capped


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
