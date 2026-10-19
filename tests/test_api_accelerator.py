"""studio_api.accelerator and the uvicorn arguments of python -m studio_api: no GPU probe runs, no server starts."""
import unittest
from types import SimpleNamespace
from unittest import mock

from studio_api import __main__ as entry
from studio_api import accelerator

CUDA = {"available": True, "kind": "cuda", "name": "RTX", "memory_bytes": 1, "reason": ""}
MPS = {"available": True, "kind": "mps", "name": "Apple GPU (Metal)", "memory_bytes": 1, "reason": ""}
NONE = {"available": False, "kind": "", "name": "", "memory_bytes": 0, "reason": "RuntimeError: No CUDA or Metal GPU was found."}


class MachineTest(unittest.TestCase):
    def probe(self, system="Linux", machine="x86_64", smi="/usr/bin/nvidia-smi", code=0, out=b"GPU 0: RTX (UUID: x)\n"):
        run = mock.Mock(return_value=SimpleNamespace(returncode=code, stdout=out))
        found = accelerator.machine_accelerator(lambda: system, lambda: machine, lambda name: smi, run)
        return found, run

    def test_apple_silicon_has_mps(self):
        self.assertEqual(self.probe("Darwin", "arm64", smi=None)[0], "mps")

    def test_an_intel_mac_has_none(self):
        self.assertEqual(self.probe("Darwin", "x86_64", smi=None)[0], "")

    def test_nvidia_smi_listing_a_gpu_is_cuda(self):
        found, run = self.probe()
        self.assertEqual(found, "cuda")
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["/usr/bin/nvidia-smi", "-L"])

    def test_no_smi_or_no_gpu_listed_is_none(self):
        self.assertEqual(self.probe(smi=None)[0], "")
        self.assertEqual(self.probe(code=9)[0], "")
        self.assertEqual(self.probe(out=b"No devices were found\n")[0], "")

    def test_an_smi_that_cannot_run_is_none(self):
        run = mock.Mock(side_effect=OSError)
        self.assertEqual(accelerator.machine_accelerator(lambda: "Linux", lambda: "x86_64", lambda n: "/x", run), "")


class CheckTest(unittest.TestCase):
    def test_a_gpu_that_torch_does_not_use_is_refused(self):
        for has in ("mps", "cuda"):
            with self.assertRaises(SystemExit) as caught:
                accelerator.check(probe=lambda: has, status=lambda: NONE)
            message = str(caught.exception)
            self.assertIn(has, message)
            self.assertIn("--allow-cpu", message)
            self.assertIn("No CUDA or Metal GPU was found", message)

    def test_allow_cpu_runs_on_the_cpu(self):
        line = accelerator.check(allow_cpu=True, probe=lambda: "mps", status=lambda: NONE)
        self.assertEqual(line, "device: cpu (--allow-cpu)")

    def test_a_cpu_only_machine_runs_on_the_cpu(self):
        self.assertEqual(accelerator.check(probe=lambda: "", status=lambda: NONE), "device: cpu")

    def test_the_gpu_in_use_is_logged_and_the_machine_not_asked(self):
        probe = mock.Mock()
        self.assertEqual(accelerator.check(probe=probe, status=lambda: CUDA), "device: cuda (RTX)")
        self.assertEqual(accelerator.check(probe=probe, status=lambda: MPS), "device: mps (Apple GPU (Metal))")
        probe.assert_not_called()


class StartupTest(unittest.TestCase):
    def test_start_refuses_before_serving(self):
        with mock.patch.object(accelerator, "machine_accelerator", return_value="mps"), \
                mock.patch("studio_backend.gpu_status", return_value=NONE), mock.patch("uvicorn.run") as run:
            with self.assertRaises(SystemExit):
                entry.main([])
            with self.assertRaises(SystemExit):
                entry.main(["--reload"])
        run.assert_not_called()

    def test_build_refuses_too(self):
        with mock.patch.object(accelerator, "machine_accelerator", return_value="cuda"), \
                mock.patch("studio_backend.gpu_status", return_value=NONE):
            with self.assertRaises(SystemExit):
                entry.make([])
            entry.make(["--fake"])[0].state.broadcaster.close()  # --fake is never checked

    def test_fake_and_allow_cpu_skip_the_check(self):
        with mock.patch.object(accelerator, "check") as check, mock.patch("uvicorn.run"):
            entry.main(["--fake"])
            check.assert_not_called()
            with mock.patch.object(entry, "build"), mock.patch.dict("os.environ"):
                entry.main(["--allow-cpu"])
            check.assert_called_with(True)


class GracefulShutdownTest(unittest.TestCase):
    """An open /events stream never ends, so --reload would wait for it forever (#322): uvicorn must cut it after a moment."""

    def test_both_runs_cut_open_streams_after_a_short_wait(self):
        self.assertTrue(0 < entry.GRACEFUL_SHUTDOWN <= 5)
        for argv in (["--fake"], ["--fake", "--reload"]):
            with mock.patch("uvicorn.run") as run, mock.patch.object(entry, "build"), mock.patch.dict("os.environ"):
                entry.main(argv)
            self.assertEqual(run.call_args.kwargs["timeout_graceful_shutdown"], entry.GRACEFUL_SHUTDOWN, argv)


if __name__ == "__main__":
    unittest.main()
