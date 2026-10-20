"""tasks.py dev: the commands it starts, the device notice and the wait for GET /node. No server is started."""
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

import tasks


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class DevCommandsTest(unittest.TestCase):
    def test_the_api_and_the_page_listen_on_loopback_only(self):
        api, web, env = tasks.dev_commands("pnpm")
        self.assertEqual(api[api.index("--bind") + 1], "127.0.0.1")
        self.assertEqual(web[web.index("--hostname") + 1], "127.0.0.1")
        for command in (api, web):
            self.assertNotIn("0.0.0.0", command)
            self.assertNotIn("::", command)
        self.assertNotIn("--allow-host", api)  # the host's tailscale names its own node
        self.assertEqual(env, {"NEXT_PUBLIC_API_URL": "/api"})

    def test_the_api_reloads_on_the_port_serve_points_at(self):
        api, web, _ = tasks.dev_commands("pnpm", reload=True)
        self.assertIn("--reload", api)
        self.assertEqual(api[api.index("--port") + 1], str(tasks.API_PORT))
        self.assertEqual(web[web.index("--port") + 1], str(tasks.WEB_PORT))
        self.assertEqual(api[:3], [tasks.PY, "-m", "studio_api"])  # the venv that runs tasks.py, so its torch

    def test_windows_does_not_use_uvicorns_reload(self):
        for platform, uvicorn in (("win32", False), ("linux", True), ("darwin", True)):
            with mock.patch.object(tasks.sys, "platform", platform):
                self.assertEqual(tasks.reload_by_uvicorn(), uvicorn)
                self.assertEqual("--reload" in tasks.dev_commands("pnpm")[0], uvicorn)
        self.assertNotIn("--reload", tasks.dev_commands("pnpm", reload=False)[0])

    def test_serve_maps_name_the_loopback_ports(self):
        self.assertEqual(tasks.DEV_SERVE, {"/": f"http://127.0.0.1:{tasks.WEB_PORT}", "/api/": f"http://127.0.0.1:{tasks.API_PORT}"})
        self.assertEqual(tasks.APP_SERVE, {"/": f"http://127.0.0.1:{tasks.API_PORT}"})

    def test_a_listening_port_is_busy(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            self.assertTrue(tasks.port_busy(listener.getsockname()[1]))


# A leader that starts a grandchild ignoring SIGTERM (as the API's reloader worker did) and prints its pid, then exits or sleeps.
TREE = textwrap.dedent("""
    import subprocess, sys, time
    child = subprocess.Popen([sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
    print(child.pid, flush=True)
    if sys.argv[1] == "sleep":
        time.sleep(60)
""")


def pid_gone(pid):
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(.1)
    return False


@unittest.skipIf(sys.platform == "win32", "process groups are POSIX; Windows uses taskkill /T")
class StopTreeTest(unittest.TestCase):
    def start(self, mode):
        proc = tasks.spawn([sys.executable, "-c", TREE, mode], stdout=subprocess.PIPE, text=True)
        self.addCleanup(lambda: [os.killpg(proc.pid, signal.SIGKILL) if not tasks.group_gone(proc.pid) else None, proc.wait()])
        return proc, int(proc.stdout.readline())

    def test_stop_ends_a_grandchild_that_ignores_sigterm(self):
        proc, grandchild = self.start("sleep")
        tasks.stop(proc, grace=.5)
        proc.wait()
        self.assertTrue(pid_gone(grandchild))

    def test_stop_ends_the_tree_when_the_leader_already_exited(self):
        proc, grandchild = self.start("exit")
        proc.wait()
        tasks.stop(proc, grace=.5)
        self.assertTrue(pid_gone(grandchild))


class PortOwnerTest(unittest.TestCase):
    def test_owners_come_from_lsof_with_their_command_lines(self):
        run = mock.Mock(return_value=mock.Mock(stdout="13965\n15327\n13965\n"))
        with mock.patch.object(tasks.sys, "platform", "darwin"), \
                mock.patch("studio_platform.command_line", side_effect=lambda pid: f"python -m studio_api {pid}"):
            self.assertEqual(tasks.port_owners(8787, run=run), [(13965, "python -m studio_api 13965"), (15327, "python -m studio_api 15327")])
        self.assertIn("-iTCP:8787", run.call_args.args[0])

    def test_no_lsof_means_no_owners(self):
        with mock.patch.object(tasks.sys, "platform", "linux"):
            self.assertEqual(tasks.port_owners(8787, run=mock.Mock(side_effect=FileNotFoundError)), [])

    def test_the_message_names_the_pid_and_calls_our_own_api_stale(self):
        text = tasks.port_in_use_message("API", 8787, [(13965, "python -m studio_api --reload")])
        self.assertIn("pid 13965: python -m studio_api --reload", text)
        self.assertIn("stale API", text)

    def test_the_message_for_something_else_or_nothing_known(self):
        other = tasks.port_in_use_message("web", 3000, [(7, "node server.js")])
        self.assertIn("pid 7: node server.js", other)
        self.assertNotIn("stale API", other)
        self.assertEqual(tasks.port_in_use_message("API", 8787, []).split(".")[0], "Port 8787 (API) is in use")


def serve_status(handlers, host="box.tail1234.ts.net:443"):
    return {"TCP": {"443": {"HTTPS": True}}, "Web": {host: {"Handlers": {p: {"Proxy": t} for p, t in handlers.items()}}}}


class FakeTailscale:
    """`tailscale serve`: status --json reports the handlers; the other commands change them as the real one does."""
    def __init__(self, handlers):
        self.handlers, self.changes = dict(handlers), []

    def __call__(self, cmd, **_):
        args = cmd[1:]
        if args == ["serve", "status", "--json"]:
            return mock.Mock(returncode=0, stdout=json.dumps(serve_status(self.handlers)))
        self.changes.append(args)
        path = next((a.split("=", 1)[1] for a in args if a.startswith("--set-path=")), "/")
        if args[-1] == "off":
            self.handlers.pop(path, None)
        else:
            self.handlers[path] = args[-1]
        return mock.Mock(returncode=0, stdout="")


class TailnetServeTest(unittest.TestCase):
    def ensure(self, fake, want, url="https://box.tail1234.ts.net/"):
        said = []
        tasks.ensure_serve(want, "/bin/tailscale", url, run=fake, say=said.append)
        return said

    def test_the_handlers_come_from_the_443_entry(self):
        status = serve_status({"/": "http://127.0.0.1:3000"})
        status["Web"]["box.tail1234.ts.net:8443"] = {"Handlers": {"/x": {"Proxy": "http://127.0.0.1:9"}}}
        self.assertEqual(tasks.serve_handlers(status), {"/": "http://127.0.0.1:3000"})
        self.assertEqual(tasks.serve_handlers({}), {})

    def test_dev_from_the_built_app_moves_the_root_and_adds_the_api(self):
        fake = FakeTailscale(tasks.APP_SERVE)
        said = self.ensure(fake, tasks.DEV_SERVE)
        self.assertEqual(fake.handlers, tasks.DEV_SERVE)
        self.assertEqual(said[-1], "Tailnet: https://box.tail1234.ts.net/")

    def test_run_from_dev_drops_the_api_path(self):
        fake = FakeTailscale(tasks.DEV_SERVE)
        self.ensure(fake, tasks.APP_SERVE)
        self.assertEqual(fake.handlers, tasks.APP_SERVE)
        self.assertIn(["serve", "--https=443", "--set-path=/api/", "off"], fake.changes)

    def test_a_second_start_changes_nothing(self):
        for want in (tasks.DEV_SERVE, tasks.APP_SERVE):
            fake = FakeTailscale(want)
            said = self.ensure(fake, want)
            self.assertEqual(fake.changes, [])
            self.assertIn("already", said[-1])

    def test_no_tailscale_changes_nothing(self):
        fake = FakeTailscale({})
        for binary, url in ((None, "https://box.tail1234.ts.net/"), ("/bin/tailscale", "")):
            said = []
            tasks.ensure_serve(tasks.DEV_SERVE, binary, url, run=fake, say=said.append)
            self.assertIn("not running", said[0])
        self.assertEqual(fake.changes, [])

    def test_a_change_that_does_not_stick_warns(self):
        fake = FakeTailscale({})
        stuck = mock.Mock(side_effect=lambda cmd, **kw: fake(cmd, **kw) if cmd[1:3] == ["serve", "status"] else mock.Mock(returncode=1))
        said = []
        tasks.ensure_serve(tasks.DEV_SERVE, "/bin/tailscale", "https://box/", run=stuck, say=said.append)
        self.assertTrue(any(s.startswith("WARNING") for s in said))


class SetupTest(unittest.TestCase):
    def test_the_running_python_is_used_when_new_enough(self):
        self.assertEqual(tasks.find_python("/py/current", (3, 13, 1)), "/py/current")

    def test_an_old_python_looks_for_a_newer_one(self):
        found = {"python3.13": "/opt/python3.13", "python3": "/usr/bin/python3"}
        self.assertEqual(tasks.find_python("/usr/bin/python3", (3, 9, 6), found.get, lambda exe: exe == "/opt/python3.13"), "/opt/python3.13")
        with self.assertRaises(SystemExit):
            tasks.find_python("/usr/bin/python3", (3, 9, 6), found.get, lambda exe: False)

    def test_cuda_wheel_only_with_nvidia_off_a_mac(self):
        with mock.patch.object(tasks.sys, "platform", "win32"):
            self.assertEqual(tasks.torch_install(True), ["install", tasks.TORCH, "--index-url", tasks.CUDA_INDEX])
            self.assertEqual(tasks.torch_install(False), ["install", tasks.TORCH])
        with mock.patch.object(tasks.sys, "platform", "darwin"):
            self.assertEqual(tasks.torch_install(True), ["install", tasks.TORCH])  # Metal is in the default wheel

    def test_the_cuda_index_and_torch_pin_match_the_requirements(self):
        root = Path(tasks.__file__).parent
        self.assertIn(tasks.CUDA_INDEX, (root / "requirements-gpu.txt").read_text())
        self.assertIn(tasks.TORCH, (root / "requirements-training.txt").read_text())

    def test_the_tailnet_url_is_the_magicdns_name(self):
        self.assertEqual(tasks.tailnet_url(("box", "box.tail1234.ts.net")), "https://box.tail1234.ts.net/")
        self.assertEqual(tasks.tailnet_url(()), "")

    def test_stop_ends_the_whole_process_group_or_tree(self):
        proc = mock.Mock(pid=42)
        proc.poll.return_value = None
        gone = lambda pid, sig: None if sig == tasks.signal.SIGTERM else (_ for _ in ()).throw(ProcessLookupError())  # noqa: E731
        with mock.patch.object(tasks.sys, "platform", "linux"), mock.patch.object(tasks.os, "killpg", create=True, side_effect=gone) as killpg:
            tasks.stop(proc, sleep=lambda s: None)
        self.assertEqual([c.args for c in killpg.call_args_list], [(42, tasks.signal.SIGTERM), (42, 0)])
        with mock.patch.object(tasks.sys, "platform", "win32"), mock.patch.object(tasks.subprocess, "call") as call:
            tasks.stop(proc)
        self.assertEqual(call.call_args.args[0][:2], ["taskkill", "/PID"])

    def test_stop_kills_the_group_when_it_ignores_sigterm(self):
        proc = mock.Mock(pid=42)
        with mock.patch.object(tasks.sys, "platform", "linux"), mock.patch.object(tasks.os, "killpg", create=True) as killpg:
            tasks.stop(proc, grace=.3, sleep=lambda s: None)
        self.assertEqual(killpg.call_args_list[0].args, (42, tasks.signal.SIGTERM))
        self.assertEqual(killpg.call_args_list[-1].args, (42, tasks.signal.SIGKILL))


class SourceWatchTest(unittest.TestCase):
    def test_only_the_python_sources_uvicorn_watched_are_scanned(self):
        walked = [("/r", ["web", ".venv", ".git", ".claude", "data", "runs", "node_modules", "studio_api", "tests"], ["a.py", "b.txt"]),
                  ("/r/studio_api", ["__pycache__"], ["c.py", "c.pyc"])]
        seen = []

        def walk(root):
            for here, dirs, files in walked:
                yield here, dirs, files
                seen.append(list(dirs))

        with mock.patch.object(tasks.os, "stat", return_value=mock.Mock(st_mtime_ns=7)):
            found = tasks.scan_sources("/r", walk)
        self.assertEqual(set(found), {tasks.os.path.join("/r", "a.py"), tasks.os.path.join("/r/studio_api", "c.py")})
        self.assertEqual(seen[0], ["studio_api", "tests"])
        self.assertEqual(seen[1], [])

    def test_a_vanished_file_is_skipped(self):
        with mock.patch.object(tasks.os, "stat", side_effect=FileNotFoundError):
            self.assertEqual(tasks.scan_sources("/r", lambda root: [("/r", [], ["a.py"])]), {})

    def test_a_change_triggers_once_after_it_settles(self):
        states = iter([{"a": 1}, {"a": 1}, {"a": 2}, {"a": 2, "b": 1}, {"a": 2, "b": 1}, {"a": 2, "b": 1}])
        watcher = tasks.SourceWatcher(lambda: next(states))
        self.assertFalse(watcher.poll())  # unchanged
        self.assertFalse(watcher.poll())  # a changed: wait, still being written
        self.assertFalse(watcher.poll())  # b appeared: still moving
        self.assertTrue(watcher.poll())   # settled
        self.assertFalse(watcher.poll())  # reported once

    def test_decisions(self):
        step = tasks.dev_step
        self.assertEqual(step(True, True, False, True), "keep")
        self.assertEqual(step(True, True, True, True), "restart")
        self.assertEqual(step(False, True, True, True), "restart")
        self.assertEqual(step(False, True, False, True), "wait")
        self.assertEqual(step(True, False, True, True), "stop")  # Next ended: stop, even with a change pending
        self.assertEqual(step(False, True, False, False), "stop")  # uvicorn reloads itself, so an ended API is real
        self.assertEqual(step(True, True, True, False), "keep")


class NodeDeviceTest(unittest.TestCase):
    def test_it_waits_until_the_api_answers(self):
        clock, calls = Clock(), []

        def fetch():
            calls.append(clock())
            if len(calls) < 3:
                raise ConnectionRefusedError
            return {"device": "mps"}

        self.assertEqual(tasks.node_device(fetch, sleep=clock.sleep, clock=clock), "mps")
        self.assertEqual(calls, [0., 1., 2.])

    def test_it_gives_up_after_the_timeout(self):
        clock = Clock()

        def fetch():
            raise ConnectionRefusedError

        self.assertIsNone(tasks.node_device(fetch, sleep=clock.sleep, clock=clock, timeout=5.))
        self.assertEqual(clock.now, 5.)

    def test_announce_prints_the_device(self):
        said = []
        tasks.announce(fetch=lambda: {"device": "mps"}, say=said.append)
        self.assertEqual(said, ["GET /node reports device mps"])

    def test_announce_says_when_the_api_never_answered(self):
        said, clock = [], Clock()

        def fetch():
            raise OSError

        tasks.announce(fetch=fetch, say=said.append, sleep=clock.sleep, clock=clock, timeout=2.)
        self.assertIn("did not answer", said[0])


if __name__ == "__main__":
    unittest.main()
