"""tasks.py dev: the commands it starts, the device notice and the wait for GET /node. No server is started."""
import socket
import unittest

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
        api, web, _ = tasks.dev_commands("pnpm")
        self.assertIn("--reload", api)
        self.assertEqual(api[api.index("--port") + 1], str(tasks.API_PORT))
        self.assertEqual(web[web.index("--port") + 1], str(tasks.WEB_PORT))
        self.assertEqual(api[:3], [tasks.PY, "-m", "studio_api"])  # the venv that runs tasks.py, so its torch

    def test_serve_commands_name_the_loopback_ports(self):
        page, api = tasks.SERVE
        self.assertTrue(page.endswith(f"http://127.0.0.1:{tasks.WEB_PORT}"))
        self.assertTrue(api.endswith(f"--set-path=/api/ http://127.0.0.1:{tasks.API_PORT}"))

    def test_a_busy_port_is_seen(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            self.assertTrue(tasks.port_busy(listener.getsockname()[1]))
            port = listener.getsockname()[1]
        self.assertFalse(tasks.port_busy(port))


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
