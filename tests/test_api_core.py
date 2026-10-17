"""studio_api's core over FakeServices: the /events stream, jobs, auth and the static web export. The routers have their own tests."""

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from studio_api import app as api_app
from studio_api.__main__ import LOCAL
from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.services import FakeServices


class Stream:
    """A GET /events response read straight off the ASGI app. TestClient buffers a whole response and /events never ends."""

    def __init__(self, app):
        self.app, self.text, self.task, self._gone = app, "", None, asyncio.Event()

    async def open(self):
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "scheme": "http",
                 "path": "/events", "raw_path": b"/events", "query_string": b"", "root_path": "", "headers": [],
                 "client": ("test", 1), "server": ("test", 80)}

        async def receive():
            await self._gone.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                self.text += message.get("body", b"").decode()

        self.task = asyncio.ensure_future(self.app(scope, receive, send))
        await self.until(lambda: ": connected" in self.text)  # the subscription exists once this arrives
        return self

    async def until(self, done, timeout=5.0):
        end = time.monotonic() + timeout
        while not done():
            if time.monotonic() > end:
                raise AssertionError(f"timed out; the stream so far: {self.text!r}")
            await asyncio.sleep(0.01)

    def events(self):
        frames = [f for f in self.text.split("\n\n") if f.startswith("data: ")]
        return [json.loads(f[len("data: "):]) for f in frames]

    async def close(self):
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


def create(services, open=True):
    """create_app without the web/out mount, so a developer's own build does not decide what these tests see (a mount at /
    would also answer for any route a test adds afterwards)."""
    with mock.patch.object(api_app, "WEB_OUT", Path(tempfile.gettempdir()) / "no-such-web-out"):
        return create_app(services, open=open, hosts=ANY)


def shut(app):
    app.state.jobs.close()
    app.state.broadcaster.close()


class EventsTest(unittest.TestCase):
    def test_every_emit_reaches_two_subscribers_in_order(self):
        services = FakeServices()
        app = create(services)

        async def run():
            first, second = await Stream(app).open(), await Stream(app).open()
            try:
                await asyncio.to_thread(services.emit, "status", {"version": "v3"})  # another thread, as a worker would
                await asyncio.to_thread(services.emit, "queue", {"state": "running"})
                await asyncio.to_thread(services.emit, "no_payload")
                for stream in (first, second):
                    await stream.until(lambda s=stream: any(e["kind"] == "no_payload" for e in s.events()))
                def emitted(stream):
                    return [e for e in stream.events() if not (e["kind"] == "status" and "queue" in e["payload"])]
                return emitted(first), emitted(second)
            finally:
                await first.close()
                await second.close()

        try:
            one, two = asyncio.run(run())
        finally:
            shut(app)
        expected = [{"kind": "status", "payload": {"version": "v3"}}, {"kind": "queue", "payload": {"state": "running"}},
                    {"kind": "no_payload", "payload": None}]
        self.assertEqual(one, expected)
        self.assertEqual(two, expected)

    def test_a_dataclass_payload_is_sent_as_its_fields(self):
        services = FakeServices()
        app = create(services)

        async def run():
            stream = await Stream(app).open()
            try:
                await asyncio.to_thread(services.save_version, {}, "first change")  # emits "versions" with a Version
                await stream.until(lambda: any(e["kind"] == "versions" for e in stream.events()))
                return next(e for e in stream.events() if e["kind"] == "versions")
            finally:
                await stream.close()

        try:
            event = asyncio.run(run())
        finally:
            shut(app)
        self.assertEqual(event["kind"], "versions")
        self.assertEqual(event["payload"]["change"], "first change")

    def test_a_closed_stream_is_dropped(self):
        services = FakeServices()
        app = create(services)

        async def run():
            stream = await Stream(app).open()
            self.assertEqual(len(app.state.broadcaster._clients), 1)
            await stream.close()
            await asyncio.sleep(0.05)

        try:
            asyncio.run(run())
        finally:
            shut(app)
        self.assertEqual(app.state.broadcaster._clients, {})

    def test_shell_gets_initial_and_updated_status_without_a_custom_status_emit(self):
        services = FakeServices()
        services.save_version({}, "first design")
        app = create(services)

        async def run():
            stream = await Stream(app).open()
            try:
                await stream.until(lambda: any(e["kind"] == "status" for e in stream.events()))
                initial = next(e["payload"] for e in stream.events() if e["kind"] == "status")
                await asyncio.to_thread(services.pause)
                await stream.until(lambda: any(e["kind"] == "status" and e["payload"]["queue"]["state"] == "paused"
                                                for e in stream.events()))
                return initial
            finally:
                await stream.close()
        try:
            initial = asyncio.run(run())
        finally:
            shut(app)
        self.assertEqual(initial["version"], "version 1 · saved")
        self.assertEqual(initial["queue"], {"state": "idle", "running": 0, "waiting": 0})
        self.assertEqual(initial["resources"], {"device": None, "diskFreeGb": 120})


class JobsTest(unittest.TestCase):
    def setUp(self):
        self.services = FakeServices()
        self.app = create(self.services)

        @self.app.post("/test-slow")
        def slow(request: api_app.Request):
            def work(stop):
                self.services.loading = ("halfway", 0.5)
                time.sleep(0.6)  # a progress poll (every 0.25 s) sees it
                return {"answer": 42}
            return {"id": request.app.state.jobs.submit("slow", work).id}

        @self.app.post("/test-quick")
        def quick(request: api_app.Request):
            def work(stop):
                self.services.note("Copied from the other machine")  # done before the first progress poll could see it
                return {}
            return {"id": request.app.state.jobs.submit("quick", work).id}

        @self.app.post("/test-fail")
        def fail(request: api_app.Request):
            def work(stop):
                raise ValueError("no good")
            return {"id": request.app.state.jobs.submit("fail", work).id}

        self.events = []
        self.services.watch(lambda kind, payload: self.events.append((kind, payload)))

    def wait_for(self, client, job_id, status):
        end = time.monotonic() + 5
        while time.monotonic() < end:
            job = client.get(f"/jobs/{job_id}").json()
            if job["status"] == status:
                return job
            time.sleep(0.02)
        self.fail(f"job {job_id} never reached {status}: {job}")

    def test_progress_a_result_and_a_late_get(self):
        with TestClient(self.app) as client:
            job_id = client.post("/test-slow").json()["id"]
            job = self.wait_for(client, job_id, "done")
            late = client.get(f"/jobs/{job_id}").json()
            listed = client.get("/jobs").json()
        self.assertEqual(late, job)
        self.assertEqual(late["result"], {"answer": 42})
        self.assertEqual([j["id"] for j in listed], [job_id])
        states = [p["status"] for k, p in self.events if k == "job" and p["id"] == job_id]
        self.assertEqual([states[0], states[1], states[-1]], ["queued", "running", "done"])
        progress = [p for k, p in self.events if k == "job" and p["id"] == job_id and p["progress"]]
        self.assertTrue(progress, "no job event carried the progress text")
        self.assertEqual((progress[0]["progress"], progress[0]["fraction"]), ("halfway", 0.5))

    def test_a_short_jobs_last_note_stays_on_its_record_and_the_next_job_starts_clean(self):
        with TestClient(self.app) as client:
            job = self.wait_for(client, client.post("/test-quick").json()["id"], "done")
            self.assertEqual(job["progress"], "Copied from the other machine")
            failed = self.wait_for(client, client.post("/test-fail").json()["id"], "failed")
        self.assertEqual(failed["progress"], "")

    def test_a_failing_job_reports_its_error(self):
        with TestClient(self.app) as client:
            job = self.wait_for(client, client.post("/test-fail").json()["id"], "failed")
        self.assertEqual(job["error"], "ValueError: no good")

    def test_an_unknown_job_is_404(self):
        with TestClient(self.app) as client:
            self.assertEqual(client.get("/jobs/nope").status_code, 404)
            self.assertEqual(client.delete("/jobs/nope").status_code, 404)


class AuthTest(unittest.TestCase):
    def status(self, open, headers=None, path="/jobs"):
        app = create(FakeServices(), open=open)
        with TestClient(app) as client:
            return client.get(path, headers=headers or {})

    def test_refused_without_the_header_when_not_open(self):
        response = self.status(open=False)
        self.assertEqual(response.status_code, 401)
        self.assertIn("Tailscale-User-Login", response.json()["detail"])

    def test_refusal_covers_every_path(self):
        for path in ("/events", "/jobs/x", "/openapi.json", "/"):
            self.assertEqual(self.status(open=False, path=path).status_code, 401, path)

    def test_allowed_with_the_header_when_not_open(self):
        self.assertEqual(self.status(open=False, headers={"Tailscale-User-Login": "me@example.com"}).status_code, 200)

    def test_allowed_without_the_header_when_open(self):
        self.assertEqual(self.status(open=True).status_code, 200)

    def test_the_user_is_the_tailscale_login_or_local(self):
        seen = {}
        for open, headers in ((False, {"Tailscale-User-Login": "me@example.com"}), (True, {})):
            app = create(FakeServices(), open=open)

            @app.get("/test-user")
            def user(request: api_app.Request):
                return {"user": request.state.user}
            with TestClient(app) as client:
                seen[open] = client.get("/test-user", headers=headers).json()["user"]
        self.assertEqual(seen, {False: "me@example.com", True: "local"})

    def test_main_opens_the_server_for_loopback_and_fake_only(self):
        from studio_api.__main__ import make

        for argv, expect_open in ((["--fake"], True), (["--fake", "--bind", "100.64.0.1"], True)):
            app, _, _ = make(argv)
            self.assertIs(app.state.open, expect_open, argv)
            shut(app)
        for host in LOCAL:
            with mock.patch("studio_core.Workspace"), mock.patch("studio_ia.services.WorkspaceServices", return_value=FakeServices()):
                app, bound, _ = make(["--bind", host])
            self.assertTrue(app.state.open, host)
            self.assertEqual(bound, host)
            shut(app)
        with mock.patch("studio_core.Workspace"), mock.patch("studio_ia.services.WorkspaceServices", return_value=FakeServices()):
            app, _, _ = make(["--bind", "100.64.0.1"])
        self.assertFalse(app.state.open)
        shut(app)


class CorsTest(unittest.TestCase):
    """`pnpm dev` serves the UI on localhost:3000 and calls the API on another port: allowed for an open server only."""

    def allowed(self, open, origin):
        with TestClient(create(FakeServices(), open=open)) as client:
            return client.get("/jobs", headers={"Origin": origin, "Tailscale-User-Login": "me@example.com"}).headers.get(
                "access-control-allow-origin")

    def test_local_origins_are_allowed_on_an_open_server(self):
        for origin in ("http://localhost:3000", "http://127.0.0.1:3000", "http://localhost"):
            self.assertEqual(self.allowed(True, origin), origin)

    def test_other_origins_are_not(self):
        for origin in ("https://example.com", "http://localhost.evil.com", "http://127.0.0.1.evil.com:3000"):
            self.assertIsNone(self.allowed(True, origin), origin)

    def test_a_server_behind_tailscale_allows_no_cross_origin_page(self):
        self.assertIsNone(self.allowed(False, "http://localhost:3000"))


class WebOutTest(unittest.TestCase):
    def test_web_out_is_served_at_the_root_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "index.html").write_text("<h1>Groundwork</h1>", encoding="utf-8")
            (out / "queue").mkdir()
            (out / "queue" / "index.html").write_text("<h1>Queue</h1>", encoding="utf-8")
            with mock.patch.object(api_app, "WEB_OUT", out), TestClient(create_app(FakeServices(), open=True, hosts=ANY)) as client:
                root, queue = client.get("/"), client.get("/queue/")
                self.assertEqual(client.get("/jobs").status_code, 200)  # the API still wins over the mount
        self.assertIn("Groundwork", root.text)
        self.assertIn("Queue", queue.text)

    def test_nothing_is_mounted_without_web_out(self):
        with TestClient(create(FakeServices())) as client:
            self.assertEqual(client.get("/").status_code, 404)


if __name__ == "__main__":
    unittest.main()
