"""A TestCase for the studio_api routers: the app over FakeServices, a TestClient, and a live reader of GET /events.

TestClient cannot read an endless stream (it waits for the body to finish), so `Capture` drives the app's ASGI callable
directly: it asks for /events, keeps every event that arrives, and never disconnects until `close()`.
"""
import asyncio
import json
import threading
import time
import unittest
from collections import Counter

from fastapi.testclient import TestClient

from studio_api.app import create_app
from studio_ia.services import FakeServices

TIMEOUT = 10.0


class Capture:
    def __init__(self, app):
        self.app, self.events, self.connected = app, [], threading.Event()
        self._text, self._cond = "", threading.Condition()
        self._loop, self._task = asyncio.new_event_loop(), None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self.connected.wait(TIMEOUT):
            raise AssertionError("GET /events did not open")

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "path": "/events",
                 "raw_path": b"/events", "query_string": b"", "root_path": "", "scheme": "http", "headers": [],
                 "client": ("127.0.0.1", 1), "server": ("testserver", 80)}
        asked = False

        async def receive():
            nonlocal asked
            if not asked:
                asked = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == "http.response.body":
                self._take(message.get("body", b"").decode())

        self._task = asyncio.ensure_future(self.app(scope, receive, send))
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    def _take(self, text):
        with self._cond:
            self._text += text
            *blocks, self._text = self._text.split("\n\n")
            for block in blocks:
                lines = block.split("\n")
                if lines[0].startswith(":"):
                    self.connected.set()
                    continue
                data = json.loads(next(l[6:] for l in lines if l.startswith("data: ")))
                named = next((l[7:] for l in lines if l.startswith("event: ")), None)
                if named is None:  # one unnamed message carrying {"kind", "payload"}
                    named, data = data["kind"], data["payload"]
                self.events.append((named, data))
            self._cond.notify_all()

    def wait(self, ready, what="the events"):
        """Block until ready(events) is true; fail with what did arrive otherwise."""
        end = time.monotonic() + TIMEOUT
        with self._cond:
            while not ready(self.events):
                left = end - time.monotonic()
                if left <= 0:
                    raise AssertionError(f"Timed out waiting for {what}; got {Counter(k for k, _ in self.events)}")
                self._cond.wait(left)
            return list(self.events)

    def of(self, kind):
        return [payload for k, payload in self.events if k == kind]

    def close(self):
        if self._task is not None:
            self._loop.call_soon_threadsafe(self._task.cancel)
        self._thread.join(TIMEOUT)
        if not self._thread.is_alive():
            self._loop.close()


class ApiCase(unittest.TestCase):
    """setUp builds the app; `self.services` is what the routers call, `self.events` what /events carries."""

    prefix = ""  # the router's path prefix: test_every_route_answers checks all its routes were hit

    def make_services(self):
        return FakeServices()

    def setUp(self):
        self.services = self.make_services()
        self.emitted = []
        self.services.watch(lambda kind, payload: self.emitted.append(kind))
        self.app = create_app(self.services, open=True)
        self.client = TestClient(self.app)
        self.events = Capture(self.app)
        self.hit = set()
        self.addCleanup(self.stop)

    def stop(self):
        self.events.close()
        self.app.state.jobs.close()
        self.app.state.broadcaster.close()

    def call(self, method, template, expect=200, **kwargs):
        """Request `template` (a path as the OpenAPI schema spells it, {names} filled from `path=`), check the status, remember the route was hit."""
        path = kwargs.pop("path", {})
        response = self.client.request(method, template.format(**path), **kwargs)
        self.assertEqual(response.status_code, expect, f"{method} {template}: {response.text}")
        self.hit.add((method, template))
        return response.json() if response.content else None

    def job(self, method, template, **kwargs):
        """Start a long read (202) and wait for it to finish; returns the finished job as GET /jobs/{id} gives it."""
        started = self.call(method, template, 202, **kwargs)
        end = time.monotonic() + TIMEOUT
        while time.monotonic() < end:
            record = self.client.get(f"/jobs/{started['id']}").json()
            if record["status"] in ("done", "failed", "cancelled"):
                return record
            time.sleep(0.02)
        self.fail(f"job {started['id']} did not finish: {record}")

    def assertAllRoutesHit(self):
        routes = {(method.upper(), path) for path, methods in self.app.openapi()["paths"].items() if path.startswith(self.prefix)
                  for method in methods}
        self.assertTrue(routes)
        self.assertEqual(sorted(routes - self.hit), [], "routes no request reached")

    def assertDelivered(self):
        """Every event Services emitted so far, by kind and count, has come out of GET /events."""
        want = Counter(self.emitted)
        self.assertTrue(want, "nothing was emitted")
        self.events.wait(lambda events: not want - Counter(k for k, _ in events), f"the emitted events {dict(want)}")
