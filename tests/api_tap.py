"""Listen to a studio_api app's real GET /events route in-process, so a test can wait for an event a request caused.

TestClient buffers a response until the app finishes, and /events never does; this drives the ASGI app directly instead.
"""
import asyncio
import json
import threading
import time


class EventTap:
    def __init__(self, app):
        self.app, self.events, self.connected = app, [], False
        self._cond = threading.Condition()
        self._loop = asyncio.new_event_loop()
        self._stop = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        self.wait_for(lambda: self.connected, "the /events connection")
        return self

    def __exit__(self, *exc):
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(5)
        self._loop.close()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        self._stop = stop = asyncio.Event()
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "scheme": "http", "path": "/events",
                 "raw_path": b"/events", "query_string": b"", "headers": [], "client": ("127.0.0.1", 1), "server": ("testserver", 80)}

        async def receive():
            await stop.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] != "http.response.body":
                return
            for block in message.get("body", b"").decode().split("\n\n"):
                lines = block.split("\n")
                if lines[0] == ": connected":
                    with self._cond:
                        self.connected = True
                        self._cond.notify_all()
                elif lines[0].startswith("data: "):
                    event = json.loads(lines[0][6:])
                    with self._cond:
                        self.events.append((event["kind"], event["payload"]))
                        self._cond.notify_all()

        await self.app(scope, receive, send)

    def wait_for(self, check, what, timeout=5.):
        end = time.monotonic() + timeout
        with self._cond:
            while not check():
                left = end - time.monotonic()
                if left <= 0:
                    raise AssertionError(f"timed out waiting for {what}; saw {[kind for kind, _ in self.events]}")
                self._cond.wait(left)

    def wait(self, kind, count=1, timeout=5.):
        """Block until `count` events of `kind` have arrived; returns their payloads."""
        self.wait_for(lambda: len(self.of(kind)) >= count, f"{count} x {kind!r}", timeout)
        return self.of(kind)

    def of(self, kind):
        return [payload for k, payload in list(self.events) if k == kind]
