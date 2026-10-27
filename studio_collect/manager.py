"""The app's side of the collector: keep the background service running when collection is on, and register its shards.

The collector lives only while the app is open and only when the person turned it on in this session: starting the app turns a saved
"on" off and stops a collector left from an earlier session, and closing the app stops the one it started (`stand_down`).

One daemon thread per app process. It never fetches anything itself. `tick()` starts the service if the settings want it and it is not
running, and adds finished shards to the workspace (studio_collect.register), then tells the page through `emit("collect")`.
"""
from __future__ import annotations

import threading

from studio_collect import config as cfg
import studio_platform
from studio_collect import register, service
from studio_collect.store import Journal, State

INTERVAL = 30.0


class Manager:
    def __init__(self, workspace, emit=lambda kind, payload=None: None, interval=INTERVAL, ensure=service.ensure_running, running=service.running,
                 terminate=studio_platform.terminate):
        self.workspace, self.emit, self.interval, self.ensure, self.running = workspace, emit, interval, ensure, running
        self.terminate = terminate
        self.stop = threading.Event()
        self.thread = None
        self.lock = threading.Lock()

    @property
    def root(self):
        return self.workspace.root

    def stand_down(self):
        """Stop the collector process if one runs and turn collection off, so it never carries on without the app or starts by itself."""
        with self.lock:
            if self.running(self.root):
                self.terminate(State(self.root).data.get("pid"))
            if cfg.load(self.root)["enabled"]:
                cfg.save(self.root, {"enabled": False})

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.stand_down()
            self.stop.clear()
            self.thread = threading.Thread(target=self._loop, name="collect-manager", daemon=True)
            self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.stand_down()

    def _loop(self):
        while not self.stop.is_set():
            try:
                self.tick()
            except Exception as error:    # noqa: BLE001 - a failed pass is retried at the next one; the journal (and so the page) says why
                try:
                    Journal(self.root).append("error", message=f"collect pass failed: {type(error).__name__}: {error}")
                except OSError:
                    pass
            self.stop.wait(self.interval)

    def tick(self):
        """One pass (also called right after a settings change so the page does not wait for the timer)."""
        with self.lock:
            changed = False
            if cfg.load(self.root)["enabled"] and not self.running(self.root):
                changed = self.ensure(self.root) is not None
            if register.pending(self.root):
                report = register.register(self.workspace)
                changed = report["shards"] > 0 or report["failed"] > 0 or changed
        if changed:
            self.emit("collect", {})
        return changed
