"""The app's side of the collector: keep the background service running when collection is on, and register its shards.

One daemon thread per app process. It never fetches anything itself. `tick()` starts the service if the settings want it and it is not
running, and adds finished shards to the workspace (studio_collect.register), then tells the page through `emit("collect")`.
"""
from __future__ import annotations

import threading

from studio_collect import config as cfg
from studio_collect import register, service

INTERVAL = 30.0


class Manager:
    def __init__(self, workspace, emit=lambda kind, payload=None: None, interval=INTERVAL, ensure=service.ensure_running, running=service.running):
        self.workspace, self.emit, self.interval, self.ensure, self.running = workspace, emit, interval, ensure, running
        self.stop = threading.Event()
        self.thread = None
        self.lock = threading.Lock()

    @property
    def root(self):
        return self.workspace.root

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.stop.clear()
            self.thread = threading.Thread(target=self._loop, name="collect-manager", daemon=True)
            self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def _loop(self):
        while not self.stop.is_set():
            try:
                self.tick()
            except Exception:    # noqa: BLE001 - a failed pass is retried at the next one; the page shows the service's own state
                pass
            self.stop.wait(self.interval)

    def tick(self):
        """One pass (also called right after a settings change so the page does not wait for the timer)."""
        with self.lock:
            changed = False
            if cfg.load(self.root)["enabled"] and not self.running(self.root):
                changed = self.ensure(self.root) is not None
            if register.pending(self.root):
                changed = register.register(self.workspace)["shards"] > 0 or changed
        if changed:
            self.emit("collect", {})
        return changed
