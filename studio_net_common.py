"""Shared, main-thread-only helpers for Network designer views."""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk

from studio_widgets import MUTED, label


class NetworkView(ttk.Frame):
    title = ""

    def __init__(self, parent, host):
        super().__init__(parent, padding=12)
        self.host = host
        label(self, self.title, "Title.TLabel").pack(anchor="w", pady=(0, 8))
        self.message = tk.StringVar(value="")
        label(self, textvariable=self.message, style="Muted.TLabel", wraplength=900,
              foreground=MUTED).pack(anchor="w", fill="x", pady=(0, 8))

    @property
    def spec(self):
        return self.host.net_model_spec

    def checkpoint(self):
        if self.host.arch_id is None:
            raise ValueError("Select a saved design first.")
        weights = self.host.arch_store.weights(self.host.arch_id)
        if not weights:
            raise ValueError("Train this design or register a checkpoint to inspect it.")
        return self.host.arch_store.weight_path(self.host.arch_id, weights[0]["version"])

    def work(self, task, done):
        """Run a numerical service off Tk's thread and deliver its result through the host queue."""
        self.message.set("Computing from the selected checkpoint…")
        def worker():
            try:
                outcome = (True, task())
            except Exception as error:
                outcome = (False, str(error))
            self.host.net_view_events.put((self, done, outcome))
        threading.Thread(target=worker, daemon=True, name="network-inspection").start()

    def refresh(self):
        pass
