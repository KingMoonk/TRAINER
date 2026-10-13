"""Shared, main-thread-only helpers for Network designer views."""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk

import numpy as np

from studio_widgets import MUTED, label


def to_numpy(value):
    """Engine hooks and weights may be torch tensors; display code works on NumPy arrays."""
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


class NetworkView(ttk.Frame):
    title = ""

    def __init__(self, parent, host):
        super().__init__(parent, padding=12)
        self.host = host
        self.request_id = 0
        label(self, self.title, "Title.TLabel").pack(anchor="w", pady=(0, 8))
        self.message = tk.StringVar(value="")
        label(self, textvariable=self.message, style="Muted.TLabel", wraplength=900,
              foreground=MUTED).pack(anchor="w", fill="x", pady=(0, 8))

    @property
    def spec(self):
        return self.host.net_model_spec

    def checkpoint_row(self):
        """The weight version picked in Versions › Weights, or the newest one."""
        if self.host.arch_id is None:
            raise ValueError("Select a saved design first.")
        weights = self.host.arch_store.weights(self.host.arch_id)
        if not weights:
            raise ValueError("Train this design or register a checkpoint to inspect it.")
        versions = getattr(self.host, "net_views", {}).get("Versions")
        picked = versions.trees["Weights"].selection() if versions is not None else ()
        return next((row for row in weights if picked and str(row["version"]) == picked[0]), weights[0])

    def checkpoint(self, row=None):
        row = row or self.checkpoint_row()
        return self.host.arch_store.weight_path(self.host.arch_id, row["version"])

    def checkpoint_spec(self, row):
        """The design the checkpoint was trained from; the edited design when the checkpoint has no revision."""
        if row.get("revision") is None:
            return self.spec
        try:
            return self.host.arch_store.revision_model_spec(self.host.arch_id, row["revision"])
        except (ValueError, KeyError):
            return self.spec

    def provenance(self, row):
        text = f"Checkpoint v{row['version']}" + (f" · design r{row['revision']}" if row.get("revision") is not None else "")
        if row.get("run_dir"):
            text += f" · run {row['run_dir']}"
        if self.checkpoint_spec(row) != self.spec:
            text += " · warning: the edited design differs from this checkpoint's; blocks and hooks follow the checkpoint"
        return text

    def work(self, task, done):
        """Run a numerical service off Tk's thread; only the latest request's result is delivered."""
        self.message.set("Computing from the selected checkpoint…")
        self.request_id += 1
        request = self.request_id

        def deliver(*outcome):
            if request == self.request_id:  # a newer request superseded this one
                done(*outcome)

        def worker():
            try:
                outcome = (True, task())
            except Exception as error:
                outcome = (False, str(error))
            self.host.events.put(("net_view", (self, deliver, outcome)))
        threading.Thread(target=worker, daemon=True, name="network-inspection").start()

    def refresh(self):
        pass
