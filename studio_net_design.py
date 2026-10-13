"""Canonical ModelSpec design controls and vertical residual stream."""
from __future__ import annotations

from tkinter import ttk

from studio_circuit import ResidualStream
from studio_net_common import NetworkView
from studio_widgets import label


class DesignView(NetworkView):
    title = "Design"

    def __init__(self, parent, host):
        super().__init__(parent, host)
        row = ttk.Frame(self)
        row.pack(fill="x", pady=(0, 8))
        self.entries = {}
        for key, title in (("d_model", "Width"), ("n_layer", "Blocks"), ("n_ctx", "Context"), ("dropout", "Dropout")):
            group = ttk.Frame(row)
            group.pack(side="left", padx=(0, 12))
            label(group, title).pack(anchor="w")
            entry = ttk.Entry(group, width=10)
            entry.pack()
            entry.bind("<Return>", lambda _event, k=key: self.apply(k))
            entry.bind("<FocusOut>", lambda _event, k=key: self.apply(k))
            self.entries[key] = entry
        self.stream = ResidualStream(self, self.select_slot)
        self.stream.pack(fill="both", expand=True)

    def select_slot(self, key):
        self.host.net_model_selected = key
        self.host.net_show_view("Sandbox")

    def apply(self, key):
        if not self.spec:
            return
        value = self.entries[key].get().strip()
        try:
            value = float(value) if key == "dropout" else int(value)
        except ValueError:
            self.message.set(f"{key} needs a number.")
            return
        if self.spec["dims"][key] == value:
            return
        self.host.net_apply_model_edit(lambda spec: spec["dims"].__setitem__(key, value), f"set {key} to {value}")

    def refresh(self):
        if not self.spec:
            return
        for key, entry in self.entries.items():
            value = str(self.spec["dims"][key])
            if entry.get() != value and entry != self.focus_get():
                entry.delete(0, "end")
                entry.insert(0, value)
        self.stream.set_spec(self.spec, self.host.net_model_estimate)
        estimate = self.host.net_model_estimate
        self.message.set(f"{estimate['total']:,} parameters · {estimate['train_flops_per_token']:,} approximate training FLOPs/token · "
                         f"{estimate['kv_bytes_per_token']:,} KV bytes/token" if estimate else self.host.net_model_error)
