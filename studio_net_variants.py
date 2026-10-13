"""Comparable named ModelSpec variants and parameter estimates."""
from __future__ import annotations

import copy
import tkinter as tk
from tkinter import ttk

from studio_arch import model_spec_presets
from studio_estimate import estimate
from studio_net_common import NetworkView
from studio_spec import block_at
from studio_widgets import label


def block_differences(old, new):
    """One line per block whose norm, mixer or channel differs; blocks only one design has are listed too."""
    lines = []
    for index in range(max(old["dims"]["n_layer"], new["dims"]["n_layer"])):
        before = block_at(old, index) if index < old["dims"]["n_layer"] else None
        after = block_at(new, index) if index < new["dims"]["n_layer"] else None
        if before is None or after is None:
            lines.append(f"block {index + 1}: {'added' if before is None else 'removed'}")
            continue
        changes = [f"{slot} {before[slot]['id']} → {after[slot]['id']}" for slot in ("norm", "mixer", "channel")
                   if before[slot]["id"] != after[slot]["id"]]
        if changes:
            lines.append(f"block {index + 1}: " + ", ".join(changes))
    return lines or ["blocks: same modules in every block"]


class VariantsView(NetworkView):
    title = "Variants"

    def __init__(self, parent, host):
        super().__init__(parent, host)
        self.variant = tk.StringVar(value="")
        row = ttk.Frame(self)
        row.pack(fill="x")
        label(row, "Named design").pack(side="left")
        self.select = ttk.Combobox(row, textvariable=self.variant, state="readonly", width=32)
        self.select.pack(side="left", padx=8)
        self.select.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        ttk.Button(row, text="Apply to design", command=self.apply).pack(side="left")
        self.details = ttk.Frame(self)
        self.details.pack(fill="both", expand=True, pady=(12, 0))

    def apply(self):
        key = self.variant.get()
        preset = model_spec_presets().get(key)
        if preset:
            self.host.net_apply_model_edit(lambda spec: (spec.clear(), spec.update(copy.deepcopy(preset))),
                                           f"apply {key} preset")

    def refresh(self):
        presets = model_spec_presets()
        self.select.configure(values=list(presets))
        if self.variant.get() not in presets:
            self.variant.set(next(iter(presets)))
        for child in self.details.winfo_children():
            child.destroy()
        if not self.spec:
            return
        chosen = presets[self.variant.get()]
        base_est, chosen_est = estimate(self.spec), estimate(chosen)
        label(self.details, f"Current: {base_est['total']:,} parameters · {base_est['kv_bytes_per_token']:,} KV bytes/token").pack(anchor="w")
        label(self.details, f"{chosen['name']}: {chosen_est['total']:,} parameters · {chosen_est['kv_bytes_per_token']:,} KV bytes/token").pack(anchor="w", pady=(4, 12))
        for slot in ("tokenizer", "embedding", "position", "head"):
            label(self.details, f"{slot}: {self.spec[slot]['id']} → {chosen[slot]['id']}").pack(anchor="w")
        for line in block_differences(self.spec, chosen):
            label(self.details, line).pack(anchor="w")
        self.message.set("Parameter and cache values come from the selected modules' estimators.")
