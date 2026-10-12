"""Registry-driven module swapping and configuration."""
from __future__ import annotations

import copy
import json
import tkinter as tk
from tkinter import ttk

import studio_registry as registry
from studio_net_common import NetworkView
from studio_slots import BLOCK_SLOTS, TRAIN_SLOTS
from studio_spec import block_at, set_block
from studio_validate import why_blocked
from studio_widgets import label, text_box

SLOTS = ("tokenizer", "embedding", "position", "norm", "mixer", "channel", "head",
         "objective", "optimizer", "schedule", "init", "precision")


class SandboxView(NetworkView):
    title = "Sandbox"

    def __init__(self, parent, host):
        super().__init__(parent, host)
        controls = ttk.Frame(self)
        controls.pack(fill="x")
        label(controls, "Slot").pack(side="left")
        self.slot = tk.StringVar(value="mixer")
        select = ttk.Combobox(controls, textvariable=self.slot, values=SLOTS, state="readonly", width=15)
        select.pack(side="left", padx=(6, 16))
        select.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        label(controls, "Scope").pack(side="left")
        self.scope = tk.StringVar(value="All blocks")
        self.scope_box = ttk.Combobox(controls, textvariable=self.scope, state="readonly", width=16)
        self.scope_box.pack(side="left", padx=(6, 16))
        self.scope_box.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        ttk.Button(controls, text="View spec JSON", command=self.show_json).pack(side="right")
        self.options = ttk.Frame(self)
        self.options.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=(10, 0))
        self.detail = ttk.Frame(self)
        self.detail.pack(side="right", fill="both", expand=True, pady=(10, 0))

    def _block(self):
        value = self.scope.get()
        return None if value == "All blocks" else int(value.split()[1]) - 1

    def _entry(self):
        spec, slot = self.spec, self.slot.get()
        if slot in BLOCK_SLOTS:
            return (spec["block"] if self._block() is None else block_at(spec, self._block()))[slot]
        return spec["train"][slot] if slot in TRAIN_SLOTS else spec[slot]

    def _set(self, spec, entry):
        slot = self.slot.get()
        if slot in BLOCK_SLOTS:
            if self._block() is None:
                spec["block"][slot] = entry
                for override in spec.get("blocks", []):
                    override.pop(slot, None)
            else:
                set_block(spec, self._block(), slot, entry)
        elif slot in TRAIN_SLOTS:
            spec["train"][slot] = entry
        else:
            spec[slot] = entry

    def choose(self, short):
        reason = why_blocked(self.spec, self.slot.get(), short, self._block())
        if reason:
            self.message.set(reason)
            return
        self.host.net_apply_model_edit(lambda spec: self._set(spec, {"id": short}),
                                       f"swap {self.slot.get()} to {short}")

    def configure_field(self, name, raw):
        slot, entry = self.slot.get(), copy.deepcopy(self._entry())
        fields = {field["name"]: field for field in registry.fields(registry.get(slot, entry["id"]))}
        field = fields[name]
        try:
            value = raw.lower() in ("true", "1", "yes") if field["type"] == "bool" else int(raw) if field["type"] == "int" else float(raw) if field["type"] == "float" else raw
        except ValueError:
            self.message.set(f"{name} needs a {field['type']} value.")
            return
        entry["config"][name] = value
        self.host.net_apply_model_edit(lambda spec: self._set(spec, entry), f"set {slot}.{name}")

    def refresh(self):
        if not self.spec:
            return
        slot = self.slot.get()
        scopes = ["All blocks"] + [f"Block {i + 1}" for i in range(self.spec["dims"]["n_layer"])]
        self.scope_box.configure(values=scopes, state="readonly" if slot in BLOCK_SLOTS else "disabled")
        if self.scope.get() not in scopes:
            self.scope.set("All blocks")
        for frame in (self.options, self.detail):
            for child in frame.winfo_children():
                child.destroy()
        entry = self._entry()
        label(self.options, f"Swap {slot}", "CardTitle.TLabel").pack(anchor="w", pady=(0, 6))
        for module in registry.for_slot(slot):
            reason = why_blocked(self.spec, slot, module.short, self._block())
            line = ttk.Frame(self.options)
            line.pack(fill="x", pady=2)
            state = "disabled" if reason else "normal"
            ttk.Button(line, text=module.name, command=lambda s=module.short: self.choose(s), state=state,
                       width=23).pack(side="left")
            label(line, "selected" if module.short == entry["id"] else reason or module.blurb,
                  style="Muted.TLabel", wraplength=340).pack(side="left", padx=(8, 0))
        module = registry.get(slot, entry["id"])
        label(self.detail, module.name, "CardTitle.TLabel").pack(anchor="w")
        label(self.detail, module.blurb, style="Muted.TLabel", wraplength=360).pack(anchor="w", pady=(3, 8))
        for field in registry.fields(module):
            line = ttk.Frame(self.detail)
            line.pack(fill="x", pady=3)
            label(line, field["name"], width=18).pack(side="left")
            value = str(entry["config"].get(field["name"], field["default"]))
            variable = tk.StringVar(value=value)
            choices = field.get("choices")
            widget = ttk.Combobox(line, textvariable=variable, values=choices, state="readonly") if choices else ttk.Entry(line, textvariable=variable)
            widget.pack(side="left", fill="x", expand=True)
            widget.bind("<<ComboboxSelected>>", lambda _event, n=field["name"], v=variable: self.configure_field(n, v.get()))
            widget.bind("<Return>", lambda _event, n=field["name"], v=variable: self.configure_field(n, v.get()))
            widget.bind("<FocusOut>", lambda _event, n=field["name"], v=variable: self.configure_field(n, v.get()))
        label(self.detail, "Hooks: " + (", ".join(name for name, _shape in module.hooks) or "none"),
              style="Muted.TLabel", wraplength=360).pack(anchor="w", pady=(10, 2))
        label(self.detail, "Backward: " + (module.backward or "Module-defined"), style="Muted.TLabel",
              wraplength=360).pack(anchor="w")
        self.message.set(self.host.net_model_error or f"Source: {registry.source_path(module)}")

    def show_json(self):
        window = tk.Toplevel(self)
        window.title("ModelSpec JSON")
        area = text_box(window, height=30)
        area.pack(fill="both", expand=True)
        area.insert("1.0", json.dumps(self.spec, indent=2, ensure_ascii=False))
