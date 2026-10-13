"""Shared checkpoint probe controls; numerical work runs through NetworkView.work."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from studio_net_common import NetworkView
from studio_spec import block_at
from studio_validate import hooks_of
from studio_widgets import label, set_text, text_box


class ProbeView(NetworkView):
    needed = ()

    def __init__(self, parent, host):
        super().__init__(parent, host)
        controls = ttk.Frame(self)
        controls.pack(fill="x")
        label(controls, "Prompt").pack(side="left")
        self.prompt = ttk.Entry(controls, width=52)
        self.prompt.insert(0, "Name the largest planet in our solar system. The largest planet is")
        self.prompt.pack(side="left", padx=(6, 12))
        label(controls, "Block").pack(side="left")
        self.block = tk.StringVar(value="1")
        self.block_box = ttk.Combobox(controls, textvariable=self.block, state="readonly", width=5)
        self.block_box.pack(side="left", padx=(5, 12))
        label(controls, "Head").pack(side="left")
        self.head = tk.StringVar(value="1")
        self.head_box = ttk.Entry(controls, textvariable=self.head, width=5)
        self.head_box.pack(side="left", padx=(5, 12))
        self.button = ttk.Button(controls, text="Inspect checkpoint", command=self.inspect)
        self.button.pack(side="left")
        self.output = text_box(self, height=20, state="disabled")
        self.output.pack(fill="both", expand=True, pady=(10, 0))

    def selected(self):
        block, head = int(self.block.get()) - 1, int(self.head.get()) - 1
        if not 0 <= block < self.spec["dims"]["n_layer"] or head < 0:
            raise ValueError("Choose an existing block and a positive head number.")
        return block, head

    def refresh(self):
        if not self.spec:
            return
        self.block_box.configure(values=[str(i + 1) for i in range(self.spec["dims"]["n_layer"])])
        if self.block.get() not in self.block_box.cget("values"):
            self.block.set("1")
        try:
            block, _ = self.selected()
        except ValueError:
            self.button.configure(state="disabled")
            return
        available = hooks_of(self.spec)
        missing = [name for name in self.needed if f"block{block}.{name}" not in available]
        if missing:
            module = block_at(self.spec, block)["mixer"]["id"]
            self.message.set(f"Block {block + 1} uses {module}; this view needs {missing[0]}.")
            self.button.configure(state="disabled")
        else:
            self.message.set("Select a checkpoint and prompt, then inspect real model values.")
            self.button.configure(state="normal")

    def inspect(self):
        try:
            source, text = self.checkpoint(), self.prompt.get().strip()
            block, head = self.selected()
            if not text:
                raise ValueError("Enter a prompt first.")
        except ValueError as error:
            self.message.set(str(error))
            return
        self.work(lambda: self.compute(source, text, block, head), self.finish)

    def compute(self, source, text, block, head):
        raise NotImplementedError

    def finish(self, ok, result):
        if not ok:
            self.message.set(str(result))
            return
        set_text(self.output, self.describe(result))
        self.message.set("Computed from the selected checkpoint.")

    def describe(self, result):
        return str(result)
