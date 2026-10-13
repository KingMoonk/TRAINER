"""Shared checkpoint probe controls; numerical work runs through NetworkView.work."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from studio_inspect import probe_model
from studio_net_common import NetworkView
from studio_spec import block_at
from studio_validate import hooks_of
from studio_widgets import label, set_text, text_box


def fit_prompt(source, text):
    """Token ids for text, keeping the last n_ctx tokens; returns (ids, tokens dropped from the front)."""
    model, tokenizer = probe_model(source)
    ids = list(tokenizer.encode(text))
    keep = model.dims.n_ctx
    return ids[-keep:], max(0, len(ids) - keep)


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

    def selected(self, spec=None):
        spec = spec or self.spec
        block, head = int(self.block.get()) - 1, int(self.head.get()) - 1
        if not 0 <= block < spec["dims"]["n_layer"] or head < 0:
            raise ValueError("Choose an existing block and a positive head number.")
        return block, head

    def probe_spec(self):
        """Blocks and hooks come from the picked checkpoint's design, not the one being edited."""
        try:
            return self.checkpoint_spec(self.checkpoint_row())
        except ValueError:
            return self.spec

    def refresh(self):
        if not self.spec:
            return
        spec = self.probe_spec()
        self.block_box.configure(values=[str(i + 1) for i in range(spec["dims"]["n_layer"])])
        if self.block.get() not in self.block_box.cget("values"):
            self.block.set("1")
        try:
            block, _ = self.selected(spec)
        except ValueError:
            self.button.configure(state="disabled")
            return
        available = hooks_of(spec)
        missing = [name for name in self.needed if f"block{block}.{name}" not in available]
        if missing:
            module = block_at(spec, block)["mixer"]["id"]
            self.message.set(f"Block {block + 1} uses {module}; this view needs {missing[0]}.")
            self.button.configure(state="disabled")
        else:
            self.message.set("Pick a checkpoint in Versions › Weights (default: newest) and a prompt, then inspect real model values.")
            self.button.configure(state="normal")

    def inspect(self):
        try:
            row = self.checkpoint_row()
            source, text = self.checkpoint(row), self.prompt.get().strip()
            block, head = self.selected(self.checkpoint_spec(row))
            if not text:
                raise ValueError("Enter a prompt first.")
        except ValueError as error:
            self.message.set(str(error))
            return
        header = f"{self.provenance(row)} · prompt {text!r} · block {block + 1} · head {head + 1}"

        def task():
            ids, dropped = fit_prompt(source, text)
            return block, head, header, dropped, self.compute(source, ids, block, head)
        self.work(task, self.finish)

    def compute(self, source, ids, block, head):
        """ids: the prompt's token ids, already cut to the checkpoint's context length."""
        raise NotImplementedError

    def finish(self, ok, result):
        if not ok:
            self.message.set(str(result))
            return
        block, head, header, dropped, payload = result
        self.block.set(str(block + 1))  # describe() labels from these: show what was computed, not what was typed since
        self.head.set(str(head + 1))
        set_text(self.output, self.describe(payload))
        self.message.set(header + (f" · prompt longer than the context: the first {dropped} tokens were dropped" if dropped else ""))

    def describe(self, result):
        return str(result)
