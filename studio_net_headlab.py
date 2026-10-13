"""Measured attention-head ablations and pattern statistics."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from studio_inspect import ablate, forward
from studio_net_probe import ProbeView


class HeadLabView(ProbeView):
    title = "Head lab"
    needed = ("attn.head_out", "attn.pattern")

    def __init__(self, parent, host):
        super().__init__(parent, host)
        self.mode = tk.StringVar(value="zero")
        controls = ttk.Frame(self)
        controls.pack(fill="x", before=self.output)
        ttk.Radiobutton(controls, text="Zero ablation", variable=self.mode, value="zero").pack(side="left")
        ttk.Radiobutton(controls, text="Mean ablation", variable=self.mode, value="mean").pack(side="left", padx=12)

    def inspect(self):
        self._requested_mode = self.mode.get()
        super().inspect()

    def compute(self, source, text, block, head):
        data = forward(source, text)
        count = data["hooks"][f"block{block}.attn.head_out"].shape[1]
        return ablate(source, text, mode=self._requested_mode, heads=[(block, index) for index in range(count)])

    def describe(self, result):
        lines = [f"Baseline objective loss: {result['baseline']:.5f}",
                 f"Intervention: {result['mode']} · one head removed at a time", "",
                 "Head     Δ loss      Entropy      Mean distance      First-token mass"]
        for row in result["heads"]:
            lines.append(f"{row['head'] + 1:>4}   {row['delta']:>+9.5f}   {row['entropy']:>9.4f}   "
                         f"{row['mean_distance']:>14.3f}   {row['first_token_mass']:>16.3f}")
        lines.append("\nThese are measurements on this prompt, not labels for what a head learned.")
        return "\n".join(lines)
