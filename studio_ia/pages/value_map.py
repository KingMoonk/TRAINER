"""Value map: W_V and its row block of W_O as two funnels around a head_dim-wide bottleneck."""
from __future__ import annotations

import tkinter as tk

import numpy as np

from studio_ia.copy import VALUE_MAP
from studio_ia.pages.base import Page
from studio_ia.pages.trace import LI, ask
from studio_ia.theme import C, GUTTER, font, tk_label
from studio_ia.widgets import Card, Kicker, Segmented, info_icon

HEADS = ("1", "2", "4", "8")
PALETTE = (C["amber"], C["blue"], C["violet"], C["pink"], "#b8a4ff", C["sage"], "#f09a5e", "#d0a8e8")


def rank_of(singular, tol=1e-6):
    """Numerical rank: singular values above tol of the largest."""
    s = np.asarray(singular, dtype=float)
    return int((s > tol * s.max()).sum()) if s.size and s.max() > 0 else 0


class FunnelCanvas(tk.Canvas):
    """x (d) -> W_V -> v (head_dim) -> W_O -> delta x (d). Heights are to scale, with an 8px floor so the bottleneck shows."""

    def __init__(self, parent, width=720, height=260):
        super().__init__(parent, width=width, height=height, bg=C["panel"], highlightthickness=0)
        self.w, self.h = width, height

    def set(self, d, hd, heads, texts):
        self.delete("all")
        mid, full = self.h / 2 - 8, 200
        narrow = max(8, full * hd / d)
        x0, x1, x2, x3, x4 = 20, 150, 330, 510, 640
        for x, label in ((x0, texts["x"]), (x4, texts["dx"])):
            self.create_rectangle(x, mid - full / 2, x + 24, mid + full / 2, fill=C["panel2"], outline=C["border"])
            self.create_text(x + 12, mid + full / 2 + 14, text=label, fill=C["muted"], font=font(12, mono=True))
        for cx, label in (((x1 + x2) / 2, texts["w_v"]), ((x2 + x3) / 2, texts["w_o"])):
            self.create_text(cx, 14, text=label, fill=C["text2"], font=font(13, bold=True))
        self.create_polygon(x0 + 24, mid - full / 2, x2, mid - narrow / 2, x2, mid + narrow / 2, x0 + 24, mid + full / 2, fill=C["amber_bg"], outline=C["amber"])
        self.create_polygon(x2, mid - narrow / 2, x4, mid - full / 2, x4, mid + full / 2, x2, mid + narrow / 2, fill=C["blue_bg"], outline=C["blue"])
        self.create_rectangle(x2 - 6, mid - narrow / 2, x2 + 6, mid + narrow / 2, fill=C["violet"], outline="")
        self.create_text(x2, mid + full / 2 + 14, text=texts["v"], fill=C["violet"], font=font(12, mono=True))
        # the other heads write into other directions: coloured stripes on the output side
        for i in range(min(heads, len(PALETTE))):
            y = mid - full / 2 + full * i / heads
            self.create_rectangle(x4 + 24, y, x4 + 34, y + full / heads - 1, fill=PALETTE[i], outline="")


class ValueMapPage(Page):
    title = VALUE_MAP["h1"]
    crumb = (*LI, "Value map")

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.heads, self.block, self.head = 4, 0, 0

    def set_heads(self, heads):
        self.heads = int(heads)
        self.block = self.head = 0
        self.refresh()

    def pick(self, block, head):
        self.block, self.head = block, head
        self.refresh()

    def refresh(self):
        for child in self.body.winfo_children():
            child.destroy()
        info, error = ask(self.services, "inspect_value_map", self.heads, block=self.block, head=self.head)
        if info is None:
            tk_label(self.body, error, 14, "muted", wraplength=720).pack(anchor="w")
            return
        d, heads = int(info["d_model"]), self.heads
        fit = lambda s: s.replace("256", str(d))  # noqa: E731 - the copy is written for d_model 256
        if d % heads:
            tk_label(self.body, f"{d} does not split into {heads} equal heads.", 14, "amber").pack(anchor="w")
            return
        hd = d // heads
        row = tk.Frame(self.body, bg=C["bg"])
        row.pack(fill="x")
        tk_label(row, fit(VALUE_MAP["d_model"]), 14, "muted").pack(side="left", padx=(0, 8))
        Segmented(row, HEADS, str(heads), command=self.set_heads, bg="bg").pack(side="left")
        tk_label(self.body, fit(VALUE_MAP["head_dim"]).format(heads=heads) + f" {hd}", 18, "text", font=font(18, bold=True)).pack(anchor="w", pady=GUTTER)
        card = Card(self.body)
        card.pack(fill="x")
        notes = (f"{fit(VALUE_MAP['together'])} {hd}. " + VALUE_MAP["each_head"].format(hd=hd), VALUE_MAP["weighting"].format(hd=hd),
                 VALUE_MAP["params"].format(per_head=f"{2 * d * hd:,}", heads=heads, total=f"{2 * d * d:,}", d=d))
        head = tk.Frame(card, bg=C["panel"])
        head.pack(anchor="w", pady=(0, 6))
        Kicker(head, VALUE_MAP["one_head"], bg="panel").pack(side="left")
        info_icon(head, "\n\n".join(notes)).pack(side="left", padx=(4, 0))
        self.funnels = FunnelCanvas(card)
        self.funnels.pack(anchor="w")
        self.funnels.set(d, hd, heads, dict(x=fit(VALUE_MAP["x_label"]), dx=fit(VALUE_MAP["delta_x"]), v=VALUE_MAP["v_label"].format(hd=hd),
                                            w_v=f"{VALUE_MAP['w_v']}  {d} → {hd}", w_o=f"{VALUE_MAP['w_o']}  {hd} → {d}"))
        singular = info.get("singular")
        if singular is not None and heads == info["heads"]:
            rank = rank_of(singular)
            blocks = info.get("blocks") or {}
            if len(blocks) > 1 or heads > 1:
                pickers = tk.Frame(card, bg=C["panel"])
                pickers.pack(anchor="w", pady=(GUTTER, 0))
                Segmented(pickers, [f"Block {b + 1}" for b in blocks], f"Block {self.block + 1}",
                          command=lambda v: self.pick(int(v.split()[1]) - 1, 0)).pack(side="left", padx=(0, GUTTER))
                Segmented(pickers, [f"Head {h + 1}" for h in range(heads)], f"Head {self.head + 1}",
                          command=lambda v: self.pick(self.block, int(v.split()[1]) - 1)).pack(side="left")
            tk_label(card, VALUE_MAP["measured"].format(block=self.block + 1, head=self.head + 1, rank=rank, hd=hd),
                     13, "text2", "panel", wraplength=760).pack(anchor="w", pady=(GUTTER, 4))
            bars = tk.Canvas(card, width=420, height=60, bg=C["panel"], highlightthickness=0)
            bars.pack(anchor="w")
            top = max(float(np.max(singular)), 1e-12)
            n = len(singular)
            for i, s in enumerate(singular):
                h = 56 * float(s) / top
                bars.create_rectangle(i * 420 / n + 1, 58 - h, (i + 1) * 420 / n - 1, 58, fill=C["violet"] if i < hd else C["border"], width=0)
        elif heads == info["heads"]:
            tk_label(card, VALUE_MAP["measured_none"], 13, "dim", "panel").pack(anchor="w", pady=(GUTTER, 0))
        else:
            tk_label(card, VALUE_MAP["measured_other"].format(heads=info["heads"]), 13, "dim", "panel").pack(anchor="w", pady=(GUTTER, 0))
