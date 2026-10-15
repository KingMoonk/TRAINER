"""Shared widgets. Plain tk widgets in token colours; every clickable one is keyboard-focusable."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from studio_ia.theme import C, GUTTER, STATES, font, tk_label


def _activate(widget, command):
    """Click, Return and space all fire `command` (keyboard parity for non-button widgets).

    Also draws the focus ring: a 1px violet outline that is invisible (same colour as the widget) until it has focus.
    """
    if not int(widget.cget("highlightthickness")):
        widget.configure(highlightthickness=1, highlightbackground=widget.cget("bg"))
    widget.configure(highlightcolor=C["violet"])
    for seq in ("<Button-1>", "<Return>", "<space>"):
        widget.bind(seq, lambda _e: command())


class Card(tk.Frame):
    """Panel with a 1px border and 16px padding; pack children straight into it."""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C["panel"], highlightthickness=1, highlightbackground=C["border"],
                         padx=GUTTER, pady=GUTTER, **kw)


class Kicker(tk.Label):
    """12px uppercase label. ponytail: Tk cannot track letters, so 0.08em is approximated with plain caps."""

    def __init__(self, parent, text="", color="muted", bg="bg"):
        super().__init__(parent, text=text.upper(), font=font(12, bold=True), fg=C.get(color, color), bg=C.get(bg, bg), anchor="w")


class Chip(tk.Label):
    """State pill. state is one of theme.STATES: unsaved, needs_training, training, live, attention, pieces, neutral."""

    def __init__(self, parent, text="", state="neutral"):
        super().__init__(parent, font=font(12, bold=True), padx=10, pady=3)
        self.set(text, state)

    def set(self, text, state="neutral"):
        fg, bg = STATES[state]
        self.configure(text=text, fg=fg, bg=bg)


class Segmented(tk.Frame):
    """Row of exclusive choices. `.value`, `.set(v)`, command(v) on change. Arrow keys move, space/Return pick."""

    accent = "violet"

    def __init__(self, parent, options, value=None, command=None, bg="panel"):
        super().__init__(parent, bg=C[bg])
        self.options, self.value, self.command = list(options), value, command
        self.buttons = {}
        for opt in self.options:
            b = tk.Label(self, text=opt, font=font(14), padx=14, pady=10, takefocus=1, cursor="hand2",
                         highlightthickness=1, highlightbackground=C["border"], highlightcolor=C["violet"])
            b.pack(side="left", padx=(0, 4))
            _activate(b, lambda o=opt: self.pick(o))
            self.buttons[opt] = b
        self._paint()

    def _paint(self):
        for opt, b in self.buttons.items():
            on = opt == self.value
            b.configure(bg=C[self.accent] if on else C["panel2"], fg=C["bg"] if on else C["text"])

    def set(self, value):
        self.value = value
        self._paint()

    def pick(self, value):
        self.set(value)
        if self.command:
            self.command(value)


class GuessChips(Segmented):
    """A guess before every run: pick one label; `.value` is None until the person guesses."""

    accent = "pink"

    def __init__(self, parent, labels, command=None, bg="panel"):
        super().__init__(parent, labels, None, command, bg)


class Toggle(tk.Frame):
    """On/off switch with a label. `.value`, `.set(bool)`, command(bool)."""

    def __init__(self, parent, text, value=False, command=None, bg="panel"):
        super().__init__(parent, bg=C[bg])
        self.value, self.command = bool(value), command
        self.track = tk.Label(self, width=4, font=font(12, bold=True), takefocus=1, cursor="hand2",
                              highlightthickness=1, highlightbackground=C["border"], highlightcolor=C["violet"])
        self.track.pack(side="left")
        tk_label(self, text, 14, "text", bg).pack(side="left", padx=(10, 0))
        _activate(self.track, self.flip)
        self.set(self.value)

    def set(self, value):
        self.value = bool(value)
        self.track.configure(text="on" if self.value else "off", bg=C["violet"] if self.value else C["panel2"],
                             fg=C["bg"] if self.value else C["muted"])

    def flip(self):
        self.set(not self.value)
        if self.command:
            self.command(self.value)


class KofN(tk.Label):
    """"k of n label": small samples are always shown as counts, never as a bare percentage."""

    def __init__(self, parent, k=None, n=None, label="", bg="panel"):
        super().__init__(parent, font=font(15, mono=True), fg=C["text"], bg=C[bg], anchor="w")
        self.label = label
        self.set(k, n)

    def set(self, k, n, label=None):
        self.label = self.label if label is None else label
        head = "not counted yet ·" if k is None or n is None else f"{k} of {n}"
        self.configure(text=f"{head} {self.label}".strip())


class StateTray(tk.Frame):
    """Always-visible state line: kicker + sentence, tinted by state."""

    def __init__(self, parent, state="neutral", kicker="", text=""):
        super().__init__(parent, highlightthickness=1)
        self.kicker = Kicker(self)
        self.kicker.pack(anchor="w", padx=GUTTER, pady=(12, 0))
        self.text = tk.Label(self, font=font(14), anchor="w", justify="left", wraplength=640, fg=C["text2"])
        self.text.pack(anchor="w", padx=GUTTER, pady=(2, 12))
        self.set(state, kicker, text)

    def set(self, state, kicker, text):
        fg, bg = STATES[state]
        self.configure(bg=bg, highlightbackground=fg)
        self.kicker.configure(text=kicker.upper(), fg=fg, bg=bg)
        self.text.configure(text=text, bg=bg)


class HowItWorks(tk.Frame):
    """Collapsible 'why and how'. Pack content into `.body`. Name what is inside; never 'Advanced'."""

    def __init__(self, parent, title="How this works", open_=False, bg="panel"):
        super().__init__(parent, bg=C[bg])
        self.title, self.open = title, open_
        self.head = tk_label(self, "", 14, "muted", bg, takefocus=1, cursor="hand2", highlightthickness=1,
                             highlightbackground=C[bg], highlightcolor=C["violet"])
        self.head.pack(anchor="w")
        _activate(self.head, self.toggle)
        self.body = tk.Frame(self, bg=C[bg])
        self.toggle(self.open)

    def toggle(self, open_=None):
        self.open = (not self.open) if open_ is None else open_
        self.head.configure(text=("▾ " if self.open else "▸ ") + self.title)
        if self.open:
            self.body.pack(fill="x", pady=(6, 0))
        else:
            self.body.pack_forget()


class Progress(tk.Canvas):
    """Thin bar; `.set(fraction)` with None = empty."""

    def __init__(self, parent, height=6, color="violet", bg="panel"):
        super().__init__(parent, height=height, bg=C["panel2"], highlightthickness=0)
        self.color, self.fraction = color, None
        self.bind("<Configure>", lambda _e: self._draw())

    def set(self, fraction):
        self.fraction = fraction
        self._draw()

    def _draw(self):
        self.delete("all")
        if self.fraction:
            self.create_rectangle(0, 0, self.winfo_width() * min(1, max(0, self.fraction)), self.winfo_height(), fill=C[self.color], width=0)


class HeatGrid(tk.Canvas):
    """rows x cols of cells; `.set(matrix, row_labels, col_labels)`; values 0..1, None = no data (dim)."""

    CELL = 28

    def __init__(self, parent, color="violet", bg="panel"):
        super().__init__(parent, bg=C[bg], highlightthickness=0, height=40)
        self.color = color
        self.set([])

    @staticmethod
    def _mix(a, b, t):
        pa, pb = [int(x[i:i + 2], 16) for x in (a,) for i in (1, 3, 5)], [int(x[i:i + 2], 16) for x in (b,) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(pa, pb))

    def set(self, matrix, row_labels=(), col_labels=()):
        self.delete("all")
        pad = 56 if row_labels else 0
        top = 18 if col_labels else 0
        for r, row in enumerate(matrix):
            if r < len(row_labels):
                self.create_text(pad - 6, top + r * self.CELL + self.CELL / 2, text=row_labels[r], anchor="e", fill=C["muted"], font=font(11))
            for c, v in enumerate(row):
                x, y = pad + c * self.CELL, top + r * self.CELL
                fill = C["panel2"] if v is None else self._mix(C["panel2"], C[self.color], max(0.0, min(1.0, v)))
                self.create_rectangle(x, y, x + self.CELL - 2, y + self.CELL - 2, fill=fill, width=0)
        for c, label in enumerate(col_labels):
            self.create_text(pad + c * self.CELL + self.CELL / 2, 8, text=label, fill=C["muted"], font=font(11))
        cols = max((len(r) for r in matrix), default=0)
        self.configure(width=pad + cols * self.CELL, height=max(40, top + len(matrix) * self.CELL))


def plot_series(d):
    """The lines of a BpcPlot as (points, colour, width, dash): the faint per-step and bold smoothed lines, then each
    `series` entry {points, color (theme name or hex), width=2, dash=None}. Lines under two points are dropped."""
    lines = [(d.get("steps"), C["border"], 1, None), (d.get("smooth"), C["violet"], 3, None)]
    lines += [(s["points"], C.get(s.get("color", "violet"), s.get("color")), s.get("width", 2), s.get("dash")) for s in d.get("series") or []]
    return [(list(pts), color, width, dash) for pts, color, width, dash in lines if pts and len(pts) >= 2]


class BpcPlot(tk.Canvas):
    """Error in bits per character. Faint per-step line, bold smoothed line, held-back dots, reference lines
    above an axis break, shaded unrun remainder and a learning-rate lane with phase bands underneath.

    set(steps=[(x, y)], smooth=[(x, y)], heldout=[(x, y)], refs=[(name, value)], total=None, progress=None,
        lr=[(x, lr)], phases=[(start, end, name)], band=None, series=[{points, color, width, dash}])
    `series` adds lines of their own colour and optional dash pattern, e.g. one per seed or a dashed held-back curve.
    Reference lines above the data range go in the strip above the break; `band` = (lo, hi) shades the noise band.
    With no data it says so."""

    PHASE = {"warm-up": "amber", "steady": "violet", "cool-down": "blue"}
    M = dict(l=56, r=16, t=14, b=22)

    def __init__(self, parent, width=640, height=300, lane=56):
        super().__init__(parent, width=width, height=height + lane, bg=C["panel"], highlightthickness=0)
        self.h, self.lane, self.data = height, lane, {}
        self.bind("<Configure>", lambda _e: self._draw())

    def set(self, **data):
        self.data = data
        self._draw()

    def _draw(self):
        self.delete("all")
        d, m = self.data, self.M
        w, h = max(self.winfo_width(), 200), self.h
        x0, x1 = m["l"], w - m["r"]
        pts = [p for k in ("steps", "smooth", "heldout") for p in d.get(k) or []]
        pts += [p for s in d.get("series") or [] for p in s["points"]]
        if not pts:
            self.create_text(w / 2, h / 2, text="No points yet: the curve appears once training logs its first steps.", fill=C["dim"], font=font(13))
            return
        total = d.get("total") or max(p[0] for p in pts) or 1
        ys = [p[1] for p in pts]
        lo, hi = min(ys), max(ys)
        lo, hi = lo - 0.05 * (hi - lo or 1), hi + 0.05 * (hi - lo or 1)
        refs = sorted(d.get("refs") or [], key=lambda r: -r[1])
        high = [r for r in refs if r[1] > hi]
        strip = 18 * len(high) + (14 if high else 0)  # room for the lines above the axis break
        yt, yb = m["t"] + strip, h - m["b"]

        def X(x):
            return x0 + (x1 - x0) * x / total

        def Y(y):
            return yb - (yb - yt) * (y - lo) / (hi - lo)

        prog = d.get("progress")
        if prog is not None and prog < total:
            self.create_rectangle(X(prog), m["t"], x1, yb, fill=C["panel2"], width=0)
        if d.get("band"):
            self.create_rectangle(x0, Y(d["band"][1]), x1, Y(d["band"][0]), fill=C["blue_bg"], width=0)
        self.create_line(x0, yb, x1, yb, fill=C["border"])
        self.create_line(x0, m["t"], x0, yb, fill=C["border"])
        for i, (name, val) in enumerate(high):
            y = m["t"] + 9 + 18 * i
            self.create_line(x0, y, x1, y, fill=C["muted"], dash=(4, 3))
            self.create_text(x0 - 6, y, text=f"{val:.2f}", anchor="e", fill=C["muted"], font=font(11, mono=True))
            self.create_text(x1, y - 8, text=name, anchor="e", fill=C["muted"], font=font(11))
        if high:  # axis break: two slashes across the y axis
            for dy in (0, 5):
                self.create_line(x0 - 6, yt - 12 + dy, x0 + 6, yt - 18 + dy, fill=C["text2"])
        for name, val in refs:
            if lo <= val <= hi:
                self.create_line(x0, Y(val), x1, Y(val), fill=C["muted"], dash=(4, 3))
                self.create_text(x1, Y(val) - 8, text=name, anchor="e", fill=C["muted"], font=font(11))
        for v in (lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo)):
            self.create_text(x0 - 6, Y(v), text=f"{v:.2f}", anchor="e", fill=C["muted"], font=font(11, mono=True))
        for points, color, width, dash in plot_series(d):
            self.create_line(*[c for p in points for c in (X(p[0]), Y(p[1]))], fill=color, width=width, dash=dash)
        for x, y in d.get("heldout") or []:
            self.create_oval(X(x) - 4, Y(y) - 4, X(x) + 4, Y(y) + 4, fill=C["blue"], outline=C["panel"])
        # learning-rate lane
        lt, lb = h + 6, h + self.lane - 4
        for start, end, name in d.get("phases") or []:
            col = C[self.PHASE.get(name, "dim")]
            self.create_rectangle(X(start), lt, X(end), lb, fill=C["panel2"], outline=col)
            self.create_text((X(start) + X(end)) / 2, lb - 8, text=name, fill=col, font=font(11))
        lr = d.get("lr") or []
        top = max((v for _, v in lr), default=0) or 1
        line = [c for x, v in lr for c in (X(x), lb - (lb - lt - 16) * v / top - 14)]
        if len(line) >= 4:
            self.create_line(*line, fill=C["text2"], width=2)
        self.create_text(x0 - 6, (lt + lb) / 2, text="LR", anchor="e", fill=C["muted"], font=font(11, bold=True))


def button(parent, text, command, primary=False):
    """ttk button; primary ones are >= 44px tall (see theme.apply)."""
    return ttk.Button(parent, text=text, command=command, style="Primary.TButton" if primary else "TButton", takefocus=True)
