"""Inside a head: five steps from raw q·k to the weighted sum, derived client-side and checked against the model."""
from __future__ import annotations

import math
import re
import tkinter as tk

import numpy as np

from studio_ia.copy import HEAD_ZOOM
from studio_ia.pages.base import Page
from studio_ia.pages.trace import LI, ask, prompt_text, show_tok
from studio_ia.theme import C, GUTTER, font, tk_label
from studio_ia.widgets import Card, HeatGrid, Kicker, Segmented, button

MAX_TOKENS = 12
TOLERANCE = 1e-4
MASK_BG, WHITE = "#0a0b0e", "#e8edf5"


def derive(forward, block, head):
    """Steps 1-5 for one head from a studio_inspect.forward result, in the reference orientation [key i, query j].

    raw = attn.scores, scaled = attn.logits (post-scale, post-bias, so μP and ALiBi are included), masked = -inf where
    i > j, weights = softmax down each column. `error` = max |derived softmax - attn.pattern|; it must be < 1e-4.
    """
    hooks = forward["hooks"]

    def part(name):
        return np.asarray(hooks[f"block{block}.attn.{name}"], dtype=np.float64)[0]
    raw, scaled, pattern = part("scores")[head].T, part("logits")[head].T, part("pattern")[head].T
    q, v = part("q"), part("v")
    T = raw.shape[0]
    masked = np.where(np.arange(T)[:, None] > np.arange(T)[None, :], -np.inf, scaled)
    e = np.exp(masked - masked.max(axis=0, keepdims=True))
    weights = e / e.sum(axis=0, keepdims=True)
    d_head = q.shape[-1]
    kv = head * v.shape[0] // q.shape[0]  # grouped-query attention shares k/v between heads
    return dict(tokens=list(forward["tokens"][0]), raw=raw, scaled=scaled, masked=masked, weights=weights, pattern=pattern,
                error=float(np.abs(weights - pattern).max()), d_head=d_head, v=v[kv],
                plain_scale=bool(np.allclose(scaled, raw / math.sqrt(d_head), atol=1e-6)))


def attention_blocks(forward):
    """{block index: head count} for blocks that have an attention pattern."""
    found = {}
    for name, value in forward["hooks"].items():
        m = re.fullmatch(r"block(\d+)\.attn\.pattern", name)
        if m:
            found[int(m.group(1))] = int(np.asarray(value).shape[1])
    return dict(sorted(found.items()))


class ScoreGrid(tk.Canvas):
    """Rows = keys, columns = queries. `.set(cells, rows, cols, selected)`; cells[i][j] = (text, fill, text colour)."""

    W, H, LEFT, TOP = 56, 34, 84, 40

    def __init__(self, parent, on_pick):
        super().__init__(parent, bg=C["panel"], highlightthickness=1, highlightbackground=C["border"],
                         highlightcolor=C["violet"], takefocus=1)
        self.on_pick, self.texts, self.selected = on_pick, [], (0, 0)
        self.bind("<Button-1>", self._click)
        for key, (di, dj) in {"<Up>": (-1, 0), "<Down>": (1, 0), "<Left>": (0, -1), "<Right>": (0, 1)}.items():
            self.bind(key, lambda _e, d=(di, dj): self._move(*d))

    def _click(self, event):
        self.focus_set()
        i, j = (event.y - self.TOP) // self.H, (event.x - self.LEFT) // self.W
        if 0 <= i < len(self.texts) and 0 <= j < len(self.texts[0]):
            self.on_pick(int(i), int(j))

    def _move(self, di, dj):
        n = len(self.texts)
        self.on_pick(min(n - 1, max(0, self.selected[0] + di)), min(n - 1, max(0, self.selected[1] + dj)))

    def set(self, cells, rows, cols, selected, bold_col=None):
        self.delete("all")
        self.selected = selected
        self.texts = [[c[0] for c in row] for row in cells]
        n = len(cells)
        self.configure(width=self.LEFT + n * self.W + 8, height=self.TOP + n * self.H + 8)
        for j, label in enumerate(cols):
            self.create_text(self.LEFT + j * self.W + self.W / 2, 14, text=show_tok(label)[:7], fill=C["amber"] if j == bold_col else C["text2"], font=font(11))
            self.create_text(self.LEFT + j * self.W + self.W / 2, 29, text=f"q{j + 1}", fill=C["dim"], font=font(10))
        for i, label in enumerate(rows):
            self.create_text(self.LEFT - 6, self.TOP + i * self.H + self.H / 2, text=f"{show_tok(label)[:8]}  k{i + 1}", anchor="e", fill=C["text2"], font=font(11))
            for j, (text, fill, fg) in enumerate(cells[i]):
                x, y = self.LEFT + j * self.W, self.TOP + i * self.H
                on = (i, j) == selected
                self.create_rectangle(x + 1, y + 1, x + self.W - 1, y + self.H - 1, fill=fill, outline=C["violet"] if on else fill, width=2 if on else 1)
                self.create_text(x + self.W / 2, y + self.H / 2, text=text, fill=fg, font=font(11, mono=True))


def heat(value, scale):
    """Amber for positive, blue for negative, stronger with size."""
    a = min(1.0, abs(value) / scale) * .85 + .08
    return HeatGrid._mix(C["panel"], C["amber"] if value >= 0 else C["blue"], a)


class HeadZoomPage(Page):
    title = HEAD_ZOOM["h1"]
    lead = HEAD_ZOOM["lead"]
    crumb = (*LI, "Inside a head")

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.block = self.head = None
        self.step, self.cell, self.d, self.forward, self.error = 1, None, None, None, None

    def select(self, block, head):
        self.block, self.head = block, head
        self.refresh()

    def refresh(self):
        for child in self.body.winfo_children():
            child.destroy()
        self.forward, self.error = self._fetch()
        if self.forward is None:
            tk_label(self.body, self.error, 14, "muted", wraplength=720).pack(anchor="w")
            return
        blocks = attention_blocks(self.forward)
        if not blocks:
            tk_label(self.body, "This model has no attention head to look inside.", 14, "muted").pack(anchor="w")
            return
        if self.block not in blocks or not 0 <= self.head < blocks[self.block]:
            self.block, self.head = next(iter(blocks)), 0
        self.d = derive(self.forward, self.block, self.head)
        T = len(self.d["tokens"])
        if self.cell is None or max(self.cell) >= T:
            self.cell = (max(0, T - 2), T - 1)
        pick = tk.Frame(self.body, bg=C["bg"])
        pick.pack(fill="x", pady=(0, GUTTER))
        Segmented(pick, [f"Block {b + 1}" for b in blocks], f"Block {self.block + 1}", bg="bg",
                  command=lambda v: self.select(int(v.split()[1]) - 1, 0)).pack(side="left", padx=(0, GUTTER))
        Segmented(pick, [f"Head {h + 1}" for h in range(blocks[self.block])], f"Head {self.head + 1}", bg="bg",
                  command=lambda v: self.select(self.block, int(v.split()[1]) - 1)).pack(side="left")
        tk_label(self.body, HEAD_ZOOM["head_label"].format(block=self.block + 1, head=self.head + 1, tokens=T), 14, "muted").pack(anchor="w")
        self.banner = tk_label(self.body, "", 13, "amber", "amber_bg", wraplength=760, padx=GUTTER, pady=8)
        if self.d["error"] <= TOLERANCE:
            tk_label(self.body, HEAD_ZOOM["checked"].format(error=self.d["error"], tol=TOLERANCE), 12, "dim", wraplength=760).pack(anchor="w")
        else:
            self.banner.configure(text=f"The numbers derived here differ from the model's own attention pattern by {self.d['error']:.2g} "
                                       f"(allowed {TOLERANCE:g}). A hook may be wrong, so do not trust this picture.")
            self.banner.pack(anchor="w", fill="x", pady=(6, 0))
        steps = tk.Frame(self.body, bg=C["bg"])
        steps.pack(fill="x", pady=GUTTER)
        names = [HEAD_ZOOM[k] for k in ("raw_scores", "scale_step", "mask_step", "softmax_step", "weighted_step")]
        self.step_buttons = []
        for n, name in enumerate(names, 1):
            b = button(steps, f"{n}. {name}", lambda n=n: self.set_step(n), primary=n == self.step)
            b.pack(side="left", padx=(0, 6))
            self.step_buttons.append(b)
        self.unit = tk_label(self.body, "", 14, "text2", font=font(14, mono=True))
        self.unit.pack(anchor="w", pady=(0, 6))
        row = tk.Frame(self.body, bg=C["bg"])
        row.pack(fill="x")
        self.grid_view = ScoreGrid(row, self._pick)
        self.grid_view.pack(side="left", anchor="n")
        self.side = Card(row)
        self.side.pack(side="left", fill="both", expand=True, padx=(GUTTER, 0), anchor="n")
        tk_label(self.body, HEAD_ZOOM["rows_keys"], 11, "dim").pack(anchor="w", pady=(6, 0))
        self._draw()

    def _fetch(self):
        text = prompt_text()
        data, error = ask(self.services, "inspect_forward", text)
        if data is not None and data["ids"].shape[1] > MAX_TOKENS:  # truncate from the left, keep the query: re-run on the tail
            data, error = ask(self.services, "inspect_forward", [int(t) for t in data["ids"][0, -MAX_TOKENS:]])
        return data, error

    def set_step(self, n):
        self.step = n
        self._draw()

    def _pick(self, i, j):
        self.cell = (i, j)
        self._draw()

    def cells(self):
        """cells[i][j] = (text, fill, colour) for the current step; also what the tests read."""
        d, step = self.d, self.step
        T = len(d["tokens"])
        scale = max(1e-9, float(np.abs(d["scaled"]).max())) if step >= 2 else max(1e-9, float(np.abs(d["raw"]).max()))
        out = []
        for i in range(T):
            line = []
            for j in range(T):
                masked = i > j
                if step == 1:
                    cell = (f"{d['raw'][i, j]:.1f}", heat(d["raw"][i, j], scale), C["text"])
                elif step == 2:
                    cell = (f"{d['scaled'][i, j]:.2f}", heat(d["scaled"][i, j], scale), C["text"])
                elif masked:
                    cell = ("−∞" if step == 3 else "0", MASK_BG, C["dim"])
                elif step == 3:
                    cell = (f"{d['scaled'][i, j]:.2f}", heat(d["scaled"][i, j], scale), C["text"])
                else:
                    w = d["weights"][i, j]
                    dim = step == 5 and j != T - 1
                    cell = (f"{w:.2f}", HeatGrid._mix(C["panel"], WHITE, w * .25 if dim else w * .92 + .04),
                            C["dim"] if dim else C["bg"] if w > .5 else C["text"])
                line.append(cell)
            out.append(line)
        return out

    def _draw(self):
        d, step = self.d, self.step
        for n, b in enumerate(self.step_buttons, 1):
            b.configure(style="Primary.TButton" if n == step else "TButton")
        cells = self.cells()
        self.grid_view.set(cells, d["tokens"], d["tokens"], self.cell, bold_col=len(d["tokens"]) - 1 if step == 5 else None)
        side = self.side
        for child in side.winfo_children():
            child.destroy()
        keys = ("raw", "scale", "mask", "softmax", "weighted")
        key = keys[step - 1]
        sqrt_d = math.sqrt(d["d_head"])
        self.unit.configure(text=HEAD_ZOOM[key + "_unit"].format(sqrt_d=f"{sqrt_d:.3g}", token=show_tok(d["tokens"][-1])))
        Kicker(side, HEAD_ZOOM["step_label"].format(n=step), "violet", "panel").pack(anchor="w")
        tk_label(side, HEAD_ZOOM[{"raw": "raw_scores"}.get(key, key + "_step")], 18, "text", "panel", font=font(18, bold=True)).pack(anchor="w")
        tk_label(side, HEAD_ZOOM[key + "_formula"].format(d=d["d_head"], sqrt_d=f"{sqrt_d:.3g}"), 13, "text2", "panel", font=font(13, mono=True), wraplength=300).pack(anchor="w", pady=4)
        tk_label(side, HEAD_ZOOM[key + "_why"].format(d=d["d_head"], sqrt_d=f"{sqrt_d:.3g}"), 13, "muted", "panel", wraplength=300).pack(anchor="w")
        tk_label(side, self._note(), 12, "muted", "panel", wraplength=300).pack(anchor="w", pady=(6, 0))
        if step == 2 and not d["plain_scale"]:
            tk_label(side, "These are the model's real pre-mask scores, so a position bias or a μP scale is included.", 12, "dim", "panel", wraplength=300).pack(anchor="w", pady=(4, 0))
        i, j = self.cell
        t = d["tokens"]
        Kicker(side, HEAD_ZOOM["selected_cell"], bg="panel").pack(anchor="w", pady=(GUTTER, 0))
        tk_label(side, f"k{i + 1} “{show_tok(t[i])}” · q{j + 1} “{show_tok(t[j])}”", 13, "text", "panel").pack(anchor="w")
        masked = "−∞" if i > j else f"{d['scaled'][i, j]:.2f}"
        for label, value in ((HEAD_ZOOM["cell_label"], f"{d['raw'][i, j]:.2f}"),
                             (HEAD_ZOOM["scaled"].format(d=d["d_head"], sqrt_d=f"{sqrt_d:.3g}"), f"{d['scaled'][i, j]:.2f}"),
                             (HEAD_ZOOM["after_mask"], masked), (HEAD_ZOOM["softmax_weight"], f"{d['weights'][i, j]:.2f}")):
            r = tk.Frame(side, bg=C["panel"])
            r.pack(fill="x")
            tk_label(r, label, 12, "muted", "panel").pack(side="left")
            tk_label(r, value, 13, "text", "panel", font=font(13, mono=True)).pack(side="right")
        if step == 5:
            self._contributors(side)

    def _note(self):
        d, last = self.d, len(self.d["tokens"]) - 1
        t = d["tokens"]
        top = np.argsort(d["weights"][:, last])[::-1]
        if self.step == 1:
            return HEAD_ZOOM["raw_note"]
        if self.step == 2:
            return HEAD_ZOOM["scale_note"]
        if self.step == 3:
            return HEAD_ZOOM["mask_note"]
        if self.step == 4:
            a, b = top[0], top[1 if len(top) > 1 else 0]
            return HEAD_ZOOM["softmax_note"].format(token=show_tok(t[last]), p1=f"{d['weights'][a, last]:.2f}", token1=show_tok(t[a]),
                                                    p2=f"{d['weights'][b, last]:.2f}", token2=show_tok(t[b]))
        return HEAD_ZOOM["weighted_note"].format(token=show_tok(t[last]))

    def _contributors(self, side):
        d, last = self.d, len(self.d["tokens"]) - 1
        t = d["tokens"]
        Kicker(side, HEAD_ZOOM["output_at"].format(token=show_tok(t[last])), bg="panel").pack(anchor="w", pady=(GUTTER, 4))
        for i in np.argsort(d["weights"][:, last])[::-1][:5]:
            w = float(d["weights"][i, last])
            r = tk.Frame(side, bg=C["panel"])
            r.pack(fill="x")
            tk_label(r, f"{show_tok(t[i])}", 13, "text", "panel", width=10).pack(side="left")
            bar = tk.Canvas(r, width=120, height=10, bg=C["panel2"], highlightthickness=0)
            bar.create_rectangle(0, 0, 120 * w, 10, fill=C["text2"], width=0)
            bar.pack(side="left", padx=6)
            tk_label(r, f"{w:.2f} · {round(w * 100)}%", 12, "muted", "panel").pack(side="left")
        tk_label(side, HEAD_ZOOM["result_is"].format(dim=d["d_head"], token=show_tok(t[int(np.argmax(d["weights"][:, last]))]), d=self.forward["hooks"]["embed"].shape[-1]), 12, "muted", "panel", wraplength=300).pack(anchor="w", pady=(6, 0))
        link = tk_label(side, HEAD_ZOOM["where_meet"] + " ›", 13, "violet", "panel", cursor="hand2", takefocus=1)
        link.pack(anchor="w", pady=(6, 0))
        for seq in ("<Button-1>", "<Return>", "<space>"):
            link.bind(seq, lambda _e: self.navigate("value_map"))
