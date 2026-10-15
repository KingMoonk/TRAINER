"""Head lab: heads as tiles on prepared text, rank them by importance, switch some off, test their attention patterns."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path

import numpy as np

import studio_inspect
from studio_ia import threads
from studio_ia.copy import HEAD_LAB
from studio_ia.pages.base import Page
from studio_ia.pages.trace import LI, ask, current, prompt_text, show_tok
from studio_ia.theme import C, GUTTER, font, tk_label
from studio_ia.widgets import Card, GuessChips, HeatGrid, HowItWorks, Kicker, Segmented, _activate, button

TESTS = (("previous", "previous_word", "previous_desc"), ("first", "parks_first", "parks_desc"), ("induction", "copying_pattern", "copying_desc"))
THRESHOLD = .5  # a share of attention above this counts as "passes"; a description of where it looks, not a verdict
GUESSES = (HEAD_LAB["lot"], HEAD_LAB["little"], HEAD_LAB["not_at_all"])
MODES = {HEAD_LAB["mode_mean"]: "mean", HEAD_LAB["mode_zero"]: "zero"}


def best_pattern(values, threshold=THRESHOLD):
    """Name of the test a head scores highest on ('previous' | 'first' | 'induction') or None if none reaches the threshold.

    A description of where the head looks, never of what it does."""
    name = max(TESTS, key=lambda t: values[t[0]])[0]
    return name if values[name] >= threshold else None


PATTERN_WORDS = {"previous": "looks at the previous word", "first": "parks on the first word", "induction": "copying pattern"}
NO_PATTERN = "no clear pattern"


def head_pattern(stats, head):
    """Words for where one head looks, from head_tests() output ({test: per-head scores}); NO_PATTERN when no test reaches the threshold."""
    return PATTERN_WORDS.get(best_pattern({t[0]: float(stats[t[0]][head]) for t in TESTS}), NO_PATTERN)


def top_relied(rises, k=3):
    """The k heads whose switch-off raises the error most: [((block, head), rise)] largest first."""
    return sorted(rises.items(), key=lambda kv: -kv[1])[:k]


class HeadLabPage(Page):
    title = HEAD_LAB["h1"]
    lead = HEAD_LAB["lead"]
    crumb = (*LI, "Head lab")

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.sel, self.off, self.rises, self.tested, self.ranked = None, set(), None, set(), False
        self.guess, self.changed, self._token, self.mode, self.notice = None, None, 0, "mean", None

    # data -------------------------------------------------------------------
    def refresh(self):
        for child in self.body.winfo_children():
            child.destroy()
        prepared, error = ask(self.services, "inspect_prepared")
        data = None
        if prepared is not None:
            data, error = ask(self.services, "inspect_forward", prepared["ids"])
        if data is None:
            tk_label(self.body, error, 14, "muted", wraplength=720).pack(anchor="w")
            return
        self.tokens = prepared["tokens"]
        self.patterns = {int(n.split(".")[0][5:]): np.asarray(v)[0:1] for n, v in data["hooks"].items() if n.endswith("attn.pattern")}
        self.stats = {b: studio_inspect.head_tests(p, prepared["ids"]) for b, p in self.patterns.items()}
        heads = [(b, h) for b, p in self.patterns.items() for h in range(p.shape[1])]
        if not heads:
            tk_label(self.body, "This model has no attention heads to experiment on.", 14, "muted").pack(anchor="w")
            return
        if self.sel not in heads:
            self.sel = heads[0]
        self.n_heads = len(heads)
        self._draw()

    def select(self, block, head):
        self.sel = (block, head)

    # drawing ----------------------------------------------------------------
    def _draw(self):
        for child in self.body.winfo_children():
            child.destroy()
        version = current(self.services)
        if version and version.best_checkpoint:
            step = self.services.checkpoint_step(version)
            ckpt = f"at step {step:,}" if step else Path(version.best_checkpoint).name
            intro = HEAD_LAB["checkpoint"].format(version=version.number, checkpoint=ckpt)
        else:
            intro = HEAD_LAB["no_checkpoint"]
        tk_label(self.body, intro, 13, "muted", wraplength=760).pack(anchor="w")
        top = tk.Frame(self.body, bg=C["bg"])
        top.pack(fill="x", pady=(GUTTER, 4))
        Kicker(top, f"{self.n_heads} HEADS · CLICK ONE").pack(side="left")
        tk_label(self.body, HEAD_LAB["heads_note"], 13, "muted", wraplength=760).pack(anchor="w")
        bar = tk.Frame(self.body, bg=C["bg"])
        bar.pack(fill="x", pady=8)
        self.rank_button = button(bar, HEAD_LAB["hide_importance"] if self.ranked else HEAD_LAB["rank_heads"], self.rank, primary=True)
        self.rank_button.pack(side="left")
        if self.off:
            button(bar, HEAD_LAB["switch_all_on"], self.all_on).pack(side="left", padx=8)
        tk_label(bar, HEAD_LAB["mode_label"], 13, "muted").pack(side="left", padx=(GUTTER, 6))
        label = next(k for k, v in MODES.items() if v == self.mode)
        Segmented(bar, list(MODES), label, command=lambda v: self.set_mode(MODES[v]), bg="bg").pack(side="left")
        if self.notice:
            tk_label(self.body, self.notice, 13, "amber", wraplength=760).pack(anchor="w")
        self.tiles = {}
        grid = tk.Frame(self.body, bg=C["bg"])
        grid.pack(fill="x")
        for row, (b, p) in enumerate(self.patterns.items()):
            tk_label(grid, f"Block {b + 1}", 13, "muted").grid(row=row, column=0, sticky="w", padx=(0, GUTTER))
            for h in range(p.shape[1]):
                self._tile(grid, row, b, h, p[0, h])
        tk_label(self.body, HEAD_LAB["grid_note_ranked"] if self.ranked else HEAD_LAB["grid_note_normal"], 13, "muted", wraplength=760).pack(anchor="w", pady=8)
        if self.ranked and self.rises:
            Kicker(self.body, HEAD_LAB["most_relied"]).pack(anchor="w")
            for (b, h), rise in top_relied(self.rises):
                tk_label(self.body, f"Block {b + 1} · head {h + 1}   " + HEAD_LAB["rank_unit"].format(rise=rise), 13, "text2", font=font(13, mono=True)).pack(anchor="w")
        self._selected()
        self._changed()
        how = HowItWorks(self.body, HEAD_LAB["how_switching"], bg="bg")
        how.pack(anchor="w", fill="x", pady=GUTTER)
        for head, text in ((HEAD_LAB["switching_note"], HEAD_LAB["replaces_output"]), (HEAD_LAB["ranking_note"], HEAD_LAB["switches_each"]),
                           (HEAD_LAB["tests_note"], HEAD_LAB["measure_attention"])):
            tk_label(how.body, f"{head} {text}", 13, "muted", "bg", wraplength=760).pack(anchor="w", pady=2)
        tk_label(how.body, HEAD_LAB["mean_exact"], 13, "muted", "bg", wraplength=760).pack(anchor="w", pady=2)
        link = tk_label(how.body, HEAD_LAB["step_through"] + " ›", 13, "violet", "bg", cursor="hand2", takefocus=1)
        link.pack(anchor="w", pady=2)
        _activate(link, self._open_zoom)

    def _open_zoom(self):
        page = self.navigate("head_zoom")
        if page is not None and hasattr(page, "select"):
            page.select(*self.sel)

    def _tile_sub(self, key):
        if key in self.off:
            return HEAD_LAB["head_off"], "amber"
        if self.ranked and self.rises and key in self.rises:
            return HEAD_LAB["rank_unit"].format(rise=self.rises[key]), "text2"
        return head_pattern(self.stats[key[0]], key[1]), "text2"  # from the stats refresh() already has: no button first

    def _tile(self, grid, row, b, h, pattern):
        key = (b, h)
        sub, color = self._tile_sub(key)
        T = pattern.shape[0]
        on = key == self.sel
        frame = tk.Frame(grid, bg=C["panel2"] if on else C["bg"], highlightthickness=2 if on else 1,
                         highlightbackground=C["violet"] if on else C["amber"] if key in self.off else C["border"], takefocus=1, cursor="hand2", padx=6, pady=4)
        frame.grid(row=row, column=h + 1, padx=4, pady=4)
        heat = HeatGrid(frame, bg="panel2" if on else "bg")
        heat.CELL = 9
        cells = [[None if k > q else (0.0 if key in self.off else float(min(1.0, pattern[q, k] * 1.5))) for k in range(T)] for q in range(T)]
        heat.set(cells)
        heat.pack()
        tk_label(frame, f"Head {h + 1}", 12, "text", "panel2" if on else "bg").pack(anchor="w")
        tk_label(frame, sub, 11, color, "panel2" if on else "bg").pack(anchor="w")
        self.tiles[key] = frame
        _activate(frame, lambda k=key: self._pick(k))
        for w in frame.winfo_children():
            w.bind("<Button-1>", lambda _e, k=key: self._pick(k))

    def _pick(self, key):
        self.sel = key
        self._draw()

    def _selected(self):
        b, h = self.sel
        card = Card(self.body)
        card.pack(fill="x", pady=(0, GUTTER))
        off = self.sel in self.off
        tk_label(card, f"Block {b + 1} · head {h + 1}", 18, "text", "panel", font=font(18, bold=True)).pack(anchor="w")
        tk_label(card, HEAD_LAB["head_off"] if off else HEAD_LAB["head_on"], 13, "amber" if off else "muted", "panel").pack(anchor="w")
        big = tk.Frame(card, bg=C["panel"])
        big.pack(anchor="w", pady=8)
        heat = HeatGrid(big)
        heat.CELL = 22
        T = len(self.tokens)
        pattern = self.patterns[b][0, h]
        heat.set([[None if k > q else float(min(1.0, pattern[q, k] * 1.5)) for k in range(T)] for q in range(T)], [show_tok(t)[:6] for t in self.tokens])
        heat.pack(side="left")
        tk_label(card, HEAD_LAB["prepared_text"], 12, "muted", "panel", wraplength=640).pack(anchor="w")
        if not off:
            tk_label(card, HEAD_LAB["before_switch"], 13, "text2", "panel").pack(anchor="w", pady=(8, 4))
            chips = GuessChips(card, GUESSES, command=lambda v: setattr(self, "guess", v))
            chips.pack(anchor="w")
            if self.guess:
                chips.set(self.guess)
        row = tk.Frame(card, bg=C["panel"])
        row.pack(anchor="w", pady=8)
        self.toggle_button = button(row, HEAD_LAB["switch_on"] if off else HEAD_LAB["switch_off"], self.flip, primary=off)
        self.toggle_button.pack(side="left")
        self.test_button = button(row, HEAD_LAB["test_pattern"], self.test)
        self.test_button.pack(side="left", padx=8)
        if self.sel in self.tested:
            self._tests(card)

    def _tests(self, card):
        b, h = self.sel
        Kicker(card, HEAD_LAB["three_tests"], bg="panel").pack(anchor="w", pady=(8, 4))
        for key, name, what in TESTS:
            value = float(self.stats[b][key][h])
            r = tk.Frame(card, bg=C["panel"])
            r.pack(fill="x", pady=2)
            tk_label(r, f"{HEAD_LAB[name]}", 14, "text", "panel", width=28).pack(side="left")
            tk_label(r, f"{value:.2f}  " + ("passes" if value >= THRESHOLD else "no"), 14, "sage" if value >= THRESHOLD else "muted", "panel", font=font(14, mono=True)).pack(side="left")
            tk_label(card, HEAD_LAB[what], 12, "muted", "panel", wraplength=640).pack(anchor="w")
        tk_label(card, "These describe where the head looks, not what it does with what it sees. Switching it off shows how much the model relies on it, not why.",
                 12, "dim", "panel", wraplength=640).pack(anchor="w", pady=(6, 0))

    def _changed(self):
        card = Card(self.body)
        card.pack(fill="x")
        n = len(self.off)
        Kicker(card, HEAD_LAB["what_changed"].format(count=n) + ("S" if n > 1 else "") if n else HEAD_LAB["what_changes"], bg="panel").pack(anchor="w")
        if not n:
            tk_label(card, HEAD_LAB["switch_head"], 13, "muted", "panel", wraplength=760).pack(anchor="w", pady=4)
            return
        tk_label(card, HEAD_LAB["example_prompt"], 14, "text2", "panel").pack(anchor="w", pady=4)
        c = self.changed or {}
        for label, reply in ((HEAD_LAB["all_heads_on"], c.get("on", "…")), (HEAD_LAB["with_heads_off"].format(count=n), c.get("off", "…"))):
            tk_label(card, label, 12, "muted", "panel").pack(anchor="w")
            tk_label(card, reply if reply is not None else "—", 14, "text", "panel", wraplength=760).pack(anchor="w", pady=(0, 6))
        bits = c.get("bits")
        text = HEAD_LAB["to_bits"] if not bits else f"{bits['before']:.2f} → {bits['after']:.2f} bits/char"
        tk_label(card, f"{HEAD_LAB['error_change']}: {text}", 14, "text", "panel", font=font(14, mono=True)).pack(anchor="w")
        tk_label(card, f"{HEAD_LAB['you_guessed']}: {self.guess.lower() if self.guess else 'no guess'}", 13, "pink", "panel").pack(anchor="w", pady=(4, 0))
        tk_label(card, HEAD_LAB["big_rise"], 12, "muted", "panel", wraplength=760).pack(anchor="w", pady=(6, 0))

    # actions ----------------------------------------------------------------
    def flip(self):
        self.off ^= {self.sel}
        self._update()

    def set_mode(self, mode):
        self.mode, self.ranked, self.rises = mode, False, None
        self._update()

    def all_on(self):
        self.off = set()
        self._update()

    def test(self):
        self.tested.add(self.sel)  # opens this head's card of the three tests
        self._draw()

    def _update(self):
        self.changed = None
        self._draw()
        if self.off:
            self._work(self._compute, self._finish)

    def _compute(self):
        version, off = current(self.services), sorted(self.off)
        out = {}
        for key, heads in (("on", ()), ("off", off)):
            try:
                out[key] = self.services.inspect_reply(version, prompt_text(), heads, self.mode)
            except (NotImplementedError, ValueError):
                out[key] = None
        try:
            out["bits"] = self.services.inspect_heads_off(version, off, self.mode)
        except (NotImplementedError, ValueError):
            out["bits"] = None
        return out

    def _finish(self, result):
        self.changed = result
        self._draw()

    def rank(self):
        if self.ranked:
            self.ranked = False
            self._draw()
        else:
            self.rank_button.configure(state="disabled")
            self._work(lambda: ask(self.services, "inspect_rank_heads", mode=self.mode), self._ranked)

    def _ranked(self, result):
        rows, error = result
        self.notice = error
        if rows is not None:
            self.rises = {(r["block"], r["head"]): float(r["rise"]) for r in rows}
            self.ranked = True
        self._draw()

    def _work(self, fn, done):
        """Run fn on a worker thread; results are picked up by polling so Tk is only touched from its own thread."""
        self._token += 1
        token, box = self._token, {}

        def run():
            box["value"] = fn()
        thread = threads.start(run)

        def poll():
            if thread.is_alive() and self.winfo_exists():
                self.after(60, poll)
                return
            if token == self._token and self.winfo_exists():
                done(box["value"])
        self.after(60, poll)
