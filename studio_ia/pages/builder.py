"""Build your model: the picture is the editor. Draft edits live on this page until saved."""
from __future__ import annotations

import copy
import os
import tkinter as tk
import uuid

from studio_spec import validate_spec
from studio_versions import _facts, spec_diff
from studio_warmstart import NOTES
from studio_ia.copy import BUILDER as T
from studio_ia.pages.base import Page
from studio_ia.services import Job
from studio_ia.theme import C, GUTTER, font, tk_label
from studio_ia.widgets import BpcPlot, Card, Chip, Kicker, StateTray, button

FF = (2, 4, 8)  # thinking-room width as a multiple of d_model: narrow / normal / wide
TOKENIZERS = ("byte", "bpe_custom", "bpe_gpt2")
VS = (256, 8192, 50257)  # word-list sizes behind letters / small / large
WORDS = ("Letters", "Small word list", "Large word list")
WIDTHS = ("Narrow", "Normal", "Wide")
HEADS = (1, 4, 8)
PROMPT = "The largest planet in our solar system is"
LINKS = (("trace", "trace"), ("head_zoom", "inside_head"), ("value_map", "value_map"), ("loss", "loss"),
         ("head_lab", "head_lab"), ("inside3d", "3d"), ("compare", "compare_runs"), ("quicktest", "quick_test"))
EXPECT = dict(
    more="More blocks: each word gets more rounds of processing, and every training step takes longer. With little data it may memorize rather than learn; watch the test line.",
    fewer="Fewer blocks: smaller and faster. It may not get its error as low.",
    heads="Different look-back heads: same size, same speed. Each block splits its looking into {how} views. Whether that helps at this size is something to test, not assume.",
    wider="More thinking room: more to learn in every block, so bigger and slower.",
    narrower="Less thinking room: smaller and faster, with less capacity in every block.",
    letters="Letters: the word table becomes tiny, but the model has to learn spelling, and your text turns into about 5 times as many pieces, so each step covers less text.",
    small="Small word list: built from your own data. A much smaller table; text splits into somewhat more pieces.",
    large="Large word list: GPT-2’s list. A big table, short sequences.",
)


def cfg(s):
    """s = dict(L, H, W, V, d, T) -> sizes for the cost formulas."""
    return dict(V=VS[s["V"]], d=s["d"], T=s["T"], L=s["L"], H=s["H"], ff=FF[s["W"]] * s["d"])


def P(c):
    return c["V"] * c["d"] + c["T"] * c["d"] + c["L"] * (4 * c["d"] ** 2 + 4 * c["d"] + 2 * c["d"] * c["ff"] + c["ff"] + c["d"] + 4 * c["d"]) + 2 * c["d"]


def C_(c):
    return 6 * (c["L"] * (4 * c["d"] ** 2 + 2 * c["d"] * c["ff"]) + c["V"] * c["d"]) + 6 * c["L"] * c["T"] * c["d"]


def TM(c):
    return 16 * P(c)


def activations(c):
    return 2 * c["L"] * c["d"] * 4 * c["T"]


def say(x, y, more, less):
    r = y / x
    if abs(r - 1) < 0.05:
        return T["about_same"]
    return T["about_multiplier"].format(mult=f"{r if r > 1 else 1 / r:.1f}") + " " + (more if r > 1 else less)


def bar(x, y):
    m = max(x, y)
    return round(x / m * 100), round(y / m * 100)


def read_spec(spec):
    """ModelSpec -> the four editable choices plus d and T."""
    f = _facts(spec)
    word = {"letters": 0, "small word list": 1, "large word list": 2}.get(f["words"], 1)
    return dict(L=f["blocks"], H=f["heads"] or 4, W=0 if f["room"] < 4 else 1 if f["room"] == 4 else 2, V=word,
                d=f["width"], T=f["context"])


def write_spec(base, s):
    spec = copy.deepcopy(validate_spec(base))
    spec["dims"]["n_layer"] = s["L"]
    spec["block"]["mixer"]["config"]["heads"] = s["H"]
    spec["block"]["channel"]["config"]["d_ff"] = FF[s["W"]] * s["d"]
    spec["tokenizer"] = {"id": TOKENIZERS[s["V"]], "config": {}}
    return validate_spec(spec)


def fits(c):
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None
    return TM(c) + 16 * activations(c) <= ram  # 16 rows per step


class BuilderPage(Page):
    title, lead = T["h1"], T["lead"]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        from studio_spec import default_spec
        self.default = default_spec()
        self.phase, self.base, self.saved, self.job_id = "edit", None, None, None
        self.cur = read_spec(self.default)
        self.draft = dict(self.cur)
        self.tray = self.chips = self.costs = self.expect = self.note = None
        self.chip_texts, self.cost_say, self.expect_texts = [], {}, []

    # state ------------------------------------------------------------------
    @property
    def n(self):
        return self.base.number if self.base else 0

    @property
    def nv(self):
        return f"version {self.n + 1}"

    def refresh(self):
        if self.phase == "edit":
            base = self.services.current_version()
            changed = (base.number if base else 0) != self.n
            self.base = base
            self.cur = read_spec(base.spec) if base and base.spec else read_spec(self.default)
            if changed:
                self.draft = dict(self.cur)
        self.render()

    def on_event(self, kind, payload):
        if kind == "queue" and self.phase == "queued":
            q = self.services.queue()
            if q.job and q.job.id == self.job_id:
                self.phase = "training"
        if kind in ("queue", "versions"):
            self.render()

    def edit(self, **patch):
        if self.phase == "edit":
            self.draft.update(patch)
            self.render()

    def spec_for(self, s):
        return write_spec(self.base.spec if self.base and self.base.spec else self.default, s)

    def changes(self):
        try:
            return spec_diff(self.spec_for(self.cur), self.spec_for(self.draft))
        except ValueError:
            return []

    def word_changed(self):
        return self.draft["V"] != self.cur["V"]

    def start_note(self):
        a, b = self.cur, self.draft
        if self.word_changed():
            return NOTES["scratch"]
        if b["H"] != a["H"]:
            return NOTES["heads"]
        if b["L"] < a["L"] or b["W"] < a["W"]:
            return NOTES["removed"]
        return NOTES["same"]

    def last_blocks_change(self):
        """(from, to, v_from, v_to, bpc_moved|None) for the latest block-count change in the history, else None."""
        vs = [v for v in self.services.versions() if v.spec]
        for i in range(len(vs) - 1, 0, -1):
            a, b = _facts(vs[i - 1].spec)["blocks"], _facts(vs[i].spec)["blocks"]
            if a != b:
                x, y = vs[i - 1].scorecard.get("bpc"), vs[i].scorecard.get("bpc")
                moved = f"{abs(y - x):.2f}" if isinstance(x, (int, float)) and isinstance(y, (int, float)) else "[N]"
                return a, b, vs[i - 1].number, vs[i].number, moved
        return None

    def expectations(self):
        a, b, out = self.cur, self.draft, []
        if b["L"] > a["L"]:
            out.append(EXPECT["more"])
        if b["L"] < a["L"]:
            out.append(EXPECT["fewer"])
        if b["L"] != a["L"] and (last := self.last_blocks_change()):
            out.append(f"Last time blocks changed ({last[0]} → {last[1]}, version {last[2]} → {last[3]}), error moved by {last[4]} bits per character.")
        if b["H"] != a["H"]:
            out.append(EXPECT["heads"].format(how="more, narrower" if b["H"] > a["H"] else "fewer, wider"))
        if b["W"] > a["W"]:
            out.append(EXPECT["wider"])
        if b["W"] < a["W"]:
            out.append(EXPECT["narrower"])
        if self.word_changed():
            out.append(EXPECT[("letters", "small", "large")[b["V"]]])
        return out or [T["change_something"]]

    # actions ----------------------------------------------------------------
    def discard(self):
        self.draft = dict(self.cur)
        self.render()

    def save(self):
        try:
            spec = self.spec_for(self.draft)
        except ValueError as error:
            self.note = str(error)
            return self.render()
        self.saved = self.services.save_version(spec, ", ".join(self.changes()) or "no change")
        self.phase = "saved"
        self.render()

    def queue_it(self, from_scratch):
        scratch = from_scratch or self.word_changed()
        self.job_id = uuid.uuid4().hex[:8]
        self.services.enqueue(Job(self.job_id, "pretrain", f"Version {self.n + 1} {'training' if scratch else 'catch-up'}",
                                  params={"version": self.n + 1, "mode": "from_scratch" if scratch else "catch_up"}))
        self.phase = "queued"
        self.render()

    def to_front(self):
        self.services.move_up(self.job_id)
        self.phase = "training"
        self.render()

    def new_changes(self):
        self.base = self.services.current_version() or self.saved
        self.cur = dict(self.draft)
        self.phase = "edit"
        self.render()

    # view -------------------------------------------------------------------
    def render(self):
        for child in self.body.winfo_children():
            child.destroy()
        self.body.columnconfigure(0, weight=1)
        a, b = cfg(self.cur), cfg(self.draft)
        self._pill()
        top = tk.Frame(self.body, bg=C["bg"])
        top.pack(fill="x", pady=(0, GUTTER))
        left = tk.Frame(top, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0, GUTTER))
        right = tk.Frame(top, bg=C["bg"], width=340)
        right.pack(side="left", fill="y")
        self._tower(left)
        self._controls(right)
        self._costs(a, b)
        self._expect()
        self._tray()
        self._charts()
        self._look_inside()

    def _pill(self):
        job = self.services.queue().job
        text = T["run_pill"].format(version=f"Version {self.n}", status="is training") if job else \
            (f"Version {self.n} is {self.base.status}" if self.base else "No saved version yet")
        Chip(self.body, text, "training" if job else "neutral").pack(anchor="w", pady=(0, GUTTER))

    def _tower(self, parent):
        tk_label(parent, f"{T['next_word']}  ·  {T['next_word_desc']}", 13, "muted").pack(anchor="w")
        cur, dr, editing = self.cur, self.draft, self.phase == "edit"
        for i in range(max(cur["L"], dr["L"]) - 1, -1, -1):
            added, removed = i >= cur["L"], i >= dr["L"]
            tag = "new" if added else "removed" if removed else ("changed" if (dr["H"] != cur["H"] or dr["W"] != cur["W"]) else "")
            row = tk.Frame(parent, bg=C["amber_bg"] if added else C["panel2"], highlightthickness=1,
                           highlightbackground=C["amber"] if added else C["border"])
            row.pack(fill="x", pady=2)
            tk_label(row, f"Block {i + 1}", 13, "dim" if removed else "text", row["bg"], width=9).pack(side="left", padx=8, pady=8)
            for _ in range(cur["H"] if removed else dr["H"]):
                tk_label(row, "●", 13, "pink", row["bg"]).pack(side="left")
            tk.Frame(row, bg=C["blue"], width=[36, 72, 144][cur["W"] if removed else dr["W"]], height=8).pack(side="left", padx=10)
            if tag:
                tk_label(row, tag, 12, "amber", row["bg"]).pack(side="right", padx=8)
        step = tk.Frame(parent, bg=C["bg"])
        step.pack(anchor="w", pady=8)
        tk_label(step, f"{T['blocks']}  ", 14, "text", "bg").pack(side="left")
        self.minus = button(step, "−", lambda: self.edit(L=max(1, dr["L"] - 1)))
        self.plus = button(step, "+", lambda: self.edit(L=min(12, dr["L"] + 1)))
        self.minus.pack(side="left")
        self.blocks_label = tk_label(step, str(dr["L"]), 18, "amber" if dr["L"] != cur["L"] else "text", "bg", width=3, anchor="center")
        self.blocks_label.pack(side="left")
        self.plus.pack(side="left")
        if dr["L"] != cur["L"]:
            tk_label(step, f"  was {cur['L']}", 13, "muted", "bg").pack(side="left")
        tk_label(step, f"  {T['blocks_desc']}", 12, "dim", "bg").pack(side="left")
        for b in (self.minus, self.plus):
            b.state(["!disabled"] if editing else ["disabled"])
        tk_label(parent, T["your_text_title"], 13, "muted").pack(anchor="w", pady=(8, 2))
        chips = tk.Frame(parent, bg=C["bg"])
        chips.pack(anchor="w", fill="x")
        v = dr["V"]
        pieces = [("␣" if c == " " else c) for c in PROMPT] if v == 0 else \
            [" ".join(PROMPT.split()[:1])] + [" " + w for w in PROMPT.split()[1:]] if v == 2 else ["pieces depend on your data"]
        for p in pieces:
            Chip(chips, p, "pieces").pack(side="left", padx=1, pady=1)
        count = "41 pieces, one per character" if v == 0 else "8 pieces" if v == 2 else "built from your data when you save"
        tk_label(parent, count, 12, "muted").pack(anchor="w")
        tk_label(parent, T["look_back_heads_gloss"], 12, "pink").pack(anchor="w", pady=(8, 0))
        tk_label(parent, T["thinking_room_gloss"], 12, "blue").pack(anchor="w")

    def _seg(self, parent, options, now, was, key, values):
        row = tk.Frame(parent, bg=C["panel"])
        row.pack(anchor="w", pady=4)
        for opt, val in zip(options, values):
            col = tk.Frame(row, bg=C["panel"])
            col.pack(side="left", padx=(0, 4))
            on = self.draft[key] == val
            lab = tk.Label(col, text=opt, font=font(14), padx=12, pady=8, takefocus=1, cursor="hand2", highlightthickness=1,
                           bg=C["violet"] if on else C["panel2"], fg=C["bg"] if on else C["text"],
                           highlightbackground=C["violet"] if (was == val and not on) else C["border"])
            lab.pack()
            for seq in ("<Button-1>", "<Return>", "<space>"):
                lab.bind(seq, lambda _e, v=val: self.edit(**{key: v}))
            # ponytail: Tk borders cannot be dashed, so the old choice gets a violet outline and a small "was".
            tk_label(col, "was" if (was == val and not on) else " ", 11, "violet", "panel", anchor="center").pack()

    def _controls(self, parent):
        for name, hint, color, opts, key, values in (
                ("Look-back heads", "how many things each block looks for at once", "pink", [str(h) for h in HEADS], "H", HEADS),
                ("Thinking room", "space to work on each word", "blue", list(WIDTHS), "W", (0, 1, 2)),
                ("Word list", T["your_text_title"], "sage", list(WORDS), "V", (0, 1, 2))):
            card = Card(parent)
            card.pack(fill="x", pady=(0, 8))
            tk_label(card, name, 15, color, "panel", font=font(15, bold=True)).pack(anchor="w")
            tk_label(card, hint, 12, "muted", "panel").pack(anchor="w")
            self._seg(card, opts, self.draft[key], self.cur[key], key, values)

    def _costs(self, a, b):
        card = Card(self.body)
        card.pack(fill="x", pady=(0, GUTTER))
        Kicker(card, T["what_costs"], bg="panel").pack(anchor="w")
        self.cost_say = {}
        for name, f, more, less in ((T["size_label"], P, T["cost_bigger"], T["cost_smaller"]),
                                    (T["time_per_step"], C_, T["cost_slower"], T["cost_faster"]),
                                    (T["memory_while_training"], TM, T["cost_more"], T["cost_less"])):
            x, y = f(a), f(b)
            s = say(x, y, more, less)
            self.cost_say[name] = s
            same = s == T["about_same"]
            row = tk.Frame(card, bg=C["panel"])
            row.pack(fill="x", pady=4)
            tk_label(row, name, 14, "text", "panel", width=24).pack(side="left")
            bars = tk.Frame(row, bg=C["panel"])
            bars.pack(side="left", fill="x", expand=True)
            for label, pct, color in (("now", bar(x, y)[0], "dim"), ("new", bar(x, y)[1], "violet" if same else "amber")):
                line = tk.Frame(bars, bg=C["panel"])
                line.pack(fill="x")
                tk_label(line, label, 11, "dim", "panel", width=4).pack(side="left")
                tk.Frame(line, bg=C[color], width=max(2, pct * 3), height=7).pack(side="left", pady=1)
            tk_label(row, s, 13, "muted" if same else "amber", "panel", width=22).pack(side="left")
        ok = fits(b)
        verdict = T["training_memory_fits"] if ok is None else ("fits" if ok else "doesn't fit")
        self.fits_text = f"{T['training_memory']} {verdict} {T['training_memory_desc']}"
        tk_label(card, self.fits_text, 12, "muted", "panel", wraplength=780).pack(anchor="w", pady=(6, 0))

    def _expect(self):
        card = Card(self.body)
        card.pack(fill="x", pady=(0, GUTTER))
        Kicker(card, T["what_expect"], bg="panel").pack(anchor="w")
        self.expect_texts = self.expectations()
        for x in self.expect_texts:
            tk_label(card, x, 14, "text2", "panel", wraplength=780).pack(anchor="w", pady=2)

    def _tray(self):
        ph, n, nv = self.phase, self.n, self.nv
        Nv = nv[0].upper() + nv[1:]
        scratch = self.word_changed()
        self.chip_texts = []
        actions = []
        if ph == "edit":
            self.chip_texts = self.changes()
            if self.chip_texts:
                state, kicker, text = "unsaved", T["not_saved"], T["not_saved_text"].format(version=nv)
                actions = [(T["quick_test_first"], lambda: self.navigate("quicktest"), False),
                           (f"Save as {nv}", self.save, True), (T["discard"], self.discard, False)]
            else:
                state, kicker, text = "neutral", T["no_changes"], T["no_changes_text"].format(version=n)
        elif ph == "saved":
            state = "needs_training"
            kicker = T["saved_text_needs_training" if scratch else "saved_text_needs_catchup"].format(new_version=nv)
            keeps = T["saved_text"].format(startNote=self.start_note(), current_version=f"version {n}", new_version=nv) if n else self.start_note()
            text = keeps
            actions = [(T["add_training" if scratch else "add_catchup"], lambda: self.queue_it(False), True)]
            if not scratch:
                actions.append((T["start_from_scratch"], lambda: self.queue_it(True), False))
        elif ph == "queued":
            state, kicker, text = "live", T["queued"], T["queued_text"].format(new_version=Nv)
            actions = [(T["move_to_front"], self.to_front, True)]
        else:
            state = "training"
            kicker = T["training"].format(new_version=nv, status="IS TRAINING" if scratch else "IS CATCHING UP")
            text = T["training_text"].format(current=n, new_version=nv)
            actions = [(T["start_new_changes"], self.new_changes, True)]
        if self.note:
            text += " " + self.note
            self.note = None
        self.tray = StateTray(self.body, state, kicker, text)
        self.tray.pack(fill="x", pady=(0, 8))
        self.tray_kicker = kicker
        self.tray_text = text
        row = tk.Frame(self.body, bg=C["bg"])
        row.pack(fill="x", pady=(0, GUTTER))
        self.chips = tk.Frame(row, bg=C["bg"])
        self.chips.pack(anchor="w", pady=(0, 6))
        for c in self.chip_texts:
            Chip(self.chips, c, "unsaved").pack(side="left", padx=(0, 6))
        self.actions = {}
        for label, cmd, primary in actions:
            self.actions[label] = btn = button(row, label, cmd, primary)
            btn.pack(side="left", padx=(0, 8))

    def _charts(self):
        card = Card(self.body)
        card.pack(fill="x", pady=(0, GUTTER))
        Kicker(card, T["while_training"], bg="panel").pack(anchor="w")
        tk_label(card, f"{T['example_shape']}  {T['version_label'].format(version=self.n)}", 12, "dim", "panel").pack(anchor="w")
        self.plot = BpcPlot(card, 640, 200)
        self.plot.pack(anchor="w", pady=6)
        Kicker(card, T["when_answering"], bg="panel").pack(anchor="w", pady=(8, 0))
        tk_label(card, T["example_prompt"], 14, "text", "panel").pack(anchor="w")
        for who in (f"version {self.n}", self.nv):
            tk_label(card, f"{who}: {T['its_answer_appears']}", 13, "muted", "panel", font=font(13, serif=True)).pack(anchor="w")
        tk_label(card, T["answering_note"], 12, "dim", "panel", wraplength=780).pack(anchor="w", pady=(6, 0))

    def _look_inside(self):
        Kicker(self.body, T["look_inside"]).pack(anchor="w")
        row = tk.Frame(self.body, bg=C["bg"])
        row.pack(anchor="w", pady=6)
        self.links = {}
        for route, key in LINKS:
            link = tk_label(row, T[key], 14, "violet", cursor="hand2", takefocus=1)
            link.pack(side="left", padx=(0, 16))
            self.links[route] = link
            for seq in ("<Button-1>", "<Return>", "<space>"):
                link.bind(seq, lambda _e, r=route: self.navigate(r))
