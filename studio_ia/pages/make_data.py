"""Make extra data: problems a rule can check (safe), or local-model drafts that go to review (never straight to training).

Talks only to services (generate_problems, make_drafts, draft_count). Decision logic is in plain functions at the top.
"""
from __future__ import annotations

import random
import tkinter as tk
from tkinter import ttk

import studio_rules as rules  # rule names only
from studio_ia.copy import MAKEDATA as T
from studio_ia.pages.base import Page
from studio_ia.theme import C, GUTTER, entry, font, tk_label
from studio_ia.widgets import Kicker, _activate, tip

MAX_N = 100000
# Which card makes which data kind, how it is described, and how it is labelled for safety.
CARDS = (
    dict(key="problems", makes="problems", safe_color="blue", name=T["problems_rule"], note=T["problems_note"], makes_line=T["problems_makes"],
         safe=T["problems_safe"], example=T["problems_example"], where=T["problems_where"], go=T["problems_go"]),
    dict(key="rewrite", makes="conversations", safe_color="amber", name=T["rewrite_model"], note=T["rewrite_note"], makes_line=T["rewrite_makes"],
         safe=T["rewrite_safe"], example=T["rewrite_example"], where=T["rewrite_where"], go=T["rewrite_go"]),
    dict(key="prose", makes="text", safe_color="amber", name=T["prose_model"], note=T["prose_note"], makes_line=T["prose_makes"],
         safe=T["prose_safe"], example=T["prose_example"], where=T["prose_where"], go=T["prose_go"]),
)


def card_for(kind):
    """Index of the card that makes `kind`, 0 if unknown."""
    return next((i for i, c in enumerate(CARDS) if c["makes"] == kind), 0)


def stores_directly(card):
    """Only rule-made problems skip review: the rule decides what is right. Local-model output is always a draft."""
    return card["makes"] == "problems"


def parse_count(text):
    """A whole number 1..MAX_N, else None."""
    try:
        n = int(str(text).strip())
    except ValueError:
        return None
    return n if 1 <= n <= MAX_N else None


def preview_lines(report):
    """Lines of the 'Try 5 first' preview: each prompt with its answer, then k of n answers passing their own rule."""
    lines = [f"{i['prompt']}  →  {i.get('answer', T['rule_decides'])}" for i in report["items"][:5]]
    if report["checked"]:
        lines.append(T["checked"].format(passed=report["passed"], checked=report["checked"]))
    return [T["preview"], *lines]


def problems_summary(report):
    return T["added_problems"].format(added=report["added"]) + (T["dup_skipped"].format(dup=report["duplicates"]) if report["duplicates"] else ".")


def drafts_summary(kind, added):
    return T["drafts_waiting"].format(added=added, kind=kind.title())


class MakeDataPage(Page):
    title = T["h1"]
    lead = T["lead"]

    def __init__(self, parent, services, navigate=None, seed=None, **kw):
        super().__init__(parent, services, navigate, **kw)
        self.rng, self.k = random.Random(seed), 0
        body = self.body
        row = tk.Frame(body, bg=C["bg"])
        row.pack(fill="x")
        self.cards = []
        for i, card in enumerate(CARDS):
            box = tk.Frame(row, bg=C["panel"], takefocus=1, cursor="hand2", highlightthickness=1, highlightbackground=C["border"],
                           highlightcolor=C["violet"], padx=GUTTER, pady=GUTTER)
            box.pack(side="left", fill="both", expand=True, padx=(0, 8))
            labels = [tk_label(box, card["name"], 15, "text", "panel", font=font(15, bold=True), wraplength=250),
                      tk_label(box, card["makes_line"], 13, "text2", "panel"),
                      tk_label(box, card["safe"], 13, card["safe_color"], "panel", wraplength=250)]
            for w in labels:
                w.pack(fill="x", pady=2)
            _activate(box, lambda i=i: self.pick(i))
            for w in labels:
                w.bind("<Button-1>", lambda _e, i=i: self.pick(i))
            tip(box, card["note"])
            self.cards.append((box, labels))
        Kicker(body, T["kicker"]).pack(fill="x", pady=(GUTTER, 4))
        self.example = tk_label(body, "", 14, "text", "panel", font=font(14, mono=True), padx=GUTTER, pady=10)
        self.example.pack(fill="x")
        self.where_tip = ""
        ctl = tk.Frame(body, bg=C["bg"])
        ctl.pack(fill="x", pady=(GUTTER, 0))
        tk_label(ctl, T["how_many"], 14, "text").pack(side="left")
        self.n = tk.StringVar(value="100")
        entry(ctl, 14, textvariable=self.n, width=8).pack(side="left", fill="y", padx=8)
        self.rule = ttk.Combobox(ctl, values=[T["all_rules"], *(rules.LABELS[r] for r in rules.RULES)], state="readonly", width=22)
        self.rule.set(T["all_rules"])
        self.rule.pack(side="left", fill="y", padx=8)
        self.go = ttk.Button(ctl, style="Primary.TButton", command=lambda: self.run(trial=False))
        self.go.pack(side="left", padx=4)
        tip(self.go, lambda: self.where_tip)
        self.try5 = ttk.Button(ctl, text=T["try_5"], command=lambda: self.run(trial=True))
        self.try5.pack(side="left", padx=4)
        self.review = ttk.Button(ctl, command=lambda: self.open_data(CARDS[self.k]["makes"]))
        self.review.pack(side="right", padx=4)
        ttk.Button(ctl, text=T["your_data"], command=lambda: self.open_data(None)).pack(side="right", padx=4)
        self.status = tk_label(body, "", 14, "text2", wraplength=760)
        self.status.pack(fill="x", pady=GUTTER)
        self.show(CARDS[0]["makes"])

    def show(self, kind):
        """Select the card that makes `kind` (links from Your data call this)."""
        self.k = card_for(kind)
        self.status.configure(text="")
        self.refresh()

    def pick(self, i):
        self.show(CARDS[i]["makes"])

    def open_data(self, kind):
        page = self.navigate("your_data")
        if page is not None and kind:
            page.show(kind)

    def refresh(self):
        card = CARDS[self.k]
        for i, (box, labels) in enumerate(self.cards):
            on = i == self.k
            bg = C["panel2"] if on else C["panel"]
            box.configure(bg=bg, highlightbackground=C["violet"] if on else C["border"])
            for w in labels:
                w.configure(bg=bg)
        self.example.configure(text=card["example"])
        self.where_tip = card["where"]
        self.go.configure(text=card["go"])
        self.rule.configure(state="readonly" if stores_directly(card) else "disabled")
        self.try5.configure(state="normal" if stores_directly(card) else "disabled")
        self.review.configure(text=T["review_drafts"].format(count=self.services.draft_count()))

    def run(self, trial=False):
        """Problems: Try 5 first previews without storing; Generate stores them. Local model: always stored as drafts."""
        card = CARDS[self.k]
        n = 5 if trial else parse_count(self.n.get())
        if n is None:
            self.status.configure(text=T["bad_count"])
            return
        if stores_directly(card):
            chosen = next((r for r in rules.RULES if rules.LABELS[r] == self.rule.get()), None)

            def work():
                report = self.services.generate_problems(chosen, n, store=not trial, rng=self.rng)
                return "\n".join(preview_lines(report)) if trial else problems_summary(report)
            title = "Making problems"
        else:
            def work():
                self.services.note(f"Asking the local model for {n:,} drafts")
                try:
                    return drafts_summary(card["makes"], self.services.make_drafts(card["makes"], n))
                except RuntimeError as error:
                    return T["draft_failed"].format(why=error)
            title = "Writing drafts"
        self.background(work, self._made, title)

    def _made(self, text):
        self.status.configure(text=text)
        self.refresh()
