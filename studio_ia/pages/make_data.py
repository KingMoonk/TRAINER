"""Make extra data: problems a rule can check (safe), or local-model drafts that go to review (never straight to training).

Plain ttk.Frame with refresh(). `drafter(kind, n) -> list[row]` is the local-model hook (None = no model connected);
rows are {"user","assistant"} for rewrites and {"text"} for prose. Drafts are stored with quality "Review".
ponytail: stand-alone until studio_ia.pages.base / widgets land; swap the Frame base for the shared Page then.
"""
import random
import tkinter as tk
from tkinter import ttk

import studio_data_kinds as dk
import studio_rules as rules

BG, PANEL, EDGE, TEXT, MUTED, VIOLET = "#0e1014", "#15181e", "#2a2f3a", "#e6e8ec", "#8f97a6", "#9f8cff"
KINDS = [
    dict(name="Problems a rule can check", note="Sort, reverse, count, copy, small addition, length limits, ending cleanly.",
         makes="problems", safe="safe: the rule decides what is right", color="#6cb4ee", go="Generate problems",
         example="Reverse: groundwork  →  rule: reverse", where="Goes to Your data → Problems. Used by Post-training round 2.",
         dataset="Generated from rules"),
    dict(name="Rewrite with your local model", makes="conversations", color="#e5b567", go="Make drafts",
         note="Rephrases prompts and answers you already have, so round 1 sees more variety. Uses your local model (Ollama).",
         safe="for review only: it can introduce mistakes", example="[your prompt, reworded]  →  [the same answer, reworded]",
         where="Goes to Your data → Conversations as drafts. Nothing is used until you mark it Good.", dataset="Rewritten conversations"),
    dict(name="Generate prose with your local model", makes="text", color="#e5b567", go="Make drafts",
         note="Short passages in a style or topic you pick.", safe="for review only: it can introduce mistakes",
         example="[a paragraph about tides, in plain English]", where="Goes to Your data → Text as drafts. Kept small next to real text.",
         dataset="Generated prose"),
]
ALL = "All rules"


class MakeDataPage(ttk.Frame):
    def __init__(self, master, workspace=None, services=None, drafter=None, on_navigate=None, seed=None):
        super().__init__(master)
        self.workspace = workspace or services.workspace
        self.drafter = drafter
        self.on_navigate = on_navigate or (lambda page, **kw: None)
        self.rng = random.Random(seed)
        self.k = 0
        self.configure(style="MD.TFrame")
        ttk.Style(self).configure("MD.TFrame", background=BG)
        tk.Label(self, text="Make extra data", bg=BG, fg=TEXT, font=("TkDefaultFont", 24, "bold")).pack(anchor="w", padx=16, pady=(14, 0))
        tk.Label(self, text="Optional. Generate more, then keep what passes review.", bg=BG, fg=MUTED).pack(anchor="w", padx=16)
        row = tk.Frame(self, bg=BG)
        row.pack(fill="x", padx=16, pady=8)
        self.cards = []
        for i, kind in enumerate(KINDS):
            card = tk.Button(row, justify="left", anchor="w", bd=0, padx=10, pady=8, wraplength=260, command=lambda i=i: self.pick(i),
                             highlightthickness=1, fg=TEXT, text=f"{kind['name']}\n{kind['note']}\nmakes: {kind['makes']}\n{kind['safe']}")
            card.pack(side="left", expand=True, fill="both", padx=3)
            self.cards.append(card)
        self.kicker = tk.Label(self, text="EXAMPLE OF WHAT IT MAKES", bg=BG, fg=MUTED, anchor="w")
        self.kicker.pack(fill="x", padx=16)
        self.example = tk.Label(self, bg=PANEL, fg=TEXT, anchor="w", justify="left", padx=10, pady=8)
        self.example.pack(fill="x", padx=16)
        self.where = tk.Label(self, bg=BG, fg=MUTED, anchor="w")
        self.where.pack(fill="x", padx=16, pady=4)
        ctl = tk.Frame(self, bg=BG)
        ctl.pack(fill="x", padx=16)
        tk.Label(ctl, text="How many", bg=BG, fg=TEXT).pack(side="left")
        self.n = tk.IntVar(value=100)
        tk.Spinbox(ctl, from_=1, to=100000, textvariable=self.n, width=8).pack(side="left", padx=6)
        self.rule = ttk.Combobox(ctl, values=[ALL, *rules.RULES], state="readonly", width=16)
        self.rule.set(ALL)
        self.rule.pack(side="left", padx=6)
        self.go = tk.Button(ctl, command=lambda: self.run(self.n.get()))
        self.go.pack(side="left", padx=3)
        tk.Button(ctl, text="Try 5 first", command=lambda: self.run(5, trial=True)).pack(side="left", padx=3)
        self.review = tk.Button(ctl, command=lambda: self.on_navigate("your_data", kind=KINDS[self.k]["makes"]))
        self.review.pack(side="right", padx=3)
        tk.Button(ctl, text="Your data", command=lambda: self.on_navigate("your_data")).pack(side="right", padx=3)
        self.status = tk.Label(self, bg=BG, fg=MUTED, anchor="w", justify="left")
        self.status.pack(fill="x", padx=16, pady=6)
        self.refresh()

    def pick(self, k):
        self.k = k
        self.refresh()

    def drafts(self):
        return sum(1 for d in self.workspace.state["datasets"] if d["name"] in {x["dataset"] for x in KINDS[1:]}
                   for e in self.workspace.iter_examples(d["id"]) if e.get("quality") == "Review")

    def refresh(self):
        kind = KINDS[self.k]
        for i, card in enumerate(self.cards):
            on = i == self.k
            card.configure(bg="#1b1f27" if on else BG, highlightbackground=VIOLET if on else EDGE)
        self.example.configure(text=kind["example"])
        self.where.configure(text=kind["where"])
        self.go.configure(text=kind["go"])
        self.rule.configure(state="readonly" if kind["makes"] == "problems" else "disabled")
        self.review.configure(text=f"Review drafts ({self.drafts()})")

    def run(self, n, trial=False):
        """Problems: Try 5 previews without storing; Generate stores them. Local model: always stored as drafts (Review)."""
        kind = KINDS[self.k]
        if n < 1:
            return
        if kind["makes"] == "problems":
            names = rules.RULES if self.rule.get() == ALL else (self.rule.get(),)
            items = [i for j, r in enumerate(names) for i in rules.generate(r, self.rng, n // len(names) + (j < n % len(names)))]
            if trial:
                self.status.configure(text="Preview, nothing saved:\n" + "\n".join(f"{i['prompt']}  →  {i.get('answer', '(rule decides)')}" for i in items[:5]))
                return
            added, dup = dk.add_problems(self.workspace, dk.ensure_dataset(self.workspace, kind["dataset"]), items)
            self.status.configure(text=f"Added {added} problems to Your data → Problems" + (f" ({dup} duplicates skipped)." if dup else "."))
        else:
            if self.drafter is None:
                self.status.configure(text="No local model connected. Set up Ollama in Post-training, then try again.")
                return
            rows = [{**r, "quality": "Review", "synthetic": True, "source": "synthetic"} for r in self.drafter(kind["makes"], n)]
            added = self.workspace.add(dk.ensure_dataset(self.workspace, kind["dataset"]), rows, "synthetic")[0]
            self.status.configure(text=f"{added} drafts waiting in Your data → {kind['makes'].title()} with status Review. Nothing is used until you mark it Good.")
        self.refresh()
