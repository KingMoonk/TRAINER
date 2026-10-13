"""Trace a position: one row per stage of one position's stream (size, relative change, cosine, logit-lens guess)."""
from __future__ import annotations

from tkinter import ttk

from studio_ia.copy import TRACE
from studio_ia.pages.base import Page
from studio_ia.theme import GUTTER, font, tk_label
from studio_ia.widgets import Card, Kicker

LI = ("Network designer", "Look inside")
NO_NUMBERS = "Numbers appear when the engine has run your prompt."


def current(services):
    """The version Chat uses (best, else newest), or None."""
    try:
        return services.best_version() or services.current_version()
    except NotImplementedError:
        return None


def ask(services, method, *args, **kw):
    """(result, None) or (None, message): an inspect call that cannot run yet is a sentence, not a crash."""
    try:
        return getattr(services, method)(current(services), *args, **kw), None
    except (NotImplementedError, ValueError) as error:
        return None, str(error) if isinstance(error, ValueError) and str(error) else NO_NUMBERS


def show_tok(token):
    return str(token).replace(" ", "·").replace("\n", "⏎") or "·"


def prompt_text():
    return f"{TRACE['example_text']} {TRACE['last_position']}"


def num(value, spec="{:.2f}"):
    return "—" if value is None else spec.format(value)


class TracePage(Page):
    title = TRACE["h1"]
    lead = TRACE["lead"]
    crumb = (*LI, "Trace a position")

    def refresh(self):
        for child in self.body.winfo_children():
            child.destroy()
        data, error = ask(self.services, "inspect_trace", prompt_text())
        card = Card(self.body)
        card.pack(fill="x")
        Kicker(card, TRACE["example_prompt"], bg="panel").pack(anchor="w")
        if data is None:
            tk_label(card, error, 14, "muted", "panel", wraplength=720).pack(anchor="w", pady=(8, 0))
            return
        tokens = data["tokens"]
        tk_label(card, "".join(tokens[:-1]), 18, "text", "panel").pack(anchor="w", pady=(8, 0))
        tk_label(card, tokens[-1], 18, "amber", "panel", font=font(18, bold=True)).pack(anchor="w")
        table = ttk.Frame(self.body)
        table.pack(fill="x", pady=GUTTER)
        heads = (TRACE["stage"], TRACE["size_norm"], TRACE["relative_change"], TRACE["direction_kept"], TRACE["top_guess"])
        for c, text in enumerate(heads):
            tk_label(table, text, 12, "muted", wraplength=190).grid(row=0, column=c, sticky="w", padx=(0, GUTTER), pady=(0, 6))
        what = {"embedding": TRACE["embedding_what"], "mixer": "mixer adds", "channel": "MLP adds", "output": TRACE["output_what"]}
        self.cells = []
        for r, row in enumerate(data["rows"], 1):
            name = TRACE["embedding"] if row["kind"] == "embedding" else TRACE["output"] if row["kind"] == "output" else row["stage"]
            guess = f"“{row['top_token']}”  {row['top_p']:.2f}" + (" ≈" if row["approximate"] else "")
            cells = [name, num(row["norm"], "{:.2f}"), num(row["rel_change"]), num(row["cosine"]), guess]
            self.cells.append(cells)
            for c, text in enumerate(cells):
                color = "muted" if c == 4 and row["approximate"] else "text"
                tk_label(table, text, 14, color, font=font(14, mono=c > 0)).grid(row=2 * r, column=c, sticky="w", padx=(0, GUTTER))
            tk_label(table, what[row["kind"]], 12, "dim").grid(row=2 * r + 1, column=0, columnspan=5, sticky="w", pady=(0, 8))
        tk_label(self.body, TRACE["relative_note"], 13, "muted", wraplength=760).pack(anchor="w")
