"""Loss: one prediction, top-5 plus 'all others', p, the one-hot target, p - y, cross-entropy in nats and the bits-per-character card."""
from __future__ import annotations

import math
import tkinter as tk

import numpy as np

from studio_ia.copy import LOSS
from studio_ia.pages.base import Page
from studio_ia.pages.trace import LI, ask, prompt_text, show_tok
from studio_ia.theme import C, GUTTER, font, tk_label
from studio_ia.widgets import Card, Kicker, Segmented

LN2 = math.log(2)
UNTRAINED, TRAINED = LOSS["step_0"], LOSS["after_training"]


def fmt_p(p):
    return f"{p:.1e}" if p < .001 else f"{p:.2f}"


def prompt_with_target():
    """The board's example plus the piece to predict: the model sees everything before the last piece."""
    return f"{prompt_text()} Jupiter"


def loss_rows(p, target, pieces, k=5):
    """The k most likely pieces plus the target if it is not among them (target first), then 'all others': (token, p, y, p - y) rows."""
    p = np.asarray(p, dtype=np.float64)
    top = [int(i) for i in np.argsort(-p)[:k]]
    order = [i for i in top if i != target][:k - 1 if target in top else k]
    rows = [(pieces[i], float(p[i]), 1 if i == target else 0, float(p[i]) - (1 if i == target else 0)) for i in [target, *order]]
    rest = 1.0 - sum(r[1] for r in rows)
    return rows, max(0.0, rest), len(p) - len(rows)


class LossPage(Page):
    title = LOSS["h1"]
    lead = LOSS["lead"]
    crumb = (*LI, "Loss")

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.trained = True

    def pick(self, label):
        self.trained = label == TRAINED
        self.refresh()

    def refresh(self):
        for child in self.body.winfo_children():
            child.destroy()
        data, error = ask(self.services, "inspect_loss", prompt_with_target(), trained=self.trained)
        toggle = Segmented(self.body, [UNTRAINED, TRAINED], TRAINED if self.trained else UNTRAINED, command=self.pick, bg="bg")
        toggle.pack(anchor="w", pady=(0, GUTTER))
        if data is None:
            tk_label(self.body, error, 14, "muted", wraplength=720).pack(anchor="w")
            return
        p, target, pieces = np.asarray(data["probabilities"], dtype=np.float64), int(data["target"]), data["pieces"]
        V = len(p)
        token = show_tok(pieces[target])
        context = "".join(data["tokens"][:data["pos"]]) if data["pos"] >= 0 else ""
        self.loss = -math.log(max(float(p[target]), 1e-300))
        card = Card(self.body)
        card.pack(fill="x")
        tk_label(card, f"…{context[-24:]} ?", 16, "text2", "panel").pack(anchor="w")
        table = tk.Frame(card, bg=C["panel"])
        table.pack(fill="x", pady=8)
        for c, text in enumerate((LOSS["token"], LOSS["probability"], "", LOSS["target"], LOSS["gradient"])):
            tk_label(table, text, 12, "muted", "panel").grid(row=0, column=c, sticky="w", padx=(0, GUTTER))
        rows, rest, n_rest = loss_rows(p, target, pieces)
        self.table = []
        for r, (tok, prob, y, g) in enumerate(rows, 1):
            self._row(table, r, show_tok(tok), prob, str(y), ("−" if g < 0 else "+") + fmt_p(abs(g)), y)
        self._row(table, len(rows) + 1, f"all {n_rest:,} others", rest, "0 each", "+p each", 0)
        tk_label(card, f"Softmax over all {V:,} tokens. The target is one-hot: 1 for “{token}”, 0 for everything else. "
                       "The gradient on each logit is simply p − target.", 13, "muted", "panel", wraplength=760).pack(anchor="w")
        if data.get("grad_error") is not None:
            tk_label(card, f"Checked against the engine's own backward pass: largest difference {data['grad_error']:.1e}.", 12, "dim", "panel").pack(anchor="w", pady=(4, 0))
        ce = Card(self.body)
        ce.pack(fill="x", pady=GUTTER)
        Kicker(ce, LOSS["cross_entropy"], bg="panel").pack(anchor="w")
        tk_label(ce, f"{LOSS['loss_eq'].format(token=token)}{LOSS['loss_eq_end'].format(p=fmt_p(float(p[target])))}", 15, "text", "panel").pack(anchor="w", pady=4)
        self.loss_label = tk_label(ce, f"{self.loss:.2f} {LOSS['nats']}", 28, "text" if self.trained else "amber", "panel", font=font(28, bold=True))
        self.loss_label.pack(anchor="w")
        note = ("Falls as the model puts more probability on the target token. Measured by the engine on the trained checkpoint." if self.trained else
                f"Random weights spread probability almost evenly, about 1 / {V:,} per token, so the loss starts near ln {V:,} = {math.log(V):.2f} (measured here: {self.loss:.2f}).")
        tk_label(ce, note, 13, "muted", "panel", wraplength=760).pack(anchor="w", pady=(4, 0))
        bpc = Card(self.body)
        bpc.pack(fill="x")
        Kicker(bpc, LOSS["why_bits"], bg="panel").pack(anchor="w")
        tk_label(bpc, LOSS["nats_depend"], 13, "text2", "panel", wraplength=760).pack(anchor="w", pady=4)
        tk_label(bpc, LOSS["bpc_formula"], 13, "text", "panel", font=font(13, mono=True), wraplength=760).pack(anchor="w")
        chars = max(1, len(pieces[target]))
        self.bpc = self.loss / (chars * LN2)
        tk_label(bpc, f"This piece, “{token}”, is {chars} character{'s' * (chars != 1)}: {self.loss:.2f} ÷ ({chars} × {LN2:.3f}) = {self.bpc:.2f} {LOSS['bpc_result']}",
                 13, "text2", "panel", wraplength=760).pack(anchor="w", pady=(6, 0))

    def _row(self, table, r, tok, prob, y, g, hot):
        cells = [tok, fmt_p(prob), "", y, g]
        self.table.append(cells)
        color = "amber" if hot else "text"
        for c, text in enumerate(cells):
            if c == 2:
                bar = tk.Canvas(table, width=140, height=12, bg=C["panel2"], highlightthickness=0)
                bar.create_rectangle(0, 0, max(1, 140 * prob), 12, fill=C["amber"] if hot else C["blue"], width=0)
                bar.grid(row=r, column=c, padx=(0, GUTTER))
            else:
                tk_label(table, text, 14, color, "panel", font=font(14, mono=c > 0)).grid(row=r, column=c, sticky="w", padx=(0, GUTTER))
