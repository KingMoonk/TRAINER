"""Benchmark: the same checks on every trained version, one chart per check, tokens trained on across and the score up.
Services run the checks (studio_bench) and keep their results; this page only asks for them and draws what comes back."""
from __future__ import annotations

import tkinter as tk

import studio_bench
from studio_ia.copy import BENCH as T
from studio_ia.pages.base import Page
from studio_ia.services import plain_error
from studio_ia.theme import C, GUTTER, tk_label
from studio_ia.widgets import BpcPlot, Card, Kicker, button

LINES = ("violet", "amber", "blue", "pink", "sage", "text")  # one colour per design, in the order the designs first appear
CHARTS = (("bpc", "bpc_title", "bpc_desc", "what_bpc"), ("few_shot", "few_shot_title", "few_shot_desc", "what_few_shot"),
          ("problems", "problems_title", "problems_desc", "what_problems"))


def tokens_text(n):
    return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K" if n >= 1e3 else f"{n:.0f}"


def value_text(metric, value):
    """Bits per character with two decimals; the other two checks as a share."""
    return f"{value:.2f}" if metric == "bpc" else f"{value:.0%}"


def designs(rows):
    """{lineage: (letter, [version numbers])} for the trained versions, designs lettered in the order their first version appears."""
    out = {}
    for row in sorted(rows, key=lambda r: r["number"]):
        letter, numbers = out.setdefault(row["lineage"], (chr(ord("A") + len(out)), []))
        numbers.append(row["number"])
    return out


def pending(state, tried=()):
    """Version numbers whose suite result is missing, oldest first, leaving out the ones already tried this session."""
    if not state or not any(state["sizes"].values()):
        return []
    return [r["number"] for r in sorted(state["rows"], key=lambda r: r["number"]) if r["result"] is None and r["number"] not in tried]


def has_items(state, metric):
    """Whether the suite holds anything for this check (documents for bits per character, questions, problems)."""
    return bool(state["sizes"]["docs" if metric == "bpc" else "choice" if metric == "few_shot" else "problems"])


def plot_args(state, metric):
    """What BpcPlot.set takes for one chart: a line with a dot per version for each design, and for the multiple-choice
    chart a dashed line at what guessing gets. None when no version has a number for this check."""
    chart = studio_bench.chart_data(state["rows"], metric)
    names = list(designs(state["rows"]))
    series = [{"points": line["points"], "color": LINES[names.index(line["lineage"]) % len(LINES)], "width": 2, "dash": None, "dots": True}
              for line in chart["lines"]]
    if not series:
        return None, chart
    if chart["chance"] is not None:
        last = max(p[0] for s in series for p in s["points"])
        series.append({"points": [(0, chart["chance"]), (last, chart["chance"])], "color": "muted", "width": 1, "dash": (4, 3)})
    return {"series": series, "fmt": lambda v: value_text(metric, v), "x_fmt": tokens_text}, chart


def status_text(state):
    """One line under the header: how many trained versions still need checking, or that none do."""
    rows = state["rows"]
    if not rows:
        return T["no_versions"]
    if not any(state["sizes"].values()):
        return T["no_data"]
    left = sum(1 for r in rows if r["result"] is None)
    return T["waiting"].format(n=left, total=len(rows)) if left else T["all_done"]


def check_versions(services, numbers):
    """Run the suite on each version in turn, saying which one on the loading panel. Returns {number: plain error} for the
    ones that failed; the others carry on. Slow: call it off the Tk thread."""
    failed = {}
    for done, number in enumerate(numbers):
        services.note(T["running"].format(number=number, done=done + 1, total=len(numbers)), done / len(numbers))
        try:
            services.run_benchmark(number)
        except Exception as error:
            failed[number] = plain_error(error)
    return failed


class BenchmarkPage(Page):
    title = T["h1"]
    lead = T["lead"]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.state, self.tried, self.failed = None, set(), {}  # tried: versions checked since this page opened; a failure is not retried on its own
        self.top = Card(self.body)
        self.top.pack(fill="x", pady=(0, GUTTER))
        self.charts = tk.Frame(self.body, bg=C["bg"])
        self.charts.pack(fill="x")

    def refresh(self):
        self.slow(self.services.benchmark, self._show)

    def _show(self, state):
        self.state = state
        self.draw()
        todo = pending(state, self.tried)
        if todo:
            self._check(todo)

    def on_event(self, kind, payload):
        if kind == "versions":  # a version finished training: it needs its numbers
            self.refresh()

    def _check(self, numbers):
        self.tried.update(numbers)

        def done(failed):
            self.failed = failed
            self.refresh()
        self.background(lambda: check_versions(self.services, numbers), done, T["checking"])

    def retry(self):
        self.tried.clear()
        self.refresh()

    def draw(self):
        state = self.state
        for child in self.top.winfo_children() + self.charts.winfo_children():
            child.destroy()
        sizes = state["sizes"]
        if any(sizes.values()):
            tk_label(self.top, T["fixed"].format(docs=sizes["docs"], questions=sizes["choice"], problems=sizes["problems"]), 13, "muted", "panel",
                     wraplength=780).pack(anchor="w")
        line = tk_label(self.top, status_text(state), 14, "text2", "panel", wraplength=780)
        line.pack(anchor="w", pady=(6, 0))
        for number, error in self.failed.items():
            tk_label(self.top, T["failed"].format(number=number, error=error), 13, "amber", "panel", wraplength=780).pack(anchor="w", pady=(4, 0))
        if sum(state["dropped"].values()):
            tk_label(self.top, T["dropped"].format(n=sum(state["dropped"].values())), 13, "muted", "panel", wraplength=780).pack(anchor="w", pady=(4, 0))
        if pending(state):
            button(self.top, T["run"], self.retry).pack(anchor="w", pady=(GUTTER, 0))
        if not state["rows"] or not any(sizes.values()):
            return
        for metric, title, desc, what in CHARTS:
            self._chart(state, metric, T[title], T[desc].format(shots=studio_bench.SHOTS[-1]), T[what])

    def _chart(self, state, metric, title, desc, what):
        card = Card(self.charts)
        card.pack(fill="x", pady=(0, GUTTER))
        Kicker(card, title, bg="panel").pack(anchor="w")
        tk_label(card, desc, 13, "muted", "panel", wraplength=780).pack(anchor="w", pady=(4, 0))
        args, chart = plot_args(state, metric) if has_items(state, metric) else (None, None)
        if args is None:
            tk_label(card, T["empty_metric"] if chart else T["no_metric_data"].format(what=what), 13, "dim", "panel", wraplength=780).pack(anchor="w", pady=6)
            return
        legend = tk.Frame(card, bg=C["panel"])
        legend.pack(anchor="w", pady=6)
        names = designs(state["rows"])
        for line in chart["lines"]:
            letter, numbers = names[line["lineage"]]
            color = LINES[list(names).index(line["lineage"]) % len(LINES)]
            tk_label(legend, T["design"].format(letter=letter, numbers=", ".join(map(str, numbers))), 13, color, "panel").pack(side="left", padx=(0, GUTTER))
        if chart["chance"] is not None:
            tk_label(legend, T["chance"].format(chance=value_text(metric, chart["chance"])), 13, "muted", "panel").pack(side="left")
        plot = BpcPlot(card, 720, 180, 0)
        plot.pack(anchor="w", pady=4)
        plot.set(**args)
        tk_label(card, T["axis"], 12, "muted", "panel").pack(anchor="w")
        if chart["unplaced"]:
            tk_label(card, T["unplaced"].format(numbers=", ".join(map(str, chart["unplaced"]))), 12, "amber", "panel", wraplength=780).pack(anchor="w")
