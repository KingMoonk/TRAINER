"""Beginner / Advanced mode, plain-language helpers and time estimates (no engine behaviour lives here)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from studio_widgets import ACCENT, BG, EDGE, FONT, MUTED, PANEL, TEXT, GlossaryDialog, Tooltip, label

MODES = ("beginner", "advanced")

# Model sizes offered on the Training setup page: (name, dimensions, one-sentence explanation).
SIZE_PRESETS = [
    ("Micro", dict(width=32, layers=1, heads=2, context=128, expansion=2),
     "Tiny and quick (about a minute). Good for checking that everything works; it only learns simple patterns."),
    ("Small", dict(width=64, layers=2, heads=4, context=256, expansion=4),
     "The best first choice. Picks up clear patterns from a few thousand examples in a few minutes."),
    ("Medium", dict(width=128, layers=4, heads=4, context=512, expansion=4),
     "More capacity for bigger datasets, but several times slower. Best with a GPU."),
]
HIDDEN_FIELDS_NOTE = "Advanced mode shows every setting."


def device_kind(requested, gpu_info):
    """'cuda' when the run will use the GPU, otherwise 'cpu'."""
    if requested == "cpu":
        return "cpu"
    if gpu_info and gpu_info.get("available"):
        return "cuda"
    return "cpu"


def step_seconds(cfg, params, kind, scale=1.0):
    """Rough seconds for one optimizer step. Fitted to measurements of the NumPy CPU and CuPy CUDA backends."""
    batch, context, accumulation = int(cfg.get("batch_size", 4)), int(cfg.get("context", 256)), int(cfg.get("accumulation", 1))
    tokens = batch * context
    if kind == "cuda":
        seconds = accumulation * (0.04 + 0.05 * int(cfg.get("layers", 2))) * max(0.25, tokens / 1024) ** 0.3
    else:
        tokens_per_second = 4.0e6 / max(float(params), 1000.0) ** 0.57
        seconds = tokens * accumulation / tokens_per_second
    return seconds * scale


def estimate_seconds(cfg, params, kind, scale=1.0):
    return int(cfg.get("steps", 0)) * step_seconds(cfg, params, kind, scale)


def format_duration(seconds):
    if seconds < 45:
        return "under a minute"
    minutes = seconds / 60
    if minutes < 90:
        return f"about {max(1, round(minutes))} min"
    hours = minutes / 60
    return f"about {hours:.1f} hours" if hours < 10 else f"about {round(hours)} hours"


def plain_training_status(history, total=None, finished=False):
    """One plain-language sentence about how a run is going, from its metric events."""
    points = [h for h in history if h.get("loss") is not None]
    if not points:
        return "Not started yet. Press Start training on the Training setup page; the chart fills in as the model learns."
    first, last = float(points[0]["loss"]), float(points[-1]["loss"])
    vals = [float(h["val_loss"]) for h in history if h.get("val_loss") is not None]
    step = points[-1].get("step", 0)
    end = "Next: chat with your model." if finished else ""
    lead = "Finished. " if finished else ""
    if len(vals) >= 4 and vals[-1] > min(vals) * 1.05 and vals[-1] > vals[-2] and last < first:
        return (f"{lead}Careful: the blue test line is rising while the green line falls, so the model is starting to memorize. "
                f"Stop and save now, or add more data. {end}").strip()
    if len(points) < 4 or (total and step < max(5, total * 0.05)):
        return f"{lead}Warming up (step {step:,}). Early numbers jump around; wait a little before judging. {end}".strip()
    if last < first * 0.9 and (len(points) < 8 or last < float(points[len(points) * 2 // 3]["loss"]) * 0.985):
        return f"{lead}Learning well: the error fell from {first:.2f} to {last:.2f}. Let it run until the line flattens out. {end}".strip()
    return (f"{lead}The error has levelled off near {last:.2f}, so extra steps help less now. "
            f"Stopping is fine, or try more data or a bigger model. {end}").strip()


class UXMixin:
    """Beginner/Advanced switch, tooltips glue and the sidebar footer (mode toggle + glossary)."""

    def __init__(self, workspace=None):
        self.advanced_widgets = []     # shown only in Advanced mode
        self.beginner_widgets = []     # shown only in Beginner mode (hints that point at Advanced features)
        self.ui_mode = "beginner"
        super().__init__(workspace)
        saved = self.workspace.state.get("uiMode")
        self.ui_mode = saved if saved in MODES else "beginner"
        self.apply_ui_mode()

    # ------------------------------------------------------------------ registry
    def is_advanced(self):
        return self.ui_mode == "advanced"

    @staticmethod
    def _entry(widget, info):
        manager = widget.winfo_manager()
        if info:
            kind = manager or ("grid" if {"row", "column", "sticky"} & set(info) else "pack")
        else:
            kind = manager
        if kind == "pack":
            captured = widget.pack_info() if manager == "pack" else {}
            captured.pop("in", None)
            captured.update(info)
            siblings = widget.master.pack_slaves()
            index = siblings.index(widget) if widget in siblings else len(siblings)
            # Later siblings may not exist yet when this is called, so remember the earlier ones too.
            return {"widget": widget, "manager": "pack", "info": captured, "after": siblings[index + 1:], "before": siblings[:index]}
        if kind == "grid":
            captured = widget.grid_info() if manager == "grid" else {}
            captured.pop("in", None)
            captured.update(info)
            return {"widget": widget, "manager": "grid", "info": captured, "after": []}
        return {"widget": widget, "manager": "", "info": {}, "after": []}

    def mark_advanced(self, widget, **info):
        """Register `widget` (already pack/grid-managed) so it is hidden in Beginner mode. Returns the widget."""
        self.advanced_widgets.append(self._entry(widget, info))
        return widget

    def mark_beginner_only(self, widget, **info):
        self.beginner_widgets.append(self._entry(widget, info))
        return widget

    def _set_visible(self, entry, visible):
        widget = entry["widget"]
        try:
            if entry["manager"] == "pack":
                if visible and not widget.winfo_manager():
                    anchor = next((s for s in entry["after"] if s.winfo_exists() and s.winfo_manager() == "pack"), None)
                    options = dict(entry["info"])
                    previous = next((s for s in reversed(entry["before"]) if s.winfo_exists() and s.winfo_manager() == "pack"), None)
                    if anchor is not None:
                        options["before"] = anchor
                    elif previous is not None:
                        options["after"] = previous
                    widget.pack(**options)
                elif not visible and widget.winfo_manager():
                    widget.pack_forget()
            elif entry["manager"] == "grid":
                if visible and not widget.winfo_manager():
                    widget.grid(**entry["info"])
                elif not visible and widget.winfo_manager():
                    widget.grid_remove()
        except tk.TclError:
            pass

    def apply_ui_mode(self):
        advanced = self.is_advanced()
        for entry in self.advanced_widgets:
            self._set_visible(entry, advanced)
        for entry in self.beginner_widgets:
            self._set_visible(entry, not advanced)
        self.refresh_nav()
        if hasattr(self, "mode_buttons"):
            for mode, button_ in self.mode_buttons.items():
                button_.configure(style="ModeOn.TButton" if mode == self.ui_mode else "Mode.TButton")
        if hasattr(self, "refresh_next_label"):
            self.refresh_next_label()
        self.update_idletasks()

    def set_ui_mode(self, mode):
        if mode not in MODES:
            raise ValueError("Mode must be 'beginner' or 'advanced'.")
        self.ui_mode = mode
        self.workspace.state["uiMode"] = mode
        self.workspace.save()
        self.apply_ui_mode()
        if mode == "beginner" and getattr(self, "active_page", None) in self.hidden_pages():
            self.show_page("blueprint")
        self.status_var.set("Beginner mode: the essentials only." if mode == "beginner" else "Advanced mode: every setting is visible.")

    def hidden_pages(self):
        return set() if self.is_advanced() else {"network"}

    # ------------------------------------------------------------------ sidebar
    def _style(self):
        super()._style()
        style = ttk.Style(self)
        style.configure("Mode.TButton", background=PANEL, foreground=MUTED, padding=(8, 6), font=(FONT, 9))
        style.map("Mode.TButton", background=[("active", "#354763")])
        style.configure("ModeOn.TButton", background=ACCENT, foreground="#081c19", padding=(8, 6), font=(FONT, 9, "bold"))
        style.map("ModeOn.TButton", background=[("active", "#a0f0d8")])
        style.configure("Hint.TLabel", background=BG, foreground=ACCENT, font=(FONT, 10))

    def _build_sidebar_footer(self, footer):
        label(footer, "DETAIL LEVEL", foreground=ACCENT, font=(FONT, 9, "bold")).pack(anchor="w")
        row = ttk.Frame(footer)
        row.pack(fill="x", pady=(5, 8))
        self.mode_buttons = {}
        for mode, text, tip in (("beginner", "Beginner", "Shows only the essentials. Everything else is tucked away, not removed."),
                                ("advanced", "Advanced", "Shows every setting, including the Network designer and expert buttons.")):
            widget = ttk.Button(row, text=text, style="Mode.TButton", command=lambda m=mode: self.set_ui_mode(m))
            widget.pack(side="left", fill="x", expand=True, padx=(0, 4) if mode == "beginner" else 0)
            Tooltip(widget, tip)
            self.mode_buttons[mode] = widget
        glossary = ttk.Button(footer, text="?  Glossary", style="Mode.TButton", command=self.open_glossary)
        glossary.pack(fill="x", pady=(0, 10))
        Tooltip(glossary, "Plain-English meanings for every term used in the studio.")
        super()._build_sidebar_footer(footer)

    def open_glossary(self, term=None):
        dialog = GlossaryDialog(self, term)
        self.glossary_dialog = dialog
        return dialog

    def _layout(self):
        super()._layout()
        self.mark_advanced(self.nav["network"])

    def refresh_nav(self):
        """Sidebar numbers follow the visible pages, so Beginner mode reads 01-07 without a gap."""
        for key, *_ in self.PAGES:
            self.nav[key].configure(text=self.nav_label(key))
