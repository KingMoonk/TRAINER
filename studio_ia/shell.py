"""Studio window: sidebar (four areas + machine strip) and the page router."""
from __future__ import annotations

import tkinter as tk
from collections import namedtuple
from tkinter import ttk

from studio_ux import format_duration
from studio_ia import services as services_module
from studio_ia.pages.base import Page
from studio_ia.theme import C, GUTTER, SIDEBAR_W, apply, font, tk_label
from studio_ia.widgets import Kicker, Progress

Route = namedtuple("Route", "area title lead crumb")

AREAS = [
    ("designer", "Network designer", "Shape the model, then see what it costs."),
    ("pretraining", "Pretraining", "Feed it text; it learns to predict the next word."),
    ("post", "Post-training", "Teach it how to answer."),
    ("chat", "Chat", "Talk to any trained version."),
]
AREA_HOME = {"designer": "builder", "pretraining": "pretraining", "post": "post_training", "chat": "chat"}
LI = ("Network designer", "Look inside")

# route -> Route(area, h1, lead, breadcrumb). area None = reached from the machine strip.
ROUTES = {
    "builder": Route("designer", "Build your model", "Change the picture. See what it costs and what it does.", ()),
    "quicktest": Route("designer", "Quick test", "Try a design idea for a few minutes on a tiny dataset, before you save it and train it for real.", ("Network designer",)),
    "compare": Route("designer", "Compare runs", "Change one thing, hold the rest equal, run several seeds. A difference counts only if it is larger than the spread across seeds.", ("Network designer",)),
    "queue": Route(None, "This computer", "What it’s training now, what’s waiting, and when it’s allowed to work. Every “Train” or “Run” button in the app adds to this queue.", ()),
    "trace": Route("designer", "Trace a position", "Follow one position's stream through the model. Numbers appear when the engine has run your prompt.", (*LI, "Trace a position")),
    "head_zoom": Route("designer", "Inside a head", "One attention head, one step at a time, on a real prompt.", (*LI, "Inside a head")),
    "value_map": Route("designer", "Value map", "How a head's values are projected down and back up, and why the heads add up.", (*LI, "Value map")),
    "loss": Route("designer", "Loss", "What one wrong or right prediction costs, and the gradient it sends back.", (*LI, "Loss")),
    "head_lab": Route("designer", "Head lab", "Where does each attention head look, and how much does the model rely on it? Switch heads off and watch the answer change.", (*LI, "Head lab")),
    "inside3d": Route("designer", "Inside the network", "Build the model from the inside and watch it train, from a window of positions down to single weights.", (*LI, "3D")),
    "pretraining": Route("pretraining", "Pretraining", "Feed it text. It learns by predicting the next word, over and over.", ()),
    "training_run": Route("pretraining", "Pretraining · version [N]", "Step k of n, the schedule phase and the error so far.", ()),
    "your_data": Route("pretraining", "Your data", "Three kinds, because each one trains a different stage.", ()),
    "make_data": Route("pretraining", "Make extra data", "Optional. Generate more, then keep what passes review.", ()),
    "post_training": Route("post", "Post-training", "Four rounds, always in this order. Same design throughout; only the weights shift.", ()),
    "chat": Route("chat", "Chat", "Talk to any trained version. Compare two to see what a change really did.", ()),
}


def _data_page(cls):
    """Adapt a data page (workspace-based constructor) to the shell's (parent, services, navigate) factory."""
    def make(parent, services, navigate):
        def go(route, kind=None, **_):
            page = navigate(route)
            if kind and hasattr(page, "show"):
                page.show(kind)
        page = cls(parent, services=services, on_navigate=go)
        page.on_event = lambda kind, payload: None
        return page
    return make


class Studio(tk.Tk):
    def __init__(self, services=None):
        super().__init__()
        self.title("Groundwork Studio")
        self.geometry("1280x820")
        self.minsize(900, 600)
        apply(self)
        self.services = services or services_module.FakeServices()
        self.services.bind_ui(self)
        self.pages: dict[str, Page] = {}
        self.factories: dict[str, type] = {}  # route -> Page subclass; page builders register here
        from studio_ia.pages.builder import BuilderPage
        from studio_ia.pages.chat import ChatPage
        self.factories["builder"], self.factories["chat"] = BuilderPage, ChatPage
        from studio_ia.pages.pretraining import PretrainingPage
        from studio_ia.pages.training_run import TrainingRunPage
        self.factories["pretraining"], self.factories["training_run"] = PretrainingPage, TrainingRunPage
        from studio_ia.pages.post_training import PostTrainingPage
        self.factories["post_training"] = PostTrainingPage
        from studio_ia.pages.compare import ComparePage
        from studio_ia.pages.quicktest import QuickTestPage
        self.factories["compare"], self.factories["quicktest"] = ComparePage, QuickTestPage
        from studio_ia.pages.queue import QueuePage
        self.factories["queue"] = QueuePage
        if self.services.workspace is not None:  # the data pages read the workspace directly; the fake façade has none
            from studio_ia.pages.make_data import MakeDataPage
            from studio_ia.pages.your_data import YourDataPage
            self.factories["your_data"], self.factories["make_data"] = _data_page(YourDataPage), _data_page(MakeDataPage)
        self.route = None
        self._area_rows = {}
        self._build_sidebar()
        self.main = ttk.Frame(self)
        self.main.pack(side="left", fill="both", expand=True)
        from studio_ia.pages import head_lab, head_zoom, loss, trace, value_map  # look-inside pages (S7)
        for route, cls in (("trace", trace.TracePage), ("head_zoom", head_zoom.HeadZoomPage), ("value_map", value_map.ValueMapPage),
                           ("loss", loss.LossPage), ("head_lab", head_lab.HeadLabPage)):
            self.register(route, cls)
        self.unsubscribe = self.services.subscribe(self.on_event)
        self.refresh_strip()
        self.show("builder")

    # sidebar ----------------------------------------------------------------
    def _build_sidebar(self):
        side = tk.Frame(self, bg=C["panel"], width=SIDEBAR_W, highlightthickness=1, highlightbackground=C["border2"])
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        tk_label(side, "Groundwork Studio", 17, "text", "panel", font=font(17, bold=True)).pack(fill="x", padx=GUTTER, pady=GUTTER)
        for key, name, gloss in AREAS:
            row = tk.Frame(side, bg=C["panel"], takefocus=1, cursor="hand2", highlightthickness=1,
                           highlightbackground=C["panel"], highlightcolor=C["violet"])
            row.pack(fill="x")
            stripe = tk.Frame(row, bg=C["panel"], width=3)
            stripe.pack(side="left", fill="y")
            inner = tk.Frame(row, bg=C["panel"])
            inner.pack(side="left", fill="x", expand=True, padx=(GUTTER - 3, GUTTER), pady=10)
            title = tk_label(inner, name, 15, "text", "panel", font=font(15, bold=True))
            sub = tk_label(inner, gloss, 12, "muted", "panel", wraplength=SIDEBAR_W - 2 * GUTTER - 3)
            title.pack(fill="x")
            sub.pack(fill="x")
            for w in (row, stripe, inner, title, sub):
                w.bind("<Button-1>", lambda _e, k=key: self.show(AREA_HOME[k]))
            for seq in ("<Return>", "<space>"):
                row.bind(seq, lambda _e, k=key: self.show(AREA_HOME[k]))
            self._area_rows[key] = (row, stripe, inner, title, sub)
        # machine strip
        tk.Frame(side, bg=C["border2"], height=1).pack(side="bottom", fill="x")
        strip = tk.Frame(side, bg=C["panel"])
        strip.pack(side="bottom", fill="x", padx=GUTTER, pady=GUTTER)
        Kicker(strip, "This computer", bg="panel").pack(fill="x")
        self.strip_now = tk_label(strip, "", 13, "text", "panel", wraplength=SIDEBAR_W - 2 * GUTTER)
        self.strip_bar = Progress(strip)
        self.strip_next = tk_label(strip, "", 12, "muted", "panel", wraplength=SIDEBAR_W - 2 * GUTTER)
        self.strip_chat = tk_label(strip, "", 12, "muted", "panel", wraplength=SIDEBAR_W - 2 * GUTTER)
        self.strip_link = tk_label(strip, "Open the queue ›", 12, "violet", "panel", cursor="hand2", takefocus=1,
                                   highlightthickness=1, highlightbackground=C["panel"], highlightcolor=C["violet"])
        self.strip_now.pack(fill="x", pady=(6, 4))
        self.strip_bar.pack(fill="x")
        self.strip_next.pack(fill="x", pady=(4, 0))
        self.strip_chat.pack(fill="x")
        self.strip_link.pack(fill="x", pady=(6, 0))
        for seq in ("<Button-1>", "<Return>", "<space>"):
            self.strip_link.bind(seq, lambda _e: self.show("queue"))

    @property
    def strip_text(self):
        return "\n".join(w.cget("text") for w in (self.strip_now, self.strip_next, self.strip_chat))

    def refresh_strip(self):
        q = self.services.queue()
        waiting = len(q.up_next)
        if q.job is None:
            self.strip_now.configure(text="Nothing is training")
            self.strip_bar.set(None)
            left = ""
        else:
            state = "paused" if q.paused else (q.phase or "training")
            self.strip_now.configure(text=f"{q.job.label or q.job.kind} · {state}")
            self.strip_bar.set(q.step / q.total if q.total else None)
            left = format_duration(q.eta_s) + " left · " if q.eta_s is not None else "[N] min left · "
        self.strip_next.configure(text=f"{left}{waiting} queued")
        try:
            v = self.services.best_version() or self.services.current_version()
        except NotImplementedError:
            v = None
        self.strip_chat.configure(text=f"Chat uses version {v.number} · best so far" if v else "Chat has no trained version yet")

    def _highlight(self, area):
        for key, (row, stripe, inner, title, sub) in self._area_rows.items():
            on = key == area
            bg = C["violet_bg"] if on else C["panel"]
            stripe.configure(bg=C["violet"] if on else C["panel"])
            for w in (row, inner, title, sub):
                w.configure(bg=bg)
            row.configure(highlightbackground=bg)

    # router -----------------------------------------------------------------
    def register(self, route, page_class):
        self.factories[route] = page_class
        old = self.pages.pop(route, None)
        if old is not None:
            old.destroy()

    def show(self, route):
        if route not in ROUTES:
            raise KeyError(route)
        spec = ROUTES[route]
        if route not in self.pages:
            cls = self.factories.get(route)
            if cls is not None:
                page = cls(self.main, self.services, self.show)
            else:
                page = Page(self.main, self.services, self.show, title=spec.title, lead=spec.lead, crumb=spec.crumb)
            self.pages[route] = page
        if self.route:
            self.pages[self.route].pack_forget()
        self.route = route
        page = self.pages[route]
        page.pack(fill="both", expand=True)
        self._highlight(spec.area)
        page.refresh()
        return page

    def on_event(self, kind, payload):
        if kind == "queue":
            self.refresh_strip()
        if self.route:
            self.pages[self.route].on_event(kind, payload)

    def destroy(self):
        self.unsubscribe()
        super().destroy()


def main():
    services = None
    make = getattr(services_module, "WorkspaceServices", None)  # provided once the real façade lands
    if make is not None:
        try:
            from studio_core import Workspace
            services = make(Workspace())  # the repo-root workspace, as studio_gui builds it
        except Exception:
            services = None
    Studio(services).mainloop()


if __name__ == "__main__":
    main()
