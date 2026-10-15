"""Page base class: standard header (breadcrumb, h1, lead) over a scrolling body."""
from __future__ import annotations

from tkinter import ttk

from studio_widgets import ScrollPage
from studio_ia.theme import C, GUTTER, tk_label
from studio_ia.widgets import _activate


class Throttle:
    """One pass per `interval` seconds: ready(now) is True (and starts the next interval) or False; wait(now) is the time left."""

    def __init__(self, interval=1.0):
        self.interval, self.last = interval, None

    def ready(self, now):
        if self.last is None or now - self.last >= self.interval:
            self.last = now
            return True
        return False

    def wait(self, now):
        return 0.0 if self.last is None else max(0.0, self.interval - (now - self.last))


class Page(ttk.Frame):
    """Subclass and set `title`, `lead`, optional `crumb` (tuple of path parts after the leading "‹ ")."""

    title = ""
    lead = ""
    crumb: tuple = ()

    def __init__(self, parent, services, navigate=None, title=None, lead=None, crumb=None):
        super().__init__(parent)
        self.services = services
        self.navigate = navigate or (lambda route: None)
        self.title = title if title is not None else self.title
        self.lead = lead if lead is not None else self.lead
        self.crumb = tuple(crumb) if crumb is not None else self.crumb
        scroll = ScrollPage(self)
        scroll.canvas.configure(bg=C["bg"])
        scroll.pack(fill="both", expand=True)
        outer = ttk.Frame(scroll.body)
        outer.pack(fill="both", expand=True, padx=2 * GUTTER, pady=2 * GUTTER)
        self.header = ttk.Frame(outer)
        self.header.pack(fill="x")
        if self.crumb:
            link = tk_label(self.header, "‹ " + " / ".join(self.crumb), 13, "muted", cursor="hand2", takefocus=1)
            link.pack(anchor="w", pady=(0, GUTTER // 2))
            _activate(link, lambda: self.navigate("builder"))
        self.h1 = ttk.Label(self.header, text=self.title, style="H1.TLabel")
        self.h1.pack(anchor="w")
        self.lead_label = ttk.Label(self.header, text=self.lead, style="Lead.TLabel", wraplength=760, justify="left")
        self.lead_label.pack(anchor="w", pady=(GUTTER // 2, 0))
        self.body = ttk.Frame(outer)
        self.body.pack(fill="both", expand=True, pady=(2 * GUTTER, 0))

    def refresh(self):
        """Re-read services and redraw. Called when the page is shown."""

    def preload(self):
        """The slow service reads refresh() is about to make, and nothing else: it can run off the Tk thread, so no widgets."""

    def reload(self):
        """refresh() after the stored data may have changed. The shell swaps in a version that loads behind its panel."""
        self.refresh()

    def quietly(self, work, then):
        """then(work()) for a read the page can show without: the shell runs it on its worker with no loading panel."""
        then(work())

    def quiet_reload(self):
        """reload() for data that changed in the background: the shell re-reads it without covering the page."""
        self.reload()

    def slow(self, work, then):
        """then(work()) for a service read that may be long. The shell swaps in a version that keeps the Tk thread free."""
        then(work())

    def background(self, work, then, title=""):
        """then(work()) for work that is always long or changes data (a local-model call, an import): the shell
        swaps in a version that runs it on its worker, never on the Tk thread. work() touches no widgets."""
        then(work())

    def on_event(self, kind, payload):
        """Services event (queue, versions, ...), delivered on the Tk thread while the page is shown."""
