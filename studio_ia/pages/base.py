"""Page base class: standard header (breadcrumb, h1 and its ⓘ lead) over a scrolling body."""
from __future__ import annotations

from tkinter import ttk

from studio_widgets import ScrollPage
from studio_ia.theme import C, GUTTER, tk_label
from studio_ia.widgets import _activate, info_icon


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
        row = ttk.Frame(self.header)
        row.pack(anchor="w")
        self.h1 = ttk.Label(row, text=self.title, style="H1.TLabel")
        self.h1.pack(side="left")
        self.lead_icon = info_icon(row, lambda: self.lead, "bg")  # what the page is for: a hover, not a paragraph
        self.set_lead(self.lead)
        self.body = ttk.Frame(outer)
        self.body.pack(fill="both", expand=True, pady=(2 * GUTTER, 0))

    def set_lead(self, text):
        self.lead = text
        self.lead_icon.pack_forget()
        if text:
            self.lead_icon.pack(side="left", padx=(8, 0), pady=(10, 0))

    def refresh(self):
        """Re-read services and redraw. Called when the page is shown."""

    topics: tuple = ()  # snapshot topics the page reads (studio_ia.snapshots); the shell calls redraw() when they change

    def redraw(self, changed):
        """While the page is shown, once per frame in which any of `topics` changed (`changed`, a set). Read the snapshots only."""

    def preload(self):
        """The slow service reads refresh() is about to make, and nothing else: it can run off the Tk thread, so no widgets."""

    opens_at_once = False  # True: the shell shows the page without its loading panel and refresh() asks warmed() first
    _warming = _warm = False

    def warmed(self):
        """For an opens_at_once page: True when preload()'s reads are ready to make on the Tk thread. Otherwise they start
        on the slow lane, refresh() runs again once they are in, and the page draws what it can without them meanwhile."""
        if self._warm or self.services.ready(self.preload):
            return True
        if not self._warming:
            self._warming = True
            self.quietly(self._read_ahead, self._warmed, lane="slow")
        return not self._warming

    def _read_ahead(self):
        try:
            self.preload()
        except Exception:  # refresh() meets the same error on the Tk thread
            pass

    def _warmed(self, _none):
        self._warming = False
        if self.winfo_exists() and self.winfo_manager():  # a page that is not shown reads again when it is
            self._warm = True  # this refresh reads in place whatever ready() says now: a read that failed or went stale never loops
            try:
                self.refresh()
            finally:
                self._warm = False

    def reload(self):
        """refresh() after the stored data may have changed. The shell swaps in a version that loads behind its panel."""
        self.refresh()

    def quietly(self, work, then, lane="fast"):
        """then(work()) for a read the page can show without: the shell runs it on its fast or slow ("slow": scans, anything
        that re-counts the stored data) lane with no loading panel."""
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
