"""Studio window: sidebar (four areas + machine strip) and the page router."""
from __future__ import annotations

import gc
import queue
import threading
import tkinter as tk
import traceback
from collections import namedtuple
from tkinter import ttk

from studio_timing import timed
from studio_ux import format_duration
from studio_ia import services as services_module
from studio_ia import threads
from studio_ia.pages.base import Page
from studio_ia.theme import C, GUTTER, SIDEBAR_W, apply, font, tk_label
from studio_ia.widgets import Progress, _activate

Route = namedtuple("Route", "area title lead crumb")

AREAS = [("designer", "Network designer"), ("pretraining", "Pretraining"), ("post", "Post-training"), ("chat", "Chat"),
         ("machine", "This computer"), ("data", "Your data")]
AREA_HOME = {"designer": "builder", "pretraining": "pretraining", "post": "post_training", "chat": "chat", "machine": "queue", "data": "your_data"}
UTILITY_AREAS = ("machine", "data")  # not stages of the pipeline: set off from the four above by a rule
LI = ("Network designer", "Look inside")
NEXT_EVENTS = ("queue", "job", "blocked", "versions", "checkpoint", "data", "download_done", "import_done", "downloads_finished")  # what can change the Next row

# route -> Route(area, h1, lead, breadcrumb): area is the sidebar row that lights up on that page.
ROUTES = {
    "builder": Route("designer", "Build your model", "Change the picture. See what it costs and what it does.", ()),
    "quicktest": Route("designer", "Quick test", "Try a design idea for a few minutes on a tiny dataset, before you save it and train it for real.", ("Network designer",)),
    "compare": Route("designer", "Compare runs", "Change one thing, hold the rest equal, run several seeds. A difference counts only if it is larger than the spread across seeds.", ("Network designer",)),
    "queue": Route("machine", "This computer", "What it’s training now, what’s waiting, and when it’s allowed to work. Every “Train” or “Run” button in the app adds to this queue.", ()),
    "trace": Route("designer", "Trace a position", "Follow one position's stream through the model. Numbers appear when the engine has run your prompt.", (*LI, "Trace a position")),
    "head_zoom": Route("designer", "Inside a head", "One attention head, one step at a time, on a real prompt.", (*LI, "Inside a head")),
    "value_map": Route("designer", "Value map", "How a head's values are projected down and back up, and why the heads add up.", (*LI, "Value map")),
    "loss": Route("designer", "Loss", "What one wrong or right prediction costs, and the gradient it sends back.", (*LI, "Loss")),
    "head_lab": Route("designer", "Head lab", "Where does each attention head look, and how much does the model rely on it? Switch heads off and watch the answer change.", (*LI, "Head lab")),
    "inside3d": Route("designer", "Inside the network", "Build the model from the inside and watch it train, from a window of positions down to single weights.", (*LI, "3D")),
    "pretraining": Route("pretraining", "Pretraining", "Feed it text. It learns by predicting the next word, over and over.", ()),
    "training_run": Route("pretraining", "Pretraining", "Step k of n, the schedule phase and the error so far.", ()),
    "your_data": Route("data", "Your data", "Three kinds, because each one trains a different stage.", ()),
    "make_data": Route("data", "Make extra data", "Optional. Generate more, then keep what passes review.", ()),
    "post_training": Route("post", "Post-training", "Four rounds, always in this order. Same design throughout; only the weights shift.", ()),
    "chat": Route("chat", "Chat", "Talk to any trained version. Compare two to see what a change really did.", ()),
}


def strip_lines(q, version=None):
    """The machine strip's text for a QueueState: (now line, queued line, chat line, chat line is the Resume link).
    Idle and paused by the user reads "Paused by you", and the chat line turns into the way out of it."""
    waiting = len(q.up_next)
    if q.job is None:
        now = "Paused by you" if q.paused else "Nothing is training"
        left = ""
    else:
        now = f"{q.job.label or q.job.kind} · {'paused' if q.paused else q.phase or 'training'}"
        left = format_duration(q.eta_s) + " left · " if q.eta_s is not None else ""
    if q.job is None and q.paused:
        chat = "Resume ›"
    else:
        chat = f"Chat uses version {version.number} · best so far" if version else "Chat has no trained version yet"
    return now, f"{left}{waiting} queued", chat, q.job is None and q.paused


class Studio(tk.Tk):
    def __init__(self, services=None):
        # Free the Tk objects of earlier, destroyed Studios now, before this interpreter exists and outside any Tk
        # callback. Left to the collector they can be freed during this app's update(), which crashes Tk on macOS.
        gc.collect()
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
        from studio_ia.pages.make_data import MakeDataPage
        from studio_ia.pages.your_data import YourDataPage
        self.factories["your_data"], self.factories["make_data"] = YourDataPage, MakeDataPage
        self.route = None
        self._area_rows = {}
        self._build_sidebar()
        self.main = ttk.Frame(self)
        self.main.pack(side="left", fill="both", expand=True)
        self._build_loader()
        from studio_ia.pages import head_lab, head_zoom, loss, trace, value_map  # look-inside pages (S7)
        from studio_ia.pages import inside3d  # look-inside 3D page (S11)
        self.register("inside3d", inside3d.Inside3DPage)
        for route, cls in (("trace", trace.TracePage), ("head_zoom", head_zoom.HeadZoomPage), ("value_map", value_map.ValueMapPage),
                           ("loss", loss.LossPage), ("head_lab", head_lab.HeadLabPage)):
            self.register(route, cls)
        self.unsubscribe = self.services.subscribe(self.on_event)
        threads.attach()
        self._settle()
        self._pump()
        self.services.start_queue()  # jobs saved as waiting by the last session start again
        with timed("shell.refresh_strip"):
            self.refresh_strip()
        self.refresh_next()
        with timed("shell.show_builder"):
            self.show("builder")
        self.quietly(self.services.refresh_downloads, lambda _rebuilt: None)  # downloads an older app converted get rebuilt once

    # sidebar ----------------------------------------------------------------
    def _build_sidebar(self):
        side = tk.Frame(self, bg=C["panel"], width=SIDEBAR_W, highlightthickness=1, highlightbackground=C["border2"])
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        tk_label(side, "Groundwork Studio", 17, "text", "panel", font=font(17, bold=True)).pack(fill="x", padx=GUTTER, pady=GUTTER)
        self._build_next(side)
        for key, name in AREAS:
            if key == UTILITY_AREAS[0]:
                tk.Frame(side, bg=C["border2"], height=1).pack(fill="x", padx=GUTTER, pady=(6, 6))
            row = tk.Frame(side, bg=C["panel"], takefocus=1, cursor="hand2", highlightthickness=1,
                           highlightbackground=C["panel"], highlightcolor=C["violet"])
            row.pack(fill="x")
            stripe = tk.Frame(row, bg=C["panel"], width=3)
            stripe.pack(side="left", fill="y")
            inner = tk.Frame(row, bg=C["panel"])
            inner.pack(side="left", fill="x", expand=True, padx=(GUTTER - 3, GUTTER), pady=10)
            title = tk_label(inner, name, 15, "text", "panel", font=font(15, bold=True))
            title.pack(fill="x")
            _activate(row, lambda k=key: self.show(AREA_HOME[k]))
            for w in (stripe, inner, title):
                w.bind("<Button-1>", lambda _e, k=key: self.show(AREA_HOME[k]))
            self._area_rows[key] = (row, stripe, inner, title)
        # machine strip
        tk.Frame(side, bg=C["border2"], height=1).pack(side="bottom", fill="x")
        strip = tk.Frame(side, bg=C["panel"])
        strip.pack(side="bottom", fill="x", padx=GUTTER, pady=GUTTER)
        self.strip_now = tk_label(strip, "", 13, "text", "panel", wraplength=SIDEBAR_W - 2 * GUTTER)
        self.strip_bar = Progress(strip)
        self.strip_next = tk_label(strip, "", 12, "muted", "panel", wraplength=SIDEBAR_W - 2 * GUTTER)
        self.strip_chat = tk_label(strip, "", 12, "muted", "panel", wraplength=SIDEBAR_W - 2 * GUTTER)
        self._strip_pending = False
        self._strip_version, self._strip_loaded = None, False  # the version the chat line names, once it has been read
        self._strip_loading, self._strip_again = False, False  # a read of it is on the worker; another was asked for meanwhile
        self._strip_resume = False  # the chat line is the "Resume" link only while the queue is paused by the user and idle
        for seq in ("<Button-1>", "<Return>", "<space>"):
            self.strip_chat.bind(seq, self._strip_chat_click)
        self.strip_link = tk_label(strip, "Open the queue ›", 12, "violet", "panel", cursor="hand2", takefocus=1,
                                   highlightthickness=1, highlightbackground=C["panel"], highlightcolor=C["violet"])
        self.strip_now.pack(fill="x", pady=(6, 4))
        self.strip_bar.pack(fill="x")
        self.strip_next.pack(fill="x", pady=(4, 0))
        self.strip_chat.pack(fill="x")
        self.strip_link.pack(fill="x", pady=(6, 0))
        for seq in ("<Button-1>", "<Return>", "<space>"):
            self.strip_link.bind(seq, lambda _e: self.show("queue"))

    def _build_next(self, side):
        """The "Next" row: services.next_step() as a kicker, a title and a line; a click opens the page that does it."""
        box = tk.Frame(side, bg=C["panel2"], takefocus=1, cursor="hand2", highlightthickness=1, highlightbackground=C["border2"],
                       highlightcolor=C["violet"])
        box.pack(fill="x", padx=GUTTER, pady=(0, GUTTER))
        wrap = SIDEBAR_W - 4 * GUTTER
        self.next_kicker = tk_label(box, "NEXT", 12, "violet", "panel2", font=font(12, bold=True))
        self.next_title = tk_label(box, "", 15, "text", "panel2", font=font(15, bold=True), wraplength=wrap)
        self.next_line = tk_label(box, "", 12, "muted", "panel2", wraplength=wrap)
        for w in (self.next_kicker, self.next_title, self.next_line):
            w.pack(fill="x", padx=10, pady=(6, 0) if w is self.next_kicker else (0, 0))
        self.next_line.pack_configure(pady=(0, 8))
        self.next_route, self._next_pending, self._next_again = None, False, False
        for w in (box, self.next_kicker, self.next_title, self.next_line):
            for seq in ("<Button-1>", "<Return>", "<space>"):
                w.bind(seq, self._next_click)
        self.next_box = box

    @property
    def next_text(self):
        return "\n".join(w.cget("text") for w in (self.next_title, self.next_line))

    def refresh_next(self):
        """Ask services.next_step() on the worker (it may read the whole text summary) and repaint the row when it answers."""
        if self._next_pending:
            self._next_again = True  # an answer is on its way, but the state changed since it was asked
            return
        self._next_pending = True

        def ask():
            try:
                return self.services.next_step()
            except NotImplementedError:  # a Services with no workspace behind it: the row keeps what it had
                return None
            except Exception:  # a failed read must not leave the row waiting for an answer that never comes
                traceback.print_exc()
                return None
        self.quietly(ask, self._paint_next)

    def _paint_next(self, step):
        self._next_pending = False
        if step is not None:
            self.next_route = step.route
            self.next_title.configure(text=step.title)
            self.next_line.configure(text=step.line)
        if self._next_again:
            self._next_again = False
            self.refresh_next()

    def _next_click(self, _event=None):
        if self.next_route:
            self.show(self.next_route)

    @property
    def strip_text(self):
        return "\n".join(w.cget("text") for w in (self.strip_now, self.strip_next, self.strip_chat))

    def refresh_strip(self):
        """Repaint the strip now from the queue; the version the chat line names is read where it is cheap (scored bands are
        kept), and otherwise on the worker, the line keeping its last text until it arrives."""
        def work():
            try:
                return self.services.best_version() or self.services.current_version()
            except NotImplementedError:
                return None
        if self.services.ready(work):
            self._strip_version, self._strip_loaded = work(), True
        elif self._strip_loading:
            self._strip_again = True
        else:
            self._strip_loading = True

            def load():
                try:
                    return work(), None
                except Exception as error:  # noqa: BLE001 - reported on the Tk thread below
                    return None, error

            def done(result):
                version, error = result
                self._strip_loading = False
                if error is None:
                    self._strip_version, self._strip_loaded = version, True
                again, self._strip_again = self._strip_again, False
                if again:
                    self.refresh_strip()
                else:
                    self._paint_strip()
                if error is not None:
                    self.report_callback_exception(type(error), error, error.__traceback__)
            self.quietly(load, done)
        self._paint_strip()

    def _paint_strip(self):
        q = self.services.queue()
        now, queued, chat, resume = strip_lines(q, self._strip_version)
        self.strip_now.configure(text=now)
        if q.job is None:
            self.strip_bar.set(None)
        else:
            self.strip_bar.set(q.step / q.total if q.total else None)
        self.strip_next.configure(text=queued)
        if self._strip_loaded or resume:
            self.strip_chat.configure(text=chat, fg=C["violet"] if resume else C["muted"], cursor="hand2" if resume else "", takefocus=int(resume))
        self._strip_resume = resume

    def _strip_tick(self):
        self._strip_pending = False
        self.refresh_strip()

    def _strip_chat_click(self, _event=None):
        if self._strip_resume:
            self.services.resume()

    def _highlight(self, area):
        for key, (row, stripe, inner, title) in self._area_rows.items():
            on = key == area
            bg = C["violet_bg"] if on else C["panel"]
            stripe.configure(bg=C["violet"] if on else C["panel"])
            for w in (row, inner, title):
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
            page.slow, page.background, page.reload = self.slow, self.background, lambda: self.show(route)
            page.quiet_reload = lambda: self._quiet_reload(route)
            page.quietly = self.quietly
        page = self.pages[route]
        self._highlight(spec.area)
        self._wanted = route  # a later click wins over a page still loading

        def warm():
            try:
                page.preload()
            except Exception:  # refresh() meets the same error on the Tk thread, as it always did
                pass

        def loaded(_none):
            self._loading.discard(route)
            if self._wanted == route:
                self._reveal(route)
        if type(page).preload is Page.preload:
            self._reveal(route)
        elif route not in self._loading:  # events that arrive while it loads do not queue the same read again
            self._loading.add(route)
            self.slow(warm, loaded, f"Opening {spec.title}")
        return page

    def _reveal(self, route):
        page = self.pages[route]
        if self.route and self.route != route:
            self.pages[self.route].pack_forget()
        self.route = route
        if not page.winfo_manager():
            page.pack(fill="both", expand=True)
        self.loader.place_forget()
        page.refresh()

    def _quiet_reload(self, route):
        """Re-read a page's data on the worker without the loading panel; the page stays usable and redraws when done."""
        page = self.pages[route]

        def done(_none):
            if self.route == route:
                page.refresh()
        self.quietly(page.preload, done)

    # long reads: one worker takes them from a queue; the Tk thread only polls for results ---------------------
    def _build_loader(self):
        self.jobs, self.results, self._waiting, self._wanted, self._loading = queue.Queue(), queue.Queue(), 0, None, set()
        self._quiet = 0  # jobs from quietly(): on the worker, no panel
        threading.Thread(target=self._work, daemon=True).start()
        self.loader = tk.Frame(self.main, bg=C["bg"])
        box = tk.Frame(self.loader, bg=C["bg"])
        box.place(relx=.5, rely=.4, anchor="center", width=440)
        self.load_title = tk_label(box, "", 17, font=font(17, bold=True))
        self.load_text = tk_label(box, "", 13, "muted", wraplength=440)
        self.load_bar = Progress(box)
        self.load_title.pack(fill="x")
        self.load_text.pack(fill="x", pady=(6, 8))
        self.load_bar.pack(fill="x")

    def _work(self):
        while True:
            work, then, quiet = self.jobs.get()
            if work is None:
                if then:  # the shell was destroyed mid-job: its hold is released once that job has finished
                    threads.done()
                return
            try:
                self.results.put((then, work(), None, quiet))
            except Exception as error:
                self.results.put((then, None, error, quiet))

    def slow(self, work, then, title="Working"):
        """then(work()) on the Tk thread. work() is service reads only; when one is not cached it runs on the worker
        behind the loading panel, which shows what services.note() says is being read and how far along it is."""
        if self.services.ready(work):
            return then(work())
        self.background(work, then, title)

    def background(self, work, then, title="Working"):
        """then(work()) with work() always on the worker: for work that is long every time or changes data."""
        self.services.note("")
        self.load_title.configure(text=title or "Working")
        self.loader.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.loader.lift()
        self._submit(work, then, quiet=False)

    def quietly(self, work, then):
        """then(work()) with work() on the worker and no loading panel, for a refresh the page can wait for."""
        self._submit(work, then, quiet=True)

    def _submit(self, work, then, quiet):
        busy = self._waiting + self._quiet
        if not busy:
            threads.hold()  # Tk objects may only be freed on the Tk thread: see studio_ia/threads.py
        self.jobs.put((work, then, quiet))
        if quiet:
            self._quiet += 1
        else:
            self._waiting += 1
        if not busy:
            self._poll()

    def _poll(self):
        if self._waiting:
            text, fraction = self.services.loading
            self.load_text.configure(text=text or "Getting things ready")
            self.load_bar.set(fraction)
        while self._waiting + self._quiet:
            try:
                then, value, error, quiet = self.results.get_nowait()
            except queue.Empty:
                break
            if quiet:
                self._quiet -= 1
            else:
                self._waiting -= 1
                if not self._waiting:
                    self.loader.place_forget()
            if not self._waiting + self._quiet:
                threads.release()
            if error is None:
                then(value)
            else:
                self.report_callback_exception(type(error), error, error.__traceback__)
        if self._waiting + self._quiet:
            self.after(60, self._poll)

    def _pump(self):
        self.services.pump()  # events the queue runner and other workers emitted, delivered here on the Tk thread
        self._pump_id = self.after(50, self._pump)

    def _settle(self):
        threads.settle()  # page and download threads that finished give the collector back, here on the Tk thread
        self._settle_id = self.after(250, self._settle)

    def on_event(self, kind, payload):
        if kind in NEXT_EVENTS:
            self.refresh_next()
        if kind in ("queue", "job", "blocked"):
            self.refresh_strip()
        elif kind == "train" and not self._strip_pending:  # steps arrive several times a second: redraw the strip once a second
            self._strip_pending = True
            self.after(1000, self._strip_tick)
        if self.route:
            self.pages[self.route].on_event(kind, payload)

    def destroy(self):
        self.jobs.put((None, bool(self._waiting + self._quiet), False))
        threads.detach()
        self.after_cancel(self._settle_id)
        self.after_cancel(self._pump_id)
        self.unsubscribe()
        for pending in self.tk.splitlist(self.tk.call("after", "info")):
            self.after_cancel(pending)  # an `after` left behind fires in the next app's update() into deleted commands
        close = getattr(self.services, "close", None)
        if close:
            close()  # stops the queue runner; a job cut short goes back to the front of the queue
        super().destroy()


def main():
    try:
        from studio_core import ROOT, Workspace
        Studio(services_module.WorkspaceServices(Workspace())).mainloop()  # the repo-root workspace
        return 0
    except Exception:
        from tkinter import messagebox
        report = traceback.format_exc()
        (ROOT / "studio-error.log").write_text(report, encoding="utf-8")
        messagebox.showerror("Groundwork could not start", "See studio-error.log for details.\n\n" + report.splitlines()[-1])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
