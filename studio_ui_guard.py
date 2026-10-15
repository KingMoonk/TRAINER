"""Keeps slow work off the Tk thread, and says so when it is not.

Two guards, both cheap:

* `@slow("what")` marks a primitive that takes a noticeable time however it is called (scoring a checkpoint, loading one,
  building or running a model, training a tokenizer, hashing a file, scanning a table). When one runs on the thread the shell
  bound with `bind()`, it raises OnUIThread under tests (GROUNDWORK_STRICT_UI=1, which tasks.py sets) and otherwise logs a
  warning with the stack, once per call site. A read the Tk thread may make asks WorkspaceServices.ready() first, whose probe
  raises _Cold or _Slow before it reaches a primitive, so it is not caught here.
* `StallWatch` is a heartbeat the Tk loop bumps from `after()` and a daemon thread that watches it. When the beat is
  older than `limit`, the Tk thread's stack goes to logs/ui-stalls.log, once per stall, so a freeze that no test covered
  still names the call that held the thread.
"""
import functools
import inspect
import logging
import os
import sys
import threading
import time
import traceback

STRICT = "GROUNDWORK_STRICT_UI"  # "1": a slow primitive on the UI thread raises OnUIThread; anything else: it is logged once per call site
STALLS = "GROUNDWORK_STALL_LOG"  # "0": no heartbeat and no ui-stalls.log (tasks.py sets it for every test run)
_log = logging.getLogger("groundwork.ui")
_ui = None  # ident of the bound UI thread
_seen = set()  # (what, file, line) already logged


class OnUIThread(RuntimeError):
    """A @slow primitive ran on the UI thread."""


def bind(ident=None):
    """The shell, on its Tk thread (Services.bind_ui), or a test standing in for it."""
    global _ui
    _ui = threading.get_ident() if ident is None else ident


def unbind(ident=None):
    """Forget the UI thread (with ident: only if it is still that one, so a closing app does not unbind the next)."""
    global _ui
    if ident is None or _ui == ident:
        _ui = None


def ui_thread():
    return _ui


def on_ui_thread():
    return _ui is not None and threading.get_ident() == _ui


def strict():
    return os.environ.get(STRICT) == "1"


def check(what, depth=2):
    """Called by a @slow primitive's wrapper as it starts. depth is how many frames up the call site is: check, the wrapper, then its caller."""
    if not on_ui_thread():
        return
    frame = sys._getframe(depth)
    site = (what, frame.f_code.co_filename, frame.f_lineno)
    message = f"{what} ran on the UI thread ({frame.f_code.co_filename}:{frame.f_lineno}). Run it on a worker and emit an event when it is in."
    if strict():
        raise OnUIThread(message)
    if site not in _seen:
        _seen.add(site)
        _log.warning("%s\n%s", message, "".join(traceback.format_stack(frame)))


def slow(what, when=None):
    """Decorator: the function (or generator function) takes long enough to freeze the window; see check().
    when(*args, **kwargs), if given, says whether this call is that long: a scan of a small store is read in place by design
    (WorkspaceServices.ready), so only a scan of a large one counts."""
    def wrap(fn):
        if inspect.isgeneratorfunction(fn):
            @functools.wraps(fn)
            def generator(*args, **kwargs):
                if when is None or when(*args, **kwargs):
                    check(what)  # at the first next(): the thread that reads it
                yield from fn(*args, **kwargs)
            return generator

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if when is None or when(*args, **kwargs):
                check(what)
            return fn(*args, **kwargs)
        return wrapper
    return wrap


class StallWatch:
    """beat() from the Tk loop every ~100 ms; check() (the daemon thread, or a test with a fake clock) writes the Tk thread's
    stack to `path` once the beat is older than `limit`, then a line with the stall's length when beats resume.

    The loop allocates nothing the cyclic collector tracks, so this thread never runs a collection that could free Tk objects
    off the Tk thread; the one path that does allocate (writing a stall) holds the collector first (threads.hold_until_settled)."""

    def __init__(self, path, ui_ident, limit=.3, interval=.1, clock=time.monotonic, frames=sys._current_frames, hold=None):
        self.path, self.ui_ident, self.limit, self.interval = path, ui_ident, limit, interval
        self.clock, self.frames, self.hold = clock, frames, hold
        self._beat, self._stalled, self._running = clock(), None, False

    def beat(self):
        self._beat = self.clock()

    def check(self):
        """One look at the beat. True when a stall was written by this call."""
        beat = self._beat
        age = self.clock() - beat
        if self._stalled is None:
            if age <= self.limit:
                return False
            self._stalled = beat
            if self.hold:
                self.hold()
            self._write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} the UI thread has not answered for {age * 1000:.0f} ms; where it is:\n"
                        + self._stack())
            return True
        if beat != self._stalled:  # beating again
            self._write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} the stall lasted about {(beat - self._stalled) * 1000:.0f} ms\n\n")
            self._stalled = None
        return False

    def _stack(self):
        frame = self.frames().get(self.ui_ident)
        return "".join(traceback.format_stack(frame)) if frame is not None else "(the UI thread is gone)\n"

    def _write(self, text):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(text)
        except OSError:  # a log that cannot be written must not take the watcher down
            pass

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, name="ui-stall-watch", daemon=True).start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            time.sleep(self.interval)
            self.check()


def stalls_enabled():
    return os.environ.get(STALLS) != "0"
