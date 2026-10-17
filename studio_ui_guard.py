"""Keeps slow work off the Tk thread, and says so when it is not.

`@slow("what")` marks a primitive that takes a noticeable time however it is called (scoring a checkpoint, loading one,
building or running a model, training a tokenizer, hashing a file, scanning a table). When one runs on the thread the shell
bound with `bind()`, it raises OnUIThread under tests (GROUNDWORK_STRICT_UI=1, which tasks.py sets) and otherwise logs a
warning with the stack, once per call site. A read the Tk thread may make asks WorkspaceServices.ready() first, whose probe
raises _Cold or _Slow before it reaches a primitive, so it is not caught here.
"""
import functools
import inspect
import logging
import os
import sys
import threading
import traceback

STRICT = "GROUNDWORK_STRICT_UI"  # "1": a slow primitive on the UI thread raises OnUIThread; anything else: it is logged once per call site
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
