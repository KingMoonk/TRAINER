"""Background threads that are safe next to Tk.

Tk objects (StringVar, PhotoImage...) may only be freed on the Tk thread, and the cyclic collector runs on whichever
thread allocates. A worker that triggers a collection which frees a dead page's variables aborts the process
(Tcl_AsyncDelete). So the collector stays off while any worker runs, and the Tk thread collects once they are done.
hold/release are counted, so the shell's worker, the page threads and the downloads compose.
"""
import gc
import threading

_lock = threading.Lock()
_holds = 0
_attached = False  # only an app with a Tk thread needs the collector held
_done = []  # threads that finished, to be released by the Tk thread


def attach():
    """Call on the Tk thread when the app starts."""
    global _attached
    _attached = True


def detach():
    """Call on the Tk thread when the app is destroyed. Threads still running keep their holds: the collector stays
    off until they finish and the next app settles them, so a dead app's Tk objects are never freed off the Tk thread."""
    global _attached
    _attached = False
    settle()


def hold():
    """Returns whether a hold was taken: only an attached app takes one."""
    global _holds
    with _lock:
        if not _attached:
            return False
        if not _holds:
            gc.disable()
        _holds += 1
        return True


def release():
    """Tk thread only: the last release collects here, where freeing Tk objects is allowed, then re-enables."""
    global _holds
    with _lock:
        if not _holds:
            return
        _holds -= 1
        last = not _holds
    if last:
        gc.collect()
        gc.enable()


def start(target, *args):
    """Run target(*args) on a daemon thread with the collector held until settle() sees it finished."""
    held = hold()

    def run():
        try:
            target(*args)
        finally:
            if held:
                done()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def done():
    """Called by a held thread as it finishes: its hold is released by the next settle() on the Tk thread."""
    with _lock:
        _done.append(threading.current_thread())


def settle():
    """Tk thread: release the hold of every thread that has finished. Cheap enough to call every few hundred ms."""
    with _lock:
        finished, _done[:] = list(_done), []
    for _ in finished:
        release()
