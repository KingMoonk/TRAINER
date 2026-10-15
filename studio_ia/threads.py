"""Background threads that are safe next to Tk.

Tk objects (StringVar, PhotoImage...) may only be freed on the Tk thread, and the cyclic collector runs on whichever
thread allocates. A worker that triggers a collection which frees a dead page's variables aborts the process
(Tcl_AsyncDelete). So the collector stays off while any worker runs, and the Tk thread collects once they are done.
hold/release are counted, so the shell's lanes, the page threads and the downloads compose.
"""
import gc
import queue
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
    """Call on the Tk thread when the app is destroyed; puts the collector back."""
    global _attached, _holds
    with _lock:
        _attached, _holds = False, 0
        _done.clear()
    gc.enable()


def hold():
    global _holds
    with _lock:
        if _attached:
            if not _holds:
                gc.disable()
            _holds += 1


def release():
    """Tk thread only: the last release collects here, where freeing Tk objects is allowed, then re-enables."""
    global _holds
    with _lock:
        if not _attached or not _holds:
            return
        _holds -= 1
        last = not _holds
    if last:
        gc.collect()
        gc.enable()


def start(target, *args):
    """Run target(*args) on a daemon thread with the collector held until settle() sees it finished."""
    hold()

    def run():
        try:
            target(*args)
        finally:
            with _lock:
                _done.append(threading.current_thread())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def settle():
    """Tk thread: release the hold of every thread that has finished. Cheap enough to call every few hundred ms."""
    with _lock:
        finished, _done[:] = list(_done), []
    for _ in finished:
        release()


class Lane:
    """One long-lived worker thread that runs jobs in the order they were submitted. submit() and finished() are Tk thread only.
    The lane holds the collector from its first queued job until finished() has handed back the last result, so two busy
    lanes are two holds."""

    def __init__(self):
        self._jobs, self._done, self.pending = queue.Queue(), queue.Queue(), 0
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            work, tag = job
            try:
                self._done.put((tag, work(), None))
            except Exception as error:
                self._done.put((tag, None, error))

    def submit(self, work, tag=None):
        if not self.pending:
            hold()
        self.pending += 1
        self._jobs.put((work, tag))

    def finished(self):
        """[(tag, value, error)] for the jobs that are done since the last call; the lane gives its hold back when it has none left."""
        out = []
        while self.pending:
            try:
                out.append(self._done.get_nowait())
            except queue.Empty:
                break
            self.pending -= 1
        if out and not self.pending:
            release()
        return out

    def stop(self):
        self._jobs.put(None)
