"""Queue jobs run here, one process per job, so heavy work never holds the app's GIL or stalls its Tk thread.

The app (Handle) starts `python -m studio_worker run <spec.json>` and the two talk over a localhost socket, one JSON object per
line. The worker sends {"event": ...} for each event its handler emits, then {"params": the params the handler changed} and
{"error": sentence} or {"done": true}. The app sends {"cmd": "stop" | "pause" | "resume"}. A socket, not stdin and stdout: a
frozen windowed build has no usable standard streams. The socket closing means the app has gone, so the worker stops its job
and exits within EXIT_GRACE. The worker ends its side of the stream with shutdown(SHUT_WR), a FIN, and exits only once the app has
closed its side (or EXIT_GRACE has passed): a process that exits with the socket open can end the connection with a reset, and on
Windows a reset discards whatever the app has not read yet, the result included (#118). Specs live in data/workers/<job id>.json
with the worker's pid until the job ends; one left behind by an app that crashed names a process the next launch stops (Runner._reap).
"""
import dataclasses
import functools
import importlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent
EXIT_GRACE = 30.   # seconds a worker whose app has gone gets to stop its job and save before it exits anyway
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def threads(cores=None):
    """Compute threads for a job: every core but one, so the app's Tk thread always has a core of its own."""
    return max(1, (cores or os.cpu_count() or 1) - 1)


def thread_env(env, cores=None):
    """`env` with each math library capped at threads(): a lower value already set is kept."""
    cap, out = threads(cores), dict(env)
    for name in THREAD_VARS:
        try:
            out[name] = str(min(int(env.get(name) or cap), cap))
        except ValueError:
            out[name] = str(cap)
    return out


def handler_ref(handler):
    """Where the worker finds `handler`: its module, its name in that module, and the folder the module's top package sits in."""
    module = sys.modules[handler.__module__]
    path = getattr(module, "__file__", None)
    return {"module": handler.__module__, "name": handler.__qualname__,
            "path": str(Path(path).resolve().parents[handler.__module__.count(".")]) if path else None}


def _plain(value):
    return value.tolist() if hasattr(value, "tolist") else str(value)  # numpy numbers and arrays; Paths and the rest as text


def _line(message):
    return (json.dumps(message, default=_plain) + "\n").encode("utf-8")


def crash_message(code):
    return f"The job stopped unexpectedly: its process ended with code {code}. Details are in logs/worker.log."


# --------------------------------------------------------------------------- app side

class Handle:
    """One job running in its worker process, seen from the app. `emit` gets each event as the handler emitted it; `done` is set
    once the process has ended, with `params` (what the handler changed) and `error` ('' or a sentence) filled in."""

    def __init__(self, job, handler, context, emit, folder, log_path, on_end=None):
        self.job, self.emit, self.on_end = job, emit, on_end
        self.params, self.error, self.done = {}, "", threading.Event()
        self._finished, self._conn, self._lock = False, None, threading.Lock()
        self._want, self._sent = {"stop": False, "pause": False}, {"stop": False, "pause": False}
        self._token = secrets.token_hex(16)
        self._server = socket.create_server(("127.0.0.1", 0))
        self._server.settimeout(.2)
        folder.mkdir(parents=True, exist_ok=True)
        self.spec = folder / f"{job.id}.json"
        spec = {"job": dataclasses.asdict(job), "handler": handler_ref(handler), "context": context,
                "port": self._server.getsockname()[1], "token": self._token}
        _write(self.spec, spec)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        from studio_core import module_command
        try:
            with open(log_path, "ab") as log:
                self.proc = subprocess.Popen(module_command("studio_worker", "run", str(self.spec)), cwd=str(REPO), env=thread_env(os.environ),
                                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **options)
        except BaseException:
            self._server.close()
            self.spec.unlink(missing_ok=True)
            raise
        _write(self.spec, {**spec, "pid": self.proc.pid})
        threading.Thread(target=self._read, daemon=True, name=f"worker {job.id}").start()

    def sync(self, stop, pause):
        """Pass the runner's stop and pause events on; only changes are sent. A stop is never taken back."""
        with self._lock:
            self._want = {"stop": self._want["stop"] or stop.is_set(), "pause": pause.is_set()}
            self._flush()

    def stop(self):
        with self._lock:
            self._want["stop"] = True
            self._flush()

    def _flush(self):
        if self._conn is None:
            return  # sent once the worker connects
        for key, on in self._want.items():
            if self._sent[key] != on:
                try:
                    self._conn.sendall(_line({"cmd": key if on else "resume"}))
                except OSError:
                    return  # the worker has gone; _read reports it
                self._sent[key] = on

    def _accept(self):
        """The worker's connection once it has proved it is ours, or None when the process ended before connecting."""
        while True:
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                if self.proc.poll() is not None:
                    return None, None
                continue
            conn.settimeout(None)
            lines = conn.makefile("r", encoding="utf-8")
            if lines.readline().strip() == self._token:
                return conn, lines
            lines.close()
            conn.close()

    def _read(self):
        conn = lines = None
        try:
            conn, lines = self._accept()
            if conn is not None:
                with self._lock:
                    self._conn = conn
                    self._flush()
                for line in lines:
                    message = json.loads(line)
                    if "event" in message:
                        self.emit(message["event"])
                    elif "params" in message:
                        self.params = message["params"]
                    elif "error" in message:
                        self.error, self._finished = message["error"] or "Error", True
                    elif "done" in message:
                        self._finished = True
        except (OSError, ValueError):
            pass  # the stream broke: a done or error already read still stands
        finally:
            with self._lock:
                self._conn = None
            for part in (lines, conn):
                if part is not None:
                    part.close()  # before waiting: the worker waits for this close before it exits
            code = self.proc.wait()
            if not self._finished:
                self.error = crash_message(code)
            self._server.close()
            self.spec.unlink(missing_ok=True)
            self.done.set()
            if self.on_end:
                self.on_end()


def _write(path, value):
    from studio_core import atomic_json
    atomic_json(path, value)


def strays(folder):
    """(spec path, pid) for each worker spec left in `folder`; the pid is None when the file never got one or cannot be read."""
    for spec in sorted(Path(folder).glob("*.json")):
        try:
            pid = json.loads(spec.read_text(encoding="utf-8")).get("pid")
        except (OSError, ValueError, AttributeError):
            pid = None
        yield spec, pid


def reap(folder, *, command_line=None, alive=None, terminate=None):
    """Stop the workers an app that crashed left running, then forget their specs. Only a process whose command line names its
    own spec is stopped: after a reboot the pid may be anyone's. Returns the pids it stopped."""
    import studio_passive_watch as procs
    command_line, alive, terminate = command_line or procs.command_line, alive or procs.alive, terminate or procs.terminate
    stopped = []
    for spec, pid in strays(folder):
        if pid and alive(pid):
            line = command_line(pid) or ""
            if "studio_worker" in line and str(spec) in line:
                terminate(pid)
                stopped.append(pid)
        spec.unlink(missing_ok=True)
    return stopped


# --------------------------------------------------------------------------- worker side

class JobContext:
    """What a handler may ask of the app, from inside the worker: the workspace folders, and reads of the stored data. The reads
    are the app's own code (WorkspaceServices.reader) without its runner or window, made the first time a handler asks."""

    def __init__(self, root, run_dir):
        self.root, self.run_dir = Path(root), Path(run_dir)
        self._reader = None

    def _data(self):
        if self._reader is None:
            from studio_ia.services import WorkspaceServices
            self._reader = WorkspaceServices.reader(self.root)
        return self._reader

    def examples(self, params):
        return self._data().examples(params)

    def post_rows(self, kind):
        return self._data().post_rows(kind)


def run(spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    conn = socket.create_connection(("127.0.0.1", spec["port"]))
    conn.sendall((spec["token"] + "\n").encode("utf-8"))
    stop, pause, closed, sending = threading.Event(), threading.Event(), threading.Event(), threading.Lock()

    def send(message):
        data = _line(message)
        with sending:
            try:
                conn.sendall(data)
            except OSError:
                stop.set()  # the app has gone

    def listen():
        try:
            with conn.makefile("r", encoding="utf-8") as lines:
                for line in lines:
                    command = json.loads(line).get("cmd")
                    if command == "stop":
                        stop.set()
                    elif command == "pause":
                        pause.set()
                    elif command == "resume":
                        pause.clear()
        except (OSError, ValueError):
            pass
        closed.set()  # the app has closed its side
        stop.set()   # the app has gone, or the job is over: stop, and exit after EXIT_GRACE even if the handler never notices
        pause.clear()
        timer = threading.Timer(EXIT_GRACE, os._exit, (3,))
        timer.daemon = True
        timer.start()
    threading.Thread(target=listen, daemon=True).start()
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: stop.set())

    from studio_queue import Job
    from studio_timing import timed
    job = Job(**spec["job"])
    before = {k: json.dumps(v, default=_plain, sort_keys=True) for k, v in job.params.items()}
    ref, error = spec["handler"], ""
    try:
        if ref["path"] and ref["path"] not in sys.path:
            sys.path.insert(0, ref["path"])
        handler = functools.reduce(getattr, ref["name"].split("."), importlib.import_module(ref["module"]))
        with timed(f"job {ref['name']}"):
            handler(job, JobContext(**spec["context"]), lambda event: send({"event": event}), stop, pause)
    except Exception as exc:  # noqa: BLE001 - every failure becomes the job's error; the traceback goes to the log
        traceback.print_exc()
        error = str(exc) or type(exc).__name__
    changed = {k: v for k, v in job.params.items() if before.get(k) != json.dumps(v, default=_plain, sort_keys=True)}
    send({"params": changed})
    send({"error": error} if error else {"done": True})
    with sending:
        try:
            conn.shutdown(socket.SHUT_WR)  # the stream ends with a FIN after the result, never a reset
        except OSError:
            pass  # the app has gone
    closed.wait(EXIT_GRACE)  # the app has read the result and closed its side; listen still holds the socket, so close() alone would not
    conn.close()
    return 1 if error else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "run":
        print("usage: python -m studio_worker run <spec.json>", file=sys.stderr)
        return 2
    return run(argv[1])


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # no waiting on threads a handler left behind; the result has been sent
