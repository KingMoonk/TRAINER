"""Non-blocking process bridge between the studio GUI and a separate viewer process.

GUI side: VizBridge launches the viewer script and exchanges dict messages over a
loopback multiprocessing connection. Viewer side: ViewerLink connects back to the GUI.
"""
from __future__ import annotations

import argparse
import atexit
import os
import queue
import subprocess
import sys
import threading
import time
import weakref
from collections import deque
from multiprocessing.connection import AuthenticationError, Client, Listener
from pathlib import Path

_COALESCE = ("telemetry", "spec", "viz_snapshot", "snapshot", "summary", "selection", "stats")
_POLL_S = 0.1
_BRIDGES = weakref.WeakSet()  # one exit hook for every live bridge; a reloaded module's old bridges are not pinned


@atexit.register
def _stop_all():
    for bridge in list(_BRIDGES):
        bridge.stop()


class _Session:
    """State for one launch of the viewer; threads get it explicitly so restarts stay isolated."""

    def __init__(self):
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.ordered = deque()
        self.pending = {}
        self.inbound = queue.Queue()
        self.closed = False
        self.disconnected = False
        self.connected = False
        self.stopping = False
        self.conn = None


def _reader(conn, sess):
    """Move received messages into the inbound queue; mark the link down on EOF or error."""
    while not sess.closed:
        try:
            if conn.poll(_POLL_S):
                sess.inbound.put(conn.recv())
        except (EOFError, OSError, ValueError):
            break
    sess.disconnected = True
    sess.inbound.put({"type": "closed"})


def _writer(conn, sess):
    """Drain the outbound queues: ordered messages first, then the latest coalesced ones."""
    while True:
        sess.event.wait()
        sess.event.clear()
        if sess.closed:
            return
        with sess.lock:
            batch = []
            for item in sess.ordered:
                if isinstance(item, str):  # coalesced kind marker: send its latest message
                    msg = sess.pending.pop(item, None)
                    if msg is not None:
                        batch.append(msg)
                else:
                    batch.append(item)
            sess.ordered.clear()
            sess.pending.clear()
        for msg in batch:
            try:
                conn.send(msg)
            except (OSError, EOFError, ValueError):
                sess.disconnected = True
                with sess.lock:
                    sess.ordered.clear()
                    sess.pending.clear()
                return
            except Exception as error:  # unpicklable message: drop it, keep the link, and tell the GUI
                print(f"viewer link: dropped a {msg.get('type')!r} message: {error}", file=sys.stderr, flush=True)
                sess.inbound.put({"type": "send_error", "message": f"Could not send a {msg.get('type')!r} message to the viewer: {error}"})


def _accept(listener, sess):
    """Wait for the viewer to connect, then start its reader and writer threads."""
    while not sess.closed:
        try:
            conn = listener.accept()
        except AuthenticationError:
            continue
        except (OSError, EOFError):
            return
        if sess.closed:
            conn.close()
            return
        sess.conn = conn
        sess.connected = True
        threading.Thread(target=_reader, args=(conn, sess), daemon=True).start()
        threading.Thread(target=_writer, args=(conn, sess), daemon=True).start()
        return


def _finish(proc, sess, listener, port, authkey):
    """Wait for a viewer that was asked to quit, terminate it if it lingers, then close the link."""
    if sess.connected and not sess.disconnected:
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1)
    sess.closed = True
    sess.event.set()
    if listener is not None:
        if not sess.connected:
            # Unblock accept() (it cannot be interrupted by closing on Windows).
            poke = threading.Thread(target=_poke, args=(port, authkey), daemon=True)
            poke.start()
            poke.join(1.0)
        try:
            listener.close()
        except OSError:
            pass
    if sess.conn is not None:
        try:
            sess.conn.close()
        except OSError:
            pass


def _poke(port, authkey):
    """Connect once so a pending Listener.accept() returns."""
    try:
        Client(("127.0.0.1", port), authkey=authkey).close()
    except Exception:
        pass


class VizBridge:
    """GUI-side end of the viewer link. Every method returns promptly; safe on the GUI thread."""

    def __init__(self, script_path, python=sys.executable, extra_args=(), command=None):
        self.script = Path(script_path).resolve()
        self.python = str(python)
        self.command = list(command) if command else None  # frozen app: the executable plus its run-this-module arguments, no script path
        self.extra_args = [str(a) for a in extra_args]
        self._proc = None
        self._sess = None
        self._listener = None
        self._authkey = b""
        self._port = 0
        self.log_path = None
        _BRIDGES.add(self)

    def start(self, log_path=None):
        """Launch the viewer process, its stderr going to log_path when given. No-op if it is already running."""
        if self.running:
            return
        if self._proc is not None:
            self.stop(wait=False)
        sess = _Session()
        self._authkey = os.urandom(16)
        listener = Listener(("127.0.0.1", 0), authkey=self._authkey)
        self._port = listener.address[1]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        # the key goes over stdin: a command line is readable by every user on the machine
        cmd = [*(self.command or [self.python, str(self.script)]), "--port", str(self._port), *self.extra_args]
        log = None
        try:
            if log_path is not None:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                log = open(log_path, "wb")
            proc = subprocess.Popen(cmd, cwd=None if self.command else str(self.script.parent), creationflags=flags,
                                    stdin=subprocess.PIPE, stderr=log)
        except BaseException:
            listener.close()
            raise
        finally:
            if log is not None:
                log.close()  # the child holds its own handle
        try:
            proc.stdin.write(self._authkey.hex().encode() + b"\n")
            proc.stdin.close()
        except OSError:
            pass  # the viewer already died; the caller sees it through exit_code
        self._proc, self._sess, self._listener, self.log_path = proc, sess, listener, log_path
        threading.Thread(target=_accept, args=(listener, sess), daemon=True).start()

    def stop(self, wait=True):
        """Ask the viewer to quit, wait up to 2 s, then terminate. Idempotent.

        wait=False returns at once and does the waiting on a background thread, so the GUI never stalls."""
        proc, sess = self._proc, self._sess
        if proc is None or sess is None or sess.closed or sess.stopping:
            return
        if sess.connected and not sess.disconnected:
            self._put({"type": "quit"})
        sess.stopping = True
        listener, self._listener = self._listener, None
        args = (proc, sess, listener, self._port, self._authkey)
        if wait:
            _finish(*args)
        else:
            threading.Thread(target=_finish, args=args, daemon=True, name="viewer-stop").start()

    def _put(self, msg):
        sess = self._sess
        kind = msg.get("type")
        with sess.lock:
            if kind in _COALESCE:
                if kind in sess.pending:  # only the latest survives, at its latest position
                    sess.ordered.remove(kind)
                sess.pending[kind] = msg
                sess.ordered.append(kind)
            else:
                sess.ordered.append(msg)
        sess.event.set()

    def send(self, msg):
        """Queue a message for the viewer. Never blocks; no-op if not started or dead."""
        sess = self._sess
        if not isinstance(msg, dict) or sess is None or sess.closed or sess.stopping or sess.disconnected:
            return
        if self._proc.poll() is not None:
            return
        self._put(msg)

    def poll(self):
        """Return all messages received from the viewer so far. Never blocks."""
        sess = self._sess
        out = []
        if sess is None:
            return out
        while True:
            try:
                out.append(sess.inbound.get_nowait())
            except queue.Empty:
                return out

    @property
    def running(self):
        """True while the viewer process is alive and the link is up or still connecting."""
        sess = self._sess
        return (self._proc is not None and sess is not None and not sess.closed and not sess.stopping
                and not sess.disconnected and self._proc.poll() is None)

    @property
    def connected(self):
        """True once the viewer has connected back (it may have disconnected since)."""
        return self._sess is not None and self._sess.connected

    @property
    def exit_code(self):
        """The viewer's exit code, or None while it runs or before start."""
        return None if self._proc is None else self._proc.poll()


class ViewerLink:
    """Viewer-side end of the link, built from parse_viewer_args()."""

    def __init__(self, port, authkey_hex, timeout=5.0):
        key = bytes.fromhex(authkey_hex)
        deadline = time.monotonic() + timeout
        while True:
            try:
                self._conn = Client(("127.0.0.1", int(port)), authkey=key)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise ConnectionError(f"could not reach the studio on port {port}") from None
                time.sleep(0.1)
        self._inbound = queue.Queue()
        self._lock = threading.Lock()
        self._connected = True
        self._closed = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self._closed:
            try:
                if self._conn.poll(_POLL_S):
                    self._inbound.put(self._conn.recv())
            except (EOFError, OSError, ValueError):
                break
        self._connected = False

    @property
    def connected(self):
        return self._connected

    def send(self, msg):
        """Send a message to the studio; marks the link down on failure."""
        with self._lock:
            if not self._connected:
                return
            try:
                self._conn.send(msg)
            except (OSError, EOFError, ValueError):
                self._connected = False

    def poll(self):
        """Return all messages received from the studio so far. Never blocks."""
        out = []
        while True:
            try:
                out.append(self._inbound.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        self._closed = True
        self._connected = False
        try:
            self._conn.close()
        except OSError:
            pass


def parse_viewer_args(argv=None, stdin=None):
    """Return (port, authkey_hex): the port from the command line, the key from its first stdin line
    (VizBridge writes it there; --authkey still works for manual runs). (None, None) without a port."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--authkey", default=None)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else list(argv))
    if args.port is None:
        return None, None
    key = args.authkey or (stdin or sys.stdin).readline().strip()
    return (args.port, key) if key else (None, None)
