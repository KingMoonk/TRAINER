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
from collections import deque
from multiprocessing.connection import AuthenticationError, Client, Listener
from pathlib import Path

_COALESCE = ("telemetry", "spec")  # only the latest of each kind is kept while queued
_POLL_S = 0.1


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
            except Exception:
                continue  # unpicklable message: drop it, keep the link


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
        self._atexit = False

    def start(self):
        """Launch the viewer process. No-op if it is already running."""
        if self.running:
            return
        if self._proc is not None:
            self.stop()
        self._sess = _Session()
        self._authkey = os.urandom(16)
        self._listener = Listener(("127.0.0.1", 0), authkey=self._authkey)
        self._port = self._listener.address[1]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        cmd = [*(self.command or [self.python, str(self.script)]), "--port", str(self._port),
               "--authkey", self._authkey.hex(), *self.extra_args]
        self._proc = subprocess.Popen(cmd, cwd=None if self.command else str(self.script.parent), creationflags=flags)
        threading.Thread(target=_accept, args=(self._listener, self._sess), daemon=True).start()
        if not self._atexit:
            atexit.register(self.stop)
            self._atexit = True

    def stop(self):
        """Ask the viewer to quit, wait up to 2 s, then terminate. Idempotent."""
        proc, sess = self._proc, self._sess
        if proc is None or sess is None or sess.closed:
            return
        if sess.connected and not sess.disconnected:
            self._put({"type": "quit"})
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
        self._shutdown(sess)

    def _shutdown(self, sess):
        sess.closed = True
        sess.event.set()
        listener = self._listener
        if listener is not None:
            if not sess.connected:
                # Unblock accept() (it cannot be interrupted by closing on Windows).
                poke = threading.Thread(target=_poke, args=(self._port, self._authkey), daemon=True)
                poke.start()
                poke.join(1.0)
            try:
                listener.close()
            except OSError:
                pass
        self._listener = None
        if sess.conn is not None:
            try:
                sess.conn.close()
            except OSError:
                pass

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
        if not isinstance(msg, dict) or sess is None or sess.closed or sess.disconnected:
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
        return (self._proc is not None and sess is not None and not sess.closed
                and not sess.disconnected and self._proc.poll() is None)


class ViewerLink:
    """Viewer-side end of the link, built from the --port/--authkey arguments."""

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


def parse_viewer_args(argv=None):
    """Return (port, authkey_hex) from the viewer command line, or (None, None) if absent."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--authkey", default=None)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else list(argv))
    if args.port is None or not args.authkey:
        return None, None
    return args.port, args.authkey
