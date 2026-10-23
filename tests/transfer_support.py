"""A loopback peer for the parallel-transfer tests and the benchmark: a real HTTP server whose every connection is slow, as one stream
of a long, lossy link is (a fixed delay before the answer, and a ceiling on the bytes per second one connection carries).

`Peer(blob)` serves `GET /blob` (with Range, unless `ranges=False`), takes `POST /up` and, unless `parts=False`, `POST /up/parts`
(assembled by studio_parts.Assembler). `dev=True` puts the API under `/api` and answers a web page anywhere else, as `make dev` does.
"""
import hashlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import studio_parts


class Peer:
    def __init__(self, blob=b"", latency=0., rate=None, ranges=True, parts=True, dev=False, rendezvous=0):
        self.blob, self.latency, self.rate, self.ranges, self.parts, self.dev = blob, latency, rate, ranges, parts, dev
        # rendezvous=N: a part request (not part 0 of a download) is held until N part requests are in flight at once, so N
        # connections open together exactly when the client really runs N streams, however loaded the machine is. `peak` is the
        # most part requests ever in flight; `met` says whether N was reached (a client that serialises them waits out `patience`).
        self.rendezvous, self.patience, self.inflight, self.peak, self.met = rendezvous, 10., 0, 0, threading.Event()
        self.requests, self.uploads, self.fail = [], [], {}  # fail: a Range header -> how many times to cut that answer short
        self.assembler, self.lock = studio_parts.Assembler(), threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self.handler())
        self.server.daemon_threads = True
        self.server.handle_error = lambda request, address: None  # a client that hung up mid-answer (a cancel) is not news
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)

    @property
    def address(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def track(self, held):
        """Count a part request as in flight; when `held`, wait for the rendezvous first."""
        with self.lock:
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
            if self.rendezvous and self.inflight >= self.rendezvous:
                self.met.set()
        if held and self.rendezvous and not self.met.wait(self.patience):
            self.met.set()  # never reached: let the rest through at once, the test fails on `peak`

    def untrack(self):
        with self.lock:
            self.inflight -= 1

    def pace(self, size):
        if self.rate:
            time.sleep(size / self.rate)

    def handler(self):
        peer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def route(self):
                url = urlparse(self.path)
                path = url.path
                if peer.dev:
                    if not path.startswith("/api/"):
                        page = b"<!doctype html><html><body>web page</body></html>"
                        self.reply(200, page, "text/html")
                        return None
                    path = path[4:]
                return path, {key: values[0] for key, values in parse_qs(url.query).items()}

            def reply(self, status, body, kind="application/octet-stream", headers=None):
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                for start in range(0, len(body), 16384):
                    peer.pace(len(body[start:start + 16384]))
                    self.wfile.write(body[start:start + 16384])

            def do_GET(self):
                found = self.route()
                if found is None:
                    return
                path, _ = found
                header = self.headers.get("Range")
                peer.track(held=bool(header) and not header.startswith("bytes=0-"))
                try:
                    self.get(path, header)
                finally:
                    peer.untrack()

            def get(self, path, header):
                time.sleep(peer.latency)
                with peer.lock:
                    peer.requests.append(("GET", path, header))
                    cut = peer.fail.get(header, 0)
                    if cut:
                        peer.fail[header] = cut - 1
                if path != "/blob":
                    self.reply(404, b'{"detail": "no"}', "application/json")
                    return
                blob = peer.blob
                span = studio_parts.byte_range(header, len(blob)) if peer.ranges else None
                if span is None:
                    self.reply(200, blob)
                    return
                first, last = span
                part = blob[first:last + 1]
                if cut:  # promise the part, send half of it, and hang up
                    self.send_response(206)
                    self.send_header("Content-Length", str(len(part)))
                    self.send_header("Content-Range", f"bytes {first}-{last}/{len(blob)}")
                    self.end_headers()
                    self.wfile.write(part[:len(part) // 2])
                    self.close_connection = True
                    return
                self.reply(206, part, headers={"Content-Range": f"bytes {first}-{last}/{len(blob)}"})

            def do_POST(self):
                found = self.route()
                if found is None:
                    return
                path, query = found
                counted = path == "/up/parts" and query.get("count") != "0"
                if counted:
                    peer.track(held=True)
                try:
                    self.post(path, query)
                finally:
                    if counted:
                        peer.untrack()

            def post(self, path, query):
                time.sleep(peer.latency)
                size = int(self.headers.get("Content-Length") or 0)
                body = bytearray()
                while len(body) < size:
                    chunk = self.rfile.read(min(16384, size - len(body)))
                    if not chunk:
                        break
                    body += chunk
                    peer.pace(len(chunk))
                body = bytes(body)
                with peer.lock:
                    peer.requests.append(("POST", path, query.get("part")))
                if path == "/up":
                    with peer.lock:
                        peer.uploads.append(body)
                    self.reply(202, b'{"accepted": true}', "application/json")
                elif path == "/up/parts" and peer.parts:
                    if int(query["count"]) == 0:
                        self.reply(202, b'{"accepted": true}', "application/json")
                        return
                    try:
                        whole, _ = peer.assembler.add(("up",), int(query["part"]), int(query["count"]), int(query["total"]), query["sha256"],
                                                      body, query["part_sha256"])
                    except studio_parts.BadPart as exc:
                        self.reply(400, ('{"detail": "%s"}' % exc).encode(), "application/json")
                        return
                    if whole is not None:
                        with peer.lock:
                            peer.uploads.append(whole)
                    self.reply(202, b'{"accepted": true}', "application/json")
                else:
                    self.reply(404, b'{"detail": "Not Found"}', "application/json")

        return Handler


def blob(size, seed=7):
    """`size` bytes that differ everywhere, so a part put in the wrong place shows."""
    out, block = bytearray(), seed.to_bytes(4, "big")
    while len(out) < size:
        block = hashlib.sha256(block).digest()
        out += block
    return bytes(out[:size])
