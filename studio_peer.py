"""The one HTTP client nodes use to call each other's API (docs/distributed.md, "Identity"). Pure urllib: no server, no GPU.

`PeerClient(address)` takes a node as the workspace config lists it (a tailnet name, which gets studio_cluster.DEFAULT_PORT,
or a URL) and calls paths on it with JSON or binary bodies. A connection that fails or times out is retried with jittered
backoff; an answer, even an error status, never is: a 4xx means the request was wrong and a 5xx may have had an effect.
A node in `make dev` puts its web page at `/` and its API under `/api/` (tasks.py DEV_SERVE): when a peer answers with a web
page instead of the API, the call is made once more under `/api`, and the client keeps that prefix.
Every failure is a PeerError, with the status when the peer answered.

One TCP stream over a long, lossy link is capped by its loss (a 120 ms path with 1-2% loss holds one stream near 0.1-0.5 MB/s
whatever the lines at either end can carry), so a big body goes over several streams (docs/distributed.md, "Parallel transfers"):
get_bytes and get_file ask for the first `part` bytes with a Range header, and when the peer answers 206 with the body's size
they fetch the other parts at the same time, each retried on its own, and put them back in order; post_bytes cuts a body into
parts posted at the same time to `<path>/parts`, which the receiver assembles and checks against the whole body's hash. A peer
that answers 200 to a range, or has no parts route (404), gets one stream, as before.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import random
import shutil
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_PORT = 8787  # studio_cluster.DEFAULT_PORT; repeated so this module imports nothing of the app
TIMEOUT = 10.  # seconds per attempt
RETRIES = 2  # extra attempts after a connection failure
BLOCK = 1 << 20  # bytes read at a time when a body goes to a file
STREAMS = 8  # parallel streams for a big body (PeerClient `streams`; 1 turns it off)
PART = 4 << 20  # bytes per part: a body of at most this goes over one stream
PART_RETRIES = 2  # attempts added to the client's `retries` for one part: a part is retried on its own, not the whole body
READ = 64 << 10  # bytes read at a time inside a part, so a cancel is seen within a fraction of a second
POLL = .1  # seconds between looks at `cancel` while parts are in flight
BACKOFF = .5  # seconds before the first retry; doubled each time, with up to half again of jitter
API_PREFIX = "/api"  # where the API is on a node whose `/` is the dev web server
_observers = []  # callables told about every call (observe); while empty, a call costs one truthiness check


def observe(callback):
    """Tell `callback(event)` about every call a PeerClient in this process makes, once it is answered or has failed (the API's
    traffic console, studio_api.traffic). `event` is a dict: remote (the address as configured), method, path ("/..."), params,
    request (the bytes sent, or None), request_type, status (None when the peer was not reached), seconds, response (the bytes
    read, None when they went to a file or there was none), response_bytes, error (text, or ""), started (time.time()). The
    callback runs on the calling thread and must not raise. Returns the function that stops the notifications."""
    _observers.append(callback)

    def stop():
        if callback in _observers:
            _observers.remove(callback)
    return stop


class PeerError(Exception):
    """A call to a peer that failed. `status` is the HTTP status when the peer answered, None when it was not reached."""

    def __init__(self, message, status=None, detail=""):
        super().__init__(message)
        self.status, self.detail = status, detail


class Cancelled(PeerError):
    """A download the caller cancelled (PeerClient.get_bytes `cancel`)."""


def base_url(address):
    """The scheme, host and port of a configured node: a URL as given (any path dropped, so `https://box.tail1234.ts.net` stays on
    443, where `tailscale serve` listens), a bare name on DEFAULT_PORT."""
    address = address.strip()
    bare = "://" not in address
    scheme, _, rest = (f"http://{address}" if bare else address).partition("://")
    host = rest.split("/", 1)[0]
    if not host:
        raise ValueError(f"Not a node address: {address!r}.")
    if bare and ":" not in host.rsplit("]", 1)[-1]:
        host += f":{DEFAULT_PORT}"
    return f"{scheme}://{host}"


class PeerClient:
    def __init__(self, address, timeout=TIMEOUT, retries=RETRIES, opener=None, sleep=time.sleep, rng=None, streams=STREAMS, part=PART):
        self.address, self.base = address, base_url(address)
        self.timeout, self.retries = timeout, retries
        self.streams, self.part = max(1, int(streams)), max(1, int(part))  # settable: a member applies its own `streams` option
        self.parts_route = None  # whether the peer has the `/parts` upload route: None until asked, then True or False
        self._open = opener or urllib.request.urlopen  # urlopen(request, timeout=...) -> a response with .read() and .status
        self._sleep, self._rng = sleep, rng or random.Random()

    def url(self, path, params=None):
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        return f"{self.base}/{path.lstrip('/')}{query}"

    def get_json(self, path, params=None, timeout=None):
        return _json(self._call("GET", path, params, None, None, timeout))

    def post_json(self, path, body=None, params=None, timeout=None):
        data = json.dumps(body if body is not None else {}).encode("utf-8")
        return _json(self._call("POST", path, params, data, "application/json", timeout))

    def delete_json(self, path, params=None, timeout=None):
        return _json(self._call("DELETE", path, params, None, None, timeout))

    interruptible = True  # get_bytes takes `cancel`

    def get_bytes(self, path, params=None, timeout=None, cancel=None):
        """GET a binary body into memory, BLOCK bytes at a time (over `streams` parallel ranges when the peer serves them and the body
        is bigger than one part). `cancel()`, when it returns true, ends the call with Cancelled (no retry) within a fraction of a
        second, so a download of tens of megabytes can be stopped."""
        if self.streams > 1:
            return self._ranged(path, params, timeout, cancel, _Memory())
        return self._call("GET", path, params, None, None, timeout, None, cancel)

    def get_file(self, path, target, params=None, timeout=None):
        """GET a binary body straight into the file `target`, a block at a time (a version's weights are too big to hold in memory).
        `timeout` bounds each wait for the peer, not the whole transfer. A retry starts the file over."""
        if self.streams > 1:
            with _File(target) as sink:
                self._ranged(path, params, timeout, None, sink)
            return

        def save(response):
            with open(target, "wb") as stream:
                shutil.copyfileobj(response, stream, BLOCK)
        self._call("GET", path, params, None, None, timeout, save)

    def post_bytes(self, path, data, params=None, timeout=None):
        """POST a binary body (an .npz); extra fields go in the query string. A body bigger than one part goes as parts posted at
        the same time to `<path>/parts` (see _post_parts) when the peer has that route."""
        data = bytes(data)
        if self.streams > 1 and len(data) > self.part and self.parts_route is not False:
            sent = self._post_parts(path, data, params, timeout)
            if sent is not None:
                return sent
        return _json(self._call("POST", path, params, data, "application/octet-stream", timeout))

    # -- parallel transfers
    def _ranged(self, path, params, timeout, cancel, sink):
        """Fetch a body in parts of `self.part` bytes over `self.streams` connections into `sink`. The first request asks for part 0;
        its answer says whether the peer ranges (206, with the body's size in Content-Range) and the other parts start at once, while
        part 0 is still arriving. A 200 is the whole body (the peer does not range): it is read as one stream, nothing is fetched twice."""
        job = _Job(cancel)
        pool = ThreadPoolExecutor(max_workers=self.streams, thread_name_prefix="peer-range")
        try:
            pending, started = {pool.submit(self._first_part, path, params, timeout, job, sink)}, False
            while pending:
                done, pending = wait(pending, timeout=POLL, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()  # a failed part ends the whole transfer
                job.check()
                if not started and job.size.is_set():
                    started = True
                    for start in range(self.part, job.total or 0, self.part):
                        pending.add(pool.submit(self._next_part, path, params, timeout, job, sink, start))
            return sink.result()
        except BaseException:
            job.abort.set()
            raise
        finally:
            pool.shutdown(wait=False, cancel_futures=True)  # a part mid-read ends at its next READ bytes; nothing waits for it

    def _first_part(self, path, params, timeout, job, sink):
        def read(response):
            if getattr(response, "status", 200) != 206:  # the peer does not range: this is the whole body
                job.total = None
                job.size.set()
                return sink.whole(response, job)
            start, end, total = _content_range(response)
            if start != 0:
                raise PeerError(f"{self.address} answered a range from {start}, not 0")
            job.total = total
            job.size.set()  # the other parts can begin now
            part = _read_exact(response, min(end + 1, total), job)
            sink.put(0, part)
            return len(part)
        try:
            self._call("GET", path, params, None, None, timeout, read, None, {"Range": f"bytes=0-{self.part - 1}"}, PART_RETRIES)
        finally:
            job.size.set()  # a failure ends the loop through the future, which must not wait for a size

    def _next_part(self, path, params, timeout, job, sink, start):
        end = min(start + self.part, job.total) - 1

        def read(response):
            if getattr(response, "status", 200) != 206:
                raise PeerError(f"{self.address} did not answer the range {start}-{end} with 206", status=getattr(response, "status", 200))
            first, last, total = _content_range(response)
            if (first, last, total) != (start, end, job.total):
                raise PeerError(f"{self.address} answered bytes {first}-{last}/{total} for {start}-{end}/{job.total}: the body changed")
            part = _read_exact(response, end - start + 1, job)
            sink.put(start, part)
            return len(part)
        self._send("GET", path, params, None, None, timeout, read, None, {"Range": f"bytes={start}-{end}"}, PART_RETRIES)

    def _post_parts(self, path, data, params, timeout):
        """POST `data` as parts at the same time to `<path>/parts`. Each carries its index (`part`), the `count`, the body's `total`
        size and `sha256`, and its own `part_sha256`; the receiver answers 202 to each and, once all are in, checks the whole against
        `sha256` before it uses them. A first call with `count=0` asks whether the route exists: a peer that predates it answers 404,
        and None is returned (the caller posts the whole body in one request, as before, and this client stops asking)."""
        route, whole = f"{path.rstrip('/')}/parts", hashlib.sha256(data).hexdigest()
        base = {**(params or {}), "total": len(data), "sha256": whole}
        if self.parts_route is None:
            try:
                self._call("POST", route, {**base, "count": 0}, b"", "application/octet-stream", timeout)
            except PeerError as exc:
                if exc.status != 404:
                    raise
                self.parts_route = False
                return None
            self.parts_route = True
        count = -(-len(data) // self.part)
        abort = threading.Event()

        def send(index):
            body = data[index * self.part:(index + 1) * self.part]
            query = {**base, "count": count, "part": index, "part_sha256": hashlib.sha256(body).hexdigest()}
            if abort.is_set():
                raise Cancelled("The upload was stopped by a failed part.")
            return _json(self._send("POST", route, query, body, "application/octet-stream", timeout, None, None, None, PART_RETRIES))
        pool = ThreadPoolExecutor(max_workers=min(self.streams, count), thread_name_prefix="peer-parts")
        try:
            pending, answer = {pool.submit(send, index) for index in range(count)}, None
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    answer = future.result() or answer
            return answer if answer is not None else {}
        except BaseException:
            abort.set()
            raise
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _call(self, method, path, params, data, content_type, timeout, read=None, cancel=None, headers=None, retries=None):
        try:
            return self._send(method, path, params, data, content_type, timeout, read, cancel, headers, retries)
        except PeerError as first:
            if first.status is None or not is_page(first.status, first.detail) or self.base.endswith(API_PREFIX):
                raise
            self.base += API_PREFIX  # a web page, not the API: the node is in `make dev`
            try:
                return self._send(method, path, params, data, content_type, timeout, read, cancel, headers, retries)
            except PeerError as second:
                if second.status is not None and is_page(second.status, second.detail):  # no API under /api either
                    self.base = self.base[:-len(API_PREFIX)]
                    raise first from None
                raise

    def _send(self, method, path, params, data, content_type, timeout, read, cancel=None, headers=None, retries=None):
        url = self.url(path, params)
        request = urllib.request.Request(url, data=data, method=method)
        if content_type:
            request.add_header("Content-Type", content_type)
        for name, value in (headers or {}).items():
            request.add_header(name, value)
        retries = self.retries + (retries or 0)
        delay = BACKOFF
        started, clock = time.time(), time.perf_counter()
        seen = dict(status=None, response=None, response_bytes=None, error="")

        def note():
            if _observers:
                event = dict(remote=self.address, method=method, path=f"/{path.lstrip('/')}", params=params, request=data,
                             request_type=content_type, seconds=time.perf_counter() - clock, started=started, **seen)
                for callback in list(_observers):
                    callback(event)
        for attempt in range(retries + 1):
            try:
                with self._open(request, timeout=timeout or self.timeout) as response:
                    status = getattr(response, "status", 200)
                    if _html(response):  # Next's 308 to a trailing slash was followed to a web page: not the API
                        detail = response.read(300).decode("utf-8", "replace")
                        seen.update(status=status, error=detail)
                        note()
                        raise PeerError(f"{self.address} answered {status} to {method} /{path.lstrip('/')} with a web page, not the API",
                                        status=status, detail=detail)
                    result = read(response) if read else _slurp(response, cancel)
                seen.update(status=status, response=None if read else result, response_bytes=len(result) if not read else result if isinstance(result, int) else None)
                note()
                return result
            except urllib.error.HTTPError as exc:  # the peer answered: never retried
                detail = _detail(exc)
                seen.update(status=exc.code, error=detail)
                note()
                raise PeerError(f"{self.address} answered {exc.code} to {method} /{path.lstrip('/')}: {detail}",
                                status=exc.code, detail=detail) from None
            except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:  # not reached, refused, reset, cut short or timed out
                if attempt == retries:
                    reason = getattr(exc, "reason", exc)
                    seen.update(error=str(reason))
                    note()
                    raise PeerError(f"{self.address} did not answer {method} /{path.lstrip('/')}: {reason}") from None
                self._sleep(delay * (1 + self._rng.random() / 2))
                delay *= 2


class _Job:
    """One ranged download in flight: the body's size once known, and the flag that ends every part."""

    def __init__(self, cancel):
        self.cancel, self.total, self.size, self.abort = cancel, None, threading.Event(), threading.Event()

    def check(self):
        if self.cancel is not None and self.cancel():
            raise Cancelled("The download was cancelled.")


class _Memory:
    """Where a ranged download goes when it is wanted in memory: parts by offset, joined in order."""

    def __init__(self):
        self.parts, self.body = {}, None

    def put(self, offset, data):
        self.parts[offset] = data

    def whole(self, response, job):
        self.body = _slurp(response, job.check)
        return len(self.body)

    def result(self):
        return self.body if self.body is not None else b"".join(self.parts[offset] for offset in sorted(self.parts))


class _File:
    """Where it goes when it is wanted in a file: each part is written at its offset."""

    def __init__(self, target):
        self.target = target

    def __enter__(self):
        self.fd = os.open(self.target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o644)
        return self

    def __exit__(self, *exc):
        os.close(self.fd)

    def put(self, offset, data):
        view = memoryview(data)
        while view:  # pwrite may write less than asked
            written = os.pwrite(self.fd, view, offset)
            view, offset = view[written:], offset + written

    def whole(self, response, job):
        os.ftruncate(self.fd, 0)
        offset = 0
        while True:
            job.check()
            chunk = response.read(BLOCK)
            if not chunk:
                return offset
            self.put(offset, chunk)
            offset += len(chunk)

    def result(self):
        return None


def _content_range(response):
    """(first, last, total) of a 206 answer's `Content-Range: bytes first-last/total`; a PeerError when it is missing or unreadable."""
    value = (getattr(response, "headers", None) or {}).get("Content-Range") or ""
    try:
        unit, _, rest = value.partition(" ")
        span, _, total = rest.partition("/")
        first, _, last = span.partition("-")
        if unit.strip().lower() != "bytes":
            raise ValueError(value)
        return int(first), int(last), int(total)
    except ValueError:
        raise PeerError(f"A 206 answer without a usable Content-Range ({value!r})", status=206) from None


def _read_exact(response, length, job):
    """`length` bytes of a part, READ at a time; a body cut short is an OSError, which the part's retry takes up again."""
    chunks, got = [], 0
    while got < length:
        if job.abort.is_set():
            raise Cancelled("The download was cancelled.")
        chunk = response.read(min(READ, length - got))
        if not chunk:
            raise OSError(f"the body ended after {got} of {length} bytes")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def _slurp(response, cancel=None):
    """The whole body, read BLOCK bytes at a time (one unbuffered read of a big body is slow over a long link)."""
    parts = []
    while True:
        if cancel is not None and cancel():
            raise Cancelled("The download was cancelled.")
        chunk = response.read(BLOCK)
        if not chunk:
            return b"".join(parts)
        parts.append(chunk)


def _html(response):
    headers = getattr(response, "headers", None)
    return "text/html" in ((headers.get("Content-Type") or "") if headers is not None else "").lower()


def is_page(status, detail):
    """Whether an error answer came from a web server, not the API: an HTML page, or Next's 308 to a trailing slash (the API's
    own slash redirect is a 307), or a 200 page (the redirect followed)."""
    head = detail.lstrip()[:15].lower()
    return status == 308 or head.startswith("<!doctype html") or head.startswith("<html")


def _json(raw):
    return json.loads(raw.decode("utf-8")) if raw else None


def _detail(exc):
    try:
        body = exc.read().decode("utf-8", "replace")
    except Exception:
        return exc.reason or ""
    try:
        detail = json.loads(body).get("detail")
        return detail if isinstance(detail, str) else json.dumps(detail)
    except (ValueError, AttributeError):
        return body[:300]
