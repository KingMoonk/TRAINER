"""The one HTTP client nodes use to call each other's API (docs/distributed.md, "Identity"). Pure urllib: no server, no GPU.

`PeerClient(address)` takes a node as the workspace config lists it (a tailnet name, which gets studio_cluster.DEFAULT_PORT,
or a URL) and calls paths on it with JSON or binary bodies. A connection that fails or times out is retried with jittered
backoff; an answer, even an error status, never is: a 4xx means the request was wrong and a 5xx may have had an effect.
A node in `make dev` puts its web page at `/` and its API under `/api/` (tasks.py DEV_SERVE): when a peer answers with a web
page instead of the API, the call is made once more under `/api`, and the client keeps that prefix.
Every failure is a PeerError, with the status when the peer answered.
"""
from __future__ import annotations

import json
import random
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_PORT = 8787  # studio_cluster.DEFAULT_PORT; repeated so this module imports nothing of the app
TIMEOUT = 10.  # seconds per attempt
RETRIES = 2  # extra attempts after a connection failure
BLOCK = 1 << 20  # bytes read at a time when a body goes to a file
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
    def __init__(self, address, timeout=TIMEOUT, retries=RETRIES, opener=None, sleep=time.sleep, rng=None):
        self.address, self.base = address, base_url(address)
        self.timeout, self.retries = timeout, retries
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
        """GET a binary body into memory, BLOCK bytes at a time. `cancel()`, when it returns true between blocks, ends the call with
        Cancelled (no retry), so a download of tens of megabytes can be stopped within a block's time."""
        return self._call("GET", path, params, None, None, timeout, None, cancel)

    def get_file(self, path, target, params=None, timeout=None):
        """GET a binary body straight into the file `target`, a block at a time (a version's weights are too big to hold in memory).
        `timeout` bounds each wait for the peer, not the whole transfer. A retry starts the file over."""
        def save(response):
            with open(target, "wb") as stream:
                shutil.copyfileobj(response, stream, BLOCK)
        self._call("GET", path, params, None, None, timeout, save)

    def post_bytes(self, path, data, params=None, timeout=None):
        """POST a binary body (an .npz); extra fields go in the query string."""
        return _json(self._call("POST", path, params, bytes(data), "application/octet-stream", timeout))

    def _call(self, method, path, params, data, content_type, timeout, read=None, cancel=None):
        try:
            return self._send(method, path, params, data, content_type, timeout, read, cancel)
        except PeerError as first:
            if first.status is None or not is_page(first.status, first.detail) or self.base.endswith(API_PREFIX):
                raise
            self.base += API_PREFIX  # a web page, not the API: the node is in `make dev`
            try:
                return self._send(method, path, params, data, content_type, timeout, read, cancel)
            except PeerError as second:
                if second.status is not None and is_page(second.status, second.detail):  # no API under /api either
                    self.base = self.base[:-len(API_PREFIX)]
                    raise first from None
                raise

    def _send(self, method, path, params, data, content_type, timeout, read, cancel=None):
        url = self.url(path, params)
        request = urllib.request.Request(url, data=data, method=method)
        if content_type:
            request.add_header("Content-Type", content_type)
        delay = BACKOFF
        started, clock = time.time(), time.perf_counter()
        seen = dict(status=None, response=None, response_bytes=None, error="")

        def note():
            if _observers:
                event = dict(remote=self.address, method=method, path=f"/{path.lstrip('/')}", params=params, request=data,
                             request_type=content_type, seconds=time.perf_counter() - clock, started=started, **seen)
                for callback in list(_observers):
                    callback(event)
        for attempt in range(self.retries + 1):
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
                seen.update(status=status, response=None if read else result, response_bytes=None if read else len(result))
                note()
                return result
            except urllib.error.HTTPError as exc:  # the peer answered: never retried
                detail = _detail(exc)
                seen.update(status=exc.code, error=detail)
                note()
                raise PeerError(f"{self.address} answered {exc.code} to {method} /{path.lstrip('/')}: {detail}",
                                status=exc.code, detail=detail) from None
            except (urllib.error.URLError, OSError) as exc:  # not reached, refused, reset or timed out
                if attempt == self.retries:
                    reason = getattr(exc, "reason", exc)
                    seen.update(error=str(reason))
                    note()
                    raise PeerError(f"{self.address} did not answer {method} /{path.lstrip('/')}: {reason}") from None
                self._sleep(delay * (1 + self._rng.random() / 2))
                delay *= 2


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
