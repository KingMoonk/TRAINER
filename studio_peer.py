"""The one HTTP client nodes use to call each other's API (docs/distributed.md, "Identity"). Pure urllib: no server, no GPU.

`PeerClient(address)` takes a node as the workspace config lists it (a tailnet name, which gets studio_cluster.DEFAULT_PORT,
or a URL) and calls paths on it with JSON or binary bodies. A connection that fails or times out is retried with jittered
backoff; an answer, even an error status, never is: a 4xx means the request was wrong and a 5xx may have had an effect.
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


class PeerError(Exception):
    """A call to a peer that failed. `status` is the HTTP status when the peer answered, None when it was not reached."""

    def __init__(self, message, status=None, detail=""):
        super().__init__(message)
        self.status, self.detail = status, detail


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

    def get_bytes(self, path, params=None, timeout=None):
        return self._call("GET", path, params, None, None, timeout)

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

    def _call(self, method, path, params, data, content_type, timeout, read=None):
        url = self.url(path, params)
        request = urllib.request.Request(url, data=data, method=method)
        if content_type:
            request.add_header("Content-Type", content_type)
        delay = BACKOFF
        for attempt in range(self.retries + 1):
            try:
                with self._open(request, timeout=timeout or self.timeout) as response:
                    return read(response) if read else response.read()
            except urllib.error.HTTPError as exc:  # the peer answered: never retried
                detail = _detail(exc)
                raise PeerError(f"{self.address} answered {exc.code} to {method} /{path.lstrip('/')}: {detail}",
                                status=exc.code, detail=detail) from None
            except (urllib.error.URLError, OSError) as exc:  # not reached, refused, reset or timed out
                if attempt == self.retries:
                    reason = getattr(exc, "reason", exc)
                    raise PeerError(f"{self.address} did not answer {method} /{path.lstrip('/')}: {reason}") from None
                self._sleep(delay * (1 + self._rng.random() / 2))
                delay *= 2


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
