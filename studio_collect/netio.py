"""Every request the collector makes goes through `Http`: honest User-Agent, excluded hosts refused, a pause between requests to
one host, and the daily caps for bandwidth and requests per host. No other module opens a connection.

The transport is injectable (tests pass a fake; nothing in the test suite touches a network). Redirects are followed here, one
hop at a time, so a redirect to an excluded host or to a path robots.txt forbids is refused like any other request.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from urllib.parse import urljoin, urlsplit

from studio_collect import policy
from studio_collect.store import LimitReached, Stopped

MAX_REDIRECTS = 5
TIMEOUT = 30
CHUNK = 256 * 1024


class Refused(Exception):
    """A request the collector will not make (excluded host, robots.txt, not http(s))."""


class Reply:
    """A transport's answer: status, lower-case headers, and the body as chunks."""

    def __init__(self, status, headers, chunks, close=lambda: None):
        self.status, self.headers, self._chunks, self._close = status, {k.lower(): v for k, v in headers.items()}, chunks, close

    def chunks(self):
        return self._chunks

    def close(self):
        self._close()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def urllib_transport(url, headers):
    """The real transport. A 3xx or 4xx is returned as a Reply, not raised; a network failure raises OSError."""
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        response = opener.open(urllib.request.Request(url, headers=headers), timeout=TIMEOUT)
    except urllib.error.HTTPError as error:
        response = error
    except urllib.error.URLError as error:
        raise OSError(str(error.reason)) from error

    def chunks():
        while block := response.read(CHUNK):
            yield block
    return Reply(response.status if hasattr(response, "status") else response.code, dict(response.headers.items()), chunks(), response.close)


class Http:
    """limits: the config's "limits" dict. ledger: store.Ledger. check(): raises Stopped when the work must stop (pause, not idle)."""

    def __init__(self, ledger, limits, transport=urllib_transport, sleep=time.sleep, clock=time.monotonic, guard=None, check=lambda: None, agent=None):
        self.ledger, self.limits, self.transport, self.sleep, self.clock = ledger, limits, transport, sleep, clock
        self.guard, self.check, self.agent = guard, check, agent or policy.user_agent()
        self.last = {}       # host -> when the last request to it started
        self.delay = {}      # host -> a seconds-between-requests floor set by its robots.txt crawl-delay

    def interval(self, host):
        return max(float(self.limits["seconds_between_requests"]), self.delay.get(host, 0.0))

    def _wait(self, host):
        gap = self.interval(host) - (self.clock() - self.last.get(host, -1e9))
        while gap > 0:
            self.check()
            step = min(gap, 0.5)
            self.sleep(step)
            gap -= step
        self.last[host] = self.clock()

    def open(self, url, headers=None, robots=False, byte_range=None):
        """The Reply for `url` after redirects. robots=True asks `guard(url)` (the crawler's robots.txt check) before each hop.
        byte_range=(start, end) asks for those bytes (end None: to the end)."""
        for hop in range(MAX_REDIRECTS + 1):
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                raise Refused(f"Not a web address: {url}")
            host = parts.hostname.lower()
            reason = policy.excluded_reason(host)
            if reason:
                raise Refused(f"{host} is never collected. {reason}")
            if robots and self.guard is not None:
                self.guard(url)
            self.check()
            self.ledger.check(host, self.limits)
            self._wait(host)
            send = {"User-Agent": self.agent, "Accept-Encoding": "identity", **(headers or {})}
            if byte_range:
                send["Range"] = f"bytes={byte_range[0]}-{'' if byte_range[1] is None else byte_range[1]}"
            self.ledger.request(host)
            reply = self.transport(url, send)
            if reply.status in (301, 302, 303, 307, 308) and reply.headers.get("location"):
                reply.close()
                url = urljoin(url, reply.headers["location"])
                continue
            return CountedReply(reply, self)
        raise Refused(f"Too many redirects from {url}")

    def get(self, url, headers=None, robots=False, limit=20_000_000):
        """(status, headers, body bytes) of a small response; ValueError when the body is larger than `limit`."""
        reply = self.open(url, headers, robots)
        try:
            body = bytearray()
            for block in reply.chunks():
                body += block
                if len(body) > limit:
                    raise ValueError(f"{url} is larger than {limit:,} bytes.")
            return reply.status, reply.headers, bytes(body)
        finally:
            reply.close()


    def size(self, url):
        """The size in bytes of the file at `url` (asked with a one-byte range), or None when the server does not say."""
        reply = self.open(url, byte_range=(0, 0))
        try:
            if reply.status == 206:
                total = reply.headers.get("content-range", "").rsplit("/", 1)[-1]
                return int(total) if total.isdigit() else None
            if reply.status == 200:
                return int(reply.headers.get("content-length") or 0) or None
            raise OSError(f"{url} answered {reply.status}.")
        finally:
            reply.close()


class CountedReply:
    """A Reply whose bytes are counted against the daily bandwidth cap as they arrive (LimitReached the moment it is used up)."""

    def __init__(self, reply, http):
        self.reply, self.http = reply, http
        self.status, self.headers = reply.status, reply.headers

    def chunks(self):
        for block in self.reply.chunks():
            self.http.check()
            full = None
            try:
                self.http.ledger.received(len(block), self.http.limits)
            except LimitReached as error:   # this block arrived and is kept; the stream stops after it
                full = error
            yield block
            if full:
                raise full

    def close(self):
        self.reply.close()


__all__ = ["Http", "Reply", "Refused", "LimitReached", "Stopped", "urllib_transport"]
