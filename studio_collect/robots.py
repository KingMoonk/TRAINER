"""robots.txt for the crawler: fetched once a day per host, obeyed (Disallow and Crawl-delay), and never bypassed.

A missing robots.txt (404, 410) allows the site; a server error or an unreachable host forbids it until the next try, because an
answer we could not read is not permission. The crawl-delay raises Http's spacing for that host; it never lowers it.
"""
from __future__ import annotations

import time
from urllib import robotparser
from urllib.parse import urlsplit

from studio_collect import policy
from studio_collect.netio import Refused

TOKEN = "GroundworkTrainer-Collector"
TTL = 24 * 3600


class Robots:
    def __init__(self, http, clock=time.time):
        self.http, self.clock, self.cache = http, clock, {}   # host -> (parser | "deny", fetched at)

    def _load(self, url):
        parts = urlsplit(url)
        host = parts.hostname.lower()
        hit = self.cache.get(host)
        if hit and self.clock() - hit[1] < TTL:
            return hit[0]
        parser = robotparser.RobotFileParser()
        try:
            reply = self.http.open(f"{parts.scheme}://{parts.netloc}/robots.txt")
            try:
                body = b"".join(reply.chunks())[:1_000_000]
            finally:
                reply.close()
            if reply.status in (404, 410) or 400 <= reply.status < 500 and reply.status not in (401, 403, 429):
                parser.parse([])                 # no robots.txt: everything allowed
            elif reply.status in (401, 403):
                parser.disallow_all = True      # RFC 9309: access denied means the site is closed to crawlers
            elif reply.status == 200:
                parser.parse(body.decode("utf-8", errors="replace").splitlines())
            else:
                parser = "deny"
        except Refused:
            raise
        except OSError:
            parser = "deny"
        self.cache[host] = (parser, self.clock())
        if parser != "deny":
            delay = parser.crawl_delay(TOKEN)
            if delay:
                self.http.delay[host] = max(self.http.delay.get(host, 0.0), float(delay))
        return parser

    def allowed(self, url):
        parser = self._load(url)
        return parser != "deny" and parser.can_fetch(TOKEN, url)

    def guard(self, url):
        """The Http guard: Refused unless robots.txt lets this crawler fetch `url`."""
        reason = policy.excluded_reason(url)
        if reason:
            raise Refused(f"{urlsplit(url).hostname} is never collected. {reason}")
        if urlsplit(url).path == "/robots.txt":
            return
        if not self.allowed(url):
            raise Refused(f"robots.txt does not allow {url}")
