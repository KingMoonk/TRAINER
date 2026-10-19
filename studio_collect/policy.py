"""What the collector may touch: the sites it never reads and the User-Agent it sends.

A source is collected only where its terms allow it. X, Quora and Reddit forbid scraping and machine-learning use without a
licence, so no collector exists for them and `Http` refuses their hosts whatever asks (a seed list, a redirect, a link on a
page). A site can be added in a future version only with a licence or official access whose terms allow it.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

REPO_URL = "https://github.com/KingMoonk/TRAINER"

# host suffix -> why it is excluded (shown on the Collect page)
EXCLUDED = {
    "x.com": "X forbids scraping and use of its content for machine learning without a licence.",
    "twitter.com": "X (Twitter) forbids scraping and use of its content for machine learning without a licence.",
    "t.co": "X (Twitter) link shortener; X forbids scraping without a licence.",
    "quora.com": "Quora forbids scraping and use of its content for machine learning without a licence.",
    "reddit.com": "Reddit forbids scraping and use of its content for machine learning without a licence.",
    "redd.it": "Reddit forbids scraping and use of its content for machine learning without a licence.",
    "redditmedia.com": "Reddit forbids scraping and use of its content for machine learning without a licence.",
    "redditstatic.com": "Reddit forbids scraping and use of its content for machine learning without a licence.",
}
SHOWN = ("x.com", "twitter.com", "quora.com", "reddit.com")  # the hosts the page lists (the rest are their aliases)


def user_agent():
    """An honest User-Agent: it names the app, says what it does and where to read about it."""
    try:
        from studio_api.version import app_version
        version = re.sub(r"[^\w.+-]", "-", app_version()) or "dev"
    except Exception:  # noqa: BLE001 - a missing git must not stop collection
        version = "dev"
    return f"GroundworkTrainer-Collector/{version} (+{REPO_URL}; passive training-data collector; obeys robots.txt)"


def host_of(url):
    return (urlsplit(url).hostname or "").lower().rstrip(".")


def excluded_reason(host_or_url):
    """Why a host (or a URL's host) is never collected, or "" when it is not excluded. Subdomains of an excluded host are excluded."""
    host = host_of(host_or_url) if "/" in host_or_url else host_or_url.lower().rstrip(".")
    for suffix, reason in EXCLUDED.items():
        if host == suffix or host.endswith("." + suffix):
            return reason
    return ""


def excluded_sites():
    """[{host, reason}] for the Collect page."""
    return [{"host": host, "reason": EXCLUDED[host]} for host in SHOWN]
