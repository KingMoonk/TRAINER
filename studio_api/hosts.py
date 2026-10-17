"""Host header check (docs/node.md, "Host names"). A page that rebinds its own name to 127.0.0.1 is same-origin to itself, so it
could call a loopback node and forge `Tailscale-User-Login`. The browser still sends the page's own name as Host, so a request (or
websocket) whose Host is not one of this node's names is refused before auth sees it.

Allowed: localhost, any IP address (rebinding needs a name, and peers may call a node by its tailnet IP), this machine's name,
the node's Tailscale names, the bind address and the `--allow-host` names. The port is ignored. The Host must be strictly
`name`, `ipv4` or `[ipv6]`, with an optional `:port`; anything else (userinfo, text after the brackets) is refused.
Tailscale names are read at startup and again in a background thread, never while a request waits.
"""

import argparse
import json
import logging
import re
import socket
import subprocess
import threading

from studio_api.auth import tailscale_binary

log = logging.getLogger("studio_api.hosts")
STATUS_TIMEOUT = 2.  # seconds for `tailscale status --json`; a node without tailscale just has fewer names
REFRESH_EMPTY = 60.  # seconds between reads while no Tailscale name is known
REFRESH_KNOWN = 600.  # seconds between reads once some are
REFUSAL = "This node does not answer to that host name. Add it with --allow-host (docs/node.md)."
HOST = re.compile(r"(?:\[(?P<v6>[0-9a-f:.]+)\]|(?P<name>[a-z0-9_.*-]+))(?::(?P<port>\d{1,5}))?", re.IGNORECASE)


class AnyHost:
    """Pass as `hosts` to create_app to skip the check (tests, the OpenAPI dump)."""

    def __repr__(self):
        return "ANY"


ANY = AnyHost()


def is_ip(name):
    try:
        socket.inet_pton(socket.AF_INET6 if ":" in name else socket.AF_INET, name)
    except (OSError, ValueError):
        return False
    return True


def host_name(value):
    """The lower-case name in a Host header or typed name, without port, brackets or one trailing dot; None when `value` is not
    strictly `name`, `a.b.c.d` or `[ipv6]`, each with an optional `:port`."""
    match = HOST.fullmatch(value.strip())
    if not match:
        return None
    port = match["port"]
    if port is not None and int(port) > 65535:
        return None
    if match["v6"] is not None:
        return match["v6"].lower() if ":" in match["v6"] and is_ip(match["v6"]) else None
    name = match["name"].lower()
    name = name[:-1] if name.endswith(".") else name
    if not name or any(not label for label in name.split(".")):
        return None
    return name


def allow_host(value):
    """argparse type for --allow-host: a host name, or a pattern like `groundwork-dev.*.ts.net` whose `*` stands for one label."""
    name = host_name(value)
    labels = (name or "").split(".")
    if name is None or ("*" in name and (sum("*" not in v for v in labels) < 2 or "*" in labels[-1]
                                      or any("*" in v and v != "*" for v in labels))):
        raise argparse.ArgumentTypeError(f"{value!r} is not a host name like trainer.example.com (a * may stand for one label, "
                                         "as in node.*.ts.net).")
    return name


def tailscale_names(run=subprocess.run, binary=tailscale_binary):
    """This node's Tailscale names from `tailscale status --json`: its MagicDNS name and short form, its host name and its
    certificate domains; () when tailscale is missing, slow or answers nonsense."""
    path = binary()
    if not path:
        return ()
    try:
        done = run([path, "status", "--json"], capture_output=True, timeout=STATUS_TIMEOUT, check=False)
        status = json.loads(done.stdout.decode("utf-8")) if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return ()
    me = status.get("Self") if isinstance(status, dict) else None
    if not isinstance(me, dict):
        return ()
    dns = me.get("DNSName")
    certs = status.get("CertDomains")
    values = [dns, dns.split(".")[0] if isinstance(dns, str) else None, me.get("HostName"),
              *(certs if isinstance(certs, list) else [])]
    return tuple(dict.fromkeys(n for n in (host_name(v) for v in values if isinstance(v, str)) if n))


class HostNames:
    """The names a node answers to. `fixed` never changes; the Tailscale names are replaced by `refresh`."""

    def __init__(self, fixed=(), tailscale=tailscale_names):
        names = [n for n in (host_name(v) for v in fixed) if n]
        self.exact = frozenset(n for n in names if "*" not in n) | {"localhost"}
        self.patterns = tuple(re.compile(r"[^.]+".join(map(re.escape, n.split("*")))) for n in names if "*" in n)
        self.tailscale, self.read = frozenset(), tailscale
        self.stop, self.warned = threading.Event(), False

    def allowed(self, value):
        name = host_name(value)
        return name is not None and (name in self.exact or name in self.tailscale or is_ip(name)
                                     or any(p.fullmatch(name) for p in self.patterns))

    def refresh(self):
        """Read the Tailscale names now; warns once each time they go from known to empty."""
        names = frozenset(self.read())
        if not names and (self.tailscale or not self.warned):
            log.warning("No Tailscale name known for this node: requests through `tailscale serve` or Funnel are refused until "
                        "tailscale answers (read again every %.0f s) or the name is given with --allow-host.", REFRESH_EMPTY)
        self.warned = True
        if names or not self.tailscale:
            self.tailscale = names  # a failed read after a good one keeps the good names
        return names

    def start_refresh(self, first=REFRESH_EMPTY, then=REFRESH_KNOWN):
        """Read now, then again in a daemon thread: every `first` seconds while no name is known, every `then` after."""
        self.refresh()

        def loop():
            while not self.stop.wait(then if self.tailscale else first):
                self.refresh()

        threading.Thread(target=loop, name="tailscale-host-names", daemon=True).start()


def own_names(bind="", extra=(), tailscale=tailscale_names, hostname=socket.gethostname):
    """The HostNames for a node: localhost, its bound address, this machine's name (full and short) and the `--allow-host` names.
    The Tailscale names come from `refresh()` / `start_refresh()`."""
    fixed = [bind, *extra]
    try:
        name = hostname()
        fixed += [name, name.partition(".")[0]]
    except OSError:
        pass
    return HostNames(fixed, tailscale)


class HostCheck:
    """ASGI middleware, outside auth and CORS: 421 for a request, a closed handshake for a websocket, when Host is not allowed.
    The 421 carries the CORS allow-origin for a listed web origin, so the page can show the message instead of a network error."""

    def __init__(self, app, hosts=(), origins=()):
        self.app, self.names, self.origins = app, hosts if isinstance(hosts, HostNames) else HostNames(hosts), tuple(origins)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            return await self.app(scope, receive, send)
        headers = scope.get("headers", ())
        host = next((v.decode("latin-1") for k, v in headers if k == b"host"), "")
        if self.names.allowed(host):
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 1008})
        answer = [(b"content-type", b"application/json")]
        origin = next((v for k, v in headers if k == b"origin"), b"")
        if origin.decode("latin-1") in self.origins:
            answer += [(b"access-control-allow-origin", origin), (b"vary", b"Origin")]
        await send({"type": "http.response.start", "status": 421, "headers": answer})
        await send({"type": "http.response.body", "body": json.dumps({"detail": REFUSAL}).encode()})
