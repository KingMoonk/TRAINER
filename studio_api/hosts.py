"""Host header check (docs/node.md, "Host names"). A page that rebinds its own name to 127.0.0.1 is same-origin to itself, so it
could call a loopback node and forge `Tailscale-User-Login`. The browser still sends the page's own name as Host, so a request (or
websocket) whose Host is not one of this node's names is refused before auth sees it.

Allowed: localhost, any IP address (rebinding needs a name, and peers may call a node by its tailnet IP), this machine's name, the
node's Tailscale names (`tailscale status --json`, asked once at startup with a short timeout), and any `--allow-host` name.
The port is ignored.
"""

import json
import socket
import subprocess

from studio_api.auth import tailscale_binary

STATUS_TIMEOUT = 2.  # seconds for `tailscale status --json`; a node without tailscale just has fewer names
REFUSAL = "This node does not answer to that host name. Add it with --allow-host (docs/node.md)."


def host_name(value):
    """The lower-case name in a Host header or typed name, without port, brackets or trailing dot; "" if there is none."""
    text = value.strip().lower()
    if text.startswith("["):
        return text[1:].partition("]")[0]
    if text.count(":") == 1:
        text = text.partition(":")[0]
    return text.rstrip(".")


def is_ip(name):
    try:
        socket.inet_pton(socket.AF_INET6 if ":" in name else socket.AF_INET, name.split("%")[0])
    except (OSError, ValueError):
        return False
    return True


def tailscale_names(run=subprocess.run, binary=tailscale_binary):
    """This node's Tailscale names (MagicDNS name, its short form, host name) from `tailscale status --json`; () when tailscale
    is missing, slow or answers nonsense."""
    path = binary()
    if not path:
        return ()
    try:
        done = run([path, "status", "--json"], capture_output=True, timeout=STATUS_TIMEOUT, check=False)
        me = json.loads(done.stdout.decode("utf-8")).get("Self") if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
        return ()
    if not isinstance(me, dict):
        return ()
    names = [me.get("DNSName"), me.get("HostName")]
    return tuple(n for n in map(host_name, (v for v in names if isinstance(v, str))) if n)


def own_names(bind="", extra=(), tailscale=tailscale_names, hostname=socket.gethostname):
    """Every non-IP name this node answers to, besides localhost: its bound address, this machine's name, its Tailscale names
    and the `--allow-host` names, each lower-cased without port."""
    names = {"localhost", host_name(bind), *map(host_name, extra), *tailscale()}
    try:
        name = host_name(hostname())
        names.update((name, name.partition(".")[0]))
    except OSError:
        pass
    return frozenset(n for n in names if n)


class HostCheck:
    """ASGI middleware, outside auth and CORS: 421 for a request, a closed handshake for a websocket, when Host is not allowed."""

    def __init__(self, app, hosts):
        self.app, self.hosts = app, frozenset(host_name(h) for h in hosts)

    def allowed(self, value):
        name = host_name(value)
        return bool(name) and (name in self.hosts or is_ip(name))

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            return await self.app(scope, receive, send)
        host = next((v.decode("latin-1") for k, v in scope.get("headers", ()) if k == b"host"), "")
        if self.allowed(host):
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 1008})
        await send({"type": "http.response.start", "status": 421, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": json.dumps({"detail": REFUSAL}).encode()})
