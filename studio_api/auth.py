"""Who is calling (docs/distributed.md, "Identity"). The tailnet is the perimeter; a header a caller can type is not proof.

* Loopback peer (`tailscale serve` connects from 127.0.0.1): trust `Tailscale-User-Login`. serve deletes any value the client
  sent and writes its own (ipn/ipnlocal/serve.go, addTailscaleIdentityHeaders), and writes none for a tagged node.
* Any other peer address (a node calling the API on its tailnet address): ignore the header and ask the local tailscaled who
  owns the source address (`tailscale whois --json <ip>`). Let in a node with an allowed tag, or a user in the allow-list.
* `open` (--fake or a loopback bind): everyone is let in, as before.
* Funnel: `tailscale funnel` also connects from loopback but writes no `Tailscale-User-Login`. When the workspace config sets a
  token (studio_cluster.funnel_token), a loopback request without the header gets in as `funnel` if it carries that token in
  `X-Groundwork-Token`. A node with a token is not open: the token is the only way in for a caller with no Tailscale identity.
  The token is compared in constant time and is never logged or echoed.
* Browser pages on a configured origin (the Vercel site): their CORS preflight carries no token and no identity, so it is let
  through to the CORS middleware, which answers it (and refuses every other origin). The request that follows is checked as above.
"""

import asyncio
import hmac
import ipaddress
import json
import os
import shutil
import subprocess
import time

HEADER = b"tailscale-user-login"
TOKEN_HEADER = b"x-groundwork-token"
WHOIS_TIMEOUT = 3.  # seconds for the tailscale CLI to answer
CACHE_SECONDS = 60.  # how long an answer about an address is reused
MISS_SECONDS = 5.  # an address tailscaled could not name is asked again soon (tailscaled may just be starting)
TAILSCALE_PATHS = (r"C:\Program Files\Tailscale\tailscale.exe", "/Applications/Tailscale.app/Contents/MacOS/Tailscale")


def refusal(direct):
    if direct:
        return ("This node does not know the caller. Calls straight to its tailnet address are accepted only from a machine tagged "
                "with an allowed tag (default tag:trainer) or a user listed in studioConfig.cluster.allowUsers (docs/cluster.md).")
    return ("No Tailscale-User-Login header. Reach this node through `tailscale serve` (docs/node.md), "
            "or run it with --bind 127.0.0.1 or --fake.")


def token_matches(presented, token):
    """Whether `presented` (header bytes) is the configured `token` (text). An empty token or an empty header never matches."""
    expected = token.encode("utf-8")
    return bool(expected) and bool(presented) and hmac.compare_digest(presented, expected)


def tailscale_binary():
    found = shutil.which("tailscale")
    return found or next((p for p in TAILSCALE_PATHS if os.path.exists(p)), None)


def tailscale_whois(ip):
    """`tailscale whois --json <ip>` as a dict (WhoIsResponse: Node, UserProfile, CapMap), or None when tailscale is missing,
    does not know the address, or answers nonsense."""
    binary = tailscale_binary()
    if not binary:
        return None
    try:
        done = subprocess.run([binary, "whois", "--json", ip], capture_output=True, timeout=WHOIS_TIMEOUT, check=False)
        data = json.loads(done.stdout.decode("utf-8")) if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("Node"), dict) else None


def peer_ip(scope):
    """The caller's address as a string, with an IPv4 address mapped into IPv6 turned back into IPv4; "" if unknown."""
    client = scope.get("client")
    if not client:
        return ""
    try:
        addr = ipaddress.ip_address(client[0].split("%")[0])
    except ValueError:
        return ""
    return str(getattr(addr, "ipv4_mapped", None) or addr)


def is_loopback(ip):
    return bool(ip) and ipaddress.ip_address(ip).is_loopback


def caller_from_whois(who, allow_tags, allow_users):
    """The name to record for a whois answer, or "" when the caller is not allowed. A tagged node is `node:<hostname>`;
    a user's own machine is their login."""
    node = who.get("Node") or {}
    tags = node.get("Tags") or []
    if tags:  # a tagged node runs as its tags, whoever created it
        if not any(tag in allow_tags for tag in tags):
            return ""
        name = str(node.get("Name") or "").rstrip(".").split(".")[0]
        return f"node:{name or 'unknown'}"
    login = str((who.get("UserProfile") or {}).get("LoginName") or "")
    return login if login and login in allow_users else ""


class TailscaleAuth:
    def __init__(self, app, open=False, whois=tailscale_whois, allow_tags=("tag:trainer",), allow_users=(), clock=time.monotonic,
                 token="", origins=()):
        self.app, self.token, self.origins = app, token or "", tuple(origins)
        self.open = open and not self.token  # a node with a token is reached through serve or Funnel: nobody gets in unnamed
        self.whois, self.allow_tags, self.allow_users, self.clock = whois, tuple(allow_tags), tuple(allow_users), clock
        self.cache = {}  # ip -> (expires, caller)

    async def caller(self, ip):
        hit = self.cache.get(ip)
        if hit and hit[0] > self.clock():
            return hit[1]
        who = await asyncio.to_thread(self.whois, ip)
        caller = caller_from_whois(who, self.allow_tags, self.allow_users) if who else ""
        self.cache[ip] = (self.clock() + (CACHE_SECONDS if who else MISS_SECONDS), caller)
        return caller

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        ip = peer_ip(scope)
        header, presented, origin, preflight = "", b"", b"", False
        for name, value in scope.get("headers", ()):
            if name == HEADER:
                header = value.decode("latin-1")
            elif name == TOKEN_HEADER:
                presented = value
            elif name == b"origin":
                origin = value
            elif name == b"access-control-request-method":
                preflight = True
        if self.origins and preflight and scope.get("method") == "OPTIONS" and origin.decode("latin-1") in self.origins:
            return await self.app(scope, receive, send)  # the CORS middleware answers it; it has no token to check
        direct = bool(ip) and not is_loopback(ip)
        if self.open:
            user = header or "local"
        elif direct:  # the header is whatever the caller typed
            user = await self.caller(ip)
        else:
            user = header
            if not user and token_matches(presented, self.token):
                user = "funnel"
        if not user:
            await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": json.dumps({"detail": refusal(direct)}).encode()})
            return
        scope.setdefault("state", {})["user"] = user
        await self.app(scope, receive, send)
