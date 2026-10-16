"""Trust the Tailscale-User-Login header that `tailscale serve` adds; refuse anything else unless the server is open."""

import json

HEADER = b"tailscale-user-login"
REFUSED = json.dumps(
    {"detail": "No Tailscale-User-Login header. Reach this node through `tailscale serve` (docs/node.md), "
     "or run it with --bind 127.0.0.1 or --fake."}
).encode()


class TailscaleAuth:
    def __init__(self, app, open=False):
        self.app = app
        self.open = open

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        user = ""
        for name, value in scope.get("headers", ()):
            if name == HEADER:
                user = value.decode("latin-1")
        if not user and not self.open:
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": REFUSED})
            return
        scope.setdefault("state", {})["user"] = user or "local"
        await self.app(scope, receive, send)
