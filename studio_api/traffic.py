"""The API's traffic tap: every request this node answers and every call it makes to a peer, kept in a ring buffer and streamed on
GET /traffic (the web console, a bottom drawer). The alert rules the console applies are stored here too (GET and PUT /traffic/rules).

What a record holds (`Traffic.add`): id, ts (epoch seconds), node (this machine), direction ("in" a request to this API, "out" a call
studio_peer made), remote (the peer an outbound call went to, or the `node` an inbound peer call names), method, path, route (the
path template), handler (the route function), model / request_model (the schema names the route declares), status, ms, state
("closed", or "open" while a stream is still running), streaming, error, and request / response, each with headers, query and a
body view. Streams (SSE) are one record that is "open" until the stream ends, never one per chunk.

Cost: the middleware always appends (a bounded deque, so the last RING records can be backfilled to a console that connects later);
JSON encoding and the fan-out to subscribers happen only while a console is connected. Secrets are masked before a record is stored
(`redact`): the buffer never holds one. Bodies are summarised, never kept raw: a JSON body up to BODY_LIMIT bytes is parsed and
redacted; a larger one, and any binary body (.npz weights, deltas, shards), is only a size and a type.
"""

import asyncio
import collections
import json
import re
import socket
import threading
import time
import typing
import urllib.parse
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import studio_peer
from studio_api.events import KEEPALIVE, sse

RING = 2000  # records kept, oldest dropped first
BODY_LIMIT = 16 * 1024  # bytes of a JSON body that are parsed and kept
MASK = "••••••"
PREFIX = "/traffic"  # this module's own routes are not recorded (the console would feed itself)
SECRET_WORDS = {"key", "keys", "secret", "secrets", "password", "passwords", "passwd", "token", "authorization", "cookie", "cookies", "auth"}
# "tokens" is not here on purpose: it is a count (a delta's `tokens` query field), not a credential.
WORDS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
MAX_RULES = 100


def secret_name(name) -> bool:
    """Whether a header, query or body field of this name carries a secret: one of its words (`apiKey`, `X-Groundwork-Token`,
    `client_secret`) is key, token, secret, password, authorization, auth or cookie."""
    return any(word.lower() in SECRET_WORDS for word in WORDS.findall(str(name)))


def redact(value, depth=0):
    """`value` with the value of every secret-named field replaced by MASK, at any depth."""
    if depth > 24:
        return MASK
    if isinstance(value, dict):
        return {k: MASK if secret_name(k) else redact(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v, depth + 1) for v in value]
    return value


def headers_view(pairs) -> dict:
    """Header (name, value) pairs, bytes or str, as a lowercase-name dict with secret headers masked; repeats are joined by ", "."""
    out = {}
    for name, value in pairs:
        name = (name.decode("latin-1") if isinstance(name, bytes) else name).lower()
        value = value.decode("latin-1") if isinstance(value, bytes) else str(value)
        out[name] = f"{out[name]}, {value}" if name in out else value
    return redact(out)


def query_view(query) -> dict:
    """A query string (bytes, str) or a params dict as a dict; a name used twice keeps a list. Secret-named fields are masked."""
    pairs = urllib.parse.parse_qsl(query.decode("latin-1") if isinstance(query, bytes) else query, keep_blank_values=True) \
        if isinstance(query, (bytes, str)) else [(k, str(v)) for k, v in (query or {}).items()]
    out = {}
    for key, value in pairs:
        out[key] = [*out[key], value] if isinstance(out.get(key), list) else [out[key], value] if key in out else value
    return redact(out)


def body_view(content_type, prefix: bytes, total: int):
    """What the console shows of a body: None for no body; {type, bytes, json} for a JSON body up to BODY_LIMIT (redacted); {type,
    bytes, form} for a small form; otherwise {type, bytes} with truncated (a JSON body over the limit: a cut-off document cannot be
    redacted, so none of it is kept) or binary. `prefix` is the first bytes of the body, at least min(total, BODY_LIMIT + 1)."""
    if not total:
        return None
    kind = (content_type or "").split(";")[0].strip().lower()
    view = {"type": kind, "bytes": total}
    is_json = kind == "application/json" or kind.endswith("+json")
    if (is_json or kind == "application/x-www-form-urlencoded") and total > BODY_LIMIT:
        return {**view, "truncated": True}
    if is_json:
        try:
            return {**view, "json": redact(json.loads(prefix))}
        except ValueError:
            return {**view, "invalid": True}
    if kind == "application/x-www-form-urlencoded":
        return {**view, "form": query_view(prefix)}
    return {**view, "binary": True}


def _type_name(annotation):
    if annotation is None:
        return None
    origin = typing.get_origin(annotation)
    if origin is not None:
        args = ", ".join(str(_type_name(a)) for a in typing.get_args(annotation))
        return f"{getattr(origin, '__name__', str(origin))}[{args}]"
    return getattr(annotation, "__name__", str(annotation))


def route_names(route):
    """(template, handler, model, request_model) of the route FastAPI matched, each None when it did not match one."""
    if route is None:
        return None, None, None, None
    field = getattr(route, "body_field", None)
    return (getattr(route, "path", None), getattr(route, "name", None), _type_name(getattr(route, "response_model", None)),
            _type_name(getattr(field, "type_", None)) if field is not None else None)


class Traffic:
    """The ring buffer and its subscribers. Thread-safe: requests are recorded on the event loop, peer calls on whatever thread made them."""

    def __init__(self, node=None, size=RING, emit=None):
        self.size, self._node, self._emit = size, node, emit
        self._ring, self._seq, self.started = collections.deque(maxlen=size), 0, time.time()
        self._clients = {}  # asyncio.Queue -> its loop
        self._lock = threading.Lock()
        self._stops = []

    @property
    def node(self):
        if self._node is None:
            self._node = socket.gethostname()
        return self._node

    def add(self, record: dict) -> dict:
        """Store `record` (id assigned) and hand a copy to every subscriber. Returns the stored record."""
        with self._lock:
            self._seq += 1
            record = {"id": self._seq, "node": self.node, **record}
            self._ring.append(record)
            self._publish(record)
        return record

    def update(self, record: dict, **changes):
        """Change a stored record (a stream closing) and send it again; a console replaces the record with the same id."""
        with self._lock:
            record.update(changes)
            self._publish(record)

    def snapshot(self, last=None) -> list:
        with self._lock:
            return self._copies(last)

    def attach(self, last=None):
        """(the last `last` records, a queue of every later one), atomically: nothing is missed or sent twice."""
        queue = asyncio.Queue(maxsize=self.size * 2)
        with self._lock:
            backfill = self._copies(last)
            self._clients[queue] = asyncio.get_running_loop()
        return backfill, queue

    def detach(self, queue):
        with self._lock:
            self._clients.pop(queue, None)

    @property
    def subscribers(self) -> int:
        return len(self._clients)

    def _copies(self, last):
        rows = list(self._ring) if last is None else list(self._ring)[-last:] if last else []
        return [dict(r) for r in rows]

    def _publish(self, record):
        if not self._clients:  # nobody reading: no copy, no encoding
            return
        copy = dict(record)
        for queue, loop in list(self._clients.items()):
            try:
                loop.call_soon_threadsafe(_put, queue, copy)
            except RuntimeError:
                self._clients.pop(queue, None)  # its loop is closed

    # recording ------------------------------------------------------------
    def record_peer(self, event: dict):
        """A studio_peer.observe callback: one outbound call, tagged with the node it went to."""
        remote = urllib.parse.urlsplit(studio_peer.base_url(event["remote"])).hostname or event["remote"]
        request = event["request"]
        self.add({
            "direction": "out", "remote": remote, "method": event["method"], "path": event["path"], "route": None, "handler": None,
            "model": None, "request_model": None, "ts": event["started"], "status": event["status"], "ms": round(event["seconds"] * 1000, 1),
            "state": "closed", "streaming": False, "error": event["error"] if not event["status"] or event["status"] >= 400 else "",
            "request": {"headers": headers_view([("content-type", event["request_type"])] if event["request_type"] else []),
                        "query": query_view(event["params"]), "body": body_view(event["request_type"], (request or b"")[:BODY_LIMIT + 1], len(request or b""))},
            "response": {"headers": {}, "body": _peer_response(event)},
        })

    def own(self, stop):
        """Call `stop()` when this Traffic closes."""
        self._stops.append(stop)

    def watch_peers(self):
        self.own(studio_peer.observe(self.record_peer))

    def error(self, record):
        """A 5xx this node answered, as a /events event, so a console that is closed can still raise it."""
        if self._emit is not None:
            self._emit("api_error", {key: record[key] for key in ("node", "method", "path", "route", "handler", "status", "ms")})

    def close(self):
        for stop in self._stops:
            stop()
        self._stops.clear()
        with self._lock:
            self._clients.clear()


def _put(queue, record):
    try:
        queue.put_nowait(record)
    except asyncio.QueueFull:
        pass  # a reader this far behind loses records, as the ring itself would


def _peer_response(event):
    raw = event["response"]
    if raw is None:
        return None if event["response_bytes"] is None else {"type": "", "bytes": event["response_bytes"], "binary": True}
    kind = "application/json" if raw[:1] in (b"{", b"[") else ""  # the helper hands over no headers: a JSON document starts with { or [
    return body_view(kind, raw[:BODY_LIMIT + 1], len(raw))


class Call:
    """One inbound request being recorded by TrafficMiddleware."""

    def __init__(self, traffic: Traffic, scope):
        self.traffic, self.scope = traffic, scope
        self.started, self.clock = time.time(), time.perf_counter()
        self.request = Capture()
        self.response = Capture()
        self.status, self.response_headers, self.streaming, self.record, self.done = None, [], False, None, False
        self.error = ""

    def on_response_start(self, message):
        self.status, self.response_headers = message["status"], message.get("headers", [])
        kind = dict(self.response_headers_lower()).get("content-type", "")
        self.response.content_type = kind
        if kind.startswith("text/event-stream"):
            self.streaming = True
            self.record = self.traffic.add(self.view(state="open", ms=None))

    def on_response_body(self, message):
        self.response.add(message.get("body", b""))
        if not message.get("more_body"):
            self.finish()

    def response_headers_lower(self):
        return [(n.decode("latin-1").lower(), v.decode("latin-1")) for n, v in self.response_headers]

    def view(self, state, ms):
        scope = self.scope
        template, handler, model, request_model = route_names(scope.get("route"))
        query = query_view(scope.get("query_string", b""))
        remote = query.get("node") if isinstance(query.get("node"), str) else None
        return {
            "direction": "in", "remote": remote, "method": scope["method"], "path": scope["path"], "route": template, "handler": handler,
            "model": model, "request_model": request_model, "ts": self.started, "status": self.status, "ms": ms, "state": state,
            "streaming": self.streaming, "error": self.error,
            "request": {"headers": headers_view(scope.get("headers", [])), "query": query,
                        "body": body_view(_header(scope, "content-type"),self.request.prefix, self.request.total)},
            "response": {"headers": headers_view(self.response_headers),
                         "body": None if self.streaming else body_view(self.response.content_type, self.response.prefix, self.response.total)},
        }

    def finish(self):
        if self.done:
            return
        self.done = True
        if self.status is None:
            self.status = 500  # the app raised before it answered
        ms = round((time.perf_counter() - self.clock) * 1000, 1)
        if self.record is not None:  # a stream: the open record closes
            self.traffic.update(self.record, state="closed", ms=ms, error=self.error)
            record = self.record
        else:
            record = self.traffic.add(self.view(state="closed", ms=ms))
        if self.status >= 500:
            self.traffic.error(record)


class Capture:
    """The first bytes of a body, enough for body_view, and its whole length."""

    def __init__(self):
        self.prefix, self.total, self.content_type = b"", 0, ""

    def add(self, chunk: bytes):
        self.total += len(chunk)
        if len(self.prefix) <= BODY_LIMIT:
            self.prefix += chunk[:BODY_LIMIT + 1 - len(self.prefix)]


def _header(scope, name):
    for key, value in scope.get("headers", []):
        if key.decode("latin-1").lower() == name:
            return value.decode("latin-1")
    return ""


class TrafficMiddleware:
    """Pure ASGI, so a streaming response is never buffered or delayed."""

    def __init__(self, app, traffic: Traffic):
        self.app, self.traffic = app, traffic

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"].startswith(PREFIX):
            return await self.app(scope, receive, send)
        call = Call(self.traffic, scope)

        async def receiving():
            message = await receive()
            if message["type"] == "http.request":
                call.request.add(message.get("body", b""))
            return message

        async def sending(message):
            if message["type"] == "http.response.start":
                call.on_response_start(message)
            elif message["type"] == "http.response.body":
                call.on_response_body(message)
            await send(message)

        try:
            await self.app(scope, receiving, sending)
        except Exception as exc:
            call.error = f"{type(exc).__name__}: {exc}"[:300]
            raise
        finally:
            call.finish()  # a stream the client left, or an app that raised


# the alert rules ------------------------------------------------------------------------------------------------------------

Severity = Literal["success", "info", "warning", "destructive"]


class Rule(BaseModel):
    """When the console raises an alert. `source` traffic: a record matches when its method (if given), route template or path (`*`
    stands for one path segment) and status (a class like 5xx, or one code) do. `source` event: an /events event of this `kind`.
    Either may then require the field at the dotted path `where` (in the record, or in the event payload) to equal `equals`, or with
    `new_min` to be a new lowest number for the rule."""

    id: str
    label: str
    severity: Severity
    enabled: bool = True
    source: Literal["traffic", "event"]
    method: str | None = None
    path: str | None = None
    status: str | None = None
    kind: str | None = None
    where: str | None = None
    equals: str | int | float | bool | None = None
    new_min: bool = False


class Rules(BaseModel):
    rules: list[Rule]


DEFAULT_RULES = [
    Rule(id="run.finished", label="Run finished", severity="success", source="event", kind="job", where="status", equals="done"),
    Rule(id="round.closed", label="Outer step applied", severity="success", source="traffic", method="GET",
         path="/collective/*/rounds/*", status="200", where="response.body.json.state", equals="closed"),
    Rule(id="val_loss.best", label="New best L_val", severity="success", source="event", kind="train", where="val_loss", new_min=True),
    Rule(id="member.joined", label="Worker joined", severity="info", source="traffic", method="POST", path="/collective/*/join", status="2xx"),
    Rule(id="member.left", label="Worker left", severity="info", source="traffic", method="POST", path="/collective/*/leave", status="2xx"),
    Rule(id="delta.refused", label="Δ refused (stale round, generation or epoch)", severity="warning", source="traffic",
         method="POST", path="/collective/*/rounds/*/delta", status="409"),
    Rule(id="lease.fenced", label="Lease renewal refused: the job moved", severity="warning", source="traffic",
         method="POST", path="/cluster/jobs/*/renew", status="409"),
    Rule(id="api.5xx", label="Server error (5xx)", severity="destructive", source="event", kind="api_error", where="status", equals="5xx"),
    Rule(id="checkpoint.written", label="Checkpoint written", severity="info", enabled=False, source="event", kind="checkpoint",
         where="phase", equals="done"),
]

router = APIRouter(tags=["traffic"])


def current_rules(services) -> list[Rule]:
    stored = services.setting("trafficRules")
    if stored is None:
        return DEFAULT_RULES
    return [Rule.model_validate(rule) for rule in stored]


@router.get("/traffic")
async def traffic(request: Request, last: int = Query(RING, ge=0, le=RING, description="how many of the newest records to send first")):
    """Server-sent events: `record` events, each a full record (see the module docstring). The newest `last` records come first, then
    every new one as it happens; a stream's record is sent again when it closes. A `hello` event first names this node and when its buffer began (ids restart with it)."""
    tap = request.app.state.traffic

    async def stream():
        backfill, queue = tap.attach(last)
        try:
            yield ": connected\n\n"
            yield sse("hello", {"node": tap.node, "size": tap.size, "count": len(backfill), "started": tap.started})
            for number, record in enumerate(backfill):
                yield sse("record", record)
                if number % 50 == 49:
                    await asyncio.sleep(0)  # let other requests in between a big backfill
            while not await request.is_disconnected():
                try:
                    yield sse("record", await asyncio.wait_for(queue.get(), KEEPALIVE))
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            tap.detach(queue)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)


@router.get("/traffic/rules", response_model=Rules)
def get_rules(request: Request):
    """The alert rules every browser applies: the saved ones, else the defaults."""
    return Rules(rules=current_rules(request.app.state.services))


@router.put("/traffic/rules", response_model=Rules)
def put_rules(body: Rules, request: Request):
    """Replace the alert rules; every connected browser is told (`traffic_rules` event)."""
    ids = [rule.id for rule in body.rules]
    if len(body.rules) > MAX_RULES or len(set(ids)) != len(ids):
        raise HTTPException(400, f"At most {MAX_RULES} rules, each with its own id.")
    services = request.app.state.services
    services.set_setting("trafficRules", [rule.model_dump() for rule in body.rules])
    services.emit("traffic_rules", {"rules": [rule.model_dump() for rule in body.rules]})
    return body


# fake mode ------------------------------------------------------------------------------------------------------------------


def simulate(traffic: Traffic, services, peer="BEAST"):
    """`--fake`: a made-up second machine takes part in the fake run, so the console has the exchanges of a DiLoCo round to show.
    Every 10th step of the fake job is a round: the peer fetches θ, long-polls the round, uploads a Δ and renews its lease. Every
    record is marked synthetic. Returns the function that stops it."""
    def watching(kind, payload):
        if kind != "train" or not isinstance(payload, dict) or payload.get("type") != "stats" or payload["step"] % 10:
            return
        run, number = payload["job"], payload["step"] // 10
        base = {"ts": time.time(), "state": "closed", "streaming": False, "error": "", "synthetic": True}

        def add(method, path, status, ms, query=None, request_body=None, response_body=None, route=None, handler=None, model=None):
            traffic.add({**base, "method": method, "path": path, "route": route or path, "handler": handler, "model": model,
                         "request_model": None, "status": status, "ms": ms, "remote": peer, "direction": "in",
                         "request": {"headers": {}, "query": query or {}, "body": request_body},
                         "response": {"headers": {}, "body": response_body}})
        bytes_ = 4_000_000
        add("GET", f"/collective/{run}/params", 200, 240.0, {"node": peer, "round": str(number)}, None,
            {"type": "application/octet-stream", "bytes": bytes_, "binary": True}, "/collective/{run}/params", "params")
        add("GET", f"/collective/{run}/rounds/{number}", 200, 1800.0, {"node": peer, "wait": "30"}, None,
            {"type": "application/json", "bytes": 120, "json": {"round": number, "state": "closed", "member": True, "current": number, "generation": 1}},
            "/collective/{run}/rounds/{round}", "round_view", "RoundView")
        add("POST", f"/collective/{run}/rounds/{number}/delta", 202, 90.0,
            {"node": peer, "generation": "1", "epoch": "1", "tokens": "40960", "steps": "10", "seconds": "2.0", "loss": "4.1"},
            {"type": "application/octet-stream", "bytes": bytes_, "binary": True}, {"type": "application/json", "bytes": 18, "json": {"accepted": True}},
            "/collective/{run}/rounds/{round}/delta", "delta", "Accepted")
        fenced = number % 7 == 0
        add("POST", f"/cluster/jobs/{run}/renew", 409 if fenced else 200, 30.0, {"node": peer}, None,
            {"type": "application/json", "bytes": 40, "json": {"detail": "stale epoch"} if fenced else {"ok": True}},
            "/cluster/jobs/{job_id}/renew", "renew")
    return services.watch(watching)
