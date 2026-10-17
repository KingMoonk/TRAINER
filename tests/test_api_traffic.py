"""studio_api.traffic over FakeServices and a fake peer opener: the middleware's records, redaction (no secret survives), the ring bound,
GET /traffic (backfill, live stream, a stream's one open/close record), outbound peer calls tagged with the node, the alert rules and
the fake mode's traffic. No network, no sleeping on a clock."""
import io
import json
import unittest
import urllib.error
from unittest import mock

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from studio_api import traffic
from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.services import FakeServices
from studio_peer import PeerClient, PeerError
from tests.api_tap import EventTap

SECRET = "hunter2-do-not-leak"


def build(**options):
    services = FakeServices()
    app = create_app(services, open=True, hosts=ANY, **options)
    unittest.addModuleCleanup(app.state.traffic.close)  # stop listening to studio_peer

    @app.post("/echo")
    async def echo(request: Request):
        return {"got": json.loads(await request.body()), "api_key": SECRET}

    @app.post("/blob")
    async def blob(request: Request):
        return {"bytes": len(await request.body())}

    @app.get("/boom")
    def boom():
        raise HTTPException(503, "backend timeout")

    @app.get("/stream")
    def stream():
        return StreamingResponse((f"data: {n}\n\n" for n in range(3)), media_type="text/event-stream")

    return services, app


class Opener:
    """A urlopen that answers from a script: a body, an HTTP error status, or an unreachable peer."""

    def __init__(self, body=b'{"ok": true}', error=None, status=200):
        self.body, self.error, self.status = body, error, status

    def __call__(self, request, timeout=None):
        if isinstance(self.error, int):
            raise urllib.error.HTTPError(request.full_url, self.error, "refused", {}, io.BytesIO(b'{"detail": "stale epoch"}'))
        if self.error:
            raise urllib.error.URLError(self.error)
        return Response(self.body, self.status)


class Response(io.BytesIO):
    def __init__(self, body, status):
        super().__init__(body)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RedactionTest(unittest.TestCase):
    def test_a_secret_is_a_field_with_a_secret_word_in_its_name(self):
        for name in ("Authorization", "cookie", "Set-Cookie", "X-Groundwork-Token", "api_key", "apiKey", "API_KEY", "client_secret", "password", "Token"):
            self.assertTrue(traffic.secret_name(name), name)
        for name in ("tokens", "max_tokens", "monkey", "keyboard", "node", "generation", "content-type"):
            self.assertFalse(traffic.secret_name(name), name)

    def test_redact_reaches_every_depth(self):
        value = {"a": [{"inner": {"password": SECRET, "keep": 1}}], "token": [SECRET], "tokens": 5}
        self.assertEqual(traffic.redact(value), {"a": [{"inner": {"password": traffic.MASK, "keep": 1}}], "token": traffic.MASK, "tokens": 5})

    def test_no_secret_survives_a_request(self):
        services, app = build()
        with TestClient(app) as client:
            client.post("/echo?token=" + SECRET + "&node=BEAST&tokens=7", json={"password": SECRET, "nested": {"secret": SECRET, "n": 1}},
                        headers={"Authorization": f"Bearer {SECRET}", "Cookie": f"s={SECRET}", "X-Groundwork-Token": SECRET})
        (record,) = [r for r in app.state.traffic.snapshot() if r["path"] == "/echo"]
        self.assertNotIn(SECRET, json.dumps(record))
        self.assertEqual(record["request"]["headers"]["authorization"], traffic.MASK)
        self.assertEqual(record["request"]["query"], {"token": traffic.MASK, "node": "BEAST", "tokens": "7"})
        self.assertEqual(record["request"]["body"]["json"], {"password": traffic.MASK, "nested": {"secret": traffic.MASK, "n": 1}})
        self.assertEqual(record["response"]["body"]["json"]["api_key"], traffic.MASK)  # a response is redacted too
        self.assertEqual(record["remote"], "BEAST")  # a peer names itself in `node`

    def test_a_peer_call_is_redacted_too(self):
        _, app = build()
        PeerClient("beast", opener=Opener()).post_json("x", {"password": SECRET}, params={"token": SECRET, "round": 3})
        record = app.state.traffic.snapshot()[-1]
        self.assertNotIn(SECRET, json.dumps(record))
        self.assertEqual(record["request"]["query"], {"token": traffic.MASK, "round": "3"})


class MiddlewareTest(unittest.TestCase):
    def setUp(self):
        self.services, self.app = build()
        self.client = self.enterContext(TestClient(self.app))
        self.tap = self.app.state.traffic

    def test_a_request_is_recorded_with_its_route_and_models(self):
        self.client.get("/node")
        (record,) = self.tap.snapshot()
        self.assertEqual((record["direction"], record["method"], record["path"], record["route"], record["handler"], record["model"], record["status"]),
                         ("in", "GET", "/node", "/node", "node", "Node", 200))
        self.assertEqual((record["state"], record["streaming"], record["node"]), ("closed", False, self.tap.node))
        self.assertGreaterEqual(record["ms"], 0)
        self.assertIn("device", record["response"]["body"]["json"])

    def test_a_route_with_a_path_parameter_keeps_the_template(self):
        self.client.get("/collective/nope/rounds/3", params={"term": 1})
        record = self.tap.snapshot()[-1]
        self.assertEqual((record["route"], record["status"]), ("/collective/{run}/rounds/{round}", 404))

    def test_the_traffic_routes_are_not_recorded(self):
        self.client.get("/traffic/rules")
        self.assertEqual(self.tap.snapshot(), [])

    def test_binary_and_large_bodies_are_only_summarised(self):
        self.client.post("/blob", content=b"\x00" * 5000, headers={"Content-Type": "application/octet-stream"})
        big = json.dumps({"password": SECRET, "pad": "x" * traffic.BODY_LIMIT})
        self.client.post("/echo", content=big, headers={"Content-Type": "application/json"})
        binary, large = self.tap.snapshot()
        self.assertEqual(binary["request"]["body"], {"type": "application/octet-stream", "bytes": 5000, "binary": True})
        self.assertEqual(large["request"]["body"]["truncated"], True)
        self.assertNotIn(SECRET, json.dumps(large))
        self.assertNotIn("pad", json.dumps(large))

    def test_the_ring_is_bounded(self):
        small = traffic.Traffic(node="n", size=5)
        for n in range(8):
            small.add({"path": f"/{n}"})
        rows = small.snapshot()
        self.assertEqual([r["path"] for r in rows], [f"/{n}" for n in range(3, 8)])
        self.assertEqual([r["id"] for r in rows], [4, 5, 6, 7, 8])
        self.assertEqual([r["id"] for r in small.snapshot(2)], [7, 8])
        self.assertEqual(small.snapshot(0), [])
        self.assertEqual(traffic.RING, 2000)

    def test_a_5xx_is_told_to_every_browser_as_an_event(self):
        with EventTap(self.app) as tap:
            self.assertEqual(self.client.get("/boom").status_code, 503)
            (event,) = tap.wait("api_error")
        self.assertEqual((event["status"], event["path"], event["handler"]), (503, "/boom", "boom"))

    def test_nothing_is_encoded_while_no_console_is_connected(self):
        with mock.patch.object(traffic, "sse") as encoded:
            self.client.get("/node")
            self.client.get("/boom")
            self.assertEqual((self.tap.subscribers, encoded.call_count), (0, 0))
        self.assertEqual(len(self.tap.snapshot()), 2)  # recorded all the same: a console that connects later gets them


class StreamTest(unittest.TestCase):
    def setUp(self):
        self.services, self.app = build()
        self.client = self.enterContext(TestClient(self.app))

    def test_backfill_then_the_live_stream(self):
        self.client.get("/node")
        self.client.get("/boom")
        with EventTap(self.app, "/traffic?last=1") as tap:
            tap.wait("record")
            self.assertEqual([r["path"] for r in tap.of("record")], ["/boom"])  # the newest one only
            self.assertEqual(tap.of("hello")[0]["size"], traffic.RING)
            self.client.get("/node")
            tap.wait("record", 2)
        self.assertEqual([r["path"] for r in tap.of("record")], ["/boom", "/node"])

    def test_a_stream_is_one_record_that_opens_and_closes(self):
        with EventTap(self.app, "/traffic") as tap:
            self.assertEqual(self.client.get("/stream").text, "data: 0\n\ndata: 1\n\ndata: 2\n\n")
            tap.wait_for(lambda: [r["state"] for r in tap.of("record") if r["path"] == "/stream"] == ["open", "closed"], "open then closed")
        rows = [r for r in tap.of("record") if r["path"] == "/stream"]
        self.assertEqual(len({r["id"] for r in rows}), 1)
        self.assertTrue(rows[0]["streaming"] and rows[0]["ms"] is None and rows[1]["ms"] >= 0)
        self.assertIsNone(rows[1]["response"]["body"])  # chunks are never kept
        self.assertEqual(len([r for r in self.app.state.traffic.snapshot() if r["path"] == "/stream"]), 1)

    def test_the_traffic_stream_does_not_record_itself(self):
        with EventTap(self.app, "/traffic") as tap:
            tap.wait("hello")
        self.assertEqual(self.app.state.traffic.snapshot(), [])


class PeerTest(unittest.TestCase):
    def setUp(self):
        _, self.app = build()
        self.tap = self.app.state.traffic

    def test_an_outbound_call_is_tagged_with_the_node_it_went_to(self):
        PeerClient("beast.tail1234.ts.net", opener=Opener(b'{"accepted": true}', status=202)).post_bytes(
            "collective/r1/rounds/7/delta", b"\x00" * 300, params={"node": "Mac", "tokens": 1638400})
        (record,) = self.tap.snapshot()
        self.assertEqual((record["direction"], record["remote"], record["method"], record["path"], record["status"]),
                         ("out", "beast.tail1234.ts.net", "POST", "/collective/r1/rounds/7/delta", 202))
        self.assertEqual(record["request"]["query"], {"node": "Mac", "tokens": "1638400"})
        self.assertEqual(record["request"]["body"], {"type": "application/octet-stream", "bytes": 300, "binary": True})
        self.assertEqual(record["response"]["body"]["json"], {"accepted": True})

    def test_a_refusal_keeps_its_status_and_a_dead_peer_has_none(self):
        with self.assertRaises(PeerError):
            PeerClient("beast", opener=Opener(error=409)).post_json("cluster/jobs/j1/renew", {})
        with self.assertRaises(PeerError):
            PeerClient("gone", opener=Opener(error="refused"), retries=0).get_json("node")
        refused, dead = self.tap.snapshot()
        self.assertEqual((refused["remote"], refused["status"], refused["error"]), ("beast", 409, "stale epoch"))
        self.assertEqual((dead["remote"], dead["status"]), ("gone", None))
        self.assertIn("refused", dead["error"])

    def test_a_binary_answer_is_a_size(self):
        PeerClient("beast", opener=Opener(b"\x93NUMPY" + b"\x00" * 100)).get_bytes("collective/r1/params", {"round": 3})
        self.assertEqual(self.tap.snapshot()[0]["response"]["body"], {"type": "", "bytes": 106, "binary": True})

    def test_a_closed_app_stops_listening(self):
        self.app.state.traffic.close()
        PeerClient("beast", opener=Opener()).get_json("node")
        self.assertEqual(self.tap.snapshot(), [])


class RulesTest(unittest.TestCase):
    def setUp(self):
        self.services, self.app = build()
        self.client = self.enterContext(TestClient(self.app))

    def test_the_defaults_are_served_until_rules_are_saved(self):
        rules = self.client.get("/traffic/rules").json()["rules"]
        self.assertEqual([r["id"] for r in rules], [r.id for r in traffic.DEFAULT_RULES])
        self.assertEqual({r["id"] for r in rules if not r["enabled"]}, {"checkpoint.written"})
        self.assertEqual({r["severity"] for r in rules}, {"success", "info", "warning", "destructive"})

    def test_saved_rules_are_stored_server_side_and_announced(self):
        rules = self.client.get("/traffic/rules").json()["rules"]
        rules[0]["enabled"] = False
        with EventTap(self.app) as tap:
            self.assertEqual(self.client.put("/traffic/rules", json={"rules": rules}).status_code, 200)
            (event,) = tap.wait("traffic_rules")
        self.assertFalse(event["rules"][0]["enabled"])
        self.assertFalse(self.client.get("/traffic/rules").json()["rules"][0]["enabled"])
        self.assertEqual(self.services.setting("trafficRules")[0]["id"], "run.finished")

    def test_bad_rules_are_refused(self):
        rule = self.client.get("/traffic/rules").json()["rules"][0]
        self.assertEqual(self.client.put("/traffic/rules", json={"rules": [rule, rule]}).status_code, 400)
        self.assertEqual(self.client.put("/traffic/rules", json={"rules": [{**rule, "severity": "loud"}]}).status_code, 422)
        self.assertEqual(self.client.put("/traffic/rules", json={"rules": [{**rule, "id": str(n)} for n in range(traffic.MAX_RULES + 1)]}).status_code, 400)
        self.assertEqual(self.services.setting("trafficRules"), None)


class FakeModeTest(unittest.TestCase):
    def test_the_fake_run_brings_a_peer_with_it(self):
        services, app = build(fake_traffic=True)
        services.emit("train", {"job": "j1", "type": "stats", "step": 9})
        self.assertEqual(app.state.traffic.snapshot(), [])
        services.emit("train", {"job": "j1", "type": "stats", "step": 10})
        rows = app.state.traffic.snapshot()
        self.assertEqual([(r["method"], r["route"], r["status"]) for r in rows], [
            ("GET", "/collective/{run}/params", 200), ("GET", "/collective/{run}/rounds/{round}", 200),
            ("POST", "/collective/{run}/rounds/{round}/delta", 202), ("POST", "/cluster/jobs/{job_id}/renew", 200)])
        self.assertTrue(all(r["synthetic"] and r["remote"] == "BEAST" for r in rows))
        self.assertEqual(rows[2]["request"]["query"]["tokens"], "40960")

    def test_without_the_flag_nothing_is_made_up(self):
        services, app = build()
        services.emit("train", {"job": "j1", "type": "stats", "step": 10})
        self.assertEqual(app.state.traffic.snapshot(), [])


if __name__ == "__main__":
    unittest.main()
