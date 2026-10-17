"""The Host check: a DNS-rebinding page cannot reach a loopback node, and nodes, browsers and localhost still can."""
import unittest
from urllib.parse import urlsplit

from fastapi import WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import studio_cluster
from studio_api.app import create_app
from studio_api.hosts import HostCheck, host_name, own_names, tailscale_names
from studio_ia.services import FakeServices

TS = ("mini.tail1234.ts.net", "mini")
FORGED = {"Tailscale-User-Login": "victim@example.com"}


def names(**kw):
    return own_names(bind=kw.pop("bind", "127.0.0.1"), tailscale=lambda: TS, hostname=lambda: "Mini.local", **kw)


class Client:
    def client(self, hosts, **kw):
        client = TestClient(create_app(FakeServices(), hosts=hosts, **kw))
        self.addCleanup(client.close)
        client.__enter__()
        return client


class Refusal(Client, unittest.TestCase):
    def test_a_forged_login_from_loopback_with_a_foreign_host_is_refused(self):
        client = self.client(names(), open=False)
        for host in ("evil.example.com", "evil.example.com:8787", "localhost.evil.example.com", "mini.evil.example.com"):
            response = client.get("/jobs", headers={"Host": host, **FORGED})
            self.assertEqual(response.status_code, 421, host)
        self.assertEqual(client.get("/jobs", headers={"Host": "evil.example.com"}).status_code, 421)

    def test_the_same_forged_login_with_a_local_host_is_let_in(self):
        client = self.client(names(), open=False)
        for host in ("localhost", "localhost:8787", "127.0.0.1:8787", "[::1]:8787", "LOCALHOST."):
            self.assertEqual(client.get("/jobs", headers={"Host": host, **FORGED}).status_code, 200, host)

    def test_an_open_node_refuses_a_foreign_host_too(self):
        self.assertEqual(self.client(names(), open=True).get("/jobs", headers={"Host": "evil.example.com"}).status_code, 421)

    def test_a_missing_host_is_refused(self):
        self.assertEqual(self.client(names(), open=True).get("/jobs", headers={"Host": ""}).status_code, 421)

    def test_a_websocket_with_a_foreign_host_is_closed_before_it_opens(self):
        app = create_app(FakeServices(), hosts=names(), open=True)

        @app.websocket("/ws-test")
        async def echo(websocket: WebSocket):
            await websocket.accept()
            await websocket.send_text("hi")

        with TestClient(app) as client:
            with self.assertRaises(WebSocketDisconnect), client.websocket_connect("/ws-test", headers={"Host": "evil.example.com"}):
                pass
            with client.websocket_connect("/ws-test", headers={"Host": "localhost:8787"}) as ws:
                self.assertEqual(ws.receive_text(), "hi")

    def test_allow_host_adds_a_name(self):
        extra = names(extra=["Trainer.example.com:443"])
        self.assertEqual(self.client(extra, open=True).get("/jobs", headers={"Host": "trainer.example.com"}).status_code, 200)
        self.assertEqual(self.client(names(), open=True).get("/jobs", headers={"Host": "trainer.example.com"}).status_code, 421)

    def test_none_skips_the_check(self):
        self.assertEqual(self.client(None, open=True).get("/jobs", headers={"Host": "evil.example.com"}).status_code, 200)


class Peers(Client, unittest.TestCase):
    """A peer calls `studio_cluster.node_url(address)`: urllib sends Host as that URL's host and port."""

    def test_every_way_a_peer_is_addressed_reaches_the_node(self):
        client = self.client(names(bind="100.64.0.7"), open=True)
        for address in ("mini", "mini:8787", "Mini.tail1234.ts.net", "https://mini.tail1234.ts.net", "100.64.0.7", "100.64.0.7:8787",
                        "[fd7a:115c:a1e0::1]:8787", "http://mini.tail1234.ts.net:8787/anything"):
            host = urlsplit(studio_cluster.node_url(address)).netloc
            self.assertEqual(client.get("/node", headers={"Host": host}).status_code, 200, address)

    def test_a_peer_reaching_the_node_under_a_name_it_does_not_have_is_refused(self):
        client = self.client(names(), open=True)
        host = urlsplit(studio_cluster.node_url("other-box")).netloc
        self.assertEqual(client.get("/node", headers={"Host": host}).status_code, 421)


class Names(unittest.TestCase):
    def test_host_name(self):
        self.assertEqual([host_name(v) for v in ("A.B:80", "[::1]:80", "[FD7A::1]", "x.", " y ")], ["a.b", "::1", "fd7a::1", "x", "y"])

    def test_own_names_include_the_machine_and_tailscale_names(self):
        got = names(bind="100.64.0.7", extra=["x.example.com"])
        self.assertTrue({"localhost", "mini", "mini.tail1234.ts.net", "mini.local", "100.64.0.7", "x.example.com"} <= got)

    def test_tailscale_names_from_status(self):
        class Done:
            returncode, stdout = 0, b'{"Self": {"DNSName": "mini.tail1234.ts.net.", "HostName": "Mini"}}'

        self.assertEqual(tailscale_names(run=lambda *a, **k: Done, binary=lambda: "tailscale"), ("mini.tail1234.ts.net", "mini"))

    def test_tailscale_names_are_empty_when_tailscale_is_missing_or_broken(self):
        self.assertEqual(tailscale_names(binary=lambda: None), ())

        class Bad:
            returncode, stdout = 0, b"not json"

        self.assertEqual(tailscale_names(run=lambda *a, **k: Bad, binary=lambda: "tailscale"), ())


class Wiring(unittest.TestCase):
    def test_the_check_is_the_outermost_layer(self):
        app = create_app(FakeServices(), hosts=["localhost"])
        self.assertIs(app.user_middleware[0].cls, HostCheck)


if __name__ == "__main__":
    unittest.main()
