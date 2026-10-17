"""The Host check: a DNS-rebinding page cannot reach a loopback node, and nodes, browsers and localhost still can."""
import unittest
from urllib.parse import urlsplit

from fastapi import WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import studio_cluster
from studio_api.app import create_app
import argparse

from studio_api.hosts import ANY, HostCheck, HostNames, allow_host, host_name, own_names, tailscale_names
from studio_ia.services import FakeServices

TS = ("mini.tail1234.ts.net", "mini")
FORGED = {"Tailscale-User-Login": "victim@example.com"}


def names(tailscale=lambda: TS, **kw):
    out = own_names(bind=kw.pop("bind", "127.0.0.1"), tailscale=tailscale, hostname=lambda: "Mini.local", **kw)
    out.refresh()
    return out


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

    def test_any_skips_the_check(self):
        self.assertEqual(self.client(ANY, open=True).get("/jobs", headers={"Host": "evil.example.com"}).status_code, 200)


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

    def test_host_name_is_strict(self):
        for bad in ("[::1]evil.com", "[::1]:80evil", "localhost:80@x", "user@localhost", "user:pw@localhost:80", "::1", "a b", "",
                    "a..b", "localhost:", "localhost:99999", "[1.2.3.4]", "[::1", "evil.com/x", "evil.com?x", "[]", "a.com:80:90"):
            self.assertIsNone(host_name(bad), bad)

    def test_a_malformed_host_is_refused(self):
        client = self.client_for(names())
        for host in ("[::1]evil.com", "localhost:80@x", "user@localhost"):
            self.assertEqual(client.get("/jobs", headers={"Host": host}).status_code, 421, host)

    def client_for(self, hosts):
        client = TestClient(create_app(FakeServices(), hosts=hosts, open=True))
        self.addCleanup(client.close)
        return client

    def test_own_names_include_the_machine_and_tailscale_names(self):
        got = names(bind="100.64.0.7", extra=["x.example.com"])
        self.assertTrue({"localhost", "mini", "mini.tail1234.ts.net", "mini.local", "100.64.0.7", "x.example.com"} <= got.exact | got.tailscale)

    def test_a_wildcard_stands_for_one_label(self):
        got = HostNames(["groundwork-dev.*.ts.net"])
        self.assertTrue(got.allowed("groundwork-dev.tail1234.ts.net:443"))
        for bad in ("groundwork-dev.ts.net", "groundwork-dev.a.b.ts.net", "groundwork-dev.tail1234.ts.net.evil.com", "x.groundwork-dev.a.ts.net"):
            self.assertFalse(got.allowed(bad), bad)

    def test_allow_host_validates(self):
        self.assertEqual(allow_host("Trainer.Example.com:443"), "trainer.example.com")
        self.assertEqual(allow_host("a.*.ts.net"), "a.*.ts.net")
        for bad in ("*", "*.com", "a.b.*", "http://x.com", "x.com/", "a*b.com", ""):
            with self.assertRaises(argparse.ArgumentTypeError, msg=bad):
                allow_host(bad)

    def test_tailscale_names_from_status(self):
        class Done:
            returncode = 0
            stdout = (b'{"Self": {"DNSName": "mini.tail1234.ts.net.", "HostName": "Mini"}, '
                      b'"CertDomains": ["mini.tail1234.ts.net", "alias.tail1234.ts.net"]}')

        self.assertEqual(tailscale_names(run=lambda *a, **k: Done, binary=lambda: "tailscale"),
                         ("mini.tail1234.ts.net", "mini", "alias.tail1234.ts.net"))

    def test_tailscale_names_are_empty_when_tailscale_is_missing_or_broken(self):
        self.assertEqual(tailscale_names(binary=lambda: None), ())

        class Bad:
            returncode, stdout = 0, b"not json"

        self.assertEqual(tailscale_names(run=lambda *a, **k: Bad, binary=lambda: "tailscale"), ())


class Refresh(unittest.TestCase):
    def test_names_missing_at_start_arrive_later_and_a_failed_read_keeps_the_good_ones(self):
        answers = [(), ("mini.tail1234.ts.net",), ()]
        got = own_names(tailscale=lambda: answers.pop(0), hostname=lambda: "box")
        with self.assertLogs("studio_api.hosts", "WARNING"):
            got.refresh()
        self.assertFalse(got.allowed("mini.tail1234.ts.net"))
        got.refresh()
        self.assertTrue(got.allowed("mini.tail1234.ts.net"))
        with self.assertLogs("studio_api.hosts", "WARNING"):
            got.refresh()
        self.assertTrue(got.allowed("mini.tail1234.ts.net"))

    def test_the_background_thread_reads_again_and_a_request_never_does(self):
        reads = []

        def read():
            reads.append(1)
            return ("mini.tail1234.ts.net",) if len(reads) > 1 else ()

        got = own_names(tailscale=read, hostname=lambda: "box")
        with self.assertLogs("studio_api.hosts", "WARNING"):
            got.start_refresh(first=.01, then=.01)
        self.addCleanup(got.stop.set)
        deadline = __import__("time").monotonic() + 5
        while not got.allowed("mini.tail1234.ts.net") and __import__("time").monotonic() < deadline:
            __import__("time").sleep(.01)
        self.assertTrue(got.allowed("mini.tail1234.ts.net"))
        got.stop.set()
        count = len(reads)
        got.allowed("evil.example.com")
        self.assertEqual(len(reads), count)


class Defaults(Client, unittest.TestCase):
    def test_the_default_fails_closed(self):
        client = self.client(())
        self.assertEqual(client.get("/jobs", headers={"Host": "evil.example.com"}).status_code, 421)
        self.assertEqual(client.get("/jobs", headers={"Host": "localhost"}).status_code, 401)  # let through to auth

    def test_the_421_names_the_fix_and_carries_cors_for_a_listed_origin_only(self):
        site = "https://trainer.example.vercel.app"
        client = self.client(names(), origins=[site])
        listed = client.get("/jobs", headers={"Host": "evil.example.com", "Origin": site})
        self.assertEqual((listed.status_code, listed.headers["access-control-allow-origin"]), (421, site))
        self.assertIn("--allow-host", listed.json()["detail"])
        other = client.get("/jobs", headers={"Host": "evil.example.com", "Origin": "https://evil.example.com"})
        self.assertNotIn("access-control-allow-origin", other.headers)


class Wiring(unittest.TestCase):
    def test_the_check_is_the_outermost_layer(self):
        app = create_app(FakeServices(), hosts=["localhost"])
        self.assertIs(app.user_middleware[0].cls, HostCheck)


if __name__ == "__main__":
    unittest.main()
