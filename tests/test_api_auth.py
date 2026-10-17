"""The optional Vercel setup: --allow-origin / STUDIO_API_ALLOW_ORIGINS for CORS, and the Funnel token from the workspace config."""
import asyncio
import unittest
from unittest import mock

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

import studio_cluster
from studio_api.__main__ import ORIGINS, parse
from studio_api.app import create_app
from studio_api.auth import TailscaleAuth
from studio_ia.services import FakeServices

SITE = "https://trainer.example.vercel.app"
TOKEN = "made-up-token-for-tests-0123456789abcdef"  # not a real secret
FUNNEL = {"Tailscale-Funnel-Request": "?1"}
FUNNEL_PAIR = ("Tailscale-Funnel-Request", "?1")
LOGIN = {"Tailscale-User-Login": "me@example.com"}


def preflight(client, origin):
    return preflight_to(client, "/jobs", origin)


def preflight_to(client, path, origin):
    return client.options(path, headers={"Origin": origin, "Access-Control-Request-Method": "GET",
                                            "Access-Control-Request-Headers": "x-groundwork-token"})


class App(unittest.TestCase):
    def client(self, **kw):
        app = create_app(FakeServices(), **kw)
        client = TestClient(app)
        self.addCleanup(client.close)
        client.__enter__()
        return client


class Cors(App):
    def test_a_preflight_from_a_configured_origin_gets_the_cors_headers(self):
        response = preflight(self.client(origins=[SITE]), SITE)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], SITE)
        self.assertIn("x-groundwork-token", response.headers["access-control-allow-headers"].lower())

    def test_a_preflight_from_another_origin_gets_none(self):
        client = self.client(origins=[SITE])
        for origin in ("https://evil.example.com", SITE.replace("https", "http"), "http://localhost:3000"):
            response = preflight(client, origin)
            self.assertFalse([h for h in response.headers if h.startswith("access-control")], origin)
            self.assertGreaterEqual(response.status_code, 400)

    def test_only_an_authenticated_request_from_the_origin_passes_cors(self):
        client = self.client(origins=[SITE])
        refused = client.get("/jobs", headers={"Origin": SITE})
        self.assertEqual(refused.status_code, 401)
        self.assertNotIn("access-control-allow-origin", refused.headers)
        allowed = client.get("/jobs", headers={"Origin": SITE, **LOGIN})
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.headers["access-control-allow-origin"], SITE)

    def test_a_non_preflight_options_request_is_not_waved_through(self):
        self.assertEqual(self.client(origins=[SITE]).options("/jobs", headers={"Origin": SITE}).status_code, 401)

    def test_with_no_origin_configured_nothing_is_added(self):
        client = self.client(open=False)
        self.assertEqual(preflight(client, SITE).status_code, 401)
        self.assertNotIn("access-control-allow-origin", client.get("/jobs", headers={"Origin": SITE, **LOGIN}).headers)
        local = self.client(open=True)
        self.assertEqual(local.get("/jobs", headers={"Origin": "http://localhost:3000"}).headers["access-control-allow-origin"],
                         "http://localhost:3000")
        self.assertNotIn("access-control-allow-origin", local.get("/jobs", headers={"Origin": SITE}).headers)


class OpenNodeCors(App):
    """The default --bind 127.0.0.1 makes the node open and adds the local-origin CORS; the configured origin must still work."""

    def client(self, **kw):
        return super().client(open=True, origins=[SITE], token=TOKEN, **kw)

    def test_the_configured_origin_preflight_passes_on_an_open_node(self):
        response = preflight(self.client(), SITE)
        self.assertEqual((response.status_code, response.headers["access-control-allow-origin"]), (200, SITE))
        self.assertEqual(len(response.headers.get_list("vary")), 1)
        self.assertEqual([v.strip() for v in response.headers["vary"].split(",")].count("Origin"), 1)

    def test_another_origin_gets_none_and_localhost_gets_none_while_a_token_is_set(self):
        client = self.client()
        for origin in ("https://evil.example.com", "http://localhost:3000"):
            response = preflight(client, origin)
            self.assertFalse([h for h in response.headers if h.startswith("access-control")], origin)
            self.assertGreaterEqual(response.status_code, 400)

    def test_a_localhost_page_gets_no_cors_headers_even_with_the_token(self):
        for origins in ([SITE], []):
            client = super().client(open=True, origins=origins, token=TOKEN)
            response = client.get("/jobs", headers={"Origin": "http://localhost:3000", "X-Groundwork-Token": TOKEN})
            self.assertEqual(response.status_code, 200, origins)
            self.assertFalse([h for h in response.headers if h.startswith("access-control")], origins)
            self.assertFalse([h for h in preflight(client, "http://localhost:3000").headers if h.startswith("access-control")], origins)

    def test_an_authenticated_response_has_one_vary_and_one_allow_origin(self):
        response = self.client().get("/jobs", headers={"Origin": SITE, "X-Groundwork-Token": TOKEN})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get_list("vary"), ["Origin"])
        self.assertEqual(response.headers.get_list("access-control-allow-origin"), [SITE])

    def test_without_a_token_the_local_origin_and_the_configured_one_both_pass(self):
        client = super().client(open=True, origins=[SITE])
        self.assertEqual(preflight(client, SITE).headers["access-control-allow-origin"], SITE)
        self.assertEqual(preflight(client, "http://localhost:3000").headers["access-control-allow-origin"], "http://localhost:3000")
        self.assertEqual(client.get("/jobs", headers={"Origin": SITE}).headers.get_list("vary"), ["Origin"])


class MiddlewareOrder(unittest.TestCase):
    def test_auth_is_outside_the_cors_layer_for_configured_origins(self):
        app = create_app(FakeServices(), origins=[SITE], token=TOKEN)
        classes = [m.cls for m in app.user_middleware]
        self.assertEqual(classes, [TailscaleAuth, CORSMiddleware])  # first is outermost

    def test_a_preflight_from_a_listed_origin_never_reaches_a_route(self):
        app = create_app(FakeServices(), origins=[SITE], token=TOKEN)
        reached = []

        @app.options("/test-secret")
        def secret():
            reached.append(1)
            return {"secret": 1}
        with TestClient(app) as client:
            self.assertEqual(preflight_to(client, "/test-secret", SITE).status_code, 200)
            self.assertEqual(client.options("/test-secret", headers={"Origin": SITE}).status_code, 401)  # no request-method: not a preflight
            self.assertEqual(client.options("/test-secret").status_code, 401)
        self.assertEqual(reached, [])


class Funnel(App):
    def test_a_funnel_style_request_needs_the_token(self):
        client = self.client(token=TOKEN)
        self.assertEqual(client.get("/jobs").status_code, 401)
        self.assertEqual(client.get("/jobs", headers={"X-Groundwork-Token": "wrong"}).status_code, 401)
        self.assertEqual(client.get("/jobs", headers={"X-Groundwork-Token": TOKEN}).status_code, 200)

    def test_funnel_traffic_needs_the_token_even_on_an_open_node(self):
        client = self.client(open=True, token=TOKEN)
        self.assertEqual(client.get("/jobs", headers=FUNNEL).status_code, 401)
        self.assertEqual(client.get("/jobs", headers={**FUNNEL, "X-Groundwork-Token": "wrong"}).status_code, 401)
        self.assertEqual(client.get("/jobs", headers={**FUNNEL, "X-Groundwork-Token": TOKEN}).status_code, 200)

    def test_funnel_traffic_is_refused_with_no_token_configured(self):
        for kw in ({"open": True}, {"open": False}, {"open": True, "token": ""}, {}):
            client = self.client(**kw)
            for extra in ({}, {"X-Groundwork-Token": ""}, {"X-Groundwork-Token": TOKEN}, LOGIN):
                self.assertEqual(client.get("/jobs", headers={**FUNNEL, **extra}).status_code, 401, (kw, extra))
        self.assertEqual(self.client(open=True).get("/jobs").status_code, 200)  # not Funnel: open as before

    def test_a_login_header_does_not_get_funnel_traffic_in(self):
        self.assertEqual(self.client(open=False, token=TOKEN).get("/jobs", headers={**FUNNEL, **LOGIN}).status_code, 401)

    def test_a_node_with_a_token_is_not_open(self):
        self.assertEqual(self.client(open=True, token=TOKEN).get("/jobs").status_code, 401)
        self.assertEqual(self.client(open=True, token=TOKEN).get("/jobs", headers={"X-Groundwork-Token": TOKEN}).status_code, 200)

    def test_an_empty_or_unset_token_never_authenticates(self):
        for token in ("", None):
            client = self.client(token=token)
            for header in ("", " ", "None"):
                self.assertEqual(client.get("/jobs", headers={"X-Groundwork-Token": header}).status_code, 401, (token, header))
        self.assertEqual(self.client().get("/jobs", headers={"X-Groundwork-Token": ""}).status_code, 401)

    def test_the_login_header_still_works_and_the_token_is_not_echoed(self):
        client = self.client(token=TOKEN)
        self.assertEqual(client.get("/jobs", headers=LOGIN).status_code, 200)
        refused = client.get("/jobs", headers={"X-Groundwork-Token": "wrong"})
        self.assertNotIn(TOKEN, refused.text)
        self.assertNotIn("wrong", refused.text)

    def test_the_funnel_caller_is_recorded_as_funnel(self):
        app = create_app(FakeServices(), token=TOKEN)

        @app.get("/test-user")
        def user(request: Request):
            return {"user": request.state.user}
        with TestClient(app) as client:
            self.assertEqual(client.get("/test-user", headers={"X-Groundwork-Token": TOKEN}).json(), {"user": "funnel"})

    def test_a_direct_tailnet_caller_cannot_use_the_token(self):
        app = create_app(FakeServices(), token=TOKEN, whois=lambda ip: None)
        with TestClient(app, client=("100.64.0.9", 5000)) as client:
            self.assertEqual(client.get("/jobs", headers={"X-Groundwork-Token": TOKEN}).status_code, 401)

    def test_the_browser_flow_end_to_end(self):
        client = self.client(origins=[SITE], token=TOKEN)
        self.assertEqual(preflight(client, SITE).headers["access-control-allow-origin"], SITE)
        response = client.get("/jobs", headers={"Origin": SITE, "X-Groundwork-Token": TOKEN})
        self.assertEqual((response.status_code, response.headers["access-control-allow-origin"]), (200, SITE))
        self.assertEqual(client.get("/jobs", headers={"Origin": SITE}).status_code, 401)


def websocket(auth, headers=(), client=("127.0.0.1", 1)):
    """Run one websocket scope through the middleware; returns (messages sent, whether the app was reached)."""
    sent, reached = [], []

    async def inner(scope, receive, send):
        reached.append(scope["type"])

    async def send(message):
        sent.append(message)

    auth.app = inner
    asyncio.run(auth({"type": "websocket", "client": client, "headers": [(k.lower().encode(), v.encode()) for k, v in headers]},
                     None, send))
    return sent, reached


class Scopes(unittest.TestCase):
    def test_lifespan_passes_through(self):
        reached = []

        async def inner(scope, receive, send):
            reached.append(scope["type"])
        asyncio.run(TailscaleAuth(inner)({"type": "lifespan"}, None, None))
        self.assertEqual(reached, ["lifespan"])

    def test_an_unauthenticated_websocket_is_closed_before_the_app(self):
        sent, reached = websocket(TailscaleAuth(None))
        self.assertEqual((sent, reached), ([{"type": "websocket.close", "code": 1008}], []))

    def test_a_websocket_follows_the_http_rules(self):
        self.assertEqual(websocket(TailscaleAuth(None), [("Tailscale-User-Login", "me@example.com")])[1], ["websocket"])
        auth = TailscaleAuth(None, token=TOKEN)
        self.assertEqual(websocket(auth, [("X-Groundwork-Token", TOKEN)])[1], ["websocket"])
        self.assertEqual(websocket(auth, [("X-Groundwork-Token", "wrong")])[1], [])
        self.assertEqual(websocket(TailscaleAuth(None, open=True), [FUNNEL_PAIR])[1], [])
        self.assertEqual(websocket(TailscaleAuth(None, open=True))[1], ["websocket"])


class Config(unittest.TestCase):
    def test_funnel_token_from_the_workspace_config(self):
        self.assertEqual(studio_cluster.funnel_token({}), "")
        self.assertEqual(studio_cluster.funnel_token({"studioConfig": {"cluster": {"allowUsers": []}}}), "")
        self.assertEqual(studio_cluster.funnel_token({"studioConfig": {"cluster": {"funnelToken": f" {TOKEN} "}}}), TOKEN)

    def test_a_bad_token_stops_the_start_without_printing_it(self):
        bad = ("", "  ", 12, ["x"], "short-secret-value", "a" * 31, "é" * 40, "x" * 20 + "\n" + "y" * 20, "  " + "a" * 31 + "  ")
        for value in bad:
            with self.assertRaisesRegex(ValueError, "studioConfig.cluster.funnelToken") as caught:
                studio_cluster.funnel_token({"studioConfig": {"cluster": {"funnelToken": value}}})
            if isinstance(value, str) and value.strip():
                self.assertNotIn(value.strip(), str(caught.exception))

    def test_a_32_character_ascii_token_is_accepted(self):
        self.assertEqual(studio_cluster.funnel_token({"studioConfig": {"cluster": {"funnelToken": "a" * 32}}}), "a" * 32)

    def test_origins_from_the_option_and_the_variable(self):
        with mock.patch.dict("os.environ", {}, clear=False) as env:
            env.pop(ORIGINS, None)
            self.assertEqual(parse(["--fake"]).allow_origin, [])
            args = parse(["--fake", "--allow-origin", "https://A.example.com/", "--allow-origin", "http://localhost:3000"])
            self.assertEqual(args.allow_origin, ["https://a.example.com", "http://localhost:3000"])
            env[ORIGINS] = f"{SITE}, https://a.example.com ,,"
            self.assertEqual(parse(["--fake", "--allow-origin", "https://a.example.com"]).allow_origin,
                             ["https://a.example.com", SITE])

    def test_default_ports_are_dropped_and_others_kept(self):
        args = parse(["--fake", "--allow-origin", "https://X.example.com:443", "--allow-origin", "http://x.example.com:80/",
                      "--allow-origin", "https://x.example.com:8443", "--allow-origin", "http://localhost:3000",
                      "--allow-origin", "https://x.example.com", "--allow-origin", "http://[::1]:80", "--allow-origin", "https://[::1]:8443"])
        self.assertEqual(args.allow_origin, ["https://x.example.com", "http://x.example.com", "https://x.example.com:8443",
                                             "http://localhost:3000", "http://[::1]", "https://[::1]:8443"])

    def test_junk_origins_are_refused_and_the_scheme_is_lowercased(self):
        for bad in ("https://a.com\\", "https://a .com", "https://a.com:0", "https://a.com:", "https://a.com:99999", "https://a.com:80:80",
                    "https://-a.com", "https://a..com", "https://a.com\n.evil.com", "https://[::1", "https://[zz]", "https://a.com:٣"):
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit, msg=bad):
                parse(["--fake", "--allow-origin", bad])
        args = parse(["--fake", "--allow-origin", "HTTPS://A.Example.com:8443/", "--allow-origin", "HTTP://[::1]:3000"])
        self.assertEqual(args.allow_origin, ["https://a.example.com:8443", "http://[::1]:3000"])

    def test_a_malformed_origin_is_refused(self):
        for bad in ("*", "https://*.example.com", "example.com", "https://a.example.com/path", "ftp://a.example.com", "https://",
                    "https://u:p@a.example.com"):
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit, msg=bad):
                parse(["--fake", "--allow-origin", bad])
        with mock.patch.dict("os.environ", {ORIGINS: "*"}), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            parse(["--fake"])


if __name__ == "__main__":
    unittest.main()
