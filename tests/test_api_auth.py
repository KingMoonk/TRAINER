"""The optional Vercel setup: --allow-origin / STUDIO_API_ALLOW_ORIGINS for CORS, and the Funnel token from the workspace config."""
import unittest
from unittest import mock

from fastapi import Request
from fastapi.testclient import TestClient

import studio_cluster
from studio_api.__main__ import ORIGINS, parse
from studio_api.app import create_app
from studio_ia.services import FakeServices

SITE = "https://trainer.example.vercel.app"
TOKEN = "made-up-token-for-tests"
LOGIN = {"Tailscale-User-Login": "me@example.com"}


def preflight(client, origin):
    return client.options("/jobs", headers={"Origin": origin, "Access-Control-Request-Method": "GET",
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


class Funnel(App):
    def test_a_funnel_style_request_needs_the_token(self):
        client = self.client(token=TOKEN)
        self.assertEqual(client.get("/jobs").status_code, 401)
        self.assertEqual(client.get("/jobs", headers={"X-Groundwork-Token": "wrong"}).status_code, 401)
        self.assertEqual(client.get("/jobs", headers={"X-Groundwork-Token": TOKEN}).status_code, 200)

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


class Config(unittest.TestCase):
    def test_funnel_token_from_the_workspace_config(self):
        self.assertEqual(studio_cluster.funnel_token({}), "")
        self.assertEqual(studio_cluster.funnel_token({"studioConfig": {"cluster": {"allowUsers": []}}}), "")
        self.assertEqual(studio_cluster.funnel_token({"studioConfig": {"cluster": {"funnelToken": f" {TOKEN} "}}}), TOKEN)

    def test_a_bad_token_stops_the_start_without_printing_it(self):
        for value in ("", "  ", 12, ["x"]):
            with self.assertRaisesRegex(ValueError, "funnelToken") as caught:
                studio_cluster.funnel_token({"studioConfig": {"cluster": {"funnelToken": value}}})
            self.assertNotIn(str(value).strip() or "\0", str(caught.exception))

    def test_origins_from_the_option_and_the_variable(self):
        with mock.patch.dict("os.environ", {}, clear=False) as env:
            env.pop(ORIGINS, None)
            self.assertEqual(parse(["--fake"]).allow_origin, [])
            args = parse(["--fake", "--allow-origin", "https://A.example.com/", "--allow-origin", "http://localhost:3000"])
            self.assertEqual(args.allow_origin, ["https://a.example.com", "http://localhost:3000"])
            env[ORIGINS] = f"{SITE}, https://a.example.com ,,"
            self.assertEqual(parse(["--fake", "--allow-origin", "https://a.example.com"]).allow_origin,
                             ["https://a.example.com", SITE])

    def test_a_malformed_origin_is_refused(self):
        for bad in ("*", "https://*.example.com", "example.com", "https://a.example.com/path", "ftp://a.example.com", "https://",
                    "https://u:p@a.example.com"):
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit, msg=bad):
                parse(["--fake", "--allow-origin", bad])
        with mock.patch.dict("os.environ", {ORIGINS: "*"}), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            parse(["--fake"])


if __name__ == "__main__":
    unittest.main()
