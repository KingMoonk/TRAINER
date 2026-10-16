"""studio_api.auth: who a caller is, over a fake whois. No tailscale, no network."""
import asyncio
import unittest

from studio_api.auth import TailscaleAuth, caller_from_whois, peer_ip


def node(name="mini.tail1234.ts.net.", tags=("tag:trainer",), login="tagged-devices"):
    return {"Node": {"Name": name, "Tags": list(tags)}, "UserProfile": {"LoginName": login}}


class Whois:
    def __init__(self, answers):
        self.answers, self.asked = answers, []

    def __call__(self, ip):
        self.asked.append(ip)
        return self.answers.get(ip)


def call(auth, client, headers=()):
    """Run one request through the middleware; returns (status or 200, scope state)."""
    seen = {}

    async def inner(scope, receive, send):
        seen.update(scope["state"])
        await send({"type": "http.response.start", "status": 200, "headers": []})

    sent = []

    async def send(message):
        sent.append(message)

    auth.app = inner
    scope = {"type": "http", "client": client, "headers": [(k.lower().encode(), v.encode()) for k, v in headers]}
    asyncio.run(auth(scope, None, send))
    return sent[0]["status"], seen


class Identity(unittest.TestCase):
    def auth(self, answers, **kw):
        self.whois = Whois(answers)
        self.now = [0.]
        return TailscaleAuth(None, whois=self.whois, clock=lambda: self.now[0], **kw)

    def test_a_tagged_node_calling_directly_gets_in_as_its_hostname(self):
        status, state = call(self.auth({"100.64.0.7": node()}), ("100.64.0.7", 5000))
        self.assertEqual((status, state["user"]), (200, "node:mini"))

    def test_the_header_is_ignored_on_the_direct_path(self):
        auth = self.auth({})
        status, _ = call(auth, ("100.64.0.9", 5000), [("Tailscale-User-Login", "me@example.com")])
        self.assertEqual(status, 401)

    def test_an_untagged_unknown_or_wrongly_tagged_caller_gets_401(self):
        auth = self.auth({"100.64.0.2": node(tags=("tag:other",)), "100.64.0.3": node(tags=(), login="eve@example.com")})
        for ip in ("100.64.0.2", "100.64.0.3", "100.64.0.4"):
            self.assertEqual(call(auth, (ip, 1))[0], 401, ip)

    def test_an_allowed_user_gets_in_by_login(self):
        auth = self.auth({"100.64.0.3": node(tags=(), login="me@example.com")}, allow_users=["me@example.com"])
        self.assertEqual(call(auth, ("100.64.0.3", 1)), (200, {"user": "me@example.com"}))

    def test_serve_path_trusts_the_header_from_loopback(self):
        auth = self.auth({})
        self.assertEqual(call(auth, ("127.0.0.1", 1), [("Tailscale-User-Login", "me@example.com")]), (200, {"user": "me@example.com"}))
        self.assertEqual(call(auth, ("::1", 1))[0], 401)
        self.assertEqual(self.whois.asked, [])

    def test_ipv4_mapped_addresses_are_looked_up_as_ipv4(self):
        self.assertEqual(peer_ip({"client": ("::ffff:100.64.0.7", 1)}), "100.64.0.7")
        self.assertEqual(call(self.auth({"100.64.0.7": node()}), ("::ffff:100.64.0.7", 1))[0], 200)

    def test_open_lets_everyone_in(self):
        auth = TailscaleAuth(None, open=True, whois=Whois({}))
        self.assertEqual(call(auth, ("100.64.0.9", 1)), (200, {"user": "local"}))

    def test_answers_are_cached_for_a_minute_and_misses_for_a_moment(self):
        auth = self.auth({"100.64.0.7": node()})
        call(auth, ("100.64.0.7", 1))
        call(auth, ("100.64.0.7", 2))
        self.assertEqual(self.whois.asked, ["100.64.0.7"])
        self.now[0] = 61
        call(auth, ("100.64.0.7", 3))
        self.assertEqual(len(self.whois.asked), 2)
        call(auth, ("100.64.0.8", 1))
        call(auth, ("100.64.0.8", 1))
        self.now[0] += 6
        call(auth, ("100.64.0.8", 1))
        self.assertEqual(self.whois.asked.count("100.64.0.8"), 2)

    def test_caller_from_whois(self):
        self.assertEqual(caller_from_whois(node(), ("tag:trainer",), ()), "node:mini")
        self.assertEqual(caller_from_whois(node(tags=("tag:a", "tag:trainer")), ("tag:trainer",), ()), "node:mini")
        self.assertEqual(caller_from_whois(node(tags=("tag:a",), login="me@example.com"), ("tag:trainer",), ("me@example.com",)), "")


if __name__ == "__main__":
    unittest.main()
