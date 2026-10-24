"""The collector's building blocks: licences, excluded sites, settings, cleaning, duplicates, and the polite HTTP layer."""
import tempfile
import unittest

import studio_core
from studio_collect import clean, config, licences, policy
from studio_collect.dedup import Seen
from studio_collect.netio import Http, Refused
from studio_collect.store import Ledger, LimitReached, Stopped
from tests.collect_fakes import FakeWeb, prose


class Licences(unittest.TestCase):
    def test_classes(self):
        for raw, klass in (("apache-2.0", "permissive"), ("MIT", "permissive"), ("Apache License 2.0", "permissive"), ("cc-by-4.0", "permissive"),
                           ("cc0-1.0", "permissive"), ("cc-by-sa-4.0", "share-alike"), ("odbl", "share-alike"), ("cdla-sharing-1.0", "share-alike"),
                           ("cc-by-nc-4.0", "non-commercial"), ("cc-by-nc-sa-4.0", "non-commercial"), ("cc-by-nd-4.0", "non-commercial"),
                           ("gpl-3.0", "copyleft"), ("other", "unknown"), ("", "unknown"), (None, "unknown"), ("llama2", "unknown"), ("weird-licence", "unknown")):
            self.assertEqual(licences.classify(raw)[1], klass, raw)

    def test_a_list_takes_the_strictest_class(self):
        self.assertEqual(licences.classify(["mit", "cc-by-nc-4.0"])[1], "non-commercial")

    def test_only_permissive_and_share_alike_are_kept_unless_opted_in(self):
        none = {"non_commercial": False, "unknown": False, "copyleft": False}
        self.assertTrue(licences.allowed("permissive", none) and licences.allowed("share-alike", none))
        for klass in ("unknown", "non-commercial", "copyleft"):
            self.assertFalse(licences.allowed(klass, none))
        self.assertTrue(licences.allowed("unknown", {**none, "unknown": True}))
        self.assertFalse(licences.allowed("non-commercial", {**none, "unknown": True}))


class Excluded(unittest.TestCase):
    def test_x_quora_and_reddit_with_their_subdomains(self):
        for host in ("x.com", "twitter.com", "www.quora.com", "old.reddit.com", "https://reddit.com/r/a", "i.redd.it"):
            self.assertTrue(policy.excluded_reason(host), host)
        for host in ("math.stackexchange.com", "notreddit.com", "https://en.wikibooks.org/wiki/A"):
            self.assertFalse(policy.excluded_reason(host), host)

    def test_the_user_agent_names_the_app(self):
        self.assertIn("GroundworkTrainer-Collector", policy.user_agent())

    def test_a_website_on_an_excluded_host_cannot_be_added(self):
        with self.assertRaisesRegex(ValueError, "cannot be added"):
            config.merge({"web": {"sites": [{"host": "www.reddit.com", "licence": "cc-by-4.0"}]}})

    def test_a_website_needs_a_licence_and_its_own_seeds(self):
        with self.assertRaisesRegex(ValueError, "licence"):
            config.merge({"web": {"sites": [{"host": "example.org"}]}})
        with self.assertRaisesRegex(ValueError, "must be on"):
            config.merge({"web": {"sites": [{"host": "example.org", "licence": "cc-by-4.0", "seeds": ["https://other.org/"]}]}})
        ok = config.merge({"web": {"sites": [{"host": "example.org", "licence": "cc-by-4.0"}]}})
        self.assertEqual(ok["web"]["sites"][0]["seeds"], ["https://example.org/"])


class Settings(unittest.TestCase):
    def test_defaults_are_off_and_cautious(self):
        base = config.merge({})
        self.assertFalse(base["enabled"])
        self.assertEqual(base["licences"], {"non_commercial": False, "unknown": False, "copyleft": False})
        self.assertTrue(base["idle"]["only_on_mains"] and base["idle"]["only_when_idle"] and not base["idle"]["unmetered_confirmed"])

    def test_bad_values_are_refused_in_words(self):
        for bad in ({"limits": {"disk_mb": -1}}, {"limits": {"nope": 1}}, {"topics": ["astrology"]}, {"mystery": 1}, {"limits": {"disk_mb": "big"}}):
            with self.assertRaises(ValueError):
                config.merge(bad)

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            config.save(root, {"enabled": True, "limits": {"disk_mb": 512}})
            saved = config.load(root)
            self.assertTrue(saved["enabled"])
            self.assertEqual(saved["limits"]["disk_mb"], 512)
            self.assertEqual(saved["limits"]["max_dataset_mb"], 200)


class Cleaning(unittest.TestCase):
    def test_personal_data_and_secrets_are_replaced(self):
        text = ("Write to jane.doe@example.com or call +1 415 555 0132 or (415) 555-0199. "
                "key = " + "AKIA" + "ABCDEFGHIJKLMNOP" + " and token: ghp_" + "a" * 36 + " and password = hunter2hunter2hunter2. "
                "See https://user:secretpass@example.org/x.")
        out, n = clean.strip_pii(text)
        for leaked in ("jane.doe", "555", "ABCDEFGHIJKL", "ghp_", "hunter2", "secretpass"):
            self.assertNotIn(leaked, out)
        self.assertIn("[EMAIL]", out)
        self.assertGreaterEqual(n, 6)

    def test_ordinary_numbers_survive(self):
        text = "In 1999 there were 2020 samples measured at 12.5 degrees over 3 weeks, version 1.2.3, ids 123456."
        self.assertEqual(clean.strip_pii(text), (text, 0))

    def test_quality(self):
        self.assertEqual(clean.text_problem(prose(1)), "")
        self.assertEqual(clean.text_problem("short"), "too short")
        self.assertEqual(clean.text_problem("the cat sat on the mat " * 60), "repeated phrases")
        self.assertTrue(clean.text_problem("\n".join(["same line of text here"] * 40)))
        self.assertTrue(clean.text_problem("#$%^&*() " * 60))
        self.assertEqual(clean.pair_problem("What is the boiling point of water at sea level?", "About 100 degrees Celsius at sea level."), "")
        self.assertEqual(clean.pair_problem("why?", "because"), "too short")


class Duplicates(unittest.TestCase):
    def example(self, text):
        return studio_core.normalize({"text": text})

    def test_exact_normalised_and_near_duplicates(self):
        with tempfile.TemporaryDirectory() as root:
            seen = Seen(root)
            base = prose(7, 12)
            first = self.example(base)
            keys, known = seen.exact(first)
            self.assertFalse(known)
            near, sig = seen.near(first)
            self.assertFalse(near)
            seen.add(keys, sig)
            self.assertTrue(seen.exact(self.example(base))[1])
            self.assertTrue(seen.exact(self.example("  " + base.upper().replace(" ", "  ")))[1])      # same words, other case and spacing
            self.assertTrue(seen.near(self.example(base + " One extra closing sentence."))[0])           # pending, not yet committed
            seen.commit()
            self.assertTrue(seen.near(self.example(base + " One extra closing sentence."))[0])           # after the commit, from disk
            self.assertFalse(seen.near(self.example(prose(8, 12)))[0])
            seen.close()
            again = Seen(root)
            self.assertTrue(again.exact(self.example(base))[1])
            again.close()

    def test_records_the_workspace_already_has_are_duplicates(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = studio_core.Workspace(root)
            ds = workspace.create_dataset("mine", storage="sqlite")
            workspace.add(ds, [{"text": prose(3)}])
            seen = Seen(root)
            self.assertEqual(seen.absorb_workspace(workspace), 1)
            self.assertTrue(seen.exact(self.example(prose(3)))[1])
            self.assertFalse(seen.exact(self.example(prose(4)))[1])
            self.assertEqual(seen.absorb_workspace(workspace), 0)
            seen.close()


class Polite(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.web = FakeWeb()
        self.web.add("https://ok.example/", b"hello")
        self.slept = []
        self.limits = config.merge({})["limits"]
        self.ledger = Ledger(self.tmp.name)

    def http(self, **kw):
        return Http(self.ledger, self.limits, transport=self.web, sleep=self.slept.append, **kw)

    def test_an_excluded_host_is_never_requested_not_even_by_redirect(self):
        http = self.http()
        for url in ("https://www.reddit.com/r/x", "https://x.com/a", "https://quora.com/q"):
            with self.assertRaises(Refused):
                http.get(url)
        self.web.add("https://redir.example/", b"", status=302, headers={"Location": "https://old.reddit.com/r/y"})
        with self.assertRaises(Refused):
            http.get("https://redir.example/a")
        self.assertEqual([u for u in self.web.log if "reddit" in u or "quora" in u or "x.com" in u], [])

    def test_every_request_says_who_asks(self):
        self.http().get("https://ok.example/a")
        self.assertIn("GroundworkTrainer-Collector", self.web.headers_log[0]["User-Agent"])

    def test_requests_to_one_host_are_spaced(self):
        elapsed = [0.]

        def sleep(seconds):
            self.slept.append(seconds)
            elapsed[0] += seconds

        http = Http(self.ledger, self.limits, transport=self.web, sleep=sleep, clock=lambda: elapsed[0])
        http.get("https://ok.example/a")
        http.get("https://ok.example/b")
        self.assertEqual(sum(self.slept), self.limits["seconds_between_requests"])

    def test_a_crawl_delay_raises_the_spacing_never_lowers_it(self):
        http = self.http()
        http.delay["ok.example"] = 10
        self.assertEqual(http.interval("ok.example"), 10)
        http.delay["ok.example"] = 0.1
        self.assertEqual(http.interval("ok.example"), self.limits["seconds_between_requests"])

    def test_requests_per_host_per_day_are_capped(self):
        self.limits = {**self.limits, "requests_per_host_per_day": 2}
        http = self.http()
        http.get("https://ok.example/a")
        http.get("https://ok.example/b")
        with self.assertRaisesRegex(LimitReached, "request cap"):
            http.get("https://ok.example/c")
        http.get("https://elsewhere.example/")          # another host has its own count

    def test_bandwidth_per_day_is_capped_while_streaming(self):
        self.web.add("https://big.example/", b"x" * 3_000_000)
        self.limits = {**self.limits, "bandwidth_mb_per_day": 1}
        reply = self.http().open("https://big.example/f")
        got = 0
        with self.assertRaisesRegex(LimitReached, "bandwidth"):
            for block in reply.chunks():
                got += len(block)
        self.assertLess(got, 3_000_000)

    def test_a_check_that_raises_stops_the_request(self):
        def stop():
            raise Stopped("paused")
        with self.assertRaises(Stopped):
            self.http(check=stop).get("https://ok.example/a")
        self.assertEqual(self.web.log, [])

    def test_only_web_addresses(self):
        with self.assertRaises(Refused):
            self.http().get("file:///etc/passwd")


if __name__ == "__main__":
    unittest.main()
