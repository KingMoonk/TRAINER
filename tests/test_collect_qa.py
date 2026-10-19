"""Question-and-answer sources over fakes: a Stack Exchange dump fixture, a fake GitHub API, and the polite crawler."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock
from xml.etree import ElementTree

from studio_collect import config, htmltext, register, service, stackexchange
from studio_collect.netio import Http
from studio_collect.robots import Robots
from studio_collect.store import Ledger, State
from tests.collect_fakes import FakeWeb, idle_probes, prose


def posts_xml(rows):
    root = ElementTree.Element("posts")
    for attrs in rows:
        ElementTree.SubElement(root, "row", {k: str(v) for k, v in attrs.items()})
    return b'<?xml version="1.0" encoding="utf-8"?>\n' + ElementTree.tostring(root)


def zipped(data, name="Posts.xml"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr(name, data)
    return buffer.getvalue()


def drain(root, web, **kw):
    kw.setdefault("sleep", lambda s: None)
    for _ in range(60):
        outcome = service.run(root, transport=web, probes=idle_probes(), once=True, lock=False, **kw)
        if outcome in ("idle", "off", "error", "limit", "waiting"):
            return outcome
    raise AssertionError("the service never went idle")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.web = FakeWeb()

    def shards(self):
        out = []
        for files in register.pending(self.root).values():
            for path in files:
                out += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return out

    def enable(self, **overrides):
        config.save(self.root, {"enabled": True, "idle": {"unmetered_confirmed": True}, **overrides})


class StackExchange(Base):
    SITE = "cs.stackexchange.com"

    def dump(self):
        question = "<p>Why does quicksort degrade to quadratic time when the pivot is always the smallest element of the array?</p><pre><code>sort(a)</code></pre>"
        return posts_xml([
            dict(Id=1, PostTypeId=1, AcceptedAnswerId=2, Score=5, Title="Why is quicksort slow on sorted input?", Body=question, Tags="|sorting|", OwnerUserId=11, CreationDate="2012-01-05T10:00:00"),
            dict(Id=2, PostTypeId=2, ParentId=1, Score=1, Body="<p>Because every partition peels off one element, so the recursion depth is linear and the total work is quadratic.</p>", OwnerUserId=12, CreationDate="2019-03-03T10:00:00"),
            dict(Id=3, PostTypeId=2, ParentId=1, Score=20, Body="<p>A different, higher scored answer that is not the accepted one.</p>", OwnerUserId=13, CreationDate="2019-04-03T10:00:00"),
            dict(Id=4, PostTypeId=1, Score=3, Title="What does amortized analysis of a dynamic array show?", Body="<p>Explain why appending to a growable array costs constant time on average.</p>", Tags="|arrays|", OwnerUserId=14, CreationDate="2010-01-05T10:00:00"),
            dict(Id=5, PostTypeId=2, ParentId=4, Score=7, Body="<p>Doubling the capacity means each element is copied only a constant number of times overall.</p>", OwnerUserId=15, CreationDate="2010-02-05T10:00:00"),
            dict(Id=6, PostTypeId=2, ParentId=4, Score=1, Body="<p>Low scoring answer that should lose to the better one above.</p>", OwnerUserId=16, CreationDate="2010-03-05T10:00:00"),
            dict(Id=7, PostTypeId=1, Score=2, Title="A question nobody answered well enough", Body="<p>Is there any way to do this faster than linear time in general?</p>", Tags="|x|", OwnerUserId=17, CreationDate="2020-01-05T10:00:00"),
            dict(Id=8, PostTypeId=2, ParentId=7, Score=1, Body="<p>An answer under the score threshold, not accepted either.</p>", OwnerUserId=18, CreationDate="2020-01-06T10:00:00"),
            dict(Id=9, PostTypeId=1, Score=-3, Title="A downvoted question with an answer", Body="<p>Please just do my homework for me right now.</p>", Tags="|x|", OwnerUserId=19, CreationDate="2020-01-05T10:00:00"),
            dict(Id=10, PostTypeId=2, ParentId=9, Score=9, Body="<p>Answer to the downvoted question that must not be used.</p>", OwnerUserId=20, CreationDate="2020-01-06T10:00:00"),
        ])

    def serve(self, data):
        self.web.add(f"https://archive.org/download/stackexchange/{self.SITE}.7z", data)

    def test_pairs_carry_attribution_and_the_licence_of_their_date(self):
        self.serve(zipped(self.dump()))
        self.enable(sources={"hub": False, "stackexchange": True}, stackexchange={"sites": [self.SITE], "min_score": 3}, limits={"auto_approve_mb": 100})
        self.assertEqual(drain(self.root, self.web), "idle")
        rows = {r["provenance"]["question_id"]: r for r in self.shards()}
        self.assertEqual(sorted(rows), [1, 4])                         # not Q7 (weak answer), not Q9 (negative score)
        first = rows[1]
        self.assertEqual(first["messages"][1]["content"].startswith("Because every partition"), True)   # the accepted answer, not the top-scored one
        self.assertIn("```", first["messages"][0]["content"])           # code kept as a block
        p = first["provenance"]
        self.assertEqual((p["licence"], p["question_licence"], p["answer_licence"]), ("cc-by-sa-3.0", "cc-by-sa-3.0", "cc-by-sa-4.0"))
        self.assertEqual(p["source_url"], f"https://{self.SITE}/questions/1")
        self.assertIn(f"https://{self.SITE}/users/11", p["attribution"])
        self.assertIn(f"https://{self.SITE}/users/12", p["attribution"])
        self.assertIn("CC BY-SA 3.0", p["attribution"])
        self.assertEqual(rows[4]["provenance"]["licence"], "cc-by-sa-2.5")                               # the top answer over the threshold; a 2010 post is 2.5
        self.assertEqual(rows[4]["messages"][1]["content"].startswith("Doubling"), True)

    def test_only_the_dump_is_fetched_never_a_site_page(self):
        self.serve(zipped(self.dump()))
        self.enable(sources={"hub": False, "stackexchange": True}, stackexchange={"sites": [self.SITE]})
        drain(self.root, self.web)
        self.assertTrue(all(u.startswith("https://archive.org/download/stackexchange/") for u in self.web.log), self.web.log)

    def test_a_big_dump_waits_for_approval_and_an_unreadable_archive_says_why(self):
        self.serve(b"7z\xbc\xaf\x27\x1c" + b"0" * 2_000_000)
        self.enable(sources={"hub": False, "stackexchange": True}, stackexchange={"sites": [self.SITE]}, limits={"auto_approve_mb": 1})
        drain(self.root, self.web)
        self.assertEqual(State(self.root).data["plan"][0]["status"], "needs_approval")
        self.assertEqual(self.web.asked(".7z"), [f"https://archive.org/download/stackexchange/{self.SITE}.7z"])          # only the size was asked
        self.assertEqual(len(self.web.headers_log) and self.web.headers_log[-1]["Range"], "bytes=0-0")
        config.save(self.root, {"approved": [self.SITE]})
        with mock.patch.object(stackexchange, "seven_zip_command", return_value=None), mock.patch.dict("sys.modules", {"py7zr": None}):
            drain(self.root, self.web)
        source = State(self.root).data["sources"][f"stackexchange:{self.SITE}"]
        self.assertEqual(source["status"], "failed")
        self.assertIn("7-Zip", source["error"])

    def test_licence_by_date(self):
        for day, licence in (("2011-04-07T00:00:00", "cc-by-sa-2.5"), ("2011-04-08T00:00:00", "cc-by-sa-3.0"), ("2018-05-01T00:00:00", "cc-by-sa-3.0"), ("2018-05-02T00:00:00", "cc-by-sa-4.0")):
            self.assertEqual(stackexchange.licence_for(day), licence)
        self.assertEqual(stackexchange.dump_name("stackoverflow.com"), "stackoverflow.com-Posts.7z")

    def test_a_stopped_pairing_resumes_at_its_offset(self):
        self.serve(zipped(self.dump()))
        self.enable(sources={"hub": False, "stackexchange": True}, stackexchange={"sites": [self.SITE], "min_score": 3})
        clock = iter(range(0, 10 ** 6))
        calls = {"n": 0}

        def busy(root):
            calls["n"] += (Path(root) / "data" / "collect" / "downloads" / "stackexchange-cs-stackexchange-com" / "pairs.sqlite").exists()
            return calls["n"] > 1
        out = service.run(self.root, transport=self.web, probes=idle_probes(queue_busy=busy), sleep=lambda s: None, clock=lambda: next(clock), once=True, lock=False)
        self.assertEqual(out, "stopped")
        self.assertEqual(drain(self.root, self.web), "idle")
        self.assertEqual(sorted(r["provenance"]["question_id"] for r in self.shards()), [1, 4])


class GitHub(Base):
    def setUp(self):
        super().setUp()
        repos = [
            {"full_name": "acme/lib", "html_url": "https://github.com/acme/lib", "default_branch": "main", "license": {"spdx_id": "MIT"}, "stargazers_count": 900, "size": 50,
             "fork": False, "archived": False, "owner": {"login": "acme"}},
            {"full_name": "gpl/tool", "html_url": "https://github.com/gpl/tool", "default_branch": "main", "license": {"spdx_id": "GPL-3.0"}, "stargazers_count": 5000, "size": 50,
             "fork": False, "archived": False, "owner": {"login": "gpl"}},
            {"full_name": "none/thing", "html_url": "https://github.com/none/thing", "default_branch": "main", "license": None, "stargazers_count": 4000, "size": 50,
             "fork": False, "archived": False, "owner": {"login": "none"}},
            {"full_name": "copy/lib", "html_url": "https://github.com/copy/lib", "default_branch": "main", "license": {"spdx_id": "MIT"}, "stargazers_count": 3, "size": 50,
             "fork": True, "archived": False, "owner": {"login": "copy"}},
        ]
        self.web.add("https://api.github.com/search/repositories", json.dumps({"items": repos}).encode(), headers={"x-ratelimit-remaining": "9"})
        code = "\n".join(f"def function_{i}(value):\n    # compute the number {i} from the given value and return it\n    return value * {i} + {i % 7}" for i in range(40))
        tree = [{"type": "blob", "path": "README.md", "size": 600}, {"type": "blob", "path": "src/main.py", "size": len(code)},
                {"type": "blob", "path": "node_modules/dep/index.js", "size": 500}, {"type": "blob", "path": "assets/logo.png", "size": 500},
                {"type": "blob", "path": "big.py", "size": 900_000}, {"type": "blob", "path": "LICENSE", "size": 500}, {"type": "tree", "path": "src", "size": 0}]
        self.web.add("https://api.github.com/repos/acme/lib/git/trees/", json.dumps({"tree": tree}).encode(), headers={"x-ratelimit-remaining": "8"})
        self.web.add("https://raw.githubusercontent.com/acme/lib/main/README.md", ("# Library\n\n" + prose(1, 12)).encode())
        self.web.add("https://raw.githubusercontent.com/acme/lib/main/src/main.py", code.encode())
        self.enable(sources={"hub": False, "github": True}, github={"queries": ["algorithms"], "max_repos": 5})

    def test_only_permissive_repositories_are_read_and_the_licence_is_kept(self):
        self.assertEqual(drain(self.root, self.web), "idle")
        asked = "\n".join(self.web.log)
        for never in ("gpl/tool", "none/thing", "copy/lib"):
            self.assertNotIn(never, asked)
        plan = {i["id"]: i for i in State(self.root).data["plan"]}
        self.assertIn("permissive", plan["gpl/tool"]["reason"])
        self.assertEqual(plan["none/thing"]["status"], "skipped")
        self.assertNotIn("copy/lib", plan)                                           # a fork
        paths = {r["provenance"]["path"] for r in self.shards()}
        self.assertEqual(paths, {"README.md", "src/main.py"})                       # not node_modules, not the image, not the 900 KB file, not LICENSE
        row = next(r for r in self.shards() if r["provenance"]["path"] == "src/main.py")
        self.assertEqual((row["provenance"]["licence"], row["provenance"]["source_url"]), ("mit", "https://github.com/acme/lib/blob/main/src/main.py"))
        self.assertIn("acme/lib", row["provenance"]["attribution"])

    def test_copyleft_repositories_stay_out_even_when_that_licence_class_is_opted_in(self):
        config.save(self.root, {"licences": {"copyleft": True, "unknown": True}})
        drain(self.root, self.web)
        self.assertNotIn("gpl/tool", "\n".join(self.web.log))

    def test_the_rate_limit_is_honoured_until_its_reset(self):
        now = {"t": 1_000_000.0}
        self.web.routes = [r for r in self.web.routes if "git/trees" not in r[0]]
        self.web.add("https://api.github.com/repos/acme/lib/git/trees/", b'{"message":"rate limit exceeded"}', status=403,
                     headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(now["t"]) + 600)})
        self.assertEqual(drain(self.root, self.web, clock=lambda: now["t"]), "limit")
        self.assertIn("rate limit", State(self.root).data["reason"])
        calls = len(self.web.asked("api.github.com"))
        self.assertEqual(drain(self.root, self.web, clock=lambda: now["t"]), "limit")        # still blocked: no call is made
        self.assertEqual(len(self.web.asked("api.github.com")), calls)
        now["t"] += 700
        self.web.routes = [r for r in self.web.routes if "git/trees" not in r[0]]
        self.web.add("https://api.github.com/repos/acme/lib/git/trees/", json.dumps({"tree": [{"type": "blob", "path": "README.md", "size": 600}]}).encode())
        self.assertEqual(drain(self.root, self.web, clock=lambda: now["t"]), "idle")
        self.assertTrue(self.shards())

    def test_issues_are_not_collected(self):
        drain(self.root, self.web)
        self.assertEqual([u for u in self.web.log if "issues" in u or "discussions" in u], [])


class Crawler(Base):
    HOST = "docs.example.org"

    def setUp(self):
        super().setUp()
        article = "<html><head><title>Sorting</title></head><body><nav>menu menu menu</nav><main><h1>Sorting</h1><p>" + prose(5, 10) + "</p></main><footer>copyright</footer></body></html>"
        qa = {"@context": "https://schema.org", "@type": "QAPage", "mainEntity": {
            "@type": "Question", "name": "How do I reverse a list in place?", "text": "I have a list of numbers and want it reversed without copying it.",
            "acceptedAnswer": {"@type": "Answer", "text": "<p>Call the reverse method of the list; it reorders the elements in place.</p>", "url": f"https://{self.HOST}/a#1",
                               "author": {"@type": "Person", "name": "Pat"}}}}
        qa_page = f'<html><head><title>Reverse</title><script type="application/ld+json">{json.dumps(qa)}</script></head><body><p>x</p></body></html>'
        home = ('<html><body><a href="/a">a</a><a href="/b">b</a><a href="/private/secret">secret</a><a href="https://www.reddit.com/r/x">r</a>'
                '<a href="https://other.example/">other</a><a href="/noai">noai</a><a href="/b?x=1">q</a></body></html>')
        html = {"Content-Type": "text/html; charset=utf-8"}
        self.web.add(f"https://{self.HOST}/robots.txt", b"User-agent: *\nDisallow: /private\nCrawl-delay: 5\n")
        self.web.add(f"https://{self.HOST}/", home.encode(), headers=html)
        self.web.add(f"https://{self.HOST}/a", qa_page.encode(), headers=html)
        self.web.add(f"https://{self.HOST}/b", article.encode(), headers=html)
        self.web.add(f"https://{self.HOST}/noai", b'<html><head><meta name="robots" content="noai"></head><body>' + article.encode() + b"</body></html>", headers=html)
        self.web.add(f"https://{self.HOST}/private/secret", b"<html><body>secret</body></html>", headers=html)
        self.web.add("https://www.reddit.com/", b"<html>never</html>", headers=html)
        self.web.add("https://other.example/", b"<html>never</html>", headers=html)
        self.slept = []
        self.enable(sources={"hub": False, "web": True}, web={"sites": [{"host": self.HOST, "licence": "cc-by-4.0", "pages": 20}]})

    def crawl(self):
        return drain(self.root, self.web, sleep=self.slept.append)

    def test_robots_disallow_and_crawl_delay_are_obeyed(self):
        def sleep(seconds):
            self.slept.append(seconds)
            if seconds == service.POLL:              # idle and waiting: this one process has crawled everything, so turn collection off
                config.save(self.root, {"enabled": False})
        service.run(self.root, transport=self.web, probes=idle_probes(), sleep=sleep, lock=False)
        asked = self.web.log
        self.assertEqual([u for u in asked if "/private" in u], [])
        self.assertEqual(len([u for u in asked if u.endswith("/robots.txt")]), 1)
        pages = [u for u in asked if self.HOST in u and not u.endswith("robots.txt")]
        self.assertIn(f"https://{self.HOST}/a", pages)
        self.assertGreaterEqual(sum(self.slept), 5 * (len(pages) - 1) - 1)                    # at least the crawl-delay between pages

    def test_an_excluded_host_is_never_requested(self):
        self.crawl()
        self.assertEqual([u for u in self.web.log if "reddit" in u or "quora" in u], [])
        self.assertEqual([u for u in self.web.log if "other.example" in u], [])                 # other hosts are not followed either

    def test_a_redirect_to_an_excluded_host_is_refused(self):
        config.save(self.root, {"web": {"sites": [{"host": "go.example.org", "licence": "cc-by-4.0", "seeds": ["https://go.example.org/x"]}]}})
        self.web.add("https://go.example.org/x", b"", status=302, headers={"Location": "https://www.reddit.com/r/y"})
        self.crawl()
        self.assertEqual([u for u in self.web.log if "reddit" in u], [])

    def test_a_qa_page_becomes_pairs_and_other_pages_text_with_the_site_licence(self):
        self.crawl()
        rows = {r["provenance"]["source_url"]: r for r in self.shards()}
        qa = rows[f"https://{self.HOST}/a"]
        self.assertEqual(qa["messages"][0]["content"].startswith("How do I reverse a list in place?"), True)
        self.assertIn("in place", qa["messages"][1]["content"])
        self.assertIn("Pat", qa["provenance"]["attribution"])
        text = rows[f"https://{self.HOST}/b"]
        self.assertEqual(text["provenance"]["licence"], "cc-by-4.0")
        self.assertNotIn("menu menu", text["text"])                                              # page furniture left out
        self.assertNotIn("copyright", text["text"])
        self.assertNotIn(f"https://{self.HOST}/noai", rows)                                       # the page asked not to be used for AI

    def test_the_crawl_resumes_from_its_frontier(self):
        out = service.run(self.root, transport=self.web, probes=idle_probes(), sleep=self.slept.append, once=True, lock=False)
        self.assertEqual(out, "worked")
        seen = len(self.web.log)
        self.crawl()
        self.assertEqual(self.web.log.count(f"https://{self.HOST}/"), 1)                       # the first page was not fetched again
        self.assertGreater(len(self.web.log), seen)


class Robots_(unittest.TestCase):
    def robots(self, status, body=b""):
        web = FakeWeb()
        web.add("https://a.example/robots.txt", body, status=status)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        http = Http(Ledger(tmp.name), config.merge({})["limits"], transport=web, sleep=lambda s: None)
        return Robots(http), http

    def test_missing_allows_server_error_and_access_denied_forbid(self):
        self.assertTrue(self.robots(404)[0].allowed("https://a.example/x"))
        self.assertFalse(self.robots(500)[0].allowed("https://a.example/x"))
        self.assertFalse(self.robots(403)[0].allowed("https://a.example/x"))

    def test_our_own_token_can_be_singled_out(self):
        body = b"User-agent: GroundworkTrainer-Collector\nDisallow: /\n\nUser-agent: *\nDisallow:\n"
        self.assertFalse(self.robots(200, body)[0].allowed("https://a.example/x"))


class Html(unittest.TestCase):
    def test_text_keeps_code_and_drops_scripts(self):
        text = htmltext.to_text("<p>Hello <b>there</b></p><script>alert(1)</script><pre><code>a = 1\nb = 2</code></pre><ul><li>one</li><li>two</li></ul>")
        self.assertEqual(text, "Hello there\n\n```\na = 1\nb = 2\n```\n\none\ntwo")


if __name__ == "__main__":
    unittest.main()
