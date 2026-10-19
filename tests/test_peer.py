"""studio_peer: the node-to-node HTTP client, over a fake opener. No network."""
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from studio_peer import PeerClient, PeerError, base_url


class Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Opener:
    """Answers each call with the next item: bytes (a 200 body) or an exception to raise. Records the requests."""

    def __init__(self, *answers):
        self.answers, self.requests = list(answers), []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return Response(answer)


def http_error(code, body):
    return urllib.error.HTTPError("http://x", code, "error", {}, io.BytesIO(body.encode()))


class Urls(unittest.TestCase):
    def test_a_bare_name_gets_the_default_port_and_a_url_keeps_its_own(self):
        self.assertEqual(base_url("windows-box"), "http://windows-box:8787")
        self.assertEqual(base_url(" http://100.64.0.7:9000/node "), "http://100.64.0.7:9000")
        self.assertEqual(base_url("https://box.tail1234.ts.net"), "https://box.tail1234.ts.net")  # tailscale serve, 443
        self.assertEqual(base_url("http://[fd7a::1]:8787"), "http://[fd7a::1]:8787")

    def test_paths_and_query(self):
        client = PeerClient("box", opener=Opener())
        self.assertEqual(client.url("/collective/r1/params", {"round": 3}), "http://box:8787/collective/r1/params?round=3")


class Calls(unittest.TestCase):
    def test_json_round_trip(self):
        opener = Opener(b'{"ok": true}')
        self.assertEqual(PeerClient("box", opener=opener).post_json("cluster/jobs", {"id": "a"}), {"ok": True})
        request, timeout = opener.requests[0]
        self.assertEqual((request.get_method(), request.full_url, json.loads(request.data)), ("POST", "http://box:8787/cluster/jobs", {"id": "a"}))
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(timeout, 10.)

    def test_bytes_both_ways(self):
        opener = Opener(b"\x00npz", b"")
        client = PeerClient("box", opener=opener)
        self.assertEqual(client.get_bytes("collective/r/shards/0"), b"\x00npz")
        self.assertIsNone(client.post_bytes("collective/r/rounds/1/delta", b"\x01", {"tokens": 5}))
        request = opener.requests[1][0]
        self.assertEqual((request.data, request.get_header("Content-type")), (b"\x01", "application/octet-stream"))
        self.assertTrue(request.full_url.endswith("?tokens=5"))

    def test_a_body_goes_to_a_file_a_block_at_a_time(self):
        class Body(Response):
            reads = []

            def read(self, size=-1):
                self.reads.append(size)
                return super().read(size)

        data = bytes(range(256)) * 9000  # more than one block
        client = PeerClient("box", opener=lambda request, timeout: Body(data))
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "weights.npz"
            self.assertIsNone(client.get_file("versions/1/weights", target, {"key": "a"}))
            self.assertEqual(target.read_bytes(), data)
        self.assertTrue(all(0 < size <= 1 << 20 for size in Body.reads), Body.reads)  # never the whole body at once
        self.assertGreater(len(Body.reads), 2)

    def test_a_download_that_drops_is_retried_from_the_start_and_an_error_status_is_not(self):
        class Drops(Response):
            def read(self, size=-1):
                raise ConnectionResetError("reset")

        answers = [Drops(b"half"), Response(b"whole")]
        opener = lambda request, timeout: answers.pop(0)  # noqa: E731
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "f"
            PeerClient("box", opener=opener, sleep=lambda s: None).get_file("x", target)
            self.assertEqual(target.read_bytes(), b"whole")
            with self.assertRaises(PeerError) as caught:
                PeerClient("box", opener=Opener(http_error(404, '{"detail": "No version 7"}'))).get_file("x", target)
            self.assertEqual((caught.exception.status, caught.exception.detail), (404, "No version 7"))

    def test_a_connection_failure_is_retried_with_growing_jittered_waits(self):
        waits = []
        opener = Opener(urllib.error.URLError("refused"), TimeoutError("slow"), b"[]")
        self.assertEqual(PeerClient("box", opener=opener, sleep=waits.append).get_json("jobs"), [])
        self.assertEqual(len(waits), 2)
        self.assertTrue(.5 <= waits[0] <= .75 and 1. <= waits[1] <= 1.5)

    def test_a_peer_that_never_answers_is_a_peer_error_without_a_status(self):
        opener = Opener(*[urllib.error.URLError("unreachable")] * 3)
        with self.assertRaises(PeerError) as caught:
            PeerClient("box", opener=opener, sleep=lambda s: None).get_json("node")
        self.assertIsNone(caught.exception.status)
        self.assertIn("did not answer", str(caught.exception))

    def test_an_error_status_is_never_retried_and_carries_the_detail(self):
        opener = Opener(http_error(409, '{"detail": "stale epoch"}'))
        with self.assertRaises(PeerError) as caught:
            PeerClient("box", opener=opener, sleep=self.fail).post_json("cluster/jobs/a/renew", {"epoch": 1})
        self.assertEqual((caught.exception.status, caught.exception.detail), (409, "stale epoch"))
        self.assertEqual(len(opener.requests), 1)


    def test_a_node_in_make_dev_is_called_under_api_from_then_on(self):
        page = "<!DOCTYPE html><html><body>404</body></html>"
        opener = Opener(http_error(404, page), b'{"device": "cuda"}', b'{"ok": true}')
        client = PeerClient("https://beast.tail1234.ts.net", opener=opener, sleep=self.fail)
        self.assertEqual(client.get_json("node"), {"device": "cuda"})
        self.assertEqual(client.post_json("cluster/jobs", {"a": 1}), {"ok": True})
        self.assertEqual([r.full_url for r, _ in opener.requests], ["https://beast.tail1234.ts.net/node",
                         "https://beast.tail1234.ts.net/api/node", "https://beast.tail1234.ts.net/api/cluster/jobs"])
        self.assertEqual(opener.requests[2][0].data, b'{"a": 1}')

    def test_a_page_under_api_too_reports_the_first_answer_and_keeps_the_root(self):
        opener = Opener(http_error(404, "<html>root</html>"), http_error(404, "<html>api</html>"), b"{}")
        client = PeerClient("https://box.tail1234.ts.net", opener=opener, sleep=self.fail)
        with self.assertRaises(PeerError) as caught:
            client.get_json("node")
        self.assertEqual((caught.exception.status, caught.exception.detail), (404, "<html>root</html>"))
        client.get_json("node")
        self.assertEqual(opener.requests[-1][0].full_url, "https://box.tail1234.ts.net/node")

    def test_a_post_redirected_by_the_dev_web_server_goes_under_api(self):
        opener = Opener(http_error(308, "/cluster/jobs/"), b'{"ok": true}')
        client = PeerClient("https://beast.tail1234.ts.net", opener=opener, sleep=self.fail)
        self.assertEqual(client.post_json("cluster/jobs", {}), {"ok": True})
        self.assertEqual(opener.requests[1][0].full_url, "https://beast.tail1234.ts.net/api/cluster/jobs")

    def test_a_json_error_from_the_api_is_not_retried_under_api(self):
        opener = Opener(http_error(404, '{"detail": "Not Found"}'))
        with self.assertRaises(PeerError):
            PeerClient("box", opener=opener, sleep=self.fail).get_json("node")
        self.assertEqual(len(opener.requests), 1)


if __name__ == "__main__":
    unittest.main()
