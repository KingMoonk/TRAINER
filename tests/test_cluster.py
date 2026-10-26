"""studio_cluster: the node registry, the requirements each kind of job carries, and placement. No network, no GPU, no window."""
import io
import json
import threading
import time
import unittest
from unittest import mock

import studio_cluster as sc
import studio_queue
from studio_cluster import NodeInfo, Requirements
from studio_peer import PeerError

GB = 2 ** 30


def node(address, device="cpu", memory=16 * GB, busy=False, online=True, hostname=None, dtypes=None):
    return NodeInfo(address=address, hostname=hostname or address, device=device, memory_bytes=memory, busy=busy, online=online,
                    dtypes=dtypes or sc.DTYPES[device])


def payload(**changes):
    return {"hostname": "windows-box", "version": "1.2.3", "device": "cuda", "device_name": "RTX 4090", "memory_bytes": 24 * GB,
            "busy": False, "dtypes": ["float32", "bfloat16"], **changes}


class Registry(unittest.TestCase):
    def test_no_nodes_listed_is_a_cluster_of_one(self):
        self.assertEqual(sc.configured_nodes({}), [])
        self.assertEqual(sc.configured_nodes({"studioConfig": {}}), [])
        own = node("local")
        self.assertEqual(sc.survey({"studioConfig": {}}, own, fetch=self.fail), [own])

    def test_listed_nodes_are_trimmed_and_deduplicated_in_order(self):
        state = {"studioConfig": {"nodes": [" windows-box ", "", "mini", "windows-box", "http://100.64.0.7:9000"]}}
        self.assertEqual(sc.configured_nodes(state), ["windows-box", "mini", "http://100.64.0.7:9000"])

    def test_a_malformed_list_is_a_plain_error(self):
        for bad in ("windows-box", [1], {"a": 1}):
            with self.assertRaisesRegex(ValueError, "studioConfig.nodes"):
                sc.configured_nodes({"studioConfig": {"nodes": bad}})

    def test_urls(self):
        port = sc.DEFAULT_PORT
        self.assertEqual(sc.node_url("windows-box"), f"http://windows-box:{port}/node")
        self.assertEqual(sc.node_url("box.tail1234.ts.net"), f"http://box.tail1234.ts.net:{port}/node")
        self.assertEqual(sc.node_url("http://100.64.0.7:9000"), "http://100.64.0.7:9000/node")
        self.assertEqual(sc.node_url("https://box.ts.net/api/"), "https://box.ts.net/node")
        self.assertEqual(sc.node_url("http://[fd7a::1]"), f"http://[fd7a::1]:{port}/node")
        self.assertEqual(sc.node_url("http://[fd7a::1]:9000/x"), "http://[fd7a::1]:9000/node")

    def test_own_info_on_cuda_reports_video_memory(self):
        status = {"available": True, "kind": "cuda", "name": "RTX 4090", "reason": "", "memory_bytes": 24 * GB}
        info = sc.own_info(busy=True, version="9", status=status, memory=64 * GB)
        self.assertEqual((info.address, info.device, info.device_name, info.memory_bytes, info.busy, info.version),
                         (sc.LOCAL, "cuda", "RTX 4090", 24 * GB, True, "9"))
        self.assertFalse(info.free)
        self.assertIn("bfloat16", info.dtypes)

    def test_own_info_on_apple_silicon_reports_the_computers_memory(self):
        status = {"available": True, "kind": "mps", "name": "Apple GPU (Metal)", "reason": "", "memory_bytes": 0}
        info = sc.own_info(status=status, memory=32 * GB)
        self.assertEqual((info.device, info.memory_bytes, info.busy, info.free), ("mps", 32 * GB, False, True))
        self.assertNotIn("bfloat16", info.dtypes)

    def test_own_info_without_a_gpu_is_a_cpu_node(self):
        status = {"available": False, "kind": "", "name": "", "reason": "No CUDA or Metal GPU was found.", "memory_bytes": 0}
        info = sc.own_info(status=status, memory=8 * GB)
        self.assertEqual((info.device, info.memory_bytes), ("cpu", 8 * GB))

    def test_own_info_without_torch_is_a_cpu_node(self):
        with mock.patch.dict("sys.modules", {"studio_backend": None}):  # importing it raises ImportError
            info = sc.own_info(memory=4 * GB)
        self.assertEqual((info.device, info.memory_bytes, info.address), ("cpu", 4 * GB, sc.LOCAL))
        self.assertTrue(info.hostname)

    def test_the_payload_round_trips(self):
        original = sc.own_info(status={"available": True, "kind": "cuda", "name": "GPU", "memory_bytes": 8 * GB}, version="1")
        body = original.payload()
        self.assertEqual(set(body), {"hostname", "version", "device", "device_name", "memory_bytes", "busy", "dtypes", "speeds", "profile", "memory_share"})
        self.assertEqual(sc.info_from_payload("box", body), sc.NodeInfo(**{**body, "dtypes": tuple(body["dtypes"])}, address="box"))

    def test_a_peer_that_answers_is_online_with_its_capabilities(self):
        asked = []

        def fetch(url, timeout):
            asked.append((url, timeout))
            return payload()

        [info] = sc.poll(["windows-box"], fetch=fetch, timeout=1.5)
        self.assertEqual(asked, [(f"http://windows-box:{sc.DEFAULT_PORT}/node", 1.5)])
        self.assertEqual((info.address, info.hostname, info.device, info.memory_bytes, info.version, info.online),
                         ("windows-box", "windows-box", "cuda", 24 * GB, "1.2.3", True))
        self.assertEqual(info.dtypes, ("float32", "bfloat16"))

    def test_a_peer_slow_for_one_poll_keeps_its_last_answer_and_is_offline_after_two(self):
        self.addCleanup(sc._ANSWERED.clear)
        answers = [payload(), TimeoutError("timed out"), payload(), PeerError("slowpoke did not answer GET /node: timed out"),
                   TimeoutError("timed out")]

        def fetch(url, timeout):
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer

        states = [sc.poll(["slowpoke"], fetch=fetch)[0] for _ in range(4)]
        self.assertEqual([i.online for i in states], [True, True, True, True])  # every slow poll followed an answer
        [offline] = sc.poll(["slowpoke"], fetch=fetch)  # the second slow poll in a row
        self.assertFalse(offline.online)
        self.assertIn("did not answer", offline.error)
        sc.poll(["refused"], fetch=lambda url, timeout: payload())
        [refused] = sc.poll(["refused"], fetch=lambda url, timeout: (_ for _ in ()).throw(ConnectionRefusedError()))
        self.assertFalse(refused.online)  # refused is not slow: offline at once

    def test_an_unreachable_peer_is_offline_and_never_raises(self):
        def fetch(url, timeout):
            if "mini" in url:
                raise TimeoutError("timed out")
            if "refused" in url:
                raise ConnectionRefusedError("no route")
            return payload()

        up, mini, refused = sc.poll(["windows-box", "mini", "refused"], fetch=fetch)
        self.assertTrue(up.online)
        for down, why in ((mini, "mini did not answer within 2 seconds."), (refused, "refused the connection")):
            self.assertFalse(down.online)
            self.assertFalse(down.free)
            self.assertIn(why, down.error)
        self.assertEqual([i.address for i in (up, mini, refused)], ["windows-box", "mini", "refused"])

    def test_nonsense_from_a_peer_is_offline(self):
        for body in (None, [], "ok", {"hostname": "x"}, {"hostname": "x", "device": "tpu"}, {"device": "cpu"}):
            [info] = sc.poll(["odd"], fetch=lambda url, timeout, body=body: body)
            self.assertFalse(info.online, body)
            self.assertIn("not like a node of this app", info.error)

    def test_a_real_unreachable_address_is_offline_not_an_exception(self):
        [info] = sc.poll(["http://127.0.0.1:9"], timeout=.5)  # the discard port: nothing listens
        self.assertFalse(info.online)
        self.assertTrue(info.error)

    def test_dead_peers_are_asked_at_the_same_time(self):
        def slow(url, timeout):
            time.sleep(.3)
            raise TimeoutError

        started = time.monotonic()
        infos = sc.poll([f"peer{i}" for i in range(6)], fetch=slow)
        self.assertEqual(len(infos), 6)
        self.assertLess(time.monotonic() - started, 1.2)  # six in a row would be 1.8 s

    def test_a_peer_that_never_answers_ends_the_poll_at_the_timeout(self):
        release = threading.Event()
        self.addCleanup(release.set)

        def fetch(url, timeout):
            if "stuck" in url:
                release.wait(10)  # a slow drip or a name lookup: the socket timeout would not end it
            return payload()

        started = time.monotonic()
        up, stuck = sc.poll(["windows-box", "stuck"], fetch=fetch, timeout=.3)
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual((up.online, stuck.online, stuck.error), (True, False, "stuck did not answer within 0.3 seconds."))

    def test_more_peers_than_threads_are_all_asked_at_once(self):
        infos = sc.poll([f"peer{i}" for i in range(20)], fetch=lambda url, timeout: (time.sleep(.2), payload())[1], timeout=1)
        self.assertTrue(all(i.online for i in infos))

    def test_offline_errors_are_sentences(self):
        raised = {"a": PeerError("a answered 403 to GET /node: not allowed", status=403), "b": json.JSONDecodeError("x", "", 0),
                  "c": ValueError("boom")}

        def fetch(url, timeout):
            raise raised[url.split("//")[1].split(":")[0]]

        a, b, c = sc.poll(["a", "b", "c"], fetch=fetch)
        self.assertEqual(a.error, "a answered 403 to GET /node: not allowed.")
        self.assertEqual(b.error, "b answered, but not like a node of this app.")
        self.assertEqual(c.error, "c could not be asked: ValueError: boom.")

    def test_survey_is_this_node_then_the_peers(self):
        own = node("local", "mps")
        found = sc.survey({"studioConfig": {"nodes": ["windows-box"]}}, own, fetch=lambda url, timeout: payload())
        self.assertEqual([n.address for n in found], ["local", "windows-box"])
        self.assertIs(found[0], own)


class Addresses(unittest.TestCase):
    def test_what_node_url_reaches_is_accepted(self):
        for good in ("windows-box", "windows-box:9000", "mini.tail1234.ts.net", "100.64.0.7:8787", "[fd7a::1]:8787", "[fd7a::1]",
                     "http://100.64.0.7:8787", "https://windows-box.tail1234.ts.net", "https://box.ts.net/anything", " box "):
            self.assertEqual(sc.check_address(good), good.strip())
            sc.node_url(good)

    def test_anything_else_is_refused_with_a_sentence(self):
        for bad, why in (("ftp://box", "http://"), ("http://", "no machine name"), ("box:abc", "port"), ("box:0", "port"), ("box:99999", "port"),
                         ("box:", "port"), ("a b", "spaces"), ("me@box", "login"), ("box?x=1", "query"), ("fd7a::1", "brackets"),
                         ("[fd7a::1", "do not match"), (":8787", "no machine name")):
            with self.assertRaisesRegex(ValueError, why):
                sc.check_address(bad)

    def test_a_list_is_stripped_and_deduplicated_and_blanks_dropped(self):
        self.assertEqual(sc.clean_nodes([" a ", "", "b", "a", "   "]), ["a", "b"])
        self.assertEqual(sc.clean_nodes([]), [])

    def test_a_list_has_a_limit(self):
        with self.assertRaisesRegex(ValueError, "up to 32"):
            sc.clean_nodes([f"box{i}" for i in range(sc.MAX_NODES + 1)])


class Defaults(unittest.TestCase):
    def test_pretraining_and_rlhf_prefer_cuda(self):
        for kind in ("pretrain", "post_round", "restore_answers"):
            self.assertEqual(sc.defaults_for(kind).device, "cuda", kind)

    def test_everything_else_takes_any_free_node(self):
        for kind in ("eval", "score", "benchmark", "datagen", "lora", "quicktest", "compare", "import_pretrained", "checkpoint_now"):
            self.assertEqual(sc.defaults_for(kind), Requirements(), kind)

    def test_inspect_prefers_the_node_with_the_most_memory(self):
        wanted = sc.defaults_for("inspect")
        self.assertTrue(wanted.most_memory)
        self.assertEqual(wanted.device, "any")

    def test_every_queue_kind_has_defaults(self):
        for kind in studio_queue.KINDS:
            self.assertIsInstance(sc.defaults_for(kind), Requirements, kind)

    def test_a_misspelt_kind_is_an_error_not_a_silent_any(self):
        with self.assertRaisesRegex(ValueError, "Unknown job kind: pretrian"):
            sc.defaults_for("pretrian")


class Placement(unittest.TestCase):
    def setUp(self):
        self.mac = node("local", "mps", 32 * GB, hostname="macbook")
        self.pc = node("windows-box", "cuda", 24 * GB, hostname="WINDOWS-BOX")
        self.mini = node("mini", "cpu", 8 * GB, hostname="mini")
        self.nodes = [self.mac, self.pc, self.mini]

    def test_pretraining_goes_to_the_cuda_node(self):
        self.assertIs(sc.place(sc.defaults_for("pretrain"), self.nodes), self.pc)

    def test_a_preference_is_soft_when_the_preferred_node_is_busy(self):
        busy = [self.mac, node("windows-box", "cuda", 24 * GB, busy=True), self.mini]
        self.assertIs(sc.place(sc.defaults_for("pretrain"), busy), self.mac)

    def test_an_offline_node_is_never_chosen(self):
        gone = [self.mac, sc.offline("windows-box", "TimeoutError"), self.mini]
        self.assertIs(sc.place(sc.defaults_for("pretrain"), gone), self.mac)

    def test_any_job_goes_to_the_first_free_node(self):
        self.assertIs(sc.place(sc.defaults_for("benchmark"), self.nodes), self.mac)
        self.assertIs(sc.place(sc.defaults_for("benchmark"), [node("local", busy=True), self.pc]), self.pc)

    def test_inspect_goes_to_the_node_with_the_most_memory(self):
        self.assertIs(sc.place(sc.defaults_for("inspect"), self.nodes), self.mac)
        self.assertIs(sc.place(sc.defaults_for("inspect"), [self.mini, self.pc]), self.pc)

    def test_memory_ranks_after_the_device_preference(self):
        wanted = Requirements(device="cuda", most_memory=True)
        self.assertIs(sc.place(wanted, self.nodes), self.pc)
        big = node("big", "cuda", 48 * GB)
        self.assertIs(sc.place(wanted, [self.pc, big]), big)

    def test_minimum_memory_is_hard(self):
        wanted = Requirements(device="cuda", min_memory_bytes=30 * GB)
        self.assertIs(sc.place(wanted, self.nodes), self.mac)  # the CUDA node is too small; the Mac fits
        self.assertIsNone(sc.place(Requirements(min_memory_bytes=64 * GB), self.nodes))

    def test_dtype_is_hard(self):
        wanted = Requirements(dtype="bfloat16")
        self.assertIs(sc.place(wanted, self.nodes), self.pc)  # the Mac's GPU does not list bfloat16
        self.assertIsNone(sc.place(Requirements(dtype="float8"), self.nodes))

    def test_none_fits(self):
        self.assertIsNone(sc.place(Requirements(), []))
        self.assertIsNone(sc.place(Requirements(), [node("a", busy=True), sc.offline("b", "down")]))

    def test_the_pinned_node_wins_over_a_better_one(self):
        self.assertIs(sc.place(sc.defaults_for("pretrain"), self.nodes, pinned="mini"), self.mini)

    def test_a_pin_matches_the_hostname_in_any_case(self):
        self.assertIs(sc.place(Requirements(), self.nodes, pinned="windows-box"), self.pc)
        self.assertIs(sc.place(Requirements(), self.nodes, pinned=" macbook "), self.mac)

    def test_a_hostname_two_nodes_share_places_nowhere_but_either_address_does(self):
        one, two = node("10.0.0.2:8787", hostname="Box"), node("10.0.0.3:8787", hostname="box")
        nodes = [self.mac, one, two]
        for typed in ("box", "BOX", " Box "):
            self.assertIsNone(sc.place(Requirements(), nodes, pinned=typed))
        self.assertIs(sc.place(Requirements(), nodes, pinned="10.0.0.2:8787"), one)
        self.assertIs(sc.place(Requirements(), nodes, pinned="10.0.0.3:8787"), two)
        self.assertIs(sc.place(Requirements(), nodes, pinned="macbook"), self.mac)

    def test_a_pinned_node_that_cannot_run_it_is_none_not_another_node(self):
        busy = [self.mac, node("windows-box", "cuda", 24 * GB, busy=True)]
        self.assertIsNone(sc.place(sc.defaults_for("pretrain"), busy, pinned="windows-box"))
        self.assertIsNone(sc.place(Requirements(min_memory_bytes=64 * GB), self.nodes, pinned="mini"))
        self.assertIsNone(sc.place(Requirements(), self.nodes, pinned="not-listed"))

    def test_placement_does_not_change_its_inputs(self):
        before = list(self.nodes)
        sc.place(sc.defaults_for("pretrain"), self.nodes)
        self.assertEqual(self.nodes, before)


if __name__ == "__main__":
    unittest.main()


class Callers(unittest.TestCase):
    def test_defaults_let_in_the_trainer_tag_only(self):
        self.assertEqual(sc.allowed_callers({}), (("tag:trainer",), ()))

    def test_configured(self):
        state = {"studioConfig": {"cluster": {"allowTags": [" tag:gpu "], "allowUsers": ["me@example.com"]}}}
        self.assertEqual(sc.allowed_callers(state), (("tag:gpu",), ("me@example.com",)))

    def test_bad_values_say_what_to_fix(self):
        for cluster, text in (("x", "studioConfig.cluster"), ({"allowTags": "tag:a"}, "allowTags"), ({"allowTags": ["trainer"]}, "tag:"),
                              ({"allowUsers": [3]}, "allowUsers")):
            with self.assertRaisesRegex(ValueError, text):
                sc.allowed_callers({"studioConfig": {"cluster": cluster}})


class FailureOfAPage(unittest.TestCase):
    def test_a_web_page_answer_says_the_api_is_not_at_that_address(self):
        for status in (200, 404, 308):
            text = sc.failure("https://box.ts.net", PeerError("answered", status=status, detail="<!DOCTYPE html><html>"))
            self.assertIn("API is not at that address", text)
            self.assertNotIn("Allow", text)


class PollThroughPeerClient(unittest.TestCase):
    def test_default_fetch_asks_the_peer_with_one_attempt(self):
        calls = []

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __init__(self):
                self.body = io.BytesIO(b'{"hostname": "box", "device": "cpu"}')

            def read(self, size=-1):
                return self.body.read(size)

        def opener(request, timeout):
            calls.append((request.full_url, timeout))
            return Resp()

        with mock.patch("studio_peer.urllib.request.urlopen", opener):
            (info,) = sc.poll(["https://box.ts.net"])
        self.assertEqual((info.online, info.hostname), (True, "box"))
        self.assertEqual(calls, [("https://box.ts.net/node", sc.POLL_TIMEOUT)])
