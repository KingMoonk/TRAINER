"""studio_peer.PeerClient's parallel transfers (#584) against a loopback peer whose every connection is slow (tests/transfer_support.py).
Run `python -m tests.test_peer_parallel` to print the single-stream against parallel throughput."""
import hashlib
import sys
import tempfile
import time
import unittest
from pathlib import Path

import studio_parts
from studio_peer import Cancelled, PeerClient, PeerError
from tests.transfer_support import Peer, blob

PART = 100_000
BODY = blob(1_000_003)  # ten full parts and a short one


def client(peer, **options):
    return PeerClient(peer.address, sleep=lambda seconds: None, **{"streams": 4, "part": PART, **options})


class RangedDownloads(unittest.TestCase):
    def test_parts_put_back_in_order_are_the_single_get_byte_for_byte(self):
        with Peer(BODY) as peer:
            single = client(peer, streams=1).get_bytes("/blob")
            ranged = client(peer).get_bytes("/blob")
            self.assertEqual(single, BODY)
            self.assertEqual(hashlib.sha256(ranged).digest(), hashlib.sha256(BODY).digest())
            self.assertEqual(ranged, BODY)
            asked = [header for method, path, header in peer.requests if header]
            self.assertEqual(len(asked), 11)
            self.assertEqual(asked[0], f"bytes=0-{PART - 1}")  # part 0 first, the size comes back with it; the rest follow

    def test_a_file_download_writes_every_part_at_its_place(self):
        with Peer(BODY) as peer, tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "weights.npz"
            client(peer).get_file("/blob", target)
            self.assertEqual(target.read_bytes(), BODY)
            target.write_bytes(b"x" * 2_000_000)  # an earlier, longer file is replaced, not overwritten in place
            client(peer).get_file("/blob", target)
            self.assertEqual(target.read_bytes(), BODY)

    def test_a_body_of_one_part_is_one_request(self):
        with Peer(BODY[:PART - 5]) as peer:
            self.assertEqual(client(peer).get_bytes("/blob"), BODY[:PART - 5])
            self.assertEqual(len(peer.requests), 1)

    def test_one_part_that_fails_is_retried_by_itself(self):
        with Peer(BODY) as peer:
            peer.fail[f"bytes={5 * PART}-{6 * PART - 1}"] = 2  # cut short twice, then fine
            self.assertEqual(client(peer).get_bytes("/blob"), BODY)
            tries = [header for method, path, header in peer.requests if header == f"bytes={5 * PART}-{6 * PART - 1}"]
            self.assertEqual(len(tries), 3)
            self.assertEqual(len(peer.requests), 13)  # the other ten parts were fetched once

    def test_a_part_that_keeps_failing_fails_the_download(self):
        with Peer(BODY) as peer:
            peer.fail[f"bytes={PART}-{2 * PART - 1}"] = 99
            with self.assertRaises(PeerError) as caught:
                client(peer).get_bytes("/blob")
            self.assertIsNone(caught.exception.status)

    def test_a_peer_that_does_not_range_is_read_as_one_stream_without_fetching_twice(self):
        with Peer(BODY, ranges=False) as peer, tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(client(peer).get_bytes("/blob"), BODY)
            self.assertEqual(len(peer.requests), 1)
            target = Path(tmp) / "out"
            client(peer).get_file("/blob", target)
            self.assertEqual(target.read_bytes(), BODY)
            self.assertEqual(len(peer.requests), 2)

    def test_cancelling_mid_download_returns_within_a_second(self):
        with Peer(blob(4_000_000), rate=100_000) as peer:  # 40 s on one connection
            began = time.perf_counter()
            stop_at = began + .3
            with self.assertRaises(Cancelled):
                client(peer, part=500_000, streams=8).get_bytes("/blob", cancel=lambda: time.perf_counter() > stop_at)
            self.assertLess(time.perf_counter() - began, 1.)

    def test_a_cancel_before_the_size_is_known_returns_fast_too(self):
        with Peer(BODY, latency=5.) as peer:
            began = time.perf_counter()
            with self.assertRaises(Cancelled):
                client(peer).get_bytes("/blob", cancel=lambda: time.perf_counter() > began + .2)
            self.assertLess(time.perf_counter() - began, 1.)

    def test_the_dev_server_prefix_works_with_ranges(self):
        with Peer(BODY, dev=True) as peer:
            ranged = client(peer)
            self.assertEqual(ranged.get_bytes("/blob"), BODY)
            self.assertTrue(ranged.base.endswith("/api"))
            self.assertEqual(ranged.get_bytes("/blob"), BODY)  # the prefix is kept: no second probe of the web page
            pages = [r for r in peer.requests if r[1] != "/blob"]
            self.assertEqual(pages, [])

    def test_a_missing_path_is_an_error_not_a_hang(self):
        with Peer(BODY) as peer:
            with self.assertRaises(PeerError) as caught:
                client(peer).get_bytes("/nothing")
            self.assertEqual(caught.exception.status, 404)

    def test_one_stream_is_the_old_behaviour(self):
        with Peer(BODY) as peer:
            self.assertEqual(client(peer, streams=1).get_bytes("/blob"), BODY)
            self.assertEqual([header for method, path, header in peer.requests], [None])


class ParallelUploads(unittest.TestCase):
    def test_parts_posted_at_once_are_assembled_into_the_body(self):
        with Peer() as peer:
            posted = client(peer).post_bytes("/up", BODY, {"node": "a"})
            self.assertEqual(posted, {"accepted": True})
            self.assertEqual(peer.uploads, [BODY])
            parts = [r for r in peer.requests if r[0] == "POST" and r[2] not in (None, "0") or r[2] == "0"]
            self.assertEqual(len(parts), 11 + 0)  # eleven parts (the probe has no `part`)

    def test_a_peer_without_the_parts_route_gets_one_post_and_is_not_asked_again(self):
        with Peer(parts=False) as peer:
            posting = client(peer)
            posting.post_bytes("/up", BODY)
            posting.post_bytes("/up", BODY)
            self.assertEqual(peer.uploads, [BODY, BODY])
            self.assertIs(posting.parts_route, False)
            self.assertEqual([path for method, path, _ in peer.requests], ["/up/parts", "/up", "/up"])  # one probe, ever

    def test_a_small_body_is_one_post(self):
        with Peer() as peer:
            client(peer).post_bytes("/up", BODY[:PART])
            self.assertEqual([path for method, path, _ in peer.requests], ["/up"])

    def test_the_dev_server_prefix_works_with_parts(self):
        with Peer(dev=True) as peer:
            posting = client(peer)
            posting.post_bytes("/up", BODY)
            self.assertEqual(peer.uploads, [BODY])
            self.assertTrue(posting.base.endswith("/api"))


class Assembly(unittest.TestCase):
    def parts(self, body, size):
        digest = hashlib.sha256(body).hexdigest()
        chunks = [body[i:i + size] for i in range(0, len(body), size)]
        return [(n, len(chunks), len(body), digest, c, hashlib.sha256(c).hexdigest()) for n, c in enumerate(chunks)]

    def test_parts_in_any_order_make_the_body_and_say_how_long_it_took(self):
        now = [10.]
        assembler = studio_parts.Assembler(clock=lambda: now[0])
        body, sent = blob(1000), None
        sent = self.parts(body, 300)
        for index in (2, 0, 3, 1):
            now[0] += 1
            whole, seconds = assembler.add("k", *sent[index], began=10.)
        self.assertEqual((whole, seconds), (body, 4.))

    def test_a_part_with_the_wrong_hash_is_rejected_and_the_rest_still_assemble(self):
        assembler, sent = studio_parts.Assembler(), self.parts(blob(1000), 500)
        n, count, total, digest, data, _ = sent[0]
        with self.assertRaises(studio_parts.BadPart):
            assembler.add("k", n, count, total, digest, data[:-1] + b"?", sent[0][5])
        self.assertIsNone(assembler.add("k", *sent[0])[0])
        self.assertEqual(assembler.add("k", *sent[1])[0], blob(1000))

    def test_a_body_that_fails_the_whole_hash_is_dropped(self):
        assembler, sent = studio_parts.Assembler(), self.parts(blob(1000), 500)
        wrong = hashlib.sha256(b"another body").hexdigest()
        assembler.add("k", sent[0][0], 2, 1000, wrong, sent[0][4], sent[0][5])
        with self.assertRaises(studio_parts.BadPart):
            assembler.add("k", sent[1][0], 2, 1000, wrong, sent[1][4], sent[1][5])
        self.assertEqual(assembler.open, {})

    def test_a_part_out_of_range_or_a_size_that_does_not_add_up_is_rejected(self):
        assembler, sent = studio_parts.Assembler(), self.parts(blob(1000), 500)
        with self.assertRaises(studio_parts.BadPart):
            assembler.add("k", 2, 2, 1000, sent[0][3], sent[0][4], sent[0][5])
        with self.assertRaises(studio_parts.BadPart):
            assembler.add("k", 0, 0, 1000, sent[0][3], sent[0][4], sent[0][5])
        assembler.add("k", sent[0][0], 2, 999, sent[0][3], sent[0][4], sent[0][5])
        with self.assertRaises(studio_parts.BadPart):
            assembler.add("k", sent[1][0], 2, 999, sent[0][3], sent[1][4], sent[1][5])

    def test_an_unfinished_body_is_forgotten_after_a_while(self):
        now = [0.]
        assembler, sent = studio_parts.Assembler(clock=lambda: now[0]), self.parts(blob(1000), 500)
        assembler.add("old", *sent[0])
        now[0] = studio_parts.KEEP + 1
        assembler.add("new", *sent[0])
        self.assertEqual(list(assembler.open), ["new"])


class Ranges(unittest.TestCase):
    def test_a_range_header_against_a_body(self):
        span = studio_parts.byte_range
        self.assertEqual(span("bytes=0-99", 1000), (0, 99))
        self.assertEqual(span("bytes=900-", 1000), (900, 999))
        self.assertEqual(span("bytes=990-5000", 1000), (990, 999))  # past the end is the end
        self.assertEqual(span("bytes=-10", 1000), (990, 999))
        self.assertIsNone(span(None, 1000))
        self.assertIsNone(span("bytes=0-1,5-6", 1000))  # several ranges: the whole body is a fine answer
        self.assertIsNone(span("items=0-1", 1000))
        for bad in ("bytes=1000-", "bytes=5-2", "bytes=-0"):
            with self.assertRaises(studio_parts.BadPart):
                span(bad, 1000)


def measure(size=4 << 20, part=256 << 10, streams=8, rate=512 << 10, latency=.12):
    """Seconds to move `size` bytes each way, over one stream and over `streams`, with every connection held to `rate` bytes/s and
    `latency` seconds before it answers (a long link: one stream's ceiling, not the line's)."""
    body, out = blob(size), {}
    with Peer(body, latency=latency, rate=rate) as peer:
        for name, count in (("single", 1), ("parallel", streams)):
            talk = PeerClient(peer.address, streams=count, part=part)
            began = time.perf_counter()
            assert talk.get_bytes("/blob") == body
            down = time.perf_counter() - began
            began = time.perf_counter()
            talk.post_bytes("/up", body)
            out[name] = {"down_s": down, "up_s": time.perf_counter() - began, "down_mb_s": size / down / 2 ** 20, "up_mb_s": size / (time.perf_counter() - began) / 2 ** 20}
        assert peer.uploads == [body, body]
    return out


class Throughput(unittest.TestCase):
    def test_eight_streams_move_a_body_several_times_faster_than_one_when_each_connection_is_slow(self):
        result = measure(size=1 << 20, part=128 << 10, streams=8, rate=2 << 20, latency=.02)
        self.assertGreater(result["parallel"]["down_mb_s"], 2.5 * result["single"]["down_mb_s"], result)
        self.assertGreater(result["parallel"]["up_mb_s"], 2.5 * result["single"]["up_mb_s"], result)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        for label, options in (("0.5 MB/s per connection, 120 ms", dict(rate=512 << 10, latency=.12)),
                               ("0.25 MB/s per connection, 120 ms", dict(rate=256 << 10, latency=.12))):
            result = measure(**options)
            print(f"{label}, 4 MiB each way, 8 streams of 256 KiB parts")
            for name, row in result.items():
                print(f"  {name:9s} down {row['down_mb_s']:5.2f} MB/s ({row['down_s']:5.2f} s)   up {row['up_mb_s']:5.2f} MB/s ({row['up_s']:5.2f} s)")
    else:
        unittest.main()
