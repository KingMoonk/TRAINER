"""Measurement correctness and aggregate shared service checks for #648."""
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from studio_transfer_discovery import SharedPacer, distribution, overlap_seconds, run_case


class MeasurementTests(unittest.TestCase):
    def test_percentiles_count_errors_without_treating_them_as_successes(self):
        rows = [{'seconds': n / 1000, 'error': None} for n in range(1, 101)] + [{'seconds': 10, 'error': 'timeout'}]
        self.assertEqual(distribution(rows), {'samples': 101, 'successes': 100, 'errors': 1,
                                              'p50_ms': 50., 'p95_ms': 95., 'p99_ms': 99., 'max_ms': 100.})

    def test_overlap_is_union_where_two_or_more_are_active(self):
        self.assertEqual(overlap_seconds([{'start': 0, 'end': 5}, {'start': 1, 'end': 3},
                                          {'start': 2, 'end': 4}]), 3)

    def test_shared_reservations_do_not_multiply_capacity_by_connections(self):
        waits, lock = [], threading.Lock()

        def sleep(delay):
            with lock:
                waits.append(delay)

        pacer = SharedPacer(100, clock=lambda: 0., sleep=sleep)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(pacer.pace, [100] * 8))
        self.assertEqual(sorted(waits), list(range(1, 9)))
        self.assertEqual(pacer.bytes, 800)

    def test_slow_controls_report_missed_cadence_and_upload_only_does_not_download(self):
        result = run_case(b'x' * 32768, 512 * 1024, .03, 1, 2, 16384, .05, .005, direction='up')
        self.assertGreater(result['control']['idle']['missed_dispatches'], 0)
        self.assertGreater(result['control']['bulk']['missed_dispatches'], 0)
        self.assertEqual({row['direction'] for row in result['transfers']}, {'up'})
        self.assertEqual(result['useful_bytes'], 65536)
        self.assertEqual(result['download_overlap_seconds'], 0)
        self.assertTrue(all(row['dispatch_lag'] >= 0 for row in result['control_raw']))

    def test_real_http_concurrent_transfers_share_cap_and_verify_payloads(self):
        for streams in (1, 8):
            with self.subTest(streams=streams):
                result = run_case(b'payload!' * 8192, 512 * 1024, .001, streams, 2, 16384, .03, .005)
                self.assertTrue(result['integrity_ok'])
                self.assertEqual(result['useful_bytes'], 4 * 65536)
                self.assertGreater(result['overlap_seconds'], 0)
                self.assertLessEqual(result['aggregate_useful_bytes_per_second'], 512 * 1024 * 1.03)
                self.assertGreater(result['control']['bulk']['samples'], 0)
                self.assertEqual(result['control']['bulk']['errors'], 0)

    def test_real_http_receiver_grants_preserve_integrity_and_fair_member_completion(self):
        result = run_case(b'grant-payload' * 8192, 512 * 1024, .001, 4, 2, 4096, .03, .005,
                          direction='up', receiver_grants=True)
        self.assertTrue(result['integrity_ok'])
        self.assertEqual(result['useful_bytes'], 2 * 13 * 8192)
        self.assertEqual({row['member'] for row in result['transfers'] if row['valid']}, {0, 1})
        self.assertLessEqual(result['aggregate_useful_bytes_per_second'], 512 * 1024 * 1.03)
        self.assertEqual(result['control']['bulk']['errors'], 0)

    def test_required_and_background_exchanges_share_byte_budgets(self):
        result = run_case(b'mixed' * 8192, 512 * 1024, .001, 4, 2, 4096, .03, .005,
                          direction='exchange', receiver_grants=True, matched_geometry=True,
                          outstanding=16384, turn=8192, background_members=1)
        self.assertTrue(result['integrity_ok'], result['transfers'])
        self.assertEqual(result['useful_bytes'], 4 * 5 * 8192)
        self.assertGreater(result['ingress_peak_bytes'], 0)
        self.assertLessEqual(result['ingress_peak_bytes'], 16384)
        self.assertLessEqual(result['egress_peak_bytes'], 16384)
        self.assertEqual([row['background'] for row in result['per_member']], [False, True])
        self.assertEqual(result['control']['bulk']['errors'], 0)


if __name__ == '__main__':
    unittest.main()
