"""Application-paced synthetic loopback discovery; not a WAN or GPU benchmark.

Synthetic /blob, /up and /control endpoints exercise the real PeerClient. The
single FIFO byte pacer covers both directions and control replies across all
connections, modeling a shared half-duplex bottleneck at the application layer.
"""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from studio_parts import Assembler, byte_range
from studio_peer import PeerClient
from studio_upload_admission import AdmissionError, UploadAdmission


class SharedPacer:
    """Reserve serialized byte service, then wait without holding the lock."""

    def __init__(self, rate, clock=time.monotonic, sleep=time.sleep):
        self.rate, self.clock, self.sleep = rate, clock, sleep
        self.lock, self.finish, self.bytes = threading.Lock(), 0., 0

    def pace(self, size):
        with self.lock:
            now = self.clock()
            self.finish = max(now, self.finish) + size / self.rate
            delay = self.finish - now
            self.bytes += size
        self.sleep(delay)


def distribution(samples):
    values = sorted(row['seconds'] for row in samples if row['error'] is None)
    result = {'samples': len(samples), 'successes': len(values), 'errors': len(samples) - len(values)}
    for name, q in [('p50', .5), ('p95', .95), ('p99', .99), ('max', 1.)]:
        result[name + '_ms'] = values[max(0, math.ceil(len(values) * q) - 1)] * 1000 if values else None
    return result


def overlap_seconds(intervals):
    events = sorted([(r['start'], 1) for r in intervals] + [(r['end'], -1) for r in intervals])
    active, previous, overlap = 0, 0., 0.
    for stamp, change in events:
        if active >= 2:
            overlap += stamp - previous
        active += change
        previous = stamp
    return overlap


class DiscoveryServer(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True


class SyntheticPeer:
    def __init__(self, payload, rate, latency, outstanding=32 << 20, turn=4 << 20):
        self.payload, self.latency = payload, latency
        self.pacer, self.assembler = SharedPacer(rate), Assembler()
        self.admission_events = []
        self.admission = UploadAdmission(max_outstanding=outstanding, turn_bytes=turn, on_event=lambda kind, event: self.admission_events.append({"kind": kind, **event}))
        self.uploads, self.lock = {}, threading.Lock()
        peer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def reply(self, status, body, headers=None):
                self.send_response(status)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Connection', 'close')
                self.close_connection = True
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                for start in range(0, len(body), 16384):
                    chunk = body[start:start + 16384]
                    peer.pacer.pace(len(chunk))
                    self.wfile.write(chunk)

            def do_GET(self):
                time.sleep(peer.latency)
                url = urlparse(self.path)
                path = url.path
                query = {key: values[0] for key, values in parse_qs(url.query).items()}
                if path == '/control':
                    self.reply(200, b'{"ok":true}')
                elif path == '/up/transfers/capabilities':
                    self.reply(200, json.dumps(peer.admission.capabilities()).encode())
                elif path.startswith('/up/transfers/') and path.endswith('/grants'):
                    ident = path.split('/')[-2]
                    identity = peer.transfer_identity(query)
                    try:
                        result = peer.admission.grant(ident, query['incarnation'], identity)
                    except AdmissionError as exc:
                        self.reply(409, json.dumps({'detail': str(exc)}).encode())
                        return
                    self.reply(200, json.dumps(result).encode())
                elif path == '/blob':
                    span = byte_range(self.headers.get('Range'), len(peer.payload))
                    if span is None:
                        self.reply(200, peer.payload)
                    else:
                        first, last = span
                        self.reply(206, peer.payload[first:last + 1],
                                   {'Content-Range': f'bytes {first}-{last}/{len(peer.payload)}'})
                else:
                    self.reply(404, b'{}')

            def do_DELETE(self):
                url = urlparse(self.path)
                query = {key: values[0] for key, values in parse_qs(url.query).items()}
                peer.admission.cancel_transfer(url.path.split('/')[-1], query.get('incarnation'))
                self.reply(204, b'')

            def do_POST(self):
                self.receiving = None
                self.applying = None
                try:
                    self.post()
                finally:
                    if self.applying is not None:
                        peer.admission.end_apply(self.applying)
                    if self.receiving is not None:
                        peer.admission.end_part(self.receiving)

            def post(self):
                time.sleep(peer.latency)
                url = urlparse(self.path)
                query = {k: v[0] for k, v in parse_qs(url.query).items()}
                size = int(self.headers.get('Content-Length', '0'))
                granted_part = url.path.startswith('/up/transfers/') and url.path.endswith('/parts')
                disposition = None
                grant_id = query.get('grant_id')
                if granted_part:
                    try:
                        identity = peer.transfer_identity(query)
                        part_size = int(query['size'])
                        disposition = peer.admission.begin_part(url.path.split('/')[-2], query['incarnation'], identity,
                                                               grant_id, int(query['part']), part_size)
                        if disposition == 'duplicate':
                            self.reply(202, b'{"accepted":true}')
                            return
                        self.receiving = grant_id
                        if size != part_size:
                            raise AdmissionError('Content-Length differs from the granted part size.')
                    except (AdmissionError, KeyError, ValueError) as exc:
                        self.reply(409, json.dumps({'detail': str(exc)}).encode())
                        return
                body = bytearray()
                while len(body) < size:
                    chunk = self.rfile.read(min(16384, size - len(body)))
                    if not chunk:
                        if granted_part:
                            peer.admission.cancel_part(grant_id)
                        self.reply(400, b'{}')
                        return
                    if granted_part and len(body) + len(chunk) > part_size:
                        peer.admission.cancel_part(grant_id)
                        self.reply(409, b'{"detail":"part exceeds its grant"}')
                        return
                    peer.pacer.pace(len(chunk))
                    body.extend(chunk)
                if url.path == '/up/parts' and int(query['count']) == 0:
                    self.reply(202, b'{}')
                    return
                if url.path == '/up/transfers':
                    try:
                        manifest = json.loads(bytes(body))
                        identity = peer.transfer_identity({**query, **manifest})
                        result = peer.admission.announce(identity, manifest['total'], manifest['sha256'],
                                                         manifest['count'], manifest['part_size'])
                    except (AdmissionError, KeyError, ValueError) as exc:
                        self.reply(409, json.dumps({'detail': str(exc)}).encode())
                        return
                    self.reply(200, json.dumps(result).encode())
                    return
                if url.path.startswith('/up/transfers/') and url.path.endswith('/parts'):
                    ident = url.path.split('/')[-2]
                    identity = peer.transfer_identity(query)
                    try:
                        part = int(query['part'])
                        if len(body) != part_size:
                            raise AdmissionError('Part length differs from its grant.')
                        complete = peer.admission.finish_part(ident, query['incarnation'], identity, grant_id, part,
                                                              bytes(body), query['part_sha256'])
                        if complete is not None:
                            self.applying = ident
                            peer.admission.pending(ident, query['incarnation'], identity)
                            whole, _ = complete
                            with peer.lock:
                                peer.uploads[query['member']] = hashlib.sha256(whole).hexdigest()
                            peer.admission.mark_complete(ident, query['incarnation'], identity)
                    except (AdmissionError, KeyError, ValueError) as exc:
                        self.reply(409, json.dumps({'detail': str(exc)}).encode())
                        return
                    self.reply(202, b'{"accepted":true}')
                    return
                if url.path not in ('/up', '/up/parts'):
                    self.reply(404, b'{}')
                    return
                whole = bytes(body)
                if url.path.endswith('/parts'):
                    whole, _ = peer.assembler.add((query['member'],), int(query['part']), int(query['count']),
                                                  int(query['total']), query['sha256'], whole, query['part_sha256'])
                if whole is not None:
                    with peer.lock:
                        peer.uploads[query['member']] = hashlib.sha256(whole).hexdigest()
                self.reply(202, b'{}')

        self.server = DiscoveryServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
        self.address = f'http://127.0.0.1:{self.server.server_port}'

    @staticmethod
    def transfer_identity(query):
        return ('discovery', 1, 0, query.get('member', query.get('node', '')), int(query.get('generation', 0)),
                int(query.get('epoch', 0)), int(query.get('tokens', 0)), int(query.get('steps', 1)),
                float(query.get('seconds', 0.)), None, float(query.get('down_bps', 0.)))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def run_case(payload, rate, latency, streams, members, part, idle_seconds, control_interval, direction="exchange",
             receiver_grants=False, matched_geometry=False, outstanding=32 << 20, turn=4 << 20):
    """Real HTTP, synthetic endpoints and one shared half-duplex application service budget."""
    with SyntheticPeer(payload, rate, latency, outstanding, turn) as peer:
        controls, transfers, missed = [], [], []
        probe = PeerClient(peer.address, retries=0, timeout=30)
        origin = time.monotonic()

        def control(phase, scheduled):
            start, error = time.monotonic(), None
            try:
                if probe.get_json('/control') != {'ok': True}:
                    raise ValueError('invalid control reply')
            except Exception as exc:
                error = str(exc)
            controls.append({'phase': phase, 'scheduled': scheduled - origin, 'dispatch_lag': start - scheduled, 'start': start - origin,
                             'seconds': time.monotonic() - start, 'error': error})

        stop = threading.Event()

        def sampling(phase, until=None):
            scheduled = time.monotonic()
            while not stop.is_set() and (until is None or scheduled < until):
                if stop.wait(max(0., scheduled - time.monotonic())):
                    break
                control(phase, scheduled)
                scheduled += control_interval
                now = time.monotonic()
                # Do not silently reduce the offered load when one request takes longer than the cadence.
                while scheduled < now and (until is None or scheduled < until):
                    missed.append({'phase': phase, 'scheduled': scheduled - origin})
                    scheduled += control_interval

        sampling('idle', time.monotonic() + idle_seconds)
        digest = hashlib.sha256(payload).hexdigest()
        barrier = threading.Barrier(members)

        def member(number):
            client = PeerClient(peer.address, streams=streams, part=part, retries=0, timeout=30,
                                receiver_grants=receiver_grants)
            barrier.wait()
            for flow in (('up', 'down') if direction == 'exchange' else (direction,)):
                start, error, valid = time.monotonic(), None, False
                try:
                    if flow == 'up':
                        (client._post_parts if matched_geometry and not receiver_grants else client.post_bytes)(
                            '/up', payload, {'member': str(number)}, timeout=30)
                        valid = peer.uploads.get(str(number)) == digest
                    else:
                        valid = hashlib.sha256(client.get_bytes('/blob')).hexdigest() == digest
                    if not valid:
                        raise ValueError('payload integrity failed')
                except Exception as exc:
                    error = str(exc)
                transfers.append({'member': number, 'direction': flow, 'start': start - origin,
                                  'end': time.monotonic() - origin, 'bytes': len(payload), 'valid': valid, 'error': error})

        sampler = threading.Thread(target=sampling, args=("bulk",))
        sampler.start()
        try:
            with ThreadPoolExecutor(max_workers=members) as pool:
                list(pool.map(member, range(members)))
        finally:
            stop.set()
            sampler.join()
        began, ended = min(t['start'] for t in transfers), max(t['end'] for t in transfers)
        elapsed = ended - began
        useful = sum(t['bytes'] for t in transfers if t['valid'] and t['error'] is None)
        for row in controls:
            if row['phase'] == 'bulk' and not any(t['start'] < row['start'] + row['seconds'] and t['end'] > row['start']
                                                for t in transfers):
                row['phase'] = 'transition'
        for row in missed:
            if row['phase'] == 'bulk' and not any(t['start'] <= row['scheduled'] < t['end'] for t in transfers):
                row['phase'] = 'transition'
        summaries = {}
        for phase in ('idle', 'bulk', 'transition'):
            summaries[phase] = distribution([r for r in controls if r['phase'] == phase])
            summaries[phase]['missed_dispatches'] = sum(r['phase'] == phase for r in missed)
        per_member = []
        for number in range(members):
            rows = [row for row in transfers if row['member'] == number]
            useful_member = sum(row['bytes'] for row in rows if row['valid'] and row['error'] is None)
            duration = max(row['end'] for row in rows) - min(row['start'] for row in rows)
            per_member.append({'member': number, 'useful_bytes': useful_member, 'duration_seconds': duration,
                               'useful_bytes_per_second': useful_member / max(duration, 1e-9),
                               'errors': sum(row['error'] is not None for row in rows)})
        rates = [row['useful_bytes_per_second'] for row in per_member]
        fairness = sum(rates) ** 2 / (len(rates) * sum(rate ** 2 for rate in rates)) if rates and any(rates) else None
        return {'streams': streams, 'members': members, 'direction': direction, 'receiver_grants': receiver_grants,
                'part_bytes': part, 'matched_geometry': matched_geometry, 'transfers': transfers, 'per_member': per_member, 'jain_fairness': fairness,
                'admission_events': peer.admission_events,
                'control_raw': controls,
                'missed_control_raw': missed, 'control': summaries,
                'bulk_seconds': elapsed, 'overlap_seconds': overlap_seconds(transfers),
                'upload_overlap_seconds': overlap_seconds([t for t in transfers if t['direction'] == 'up']),
                'download_overlap_seconds': overlap_seconds([t for t in transfers if t['direction'] == 'down']),
                'useful_bytes': useful, 'aggregate_useful_bytes_per_second': useful / elapsed,
                'paced_bytes': peer.pacer.bytes, 'integrity_ok': all(t['valid'] and t['error'] is None for t in transfers)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--payload-mib', type=float, default=8)
    parser.add_argument('--rate-mib', type=float, default=2)
    parser.add_argument('--latency-ms', type=float, default=30)
    parser.add_argument('--part-kib', type=int, default=256)
    parser.add_argument('--members', type=int, default=2)
    parser.add_argument('--direction', choices=('up', 'down', 'exchange'), default='exchange')
    parser.add_argument('--admission', choices=('baseline', 'grants', 'both'), default='baseline',
                        help='upload admission policy; both compares legacy multipart with receiver grants at identical part geometry')
    parser.add_argument('--window-kib', type=int, default=32768, help='receiver outstanding byte budget in KiB')
    parser.add_argument('--turn-kib', type=int, default=4096, help='receiver per-transfer byte window in KiB')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--idle-seconds', type=float, default=3)
    parser.add_argument('--control-interval', type=float, default=.02)
    args = parser.parse_args()
    values = (args.payload_mib, args.rate_mib, args.part_kib, args.members, args.idle_seconds, args.control_interval, args.repeats, args.window_kib, args.turn_kib)
    if any(not math.isfinite(v) or v <= 0 for v in values) or not math.isfinite(args.latency_ms) or args.latency_ms < 0:
        parser.error('sizes, rate, members and durations must be positive; latency must be nonnegative')
    if args.admission == 'both' and args.part_kib > min(args.window_kib, args.turn_kib, 4096):
        parser.error('matched comparison requires parts to fit the receiver window, turn and 4 MiB grant limit')
    size = int(args.payload_mib * 2**20)
    if size < 1:
        parser.error('payload must contain at least one byte')
    payload = (hashlib.sha256(b'648 discovery').digest() * (size // 32 + 1))[:size]
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    result = {'kind': 'application-paced synthetic loopback; no WAN/GPU inference',
              'clock': 'time.monotonic', 'percentiles': 'nearest rank; successful calls only; missed dispatches reported separately',
              'bottleneck': 'one shared FIFO byte-service budget for upload reads, download writes and control replies',
              'control_endpoint': 'synthetic /control, not production progress or round-view routes',
              'stream_comparison': ('Matched multipart upload geometry for baseline/grants at both stream settings' if args.admission == 'both'
                                    else 'Legacy 1 uses whole-body HTTP; legacy 8 uses multipart/ranges; grants use negotiated parts'),
              'admission_order': 'alternates baseline/grants by repetition when --admission both is selected',
              'latency_model': 'fixed handler delay before processing each request; not TCP RTT/loss simulation',
              'platform': platform.platform(), 'python': platform.python_version(), 'revision': revision, 'dirty': dirty,
              'module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'cases': []}
    policies = (False, True) if args.admission == 'both' else (args.admission == 'grants',)
    for repetition in range(args.repeats):
        policy_order = policies if repetition % 2 == 0 else tuple(reversed(policies))
        for receiver_grants in policy_order:
            stream_order = (1, 8) if repetition % 2 == 0 else (8, 1)
            for streams in stream_order:
                case = run_case(payload, args.rate_mib * 2**20, args.latency_ms / 1000, streams,
                                args.members, args.part_kib * 1024, args.idle_seconds, args.control_interval,
                                args.direction, receiver_grants, args.admission == 'both',
                                args.window_kib * 1024, args.turn_kib * 1024)
                case['repetition'] = repetition + 1
                result['cases'].append(case)
                print(json.dumps({k: v for k, v in case.items()
                                  if k not in ('transfers', 'admission_events', 'control_raw', 'missed_control_raw')}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if not all(case['integrity_ok'] for case in result['cases']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
