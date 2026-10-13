import copy
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import studio_queue
import studio_quicktest as q
from studio_experiments import ExperimentSpec, shrunk
from studio_jobs import HANDLERS
from studio_queue import Job, Plan, Runner
from studio_spec import default_spec, validate_spec

CFG = {'device': 'cpu', 'batch_size': 2}


def spec():
    s = default_spec()
    s['dims'].update(d_model=16, n_layer=1, n_ctx=32, dropout=0.)
    return validate_spec(s)


class QuickTest(unittest.TestCase):
    def test_verdict_needs_a_clear_gap(self):
        self.assertEqual(q.verdict(3., 3.01, .05), 'about the same')
        self.assertEqual(q.verdict(3., 2.8, .05), 'lower')
        self.assertEqual(q.verdict(3., 3.2, .05), 'higher')

    def test_two_short_runs_give_verdict_and_four_checks(self):
        draft = copy.deepcopy(spec())
        draft['block']['mixer']['config']['heads'] = 2
        events = []
        r = q.run(spec(), draft, 'stories', .05, False, emit=events.append, config=CFG)  # 3 seconds each
        self.assertIn(r['verdict'], ('lower', 'about the same', 'higher'))
        self.assertEqual([c['id'] for c in r['checks']], ['starts_near_uniform', 'finite', 'memorises_one_batch', 'error_falls'])
        self.assertTrue(all(c['ok'] is not None for c in r['checks']))
        self.assertTrue(r['current']['sample'] and r['draft']['sample'])
        self.assertTrue({e['which'] for e in events if e['type'] == 'quicktest'} == {'current', 'draft'})
        self.assertLess(abs(r['current']['steps'] - r['draft']['steps']), max(r['current']['steps'], r['draft']['steps']))

    def test_shrink_keeps_shape_and_carries_rate(self):
        small = shrunk(validate_spec(default_spec()))
        self.assertEqual(small['dims']['d_model'], 16)
        self.assertEqual(small['train']['init']['id'], 'mup')

    def test_stop_returns_early(self):
        stop = threading.Event()
        stop.set()
        self.assertTrue(q.run(spec(), spec(), 'stories', .05, False, stop=stop, config=CFG).get('stopped'))


class Handlers(unittest.TestCase):
    def run_job(self, job):
        with tempfile.TemporaryDirectory() as tmp:
            services = SimpleNamespace(workspace=SimpleNamespace(run_dir=tmp), examples=lambda p: [{'text': f'doc {i} ' + 'one two three ' * 8} for i in range(6)])
            runner = Runner(Path(tmp) / 'queue.json', HANDLERS, services, tick=.02)
            runner.set_plan(Plan(hours='any', idle_only=False, pause_on_battery=False))
            self.addCleanup(runner.shutdown)
            seen = []
            runner.subscribe(lambda kind, payload: seen.append((kind, payload)))
            runner.enqueue(job)
            end = __import__('time').time() + 90
            while __import__('time').time() < end and job.status in ('waiting', 'running'):
                __import__('time').sleep(.05)
            return job, seen

    def test_quicktest_handler_completes(self):
        job, seen = self.run_job(Job('quicktest', 'qt', params={'current_spec': spec(), 'draft_spec': spec(), 'dataset': 'stories',
                                                                    'minutes': .03, 'shrink': False, 'config': CFG}))
        self.assertEqual((job.status, job.error), ('done', ''))
        self.assertIn(job.params['result']['verdict'], ('lower', 'about the same', 'higher'))
        self.assertTrue(any(p.get('type') == 'quicktest_done' for k, p in seen if k == 'train'))

    def test_compare_handler_completes(self):
        v = copy.deepcopy(spec())
        v['block']['channel']['config']['d_ff'] = 32
        exp = ExperimentSpec(spec(), [v], seeds=2, steps=5, lr_retune=False)
        from dataclasses import asdict
        job, seen = self.run_job(Job('compare', 'cmp', params={'experiment': asdict(exp), 'config': CFG}))
        self.assertEqual((job.status, job.error), ('done', ''))
        self.assertEqual(len(job.params['result']['variants']), 2)
        self.assertTrue(any(p.get('type') == 'compare' for k, p in seen if k == 'train'))


del studio_queue
