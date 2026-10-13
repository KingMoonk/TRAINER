import queue
import tempfile
import threading
import unittest

import copy
import math

from studio_experiments import ExperimentSpec, best_lr, match, plan, scaled_spec, seed_stats, shrunk, start, summarize
from studio_model import train
from studio_spec import default_spec, validate_spec


class Experiments(unittest.TestCase):
    def spec(self):
        spec = default_spec()
        spec['dims'].update(d_model=16, n_layer=1, n_ctx=32, dropout=0.)
        return validate_spec(spec)

    def test_matching_changes_one_width_and_reports_impossible(self):
        base = self.spec()
        variant = self.spec()
        variant['block']['channel']['config']['d_ff'] = 128
        for mode in ('params', 'compute'):
            result = match(base, variant, mode)
            self.assertTrue(result['matched'])
            self.assertLessEqual(result['relative_error'], .02)
            self.assertEqual(result['adjustments'][0]['field'], 'block.channel.config.d_ff')
        variant['block']['channel'] = {'id': 'none'}
        self.assertFalse(match(base, variant)['matched'])
        self.assertTrue(match(base, variant, 'tokens')['matched'])

    def test_seed_spread_is_descriptive(self):
        values = seed_stats([1., 2., 3.], [1., 2., 2.5])
        self.assertEqual(values['mean'], 2.)
        self.assertTrue(values['within_noise'])
        self.assertIsNone(seed_stats([1.], [2.])['within_noise'])
        with self.assertRaises(ValueError):
            seed_stats([float('nan')])

    def test_worker_token_budget_and_reopen(self):
        spec = self.spec()
        experiment = ExperimentSpec(spec, [spec], seeds=[3], token_budget=1, steps=10)
        events = queue.Queue()
        with tempfile.TemporaryDirectory() as root:
            thread = start(experiment, [{'text': 'one two three ' * 8}], root, events,
                           config={'batch_size': 1, 'validation': 0., 'objective': 'all', 'device': 'cpu'})
            thread.join(10)
            self.assertFalse(thread.is_alive())
            loaded = ExperimentSpec.load(root + '/experiment.json')
            self.assertEqual(loaded.seeds, [3])
        values = list(events.queue)
        self.assertFalse([value for value in values if value['type'] == 'experiment_error'], values)
        done = [value for value in values if value['type'] == 'experiment_done'][0]
        self.assertEqual(len(done['runs']), 2)
        for row in done['runs']:
            self.assertEqual(row['record']['step'], 1)
            self.assertEqual(row['record']['budget_reached'], 'tokens')
            self.assertGreaterEqual(row['record']['trained_tokens'], 1)

    def test_wall_clock_budget_and_cancellation(self):
        cfg = dict(width=16, layers=1, heads=2, context=32, steps=100, batch_size=1,
                   validation=0., device='cpu', objective='all')
        with tempfile.TemporaryDirectory() as root:
            record = train(cfg, [{'text': 'one two ' * 20}], root, wall_seconds=.001)
            self.assertEqual(record['budget_reached'], 'wall-clock')
            self.assertLess(record['step'], 100)
        stop = threading.Event()
        stop.set()
        events = queue.Queue()
        with tempfile.TemporaryDirectory() as root:
            thread = start(ExperimentSpec(self.spec()), [{'text': 'one two'}], root, events, stop=stop)
            thread.join(5)
            self.assertTrue(list(events.queue)[-1]['cancelled'])


class Planned(unittest.TestCase):
    spec = Experiments.spec

    def variants(self, n):
        out = []
        for i in range(n):
            v = copy.deepcopy(self.spec())
            v['block']['channel']['config']['d_ff'] = 32 * (i + 1)
            out.append(v)
        return out

    def test_plan_expands_runs_and_sweeps(self):
        e = ExperimentSpec(self.spec(), self.variants(3), seeds=3)
        p = plan(e)
        self.assertEqual((p['n_runs'], p['n_sweeps']), (12, 9))  # base + 3 variants, 3 seeds; base is not swept
        self.assertEqual({r['seed'] for r in p['runs']}, set(e.seed_list()))
        self.assertEqual(sorted({s['factor'] for s in p['sweeps']}), [.5, 1., 2.])
        e.lr_retune = False
        self.assertEqual(plan(e)['n_sweeps'], 0)
        self.assertEqual(plan(ExperimentSpec(self.spec(), self.variants(1), seeds=[1, 2], mode='wall-clock'))['runs'][0]['steps'], None)

    def test_compute_mode_gives_cheaper_variants_more_steps(self):
        cheap = self.spec()
        cheap['block']['channel']['config']['d_ff'] = 8
        p = plan(ExperimentSpec(self.spec(), [cheap], seeds=1, mode='compute', steps=100))
        self.assertGreaterEqual(p['variants'][1]['steps'], 100)

    def test_lr_sweep_picks_best_finite(self):
        self.assertEqual(best_lr([(.5e-3, 3.2), (1e-3, 3.0), (2e-3, 3.4)]), 1e-3)
        self.assertEqual(best_lr([(.5e-3, 3.2), (1e-3, math.nan), (2e-3, 2.9)]), 2e-3)
        self.assertEqual(best_lr([(.5e-3, math.nan), (1e-3, math.nan), (2e-3, math.inf)]), 1e-3)

    def test_summary_spread_and_inside_the_spread(self):
        def run(variant, seed, bpc):
            return {'variant': variant, 'seed': seed, 'bpc': bpc, 'curve': [(1, bpc + 1), (2, bpc)], 'pieces_per_s': 10., 'peak_mb': 5., 'lr': 1e-3}
        rows = summarize([run(0, 1, 3.0), run(0, 2, 3.1), run(0, 3, 2.9), run(1, 1, 3.05), run(1, 2, 3.15), run(1, 3, 2.95),
                          run(2, 1, 2.5), run(2, 2, 2.6), run(2, 3, 2.4)])
        base, same, better = rows
        self.assertAlmostEqual(base['spread'], .2)
        self.assertAlmostEqual(same['delta'], .05)
        self.assertTrue(same['inside_spread'])
        self.assertAlmostEqual(better['delta'], -.5)
        self.assertFalse(better['inside_spread'])
        self.assertEqual(better['curve'], [[1., 3.5], [2., 2.5]])
        self.assertIsNone(summarize([run(0, 1, 3.), run(1, 1, 2.)])[1]['inside_spread'])  # one seed: no spread

    def test_schedule_scales_and_shrink_uses_mup(self):
        a, b = scaled_spec(self.spec(), 100), scaled_spec(self.spec(), 1000)
        self.assertEqual(a['train']['schedule']['config']['stable_fraction'], b['train']['schedule']['config']['stable_fraction'])
        self.assertEqual((a['train']['schedule']['config']['warmup'], b['train']['schedule']['config']['warmup']), (5, 50))
        wide = validate_spec(default_spec())
        small = shrunk(wide)
        self.assertEqual(small['dims']['d_model'], 16)
        self.assertEqual(small['train']['init']['config']['base_width'], wide['dims']['d_model'])

    def test_execute_end_to_end_with_pinned_data_order(self):
        import studio_experiments as ex
        events = []
        e = ExperimentSpec(self.spec(), self.variants(1)[:1], seeds=2, steps=6, question='q')
        docs = [{'text': f'doc {i} ' + 'one two three four ' * 6} for i in range(6)]
        with tempfile.TemporaryDirectory() as root:
            result = ex.execute(e, docs, root, events.append, config={'batch_size': 2, 'device': 'cpu'})
            self.assertTrue((__import__('pathlib').Path(root) / 'results.json').exists())
        self.assertEqual(len(result['sweeps']), 3)
        self.assertEqual([len(v['bpc']) for v in result['variants']], [2, 2])
        self.assertTrue(all(math.isfinite(x) for v in result['variants'] for x in v['bpc']))
        seen = [x for x in events if x['type'] == 'compare' and x['phase'] == 'run']
        self.assertTrue(seen and {'variant', 'seed', 'step', 'bpc'} <= set(seen[0]))
        first = {(x['variant'], x['seed']): x['bpc'] for x in seen if x['step'] == 1}
        self.assertEqual(len(first), 4)
