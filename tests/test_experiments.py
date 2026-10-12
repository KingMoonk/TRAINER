import queue
import tempfile
import threading
import unittest

from studio_experiments import ExperimentSpec, match, seed_stats, start
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
