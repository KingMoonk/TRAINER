import tempfile
import unittest

import numpy as np

from studio_compile import CompiledModel
from studio_model import build_optimizer, train
from studio_snapshot import capture, load, one_step, save
from studio_spec import default_spec, validate_spec


class Snapshots(unittest.TestCase):
    def model(self):
        spec = default_spec()
        spec['dims'].update(d_model=16, n_layer=1, n_ctx=32, dropout=0.)
        return CompiledModel(validate_spec(spec), 10, device='cpu')

    def test_exact_step_matches_real_update_and_recorded_layer_values(self):
        model = self.model()
        optimizer, _ = build_optimizer(model)
        ids = np.array([[1, 2, 3]])
        targets = np.array([[2, 3, 4]])
        snapshot = one_step(model, ids, targets, optimizer=optimizer)
        before = {key: value.copy() for key, value in model.p.items()}
        logits, tape = model.forward(ids, training=True, tape=True, trace=True)
        loss, dy = model.objective(logits, targets, gradient=True)
        model.backward(dy, tape)
        optimizer.update(model, .001)
        self.assertAlmostEqual(snapshot['loss'], loss)
        for key, value in model.p.items():
            np.testing.assert_array_equal(snapshot['arrays'][f'weight::{key}'], value)
            np.testing.assert_array_equal(snapshot['arrays'][f'update::{key}'], value - before[key])
        for key, value in tape.forward_values.items():
            np.testing.assert_array_equal(snapshot['arrays'][f'forward::{key}'], value)
        for key, value in tape.backward_values.items():
            np.testing.assert_array_equal(snapshot['arrays'][f'backward::{key}'], value)

    def test_round_trip_without_pickle(self):
        snapshot = capture(self.model(), [[1, 2]], 4)
        with tempfile.TemporaryDirectory() as root:
            restored = load(save(root, snapshot))
        self.assertEqual(restored['spec'], snapshot['spec'])
        self.assertEqual(restored['step'], 4)
        for key, value in snapshot['arrays'].items():
            np.testing.assert_array_equal(restored['arrays'][key], value)

    def test_training_saves_fixed_probe_snapshots(self):
        events = []
        cfg = dict(width=16, layers=1, heads=2, context=32, steps=2, batch_size=1,
                   validation=0., dropout=0., device='cpu', objective='all', eval_every=2)
        with tempfile.TemporaryDirectory() as root:
            train(cfg, [{'text': 'count one two three ' * 5}], root, emit=events.append, snapshot_every=1)
            paths = [event['path'] for event in events if event['type'] == 'viz_snapshot']
            self.assertEqual(len(paths), 2)
            first, second = map(load, paths)
            np.testing.assert_array_equal(first['arrays']['ids'], second['arrays']['ids'])
            self.assertEqual(second['step'], 2)

    def test_retention_keeps_only_newest_snapshots(self):
        """Saving 25 snapshots with keep=20 leaves exactly the 20 newest."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as root:
            # Save 25 snapshots
            for step in range(1, 26):
                snapshot = capture(self.model(), [[1, 2]], step)
                save(root, snapshot, keep=20)
            # Check that we have exactly 20 files
            viz_dir = Path(root) / 'viz'
            files = sorted(viz_dir.glob('step_*.npz'))
            self.assertEqual(len(files), 20)
            # Check that the oldest ones are deleted (steps 1-5) and newest ones remain (steps 6-25)
            steps = [int(f.stem.split('_')[1]) for f in files]
            self.assertEqual(steps, list(range(6, 26)))

    def test_retention_with_keep_none_keeps_all(self):
        """With keep=None or 0, all snapshots are kept."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as root:
            # Save 25 snapshots with keep=None
            for step in range(1, 26):
                snapshot = capture(self.model(), [[1, 2]], step)
                save(root, snapshot, keep=None)
            viz_dir = Path(root) / 'viz'
            files = sorted(viz_dir.glob('step_*.npz'))
            self.assertEqual(len(files), 25)

    def test_retention_with_keep_zero_keeps_all(self):
        """With keep=0, all snapshots are kept."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as root:
            # Save 25 snapshots with keep=0
            for step in range(1, 26):
                snapshot = capture(self.model(), [[1, 2]], step)
                save(root, snapshot, keep=0)
            viz_dir = Path(root) / 'viz'
            files = sorted(viz_dir.glob('step_*.npz'))
            self.assertEqual(len(files), 25)

    def test_retention_does_not_delete_unrelated_files(self):
        """Unrelated files in the directory are never deleted."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as root:
            viz_dir = Path(root) / 'viz'
            viz_dir.mkdir(parents=True, exist_ok=True)
            # Create an unrelated file
            unrelated = viz_dir / 'important_data.txt'
            unrelated.write_text('important')
            # Save 25 snapshots with keep=20
            for step in range(1, 26):
                snapshot = capture(self.model(), [[1, 2]], step)
                save(root, snapshot, keep=20)
            # Check that the unrelated file still exists
            self.assertTrue(unrelated.exists())
            self.assertEqual(unrelated.read_text(), 'important')
            # Check that we have 20 snapshots + the unrelated file
            all_files = list(viz_dir.iterdir())
            self.assertEqual(len(all_files), 21)
