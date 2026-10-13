import math
import unittest

import numpy as np
import torch

import studio_inspect as inspect
from studio_compile import CompiledModel
from studio_ia.pages.head_zoom import derive
from studio_modules.tokenizer.byte import Tokenizer
from studio_spec import default_spec, validate_spec


def tiny(position='learned', heads=4):
    spec = default_spec()
    spec['dims'].update(d_model=16, n_layer=2, n_ctx=32, dropout=.1)
    spec['position'] = {'id': position, 'config': {}}
    spec['block']['mixer']['config']['heads'] = heads
    model = CompiledModel(validate_spec(spec), Tokenizer('byte').vocab_size, device='cpu')
    # random weights make the pattern non-trivial; the default init is near zero
    rng = np.random.default_rng(0)
    for key in model.p:
        model.p[key].copy_(model.p[key] + 0.5 * torch.as_tensor(rng.normal(size=tuple(model.p[key].shape)), dtype=model.p[key].dtype))
    return model


class InspectViews(unittest.TestCase):
    def setUp(self):
        self.model = tiny()
        self.tokenizer = Tokenizer('byte')
        self.ids = inspect.prepared_text(self.tokenizer, n=4, seed=1)

    def test_prepared_text_repeats_random_pieces(self):
        ids = self.ids
        self.assertEqual(len(ids), 9)
        self.assertEqual(ids[1:5], ids[5:])
        self.assertEqual(len(set(ids[1:5])), 4)
        self.assertNotEqual(ids, inspect.prepared_text(self.tokenizer, n=4, seed=2))

    def test_trace_rows_are_finite_and_cosine_bounded(self):
        result = inspect.trace(self.model, self.ids)
        rows = result['rows']
        self.assertEqual(len(rows), 1 + 2 * 2 + 1)
        self.assertEqual([r['kind'] for r in rows], ['embedding', 'mixer', 'channel', 'mixer', 'channel', 'output'])
        for row in rows:
            for key in ('norm', 'top_p'):
                self.assertTrue(math.isfinite(row[key]), row)
            self.assertTrue(0 < row['top_p'] <= 1)
            if row['kind'] != 'embedding':
                self.assertTrue(math.isfinite(row['rel_change']) and math.isfinite(row['cosine']), row)
                self.assertTrue(-1 <= row['cosine'] <= 1, row)
        self.assertIsNone(rows[0]['cosine'])
        self.assertFalse(rows[-1]['approximate'])
        self.assertEqual(rows[-1]['top_id'], int(inspect.forward(self.model, self.ids)['logits'][0, -1].argmax()))

    def test_head_tests_on_synthetic_patterns(self):
        T = 9
        previous = np.zeros((1, 2, T, T))
        previous[:, 0, np.arange(1, T), np.arange(T - 1)] = 1  # head 0 looks one piece back
        previous[:, 1, :, 0] = 1                                # head 1 parks on the first piece
        previous[:, :, 0, 0] = 1
        got = inspect.head_tests(previous, self.ids)
        self.assertAlmostEqual(got['previous'][0], 1.0)
        self.assertAlmostEqual(got['first'][1], 1.0)
        self.assertLess(got['previous'][1], .2)
        induction = np.zeros((1, 1, T, T))
        for q in range(5, T):                                    # the piece after the earlier copy of the current piece
            induction[0, 0, q, q - 3] = 1
        self.assertAlmostEqual(inspect.head_tests(induction, self.ids)['induction'][0], 1.0)

    def test_derived_softmax_matches_model_pattern(self):
        for position in ('learned', 'alibi'):
            model = tiny(position)
            forward = inspect.forward(model, [self.ids])
            for block in range(2):
                for head in range(4):
                    d = derive(forward, block, head)
                    self.assertLess(d['error'], 1e-4, (position, block, head))
                    np.testing.assert_allclose(d['weights'].sum(axis=0), 1, atol=1e-9)
                    self.assertEqual(float(np.tril(d['weights'], -1).sum()), 0.0)  # keys after the query carry no weight

    def test_rank_heads_one_entry_per_head(self):
        ranks = inspect.rank_heads(self.model, [self.ids])
        self.assertEqual(sorted((r['block'], r['head']) for r in ranks), [(b, h) for b in range(2) for h in range(4)])
        self.assertTrue(all(math.isfinite(r['rise']) for r in ranks))
        self.assertEqual(len({r['baseline'] for r in ranks}), 1)


if __name__ == '__main__':
    unittest.main()
