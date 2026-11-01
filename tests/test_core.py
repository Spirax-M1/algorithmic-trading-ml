import json
import unittest
from dataclasses import replace

import numpy as np
from trading_research.core import (Bars, evaluate, features, fit, predict,
                                   purged_split, run, samples, synthetic, validate)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.bars = synthetic(size=200)

    def test_future_prices_cannot_change_past_features(self):
        changed = self.bars.prices.copy()
        changed[100:] *= 3
        np.testing.assert_allclose(features(self.bars)[:100],
                                   features(replace(self.bars, prices=changed))[:100], equal_nan=True)

    def test_warmup_not_backfilled(self):
        self.assertTrue(np.isnan(features(self.bars)[:10]).all())

    def test_purge_excludes_boundary_labels(self):
        _, _, _, idx = samples(self.bars)
        train, test = purged_split(idx, 160)
        self.assertLess((idx[train]+1).max(), idx[test].min())

    def test_target_matches_next_open_to_close(self):
        _, y, realized, idx = samples(self.bars)
        expected = self.bars.prices[idx+1, 3]/self.bars.prices[idx+1, 0]-1
        np.testing.assert_allclose(realized, expected)
        np.testing.assert_array_equal(y, expected > 0)

    def test_serialized_model_preserves_predictions(self):
        x, y, _, _ = samples(self.bars)
        model = fit(x[:100], y[:100])
        np.testing.assert_allclose(predict(model, x[100:]),
                                   predict(json.loads(json.dumps(model)), x[100:]))
        np.testing.assert_allclose(model['mean'], x[:100].mean(axis=0))

    def test_holdout_cannot_change_model(self):
        changed = self.bars.prices.copy()
        changed[160:] *= 2
        _, first = run(self.bars)
        _, second = run(replace(self.bars, prices=changed))
        self.assertEqual(first, second)

    def test_costs_and_short_direction(self):
        result = evaluate(np.array([.01, -.02]), np.array([1, -1]), 10)
        np.testing.assert_allclose(result['net_returns'], [.009, .019])
        self.assertAlmostEqual(result['total_return'], 1.009*1.019-1)

    def test_no_trade_has_no_costs(self):
        result = evaluate(np.array([.1, -.1]), np.zeros(2), 20)
        self.assertEqual(result['total_return'], 0)
        self.assertIsNone(result['win_rate'])

    def test_drawdown_includes_initial_capital(self):
        result = evaluate(np.array([-.1, .05]), np.ones(2), 0)
        self.assertAlmostEqual(result['max_close_drawdown'], .1)

    def test_reject_bad_data_and_costs(self):
        with self.assertRaises(ValueError):
            validate(replace(self.bars, time=tuple(reversed(self.bars.time))))
        p = self.bars.prices.copy()
        p[0, 1] = 0
        with self.assertRaises(ValueError):
            validate(replace(self.bars, prices=p))
        for cost in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                evaluate(np.array([.01]), np.ones(1), cost)

    def test_feature_schema_checked(self):
        x, y, _, _ = samples(self.bars)
        model = fit(x, y)
        model['features'].reverse()
        with self.assertRaises(ValueError):
            predict(model, x)

    def test_reproducible(self):
        first, _ = run(self.bars)
        second, _ = run(synthetic(size=200))
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
