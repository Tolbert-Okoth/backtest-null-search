import unittest

import numpy as np
import pandas as pd

import nullsearch as ns


def rw(n_days=400, seed=1, drift=0.0, edge_hour=None, edge_bp=0.0):
    idx = pd.date_range("2020-01-01", periods=n_days * 24 * 12, freq="5min"); idx = idx[idx.dayofweek < 5]
    r = np.random.default_rng(seed).normal(drift, 4e-4, len(idx))
    if edge_hour is not None:
        r[idx.hour == edge_hour] += edge_bp * 1e-4 / 12
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


class Stats(unittest.TestCase):
    def test_hurdle_grows_with_n_and_shrinks_with_years(self):
        self.assertGreater(ns.noise_hurdle(10_000, 10), ns.noise_hurdle(100, 10))
        self.assertGreater(ns.noise_hurdle(1_000, 5), ns.noise_hurdle(1_000, 20))
        self.assertAlmostEqual(ns.noise_hurdle(1, 10), 0.0)

    def test_dsr_penalises_more_trials(self):
        x = np.random.default_rng(0).normal(0.0005, 0.01, 2500)
        d1, _ = ns.deflated_sharpe(x, 1); d1000, sr0 = ns.deflated_sharpe(x, 1000)
        self.assertGreater(d1, d1000); self.assertGreater(sr0, 0)

    def test_pbo_near_half_on_noise(self):
        perf = np.random.default_rng(3).normal(0, 1, (480, 40))
        self.assertTrue(0.2 < ns.pbo_cscv(perf, 8) < 0.8)

    def test_psr_bounds(self):
        x = np.random.default_rng(4).normal(0.001, 0.01, 500)
        p = ns.probabilistic_sharpe(x); self.assertTrue(0 <= p <= 1)


class NullPaths(unittest.TestCase):
    def test_sign_flip_keeps_abs_moves_and_drift(self):
        p = rw(30, drift=1e-5); q = ns.sign_flip(p, seed=7)
        lp, lq = np.diff(np.log(p.to_numpy())), np.diff(np.log(q.to_numpy())); mu = lp.mean()
        self.assertTrue(np.allclose(np.abs(lp - mu), np.abs(lq - mu)))
        self.assertTrue(q.index.equals(p.index)); self.assertAlmostEqual(q.iloc[0], p.iloc[0])

    def test_block_shuffle_keeps_length(self):
        p = rw(20); q = ns.block_shuffle_days(p, seed=1); self.assertEqual(len(p), len(q))


class Search(unittest.TestCase):
    def test_matrix_shape_and_bag(self):
        p = rw(200); cfgs, M, d = ns.clock_matrix(p, hours=range(0, 24, 6), holds=(1, 4), weekdays=(-1,))
        self.assertEqual(M.shape, (len(cfgs), len(d)))
        bag = ns.walk_forward_bag(M, d, ["2020-05-01", "2020-07-01"], k=3, min_obs=20)
        self.assertTrue(len(bag) > 0 and np.isfinite(bag).all())

    def test_planted_edge_is_found(self):
        p = rw(400, seed=5, edge_hour=10, edge_bp=25.0)
        cfgs, M, d = ns.clock_matrix(p, hours=(8, 9, 10, 11), holds=(1,), weekdays=(-1,), cost_bp=0.0)
        bag = ns.walk_forward_bag(M, d, ["2020-08-01"], k=1, min_obs=50); self.assertGreater(ns.annual_ir(bag), 1.0)


if __name__ == "__main__":
    unittest.main()
