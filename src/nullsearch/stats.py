"""Selection-bias statistics for strategy searches.

References
----------
Bailey, D. H. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio. Journal of Portfolio Management 40(5).
Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2017). The Probability of Backtest Overfitting. Journal of Computational Finance 20(4).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329


def noise_hurdle(n_configs: int, years: float) -> float:
    """Approximate annualised Sharpe ratio the best of `n_configs` zero-edge strategies reaches by luck alone: sqrt(2 ln N) / sqrt(years)."""
    if n_configs < 2 or years <= 0:
        return 0.0
    return math.sqrt(2.0 * math.log(n_configs)) / math.sqrt(years)


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected maximum of `n_trials` independent Sharpe estimates with true value 0 and standard deviation `sr_std` (same periodicity)."""
    if n_trials < 2:
        return 0.0
    return sr_std * ((1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n_trials) + EULER_GAMMA * norm.ppf(1 - 1.0 / (n_trials * math.e)))


def probabilistic_sharpe(returns, sr_benchmark: float = 0.0) -> float:
    """PSR: probability that the true per-period Sharpe exceeds `sr_benchmark`, accounting for sample length, skew and kurtosis."""
    x = np.asarray(returns, float); x = x[np.isfinite(x)]; t = len(x)
    if t < 3 or x.std() == 0:
        return float("nan")
    sr = x.mean() / x.std(); g3 = skew(x); g4 = kurtosis(x, fisher=False)
    se = math.sqrt(max((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (t - 1), 1e-18))
    return float(norm.cdf((sr - sr_benchmark) / se))


def deflated_sharpe(returns, n_trials: int) -> tuple[float, float]:
    """Deflated Sharpe ratio. Returns (DSR probability, annualised hurdle SR0 assuming daily data).
    SR0 is the expected best Sharpe among `n_trials` zero-edge strategies of the same length and shape."""
    x = np.asarray(returns, float); x = x[np.isfinite(x)]; t = len(x)
    if t < 3 or x.std() == 0:
        return float("nan"), float("nan")
    sr = x.mean() / x.std(); g3 = skew(x); g4 = kurtosis(x, fisher=False)
    se = math.sqrt(max((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (t - 1), 1e-18))
    sr0 = expected_max_sharpe(n_trials, se)
    return float(norm.cdf((sr - sr0) / se)), float(sr0 * math.sqrt(252))


def pbo_cscv(perf: np.ndarray, n_splits: int = 16, metric=None) -> float:
    """Probability of Backtest Overfitting via combinatorially symmetric cross-validation.
    perf: array (T periods x N configurations) of per-period returns. Returns the share of splits where the in-sample best
    configuration ranks below the out-of-sample median."""
    perf = np.asarray(perf, float); t, n = perf.shape
    if n_splits % 2 or n_splits < 2:
        raise ValueError("n_splits must be an even number >= 2")
    metric = metric or (lambda a: a.mean(0) / np.where(a.std(0) > 0, a.std(0), np.nan))
    blocks = np.array_split(np.arange(t), n_splits); logits = []
    for combo in itertools.combinations(range(n_splits), n_splits // 2):
        ins = np.concatenate([blocks[i] for i in combo]); oos = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo])
        m_in, m_out = metric(perf[ins]), metric(perf[oos])
        best = int(np.nanargmax(m_in)); rank = (np.sum(m_out <= m_out[best]) - 0.5) / n
        logits.append(math.log(rank / (1 - rank)))
    return float(np.mean(np.array(logits) <= 0))
