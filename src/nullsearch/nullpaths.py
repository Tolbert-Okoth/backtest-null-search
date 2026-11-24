"""Null price paths: same noise, no directional structure.

`sign_flip` keeps every timestamp, every gap and the absolute size of every move (so the intraday volatility shape,
volatility clustering and jumps are preserved) and keeps the overall drift, but randomises the direction of each
move around that drift. Any time-of-day, calendar or conditional pattern a search finds on such a path is noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sign_flip(prices: pd.Series, seed: int | None = None, keep_drift: bool = True) -> pd.Series:
    """Return a null path with the same index as `prices` (strictly positive). Log returns r_t become mu + e_t (r_t - mu), e_t = +/-1."""
    p = np.asarray(prices, float)
    if (p <= 0).any():
        raise ValueError("prices must be strictly positive")
    lr = np.diff(np.log(p)); mu = lr.mean() if keep_drift else 0.0
    eps = np.random.default_rng(seed).choice([-1.0, 1.0], len(lr))
    out = np.exp(np.log(p[0]) + np.r_[0.0, np.cumsum(mu + eps * (lr - mu))])
    return pd.Series(out, index=prices.index, name=getattr(prices, "name", None))


def block_shuffle_days(prices: pd.Series, seed: int | None = None) -> pd.Series:
    """Re-order whole days (keeps each day's intraday shape, destroys multi-day structure). Useful for tests of multi-day rules,
    NOT for time-of-day searches (it keeps intraday patterns)."""
    lr = np.log(prices).diff().fillna(0.0); days = lr.index.normalize(); uniq = days.unique()
    order = np.random.default_rng(seed).permutation(len(uniq)); pieces = [lr[days == uniq[i]].to_numpy() for i in order]
    out = np.exp(np.log(prices.iloc[0]) + np.cumsum(np.concatenate(pieces)))
    return pd.Series(out, index=prices.index)
