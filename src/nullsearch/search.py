"""A transparent brute-force "clock" search with walk-forward selection, plus the null-data test that judges it.

The search space is deliberately generic: enter at hour h, hold L hours, long or short, on all days or one weekday.
Each configuration yields one net return per trading day. Selection is walk-forward: at each cut, rank configurations by
t-stat on all data before the cut and hold the equal-weight bag of the top K over the next segment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def hourly_grid(prices: pd.Series) -> pd.DataFrame:
    """Hourly close grid: rows = calendar days, columns = hour 0..47 (two days so holds can cross midnight)."""
    h = prices.resample("1h").last().dropna()
    days = pd.DatetimeIndex(sorted(set(h.index.normalize())))
    full = pd.date_range(days[0], days[-1] + pd.Timedelta(hours=47), freq="1h")
    hv = h.reindex(full).to_numpy()
    off = ((days - days[0]) // pd.Timedelta(hours=1)).astype(int)          # hour offset of each day in the continuous grid
    cols = off.to_numpy()[:, None] + np.arange(48)[None, :]
    return pd.DataFrame(hv[np.clip(cols, 0, len(hv) - 1)], index=days)


@dataclass
class ClockConfig:
    hour: int
    hold: int
    side: int          # +1 long, -1 short
    weekday: int       # -1 = all days

    def label(self) -> str:
        wd = "all" if self.weekday < 0 else "MTWTFSS"[self.weekday]
        return f"{'L' if self.side > 0 else 'S'} {self.hour:02d}h+{self.hold}h {wd}"


def clock_matrix(prices: pd.Series, hours=range(24), holds=range(1, 13), weekdays=(-1, 0, 1, 2, 3, 4), cost_bp: float = 1.0):
    """Net daily returns (bp) for every configuration. Returns (configs, matrix configs x days, day index)."""
    g = hourly_grid(prices); wd = g.index.dayofweek.to_numpy(); cfgs, rows = [], []
    for h in hours:
        for L in holds:
            if h + L > 47:
                continue
            gross = np.log(g[h + L].to_numpy() / g[h].to_numpy()) * 1e4
            for side in (1, -1):
                net = side * gross - cost_bp
                for w in weekdays:
                    m = np.isfinite(net) & ((wd == w) if w >= 0 else True)
                    cfgs.append(ClockConfig(h, L, side, w)); rows.append(np.where(m, net, np.nan))
    return cfgs, np.array(rows, dtype=np.float32), g.index


def walk_forward_bag(matrix: np.ndarray, days: pd.DatetimeIndex, cuts, k: int = 20, min_obs: int = 150) -> pd.Series:
    """Equal-weight out-of-sample returns of the in-sample top-k configurations, stitched across walk-forward segments."""
    cuts = [pd.Timestamp(c) for c in cuts]; out = []
    for i, cut in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else days[-1] + pd.Timedelta(days=1)
        tr, te = days < cut, (days >= cut) & (days < end)
        X = matrix[:, tr]; n = np.isfinite(X).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.nanmean(X, 1) / np.nanstd(X, 1) * np.sqrt(np.maximum(n, 1))
        ok = np.where((n >= min_obs) & np.isfinite(t))[0]; top = ok[np.argsort(-t[ok])][:k]
        Y = matrix[top][:, te]; out.append(pd.Series(np.nanmean(np.where(np.isfinite(Y), Y, 0.0), 0), index=days[te]))
    return pd.concat(out)


def annual_ir(daily) -> float:
    x = np.asarray(daily, float)
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 1 and x.std() > 0 else 0.0


def null_test(prices: pd.Series, cuts, n_null: int = 20, k: int = 20, seed: int = 0, null_fn=None, **grid_kw) -> dict:
    """Run the identical search on the real path and on `n_null` null paths. Returns the real OOS IR, the null distribution and
    the real IR's percentile. A real result below the null 95th percentile is what the search finds in noise."""
    from .nullpaths import sign_flip
    null_fn = null_fn or sign_flip
    cfgs, M, d = clock_matrix(prices, **grid_kw); real = annual_ir(walk_forward_bag(M, d, cuts, k))
    nulls = []
    for i in range(n_null):
        _, Mn, dn = clock_matrix(null_fn(prices, seed=seed + i), **grid_kw); nulls.append(annual_ir(walk_forward_bag(Mn, dn, cuts, k)))
    nulls = np.array(nulls)
    return dict(n_configs=len(cfgs), real_ir=real, null_irs=nulls, null_p95=float(np.quantile(nulls, 0.95)),
                percentile=float(np.mean(nulls < real)), passes=bool(real > np.quantile(nulls, 0.95)))
