"""Reproduce the README figures (synthetic data only). Run: python examples/headline.py  (needs matplotlib)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nullsearch as ns

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs"); os.makedirs(DOCS, exist_ok=True)


def synthetic(years=6, seed=11, edge_bp_per_day=0.0, edge_hour=14):
    """5-minute weekday prices with intraday volatility seasonality and an optional small edge in one hour."""
    idx = pd.date_range("2018-01-01", periods=years * 365 * 288, freq="5min"); idx = idx[idx.dayofweek < 5]
    rng = np.random.default_rng(seed); hour = idx.hour.to_numpy()
    vol = 3e-4 * (1 + 0.8 * np.exp(-((hour - 14) ** 2) / 8))                     # busier afternoon, like a cash session
    r = rng.normal(2e-7, 1, len(idx)) * vol
    r[hour == edge_hour] += edge_bp_per_day * 1e-4 / 12
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


# 1) noise hurdle
n = np.logspace(1, 5, 60)
fig, ax = plt.subplots(figsize=(7, 4))
for yrs in (5, 10, 20):
    ax.plot(n, [ns.noise_hurdle(int(v), yrs) for v in n], label=f"{yrs} years of data")
ax.set_xscale("log"); ax.set_xlabel("configurations tried (N)"); ax.set_ylabel("annual Sharpe from luck alone")
ax.set_title("The noise hurdle: what the best of N zero-edge strategies shows by chance"); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(DOCS, "noise_hurdle.png"), dpi=130); plt.close(fig)

# 2) null test on noise vs a planted edge
cuts = ["2020-01-01", "2021-07-01", "2023-01-01"]; grid = dict(hours=range(24), holds=range(1, 7), weekdays=(-1, 0, 4), cost_bp=0.5)
res = {}
for lab, edge in (("pure noise (drift only)", 0.0), ("noise + planted 1-hour edge (3 bp/day)", 3.0)):
    p = synthetic(edge_bp_per_day=edge); res[lab] = ns.null_test(p, cuts, n_null=20, k=20, seed=100, **grid)
    r = res[lab]; print(f"{lab}: {r['n_configs']} configs | real OOS IR {r['real_ir']:+.2f} | null p95 {r['null_p95']:+.2f} | percentile {r['percentile']:.2f} | passes {r['passes']}")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (lab, r) in zip(axes, res.items()):
    ax.hist(r["null_irs"], bins=10, color="#9aa5b1", label="same search on 20 sign-flip null paths")
    ax.axvline(r["null_p95"], color="#555", ls="--", label="null 95th percentile")
    ax.axvline(r["real_ir"], color="#d9480f" if not r["passes"] else "#2b8a3e", lw=2.5, label=f"real path OOS IR {r['real_ir']:+.2f}")
    ax.set_title(lab, fontsize=10); ax.set_xlabel("walk-forward out-of-sample IR (top-20 bag)"); ax.legend(fontsize=7)
fig.suptitle(f"{res[next(iter(res))]['n_configs']:,}-configuration clock search: noise vs a real edge", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(DOCS, "null_test.png"), dpi=130); plt.close(fig)
pd.DataFrame({k: dict(n_configs=v["n_configs"], real_oos_ir=round(v["real_ir"], 3), null_p95=round(v["null_p95"], 3), percentile=v["percentile"], passes=v["passes"]) for k, v in res.items()}).T.to_csv(os.path.join(DOCS, "null_test_summary.csv"))
