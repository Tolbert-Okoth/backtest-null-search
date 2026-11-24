"""nullsearch: judge a strategy search against what the same search finds in noise."""
from .nullpaths import block_shuffle_days, sign_flip
from .search import ClockConfig, annual_ir, clock_matrix, hourly_grid, null_test, walk_forward_bag
from .stats import deflated_sharpe, expected_max_sharpe, noise_hurdle, pbo_cscv, probabilistic_sharpe

__all__ = ["sign_flip", "block_shuffle_days", "ClockConfig", "clock_matrix", "hourly_grid", "walk_forward_bag", "null_test", "annual_ir",
           "noise_hurdle", "expected_max_sharpe", "probabilistic_sharpe", "deflated_sharpe", "pbo_cscv"]
__version__ = "0.1.0"
