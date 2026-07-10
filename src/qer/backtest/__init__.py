"""Phase 4: vectorised, walk-forward, transaction-cost-aware backtester.

Subphase 4.1 ships the causal core: a signal -> dollar-neutral target weights ->
T+1 execution -> IS/OOS-separable daily return series. Costs (4.2), constraints
and sizing (4.3), analytics (4.4), and the report/CLI (4.5) build on it.
"""

from __future__ import annotations

from qer.backtest.engine import Backtest, BacktestResult
from qer.backtest.schedule import rebalance_schedule, train_test_split, walk_forward_folds
from qer.backtest.weights import signal_to_weights

__all__ = [
    "Backtest",
    "BacktestResult",
    "rebalance_schedule",
    "train_test_split",
    "walk_forward_folds",
    "signal_to_weights",
]
