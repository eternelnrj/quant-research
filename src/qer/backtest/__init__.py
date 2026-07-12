"""Phase 4: vectorised, walk-forward, transaction-cost-aware backtester.

Subphase 4.1 ships the causal core: a signal -> dollar-neutral target weights ->
T+1 execution -> IS/OOS-separable daily return series. Costs (4.2), constraints
and sizing (4.3), analytics (4.4), and the report/CLI (4.5) build on it.
"""

from __future__ import annotations

from qer.backtest.constraints import (
    apply_constraints,
    beta_neutral,
    cap_positions,
    dollar_neutral,
    make_weigher,
    neutralise_sectors,
    renormalise,
    rolling_beta,
)
from qer.backtest.cost_curve import sharpe_vs_cost
from qer.backtest.costs import (
    CostedResult,
    CostModel,
    adv,
    apply_costs,
    borrow_cost,
    capacity_report,
    exclude_htb_shorts,
    htb_mask,
    impact_cost,
    linear_cost,
    turnover,
)
from qer.backtest.engine import Backtest, BacktestResult, holding_period_sweep
from qer.backtest.metrics import (
    ann_vol,
    avg_turnover,
    cagr,
    calmar,
    conditional_drawdown,
    drawdown_series,
    equity_curve,
    hit_rate,
    max_drawdown,
    monthly_return_heatmap,
    performance_summary,
    profit_factor,
    sharpe,
    sortino,
    total_return,
)
from qer.backtest.report import ReportData, build_report, compute_report_data
from qer.backtest.risk import benchmark_stats, ff5_exposures, realised_beta, rolling_sharpe
from qer.backtest.schedule import rebalance_schedule, train_test_split, walk_forward_folds
from qer.backtest.sizing import equal_weight, risk_weight, rolling_vol, signal_weight
from qer.backtest.weights import signal_to_weights

__all__ = [
    "Backtest",
    "BacktestResult",
    "holding_period_sweep",
    "rebalance_schedule",
    "train_test_split",
    "walk_forward_folds",
    "signal_to_weights",
    "CostModel",
    "CostedResult",
    "apply_costs",
    "turnover",
    "linear_cost",
    "impact_cost",
    "borrow_cost",
    "adv",
    "htb_mask",
    "exclude_htb_shorts",
    "capacity_report",
    "equal_weight",
    "signal_weight",
    "risk_weight",
    "rolling_vol",
    "cap_positions",
    "dollar_neutral",
    "neutralise_sectors",
    "beta_neutral",
    "renormalise",
    "apply_constraints",
    "rolling_beta",
    "make_weigher",
    "total_return",
    "cagr",
    "ann_vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "conditional_drawdown",
    "hit_rate",
    "profit_factor",
    "avg_turnover",
    "equity_curve",
    "drawdown_series",
    "monthly_return_heatmap",
    "performance_summary",
    "realised_beta",
    "ff5_exposures",
    "rolling_sharpe",
    "benchmark_stats",
    "sharpe_vs_cost",
    "compute_report_data",
    "build_report",
    "ReportData",
]
