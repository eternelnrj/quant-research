"""Unit tests for Phase 4.4 metrics and risk attribution (pure functions, no loader)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qer.backtest.metrics import (
    cagr,
    calmar,
    conditional_drawdown,
    hit_rate,
    max_drawdown,
    monthly_return_heatmap,
    performance_summary,
    profit_factor,
    sharpe,
    sortino,
    total_return,
)
from qer.backtest.risk import benchmark_stats, ff5_exposures, realised_beta, rolling_sharpe

IDX = pd.bdate_range("2019-01-01", periods=504)


def _noisy(seed, mu=0.0006, sd=0.01, n=504):
    return pd.Series(np.random.default_rng(seed).normal(mu, sd, n), index=IDX[:n])


def test_total_return_compounds():
    r = pd.Series([0.01, -0.02, 0.03], index=IDX[:3])
    assert total_return(r) == pytest.approx(1.01 * 0.98 * 1.03 - 1)


def test_cagr_constant_drift():
    c = 0.0005
    assert cagr(pd.Series([c] * 252, index=IDX[:252])) == pytest.approx((1 + c) ** 252 - 1)


def test_max_drawdown_single_shock_and_monotonic():
    shock = pd.Series([0.0] * 5 + [-0.20] + [0.0] * 5, index=IDX[:11])
    assert max_drawdown(shock) == pytest.approx(0.20)
    assert max_drawdown(pd.Series([0.001] * 20, index=IDX[:20])) == pytest.approx(0.0, abs=1e-12)


def test_sharpe_matches_manual():
    r = _noisy(0)
    assert sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))


def test_sortino_at_least_sharpe():
    r = _noisy(1)
    assert sortino(r) >= sharpe(r)                     # target semideviation <= total vol


def test_calmar_is_cagr_over_maxdd():
    r = _noisy(2)
    assert calmar(r) == pytest.approx(cagr(r) / max_drawdown(r))


def test_monthly_returns_compound_to_total():
    r = _noisy(3)
    monthly = monthly_return_heatmap(r).stack().to_numpy()
    assert np.isclose(np.prod(1 + monthly) - 1, total_return(r))


def test_hit_rate_and_profit_factor():
    r = pd.Series([0.02, -0.01, 0.03, -0.04], index=IDX[:4])
    assert hit_rate(r) == 0.5
    assert profit_factor(r) == pytest.approx(0.05 / 0.05)


def test_conditional_drawdown_between_zero_and_maxdd():
    r = _noisy(4, mu=0.0003)
    cdar = conditional_drawdown(r, alpha=0.05)
    assert 0.0 <= cdar <= max_drawdown(r) + 1e-9       # tail mean never worse than the worst


def test_performance_summary_has_all_keys():
    s = performance_summary(_noisy(5, n=252), turnover=pd.Series([0.5, 0.6, 0.4]))
    assert {"total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown",
            "calmar", "hit_rate", "profit_factor", "avg_turnover"} <= set(s)


def test_realised_beta_recovers_slope():
    rng = np.random.default_rng(6)
    m = pd.Series(rng.normal(0, 0.008, 504), index=IDX)
    r = 1.3 * m + rng.normal(0, 0.003, 504)
    assert realised_beta(r, m) == pytest.approx(1.3, abs=0.1)


def test_rolling_sharpe_length_and_warmup():
    rs = rolling_sharpe(_noisy(7), window=252)
    assert len(rs) == 504
    assert rs.iloc[:251].isna().all()                  # no value before the window fills
    assert rs.iloc[251:].notna().any()


def test_benchmark_stats_compares_strategy_and_spy():
    bs = benchmark_stats(_noisy(8), market_return=pd.Series(
        np.random.default_rng(80).normal(0, 0.008, 504), index=IDX))
    assert "strategy" in bs.index and "SPY_buy_hold" in bs.index
    assert {"cagr", "sharpe", "max_drawdown"} <= set(bs.columns)


def test_benchmark_stats_aligns_benchmark_to_strategy_range():
    # a benchmark spanning a longer range must be measured over the strategy's dates only
    strat = _noisy(10, n=252)                                  # year 1 only
    spy_log = pd.Series(np.random.default_rng(11).normal(0.002, 0.008, 504), index=IDX)
    bs = benchmark_stats(strat, market_return=spy_log)
    aligned = cagr(np.expm1(spy_log.reindex(strat.index)))
    full = cagr(np.expm1(spy_log))
    assert bs.loc["SPY_buy_hold", "cagr"] == pytest.approx(aligned)   # strategy-aligned
    assert not np.isclose(bs.loc["SPY_buy_hold", "cagr"], full)       # not the full range


def test_ff5_exposures_runs_and_recovers_market_loading():
    rng = np.random.default_rng(9)
    cols = {name: rng.normal(0, 0.006, 504) for name in ("smb", "hml", "rmw", "cma")}
    cols["mkt_rf"] = rng.normal(0.0004, 0.01, 504)
    r = pd.Series(0.6 * cols["mkt_rf"] + rng.normal(0, 0.003, 504), index=IDX)
    out = ff5_exposures(r, pd.DataFrame(cols, index=IDX))
    assert "alpha" in out
    betas = out.get("betas", {})
    if isinstance(betas, dict) and "mkt_rf" in betas:
        assert betas["mkt_rf"] == pytest.approx(0.6, abs=0.15)
