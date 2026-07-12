"""Integration tests for Phase 4.3: rolling vol/beta, the engine weigher hook, and
the holding-period sweep. Uses a local loader that includes SPY (for market beta).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import qer.data.loader as loader_mod
from qer.backtest import Backtest, holding_period_sweep, make_weigher, rolling_beta, rolling_vol
from qer.data.loader import DataLoader
from qer.factors.graph.base import PanelFactor


@pytest.fixture
def market_loader(tmp_path, monkeypatch):
    raw, prices, wide, proc = (
        tmp_path / "raw", tmp_path / "raw" / "prices", tmp_path / "wide", tmp_path / "processed"
    )
    for d in (prices, wide, proc):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2019-01-01", periods=350)
    tickers = [f"T{i:02d}" for i in range(30)]
    common = rng.normal(0.0, 0.008, len(dates))                  # market factor
    for t in tickers:
        beta = rng.uniform(0.3, 1.6)
        close = 100.0 * np.exp(np.cumsum(beta * common + rng.normal(0.0004, 0.01, len(dates))))
        frame = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close,
             "adj close": close, "volume": np.full(len(dates), 1e6)}, index=dates)
        frame.index.name = "date"
        frame.to_parquet(prices / f"{t}.parquet")
    pd.DataFrame(
        [{"ticker": t, "start_date": pd.Timestamp("2018-01-01"),
          "end_date": pd.Timestamp("2099-12-31")} for t in tickers]
    ).to_parquet(proc / "membership.parquet")
    spy_close = 100.0 * np.exp(np.cumsum(common))
    pd.DataFrame({"adj close": spy_close}, index=dates).to_parquet(raw / "SPY.parquet")
    for name, val in [("RAW_DIR", raw), ("PRICES_DIR", prices), ("WIDE_DIR", wide),
                      ("MEMBERSHIP_FILE", proc / "membership.parquet"),
                      ("SPY_FILE", raw / "SPY.parquet")]:
        monkeypatch.setattr(loader_mod, name, val)
    return DataLoader()


def _factor(loader, seed=0):
    rng = np.random.default_rng(seed)
    sig = pd.DataFrame(rng.normal(size=(len(loader.close.index), 30)),
                       index=loader.close.index, columns=loader.close.columns)
    return PanelFactor(sig, name="s", direction=1)


def test_rolling_vol_positive_after_warmup(market_loader):
    v = rolling_vol(market_loader, window=63)
    assert v.shape == market_loader.close.shape
    assert (v.iloc[100:].dropna() > 0).to_numpy().all()


def test_rolling_beta_recovers_positive_betas(market_loader):
    last = rolling_beta(market_loader, window=126).iloc[-1].dropna()
    assert len(last) > 0
    assert last.mean() > 0            # names load positively on the common market factor


def test_make_weigher_through_engine_is_dollar_neutral(market_loader):
    weigher = make_weigher("risk", n_buckets=10, max_position=0.08)
    res = Backtest(freq="M", weigher=weigher).run(market_loader, _factor(market_loader))
    assert np.isfinite(res.returns.to_numpy()).all()
    assert len(res.rebalance_dates) > 0
    wr = res.weights.loc[res.rebalance_dates]
    assert wr.sum(axis=1).abs().max() < 1e-9      # dollar-neutral (renormalise preserves it)


def test_make_weigher_beta_neutral_runs_causally(market_loader):
    weigher = make_weigher("signal", beta_neutralise=True, beta_window=126)
    res = Backtest(freq="M", weigher=weigher).run(market_loader, _factor(market_loader, 1))
    assert np.isfinite(res.returns.to_numpy()).all()
    assert len(res.rebalance_dates) > 0


def test_holding_period_sweep_all_valid(market_loader):
    sweep = holding_period_sweep(market_loader, _factor(market_loader),
                                 holding_periods=(1, 5, 21), scheme="signal")
    assert set(sweep) == {1, 5, 21}
    for r in sweep.values():
        assert np.isfinite(r.returns.to_numpy()).all()
        assert len(r.rebalance_dates) > 0
    assert len(sweep[1].rebalance_dates) > len(sweep[21].rebalance_dates)   # denser rebalancing


def test_make_weigher_caches_rolling_panel_once(market_loader, monkeypatch):
    # the rolling vol panel must be computed once per run, not once per rebalance
    import qer.backtest.constraints as constraints_mod
    calls = {"n": 0}
    orig = constraints_mod.rolling_vol

    def counting(loader, window=63, kind="simple"):
        calls["n"] += 1
        return orig(loader, window, kind)

    monkeypatch.setattr(constraints_mod, "rolling_vol", counting)
    res = Backtest(freq="M", weigher=make_weigher("risk", n_buckets=10)).run(
        market_loader, _factor(market_loader))
    assert len(res.rebalance_dates) > 1
    assert calls["n"] == 1          # cached, not recomputed each rebalance
