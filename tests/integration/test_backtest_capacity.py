"""Integration tests for Phase 4.2: ADV, hard-to-borrow, capacity, and apply_costs.

Uses a richer local loader (heterogeneous volume + shares, so market cap and ADV vary
across names) than the shared constant-volume fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import qer.data.loader as loader_mod
from qer.backtest import Backtest, CostModel, adv, apply_costs, capacity_report, htb_mask
from qer.data.loader import DataLoader
from qer.factors.graph.base import PanelFactor


@pytest.fixture
def rich_loader(tmp_path, monkeypatch):
    raw, prices, wide, proc = (
        tmp_path / "raw", tmp_path / "raw" / "prices", tmp_path / "wide", tmp_path / "processed"
    )
    for d in (prices, wide, proc):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2019-01-01", periods=300)
    tickers = [f"T{i:02d}" for i in range(30)]
    for i, t in enumerate(tickers):
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(dates))))
        vol = np.full(len(dates), (i + 1) * 1e5)          # T00 least liquid ... T29 most
        frame = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close,
             "adj close": close, "volume": vol}, index=dates)
        frame.index.name = "date"
        frame.to_parquet(prices / f"{t}.parquet")
    pd.DataFrame(
        [{"ticker": t, "start_date": pd.Timestamp("2018-01-01"),
          "end_date": pd.Timestamp("2099-12-31")} for t in tickers]
    ).to_parquet(proc / "membership.parquet")
    shares = pd.DataFrame({t: [(i + 1) * 1e8] for i, t in enumerate(tickers)},
                          index=[dates[0]])                            # T00 smallest cap
    shares.to_parquet(raw / "shares.parquet")
    for name, val in [("RAW_DIR", raw), ("PRICES_DIR", prices), ("WIDE_DIR", wide),
                      ("MEMBERSHIP_FILE", proc / "membership.parquet"),
                      ("SHARES_FILE", raw / "shares.parquet")]:
        monkeypatch.setattr(loader_mod, name, val)
    return DataLoader()


def _backtest(loader, seed=0, scheme="signal"):
    rng = np.random.default_rng(seed)
    sig = pd.DataFrame(rng.normal(size=(len(loader.close.index), 30)),
                       index=loader.close.index, columns=loader.close.columns)
    return Backtest(scheme=scheme, exec_lag=1).run(loader, PanelFactor(sig, name="s", direction=1))


def test_adv_is_positive_and_full_shape(rich_loader):
    a = adv(rich_loader, window=21)
    assert a.shape == rich_loader.close.shape
    assert (a.dropna() > 0).to_numpy().all()


def test_htb_mask_flags_small_illiquid_names(rich_loader):
    assert not rich_loader.market_cap.empty
    last = htb_mask(rich_loader, quantile=0.2).iloc[-1]
    assert bool(last["T00"]) is True        # smallest cap + least liquid -> hard to borrow
    assert bool(last["T29"]) is False       # largest cap + most liquid -> borrowable


def test_capacity_report_ranks_low_adv_names_worst(rich_loader):
    res = _backtest(rich_loader)
    cap = capacity_report(res.weights, rich_loader, aum=5e8)
    assert not cap.empty
    assert cap["max_pos_pct_adv"].is_monotonic_decreasing        # worst-first
    assert (cap["max_pos_pct_adv"] > 0).all()
    # the least-liquid name uses a far larger fraction of its ADV than the most-liquid
    assert cap.loc["T00", "max_pos_pct_adv"] > cap.loc["T29", "max_pos_pct_adv"]


def test_apply_costs_nets_gross_and_charges_only_shorts(rich_loader):
    res = _backtest(rich_loader, scheme="signal")
    costed = apply_costs(res, rich_loader, CostModel())
    assert (costed.net_returns <= costed.gross_returns + 1e-12).to_numpy().all()   # net <= gross
    for series in (costed.linear, costed.impact, costed.borrow):
        assert (series >= 0).to_numpy().all()
    # borrow accrues only where there is a short book
    no_short = res.weights.clip(upper=0.0).abs().sum(axis=1) == 0
    assert (costed.borrow[no_short] == 0).to_numpy().all()
    # trade costs land only on rebalance (trade) days
    trade_days = res.trades[(res.trades != 0).any(axis=1)].index
    assert costed.linear[costed.linear > 0].index.isin(trade_days).all()
    assert set(costed.summary()) >= {"gross_sharpe", "net_sharpe", "ann_cost"}
