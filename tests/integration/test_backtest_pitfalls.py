"""Phase 4.5: the roadmap pitfalls locked in as a regression suite, plus the report test.

One test per pitfall in roadmap section 11.4, so each defence built in 4.1-4.4 has a
permanent guard, and a test that the report is deterministic, complete, and runs on the
synthetic loader with no network or optional extras (SPY/FF5 pointed away).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import qer.backtest.report as report_mod
import qer.data.loader as loader_mod
from qer.backtest import Backtest, capacity_report, exclude_htb_shorts
from qer.backtest.costs import borrow_cost, impact_cost
from qer.backtest.report import build_report
from qer.backtest.schedule import train_test_split
from qer.data.loader import DataLoader
from qer.factors.graph.base import PanelFactor


def _write_prices(prices_dir, dates, tickers, close_by_ticker, hl_mult=1.0, volume=2e6):
    for t in tickers:
        close = close_by_ticker[t]
        frame = pd.DataFrame(
            {"open": close, "high": close * hl_mult, "low": close / hl_mult,
             "close": close, "adj close": close, "volume": np.full(len(dates), volume)},
            index=dates)
        frame.index.name = "date"
        frame.to_parquet(prices_dir / f"{t}.parquet")


def _base_loader(root, monkeypatch, hl_mult=1.0, seed=0):
    prices, wide, proc, raw = root / "raw" / "prices", root / "wide", root / "processed", root / "raw"
    for d in (prices, wide, proc):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=350)
    tickers = [f"T{i:02d}" for i in range(30)]
    close_by_ticker = {t: 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(dates)))) for t in tickers}
    _write_prices(prices, dates, tickers, close_by_ticker, hl_mult=hl_mult)
    pd.DataFrame([{"ticker": t, "start_date": pd.Timestamp("2018-01-01"),
                   "end_date": pd.Timestamp("2099-12-31")} for t in tickers]).to_parquet(
        proc / "membership.parquet")
    for name, val in [("RAW_DIR", raw), ("PRICES_DIR", prices), ("WIDE_DIR", wide),
                      ("MEMBERSHIP_FILE", proc / "membership.parquet"),
                      ("SPY_FILE", raw / "no_spy.parquet")]:      # SPY absent -> benchmark skipped
        monkeypatch.setattr(loader_mod, name, val)
    monkeypatch.setattr(report_mod, "FF5_FILE", raw / "no_ff5.parquet")   # FF5 absent -> skipped
    return DataLoader(), dates, tickers, close_by_ticker


@pytest.fixture
def pit_loader(tmp_path, monkeypatch):
    loader, _dates, _tickers, _close = _base_loader(tmp_path, monkeypatch)
    return loader


def _factor(loader, seed=0):
    rng = np.random.default_rng(seed)
    sig = pd.DataFrame(rng.normal(size=(len(loader.close.index), len(loader.close.columns))),
                       index=loader.close.index, columns=loader.close.columns)
    return PanelFactor(sig, name="pit_factor", direction=1)


# --- pitfall 1: trading on the close at the close ---------------------------

def test_pitfall_trading_on_close(pit_loader):
    sig = pd.DataFrame(np.random.default_rng(0).normal(size=pit_loader.close.shape),
                       index=pit_loader.close.index, columns=pit_loader.close.columns)
    base = Backtest(exec_lag=1).run(pit_loader, PanelFactor(sig, name="a", direction=1)).returns
    t0 = pit_loader.close.index[200]
    sig2 = sig.copy()
    future = sig2.index > t0
    sig2.loc[future] = np.random.default_rng(1).normal(size=(int(future.sum()), sig2.shape[1]))
    perturbed = Backtest(exec_lag=1).run(pit_loader, PanelFactor(sig2, name="b", direction=1)).returns
    assert base.loc[:t0].equals(perturbed.loc[:t0])           # future cannot move the past

    clair = pit_loader.get_returns("simple").shift(-1)         # tomorrow's return as signal
    fc = PanelFactor(clair, name="c", direction=1)
    assert (Backtest(scheme="equal", exec_lag=0).run(pit_loader, fc).returns.mean()
            > Backtest(scheme="equal", exec_lag=1).run(pit_loader, fc).returns.mean())


# --- pitfall 2: linear costs in everything ----------------------------------

def test_pitfall_linear_costs_impact_convex_and_capacity(pit_loader):
    a, b = float(impact_cost(1e6, 1e8, 0.1)), float(impact_cost(2e6, 1e8, 0.1))
    assert b / a == pytest.approx(np.sqrt(2.0))               # sqrt(participation), not linear
    res = Backtest(scheme="signal").run(pit_loader, _factor(pit_loader))
    cap = capacity_report(res.weights, pit_loader, aum=5e8)
    assert "max_pos_pct_adv" in cap.columns and len(cap) > 0  # capacity is reported


# --- pitfall 3: free-money shorts -------------------------------------------

def test_pitfall_free_money_shorts():
    assert borrow_cost({"A": 0.5, "B": 0.5}, 75.0) == 0.0     # long-only pays nothing
    assert borrow_cost({"A": 0.5, "B": -0.5}, 75.0) > 0.0     # shorts accrue borrow
    idx = pd.bdate_range("2020-01-01", periods=1)
    w = pd.DataFrame({"A": [-0.5], "B": [-0.5]}, index=idx)
    htb = pd.DataFrame({"A": [True], "B": [False]}, index=idx)
    out = exclude_htb_shorts(w, htb)
    assert out["A"].iloc[0] == 0.0 and out["B"].iloc[0] == -0.5   # HTB removed from short book


# --- pitfall 4: optimising across the whole sample --------------------------

def test_pitfall_optimising_whole_sample(pit_loader, tmp_path):
    cal = pit_loader.close.index
    is_, oos = train_test_split(cal, cal[200])
    assert is_.intersection(oos).empty and len(is_) + len(oos) == len(cal)
    _, data = build_report(pit_loader, _factor(pit_loader), out_dir=tmp_path / "rep",
                           scheme="signal", oos_split=str(cal[200].date()))
    assert "OOS" in data.config["evaluated_on"]               # headline is OOS


# --- pitfall 5: stop-loss with look-ahead (no intraday triggers) ------------

def test_pitfall_no_intraday_dependence(tmp_path, monkeypatch):
    # two books with identical CLOSE but different high/low must give identical returns
    la, dates, tickers, close = _base_loader(tmp_path / "a", monkeypatch, hl_mult=1.0, seed=7)
    sig = pd.DataFrame(np.random.default_rng(3).normal(size=(len(dates), len(tickers))),
                       index=dates, columns=tickers)
    ra = Backtest(scheme="signal").run(la, PanelFactor(sig, name="s", direction=1)).returns

    prices_b = tmp_path / "b" / "raw" / "prices"
    prices_b.mkdir(parents=True, exist_ok=True)
    _write_prices(prices_b, dates, tickers, close, hl_mult=1.5)   # same close, different high/low
    monkeypatch.setattr(loader_mod, "PRICES_DIR", prices_b)
    rb = Backtest(scheme="signal").run(DataLoader(), PanelFactor(sig, name="s", direction=1)).returns
    assert np.allclose(ra.to_numpy(), rb.to_numpy())             # engine uses day-end close only


# --- report: deterministic, complete, no network/extras ---------------------

def test_report_is_deterministic_complete_and_offline(pit_loader, tmp_path):
    factor = _factor(pit_loader)
    path, data = build_report(pit_loader, factor, out_dir=tmp_path / "rep", fmt="html",
                              scheme="signal", oos_split="2020-01-01", n_trials=108)
    assert path.exists()
    assert data.benchmark is None and data.ff5 is None          # ran without SPY/FF5
    assert len(data.capacity) > 0 and data.deflated_sharpe is not None
    html = path.read_text()
    for section in ("Configuration", "Performance", "Sharpe vs assumed cost",
                    "Monthly returns", "Capacity", "Equity", "Rolling Sharpe"):
        assert section in html                                  # every required section present
    # the cost curve at the assumed spread equals the headline net Sharpe (same period + model)
    assert data.cost_curve.loc[data.config["spread_bps"], "net_sharpe"] == pytest.approx(
        data.performance_net["sharpe"])

    # determinism: an identical config yields a byte-identical metrics table
    t1 = data.metrics_table()
    t2 = build_report(pit_loader, factor, out_dir=tmp_path / "rep2", scheme="signal",
                      oos_split="2020-01-01", n_trials=108)[1].metrics_table()
    assert t1.equals(t2)
