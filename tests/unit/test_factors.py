"""Unit tests for the Phase 2 factor library and panel-based IC harness.

Pure and network-free. Pure-math factor tests build small DataFrames directly;
loader-dependent tests (size, value, quality, IC) stand up a tiny synthetic data
layout under tmp_path and redirect the loader's path globals at it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import qer.data.fundamentals as fund_mod
import qer.data.loader as loader_mod
import qer.factors as F
from qer.data.loader import DataLoader
from qer.diagnostics.factor_ic import (
    compute_factor_ic,
    newey_west_tstat,
    summarize_ic,
)
from qer.factors.base import Factor, compute_factor_panel
from qer.factors.liquidity import amihud_panel
from qer.factors.momentum import momentum_12_1, momentum_panel
from qer.factors.reversal import reversal_panel
from qer.factors.volatility import volatility_panel

# ---------------------------------------------------------------------------
# Pure factor-math tests (no loader)
# ---------------------------------------------------------------------------


@pytest.fixture
def prices():
    idx = pd.bdate_range("2018-01-01", periods=600)
    rng = np.random.default_rng(1)
    data = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, (600, 6)), axis=0))
    return pd.DataFrame(data, index=idx, columns=[f"T{i}" for i in range(6)])


def test_registry_conformance():
    assert F.FACTORS, "no factors registered"
    for fac in F.all_factors():
        assert isinstance(fac, Factor)
        assert isinstance(fac.name, str) and fac.name
        assert fac.direction in (-1, 1), f"{fac.name} has bad direction {fac.direction}"


def test_momentum_panel_matches_per_date(prices):
    panel = momentum_panel(prices, 252, 21)
    for t in prices.index[300::50]:
        per_date = momentum_12_1(prices, t)  # raw levels, log_prices=False
        pd.testing.assert_series_equal(panel.loc[t], per_date, check_names=False)


def test_reversal_panel_formula(prices):
    panel = reversal_panel(prices, window=21)
    logp = np.log(prices)
    expected = logp - logp.shift(21)
    pd.testing.assert_frame_equal(panel, expected)


def test_volatility_panel_is_rolling_std():
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(2)
    rets = pd.DataFrame(rng.normal(0, 0.01, (200, 3)), index=idx, columns=list("ABC"))
    panel = volatility_panel(rets, window=60)
    expected = rets.rolling(60, min_periods=40).std(ddof=1)
    pd.testing.assert_frame_equal(panel, expected)


def test_amihud_formula_small():
    idx = pd.bdate_range("2021-01-01", periods=40)
    rets = pd.DataFrame({"A": np.r_[np.nan, np.full(39, 0.01)]}, index=idx)
    dv = pd.DataFrame({"A": np.full(40, 1e6)}, index=idx)
    panel = amihud_panel(rets, dv, window=21)
    # |0.01| / 1e6 * 1e6 = 0.01 once the window fills
    assert np.isclose(panel["A"].dropna().iloc[-1], 0.01)


@pytest.mark.parametrize(
    "fn,kw",
    [(momentum_panel, dict(lookback_days=252, skip_days=21)), (reversal_panel, dict(window=21))],
)
def test_no_lookahead(prices, fn, kw):
    """Mutating rows strictly after t must not change the panel value at t."""
    full = fn(prices, **kw)
    t = prices.index[400]
    mutated = prices.copy()
    mutated.loc[mutated.index > t] = mutated.loc[mutated.index > t] * 3.0
    after = fn(mutated, **kw)
    pd.testing.assert_series_equal(full.loc[t], after.loc[t], check_names=False)


def test_newey_west_reduces_tstat_on_autocorrelated_series():
    rng = np.random.default_rng(0)
    n = 2000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 0.05)
    x = x + 0.02
    naive = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    nw = newey_west_tstat(x, lags=20)
    assert np.isclose(newey_west_tstat(x, lags=0), naive, rtol=1e-3)
    assert abs(nw) < abs(naive)  # positive autocorrelation must shrink the t-stat


# ---------------------------------------------------------------------------
# Loader-dependent tests
# ---------------------------------------------------------------------------


@pytest.fixture
def qer_loader(tmp_path, monkeypatch):
    """Synthetic price/volume/SPY/shares/fundamentals layout + redirected loader."""
    raw = tmp_path / "raw"
    prices = raw / "prices"
    wide = tmp_path / "wide"
    processed = tmp_path / "processed"
    for d in (prices, wide, processed):
        d.mkdir(parents=True, exist_ok=True)

    membership_file = processed / "membership.parquet"
    spy_file = raw / "SPY.parquet"
    shares_file = raw / "shares.parquet"
    fundamentals_file = processed / "fundamentals.parquet"

    monkeypatch.setattr(loader_mod, "RAW_DIR", raw)
    monkeypatch.setattr(loader_mod, "PRICES_DIR", prices)
    monkeypatch.setattr(loader_mod, "WIDE_DIR", wide)
    monkeypatch.setattr(loader_mod, "MEMBERSHIP_FILE", membership_file)
    monkeypatch.setattr(loader_mod, "SPY_FILE", spy_file)
    monkeypatch.setattr(loader_mod, "SHARES_FILE", shares_file)
    monkeypatch.setattr(fund_mod, "FUNDAMENTALS_FILE", fundamentals_file)

    idx = pd.bdate_range("2021-01-04", periods=420)
    rng = np.random.default_rng(11)
    tickers = [f"T{i:02d}" for i in range(40)]
    mret = rng.normal(0.0003, 0.01, len(idx))
    for k, t in enumerate(tickers):
        beta = 0.6 + 1.0 * k / 40
        r = beta * mret + rng.normal(0.0002 * (k - 20), 0.012, len(idx))
        px = 100 * np.exp(np.cumsum(r))
        vol = rng.lognormal(15, 0.4, len(idx))
        pd.DataFrame(
            {"Open": px, "High": px, "Low": px, "Close": px, "Adj Close": px, "Volume": vol},
            index=idx,
        ).to_parquet(prices / f"{t}.parquet")

    pd.DataFrame(
        {"ticker": tickers, "start_date": idx[0], "end_date": pd.Timestamp("2099-12-31")}
    ).to_parquet(membership_file)

    spy_px = 100 * np.exp(np.cumsum(mret))
    pd.DataFrame({"Adj Close": spy_px, "Close": spy_px}, index=idx).to_parquet(spy_file)
    pd.DataFrame(rng.uniform(1e8, 1e9, (1, 40)), index=[idx[0]], columns=tickers).to_parquet(
        shares_file
    )

    rows = []
    for t in tickers:
        for fld in ("book_equity", "gross_profit", "total_assets"):
            rows.append(
                {
                    "ticker": t,
                    "available_date": idx[0],
                    "field": fld,
                    "value": rng.uniform(1e9, 5e10),
                }
            )
    pd.DataFrame(rows).to_parquet(fundamentals_file)

    return SimpleNamespace(loader=DataLoader(), idx=idx, tickers=tickers)


def test_loader_phase2_views(qer_loader):
    loader = qer_loader.loader
    assert loader.dollar_volume.shape == loader.close.shape
    assert loader.market_return.notna().sum() > 300
    assert loader.market_cap.shape[1] == len(qer_loader.tickers)


def test_momentum_positive_ic_on_drift(qer_loader):
    """On monotone-drift data, oriented momentum IC must be strongly positive."""
    loader = qer_loader.loader
    dates = loader.close.index[300::15]
    ic = compute_factor_ic(
        loader, F.get_factor("momentum_12_1"), horizons=(21,), dates=dates, min_cross_section=20
    )
    assert ic[21].mean() > 0.2


def test_all_factors_produce_finite_ic(qer_loader):
    loader = qer_loader.loader
    dates = loader.close.index[300::20]
    for fac in F.all_factors():
        ic = compute_factor_ic(loader, fac, horizons=(21,), dates=dates, min_cross_section=15)
        assert len(ic[21]) > 0, f"{fac.name} produced no IC"
        assert ic[21].abs().max() <= 1.0


def test_compute_factor_ic_multi_horizon(qer_loader):
    loader = qer_loader.loader
    dates = loader.close.index[300::20]
    ic = compute_factor_ic(
        loader, F.get_factor("reversal_1m"), horizons=(1, 5, 21), dates=dates, min_cross_section=15
    )
    assert set(ic) == {1, 5, 21}


def test_compute_factor_panel_oriented_flips_sign(qer_loader):
    loader = qer_loader.loader
    fac = F.get_factor("reversal_1m")  # direction -1
    dates = loader.close.index[300::40]
    raw = compute_factor_panel(loader, fac, dates=dates, oriented=False)
    ori = compute_factor_panel(loader, fac, dates=dates, oriented=True)
    pd.testing.assert_frame_equal(ori, raw * -1)


def test_summarize_ic_adds_newey_west_key(qer_loader):
    loader = qer_loader.loader
    dates = loader.close.index[300::10]
    ic = compute_factor_ic(
        loader, F.get_factor("momentum_12_1"), horizons=(21,), dates=dates, min_cross_section=15
    )[21]
    base = summarize_ic(ic)
    assert "t_stat_nw" not in base
    with_nw = summarize_ic(ic, newey_west_lags=20)
    assert "t_stat_nw" in with_nw


def test_dollar_volume_and_market_cap_use_raw_not_adjusted_close(tmp_path, monkeypatch):
    """Dollar volume and market cap must use the RAW close, not adj close.

    Under a split the two diverge: adjusted close back-adjusts pre-split prices,
    so adj_close * volume understates true dollar volume by the split factor.
    Regression guard for a bug that previously shipped (adj close was used).
    """
    prices = tmp_path / "raw" / "prices"
    prices.mkdir(parents=True)
    processed = tmp_path / "processed"
    processed.mkdir()
    membership = processed / "m.parquet"
    shares_file = tmp_path / "raw" / "shares.parquet"

    monkeypatch.setattr(loader_mod, "PRICES_DIR", prices)
    monkeypatch.setattr(loader_mod, "WIDE_DIR", tmp_path / "wide")
    monkeypatch.setattr(loader_mod, "MEMBERSHIP_FILE", membership)
    monkeypatch.setattr(loader_mod, "SHARES_FILE", shares_file)

    idx = pd.bdate_range("2020-01-01", periods=10)
    raw_close = np.array([100, 100, 100, 100, 100, 50, 50, 50, 50, 50], float)  # 2:1 split
    adj_close = np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50], float)
    volume = np.full(10, 1_000_000.0)
    pd.DataFrame(
        {
            "Open": raw_close,
            "High": raw_close,
            "Low": raw_close,
            "Close": raw_close,
            "Adj Close": adj_close,
            "Volume": volume,
        },
        index=idx,
    ).to_parquet(prices / "AAA.parquet")
    pd.DataFrame(
        {"ticker": ["AAA"], "start_date": [idx[0]], "end_date": [pd.Timestamp("2099-12-31")]}
    ).to_parquet(membership)
    pd.DataFrame({"AAA": [2_000_000.0]}, index=[idx[0]]).to_parquet(shares_file)

    loader = DataLoader()
    dv = loader.dollar_volume["AAA"].values
    assert np.allclose(dv, raw_close * volume)  # raw close * volume
    assert not np.allclose(dv, adj_close * volume)  # NOT adjusted close
    # market cap likewise uses raw close (price * shares outstanding)
    mc = loader.market_cap["AAA"].values
    assert np.allclose(mc, raw_close * 2_000_000.0)


def test_dollar_volume_and_market_cap_use_raw_close(tmp_path, monkeypatch):
    """dollar_volume/market_cap must use RAW close, not the adjusted series.

    Regression test for the split/dividend mismatch: pairing as-reported volume
    (or share counts) with adjusted close scales each name by its own cumulative
    adjustment factor. Here Adj Close = 0.5 * Close, so the adjusted-close bug
    would halve both quantities.
    """
    raw = tmp_path / "raw"
    prices = raw / "prices"
    processed = tmp_path / "processed"
    for d in (prices, processed, tmp_path / "wide"):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(loader_mod, "RAW_DIR", raw)
    monkeypatch.setattr(loader_mod, "PRICES_DIR", prices)
    monkeypatch.setattr(loader_mod, "WIDE_DIR", tmp_path / "wide")
    monkeypatch.setattr(loader_mod, "MEMBERSHIP_FILE", processed / "m.parquet")
    monkeypatch.setattr(loader_mod, "SHARES_FILE", raw / "shares.parquet")

    idx = pd.bdate_range("2020-01-01", periods=10)
    raw_close = pd.Series(100.0, index=idx)
    volume = pd.Series(1_000.0, index=idx)
    for t in ["AAA", "BBB"]:
        pd.DataFrame(
            {
                "Open": raw_close,
                "High": raw_close,
                "Low": raw_close,
                "Close": raw_close,  # RAW close = 100
                "Adj Close": raw_close * 0.5,  # adjusted = 50 (a split/div away)
                "Volume": volume,
            },
            index=idx,
        ).to_parquet(prices / f"{t}.parquet")
    pd.DataFrame(
        {"ticker": ["AAA", "BBB"], "start_date": idx[0], "end_date": pd.Timestamp("2099-12-31")}
    ).to_parquet(processed / "m.parquet")
    pd.DataFrame({"AAA": [2.0], "BBB": [2.0]}, index=[idx[0]]).to_parquet(raw / "shares.parquet")

    loader = DataLoader()
    # raw close (100) * volume (1000) = 100_000, NOT adjusted (50*1000 = 50_000)
    assert np.isclose(loader.dollar_volume.loc[idx[-1], "AAA"], 100_000.0)
    # raw close (100) * shares (2) = 200, NOT adjusted (50*2 = 100)
    assert np.isclose(loader.market_cap.loc[idx[-1], "AAA"], 200.0)


def test_market_return_raises_without_adj_close(tmp_path, monkeypatch):
    """market_return must refuse a SPY file lacking adj close rather than
    silently using raw close (which would drop dividends)."""
    spy_file = tmp_path / "SPY.parquet"
    idx = pd.bdate_range("2020-01-01", periods=10)
    # only a raw Close column - no Adj Close
    pd.DataFrame({"Close": np.linspace(300, 310, 10)}, index=idx).to_parquet(spy_file)
    monkeypatch.setattr(loader_mod, "SPY_FILE", spy_file)
    loader = DataLoader()
    with pytest.raises(KeyError, match="adj close"):
        _ = loader.market_return
