"""Integration test for the Subphase 3.1 harness-reuse spike.

The gate for Subphase 3.1: a classical factor pushed through the graph
evaluation path (its panel wrapped in a ``PanelFactor``) must reproduce its
native IC and decile-Sharpe to numerical tolerance. We build a small fixed-seed
synthetic dataset on disk, repoint the loader's path constants at it, and assert
the spike passes -- proving "a graph feature is a Factor" before any graph exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import qer.data.loader as loader_mod
import qer.factors.momentum  # noqa: F401 - registers momentum_12_1
from qer.data.loader import DataLoader
from qer.graphs.spike import assert_harness_reuse, harness_reuse_spike

SENTINEL_END = pd.Timestamp("2099-12-31")


def _write_ticker(prices_dir: Path, ticker: str, dates: pd.DatetimeIndex, close: np.ndarray):
    frame = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "adj close": close, "volume": np.full(len(dates), 1_000_000.0)},
        index=dates,
    )
    frame.index.name = "date"
    frame.to_parquet(prices_dir / f"{ticker}.parquet")


@pytest.fixture
def random_loader(tmp_path, monkeypatch):
    raw, prices, wide, processed = (
        tmp_path / "raw", tmp_path / "raw" / "prices", tmp_path / "wide", tmp_path / "processed"
    )
    for d in (prices, wide, processed):
        d.mkdir(parents=True, exist_ok=True)
    membership_file = processed / "membership.parquet"

    rng = np.random.default_rng(7)
    n_tickers = 30
    dates = pd.bdate_range("2019-01-01", periods=360)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    for t in tickers:
        steps = rng.normal(0.0004, 0.012, len(dates))
        _write_ticker(prices, t, dates, 100.0 * np.exp(np.cumsum(steps)))

    pd.DataFrame(
        [{"ticker": t, "start_date": pd.Timestamp("2018-01-01"), "end_date": SENTINEL_END}
         for t in tickers]
    ).to_parquet(membership_file)

    monkeypatch.setattr(loader_mod, "RAW_DIR", raw)
    monkeypatch.setattr(loader_mod, "PRICES_DIR", prices)
    monkeypatch.setattr(loader_mod, "WIDE_DIR", wide)
    monkeypatch.setattr(loader_mod, "MEMBERSHIP_FILE", membership_file)
    return DataLoader()


def test_spike_reproduces_native_factor(random_loader):
    res = harness_reuse_spike(random_loader, factor_name="momentum_12_1")
    # non-vacuous: the harness actually produced an IC series and a tradeable LS
    assert res["n_ic_dates"] > 0
    assert np.isfinite(res["ls_sharpe_native"])
    # equivalence: wrapping the panel as a Factor changes nothing
    assert res["ic_max_abs_diff"] <= 1e-9
    assert res["ls_sharpe_diff"] <= 1e-9


def test_assert_harness_reuse_passes(random_loader):
    out = assert_harness_reuse(random_loader)  # raises on failure
    assert out["ls_sharpe_diff"] <= 1e-9
