"""Shared pytest fixtures.

``synthetic_loader`` builds a small fixed-seed price/membership dataset on disk,
repoints the loader's path constants at it, and returns a ready ``DataLoader``.
Used by the Subphase 3.2 graph-engine tests (and available to any test that
needs a self-contained loader).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import qer.data.loader as loader_mod
from qer.data.loader import DataLoader

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
def synthetic_loader(tmp_path, monkeypatch):
    raw, prices, wide, processed = (
        tmp_path / "raw", tmp_path / "raw" / "prices", tmp_path / "wide", tmp_path / "processed"
    )
    for d in (prices, wide, processed):
        d.mkdir(parents=True, exist_ok=True)
    membership_file = processed / "membership.parquet"

    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2019-01-01", periods=400)
    tickers = [f"T{i:02d}" for i in range(30)]
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
