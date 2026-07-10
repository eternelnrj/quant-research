"""Unit tests for the Phase 4.1 schedule and weight helpers (no loader needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qer.backtest.schedule import rebalance_schedule, train_test_split, walk_forward_folds
from qer.backtest.weights import signal_to_weights

TICKERS = [f"T{i:02d}" for i in range(30)]


# ---- weights ---------------------------------------------------------------

def test_weights_dollar_neutral_and_unit_gross():
    rng = np.random.default_rng(0)
    row = pd.Series(rng.normal(size=30), index=TICKERS)
    for scheme in ("equal", "signal", "rank"):
        w = signal_to_weights(row, scheme=scheme)
        assert abs(float(w.sum())) < 1e-10                  # dollar-neutral
        assert abs(float(w.abs().sum()) - 1.0) < 1e-10      # unit gross


def test_equal_scheme_longs_top_shorts_bottom():
    row = pd.Series(range(10), index=[f"N{i}" for i in range(10)], dtype=float)
    w = signal_to_weights(row, scheme="equal", n_buckets=5)   # n_side = 2
    assert set(w[w > 0].index) == {"N8", "N9"}
    assert set(w[w < 0].index) == {"N0", "N1"}


def test_constant_signal_gives_zero_weights():
    row = pd.Series(5.0, index=TICKERS)
    assert (signal_to_weights(row, scheme="signal") == 0).all()
    assert (signal_to_weights(row, scheme="rank") == 0).all()


def test_too_few_names_returns_empty():
    assert signal_to_weights(pd.Series([1.0], index=["A"])).empty
    assert signal_to_weights(pd.Series(dtype=float)).empty


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        signal_to_weights(pd.Series([1.0, 2.0], index=["A", "B"]), scheme="bogus")


# ---- schedule --------------------------------------------------------------

def test_rebalance_schedule_on_calendar_and_freq_ordering():
    cal = pd.bdate_range("2020-01-01", periods=300)
    monthly = rebalance_schedule(cal, freq="M")
    assert monthly.isin(cal).all()                          # real trading days
    assert len(monthly) >= 12
    assert len(rebalance_schedule(cal, freq="W")) > len(monthly)   # weekly is denser


def test_train_test_split_disjoint_and_covers():
    cal = pd.bdate_range("2020-01-01", periods=300)
    is_, oos = train_test_split(cal, "2020-07-01")
    assert is_.intersection(oos).empty
    assert len(is_) + len(oos) == len(cal)
    assert is_.max() < pd.Timestamp("2020-07-01") <= oos.min()


def test_walk_forward_folds_rolling_and_disjoint():
    cal = pd.bdate_range("2020-01-01", periods=300)
    folds = walk_forward_folds(cal, train=100, test=50)
    assert len(folds) >= 4
    for tr, te in folds:
        assert len(tr) == 100 and len(te) == 50
        assert tr.intersection(te).empty                    # no leakage IS -> OOS
        assert tr.max() < te.min()


def test_walk_forward_rejects_nonpositive():
    cal = pd.bdate_range("2020-01-01", periods=50)
    with pytest.raises(ValueError):
        walk_forward_folds(cal, train=0, test=10)
