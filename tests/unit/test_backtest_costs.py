"""Unit tests for the Phase 4.2 cost primitives (pure functions, no loader)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qer.backtest.costs import (
    borrow_cost,
    exclude_htb_shorts,
    impact_cost,
    linear_cost,
    turnover,
)


def test_turnover_unchanged_zero_and_linear():
    w = {"A": 0.5, "B": -0.5}
    assert turnover(w, w) == 0.0
    # flipping +w -> -w (turnover 2|w|) is twice flipping +w -> 0 (|w|)
    assert turnover({"A": 0.3}, {"A": -0.3}) == pytest.approx(2 * turnover({"A": 0.3}, {"A": 0.0}))


def test_linear_cost_is_c_times_turnover():
    assert linear_cost(2.0, 8.0) == pytest.approx(2.0 * 8e-4)
    assert linear_cost(1.0, 8.0) == pytest.approx(0.5 * linear_cost(2.0, 8.0))


def test_impact_per_unit_scales_as_sqrt_participation():
    a = float(impact_cost(1e6, 1e8, 0.1))
    b = float(impact_cost(2e6, 1e8, 0.1))
    assert b / a == pytest.approx(np.sqrt(2.0))          # double notional -> sqrt(2)x per-unit


def test_total_impact_drag_is_convex():
    # total drag for a name = |dw| * impact_cost(|dw|*aum, adv, coef) ~ size^{3/2}
    aum, adv_v, coef = 1e9, 1e8, 0.1
    d1 = 0.01 * float(impact_cost(0.01 * aum, adv_v, coef))
    d2 = 0.02 * float(impact_cost(0.02 * aum, adv_v, coef))
    assert d2 / d1 == pytest.approx(2 ** 1.5)            # convex, not linear


def test_impact_zero_or_missing_adv_is_zero():
    assert float(impact_cost(1e6, 0.0, 0.1)) == 0.0     # untradeable -> no modelled impact


def test_borrow_only_on_shorts_and_linear_in_size():
    assert borrow_cost({"A": 0.5, "B": 0.5}, 75.0) == 0.0            # long-only pays nothing
    b1 = borrow_cost({"A": 0.5, "B": -0.5}, 75.0)
    assert b1 == pytest.approx(0.5 * 75e-4 / 252)                    # exact daily formula
    assert borrow_cost({"A": 0.5, "B": -1.0}, 75.0) == pytest.approx(2 * b1)   # 2x short -> 2x


def test_exclude_htb_shorts_removes_shorts_keeps_longs():
    idx = pd.bdate_range("2020-01-01", periods=2)
    w = pd.DataFrame({"A": [0.5, 0.5], "B": [-0.5, -0.5], "C": [0.3, 0.3]}, index=idx)
    htb = pd.DataFrame({"A": [False, False], "B": [True, True], "C": [True, True]}, index=idx)
    out = exclude_htb_shorts(w, htb)
    assert (out["B"] == 0.0).all()      # hard-to-borrow short -> removed
    assert (out["C"] == 0.3).all()      # hard-to-borrow long  -> kept
    assert (out["A"] == 0.5).all()      # not hard-to-borrow   -> kept


def test_exclude_htb_shorts_handles_misaligned_mask():
    # htb from a full-universe mask rarely aligns exactly with a weight path's columns/dates
    idx = pd.bdate_range("2020-01-01", periods=3)
    w = pd.DataFrame({"A": [0.5] * 3, "B": [-0.5] * 3, "C": [-0.3] * 3}, index=idx)
    htb = pd.DataFrame({"B": [True, True], "D": [True, True]}, index=idx[:2])   # missing C, date; extra D
    out = exclude_htb_shorts(w, htb)                      # must not raise
    assert (out["B"].iloc[:2] == 0.0).all()              # flagged short -> zeroed where known
    assert out["B"].iloc[2] == -0.5                       # unknown date -> treated as borrowable
    assert (out["C"] == -0.3).all()                       # absent from mask -> borrowable
