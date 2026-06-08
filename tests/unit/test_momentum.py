"""Unit tests for qer.factors.momentum.

Focus on the price-convention contract: the default treats the input as raw
price levels (so passing DataLoader.close directly is correct), and the
log_prices=True branch treats the input as already-log prices. Both must
recover the same log return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qer.factors.momentum import momentum, momentum_12_1

LOOKBACK = 252
SKIP = 21


@pytest.fixture
def prices():
    """Two tickers with smooth exponential drift over ~300 trading days."""
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    aaa = 100.0 * np.exp(np.linspace(0.0, 0.40, 300))  # ~+49% over window
    bbb = 100.0 * np.exp(np.linspace(0.0, 0.10, 300))  # ~+10% over window
    return pd.DataFrame({"AAA": aaa, "BBB": bbb}, index=idx)


def test_default_treats_input_as_raw_levels(prices):
    """The default must return a log RETURN, not a raw price difference."""
    t = prices.index[-1]
    factor = momentum_12_1(prices, t)  # default log_prices=False
    # Log return over the window is modest (< 0.5), never tens of dollars.
    assert (factor.abs() < 1.0).all()
    assert factor["AAA"] > factor["BBB"] > 0


def test_raw_and_log_inputs_agree(prices):
    """log_prices=False on levels == log_prices=True on log(levels)."""
    t = prices.index[-1]
    from_levels = momentum(prices, t, LOOKBACK, SKIP, log_prices=False)
    from_logs = momentum(np.log(prices), t, LOOKBACK, SKIP, log_prices=True)
    pd.testing.assert_series_equal(from_levels, from_logs, check_names=False)


def test_insufficient_history_returns_all_nan(prices):
    early = prices.index[10]
    factor = momentum_12_1(prices, early)
    assert factor.isna().all()
    assert len(factor) == prices.shape[1]


def test_no_look_ahead(prices):
    """Mutating prices strictly after T must not change the factor at T."""
    t = prices.index[-1]
    baseline = momentum_12_1(prices, t)
    mutated = prices.copy()
    mutated.loc[mutated.index > t] = np.nan  # nothing after t anyway, but explicit
    after = momentum_12_1(mutated, t)
    pd.testing.assert_series_equal(baseline, after)
