"""
12-1 month momentum factor.

Definition: cumulative return from T-252 trading days to T-21 trading days.
The most recent month (~21 trading days) is excluded to avoid short-term
reversal contamination - stocks that ripped in the last few weeks tend to
mean-revert and are noise to a momentum signal.

Conventions:
- Trading days, not calendar days (252 = ~12 months, 21 = ~1 month).
- Computed in log space for cleanly additive returns.
- As-of date T uses information through close of T. The signal is
  tradeable at open or close of T+1 (signals from close of T are NOT
  tradeable at the same close - see roadmap section 6.5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Imported lazily to avoid a circular import at module load.
from qer.factors.base import Factor, register  # noqa: E402           # NEW


def momentum(
    prices_df: pd.DataFrame,
    as_of_date,
    lookback_days: int,
    skip_days: int,
    log_prices: bool = False,
) -> pd.Series:
    """momentum factor.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Date x ticker price matrix. Index must be a sorted DatetimeIndex.
        Prices should already be split- and dividend-adjusted (e.g.
        yfinance Adj Close). Raw *price levels* or *log prices* are both
        accepted - the function converts internally based on ``log_prices``.
    as_of_date : pd.Timestamp-like
        The signal date T. Only information through close of T is used.
        If T is not a trading day in prices_df, the most recent prior
        trading day is used.
    lookback_days : int
        Total lookback window in trading days.
    skip_days : int
        Recent days to exclude from the window (skip-the-recent-month).
    log_prices : bool, default False
        Whether ``prices_df`` is ALREADY in log space.
        - False (default): ``prices_df`` holds raw price *levels*; the factor
          is computed as ``log(P_end) - log(P_start)``.
        - True: ``prices_df`` already holds log prices; the factor is the
          plain difference ``logP_end - logP_start``.
        Both branches return the same log return; the flag only says whether
        the log has already been taken. The default matches the common case
        of passing ``DataLoader.close`` (raw adjusted price levels).

    Returns
    -------
    pd.Series
        Factor value (log return T-lookback_days -> T-skip_days) for each ticker, indexed
        by ticker. Tickers without enough history return NaN, NOT zero -
        downstream code should handle NaN explicitly (winsorise + drop,
        or impute). NaN is much safer than a misleading zero.

    Notes
    -----
    The function strictly uses data through `as_of_date` (no look-ahead).
    To verify: the slice `prices_df.loc[:as_of_date]` is the only data
    touched.
    """
    as_of = pd.Timestamp(as_of_date)
    if as_of < prices_df.index[0]:
        raise ValueError(
            f"as_of_date {as_of.date()} is before the price data starts "
            f"({prices_df.index[0].date()})"
        )

    # Slice to everything up to and including as_of_date. This is the
    # ONLY data the function may use; downstream computations operate on
    # this slice only. No look-ahead by construction.
    history = prices_df.loc[:as_of]

    if len(history) < lookback_days + 1:
        # Not enough trading days for ANY ticker. Return all-NaN series.
        return pd.Series(
            np.nan,
            index=prices_df.columns,
            name=f"mom_{lookback_days}d_{skip_days}d_{as_of.date()}",
        )

    # End of momentum window: skip_days back from as_of (excludes recent month).
    # Start of momentum window: lookback_days back from as_of.
    # Using integer positional indexing avoids any calendar-arithmetic bugs
    # around weekends/holidays.
    end_row = history.iloc[-skip_days - 1]  # price on the "end" date
    start_row = history.iloc[-lookback_days - 1]  # price on the "start" date

    # Log return = log(P_end) - log(P_start). Equivalent to log(P_end / P_start).
    # NaN if either price is NaN (ticker not yet listed, or delisted then).
    with np.errstate(invalid="ignore", divide="ignore"):
        if log_prices:
            # Input is already log prices: the difference IS the log return.
            factor = end_row - start_row
        else:
            # Input is raw price levels: take logs first.
            factor = np.log(end_row) - np.log(start_row)

    factor.name = f"mom_{lookback_days}d_{skip_days}d_{as_of.date()}"
    return factor


def momentum_12_1(prices_df, as_of_date, log_prices: bool = False) -> pd.Series:
    """Canonical 12-1 month momentum (252-day lookback, 21-day skip).

    See :func:`momentum` for the ``log_prices`` flag. The default (False)
    treats ``prices_df`` as raw price levels, so passing ``DataLoader.close``
    directly yields a proper log-return momentum signal.
    """
    return momentum(prices_df, as_of_date, lookback_days=252, skip_days=21, log_prices=log_prices)


# ---------------------------------------------------------------------------
# Phase 2: vectorised panel + Factor interface                                                 # NEW
# ---------------------------------------------------------------------------


def momentum_panel(
    prices_df: pd.DataFrame,
    lookback_days: int,
    skip_days: int,
    log_prices: bool = False,
) -> pd.DataFrame:
    """Vectorised ``date x ticker`` momentum panel.

    Equivalent to calling :func:`momentum` at every date, but in one pass:
    at row t the value is ``logP[t-skip] - logP[t-lookback]``, which is exactly
    ``logP.shift(skip) - logP.shift(lookback)``. Look-ahead-safe (each row uses
    only earlier rows).
    """
    logp = prices_df if log_prices else np.log(prices_df)
    return logp.shift(skip_days) - logp.shift(lookback_days)


class Momentum12_1(Factor):
    """Canonical 12-1 momentum as a registered Factor (252d lookback, 21d skip)."""

    name = "momentum_12_1"
    direction = +1

    def compute_panel(self, loader) -> pd.DataFrame:
        return momentum_panel(loader.close, lookback_days=252, skip_days=21)


register(Momentum12_1())
