"""Subphase 3.2: point-in-time trailing-return windows and the rebalance schedule.

The single look-ahead boundary of the whole graph layer lives here. A window
ends at ``as_of`` and looks strictly backward, over the point-in-time universe at
``as_of``; names without enough history in the window are DROPPED, never
zero-filled, so a fresh listing cannot contaminate a correlation matrix.
"""

from __future__ import annotations

import pandas as pd


def trailing_return_matrix(
    loader, as_of, window: int = 120, min_obs: int | None = None, kind: str = "log"
) -> pd.DataFrame:
    """``(<= window) x names`` trailing returns up to and including ``as_of``.

    Restricted to the point-in-time universe at ``as_of`` and to names with at
    least ``min_obs`` non-NaN observations in the window (default: the full
    ``window`` -- the contamination-safe choice). ``kind`` is passed to
    ``loader.get_returns``.
    """
    as_of = pd.Timestamp(as_of)
    if min_obs is None:
        min_obs = window
    rets = loader.get_returns(kind)
    sub = rets.loc[:as_of].iloc[-window:]  # <= as_of, last `window` rows
    universe = set(loader.get_universe(as_of))
    cols = [c for c in sub.columns if c in universe]
    sub = sub.loc[:, cols]
    counts = sub.notna().sum()
    keep = counts.index[counts >= min_obs]
    return sub.loc[:, keep]


def rebalance_dates(calendar, freq: str = "M", start=None, end=None) -> pd.DatetimeIndex:
    """Last trading day of each ``freq`` period drawn from the trading ``calendar``.

    ``freq`` is a pandas *period* alias ("M" month, "W" week, "Q" quarter). Using
    the actual trading calendar (not a synthetic month-end) guarantees every
    snapshot date is a real trading day.
    """
    cal = pd.DatetimeIndex(calendar).drop_duplicates().sort_values()
    if start is not None:
        cal = cal[cal >= pd.Timestamp(start)]
    if end is not None:
        cal = cal[cal <= pd.Timestamp(end)]
    if len(cal) == 0:
        return pd.DatetimeIndex([])
    last = pd.Series(cal, index=cal).groupby(cal.to_period(freq)).last()
    return pd.DatetimeIndex(last.to_numpy()).sort_values()
