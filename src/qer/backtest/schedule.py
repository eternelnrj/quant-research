"""Phase 4.1: rebalance schedule and in-sample / out-of-sample splitting.

Thin, dependency-light helpers. ``rebalance_schedule`` reuses the Phase-3
``graphs.windows.rebalance_dates`` so the backtest rebalances on the same real
trading days the graph engine snapshots on. The split helpers exist so callers
can tune on IS folds and report OOS; the 4.1 engine itself fits nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.graphs.windows import rebalance_dates


def rebalance_schedule(calendar, freq="M", start=None, end=None) -> pd.DatetimeIndex:
    """Rebalance dates on ``calendar``.

    ``freq`` is a pandas period alias -- ``"W"`` weekly, ``"M"`` monthly, ``"Q"``
    quarterly (reuses ``graphs.windows.rebalance_dates``) -- or an ``int`` meaning
    "every N trading days" (used by the holding-period sweep). Every rebalance is a
    real trading day.
    """
    if isinstance(freq, (int, np.integer)):
        if int(freq) < 1:
            raise ValueError("integer freq (every-N-days) must be >= 1")
        cal = pd.DatetimeIndex(calendar).drop_duplicates().sort_values()
        if start is not None:
            cal = cal[cal >= pd.Timestamp(start)]
        if end is not None:
            cal = cal[cal <= pd.Timestamp(end)]
        return cal[:: int(freq)]
    return rebalance_dates(calendar, freq=freq, start=start, end=end)


def train_test_split(calendar, split_date) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split ``calendar`` into in-sample (``< split_date``) and out-of-sample (``>=``)."""
    cal = pd.DatetimeIndex(calendar).drop_duplicates().sort_values()
    split = pd.Timestamp(split_date)
    return cal[cal < split], cal[cal >= split]


def walk_forward_folds(calendar, train: int, test: int, step: int | None = None):
    """Rolling ``(train_dates, test_dates)`` folds, sized in trading days.

    ``step`` defaults to ``test`` (non-overlapping test windows). The engine fits
    nothing on the test dates -- these folds are for the caller's tuning/reporting.
    """
    cal = pd.DatetimeIndex(calendar).drop_duplicates().sort_values()
    n = len(cal)
    step = test if step is None else step
    if train <= 0 or test <= 0 or step <= 0:
        raise ValueError("train, test, step must be positive")
    folds = []
    start = 0
    while start + train + test <= n:
        folds.append((cal[start:start + train], cal[start + train:start + train + test]))
        start += step
    return folds
