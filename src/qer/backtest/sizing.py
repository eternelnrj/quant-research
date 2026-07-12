"""Phase 4.3: position-sizing schemes.

Each sizer maps an *oriented* signal cross-section (high value => long) to signed,
dollar-neutral, unit-gross target weights. ``equal_weight`` and ``signal_weight``
delegate to the 4.1 primitive; ``risk_weight`` is new -- it selects the top/bottom
bucket by signal and weights each side by inverse volatility (risk-parity-lite), so
lower-vol names carry more weight.
"""

from __future__ import annotations

import pandas as pd

from qer.backtest.weights import signal_to_weights


def rolling_vol(loader, window: int = 63, kind: str = "simple") -> pd.DataFrame:
    """Trailing ``window``-day return volatility (date x ticker) for risk weighting."""
    return loader.get_returns(kind).rolling(window, min_periods=max(2, window // 4)).std()


def equal_weight(signal_row, n_buckets: int = 10) -> pd.Series:
    """Equal-weight the top and bottom ``1/n_buckets`` of the signal (dollar-neutral)."""
    return signal_to_weights(signal_row, scheme="equal", n_buckets=n_buckets)


def signal_weight(signal_row) -> pd.Series:
    """Weight proportional to the demeaned signal (dollar-neutral, unit-gross)."""
    return signal_to_weights(signal_row, scheme="signal")


def risk_weight(signal_row, vol, n_buckets: int = 10) -> pd.Series:
    """Top/bottom bucket by signal, inverse-volatility weighted within each side.

    Dollar-neutral (+0.5 long, -0.5 short) and unit-gross; within a side, weight is
    proportional to ``1/vol`` so a lower-vol name gets a larger position. Missing or
    non-positive vols fall back to the cross-sectional median.
    """
    s = pd.Series(signal_row).dropna().astype(float)
    if len(s) < 2:
        return pd.Series(dtype=float)
    n_side = max(1, round(len(s) / n_buckets))
    longs = s.nlargest(n_side).index
    shorts = s.nsmallest(n_side).index.difference(longs)
    if len(longs) == 0 or len(shorts) == 0:
        return pd.Series(0.0, index=s.index)

    inv = 1.0 / pd.Series(vol).reindex(s.index).astype(float).where(lambda v: v > 0)
    inv = inv.fillna(inv.median() if inv.notna().any() else 1.0)

    w = pd.Series(0.0, index=s.index)
    lv, sv = inv.reindex(longs), inv.reindex(shorts)
    w[longs] = 0.5 * lv / lv.sum()
    w[shorts] = -0.5 * sv / sv.sum()
    return w
