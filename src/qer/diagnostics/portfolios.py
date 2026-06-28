"""Decile/quintile long-short portfolios, turnover, and the IC decay curve.

Works off an *oriented* factor panel (high score = expected high return) and a
forward-return panel. The long-short return on a date is the mean forward return
of the top bucket minus the bottom bucket. Top-bucket turnover (long-leg name
churn) is reported on its own; the cost charged against the net long-short return
counts both legs (long top + short bottom). The decay curve is mean IC as the
forward horizon lengthens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.diagnostics.factor_ic import compute_factor_ic, forward_return_panel
from qer.factors.base import compute_factor_panel


def _bucket_masks(oriented_panel: pd.DataFrame, n_buckets: int):
    ranks = oriented_panel.rank(axis=1, pct=True, method="first")
    top = ranks > (1 - 1.0 / n_buckets)
    bottom = ranks <= (1.0 / n_buckets)
    return top, bottom


def _mask_turnover(mask: pd.DataFrame) -> pd.Series:
    """One-way turnover of a boolean membership mask between consecutive dates.

    Fraction of the bucket replaced: (entries + exits) / (2 * bucket size). The
    first date (no predecessor) is dropped rather than counted as 100% new.
    """
    prev = mask.shift(1)
    mask = mask.iloc[1:]
    prev = prev.iloc[1:].astype(bool)
    changed = (mask & ~prev) | (~mask & prev)
    denom = 2 * mask.sum(axis=1).replace(0, np.nan)
    return (changed.sum(axis=1) / denom).dropna()


def long_short_returns(
    oriented_panel: pd.DataFrame, fwd: pd.DataFrame, n_buckets: int = 10
) -> pd.Series:
    """Top-minus-bottom mean forward return per date (gross)."""
    fwd = fwd.reindex(index=oriented_panel.index, columns=oriented_panel.columns)
    top, bottom = _bucket_masks(oriented_panel, n_buckets)
    ls = fwd.where(top).mean(axis=1) - fwd.where(bottom).mean(axis=1)
    ls.name = "long_short"
    return ls.dropna()


def top_bucket_turnover(oriented_panel: pd.DataFrame, n_buckets: int = 10) -> pd.Series:
    """Fraction of top-bucket (long-leg) names that change between dates.

    This is the *long leg only* - kept for turnover reporting. For the cost
    charged against a long-short return, use :func:`long_short_turnover`, which
    also counts the short leg.
    """
    top, _ = _bucket_masks(oriented_panel, n_buckets)
    return _mask_turnover(top)


def long_short_turnover(oriented_panel: pd.DataFrame, n_buckets: int = 10) -> pd.Series:
    """Combined one-way turnover of BOTH legs - long top plus short bottom.

    A long-short book trades both legs at each rebalance (buy/sell the long
    leg, cover/short the short leg), so the cost-relevant turnover is the sum.
    Charging only the long leg (the previous behaviour) undercounted total
    trading cost by roughly half for a symmetric factor.
    """
    top, bottom = _bucket_masks(oriented_panel, n_buckets)
    return _mask_turnover(top).add(_mask_turnover(bottom), fill_value=0.0)


def net_long_short(
    oriented_panel: pd.DataFrame,
    fwd: pd.DataFrame,
    n_buckets: int = 10,
    cost_per_unit_turnover: float = 0.001,
) -> pd.Series:
    """Long-short return net of an estimated linear turnover cost on both legs.

    Cost model is intentionally simple (linear in turnover) and covers trading
    only; short-borrow and market-impact costs are deferred to the Phase 4
    backtester.
    """
    gross = long_short_returns(oriented_panel, fwd, n_buckets)
    turn = long_short_turnover(oriented_panel, n_buckets).reindex(gross.index).fillna(0.0)
    return gross - cost_per_unit_turnover * turn


def ic_decay(loader, factor, horizons=(1, 5, 10, 21, 42, 63), dates=None) -> pd.Series:
    """Mean IC as a function of forward horizon - the signal-decay curve."""
    ic = compute_factor_ic(loader, factor, horizons=tuple(horizons), dates=dates)
    return pd.Series({h: ic[h].mean() for h in horizons}, name="mean_ic")


def factor_long_short(loader, factor, n_buckets=10, horizon=21, dates=None, lag=1) -> pd.Series:
    """Convenience: oriented panel -> long-short return series for a Factor.

    ``lag`` is the implementation lag in trading days between observing the
    signal and trading on it. The signal at date T may need the close of T to be
    computed, so it is not tradeable at that same close in that case; with ``lag=1`` the
    bucket formed from the signal at T is held over the forward window starting
    at T+1 (trade at the next close). ``lag=0`` reproduces the old, optimistic
    same-close convention - useful only as a frictionless upper bound.
    """
    oriented = compute_factor_panel(loader, factor, dates=None, oriented=True)
    if lag:
        oriented = oriented.shift(lag)  # trade on the prior signal, not today's
    if dates is not None:
        oriented = oriented.reindex(pd.DatetimeIndex(dates))
    fwd = forward_return_panel(loader, horizon)
    return long_short_returns(oriented, fwd, n_buckets)
