"""Phase 4.1: map a signal cross-section to dollar-neutral, unit-gross target weights."""

from __future__ import annotations

import pandas as pd

_SCHEMES = ("equal", "signal", "rank")


def signal_to_weights(signal_row, scheme: str = "equal", n_buckets: int = 10) -> pd.Series:
    """One cross-section of (oriented) signal values -> signed target weights.

    The returned weights are dollar-neutral (``sum(w) = 0``) and unit-gross
    (``sum(|w|) = 1``), so the book is a self-financing long-short. Schemes:

    * ``equal``  -- long the top ``1/n_buckets`` names, short the bottom
      ``1/n_buckets``, equal-weight within each side;
    * ``signal`` -- weight proportional to the demeaned signal;
    * ``rank``   -- weight proportional to the demeaned cross-sectional rank
      (robust to signal outliers).

    The signal must already be *oriented* (high value => long). NaNs are dropped;
    a cross-section with fewer than two valid names yields an empty Series, and a
    degenerate (constant) signal yields all-zero weights.
    """
    if scheme not in _SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {_SCHEMES}")
    s = pd.Series(signal_row).dropna().astype(float)
    if len(s) < 2:
        return pd.Series(dtype=float)

    if scheme == "equal":
        n_side = max(1, round(len(s) / n_buckets))
        longs = s.nlargest(n_side).index
        shorts = s.nsmallest(n_side).index.difference(longs)   # no name on both sides
        if len(longs) == 0 or len(shorts) == 0:
            return pd.Series(0.0, index=s.index)
        w = pd.Series(0.0, index=s.index)
        w[longs] = 0.5 / len(longs)
        w[shorts] = -0.5 / len(shorts)
        return w

    base = s if scheme == "signal" else s.rank()
    d = base - base.mean()                 # demean -> dollar neutral
    denom = float(d.abs().sum())
    if denom == 0.0:
        return pd.Series(0.0, index=s.index)
    return d / denom                       # normalise -> unit gross
