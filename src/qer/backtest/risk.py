"""Phase 4.4: risk attribution -- realised market beta, FF5 exposures, rolling Sharpe,
and side-by-side benchmark comparison.

Reuses :func:`qer.diagnostics.exposures.ff5_exposures` (one Newey-West HAC covariance)
rather than reimplementing factor regressions, so the backtest and the factor scorecard
speak the same language.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.backtest.metrics import performance_summary
from qer.diagnostics.exposures import ff5_exposures as _ff5_exposures


def realised_beta(returns, market_return) -> float:
    """OLS beta of the strategy return on the market return, ``cov(r, m) / var(m)``."""
    df = pd.concat(
        [pd.Series(returns, dtype=float).rename("r"),
         pd.Series(market_return, dtype=float).rename("m")], axis=1
    ).dropna()
    if len(df) < 2:
        return float("nan")
    var = float(df["m"].var())
    return float(df["r"].cov(df["m"]) / var) if var > 0 else float("nan")


def ff5_exposures(returns, ff5, nw_lags: int = 21) -> dict:
    """Alpha and FF5 betas with HAC t-stats (thin wrapper over the diagnostics version)."""
    return _ff5_exposures(pd.Series(returns, dtype=float).dropna(), ff5, nw_lags=nw_lags)


def rolling_sharpe(returns, window: int = 252, periods_per_year: int = 252) -> pd.Series:
    """Trailing annualised Sharpe over a rolling ``window`` (no look-ahead)."""
    r = pd.Series(returns, dtype=float)
    return r.rolling(window).mean() / r.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)


def benchmark_stats(returns, market_return=None, extra=None, periods_per_year: int = 252) -> pd.DataFrame:
    """The full metric set for the strategy alongside benchmarks -- answers "beats what?".

    ``market_return`` (SPY log return) is converted to simple and treated as a buy-and-hold
    benchmark; ``extra`` is an optional ``{name: simple_return_series}`` mapping (e.g. an
    equal-weight classical-factor composite). Benchmarks are aligned to the strategy's date
    range so every row is measured over the *same* period. Returns a DataFrame indexed by name.
    """
    returns = pd.Series(returns, dtype=float)
    rows = {"strategy": performance_summary(returns, periods_per_year=periods_per_year)}
    if market_return is not None:
        spy = np.expm1(pd.Series(market_return, dtype=float).reindex(returns.index)).dropna()
        rows["SPY_buy_hold"] = performance_summary(spy, periods_per_year=periods_per_year)
    for name, series in (extra or {}).items():
        aligned = pd.Series(series, dtype=float).reindex(returns.index).dropna()
        rows[name] = performance_summary(aligned, periods_per_year=periods_per_year)
    return pd.DataFrame(rows).T
