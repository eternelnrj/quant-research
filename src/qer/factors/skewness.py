"""Idiosyncratic skewness: skew of returns after removing market exposure.

For each ticker, regress its daily returns on the market return over a trailing
window, take the residuals, and measure their skewness. Direction is -1: names
with high positive idiosyncratic skew are "lottery-like" and tend to be
overpriced, earning lower subsequent returns.

Implementation note: beta is the trailing rolling beta (cov/var), residual is
``r - beta * r_market`` (the intercept is irrelevant - skew is location-invariant),
and skew is the trailing rolling skew of that residual. All rolling, hence
look-ahead-safe. It is an approximation (beta drifts within the skew window) but
the standard one for a daily panel.
"""

from __future__ import annotations

import pandas as pd

from qer.factors.base import Factor, register


def idio_skew_panel(
    returns: pd.DataFrame, market_return: pd.Series, window: int = 60
) -> pd.DataFrame:
    minp = window * 2 // 3
    m = market_return.reindex(returns.index)
    var_m = m.rolling(window, min_periods=minp).var(ddof=1)
    cov = returns.rolling(window, min_periods=minp).cov(m)
    beta = cov.div(var_m, axis=0)
    resid = returns - beta.mul(m, axis=0)
    return resid.rolling(window, min_periods=minp).skew()


class IdiosyncraticSkew(Factor):
    name = "idio_skew_60d"
    direction = -1  # high idiosyncratic skew => expected underperformance

    def compute_panel(self, loader) -> pd.DataFrame:
        return idio_skew_panel(loader.get_returns("log"), loader.market_return, window=60)


register(IdiosyncraticSkew())
