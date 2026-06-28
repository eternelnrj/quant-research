"""Fama-French 5-factor exposures of a portfolio return stream.

A *time-series* regression of the long-short portfolio's returns on the FF5
factors (+ intercept) - distinct from the cross-sectional IC. The intercept is
the FF5-adjusted alpha; for a graph/classical factor the question is whether
that alpha survives the known risk factors.

Plain OLS point estimates via numpy, with a *single* Newey-West (Bartlett) HAC
covariance for the whole coefficient vector - so every t-stat (alpha and each
beta) comes from one consistent, overlap-aware standard error. This replaces an
earlier split that used a plain non-HAC OLS covariance for the betas and an
ad-hoc HAC-on-the-residual-mean for alpha alone (the betas were the ones most
overstated under overlapping forward-return windows). The covariance is the
asymptotic HAC sandwich (no small-sample dof rescale).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _hac_ols_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West HAC covariance of the OLS coefficient vector.

    ``(X'X)^-1 [ G_0 + sum_L w_L (G_L + G_L') ] (X'X)^-1`` with Bartlett weights
    ``w_L = 1 - L/(lags+1)`` and score autocovariances ``G_L = sum_t g_t g_{t-L}'``,
    ``g_t = X_t * e_t``. Reduces to the white (X'X)^-1 (sum g g') (X'X)^-1 at
    ``lags=0`` and accounts for the estimation of every coefficient, not just one.
    """
    g = X * resid[:, None]  # T x k score matrix
    s = g.T @ g
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gl = g[lag:].T @ g[:-lag]
        s += w * (gl + gl.T)
    # Sandwich (X'X)^-1 S (X'X)^-1 via solves rather than an explicit inverse,
    # which is the more numerically stable way to apply A^-1 (here A = X'X is
    # symmetric PSD). For k ~ 6 well-conditioned regressors the practical
    # difference is small, but solve() is the right default.
    xtx = X.T @ X
    z = np.linalg.solve(xtx, s)  # = (X'X)^-1 S
    return np.linalg.solve(xtx, z.T).T  # = (X'X)^-1 S (X'X)^-1


def ff5_exposures(portfolio_returns: pd.Series, ff5: pd.DataFrame, nw_lags: int = 21) -> dict:
    """Regress portfolio returns on FF5 factors; return alpha, betas, HAC t-stats.

    ``ff5`` columns are the factor returns (e.g. ``mkt_rf, smb, hml, rmw, cma``);
    a ``rf`` column, if present, is subtracted from ``portfolio_returns`` first.
    All t-stats use one Newey-West HAC covariance (``nw_lags`` Bartlett lags).
    """
    df = ff5.copy()
    df.columns = [c.lower() for c in df.columns]
    factors = [c for c in df.columns if c != "rf"]
    y = portfolio_returns.copy()
    aligned = pd.concat([y.rename("y"), df], axis=1).dropna()
    if "rf" in aligned:
        aligned["y"] = aligned["y"] - aligned["rf"]

    X = np.column_stack([np.ones(len(aligned)), aligned[factors].values])
    yv = aligned["y"].values
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta

    cov = _hac_ols_cov(X, resid, lags=nw_lags)  # one sandwich for all coefficients
    se = np.sqrt(np.diag(cov))
    names = ["alpha"] + factors
    tstats = {
        n: (b / s if s > 0 and np.isfinite(s) else np.nan) for n, b, s in zip(names, beta, se)
    }
    return {
        "alpha": float(beta[0]),
        "betas": dict(zip(factors, beta[1:])),
        # All t-stats use the Newey-West (Bartlett) HAC covariance. ``t_stats``
        # is kept for back-compat; ``t_stats_nw`` is the self-documenting alias,
        # and ``t_stat_kind`` records the method so callers can't mistake these
        # for plain-OLS t-stats.
        "t_stats": tstats,
        "t_stats_nw": tstats,
        "t_stat_kind": f"newey_west_hac(lags={nw_lags})",
        "se": dict(zip(names, se)),
        "n": len(yv),
    }
