"""Subphase 3.6: incremental value of a factor over a benchmark set.

The decisive question of the whole phase: does a graph factor add value *beyond* the
classical factors, once overlap-robust inference and the trial count are accounted for?
This module answers it with a spanning regression and its consequences.

Given the factor's long-short return ``R_t`` and the ``K`` benchmark (classical) long-short
returns ``F_t``, regress ``R_t = alpha + beta' F_t + eps_t``. The intercept ``alpha`` is the
*unspanned* alpha; its Newey-West (HAC) t-statistic is the honest test (returns overlap, so
``eps`` is autocorrelated). Two classical results make ``alpha`` interpretable:

* the appraisal ratio ``IR = alpha / sd(eps)``, and the identity
  ``theta_augmented = theta_benchmark + IR**2`` -- adding the factor lifts the maximum squared
  Sharpe of the benchmark set by exactly ``IR**2``;
* for a single test asset the Gibbons-Ross-Shanken statistic reduces to the squared t of the
  intercept, ``GRS = t_alpha**2 ~ F(1, T-K-1) = t^2``.

Everything here is numpy/scipy only. The HAC sandwich is reused from
:mod:`qer.diagnostics.exposures` so alpha and betas share one consistent standard error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from qer.diagnostics.exposures import _hac_ols_cov


def _align(target, factors) -> tuple[np.ndarray, np.ndarray, list[str]]:
    F = pd.DataFrame(factors)
    # sort_index: the HAC covariance uses lagged autocovariances, so rows must be in
    # chronological order; sort=False keeps concat explicit (avoids the pandas sort warning).
    df = pd.concat([pd.Series(target).rename("_y"), F], axis=1, sort=False).dropna().sort_index()
    y = df["_y"].to_numpy(dtype=float)
    Fv = df.drop(columns="_y").to_numpy(dtype=float)
    return y, Fv, [str(c) for c in F.columns]


def benchmark_squared_sharpe(factors, ddof: int = 0) -> float:
    """Maximum squared Sharpe of the benchmark set: ``mu' Sigma^-1 mu`` (ex-post tangency).

    ``ddof=0`` (population/ML moments) makes the appraisal-ratio identity in
    :func:`incremental_alpha` exact; use ``ddof=1`` for an unbiased covariance if reporting
    the number on its own.
    """
    F = pd.DataFrame(factors).dropna()
    if F.shape[1] == 0 or len(F) <= F.shape[1]:
        return 0.0
    mu = F.mean().to_numpy()
    Sigma = np.atleast_2d(np.cov(F.to_numpy(), rowvar=False, ddof=ddof))
    theta = float(mu @ np.linalg.solve(Sigma, mu))
    return max(theta, 0.0)


def spanning_regression(target, factors, nw_lags: int = 20) -> dict:
    """OLS of ``target`` on ``[1, factors]`` with a Newey-West HAC covariance.

    Returns the intercept (unspanned alpha), the betas, HAC t-stats for every coefficient,
    the (unbiased) residual variance, and R^2.
    """
    y, F, names = _align(target, factors)
    n, k = F.shape
    X = np.column_stack([np.ones(n), F])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    dof = n - (k + 1)
    cov = _hac_ols_cov(X, resid, lags=nw_lags)
    se = np.sqrt(np.diag(cov))
    coef_names = ["alpha", *names]
    t_nw = {nm: (b / s if s > 0 and np.isfinite(s) else np.nan)
            for nm, b, s in zip(coef_names, beta, se)}
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "alpha": float(beta[0]),
        "betas": dict(zip(names, beta[1:])),
        "t_stats_nw": t_nw,
        "se": dict(zip(coef_names, se)),
        "resid_var": rss / dof if dof > 0 else np.nan,
        "r_squared": 1.0 - rss / ss_tot if ss_tot > 0 else np.nan,
        "n": n,
        "k": k,
    }


def incremental_alpha(target, factors, nw_lags: int = 20, periods_per_year: int = 252) -> dict:
    """Unspanned alpha of ``target`` over the benchmark ``factors``, and its consequences.

    Reports the intercept, its HAC t-stat/p-value (overlap-robust, the headline test), the
    single-asset GRS statistic (``= t_alpha^2`` from the iid OLS t, ``~ F(1, T-K-1)``), the
    appraisal ratio ``IR`` (per-period and annualised), and the tangency identity
    ``theta_augmented = theta_benchmark + IR^2``.
    """
    y, F, names = _align(target, factors)
    n, k = F.shape
    X = np.column_stack([np.ones(n), F])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    dof = n - (k + 1)
    alpha = float(beta[0])

    # iid OLS t of alpha -> single-asset GRS = t^2 ~ F(1, dof)
    sigma2_unb = rss / dof if dof > 0 else np.nan
    xtx_inv = np.linalg.inv(X.T @ X)
    se_iid = float(np.sqrt(sigma2_unb * xtx_inv[0, 0])) if np.isfinite(sigma2_unb) else np.nan
    t_iid = alpha / se_iid if se_iid and se_iid > 0 else np.nan
    grs = float(t_iid ** 2) if np.isfinite(t_iid) else np.nan
    grs_p = float(stats.f.sf(grs, 1, dof)) if (np.isfinite(grs) and dof > 0) else np.nan

    # overlap-robust HAC t of alpha (preferred inference)
    cov = _hac_ols_cov(X, resid, lags=nw_lags)
    se_hac = float(np.sqrt(cov[0, 0]))
    t_hac = alpha / se_hac if se_hac > 0 else np.nan
    p_hac = float(2 * stats.norm.sf(abs(t_hac))) if np.isfinite(t_hac) else np.nan

    # appraisal ratio and the tangency identity (ML residual sd, ML theta -> exact identity)
    sigma_ml = np.sqrt(rss / n)
    ir = alpha / sigma_ml if sigma_ml > 0 else np.nan
    theta_f = benchmark_squared_sharpe(pd.DataFrame(F, columns=names), ddof=0)
    return {
        "alpha": alpha,
        "hac_t": t_hac,
        "hac_pvalue": p_hac,
        "grs_stat": grs,
        "grs_pvalue": grs_p,
        "appraisal_ratio": float(ir) if np.isfinite(ir) else np.nan,
        "appraisal_ratio_annual": float(ir * np.sqrt(periods_per_year)) if np.isfinite(ir) else np.nan,
        "theta_benchmark": theta_f,
        "theta_augmented": theta_f + (ir ** 2 if np.isfinite(ir) else 0.0),
        "theta_improvement": float(ir ** 2) if np.isfinite(ir) else np.nan,
        "resid_vol": float(sigma_ml),
        "r_squared": spanning_regression(target, factors, nw_lags)["r_squared"],
        "n": n,
        "k": k,
    }


def block_bootstrap_alpha(target, factors, n_boot: int = 2000, block: int | None = None,
                          seed: int = 0, ci: float = 0.95) -> dict:
    """Circular moving-block bootstrap confidence interval for the unspanned alpha.

    Resamples length-``block`` circular blocks of the joint ``(target, factors)`` rows
    (preserving short-range dependence from overlapping returns), refits the spanning
    regression, and returns a percentile CI plus a two-sided bootstrap p-value for
    ``alpha = 0``. ``block`` defaults to ``round(n**(1/3))``.
    """
    y, F, names = _align(target, factors)
    n = len(y)
    data = np.column_stack([y, F])
    if block is None:
        block = max(1, int(round(n ** (1.0 / 3.0))))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    alphas = np.empty(n_boot)
    base = np.arange(block)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(s + base) % n for s in starts])[:n]
        d = data[idx]
        Xb = np.column_stack([np.ones(n), d[:, 1:]])
        beta_b, *_ = np.linalg.lstsq(Xb, d[:, 0], rcond=None)
        alphas[b] = beta_b[0]
    lo = (1.0 - ci) / 2.0
    p = 2.0 * min(float((alphas <= 0).mean()), float((alphas >= 0).mean()))
    return {
        "alpha_mean": float(alphas.mean()),
        "ci_low": float(np.quantile(alphas, lo)),
        "ci_high": float(np.quantile(alphas, 1.0 - lo)),
        "p_two_sided": float(min(p, 1.0)),
        "block": block,
        "n_boot": n_boot,
    }
