"""Unit tests for Subphase 3.6 incremental-value machinery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.diagnostics.incremental import (
    benchmark_squared_sharpe,
    block_bootstrap_alpha,
    incremental_alpha,
    spanning_regression,
)


def _benchmark(T=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0004, 0.01, (T, 3)), columns=["f1", "f2", "f3"])


def test_spanned_factor_has_no_alpha():
    F = _benchmark()
    rng = np.random.default_rng(1)
    spanned = 0.5 * F["f1"] - 0.3 * F["f2"] + rng.normal(0, 0.008, len(F))
    r = incremental_alpha(spanned, F)
    assert abs(r["hac_t"]) < 2.0                      # intercept not significant
    assert r["theta_improvement"] < 0.01             # negligible Sharpe gain
    assert r["r_squared"] > 0.3                       # well explained by benchmark


def test_unspanned_alpha_is_detected():
    F = _benchmark()
    rng = np.random.default_rng(2)
    target = 0.002 + 0.4 * F["f1"] + rng.normal(0, 0.008, len(F))
    r = incremental_alpha(target, F)
    assert r["alpha"] > 0
    assert r["hac_t"] > 3.0                           # significant unspanned alpha
    assert r["grs_pvalue"] < 0.01
    assert r["theta_improvement"] > 0                 # lifts the tangency Sharpe


def test_tangency_identity_is_exact():
    # theta_augmented (via alpha/IR) must equal the direct augmented max squared Sharpe
    F = _benchmark(seed=3)
    rng = np.random.default_rng(4)
    target = 0.0015 + 0.3 * F["f2"] + rng.normal(0, 0.009, len(F))
    r = incremental_alpha(target, F)
    direct = benchmark_squared_sharpe(pd.concat([F, target.rename("R")], axis=1), ddof=0)
    assert np.isclose(r["theta_augmented"], direct)
    assert np.isclose(r["theta_augmented"], r["theta_benchmark"] + r["theta_improvement"])


def test_benchmark_squared_sharpe_matches_formula():
    F = _benchmark(seed=5)
    mu = F.mean().to_numpy()
    Sigma = np.cov(F.to_numpy(), rowvar=False, ddof=0)
    expected = float(mu @ np.linalg.solve(Sigma, mu))
    assert np.isclose(benchmark_squared_sharpe(F, ddof=0), expected)


def test_hac_se_widens_under_autocorrelation():
    # positively autocorrelated residuals -> HAC SE of alpha exceeds the naive iid SE
    rng = np.random.default_rng(6)
    T = 500
    e = np.zeros(T)
    innov = rng.normal(0, 0.01, T)
    for t in range(1, T):
        e[t] = 0.6 * e[t - 1] + innov[t]
    target = pd.Series(0.001 + e)
    F = pd.DataFrame(rng.normal(0, 0.01, (T, 1)), columns=["f1"])
    sr = spanning_regression(target, F, nw_lags=20)
    hac_se = sr["se"]["alpha"]
    # naive iid SE of alpha
    X = np.column_stack([np.ones(T), F["f1"].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    resid = target.to_numpy() - X @ beta
    sigma2 = (resid @ resid) / (T - 2)
    iid_se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[0, 0])
    assert hac_se > iid_se                            # HAC is the more honest (wider) SE


def test_block_bootstrap_separates_real_from_spanned():
    F = _benchmark(seed=7)
    rng = np.random.default_rng(8)
    real = 0.002 + rng.normal(0, 0.008, len(F))
    spanned = 0.6 * F["f1"] + rng.normal(0, 0.008, len(F))
    bb_real = block_bootstrap_alpha(real, F, n_boot=600, seed=0)
    bb_span = block_bootstrap_alpha(spanned, F, n_boot=600, seed=0)
    assert bb_real["ci_low"] > 0                       # real alpha CI excludes 0
    assert bb_span["ci_low"] < 0 < bb_span["ci_high"]  # spanned CI straddles 0
