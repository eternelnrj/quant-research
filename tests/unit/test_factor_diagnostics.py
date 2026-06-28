"""Unit tests for the Phase 2 diagnostics: portfolios, FF5 exposures,
multiple-testing corrections, and the deflated Sharpe ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.diagnostics.deflated_sharpe import deflated_sharpe, expected_max_sharpe
from qer.diagnostics.exposures import ff5_exposures
from qer.diagnostics.multiple_testing import (
    benjamini_hochberg,
    bonferroni,
    pvalue_from_tstat,
)
from qer.diagnostics.portfolios import (
    long_short_returns,
    long_short_turnover,
    net_long_short,
    top_bucket_turnover,
)


def _panel(n_dates=30, n_names=20, seed=0):
    idx = pd.bdate_range("2022-01-03", periods=n_dates)
    rng = np.random.default_rng(seed)
    cols = [f"T{i:02d}" for i in range(n_names)]
    return pd.DataFrame(rng.normal(size=(n_dates, n_names)), index=idx, columns=cols)


def test_long_short_is_top_minus_bottom():
    oriented = _panel(seed=1)
    # forward return perfectly equals the oriented score -> LS must be positive
    fwd = oriented.copy()
    ls = long_short_returns(oriented, fwd, n_buckets=5)
    assert (ls > 0).all()


def test_turnover_zero_when_ranks_constant():
    idx = pd.bdate_range("2022-01-03", periods=10)
    cols = [f"T{i}" for i in range(10)]
    # identical rows -> identical buckets each day -> zero turnover
    oriented = pd.DataFrame(np.tile(np.arange(10.0), (10, 1)), index=idx, columns=cols)
    turn = top_bucket_turnover(oriented, n_buckets=5)
    assert np.allclose(turn.values, 0.0)


def test_net_long_short_below_gross_when_turnover_positive():
    oriented = _panel(seed=3)
    fwd = _panel(seed=4)
    gross = long_short_returns(oriented, fwd, n_buckets=5)
    net = net_long_short(oriented, fwd, n_buckets=5, cost_per_unit_turnover=0.01)
    common = gross.index.intersection(net.index)
    assert (net.loc[common] <= gross.loc[common] + 1e-12).all()


def test_net_long_short_charges_the_short_leg_too():
    """A stable long leg but a churning short leg must still incur cost - the
    old top-bucket-only model would have charged nothing here."""
    idx = pd.bdate_range("2022-01-03", periods=2)
    cols = [f"T{i}" for i in range(10)]
    # top two (T8,T9) stay top both days; bottom two churn T0,T1 -> T2,T3
    oriented = pd.DataFrame(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [4, 5, 0, 1, 2, 3, 6, 7, 8, 9]],
        index=idx,
        columns=cols,
        dtype=float,
    )
    fwd = pd.DataFrame(np.zeros((2, 10)), index=idx, columns=cols)  # zero gross isolates cost
    # long leg is stable -> long-only turnover is zero ...
    assert float(top_bucket_turnover(oriented, n_buckets=5).iloc[0]) == 0.0
    # ... but the short leg fully churns, so combined turnover (and cost) is positive
    assert float(long_short_turnover(oriented, n_buckets=5).iloc[0]) > 0.0
    gross = long_short_returns(oriented, fwd, n_buckets=5)
    net = net_long_short(oriented, fwd, n_buckets=5, cost_per_unit_turnover=0.01)
    assert net.iloc[-1] < gross.iloc[-1]  # cost comes purely from the short leg


def test_ff5_recovers_known_betas():
    idx = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(5)
    ff5 = pd.DataFrame(
        {
            "mkt_rf": rng.normal(0, 0.01, 500),
            "smb": rng.normal(0, 0.008, 500),
            "hml": rng.normal(0, 0.008, 500),
            "rmw": rng.normal(0, 0.006, 500),
            "cma": rng.normal(0, 0.006, 500),
            "rf": np.full(500, 0.0001),
        },
        index=idx,
    )
    true_alpha, true_beta_mkt = 0.0005, 1.3
    port = (
        true_alpha
        + true_beta_mkt * ff5["mkt_rf"]
        + 0.5 * ff5["hml"]
        + ff5["rf"]
        + rng.normal(0, 0.0005, 500)
    )
    res = ff5_exposures(pd.Series(port, index=idx), ff5)
    assert abs(res["betas"]["mkt_rf"] - true_beta_mkt) < 0.05
    assert abs(res["alpha"] - true_alpha) < 0.0003


def test_ff5_exposures_hac_covers_all_coefficients_and_widens_under_autocorrelation():
    """Every coefficient (alpha and betas) gets a HAC t-stat from one sandwich,
    and HAC widens the SE vs plain OLS where the score is autocorrelated - alpha
    (its score is the residual) and a beta whose factor is itself autocorrelated.
    This is the case the old plain-OLS beta t-stats got wrong."""
    rng = np.random.default_rng(11)
    n = 1500

    def ar1(rho, sd):
        x = np.empty(n)
        x[0] = rng.normal(0, sd)
        for t in range(1, n):
            x[t] = rho * x[t - 1] + rng.normal(0, sd)
        return x

    idx = pd.bdate_range("2020-01-01", periods=n)
    ff5 = pd.DataFrame(
        {
            "mkt_rf": ar1(0.6, 0.01),  # autocorrelated factor -> its score is autocorrelated
            "smb": rng.normal(0, 0.006, n),
            "hml": rng.normal(0, 0.006, n),
            "rmw": rng.normal(0, 0.005, n),
            "cma": rng.normal(0, 0.005, n),
        },
        index=idx,
    )
    e = ar1(0.7, 0.01)  # autocorrelated residuals -> alpha's score is autocorrelated
    y = pd.Series(1.0 * ff5["mkt_rf"].values + e, index=idx)  # true mkt beta 1, alpha 0

    res = ff5_exposures(y, ff5, nw_lags=21)
    # one sandwich -> a finite HAC t-stat and SE for alpha AND every factor
    assert set(res["se"]) == {"alpha", "mkt_rf", "smb", "hml", "rmw", "cma"}
    assert all(np.isfinite(v) for v in res["t_stats"].values())
    assert abs(res["betas"]["mkt_rf"] - 1.0) < 0.1

    # HAC SEs exceed the naive OLS SEs where the score is autocorrelated
    X = np.column_stack([np.ones(n), ff5.values])
    b = np.linalg.lstsq(X, y.values, rcond=None)[0]
    resid = y.values - X @ b
    ols_cov = (resid @ resid) / (n - X.shape[1]) * np.linalg.inv(X.T @ X)
    assert res["se"]["alpha"] > np.sqrt(ols_cov[0, 0])  # intercept score = residual (AR1)
    assert res["se"]["mkt_rf"] > np.sqrt(ols_cov[1, 1])  # factor AR1 -> score AR1


def test_bonferroni_and_bh():
    pvals = {"a": 0.001, "b": 0.04, "c": 0.2, "d": 0.5}
    bonf = bonferroni(pvals, alpha=0.05)
    assert bonf["a"]["reject"] and not bonf["b"]["reject"]  # 0.05/4 = 0.0125
    bh = benjamini_hochberg(pvals, alpha=0.05)
    # BH is at least as permissive as Bonferroni
    assert bh["a"]["reject"]
    assert sum(v["reject"] for v in bh.values()) >= sum(v["reject"] for v in bonf.values())


def test_pvalue_from_tstat_monotone():
    # default reference is the normal (df=None); p decreases as |t| grows
    assert pvalue_from_tstat(0.0) > pvalue_from_tstat(2.0) > pvalue_from_tstat(4.0)


def test_pvalue_from_tstat_explicit_reference():
    from scipy.stats import norm
    from scipy.stats import t as tdist

    # df=None -> exact standard normal (the HAC / asymptotic reference)
    assert abs(pvalue_from_tstat(2.0) - 2 * norm.sf(2.0)) < 1e-12
    assert abs(pvalue_from_tstat(2.0, df=None) - 2 * norm.sf(2.0)) < 1e-12
    # df given -> Student-t with exactly that df (fatter tails for small df)
    assert abs(pvalue_from_tstat(2.0, df=10) - 2 * tdist.sf(2.0, df=10)) < 1e-12
    assert pvalue_from_tstat(2.0, df=5) > pvalue_from_tstat(2.0, df=1000)
    # large df converges to the normal
    assert abs(pvalue_from_tstat(2.0, df=100_000) - 2 * norm.sf(2.0)) < 1e-4
    # invalid df and NaN t -> NaN
    assert np.isnan(pvalue_from_tstat(2.0, df=0))
    assert np.isnan(pvalue_from_tstat(np.nan))


def test_deflated_sharpe_overlap_correction_is_more_conservative():
    # a clear winner: fewer effective (overlap-corrected) observations -> lower DSR
    full = deflated_sharpe(0.21, n_trials=8, n_obs=1000, var_sharpe=0.015)
    overlap = deflated_sharpe(0.21, n_trials=8, n_obs=1000 // 21, var_sharpe=0.015)
    assert overlap < full


def test_deflated_sharpe_decreases_with_more_trials():
    base = deflated_sharpe(0.15, n_trials=1, n_obs=500)
    many = deflated_sharpe(0.15, n_trials=50, n_obs=500)
    assert 0.0 <= many <= base <= 1.0


def test_deflated_sharpe_uses_cross_trial_variance_not_unit_default():
    """Per-observation Sharpes must not collapse to 0 when the (small) cross-trial
    variance is supplied - the unit-variance default is the wrong scale for them."""
    sharpes = [-0.08, 0.19, -0.11, 0.03, 0.08, -0.11, 0.21, 0.04]  # per-observation, like the zoo
    var_sr = float(np.var(sharpes, ddof=1))
    best = max(sharpes)
    # with the default unit variance the benchmark (~1.46) swamps everything -> 0
    assert deflated_sharpe(best, n_trials=len(sharpes), n_obs=1000) == 0.0
    # with the empirical cross-trial variance the best factor is clearly nonzero
    dsr = deflated_sharpe(best, n_trials=len(sharpes), n_obs=1000, var_sharpe=var_sr)
    assert 0.5 < dsr < 1.0
    # a negative-Sharpe factor still reads ~0 under the corrected scale
    assert (
        deflated_sharpe(min(sharpes), n_trials=len(sharpes), n_obs=1000, var_sharpe=var_sr) < 0.05
    )


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(2) < expected_max_sharpe(10) < expected_max_sharpe(100)
