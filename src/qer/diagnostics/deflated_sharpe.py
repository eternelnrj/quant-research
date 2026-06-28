"""Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014).

A high Sharpe found after trying many strategies is partly luck. The deflated
Sharpe discounts an observed Sharpe by the number of independent trials and by
the return distribution's skew/kurtosis, returning the probability that the true
Sharpe is positive.

Normal-distribution maths come from ``scipy.stats.norm`` (already a project
dependency and used elsewhere) rather than a hand-rolled approximation.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
from scipy.stats import norm


def expected_max_sharpe(n_trials: int, var_sharpe: float = 1.0) -> float:
    """Expected maximum of ``n_trials`` independent N(0, var_sharpe) Sharpes."""
    if n_trials <= 1:
        return 0.0
    e = np.euler_gamma  # Euler-Mascheroni constant
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    return sqrt(var_sharpe) * ((1 - e) * z1 + e * z2)


def deflated_sharpe(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    var_sharpe: float = 1.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Computes Deflated Sharpe Ratio.

    ``observed_sharpe`` and the result are in per-observation units; pass the
    non-annualised Sharpe and the sample length ``n_obs``.

    ``var_sharpe`` is the variance of the Sharpe *estimates across the trials*
    (estimate it empirically from the trial Sharpes), and MUST be in the same
    units as ``observed_sharpe``. The default of 1.0 assumes unit-variance
    (standardised) Sharpes; for per-observation Sharpes it is almost always
    wrong and collapses the result to 0 - pass the real cross-trial variance.
    """
    sr0 = expected_max_sharpe(n_trials, var_sharpe=var_sharpe)  # selection benchmark
    denom = sqrt(1 - skew * observed_sharpe + (kurtosis - 1) / 4.0 * observed_sharpe**2)
    if denom <= 0 or n_obs <= 1:
        return np.nan
    z = (observed_sharpe - sr0) * sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))
