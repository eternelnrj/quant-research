"""Multiple-testing corrections across the factor zoo.

Testing eight factors and celebrating the best is p-hacking. Apply Bonferroni
(control family-wise error) or Benjamini-Hochberg (control false-discovery rate)
to the per-factor p-values before declaring anything significant.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict:
    m = len(pvalues)
    thresh = alpha / m if m else alpha
    return {k: {"p": p, "reject": p <= thresh} for k, p in pvalues.items()}


def benjamini_hochberg(pvalues: dict[str, float], alpha: float = 0.05) -> dict:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    # largest k with p_(k) <= (k/m) * alpha
    k_max = 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= (i / m) * alpha:
            k_max = i
    reject_names = {name for name, _ in items[:k_max]}
    return {k: {"p": p, "reject": k in reject_names} for k, p in pvalues.items()}


def pvalue_from_tstat(t: float, df: int | None = None) -> float:
    """Two-sided p-value from a t-statistic against an explicit reference.

    The correct reference depends on how ``t`` was computed, so the caller
    declares it rather than this function guessing:

      - ``df=None`` -> standard normal. The asymptotic reference for a
        Newey-West / HAC statistic (e.g. the IC ``t_nw``), which has no exact
        finite-sample t distribution.
      - ``df=n-1`` -> a single-sample mean test (plain, non-HAC mean/se).
      - ``df=n-k`` -> a regression coefficient with ``k`` estimated parameters.

    Hardcoding any one ``df`` is wrong for the others, so there is no default
    degrees-of-freedom assumption baked in.
    """
    if t is None or np.isnan(t):
        return np.nan
    if df is None:
        return float(2.0 * stats.norm.sf(abs(t)))
    if df < 1:
        return np.nan
    return float(2.0 * stats.t.sf(abs(t), df=df))
