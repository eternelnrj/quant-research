"""Build the factor-zoo summary table from the shared evaluation harness.

For each registered factor this computes multi-horizon IC, the Newey-West
t-stat (overlap-aware), and a decile long-short Sharpe, then applies a
Benjamini-Hochberg correction and a deflated Sharpe across the whole zoo - so
"which factors survive multiple testing" is one table rather than per factor.
Factors whose data is unavailable (no shares -> size; no fundamentals ->
value/quality) are skipped with a note, not an error.

This is pure computation with no file I/O, so it is importable from both the
CLI (``scripts/run_factor_zoo.py``) and notebooks. The CLI wrapper handles
writing the CSV and printing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.data.loader import DataLoader
from qer.diagnostics.deflated_sharpe import deflated_sharpe
from qer.diagnostics.factor_ic import compute_factor_ic, summarize_ic
from qer.diagnostics.multiple_testing import benjamini_hochberg, pvalue_from_tstat
from qer.diagnostics.portfolios import factor_long_short
from qer.factors import all_factors

DEFAULT_HORIZONS = (1, 5, 21)
DEFAULT_PRIMARY_H = 21
_MIN_HISTORY = 273  # ~ a year of trading days of warm-up before sampling


def build_factor_zoo_table(
    loader: DataLoader | None = None,
    years: int = 5,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    primary_horizon: int = DEFAULT_PRIMARY_H,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Evaluate every registered factor and return the zoo summary table.

    ``loader`` defaults to a fresh :class:`DataLoader`; pass an existing one
    (e.g. from a notebook) to avoid rebuilding it.
    """
    if loader is None:
        loader = DataLoader()
    close = loader.close
    start = close.index.max() - pd.DateOffset(years=years)
    dates = close.index[(close.index >= start) & (close.index <= close.index[-primary_horizon - 1])]
    dates = dates[dates >= close.index[_MIN_HISTORY]]

    rows = []
    for fac in all_factors():
        try:
            ic = compute_factor_ic(loader, fac, horizons=horizons, dates=dates)
            s = summarize_ic(ic[primary_horizon], newey_west_lags=primary_horizon - 1)
            ls = factor_long_short(
                loader, fac, n_buckets=n_buckets, horizon=primary_horizon, dates=dates
            )
            sr = ls.mean() / ls.std(ddof=1) if ls.std(ddof=1) > 0 else np.nan
        except (FileNotFoundError, KeyError) as exc:
            print(f"  SKIP {fac.name}: {type(exc).__name__} ({str(exc)[:60]})")
            continue
        rows.append(
            {
                "factor": fac.name,
                "direction": fac.direction,
                "mean_ic_21d": s.get("mean_ic"),
                "ic_ir": s.get("ic_ir_annualized"),
                "t_naive": s.get("t_stat"),
                "t_nw": s.get("t_stat_nw"),
                "hit_rate": s.get("hit_rate"),
                "ls_sharpe": sr,
                "n": s.get("n"),
            }
        )

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows).set_index("factor")
    n_trials = len(table)
    # t_nw is a Newey-West (HAC) statistic, so its asymptotic reference is the
    # normal (df=None) - not a single-sample-mean t with n-1 dof. The non-HAC
    # t_naive would instead warrant df = n - 1.
    pvals = {f: pvalue_from_tstat(t) for f, t in zip(table.index, table["t_nw"])}
    bh = benjamini_hochberg(pvals)
    table["p_value"] = pd.Series({k: v["p"] for k, v in bh.items()})
    table["survives_bh"] = pd.Series({k: v["reject"] for k, v in bh.items()})
    # Deflated Sharpe needs the benchmark scaled by the variance of the Sharpe
    # estimates ACROSS the trials (else the unit-variance default collapses every
    # value to 0). Estimate that variance empirically from the zoo's Sharpes.
    sr_series = table["ls_sharpe"].dropna()
    var_sr = float(sr_series.var(ddof=1)) if len(sr_series) > 1 else 1.0
    # Overlap-aware sample size: the long-short returns are primary_horizon-day
    # forward returns sampled daily, so ~primary_horizon consecutive observations
    # overlap. Treating all n as independent overstates significance, so deflate
    # the sample to n / primary_horizon effective independent observations.
    table["deflated_sharpe_ratio"] = [
        deflated_sharpe(
            sr,
            n_trials=n_trials,
            n_obs=max(int(n) // primary_horizon, 2),
            var_sharpe=var_sr,
        )
        if pd.notna(sr)
        else np.nan
        for sr, n in zip(table["ls_sharpe"], table["n"])
    ]
    return table
