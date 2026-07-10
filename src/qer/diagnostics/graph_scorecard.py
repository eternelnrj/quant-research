"""Subphase 3.6: the graph-factor scorecard -- the honest, net-of-everything gate.

For each graph factor this assembles the one-line verdict: rank-IC and its information ratio,
the decile long-short Sharpe, the *unspanned* alpha over the classical factor set (with an
overlap-robust HAC t-stat and the tangency-Sharpe improvement it implies), and the Deflated
Sharpe Ratio computed against the *full* trial count from the pre-registered grid. A factor
earns a place only if it clears these together: a real IC, a positive spanning alpha that
survives HAC inference, and a Sharpe that survives the trial-count discount.

It also offers the cluster-vs-sector confusion matrix: do the correlation-graph communities
recover known GICS structure? -- a sanity check that the graph captures real economics.

Spanning/DSR are numpy/scipy only. The confusion matrix uses community detection (the optional
``graphs`` extra) and is skipped cleanly if unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.diagnostics.deflated_sharpe import deflated_sharpe
from qer.diagnostics.factor_ic import compute_factor_ic, summarize_ic
from qer.diagnostics.incremental import incremental_alpha
from qer.diagnostics.portfolios import factor_long_short
from qer.factors import all_factors


def _classical_returns(loader, classical_factors, horizon: int, n_buckets: int) -> pd.DataFrame:
    """Long-short return of each benchmark (classical) factor -> the spanning regressors."""
    cols = {}
    for f in classical_factors:
        s = factor_long_short(loader, f, n_buckets=n_buckets, horizon=horizon).dropna()
        if len(s) > 0:
            cols[f.name] = s
    return pd.DataFrame(cols)


def spanning_alpha_vs_classical(loader, graph_factor, classical_factors=None, *,
                                horizon: int = 21, n_buckets: int = 10) -> dict:
    """Regress a graph factor's long-short return on the classical factors -> incremental alpha."""
    classical_factors = classical_factors if classical_factors is not None else all_factors()
    F = _classical_returns(loader, classical_factors, horizon, n_buckets)
    target = factor_long_short(loader, graph_factor, n_buckets=n_buckets, horizon=horizon)
    return incremental_alpha(target, F, nw_lags=max(horizon - 1, 0))


def feature_scorecard(loader, factor, n_trials, *, classical_returns=None, classical_factors=None,
                      horizon: int = 21, n_buckets: int = 10, primary_ic_horizon: int = 21,
                      var_sharpe: float = 1.0, periods_per_year: int = 252) -> dict:
    """One-line scorecard for a single graph factor (IC, Sharpe, spanning alpha, deflated Sharpe)."""
    if classical_returns is None:
        classical_factors = classical_factors if classical_factors is not None else all_factors()
        classical_returns = _classical_returns(loader, classical_factors, horizon, n_buckets)

    ic = compute_factor_ic(loader, factor, horizons=(primary_ic_horizon,))[primary_ic_horizon]
    ic_stats = summarize_ic(ic.dropna(), newey_west_lags=primary_ic_horizon - 1)

    ls = factor_long_short(loader, factor, n_buckets=n_buckets, horizon=horizon).dropna()
    sd = ls.std(ddof=1)
    sr_pp = float(ls.mean() / sd) if sd > 0 else np.nan               # per-period Sharpe
    sharpe_ann = sr_pp * np.sqrt(periods_per_year) if np.isfinite(sr_pp) else np.nan

    span = (incremental_alpha(ls, classical_returns, nw_lags=max(horizon - 1, 0))
            if len(ls) > classical_returns.shape[1] + 2 and classical_returns.shape[1] > 0 else None)

    dsr = np.nan
    if n_trials and np.isfinite(sr_pp) and len(ls) > 2:
        dsr = deflated_sharpe(sr_pp, n_trials=n_trials, n_obs=len(ls), var_sharpe=var_sharpe,
                              skew=float(ls.skew()), kurtosis=float(ls.kurtosis() + 3.0))
    return {
        "factor": factor.name,
        "mean_ic": ic_stats.get("mean_ic", np.nan),
        "ic_t_nw": ic_stats.get("t_stat", ic_stats.get("t_nw", np.nan)),
        "ls_sharpe_ann": sharpe_ann,
        "span_alpha": span["alpha"] if span else np.nan,
        "span_hac_t": span["hac_t"] if span else np.nan,
        "theta_improvement": span["theta_improvement"] if span else np.nan,
        "deflated_sharpe": dsr,
        "n_obs": len(ls),
    }


def graph_scorecard(loader, graph_factors, classical_factors=None, *, horizon: int = 21,
                    n_buckets: int = 10, primary_ic_horizon: int = 21, n_trials: int | None = None,
                    periods_per_year: int = 252) -> pd.DataFrame:
    """Assemble the scorecard table over a list of graph factors (one row each).

    ``var_sharpe`` for the Deflated Sharpe is estimated from the cross-section of the graph
    factors' own per-period Sharpes (the empirical spread of the trials), as the DSR intends.
    """
    classical_factors = classical_factors if classical_factors is not None else all_factors()
    F = _classical_returns(loader, classical_factors, horizon, n_buckets)

    # first pass: per-period Sharpes -> empirical var_sharpe across the trials
    ls_series, sr_pp = {}, {}
    for gf in graph_factors:
        s = factor_long_short(loader, gf, n_buckets=n_buckets, horizon=horizon).dropna()
        ls_series[gf.name] = s
        sd = s.std(ddof=1)
        sr_pp[gf.name] = float(s.mean() / sd) if sd > 0 and len(s) > 2 else np.nan
    valid = [v for v in sr_pp.values() if np.isfinite(v)]
    var_sharpe = float(np.var(valid, ddof=1)) if len(valid) >= 2 else 1.0

    rows = [
        feature_scorecard(loader, gf, n_trials, classical_returns=F, horizon=horizon,
                          n_buckets=n_buckets, primary_ic_horizon=primary_ic_horizon,
                          var_sharpe=var_sharpe, periods_per_year=periods_per_year)
        for gf in graph_factors
    ]
    return pd.DataFrame(rows).set_index("factor")


# ---------------------------------------------------------------------------
# Cluster-vs-sector confusion matrix
# ---------------------------------------------------------------------------

def _adjusted_rand(a, b) -> float:
    """Adjusted Rand index between two labellings (numpy only)."""
    ct = pd.crosstab(pd.Series(np.asarray(a)), pd.Series(np.asarray(b))).to_numpy(dtype=float)
    n = ct.sum()

    def comb2(x):
        return x * (x - 1.0) / 2.0

    sum_ij = comb2(ct).sum()
    ai = comb2(ct.sum(axis=1)).sum()
    bj = comb2(ct.sum(axis=0)).sum()
    total = comb2(n)
    expected = ai * bj / total if total > 0 else 0.0
    denom = 0.5 * (ai + bj) - expected
    return float((sum_ij - expected) / denom) if denom != 0 else 0.0


def cluster_sector_matrix(communities, sectors) -> dict:
    """Confusion matrix and adjusted Rand index of community labels vs sector labels."""
    lab = pd.Series(communities)
    sec = pd.Series(sectors)
    common = lab.index.intersection(sec.index)
    lab, sec = lab.loc[common], sec.loc[common]
    return {
        "confusion": pd.crosstab(lab, sec),
        "adjusted_rand": _adjusted_rand(lab.to_numpy(), sec.to_numpy()),
        "n": len(common),
    }


def cluster_sector_confusion(loader, as_of, sectors, *, window: int = 120,
                             method: str = "louvain") -> dict:
    """Build the correlation-graph communities as of ``as_of`` and compare to ``sectors``.

    Needs the ``graphs`` extra (community detection); raises a clear error if absent.
    """
    from qer.graphs.centrality import communities
    from qer.graphs.correlation import mst_graph, shrunk_correlation
    from qer.graphs.windows import trailing_return_matrix

    rw = trailing_return_matrix(loader, as_of, window=window, min_obs=window)
    labels = communities(mst_graph(shrunk_correlation(rw)), method=method)
    return cluster_sector_matrix(labels, sectors)
