"""Subphase 3.4: directed lead-lag networks from market-residualised cross-correlations.

A directed edge ``i -> j`` means stock ``i``'s idiosyncratic return today helps predict
stock ``j``'s return a few days later. The construction, per rebalance window:

1. residualise the market out of each stock's returns (so the edges reflect *idiosyncratic*
   lead-lag, not a shared market-wide drift);
2. for each ordered pair and lag ``k`` in 1..5, measure the lagged cross-correlation
   ``corr(e_i,t, e_j,t+k)`` -- how well ``i`` today tracks ``j`` in ``k`` days;
3. turn each into a Bartlett z-score (SE ~ 1/sqrt(T)), Benjamini-Hochberg across all
   candidate ``(i,j,k)`` hypotheses, and keep an edge if any of its lags survives FDR.

From the directed adjacency come three features: out-degree (a *leader*), in-degree (a
*follower*), and an *upstream signal* (the edge-weighted recent return of a follower's
leaders). Because lead-lag is weak and short-lived in liquid large caps, the honest-null
machinery -- :func:`shuffled_null_density` -- is built alongside the detector so that ``no
lead-lag structure here`` can be reported as a finding rather than fished into spurious edges.

Dependency-light: numpy + scipy only. Granger causality (:func:`granger_pvalue`) is an
optional, off-critical-path alternative that needs ``statsmodels``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

_LAGS = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# Residualisation
# ---------------------------------------------------------------------------

def residualize_market(returns, market=None) -> pd.DataFrame:
    """Remove the market factor from each stock's returns (OLS residual over the window).

    ``market`` defaults to the equal-weighted cross-sectional mean of the window -- the
    dominant common factor -- so the residuals carry idiosyncratic moves only. Returns a
    residual panel of the same shape (column-centred).
    """
    R = pd.DataFrame(returns)
    Rv = R.to_numpy(dtype=float)
    m = Rv.mean(axis=1) if market is None else np.asarray(market, dtype=float)
    mc = m - m.mean()
    var_m = float(mc @ mc)
    if var_m <= 0:
        return R - R.mean(axis=0)
    Rc = Rv - Rv.mean(axis=0, keepdims=True)
    beta = (Rc.T @ mc) / var_m                 # per-stock market beta
    E = Rc - np.outer(mc, beta)                # OLS residual (centred)
    return pd.DataFrame(E, index=R.index, columns=R.columns)


# ---------------------------------------------------------------------------
# Directed edges via lagged cross-correlation + Bartlett + BH
# ---------------------------------------------------------------------------

def _zscore_cols(A: np.ndarray) -> np.ndarray:
    mu = A.mean(axis=0, keepdims=True)
    sd = A.std(axis=0, ddof=0, keepdims=True)
    return (A - mu) / np.where(sd > 0, sd, 1.0)   # constant column -> zeros


def _bh_reject(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg rejection mask over a flat array of p-values."""
    m = pvals.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = pvals[order] <= thresh
    reject = np.zeros(m, dtype=bool)
    if below.any():
        kmax = np.max(np.nonzero(below)[0])
        reject[order[: kmax + 1]] = True
    return reject


def leadlag_edges(resid, lags=_LAGS, alpha: float = 0.10) -> pd.DataFrame:
    """Signed directed adjacency ``D`` (rows lead, columns follow) after BH-FDR control.

    ``D[i, j]`` is the signed cross-correlation at the strongest surviving lag of the
    hypothesis ``i -> j``, or 0 if no lag survives. Diagonal is 0.
    """
    E = pd.DataFrame(resid)
    names = E.columns
    A = E.to_numpy(dtype=float)
    W, p = A.shape
    ks = [k for k in lags if 0 < k < W]
    if p < 2 or not ks:
        return pd.DataFrame(np.zeros((p, p)), index=names, columns=names)

    Cs = np.empty((len(ks), p, p))
    zs = np.empty((len(ks), p, p))
    for li, k in enumerate(ks):
        lead = _zscore_cols(A[:-k])            # e_i at t
        follow = _zscore_cols(A[k:])           # e_j at t+k
        n = lead.shape[0]
        C = (lead.T @ follow) / n              # C[i,j] = corr(e_i,t, e_j,t+k)
        np.fill_diagonal(C, 0.0)
        Cs[li], zs[li] = C, C * np.sqrt(n)     # Bartlett z

    offdiag = np.broadcast_to(~np.eye(p, dtype=bool), zs.shape)
    pvals = 2.0 * stats.norm.sf(np.abs(zs))    # two-sided
    reject = np.zeros(zs.shape, dtype=bool)
    reject[offdiag] = _bh_reject(pvals[offdiag], alpha)

    absC_surv = np.where(reject, np.abs(Cs), -1.0)
    best = absC_surv.argmax(axis=0)                              # (p,p) best-lag index
    D = np.take_along_axis(Cs, best[None], axis=0)[0]           # signed C at best lag
    D = np.where(reject.any(axis=0), D, 0.0)
    return pd.DataFrame(D, index=names, columns=names)


# ---------------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------------

def out_degree(D, normalized: bool = True) -> pd.Series:
    """Leader strength: total absolute lead-weight from each node."""
    Dm = pd.DataFrame(D)
    od = Dm.abs().sum(axis=1).to_numpy()
    if normalized and len(Dm) > 1:
        od = od / (len(Dm) - 1)
    return pd.Series(od, index=Dm.index, name="leadlag_out_degree")


def in_degree(D, normalized: bool = True) -> pd.Series:
    """Follower strength: total absolute lead-weight into each node."""
    Dm = pd.DataFrame(D)
    idg = Dm.abs().sum(axis=0).to_numpy()
    if normalized and len(Dm) > 1:
        idg = idg / (len(Dm) - 1)
    return pd.Series(idg, index=Dm.columns, name="leadlag_in_degree")


def upstream_signal(D, resid, lookback: int = 5) -> pd.Series:
    """Edge-weighted recent residual return of each follower's leaders.

    For follower ``j``: ``sum_i D[i,j] * rbar_i / sum_i |D[i,j]|``, where ``rbar_i`` is
    ``i``'s mean residual return over the last ``lookback`` days. Predicts ``j`` to move in
    the (sign-weighted) direction its leaders just moved. Nodes with no leaders get 0.
    """
    Dm = pd.DataFrame(D)
    E = pd.DataFrame(resid)
    names = Dm.index
    rbar = E.iloc[-lookback:].mean(axis=0).reindex(names).to_numpy()
    Dv = Dm.to_numpy()
    num = Dv.T @ rbar                          # sum_i D[i,j] rbar_i
    den = np.abs(Dv).sum(axis=0)               # sum_i |D[i,j]|
    up = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return pd.Series(up, index=names, name="leadlag_upstream")


# ---------------------------------------------------------------------------
# Edge density and the shuffled null
# ---------------------------------------------------------------------------

def edge_density(D) -> float:
    """Fraction of the ``p(p-1)`` possible directed edges that are present."""
    Dv = np.asarray(D)
    p = Dv.shape[0]
    return float((Dv != 0).sum()) / (p * (p - 1)) if p > 1 else 0.0


def _mean_abs_lag1_autocorr(resid) -> float:
    """Mean absolute lag-1 autocorrelation across the residual series."""
    A = np.asarray(pd.DataFrame(resid).to_numpy(), dtype=float)
    A = A - A.mean(axis=0, keepdims=True)
    num = (A[1:] * A[:-1]).sum(axis=0)
    den = (A * A).sum(axis=0)
    ac = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return float(np.mean(np.abs(ac)))


def shuffled_null_density(resid, lags=_LAGS, alpha: float = 0.10,
                          n_shuffles: int = 200, seed: int = 0) -> np.ndarray:
    """Edge density under an independent circular-shift null.

    Each residual series is circularly shifted by its own random offset, which preserves each
    series' marginal distribution *and* its own autocorrelation but destroys *all* cross-series
    dependence -- both lagged and contemporaneous. The null therefore represents mutual
    cross-sectional independence. See :func:`leadlag_density_report` for the interpretation and
    the one caveat this design carries.
    """
    E = pd.DataFrame(resid)
    A = E.to_numpy(dtype=float)
    W, p = A.shape
    rng = np.random.default_rng(seed)
    out = np.empty(n_shuffles)
    for s in range(n_shuffles):
        shifts = rng.integers(1, W, size=p)
        A_sh = np.column_stack([np.roll(A[:, j], int(shifts[j])) for j in range(p)])
        D = leadlag_edges(pd.DataFrame(A_sh, columns=E.columns), lags=lags, alpha=alpha)
        out[s] = edge_density(D.to_numpy())
    return out


def leadlag_density_report(resid, lags=_LAGS, alpha: float = 0.10,
                           n_shuffles: int = 200, seed: int = 0) -> dict:
    """Compare observed edge density to the shuffled null: the honest-null gate.

    ``p_value`` is the fraction of null densities at least as large as the observed one; a small
    value means genuine lead-lag structure, a large value means ``no structure``.

    Caveat (what the null does and does not test). The null of
    :func:`shuffled_null_density` is mutual cross-sectional *independence*, so a small p-value
    rejects independence -- which is *directional lead-lag* only if the residuals carry no other
    cross-sectional dependence. If the residuals retain contemporaneous correlation (e.g.\\ a
    sector factor surviving the market residualisation) *and* non-trivial autocorrelation, the
    two together induce spurious *lagged* cross-correlations that the independent-shift null does
    not reproduce, which can inflate significance. Empirically this is negligible at the
    autocorrelation typical of daily large-cap residuals (mean |lag-1| <~ 0.1) and only material
    at high autocorrelation (>~ 0.3). ``mean_abs_autocorr`` is reported so the caller can judge
    trustworthiness; when it is large, read ``p_value`` as a test against complete independence,
    not against ``no lead-lag given the contemporaneous structure``, and prefer a
    contemporaneous-preserving surrogate (e.g.\\ cross-spectrum-preserving phase randomisation).
    """
    D = leadlag_edges(resid, lags=lags, alpha=alpha)
    actual = edge_density(D.to_numpy())
    null = shuffled_null_density(resid, lags=lags, alpha=alpha, n_shuffles=n_shuffles, seed=seed)
    return {
        "actual_density": actual,
        "null_mean": float(null.mean()),
        "null_q95": float(np.quantile(null, 0.95)),
        "p_value": float((null >= actual).mean()),
        "n_edges": int((D.to_numpy() != 0).sum()),
        "mean_abs_autocorr": _mean_abs_lag1_autocorr(resid),
    }


# ---------------------------------------------------------------------------
# Optional: Granger causality (needs statsmodels)
# ---------------------------------------------------------------------------

def granger_pvalue(e_lead, e_follow, maxlag: int = 5) -> float:  # pragma: no cover - optional
    """Min-over-lags p-value that ``e_lead`` Granger-causes ``e_follow`` (needs statsmodels)."""
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError as exc:
        raise ImportError("granger_pvalue needs statsmodels (optional, off critical path).") from exc
    data = np.column_stack([np.asarray(e_follow, dtype=float), np.asarray(e_lead, dtype=float)])
    try:
        res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
    except TypeError:  # statsmodels >= 0.15 removed the `verbose` argument
        res = grangercausalitytests(data, maxlag=maxlag)
    return float(min(res[k][0]["ssr_ftest"][1] for k in range(1, maxlag + 1)))
