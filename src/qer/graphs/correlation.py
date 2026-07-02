"""Subphase 3.2: correlation construction and sparsification.

Ledoit-Wolf-shrunk correlation (the cheapest defence against high-dimensional
instability), then two sparsifiers so a result is never a one-method artefact:
a hard ``|rho|`` threshold graph and a minimum spanning tree on the Mantegna
distance ``d = sqrt(2(1 - rho))``. PMFG is left optional (MST + threshold satisfy
the two-sparsifier requirement).

Adjacency is returned as a ``(ticker x ticker)`` DataFrame, weighted by *similarity*
(``|rho|``) on kept edges, so the construction stays dependency-light (numpy +
scipy only) and the centrality layer (Subphase 3.3) can build networkx/igraph
from it. Crucially the MST is weighted by ``|rho|``, not by the distance the tree
was built from -- distance weights would rank the most *dissimilar* nodes as most
central and invert eigenvector centrality (the Subphase 3.2/3.3 caveat).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree


def ledoit_wolf_covariance(returns) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward ``mu * I``.

    Returns ``(cov, shrinkage)`` where ``shrinkage`` in [0, 1] is the intensity
    toward the scaled-identity target ``mu * I`` (``mu = trace(S) / p``). This is
    the same estimator as ``sklearn.covariance.LedoitWolf`` (no external dep).
    """
    X = np.asarray(returns, dtype=float)
    n, p = X.shape
    if n < 2 or p < 1:
        raise ValueError("need at least 2 observations and 1 series")
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / n                       # MLE sample covariance (ddof=0)
    mu = np.trace(S) / p
    d2 = np.sum((S - mu * np.eye(p)) ** 2) / p
    # b_bar^2 = (1/p)(1/n^2) sum_k || x_k x_k' - S ||_F^2
    xnorm2 = np.einsum("ij,ij->i", Xc, Xc)            # x_k . x_k
    xSx = np.einsum("ij,jk,ik->i", Xc, S, Xc)         # x_k' S x_k
    s_fro2 = np.sum(S * S)
    per_k = xnorm2 ** 2 - 2.0 * xSx + s_fro2
    b_bar2 = per_k.sum() / (n * n) / p
    b2 = min(b_bar2, d2)
    shrink = 0.0 if d2 <= 0 else float(min(1.0, max(0.0, b2 / d2)))
    cov = shrink * mu * np.eye(p) + (1.0 - shrink) * S
    return cov, shrink


def shrunk_correlation(returns) -> pd.DataFrame:
    """Ledoit-Wolf-shrunk correlation matrix as a ``(ticker x ticker)`` DataFrame.

    Operates on complete columns: any series with a NaN in the window is dropped
    (the trailing-window builder already enforces this at ``min_obs = window``).
    """
    R = pd.DataFrame(returns).dropna(axis=1, how="any")
    # Drop zero-variance names (e.g. a halted/constant-price stock over the window):
    # they carry no correlation information and would otherwise enter as degenerate
    # nodes with rho == 0 to everything -- which also breaks the MST backbone.
    R = R.loc[:, R.std(ddof=0) > 0]
    if R.shape[1] < 2 or R.shape[0] < 2:
        raise ValueError("need at least 2 complete, non-constant series and 2 observations")
    cov, _ = ledoit_wolf_covariance(R.to_numpy())
    std = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    return pd.DataFrame(corr, index=R.columns, columns=R.columns)


def mantegna_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Mantegna distance ``d = sqrt(2(1 - rho))`` (a proper metric on [-1, 1])."""
    d = np.sqrt(np.clip(2.0 * (1.0 - corr.to_numpy()), 0.0, None))
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


def threshold_graph(corr: pd.DataFrame, threshold: float = 0.30) -> pd.DataFrame:
    """Hard-threshold graph: keep an edge (weighted by ``|rho|``) iff ``|rho| >= threshold``.

    Symmetric, zero diagonal. Interpretable but can fragment -- which is exactly
    why it is paired with the always-connected MST.
    """
    A = np.abs(corr.to_numpy()).astype(float)
    np.fill_diagonal(A, 0.0)
    A[A < threshold] = 0.0
    return pd.DataFrame(A, index=corr.index, columns=corr.columns)


def mst_graph(corr: pd.DataFrame) -> pd.DataFrame:
    """Minimum spanning tree on the Mantegna distance, weighted by ``|rho|``.

    Returns a symmetric ``(ticker x ticker)`` adjacency with ``n - 1`` edges
    (``2(n-1)`` nonzeros), connected by construction. Edge *structure* comes from
    the distance MST; edge *weights* are similarities ``|rho|`` (see module note).
    """
    names = corr.index
    dist = mantegna_distance(corr).to_numpy()
    offdiag = ~np.eye(len(dist), dtype=bool)
    # a genuine zero-distance edge (rho == 1 between distinct names) would be lost
    # by the sparse MST's "nonzero = edge" convention; nudge it so it survives.
    dist = dist.copy()
    dist[offdiag & (dist == 0.0)] = 1e-12
    tree = minimum_spanning_tree(dist).toarray()
    edges = tree > 0
    absorr = np.abs(corr.to_numpy())
    adj = np.zeros_like(dist)
    # Weight by similarity |rho|, but never let a structural edge vanish: an MST
    # edge with |rho| == 0 (e.g. a node uncorrelated with all others) is still part
    # of the backbone, so floor its weight so the graph stays connected (n-1 edges).
    w = absorr[edges]
    adj[edges] = np.where(w > 0.0, w, 1e-12)
    adj = np.maximum(adj, adj.T)            # symmetrise (scipy returns directed)
    np.fill_diagonal(adj, 0.0)
    return pd.DataFrame(adj, index=names, columns=names)


def pmfg_graph(corr: pd.DataFrame) -> pd.DataFrame:  # noqa: ARG001 - optional sparsifier
    """Planar maximally filtered graph (optional).

    Not implemented: MST + threshold already satisfy the two-sparsifier
    requirement. Add via ``planarity``/``mlfinlab`` if a planar filtration is
    wanted as a third axis.
    """
    raise NotImplementedError(
        "PMFG is optional; MST + threshold satisfy the two-sparsifier requirement."
    )
