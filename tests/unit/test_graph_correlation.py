"""Unit tests for Subphase 3.2 correlation construction and sparsifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse.csgraph import connected_components

from qer.graphs.correlation import (
    ledoit_wolf_covariance,
    mantegna_distance,
    mst_graph,
    shrunk_correlation,
    threshold_graph,
)


def _factor_returns(n_obs=80, n_names=40, n_factors=3, seed=0):
    """Returns driven by a few common factors -> non-trivial correlation structure."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n_obs, n_factors))
    B = rng.normal(size=(n_factors, n_names))
    idio = rng.normal(scale=0.5, size=(n_obs, n_names))
    X = F @ B + idio
    cols = [f"T{i:02d}" for i in range(n_names)]
    return pd.DataFrame(X, columns=cols)


def test_shrunk_correlation_is_a_valid_correlation_matrix():
    corr = shrunk_correlation(_factor_returns())
    A = corr.to_numpy()
    assert np.allclose(A, A.T, atol=1e-12)                  # symmetric
    assert np.allclose(np.diag(A), 1.0, atol=1e-12)         # unit diagonal
    assert A.min() >= -1.0 - 1e-9 and A.max() <= 1.0 + 1e-9  # bounded
    assert np.linalg.eigvalsh(A).min() >= -1e-8             # PSD


def test_shrinkage_improves_conditioning():
    # n close to p: the sample correlation is ill-conditioned; shrinkage helps.
    R = _factor_returns(n_obs=60, n_names=50, seed=1)
    sample_corr = R.corr().to_numpy()
    shrunk = shrunk_correlation(R).to_numpy()
    assert np.linalg.cond(shrunk) < np.linalg.cond(sample_corr)


def test_ledoit_wolf_shrinkage_in_unit_interval():
    _, shrink = ledoit_wolf_covariance(_factor_returns().to_numpy())
    assert 0.0 <= shrink <= 1.0


def test_ledoit_wolf_matches_sklearn():
    sk = pytest.importorskip("sklearn.covariance")
    X = _factor_returns(seed=2).to_numpy()
    cov_mine, _ = ledoit_wolf_covariance(X)
    cov_sk, _ = sk.ledoit_wolf(X)
    assert np.allclose(cov_mine, cov_sk, rtol=1e-6, atol=1e-10)


def test_threshold_graph_keeps_only_strong_edges():
    corr = shrunk_correlation(_factor_returns())
    thr = 0.30
    g = threshold_graph(corr, threshold=thr)
    A = g.to_numpy()
    assert np.allclose(A, A.T)                              # symmetric
    assert np.allclose(np.diag(A), 0.0)                     # no self-loops
    kept = A[A > 0]
    assert (kept >= thr - 1e-12).all()                      # every kept edge clears thr
    # an edge is kept iff |rho| >= thr (off-diagonal)
    off = ~np.eye(len(A), dtype=bool)
    expected = (np.abs(corr.to_numpy()) >= thr) & off
    assert np.array_equal(A > 0, expected)


def test_mst_is_a_connected_spanning_tree():
    corr = shrunk_correlation(_factor_returns(n_names=30))
    g = mst_graph(corr)
    A = g.to_numpy()
    n = len(A)
    assert np.allclose(A, A.T)                              # symmetric
    assert np.allclose(np.diag(A), 0.0)                     # no self-loops
    n_edges = int((A > 0).sum() // 2)
    assert n_edges == n - 1                                 # spanning tree edge count
    n_comp, _ = connected_components(A > 0, directed=False)
    assert n_comp == 1                                      # connected


def test_mantegna_distance_metric_basics():
    corr = shrunk_correlation(_factor_returns(n_names=15))
    d = mantegna_distance(corr).to_numpy()
    assert np.allclose(np.diag(d), 0.0)                     # zero self-distance
    assert (d >= -1e-12).all()                              # non-negative
    assert np.allclose(d, d.T)                              # symmetric


def test_constant_name_is_dropped_and_backbone_stays_connected():
    # a halted/constant-price name has zero-variance returns: it must be dropped,
    # not enter as a rho==0 node that falls out of the MST backbone.
    X = _factor_returns(n_obs=120, n_names=15, seed=5)
    X["T07"] = 100.0
    corr = shrunk_correlation(X)
    assert "T07" not in corr.columns                        # dropped, not degenerate
    A = mst_graph(corr).to_numpy()
    n = len(A)
    assert int((A > 0).sum() // 2) == n - 1                 # still a spanning tree
    n_comp, _ = connected_components(A > 0, directed=False)
    assert n_comp == 1                                      # connected


def test_mst_attaches_a_zero_correlation_node():
    # even a node uncorrelated with everything must stay in the connected backbone
    corr = shrunk_correlation(_factor_returns(n_names=10, seed=6))
    corr.iloc[0, 1:] = 0.0
    corr.iloc[1:, 0] = 0.0
    A = mst_graph(corr).to_numpy()
    n = len(A)
    assert int((A > 0).sum() // 2) == n - 1
    n_comp, _ = connected_components(A > 0, directed=False)
    assert n_comp == 1
