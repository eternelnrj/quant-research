"""Unit tests for Subphase 3.3 centralities, on graphs with known-analytic values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qer.graphs import centrality as C


def _adj(n, edges, w=1.0):
    """Symmetric adjacency DataFrame on n nodes from an edge list."""
    names = [f"N{i}" for i in range(n)]
    A = np.zeros((n, n))
    for i, j in edges:
        A[i, j] = A[j, i] = w
    return pd.DataFrame(A, index=names, columns=names)


def star(n):
    return _adj(n, [(0, j) for j in range(1, n)])          # node 0 = hub


def clique(n):
    return _adj(n, [(i, j) for i in range(n) for j in range(i + 1, n)])


def path(n):
    return _adj(n, [(i, i + 1) for i in range(n - 1)])


# ---- degree ---------------------------------------------------------------

def test_degree_star_hub_dominates():
    d = C.degree_centrality(star(5), normalized=True)
    assert d.idxmax() == "N0"                               # hub most connected
    assert np.isclose(d["N0"], 1.0)                         # 4/(5-1)
    assert np.allclose(d[["N1", "N2", "N3", "N4"]], 0.25)   # 1/(5-1)


def test_degree_clique_is_uniform():
    d = C.degree_centrality(clique(4), normalized=True)
    assert np.allclose(d.to_numpy(), 1.0)                   # every node connects to all


# ---- eigenvector ----------------------------------------------------------

def test_eigenvector_star_hub_is_most_central():
    e = C.eigenvector_centrality(star(6))
    assert e.idxmax() == "N0"
    assert (e >= -1e-12).all()                              # non-negative (Perron)
    assert np.isclose(np.linalg.norm(e.to_numpy()), 1.0)    # unit-normalised


def test_eigenvector_clique_is_uniform():
    e = C.eigenvector_centrality(clique(5))
    assert np.allclose(e.to_numpy(), e.iloc[0], atol=1e-9)  # symmetry => equal


def test_eigenvector_rejects_negative_weights():
    bad = clique(3)
    bad.iloc[0, 1] = bad.iloc[1, 0] = -0.5                  # a signed / distance weight
    with pytest.raises(ValueError, match="non-negative"):
        C.eigenvector_centrality(bad)


# ---- clustering -----------------------------------------------------------

def test_clustering_clique_is_one_star_and_path_zero():
    assert np.allclose(C.clustering_coefficient(clique(4)).to_numpy(), 1.0)
    assert np.allclose(C.clustering_coefficient(star(5)).to_numpy(), 0.0)
    assert np.allclose(C.clustering_coefficient(path(4)).to_numpy(), 0.0)


# ---- betweenness (optional: networkx) -------------------------------------

def test_betweenness_star_hub_path_middles():
    pytest.importorskip("networkx")
    b_star = C.betweenness_centrality(star(5))
    assert b_star.idxmax() == "N0"                          # hub bridges all leaf pairs
    assert np.allclose(b_star[["N1", "N2", "N3", "N4"]], 0.0)
    b_path = C.betweenness_centrality(path(5))
    assert b_path["N2"] > b_path["N0"]                      # middle bridges more than ends


# ---- communities + cohesion (optional: networkx) --------------------------

def test_communities_recover_two_disjoint_cliques():
    pytest.importorskip("networkx")
    # two 4-cliques joined by a single weak bridge -> two communities
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    edges += [(i, j) for i in range(4, 8) for j in range(i + 1, 8)]
    edges += [(3, 4)]                                       # bridge
    A = _adj(8, edges)
    lab = C.communities(A, method="louvain", seed=0)
    assert lab.nunique() == 2
    assert lab.iloc[:4].nunique() == 1 and lab.iloc[4:].nunique() == 1

    coh = C.community_cohesion(A, method="louvain", seed=0)
    assert (coh >= -1e-9).all() and (coh <= 1 + 1e-9).all()
    # a clique-interior node keeps almost all its weight inside its community
    assert coh["N0"] > coh["N3"]                            # N3 spends weight on the bridge


# ---- label alignment ------------------------------------------------------

def test_align_labels_removes_relabelling_noise():
    idx = [f"N{i}" for i in range(6)]
    prev = pd.Series([0, 0, 1, 1, 2, 2], index=idx)
    curr = pd.Series([2, 2, 0, 0, 1, 1], index=idx)         # same clusters, permuted ids
    aligned = C.align_labels(prev, curr)
    assert (aligned.to_numpy() == prev.to_numpy()).all()     # zero genuine migration

def test_align_labels_flags_a_real_move():
    idx = [f"N{i}" for i in range(6)]
    prev = pd.Series([0, 0, 1, 1, 2, 2], index=idx)
    curr = pd.Series([0, 0, 1, 2, 2, 2], index=idx)         # N3 moved cluster 1 -> 2
    aligned = C.align_labels(prev, curr)
    changed = (aligned.to_numpy() != prev.to_numpy())
    assert changed.sum() == 1 and idx[int(np.argmax(changed))] == "N3"
