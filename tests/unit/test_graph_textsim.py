"""Unit tests for the Subphase 3.5 text-similarity engine (numpy core)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.graphs.textsim import (
    EmbeddingStore,
    cosine_similarity_matrix,
    coverage_report,
    extract_item1,
    knn_graph,
    neighbour_return_signal,
)


def _two_clusters(seed=0):
    """12 firms: A0-A5 near [1,0,..], B0-B5 near [0,1,..] in an 8-dim space."""
    rng = np.random.default_rng(seed)
    dim = 8
    names, vecs = [], []
    for grp, axis in (("A", 0), ("B", 1)):
        for i in range(6):
            v = np.zeros(dim)
            v[axis] = 3.0
            vecs.append(v + rng.normal(0, 0.25, dim))
            names.append(f"{grp}{i}")
    return pd.DataFrame(np.array(vecs), index=names)


# ---- cosine ---------------------------------------------------------------

def test_cosine_bounds_and_self_similarity():
    emb = pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], index=["x", "y", "z"])
    S = cosine_similarity_matrix(emb)
    assert np.allclose(np.diag(S.to_numpy()), 1.0)          # self-similarity 1
    assert np.isclose(S.loc["x", "y"], 0.0)                 # orthogonal
    assert np.isclose(S.loc["x", "z"], -1.0)               # opposite


# ---- kNN ------------------------------------------------------------------

def test_knn_neighbours_are_intra_cluster():
    A = knn_graph(_two_clusters(), k=3)
    assert (np.diag(A.to_numpy()) == 0).all()               # no self-loops
    for i in range(6):
        nbrs = A.loc[f"A{i}"]
        picked = nbrs[nbrs > 0].index.tolist()
        assert len(picked) == 3                             # exactly k neighbours
        assert all(n.startswith("A") for n in picked)       # all same cluster


def test_knn_k_capped_at_n_minus_one():
    emb = pd.DataFrame(np.eye(3), index=["a", "b", "c"])
    A = knn_graph(emb, k=10)                                 # k > n-1
    assert (A.to_numpy() > 0).sum(axis=1).max() <= 2         # at most n-1 neighbours


# ---- neighbour-return signal ---------------------------------------------

def test_neighbour_signal_reflects_cluster_comovement():
    emb = _two_clusters()
    A = knn_graph(emb, k=3)
    # A-cluster had strong recent returns, B-cluster weak
    W = 30
    rw = pd.DataFrame(np.zeros((W, 12)), columns=emb.index)
    rw.iloc[-5:, :6] = 0.05                                  # A firms up
    rw.iloc[-5:, 6:] = -0.05                                 # B firms down
    sig = neighbour_return_signal(A, rw, lookback=5)
    assert sig[[f"A{i}" for i in range(6)]].mean() > sig[[f"B{i}" for i in range(6)]].mean()


def test_neighbour_signal_restricts_to_common_firms():
    emb = _two_clusters()
    A = knn_graph(emb, k=3)
    rw = pd.DataFrame(np.zeros((10, 3)), columns=["A0", "A1", "A2"])   # only 3 firms have returns
    sig = neighbour_return_signal(A, rw, lookback=5)
    assert set(sig.index) <= {"A0", "A1", "A2"}


# ---- point-in-time embedding store ---------------------------------------

def _store():
    df = pd.DataFrame([
        {"ticker": "A", "filing_date": "2020-01-01", "emb_0": 1.0, "emb_1": 0.0},
        {"ticker": "A", "filing_date": "2021-01-01", "emb_0": 0.0, "emb_1": 1.0},
        {"ticker": "B", "filing_date": "2020-06-01", "emb_0": 1.0, "emb_1": 1.0},
    ])
    return EmbeddingStore.from_frame(df)


def test_store_returns_most_recent_filing_on_or_before_date():
    store = _store()
    e = store.as_of("2020-06-15", ["A", "B"])
    assert set(e.index) == {"A", "B"}
    assert np.allclose(e.loc["A"].to_numpy(), [1.0, 0.0])    # the 2020 filing, not 2021
    e2 = store.as_of("2021-06-01", ["A"])
    assert np.allclose(e2.loc["A"].to_numpy(), [0.0, 1.0])   # now the 2021 filing


def test_store_respects_universe_and_missing_dates():
    store = _store()
    assert store.as_of("2019-01-01", ["A"]) is None          # no filing yet -> no coverage
    e = store.as_of("2020-06-15", ["A"])                     # universe excludes B
    assert list(e.index) == ["A"]


def test_coverage_report_fractions():
    store = _store()
    rep = coverage_report(store, [("2019-06-01", ["A", "B"]), ("2020-06-15", ["A", "B"])])
    assert rep.loc[0, "coverage"] == 0.0                     # before any filing
    assert rep.loc[1, "coverage"] == 1.0                     # both covered


# ---- Item 1 parsing -------------------------------------------------------

def test_extract_item1_between_headings():
    html = "<html>ITEM 1. BUSINESS We design and sell rockets. ITEM 1A. RISK FACTORS ...</html>"
    body = extract_item1(html)
    assert body == "We design and sell rockets."

def test_extract_item1_absent_returns_empty():
    assert extract_item1("Some filing with no business section header.") == ""
