"""Subphase 3.3: node centralities and communities on the correlation graph.

Each statistic maps a ``(ticker x ticker)`` adjacency -- the ``|rho|``-weighted,
symmetric output of :mod:`qer.graphs.correlation` -- to a per-ticker Series, so it
drops straight into the Subphase 3.2 panel engine as a ``snapshot_fn``. Degree,
eigenvector and clustering are pure numpy/scipy (testable in the base environment);
betweenness and communities use the optional ``graphs`` extra (networkx) and raise a
clear error if it is absent.

Weighting convention (the load-bearing subtlety of the whole subphase):

* Degree and eigenvector centrality use SIMILARITY weights ``|rho|`` -- a hub is a
  node that comoves strongly with other strong nodes.
* Betweenness is a shortest-path notion, so it uses DISTANCE ``1/|rho|`` -- a strong
  correlation is a *short* hop. Feeding similarity weights to a shortest-path routine
  (or distances to eigenvector centrality) silently inverts the ranking, so the
  conversion is explicit here and never left to the caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_matrix(adj: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
    A = np.asarray(adj, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square (ticker x ticker) matrix")
    A = A.copy()
    np.fill_diagonal(A, 0.0)
    return A, adj.index


# ---------------------------------------------------------------------------
# Core centralities (numpy/scipy only)
# ---------------------------------------------------------------------------

def degree_centrality(adj: pd.DataFrame, normalized: bool = True) -> pd.Series:
    """Weighted degree: the total edge weight incident on each node.

    With similarity (``|rho|``) weights this is how strongly a stock comoves with the
    rest of the network. If ``normalized``, divide by ``n-1`` so the scale is
    comparable across snapshots of different size (immaterial to the harness, which
    ranks cross-sectionally, but tidier).
    """
    A, idx = _as_matrix(adj)
    deg = A.sum(axis=1)
    if normalized and len(idx) > 1:
        deg = deg / (len(idx) - 1)
    return pd.Series(deg, index=idx, name="degree_centrality")


def eigenvector_centrality(adj: pd.DataFrame, tol: float = 1e-9) -> pd.Series:
    """Leading-eigenvector centrality of a NON-NEGATIVE symmetric adjacency.

    A node is central if it links to other central nodes -- the entries of the
    eigenvector of the largest eigenvalue. By Perron-Frobenius a non-negative matrix
    (here ``|rho|``-weighted) has a non-negative leading eigenvector, so the scores are
    well defined and non-negative. Passing negative entries (signed correlations, or
    distance weights) inverts the ranking and raises.
    """
    A, idx = _as_matrix(adj)
    if (A < -tol).any():
        raise ValueError(
            "eigenvector_centrality requires a non-negative adjacency; use |rho| "
            "similarity weights, not signed correlations or distance weights."
        )
    A = np.clip(A, 0.0, None)
    n = len(idx)
    if n == 1:
        return pd.Series([1.0], index=idx, name="eigenvector_centrality")
    vals, vecs = np.linalg.eigh(A)          # symmetric => real spectrum, ascending
    v = vecs[:, -1]                          # eigenvector of the largest eigenvalue
    if v.sum() < 0:                          # Perron vector is single-signed; fix sign
        v = -v
    v = np.clip(v, 0.0, None)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return pd.Series(v, index=idx, name="eigenvector_centrality")


def clustering_coefficient(adj: pd.DataFrame) -> pd.Series:
    """Local (binary) clustering coefficient: how tightly a node's neighbours link.

    On the edge structure ``B = 1(adj>0)``, for node ``i`` of degree ``k_i``,
    ``C_i = (B^3)_{ii} / (k_i (k_i - 1))`` -- the fraction of neighbour pairs that are
    themselves connected. ``C_i = 0`` when ``k_i < 2``.
    """
    A, idx = _as_matrix(adj)
    B = (A > 0).astype(float)
    k = B.sum(axis=1)
    closed = np.einsum("ij,jk,ki->i", B, B, B)   # (B^3)_ii = 2 * triangles through i
    denom = k * (k - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(denom > 0, closed / denom, 0.0)
    return pd.Series(c, index=idx, name="clustering_coefficient")


# ---------------------------------------------------------------------------
# Optional (graphs extra): betweenness + communities
# ---------------------------------------------------------------------------

def _require_networkx():
    try:
        import networkx as nx
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "this statistic needs networkx; install the 'graphs' extra "
            "(pip install -e '.[graphs]')."
        ) from e
    return nx


def _weighted_graph(A: np.ndarray, weight_attr: str, transform=lambda w: w):
    """Build a networkx graph from the upper triangle of ``A`` (weight via transform)."""
    nx = _require_networkx()
    G = nx.Graph()
    G.add_nodes_from(range(len(A)))
    ii, jj = np.triu_indices(len(A), k=1)
    for i, j in zip(ii.tolist(), jj.tolist()):
        w = A[i, j]
        if w > 0:
            G.add_edge(i, j, **{weight_attr: transform(w)})
    return G


def betweenness_centrality(adj: pd.DataFrame, normalized: bool = True) -> pd.Series:
    """Betweenness: fraction of shortest paths passing through each node.

    Shortest paths use DISTANCE ``1/|rho|`` (a strong link is a short hop), the mirror
    image of the eigenvector convention. Requires the ``graphs`` extra.
    """
    A, idx = _as_matrix(adj)
    nx = _require_networkx()
    G = _weighted_graph(A, "distance", transform=lambda w: 1.0 / w)
    bc = nx.betweenness_centrality(G, weight="distance", normalized=normalized)
    vals = np.array([bc.get(i, 0.0) for i in range(len(idx))])
    return pd.Series(vals, index=idx, name="betweenness_centrality")


def communities(adj: pd.DataFrame, method: str = "louvain", seed: int = 0) -> pd.Series:
    """Integer community label per node (``|rho|``-weighted comovement groups).

    ``method='louvain'`` uses networkx's Louvain; ``method='leiden'`` uses
    ``leidenalg`` + ``python-igraph``. Requires the ``graphs`` extra. Labels are
    arbitrary integers -- use :func:`align_labels` before comparing snapshots.
    """
    A, idx = _as_matrix(adj)
    if method == "louvain":
        nx = _require_networkx()
        G = _weighted_graph(A, "weight")
        parts = nx.community.louvain_communities(G, weight="weight", seed=seed)
    elif method == "leiden":
        parts = _leiden_partition(A, seed)
    else:
        raise ValueError(f"unknown community method {method!r}")
    label = np.full(len(idx), -1, dtype=int)
    for c, nodes in enumerate(parts):
        for node in nodes:
            label[node] = c
    return pd.Series(label, index=idx, name=f"community_{method}")


def _leiden_partition(A: np.ndarray, seed: int):  # pragma: no cover - needs leidenalg
    try:
        import igraph as ig
        import leidenalg
    except ImportError as e:
        raise ImportError(
            "communities(method='leiden') needs python-igraph + leidenalg "
            "(part of the 'graphs' extra)."
        ) from e
    ii, jj = np.triu_indices(len(A), k=1)
    edges = [(int(i), int(j)) for i, j in zip(ii, jj) if A[i, j] > 0]
    weights = [float(A[i, j]) for i, j in edges]
    g = ig.Graph(n=len(A), edges=edges)
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, weights=weights, seed=seed
    )
    return [set(c) for c in part]


def community_cohesion(adj: pd.DataFrame, method: str = "louvain", seed: int = 0) -> pd.Series:
    """Share of a node's edge weight that stays inside its own community.

    A rankable, sign-meaningful community feature: high => the stock's comovement is
    concentrated in a tight cluster (a sector/theme); low => it bridges clusters.
    Requires the ``graphs`` extra.
    """
    A, idx = _as_matrix(adj)
    labels = communities(adj, method=method, seed=seed).to_numpy()
    total = A.sum(axis=1)
    same = np.array([A[i, labels == labels[i]].sum() for i in range(len(idx))])
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(total > 0, same / total, 0.0)
    return pd.Series(frac, index=idx, name="community_cohesion")


# ---------------------------------------------------------------------------
# Cross-snapshot label alignment (scipy only)
# ---------------------------------------------------------------------------

def align_labels(prev: pd.Series, curr: pd.Series) -> pd.Series:
    """Relabel ``curr``'s community ids to best match ``prev`` on shared nodes.

    Community ids are arbitrary, so a raw diff of two snapshots' labels is mostly
    relabelling noise. This solves the assignment maximising label overlap (Hungarian
    algorithm on the contingency table) and returns ``curr`` with ids remapped, so a
    genuine migration -- not a relabelling -- shows up as a change. ``curr`` ids with
    no ``prev`` match get fresh ids beyond the existing range.
    """
    from scipy.optimize import linear_sum_assignment

    shared = prev.index.intersection(curr.index)
    if len(shared) == 0:
        return curr.copy()
    p = prev.loc[shared].to_numpy()
    c = curr.loc[shared].to_numpy()
    p_ids = np.unique(p)
    c_ids = np.unique(c)
    pidx = {int(v): i for i, v in enumerate(p_ids)}
    cidx = {int(v): i for i, v in enumerate(c_ids)}
    overlap = np.zeros((len(c_ids), len(p_ids)))
    for pv, cv in zip(p.tolist(), c.tolist()):
        overlap[cidx[int(cv)], pidx[int(pv)]] += 1.0
    rows, cols = linear_sum_assignment(-overlap)   # maximise overlap
    remap = {int(c_ids[r]): int(p_ids[cc]) for r, cc in zip(rows, cols)}
    next_id = int(max(int(p_ids.max()), int(c_ids.max()))) + 1
    out_vals = []
    for v in curr.to_numpy().tolist():
        v = int(v)
        if v not in remap:
            remap[v] = next_id
            next_id += 1
        out_vals.append(remap[v])
    return pd.Series(out_vals, index=curr.index, name=curr.name)
