"""Integration test for Subphase 3.6: the graph-factor scorecard on the harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import qer.factors.graph as GF
from qer.diagnostics.graph_scorecard import (
    cluster_sector_matrix,
    graph_scorecard,
    spanning_alpha_vs_classical,
)

COLS = {"mean_ic", "ic_t_nw", "ls_sharpe_ann", "span_alpha", "span_hac_t",
        "theta_improvement", "deflated_sharpe", "n_obs"}


def test_scorecard_table_shape_and_fields(synthetic_loader):
    gfs = [GF.get_graph_factor(n) for n in
           ("eigenvector_centrality_mst_120d", "degree_centrality_mst_120d")]
    sc = graph_scorecard(synthetic_loader, gfs, horizon=21, n_trials=108)
    assert list(sc.index) == [f.name for f in gfs]
    assert COLS <= set(sc.columns)
    # deflated Sharpe is a probability in [0, 1]
    dsr = sc["deflated_sharpe"].dropna().to_numpy()
    assert ((dsr >= 0) & (dsr <= 1)).all()
    # a populated factor has a finite IC and long-short Sharpe
    assert np.isfinite(sc.loc["eigenvector_centrality_mst_120d", "mean_ic"])
    assert np.isfinite(sc.loc["eigenvector_centrality_mst_120d", "ls_sharpe_ann"])


def test_spanning_alpha_runs_and_is_hac(synthetic_loader):
    r = spanning_alpha_vs_classical(
        synthetic_loader, GF.get_graph_factor("eigenvector_centrality_mst_120d"), horizon=21)
    assert {"alpha", "hac_t", "grs_stat", "grs_pvalue", "theta_improvement"} <= set(r)
    assert np.isfinite(r["alpha"])
    assert r["k"] >= 1                                   # regressed on >=1 classical factor
    # tangency identity holds within the scorecard path too
    assert np.isclose(r["theta_augmented"], r["theta_benchmark"] + r["theta_improvement"])


def test_cluster_sector_matrix_scores_agreement():
    idx = [f"T{i:02d}" for i in range(30)]
    sectors = pd.Series([i % 3 for i in range(30)], index=idx)
    perfect = cluster_sector_matrix(sectors.copy(), sectors)
    assert np.isclose(perfect["adjusted_rand"], 1.0)     # identical labellings
    rng = np.random.default_rng(0)
    random_lab = pd.Series(rng.integers(0, 3, 30), index=idx)
    assert cluster_sector_matrix(random_lab, sectors)["adjusted_rand"] < 0.3  # unrelated


def test_cluster_sector_confusion_builds_a_graph(synthetic_loader):
    pytest.importorskip("networkx")
    from qer.diagnostics.graph_scorecard import cluster_sector_confusion
    tickers = list(synthetic_loader.close.columns)
    sectors = pd.Series([i % 4 for i in range(len(tickers))], index=tickers)
    as_of = synthetic_loader.close.index[-1]
    out = cluster_sector_confusion(synthetic_loader, as_of, sectors, window=120)
    assert isinstance(out["confusion"], pd.DataFrame)
    assert -1.0 <= out["adjusted_rand"] <= 1.0
    assert out["n"] > 0
