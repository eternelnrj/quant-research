"""Integration test for Subphase 3.3: a correlation-network factor through the harness.

The gate: at least one correlation factor produces a clean, finite IC profile through
the standard Phase-2 harness, and its panel is causal and non-trivial. Also checks that
graph factors stay in their own registry (never the classical ``FACTORS``) and that the
neutralisation control composes with a graph panel.
"""

from __future__ import annotations

import numpy as np

import qer.factors.graph as GF
from qer.diagnostics.factor_ic import compute_factor_ic
from qer.diagnostics.portfolios import factor_long_short
from qer.factors.base import FACTORS
from qer.graphs.panel import neutralize_cross_section

WINDOW = 120


def test_graph_registry_is_separate_from_classical():
    names = {f.name for f in GF.all_graph_factors()}
    assert "eigenvector_centrality_mst_120d" in names
    assert names.isdisjoint(set(FACTORS))          # no graph factor in the classical zoo


def test_eigenvector_factor_has_a_finite_ic_profile(synthetic_loader):
    factor = GF.get_graph_factor("eigenvector_centrality_mst_120d")

    panel = factor.compute_panel(synthetic_loader)
    assert panel.index.equals(synthetic_loader.close.index)     # daily, causal grid
    populated = panel.dropna(how="all")
    assert len(populated) > 0
    # non-trivial: real cross-sectional dispersion on a populated date
    last = populated.iloc[-1].dropna()
    assert last.std() > 0

    ic = compute_factor_ic(synthetic_loader, factor, horizons=(1, 5, 21))
    ic21 = ic[21].dropna()
    assert len(ic21) >= 5                                        # a real IC series
    assert np.isfinite(ic21.to_numpy()).all()


def test_factor_is_tradeable_through_long_short(synthetic_loader):
    factor = GF.get_graph_factor("degree_centrality_mst_120d")
    ls = factor_long_short(synthetic_loader, factor, n_buckets=5, horizon=21)
    assert len(ls.dropna()) > 0
    assert np.isfinite(ls.dropna().to_numpy()).all()


def test_delta_factor_produces_a_panel(synthetic_loader):
    factor = GF.get_graph_factor("delta_eigenvector_centrality_mst_120d")
    panel = factor.compute_panel(synthetic_loader)
    vals = panel.to_numpy()
    assert np.isfinite(vals).any()
    # a change series straddles zero (some names rise, some fall between rebuilds)
    finite = vals[np.isfinite(vals)]
    assert finite.min() < 0 < finite.max()


def test_neutralisation_composes_with_a_graph_panel(synthetic_loader):
    eig = GF.get_graph_factor("eigenvector_centrality_mst_120d").compute_panel(synthetic_loader)
    deg = GF.get_graph_factor("degree_centrality_mst_120d").compute_panel(synthetic_loader)
    # neutralise eigenvector centrality against degree (a size/connectivity proxy)
    resid = neutralize_cross_section(eig, by={"degree": deg}, rank=True, min_names=10)
    common = eig.dropna(how="all").index.intersection(resid.dropna(how="all").index)
    assert len(common) > 0                                        # produced residuals
    # residual is not identical to the raw feature (something was removed)
    a = eig.loc[common].to_numpy()
    b = resid.loc[common].to_numpy()
    assert not np.allclose(np.nan_to_num(a), np.nan_to_num(b))
