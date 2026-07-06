"""Integration test for Subphase 3.4: lead-lag factors through the engine and harness.

On structureless synthetic data the lead-lag graph is (correctly) sparse, so this test checks
the plumbing -- the factor builds a causal daily panel and runs through the Phase-2 harness
without error -- rather than demanding a strong IC (the honest null is exercised at unit level
in ``test_graph_leadlag.py``). It also confirms the factors stay in the graph-only registry.
"""

from __future__ import annotations

import numpy as np

import qer.factors.graph as GF
from qer.diagnostics.factor_ic import compute_factor_ic
from qer.diagnostics.portfolios import factor_long_short
from qer.factors.base import FACTORS


def test_leadlag_factors_registered_separately():
    names = {f.name for f in GF.all_graph_factors()}
    assert {"leadlag_out_degree_120d", "leadlag_in_degree_120d", "leadlag_upstream_120d"} <= names
    assert names.isdisjoint(set(FACTORS))  # never in the classical zoo


def test_leadlag_factor_builds_and_scores(synthetic_loader):
    factor = GF.get_graph_factor("leadlag_upstream_120d")

    panel = factor.compute_panel(synthetic_loader)
    assert panel.index.equals(synthetic_loader.close.index)  # daily, causal grid
    assert panel.dropna(how="all").shape[0] > 0

    # the harness runs and returns an IC series per horizon (finite where defined)
    ic = compute_factor_ic(synthetic_loader, factor, horizons=(1, 5, 21))
    assert set(ic.keys()) == {1, 5, 21}
    finite = ic[21].dropna().to_numpy()
    assert np.isfinite(finite).all()

    # and the factor is tradeable through the long-short without error
    ls = factor_long_short(synthetic_loader, factor, n_buckets=5, horizon=21)
    assert np.isfinite(ls.dropna().to_numpy()).all()


def test_leadlag_degree_factors_build(synthetic_loader):
    for name in ("leadlag_out_degree_120d", "leadlag_in_degree_120d"):
        panel = GF.get_graph_factor(name).compute_panel(synthetic_loader)
        assert panel.index.equals(synthetic_loader.close.index)
