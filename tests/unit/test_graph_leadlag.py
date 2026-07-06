"""Unit tests for the Subphase 3.4 lead-lag engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.graphs.leadlag import (
    edge_density,
    in_degree,
    leadlag_density_report,
    leadlag_edges,
    out_degree,
    residualize_market,
    upstream_signal,
)

P = 8
NAMES = [f"N{i}" for i in range(P)]


def _planted(W=250, seed=0):
    """Independent residuals with two planted lead-lag edges: 0->1 and 2->3 (lag 1)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(W, P))
    E = base.copy()
    E[1:, 1] = 0.9 * base[:-1, 0] + 0.2 * base[1:, 1]   # 0 leads 1
    E[1:, 3] = 0.9 * base[:-1, 2] + 0.2 * base[1:, 3]   # 2 leads 3
    return pd.DataFrame(E, columns=NAMES)


def test_residualize_removes_the_market():
    rng = np.random.default_rng(1)
    W = 200
    market = rng.normal(0, 0.01, W)
    betas = rng.uniform(0.5, 1.5, P)
    idio = rng.normal(0, 0.01, (W, P))
    R = pd.DataFrame(market[:, None] * betas[None, :] + idio, columns=NAMES)
    E = residualize_market(R)
    # residuals are (near) orthogonal to the equal-weighted market
    m = R.mean(axis=1).to_numpy()
    for c in NAMES:
        assert abs(np.corrcoef(E[c].to_numpy(), m)[0, 1]) < 0.15


def test_edges_are_directional():
    D = leadlag_edges(_planted(), alpha=0.10)
    assert D.loc["N0", "N1"] > 0            # 0 leads 1
    assert D.loc["N1", "N0"] == 0           # not the reverse
    assert D.loc["N2", "N3"] > 0            # 2 leads 3
    assert D.loc["N3", "N2"] == 0


def test_no_self_edges_and_sparse_on_noise():
    rng = np.random.default_rng(2)
    noise = pd.DataFrame(rng.normal(size=(250, P)), columns=NAMES)
    D = leadlag_edges(noise, alpha=0.10)
    assert np.allclose(np.diag(D.to_numpy()), 0.0)           # no self-edges
    assert edge_density(D.to_numpy()) < 0.10                 # BH keeps false edges rare


def test_degrees_identify_leaders_and_followers():
    D = leadlag_edges(_planted(), alpha=0.10)
    od, idg = out_degree(D), in_degree(D)
    assert od["N0"] > 0 and od["N2"] > 0                     # leaders lead
    assert od[["N1", "N3"]].max() == 0                        # followers do not lead
    assert idg["N1"] > 0 and idg["N3"] > 0                    # followers are led
    assert idg[["N0", "N2"]].max() == 0                       # leaders are not led


def test_upstream_signal_follows_leaders():
    E = _planted()
    D = leadlag_edges(E, alpha=0.10)
    # force leader N0's recent residual returns strongly positive
    E2 = E.copy()
    E2.iloc[-5:, E2.columns.get_loc("N0")] = 1.0
    up = upstream_signal(D, E2, lookback=5)
    assert up["N1"] > 0                                       # follower expected up
    # a stock with no leaders gets exactly zero
    assert up["N5"] == 0.0


def test_honest_null_flags_planted_structure():
    report = leadlag_density_report(_planted(), alpha=0.10, n_shuffles=100, seed=0)
    assert report["n_edges"] >= 2
    assert report["actual_density"] > report["null_q95"]     # clearly above the null
    assert report["p_value"] < 0.05


def test_honest_null_reports_noise_as_null():
    rng = np.random.default_rng(3)
    noise = pd.DataFrame(rng.normal(size=(250, P)), columns=NAMES)
    report = leadlag_density_report(noise, alpha=0.10, n_shuffles=100, seed=0)
    assert report["p_value"] > 0.20                          # indistinguishable from the null
