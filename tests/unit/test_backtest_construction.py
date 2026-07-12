"""Unit tests for Phase 4.3 sizing and constraints (pure functions, no loader).

Each constraint is tested *in isolation* (it enforces its own limit); the single-pass
pipeline is only tested for idempotency on a feasible book and for running end to end,
since joint satisfaction is a Phase 5 (optimiser) concern.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.backtest.constraints import (
    apply_constraints,
    beta_neutral,
    cap_positions,
    dollar_neutral,
    neutralise_sectors,
    renormalise,
)
from qer.backtest.sizing import risk_weight


def test_cap_positions_bounds_size():
    out = cap_positions(pd.Series({"A": 0.4, "B": -0.6, "C": 0.1}), 0.35)
    assert out.abs().max() <= 0.35 + 1e-12
    assert out["A"] == 0.35 and out["B"] == -0.35 and out["C"] == 0.1


def test_dollar_neutral_zeroes_net():
    out = dollar_neutral(pd.Series({"A": 0.5, "B": 0.3, "C": -0.3}))
    assert abs(float(out.sum())) < 1e-12


def test_neutralise_sectors_within_cap():
    w = pd.Series({"A": 0.4, "B": 0.3, "C": -0.5, "D": -0.6})
    sec = pd.Series({"A": "tech", "B": "tech", "C": "fin", "D": "fin"})
    for s in ("tech", "fin"):
        net0 = float(neutralise_sectors(w, sec, cap=0.0)[sec[sec == s].index].sum())
        assert abs(net0) < 1e-12                                  # cap 0 -> exactly neutral
        net1 = float(neutralise_sectors(w, sec, cap=0.1)[sec[sec == s].index].sum())
        assert abs(net1) <= 0.1 + 1e-12                           # cap 0.1 -> within cap


def test_neutralise_sectors_ignores_unlabelled():
    w = pd.Series({"A": 0.5, "B": -0.5, "X": 0.3})
    out = neutralise_sectors(w, pd.Series({"A": "tech", "B": "tech"}), cap=0.0)
    assert out["X"] == 0.3                                        # no label -> untouched


def test_beta_neutral_zeroes_portfolio_beta():
    w = pd.Series({"A": 0.4, "B": 0.3, "C": -0.5, "D": -0.6})
    b = pd.Series({"A": 1.2, "B": 0.8, "C": 1.5, "D": 0.5})
    assert abs(float((beta_neutral(w, b) * b).sum())) < 1e-10


def test_renormalise_sets_unit_gross():
    assert abs(float(renormalise(pd.Series({"A": 0.2, "B": -0.3})).abs().sum()) - 1.0) < 1e-12


def test_risk_weight_favours_low_vol():
    sig = pd.Series({f"T{i}": float(i) for i in range(10)})       # T9 top, T8 next
    vol = pd.Series({f"T{i}": 0.02 for i in range(10)})
    vol["T9"], vol["T8"] = 0.01, 0.04                            # both long; T9 low vol, T8 high
    rw = risk_weight(sig, vol, n_buckets=5)                       # n_side = 2
    assert rw["T9"] > rw["T8"]                                    # lower vol -> larger position
    assert abs(float(rw.sum())) < 1e-12                           # dollar-neutral
    assert abs(float(rw.abs().sum()) - 1.0) < 1e-12              # unit gross


def test_apply_constraints_idempotent_on_feasible():
    feasible = pd.Series({"A": 0.25, "B": 0.25, "C": -0.25, "D": -0.25})
    assert np.allclose(apply_constraints(feasible, max_position=0.5).values, feasible.values)


def test_apply_constraints_runs_full_pipeline_to_unit_gross():
    w = pd.Series({"A": 0.6, "B": 0.1, "C": -0.4, "D": -0.3})
    sec = pd.Series({"A": "t", "B": "t", "C": "f", "D": "f"})
    b = pd.Series({"A": 1.1, "B": 0.9, "C": 1.3, "D": 0.7})
    out = apply_constraints(w, max_position=0.4, sectors=sec, sector_cap=0.0, betas=b)
    assert np.isfinite(out.to_numpy()).all()
    assert abs(float(out.abs().sum()) - 1.0) < 1e-9              # renormalised to unit gross
