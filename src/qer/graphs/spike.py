"""Subphase 3.1: the harness-reuse spike -- the cheapest de-risking of the phase.

Before any graph is built, prove that the thing Phase 3 produces (a
``date x ticker`` node-feature panel) flows through the Phase-2 harness exactly
like a classical factor. We do that by taking an existing classical factor,
wrapping *its own panel* in a :class:`~qer.factors.graph.base.PanelFactor`, and
asserting the wrapped factor reproduces the native factor's IC and decile-Sharpe
to numerical tolerance. If that holds, "a graph feature is a Factor" is not an
aspiration -- the evaluation path is already proven, and every later subphase
only has to supply a panel.

This is the gate for Subphase 3.1. Run it against the real data with::

    python -m scripts.run_graph_spike
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.diagnostics.factor_ic import compute_factor_ic
from qer.diagnostics.portfolios import factor_long_short
from qer.factors.base import get_factor
from qer.factors.graph.base import PanelFactor


def _series_max_abs_diff(a: pd.Series, b: pd.Series) -> float:
    """Max |a - b| over the union of indices.

    NaN-vs-NaN counts as agreement (0); a one-sided NaN (one series has a value,
    the other does not) is a genuine disagreement and returns ``inf`` rather than
    a swallowed NaN -- otherwise a divergent NaN pattern would make the spike's
    ``> tol`` check silently pass.
    """
    idx = a.index.union(b.index)
    aa, bb = a.reindex(idx), b.reindex(idx)
    both_nan = aa.isna() & bb.isna()
    one_nan = aa.isna() ^ bb.isna()
    diff = (aa - bb).abs()
    diff = diff.where(~both_nan, 0.0)
    diff = diff.where(~one_nan, np.inf)
    return float(diff.max()) if len(diff) else 0.0


def _sharpe(ls: pd.Series) -> float:
    sd = ls.std(ddof=1)
    return float(ls.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan")


def harness_reuse_spike(
    loader,
    factor_name: str = "momentum_12_1",
    horizons: tuple[int, ...] = (1, 5, 21),
    primary_horizon: int = 21,
    n_buckets: int = 10,
    dates=None,
) -> dict:
    """Run ``factor_name`` natively and via a ``PanelFactor`` wrapper; compare.

    Returns a dict with the per-horizon max IC differences, both decile-Sharpe
    values, and the headline ``ic_max_abs_diff`` / ``ls_sharpe_diff``.
    """
    native = get_factor(factor_name)
    wrapped = PanelFactor(
        native.compute_panel(loader),
        name=f"panelwrap_{factor_name}",
        direction=native.direction,
    )

    # Always include the long-short horizon among the IC horizons, so n_ic_dates
    # below is well defined regardless of what the caller passed.
    horizons = tuple(sorted(set(horizons) | {primary_horizon}))

    ic_native = compute_factor_ic(loader, native, horizons=horizons, dates=dates)
    ic_wrapped = compute_factor_ic(loader, wrapped, horizons=horizons, dates=dates)
    ic_diffs = {h: _series_max_abs_diff(ic_native[h], ic_wrapped[h]) for h in horizons}

    ls_native = factor_long_short(
        loader, native, n_buckets=n_buckets, horizon=primary_horizon, dates=dates
    )
    ls_wrapped = factor_long_short(
        loader, wrapped, n_buckets=n_buckets, horizon=primary_horizon, dates=dates
    )
    sharpe_native, sharpe_wrapped = _sharpe(ls_native), _sharpe(ls_wrapped)

    ic_max = max(ic_diffs.values()) if ic_diffs else 0.0
    # NaN-vs-NaN Sharpe counts as agreement; a one-sided NaN is a real mismatch.
    if np.isnan(sharpe_native) and np.isnan(sharpe_wrapped):
        sharpe_diff = 0.0
    elif np.isnan(sharpe_native) or np.isnan(sharpe_wrapped):
        sharpe_diff = float("inf")
    else:
        sharpe_diff = abs(sharpe_native - sharpe_wrapped)

    return {
        "factor": factor_name,
        "n_ic_dates": int(len(ic_native[primary_horizon])),
        "ic_diff_by_horizon": ic_diffs,
        "ic_max_abs_diff": ic_max,
        "ls_sharpe_native": sharpe_native,
        "ls_sharpe_wrapped": sharpe_wrapped,
        "ls_sharpe_diff": sharpe_diff,
    }


def assert_harness_reuse(loader, tol: float = 1e-9, **kwargs) -> dict:
    """Run the spike and assert native == wrapped to ``tol``. Returns the result."""
    res = harness_reuse_spike(loader, **kwargs)
    if res["ic_max_abs_diff"] > tol or res["ls_sharpe_diff"] > tol:
        raise AssertionError(
            "Harness-reuse spike FAILED: a PanelFactor wrapper did not reproduce "
            f"the native factor. ic_max_abs_diff={res['ic_max_abs_diff']:.3e}, "
            f"ls_sharpe_diff={res['ls_sharpe_diff']:.3e} (tol={tol:.1e}). "
            "The graph evaluation path is not equivalent to the classical one."
        )
    return res
