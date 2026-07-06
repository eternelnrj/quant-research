"""Subphase 3.4: lead-lag network factors (feature class B).

Three directed-network features, each built through the Subphase 3.2 engine and registered
in the separate ``GRAPH_FACTORS`` registry:

* ``leadlag_out_degree`` -- how strongly a stock *leads* others (a leader);
* ``leadlag_in_degree``  -- how strongly a stock is *led* by others (a follower);
* ``leadlag_upstream``   -- the edge-weighted recent residual return of a stock's leaders,
  the genuinely *predictive* feature of the three.

All are numpy/scipy only (no optional extra). Orientation is empirical: ``direction`` defaults
to +1 and is set from the observed IC sign during evaluation (Subphase 3.6). Because lead-lag is
weak in liquid large caps, these factors are expected to be the phase's most likely honest
failure/decay case; :func:`qer.graphs.leadlag.leadlag_density_report` is the accompanying gate.
"""

from __future__ import annotations

import pandas as pd

from qer.factors.graph.base import GraphFactor, register_graph_factor
from qer.graphs.leadlag import (
    in_degree,
    leadlag_edges,
    out_degree,
    residualize_market,
    upstream_signal,
)

_LAGS = (1, 2, 3, 4, 5)


def make_leadlag_snapshot_fn(stat: str, *, lags=_LAGS, alpha: float = 0.10,
                             min_names: int = 10, lookback: int = 5):
    """Compose a ``snapshot_fn``: returns-window -> per-ticker lead-lag statistic.

    Residualises the market, builds the directed edge set, and reduces it to ``stat`` in
    {``out_degree``, ``in_degree``, ``upstream``}. Snapshots with fewer than ``min_names``
    names return an empty Series (the engine skips them).
    """
    if stat not in {"out_degree", "in_degree", "upstream"}:
        raise ValueError(f"unknown lead-lag stat {stat!r}")

    def snapshot_fn(returns_window: pd.DataFrame, universe: list) -> pd.Series:  # noqa: ARG001
        if returns_window.shape[1] < min_names:
            return pd.Series(dtype=float)
        E = residualize_market(returns_window)
        D = leadlag_edges(E, lags=lags, alpha=alpha)
        if stat == "out_degree":
            return out_degree(D)
        if stat == "in_degree":
            return in_degree(D)
        return upstream_signal(D, E, lookback=lookback)

    return snapshot_fn


def make_leadlag_factor(stat: str, window: int = 120, **kwargs) -> GraphFactor:
    fn = make_leadlag_snapshot_fn(stat, **kwargs)
    return GraphFactor(f"leadlag_{stat}_{window}d", fn, window=window)


def _register_class_b(window: int = 120) -> None:
    for stat in ("out_degree", "in_degree", "upstream"):
        register_graph_factor(make_leadlag_factor(stat, window))


_register_class_b()
