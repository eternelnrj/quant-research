"""Subphase 3.1: ``PanelFactor`` -- adapt a precomputed panel into a ``Factor``.

A graph feature ultimately produces the same object a classical factor does: a
``date x ticker`` matrix of node-level numbers. Wrapping that matrix in a
``Factor`` lets the entire Phase-2 evaluation harness -- IC, decile long-short,
cost, FF5 exposures, deflated Sharpe -- score it with no parallel code path.

``PanelFactor`` is deliberately minimal: it stores a panel and returns it from
``compute_panel`` (ignoring the loader, since the values are already computed).
The orientation convention is unchanged from :class:`qer.factors.base.Factor` --
the stored panel is the RAW node feature and ``direction`` (set empirically from
the IC sign for real graph features) is applied by the harness.
"""

from __future__ import annotations

import pandas as pd

from qer.factors.base import Factor


class PanelFactor(Factor):
    """Wrap an already-computed ``date x ticker`` panel as a :class:`Factor`.

    Parameters
    ----------
    panel:
        RAW (un-oriented) ``date x ticker`` factor values.
    name:
        Registry name for the factor.
    direction:
        +1 if high raw value => high expected return, -1 if sign-flipped. For a
        real graph feature this is set from the IC sign in the harness, not assumed.
    """

    def __init__(self, panel: pd.DataFrame, name: str, direction: int = 1):
        if not isinstance(panel, pd.DataFrame):
            raise TypeError("PanelFactor expects a date x ticker DataFrame")
        self.name = name
        self.direction = int(direction)
        self._panel = panel

    def compute_panel(self, loader) -> pd.DataFrame:  # noqa: ARG002 - loader unused by design
        return self._panel


# ===========================================================================
# Subphase 3.3: graph factors built through the Subphase 3.2 engine, and a
# SEPARATE registry so graph features never join the classical FACTORS zoo
# (that would contaminate the classical multiple-testing denominator). Phase 3
# tracks its own trial count via the pre-registered grid + trials ledger.
# ===========================================================================

from typing import Callable  # noqa: E402

from qer.graphs.panel import build_feature_panel  # noqa: E402
from qer.graphs.windows import rebalance_dates  # noqa: E402

GRAPH_FACTORS: dict[str, Factor] = {}


def register_graph_factor(factor: Factor) -> Factor:
    """Register a graph factor under its name (idempotent). Kept apart from the
    classical ``FACTORS`` registry on purpose."""
    if factor.name not in GRAPH_FACTORS:
        GRAPH_FACTORS[factor.name] = factor
    return factor


def get_graph_factor(name: str) -> Factor:
    return GRAPH_FACTORS[name]


def all_graph_factors() -> list[Factor]:
    return list(GRAPH_FACTORS.values())


class GraphFactor(Factor):
    """A graph node statistic as a :class:`Factor`, built through the 3.2 engine.

    Given a ``snapshot_fn(returns_window, universe) -> Series`` (build a graph from the
    window, reduce it to a per-node number), this loops the monthly rebalance grid and
    forward-fills to a daily panel -- inheriting the engine's look-ahead / survivorship
    guarantees. It is the general base for feature classes A/B/C.
    """

    def __init__(
        self,
        name: str,
        snapshot_fn: Callable[[pd.DataFrame, list], pd.Series],
        *,
        window: int = 120,
        freq: str = "M",
        direction: int = 1,
        cache_dir=None,
    ):
        self.name = name
        self.direction = int(direction)
        self._snapshot_fn = snapshot_fn
        self._window = window
        self._freq = freq
        self._cache_dir = cache_dir

    def compute_panel(self, loader) -> pd.DataFrame:
        return build_feature_panel(
            loader,
            self._snapshot_fn,
            window=self._window,
            freq=self._freq,
            cache_dir=self._cache_dir,
            name=self.name if self._cache_dir else None,
        )


class DeltaGraphFactor(GraphFactor):
    """Change in a node statistic since the previous rebuild (Delta-centrality).

    Higher turnover but often more predictive than the level -- a stock *becoming*
    more central. Computed on the rebalance grid (not daily) so the change is a genuine
    snapshot-to-snapshot move, then forward-filled to the daily panel.
    """

    def compute_panel(self, loader) -> pd.DataFrame:
        level = super().compute_panel(loader)  # daily, forward-filled
        cal = loader.close.index
        snaps = pd.DatetimeIndex(
            [t for t in rebalance_dates(cal, freq=self._freq) if t in level.index]
        )
        delta = level.reindex(snaps).diff()  # change vs previous rebuild
        return delta.reindex(cal).ffill()
