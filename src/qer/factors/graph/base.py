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
