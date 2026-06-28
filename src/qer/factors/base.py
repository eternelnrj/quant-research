"""
Phase 2: the Factor interface and registry.

Every factor implements a *vectorised* ``compute_panel(loader)`` that returns the
full ``date x ticker`` matrix of raw factor values in one pass. The per-date
``compute(loader, as_of)`` is a thin slice of that panel, for notebooks and
one-off use.

Why panel-first: the evaluation harness needs the factor at many dates. Building
it per date (re-slicing and recomputing each time) is O(n_dates) of repeated
work and was the scaling bottleneck flagged in the architecture review. Almost
every cross-sectional factor is a rolling/shift transform of a price/return/
volume panel, which pandas computes for *all* dates at once - and such transforms
are look-ahead-safe automatically, since the value at row t references only rows
<= t.

Orientation convention: ``compute_panel`` returns the RAW economic quantity
(e.g. momentum return, volatility, market cap). The ``direction`` attribute
(+1 / -1) is what turns it into "high score => expected high return"; the harness
applies it, so the raw panel stays interpretable.
"""

# NEW

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Factor(ABC):
    """Base class for a cross-sectional alpha factor.

    Subclasses set ``name`` and ``direction`` and implement ``compute_panel``.
    """

    name: str
    direction: int  # +1 if high raw score => high expected return, -1 if sign-flipped

    @abstractmethod
    def compute_panel(self, loader) -> pd.DataFrame:
        """Full ``date x ticker`` RAW factor panel, vectorised over all dates.

        Must be look-ahead-safe: the value at row t may use only data through
        row t (rolling/shift transforms satisfy this by construction).
        """

    def compute(self, loader, as_of_date) -> pd.Series:
        """RAW cross-section on a single date (most recent row at/<= as_of).

        Convenience for notebooks and one-off use. Builds the full panel then
        slices it, so do NOT call this in a date loop - use
        :func:`compute_factor_panel` (a single vectorised build) for evaluation.
        """
        panel = self.compute_panel(loader)
        as_of = pd.Timestamp(as_of_date)
        sub = panel.loc[:as_of]
        if sub.empty:
            return pd.Series(index=panel.columns, dtype=float, name=self.name)
        out = sub.iloc[-1].copy()
        out.name = self.name
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Factor {self.name!r} dir={self.direction:+d}>"


# ---------------------------------------------------------------------------
# Registry — one canonical, fixed-parameter instance per factor.
# Keeping a single named spec per factor is the structural guard against
# p-hacking via parameter variants.
# ---------------------------------------------------------------------------

FACTORS: dict[str, Factor] = {}


def register(factor: Factor) -> Factor:
    """Register a factor instance under its name (idempotent across re-imports)."""
    if factor.name not in FACTORS:
        FACTORS[factor.name] = factor
    return factor


def get_factor(name: str) -> Factor:
    return FACTORS[name]


def all_factors() -> list[Factor]:
    return list(FACTORS.values())


# ---------------------------------------------------------------------------
# The efficient evaluation entry point.
# ---------------------------------------------------------------------------


def compute_factor_panel(
    loader,
    factor: Factor,
    dates=None,
    oriented: bool = False,
) -> pd.DataFrame:
    """``date x ticker`` factor panel, built in ONE vectorised pass.

    Calls ``factor.compute_panel(loader)`` exactly once instead of looping per
    date. Pass ``dates`` to restrict the rows returned; pass ``oriented=True``
    to multiply by ``factor.direction`` (so high => expected high return).
    """
    panel = factor.compute_panel(loader)
    if oriented:
        panel = panel * factor.direction
    if dates is not None:
        # idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
        # panel = panel.reindex(idx)
        panel = panel.reindex(pd.DatetimeIndex(dates))  # MODIF

    return panel
