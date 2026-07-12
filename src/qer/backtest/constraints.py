"""Phase 4.3: rules-based portfolio constraints (clip / project / renormalise).

Each constraint is a pure map on a signed weight vector that enforces *its own*
limit: :func:`cap_positions` bounds single-name size, :func:`dollar_neutral` zeroes
net exposure, :func:`neutralise_sectors` bounds per-sector net, :func:`beta_neutral`
zeroes portfolio market beta. :func:`apply_constraints` chains them in a fixed order
and renormalises to unit gross.

This is deliberately the *rules-based* version: a single sequential pass, so the
neutralisations partially undo one another and the constraints hold jointly only
approximately (each holds exactly in isolation, and the pipeline is idempotent on an
already-feasible book). Phase 5 replaces this step with a joint ``cvxpy`` projection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.backtest.sizing import equal_weight, risk_weight, rolling_vol, signal_weight


# ---------------------------------------------------------------------------
# Constraint primitives (each a pure map on a weight Series)
# ---------------------------------------------------------------------------

def cap_positions(w, max_weight: float) -> pd.Series:
    """Clip every position to ``|w_i| <= max_weight``."""
    return pd.Series(w, dtype=float).clip(lower=-max_weight, upper=max_weight)


def dollar_neutral(w) -> pd.Series:
    """Demean so ``sum(w) = 0`` (dollar-neutral)."""
    w = pd.Series(w, dtype=float)
    return w - w.mean()


def neutralise_sectors(w, sectors, cap: float = 0.0) -> pd.Series:
    """Bring each sector's net weight within ``+/- cap`` (0.0 = fully sector-neutral).

    Subtracts a per-sector constant so the sector net becomes ``clip(net, -cap, cap)``.
    Names with a missing sector label are left untouched.
    """
    w = pd.Series(w, dtype=float).copy()
    sec = pd.Series(sectors).reindex(w.index)
    for _, members in w.groupby(sec).groups.items():
        members = list(members)
        net = float(w[members].sum())
        target = float(np.clip(net, -cap, cap))
        w.loc[members] = w[members] - (net - target) / len(members)
    return w


def beta_neutral(w, betas) -> pd.Series:
    """Project out net market beta so the portfolio beta ``sum(w_i * beta_i) = 0``."""
    w = pd.Series(w, dtype=float)
    b = pd.Series(betas).reindex(w.index).astype(float).fillna(0.0)
    denom = float((b * b).sum())
    if denom == 0.0:
        return w
    return w - (float((w * b).sum()) / denom) * b


def renormalise(w, gross: float = 1.0) -> pd.Series:
    """Scale to a target gross exposure ``sum(|w|) = gross`` (scale-invariant to neutrality)."""
    w = pd.Series(w, dtype=float)
    g = float(w.abs().sum())
    return w * (gross / g) if g > 0 else w


def apply_constraints(w, *, max_position: float | None = None, sectors=None,
                      sector_cap: float = 0.0, betas=None, gross: float = 1.0) -> pd.Series:
    """Chain the constraints in the documented order and renormalise to ``gross``."""
    w = pd.Series(w, dtype=float).copy()
    if max_position is not None:
        w = cap_positions(w, max_position)
    w = dollar_neutral(w)
    if sectors is not None:
        w = neutralise_sectors(w, sectors, sector_cap)
    if betas is not None:
        w = beta_neutral(w, betas)
    return renormalise(w, gross)


# ---------------------------------------------------------------------------
# Rolling market beta (reuses the trailing cov/var pattern from the factor layer)
# ---------------------------------------------------------------------------

def rolling_beta(loader, window: int = 252, kind: str = "log") -> pd.DataFrame:
    """Trailing single-index beta of each name vs the market (date x ticker).

    ``beta = cov(r_i, r_m) / var(r_m)`` over the trailing ``window``, with all moments
    on the same (divide-by-window) convention so the ratio is consistent.
    """
    r = loader.get_returns(kind)
    m = loader.market_return.reindex(r.index)
    e_m = m.rolling(window).mean()
    var_m = (m * m).rolling(window).mean() - e_m * e_m
    cov = r.mul(m, axis=0).rolling(window).mean() - r.rolling(window).mean().mul(e_m, axis=0)
    return cov.div(var_m.replace(0.0, np.nan), axis=0)


# ---------------------------------------------------------------------------
# Compose sizing + constraints into an engine weigher
# ---------------------------------------------------------------------------

def make_weigher(scheme: str = "equal", *, n_buckets: int = 10, max_position: float | None = None,
                 sectors=None, sector_cap: float = 0.0, beta_neutralise: bool = False,
                 vol_window: int = 63, beta_window: int = 252):
    """Build a ``weigher(signal_row, as_of, loader)`` that sizes then constrains.

    Suitable for :class:`~qer.backtest.engine.Backtest`'s ``weigher`` hook. The rolling
    vol/beta panels are computed once (lazily) and cached across rebalances.
    """
    cache: dict = {}

    def weigher(row, as_of, loader):
        # Cache the rolling panels once per loader (setdefault would recompute them
        # every call, since its default argument is always evaluated).
        if cache.get("loader_id") != id(loader):
            cache.clear()
            cache["loader_id"] = id(loader)

        if scheme == "risk":
            if "vol" not in cache:
                cache["vol"] = rolling_vol(loader, vol_window)
            w = risk_weight(row, cache["vol"].loc[as_of], n_buckets)
        elif scheme == "signal":
            w = signal_weight(row)
        else:
            w = equal_weight(row, n_buckets)

        betas = None
        if beta_neutralise:
            if "beta" not in cache:
                cache["beta"] = rolling_beta(loader, beta_window)
            beta_panel = cache["beta"]
            betas = beta_panel.loc[as_of] if as_of in beta_panel.index else None

        return apply_constraints(w, max_position=max_position, sectors=sectors,
                                 sector_cap=sector_cap, betas=betas)

    return weigher
