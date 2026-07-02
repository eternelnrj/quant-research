"""Subphase 3.2: the feature-panel engine and cross-sectional neutralisation.

One engine serves every graph feature. For each monthly rebalance date it builds
a point-in-time trailing window, calls a ``snapshot_fn(returns_window, universe)
-> Series`` to get node-level numbers, assembles a ``snapshot x ticker`` frame,
then reindexes to the daily calendar and forward-fills between rebuilds. So
look-ahead, survivorship and optional caching are implemented and tested once,
and a new feature is just a different ``snapshot_fn``.

``neutralize_cross_section`` is the shared control run before any portfolio: it
returns the per-date residual of the (ranked) feature after regressing out
characteristic controls and sector dummies, so a "relational" signal that is
really a size/liquidity/sector tilt collapses to zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from qer.graphs.windows import rebalance_dates, trailing_return_matrix


def _cache_ns(cache_dir, name, window, freq, min_obs, kind) -> Path | None:
    if cache_dir is None:
        return None
    if not name:
        raise ValueError("cache_dir requires a `name` to namespace the cache")
    ns = Path(cache_dir) / f"{name}__w{window}_{freq}_m{min_obs}_{kind}"
    ns.mkdir(parents=True, exist_ok=True)
    return ns


def _cache_path(ns: Path | None, t) -> Path | None:
    return None if ns is None else ns / f"{pd.Timestamp(t):%Y%m%d}.parquet"


def build_feature_panel(
    loader,
    snapshot_fn: Callable[[pd.DataFrame, list], pd.Series],
    *,
    window: int = 120,
    freq: str = "M",
    min_obs: int | None = None,
    kind: str = "log",
    start=None,
    end=None,
    cache_dir=None,
    name: str | None = None,
) -> pd.DataFrame:
    """Loop rebalance snapshots -> per-name Series -> daily forward-filled panel.

    ``cache_dir`` (with a config-specific ``name``) caches each snapshot Series;
    the cache key encodes ``window/freq/min_obs/kind`` so a parameter change never
    reads a stale snapshot. The daily panel is forward-filled between rebuilds;
    dates before the first snapshot are NaN (you do not have a graph yet). Names
    that drop out of a later snapshot keep a forward-filled value, which the
    harness removes via its per-date universe mask.
    """
    cal = loader.close.index
    if min_obs is None:
        min_obs = window
    snaps = rebalance_dates(cal, freq=freq, start=start, end=end)
    ns = _cache_ns(cache_dir, name, window, freq, min_obs, kind)

    rows: dict[pd.Timestamp, pd.Series] = {}
    for t in snaps:
        cp = _cache_path(ns, t)
        if cp is not None and cp.exists():
            rows[t] = pd.read_parquet(cp).iloc[:, 0]
            continue
        rw = trailing_return_matrix(loader, t, window=window, min_obs=min_obs, kind=kind)
        if rw.shape[1] == 0:
            continue
        s = pd.Series(snapshot_fn(rw, list(rw.columns)))
        if cp is not None:
            s.to_frame("value").to_parquet(cp)
        rows[t] = s

    if not rows:
        return pd.DataFrame(index=cal)
    snap_panel = pd.DataFrame(rows).T
    snap_panel.index = pd.DatetimeIndex(snap_panel.index)
    snap_panel = snap_panel.sort_index()
    return snap_panel.reindex(cal).ffill()


def _ranked(s: pd.Series) -> pd.Series:
    return s.rank()


def neutralize_cross_section(
    feature: pd.DataFrame,
    by: dict[str, pd.DataFrame] | None = None,
    sectors=None,
    rank: bool = True,
    min_names: int = 10,
) -> pd.DataFrame:
    """Per-date OLS residual of ``feature`` after partialling out controls.

    ``by``      : numeric control panels, ``{name: date x ticker}`` (e.g. log market
                  cap, Amihud illiquidity). Rank-transformed when ``rank=True``.
    ``sectors`` : ticker->label Series (static) or ``date x ticker`` label frame,
                  entered as dummy columns (i.e. sector-demeaning).
    Dates with fewer than ``min_names`` usable names are left NaN. A feature that
    is an exact (monotone) function of a control neutralises to ~0.
    """
    by = by or {}
    out = pd.DataFrame(np.nan, index=feature.index, columns=feature.columns)

    for t in feature.index:
        y_full = feature.loc[t].dropna()
        if y_full.empty:
            continue
        cols = [c for c in feature.columns if c in y_full.index]
        design = {}
        for cname, panel in by.items():
            if t in panel.index:
                design[cname] = panel.loc[t]
        # align on names present in y and in every numeric control
        names = pd.Index(cols)
        for cname, cser in design.items():
            names = names.intersection(cser.dropna().index)
        names = names.intersection(y_full.index)
        if len(names) < min_names:
            continue

        y = y_full.loc[names]
        y = _ranked(y) if rank else y
        Xcols = [np.ones(len(names))]
        for cname in design:
            col = design[cname].loc[names]
            col = _ranked(col) if rank else col
            Xcols.append(col.to_numpy(dtype=float))
        # sector dummies (drop first level to avoid collinearity with the intercept)
        if sectors is not None:
            lab = sectors.loc[t] if isinstance(sectors, pd.DataFrame) and t in sectors.index else sectors
            if isinstance(lab, pd.Series):
                lab = lab.reindex(names)
                dummies = pd.get_dummies(lab, drop_first=True, dtype=float)
                for c in dummies.columns:
                    Xcols.append(dummies[c].to_numpy(dtype=float))

        X = np.column_stack(Xcols)
        yv = y.to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        out.loc[t, names] = resid

    return out
