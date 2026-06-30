"""Subphase 3.1: the pre-registered configuration grid.

The number of distinct feature configurations you *try* is the denominator of
every later significance claim. Reporting the survivors while silently discarding
the rest is exactly the factor-zoo trap the harness exists to prevent, so Phase 3
fixes the grid in writing -- before any forward return is looked at -- and feeds
its *total* size to the deflated Sharpe and the Bonferroni/BH step.

This module enumerates that grid deterministically and registers it to
``data/graphs/grid.parquet``. Registration is write-once: re-registering a grid
whose configuration set differs from the one on disk raises, so the pre-registered
denominator cannot quietly drift as the project evolves.

Each configuration is a small dict tagged with its ``feature_class`` and the axes
relevant to that class. The grid is intentionally heterogeneous -- a correlation
centrality has a window and a sparsifier, a lead-lag feature has a lag, a text
feature has a k -- because that is the honest shape of "the grid actually run".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product

import pandas as pd

from qer.config import GRAPH_GRID_FILE


@dataclass(frozen=True)
class GridSpec:
    """The axes of the Phase 3 configuration space.

    Defaults are the pre-registered grid from the implementation plan. Treat any
    change to these as a *new* grid (it changes the multiple-testing denominator),
    not an edit to the existing one.
    """

    windows: tuple[int, ...] = (120, 180, 250)
    sparsifiers: tuple[str, ...] = ("threshold", "mst", "pmfg")
    centralities: tuple[str, ...] = ("degree", "eigenvector", "betweenness", "clustering")
    centrality_transforms: tuple[str, ...] = ("level", "delta")
    communities: tuple[str, ...] = ("louvain", "leiden")
    leadlag_lags: tuple[int, ...] = (1, 2, 3, 4, 5)
    leadlag_features: tuple[str, ...] = ("in_degree", "out_degree", "upstream")
    knn: tuple[int, ...] = (5, 10, 20)
    text_features: tuple[str, ...] = ("neighbour_return",)


DEFAULT_GRID = GridSpec()


def _config_id(cfg: dict) -> str:
    """Stable 12-hex-char id from a config's canonical JSON (order-independent)."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _with_id(cfg: dict) -> dict:
    return {"config_id": _config_id(cfg), **cfg}


def enumerate_configs(spec: GridSpec = DEFAULT_GRID) -> list[dict]:
    """Deterministic, flat list of every configuration in the grid.

    Order is fixed (correlation, then lead-lag, then text) so the registered
    grid is reproducible across runs and machines.
    """
    configs: list[dict] = []

    # --- Feature class A: correlation networks -----------------------------
    # centralities: window x sparsifier x centrality x {level, delta}
    for w, sp, c, tr in product(
        spec.windows, spec.sparsifiers, spec.centralities, spec.centrality_transforms
    ):
        configs.append(
            {
                "feature_class": "correlation",
                "feature": c,
                "transform": tr,
                "window": w,
                "sparsifier": sp,
            }
        )
    # communities: window x sparsifier x community method
    for w, sp, comm in product(spec.windows, spec.sparsifiers, spec.communities):
        configs.append(
            {
                "feature_class": "correlation",
                "feature": f"community_{comm}",
                "transform": "level",
                "window": w,
                "sparsifier": sp,
            }
        )

    # --- Feature class B: lead-lag networks --------------------------------
    # lag x {in_degree, out_degree, upstream}
    for lag, feat in product(spec.leadlag_lags, spec.leadlag_features):
        configs.append(
            {
                "feature_class": "leadlag",
                "feature": feat,
                "lag": lag,
            }
        )

    # --- Feature class C: text-similarity networks -------------------------
    for k, feat in product(spec.knn, spec.text_features):
        configs.append(
            {
                "feature_class": "text",
                "feature": feat,
                "knn": k,
            }
        )

    return [_with_id(c) for c in configs]


def grid_frame(spec: GridSpec = DEFAULT_GRID) -> pd.DataFrame:
    """The enumerated grid as a tidy DataFrame (one row per configuration)."""
    df = pd.DataFrame(enumerate_configs(spec))
    # config_id first, then class/feature, then the (sparse) axis columns
    lead = ["config_id", "feature_class", "feature"]
    rest = [c for c in df.columns if c not in lead]
    return df[lead + rest]


def register_grid(
    spec: GridSpec = DEFAULT_GRID, path=GRAPH_GRID_FILE, overwrite: bool = False
) -> pd.DataFrame:
    """Write the grid to ``path`` once; refuse to silently change it later.

    If ``path`` already holds a grid with the *same* set of ``config_id``s, this
    is a no-op and the existing frame is returned. If the set differs, it raises
    unless ``overwrite=True`` -- a pre-registered denominator must not drift by
    accident. Returns the registered DataFrame.
    """
    new = grid_frame(spec)
    if path.exists() and not overwrite:
        existing = pd.read_parquet(path)
        if set(existing["config_id"]) != set(new["config_id"]):
            raise ValueError(
                f"Grid at {path} differs from the spec being registered "
                f"({len(existing)} configs on disk vs {len(new)} requested). "
                "Pre-registration is write-once; pass overwrite=True only if you "
                "deliberately intend a new grid (it changes the trial-count "
                "denominator for every significance test)."
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    new = new.assign(registered_at=pd.Timestamp.now("UTC"))
    new.to_parquet(path)
    return new


def load_grid(path=GRAPH_GRID_FILE) -> pd.DataFrame:
    """Read the registered grid, with a clear error if it was never registered."""
    if not path.exists():
        raise FileNotFoundError(
            f"No pre-registered grid at {path}. Run `make graphs-register` "
            "(or qer.graphs.grid.register_grid()) before evaluating any feature."
        )
    return pd.read_parquet(path)


def n_trials(spec_or_path=DEFAULT_GRID) -> int:
    """The honest trial count N -- the *total grid size*, not the survivor count.

    Pass a :class:`GridSpec` to count the spec, or a path to count what was
    registered on disk. This is the N to feed the deflated Sharpe and the
    Bonferroni/BH correction.
    """
    if isinstance(spec_or_path, GridSpec):
        return len(enumerate_configs(spec_or_path))
    return len(load_grid(spec_or_path))
