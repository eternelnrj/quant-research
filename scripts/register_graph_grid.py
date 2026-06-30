"""CLI: pre-register the Phase 3 configuration grid.

Thin wrapper around ``qer.graphs.grid.register_grid`` - the grid definition and
registration logic live in the package so notebooks and the harness can import
them too. This wrapper registers the default grid (write-once) and prints the
trial-count denominator that later significance tests are fed.

Usage:
    python -m scripts.register_graph_grid
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from qer.config import GRAPH_GRID_FILE
from qer.graphs.grid import DEFAULT_GRID, register_grid


def main() -> pd.DataFrame:
    spec = DEFAULT_GRID
    df = register_grid(spec)
    by_class = df["feature_class"].value_counts().to_dict()
    print(f"Registered grid at {GRAPH_GRID_FILE}")
    print(f"  total configurations (N_trials): {len(df)}")
    for cls, n in sorted(by_class.items()):
        print(f"    {cls:12s}: {n}")
    print("  spec:", json.dumps(asdict(spec)))
    return df


if __name__ == "__main__":
    main()
