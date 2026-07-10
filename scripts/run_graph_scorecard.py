"""CLI: score every registered graph factor for incremental value over the classical set.

Thin wrapper around ``qer.diagnostics.graph_scorecard.graph_scorecard`` - the evaluation
logic lives in the package so notebooks can import it too. This wrapper collects the graph
factors, feeds the deflated Sharpe the pre-registered grid's *total* trial count, builds the
table, writes the CSV, and prints it.

Usage:
    python -m scripts.run_graph_scorecard
"""

from __future__ import annotations

import pandas as pd

from qer.config import FACTOR_REPORT_DIR, GRAPH_GRID_FILE
from qer.data.loader import DataLoader
from qer.diagnostics.graph_scorecard import graph_scorecard
from qer.factors.graph import all_graph_factors
from qer.graphs.grid import n_trials


def main(horizon: int = 21, n_buckets: int = 10) -> pd.DataFrame:
    factors = all_graph_factors()
    if not factors:
        print(
            "No graph factors registered. Optional-dependency factors (betweenness, "
            "communities) register only when the `graphs` extra is importable, and the "
            "text factor only when its embedding cache exists (run `make graphs-text`)."
        )
        return pd.DataFrame()

    # Honest trial count for the deflated Sharpe: the *total* pre-registered grid size
    # (what was registered on disk if present, else the default spec) - never the winners.
    trials = n_trials(GRAPH_GRID_FILE) if GRAPH_GRID_FILE.exists() else n_trials()

    loader = DataLoader()
    table = graph_scorecard(
        loader, factors, horizon=horizon, n_buckets=n_buckets, n_trials=trials
    )
    if table.empty:
        print("No graph factors could be scored (is the data built? run `make data`).")
        return table

    FACTOR_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = FACTOR_REPORT_DIR / "graph_scorecard_summary.csv"
    table.to_csv(out)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(
        f"\nGraph scorecard ({len(table)} factors, {horizon}d forward, "
        f"deflated vs {trials} trials):\n"
    )
    print(table.to_string())
    print(f"\nSaved to {out}")
    return table


if __name__ == "__main__":
    main()
