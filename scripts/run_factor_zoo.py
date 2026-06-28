"""CLI: run every registered factor through the shared evaluation harness.

Thin wrapper around ``qer.diagnostics.factor_zoo.build_factor_zoo_table`` - the
evaluation logic lives in the package so notebooks can import it too. This
wrapper just builds the table, writes the CSV, and prints it.

Usage:
    python -m scripts.run_factor_zoo
"""

from __future__ import annotations

import pandas as pd

from qer.config import FACTOR_REPORT_DIR
from qer.diagnostics.factor_zoo import DEFAULT_PRIMARY_H, build_factor_zoo_table


def main(years: int = 5) -> pd.DataFrame:
    table = build_factor_zoo_table(years=years)
    if table.empty:
        print("No factors could be evaluated (is the data built? run `make data`).")
        return table

    FACTOR_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = FACTOR_REPORT_DIR / "factor_zoo_summary.csv"
    table.to_csv(out)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"\nFactor zoo ({len(table)} factors, {DEFAULT_PRIMARY_H}d forward):\n")
    print(table.to_string())
    print(f"\nSaved to {out}")
    return table


if __name__ == "__main__":
    main()
