"""CLI: compute and plot the 12-1 momentum information-coefficient analysis.

Thin wrapper around qer.diagnostics.factor_ic so the heavy logic stays in the
importable package (and stays unit-testable) while this module is the runnable
entry point referenced by the Makefile.

Usage:
    python -m scripts.factor_ic
"""

from qer.data.loader import DataLoader
from qer.diagnostics.factor_ic import run_momentum_ic_analysis

if __name__ == "__main__":
    loader = DataLoader()
    run_momentum_ic_analysis(loader, years=5)
