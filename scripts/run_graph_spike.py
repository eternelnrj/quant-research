"""CLI: run the Subphase 3.1 harness-reuse spike against the real data.

Thin wrapper around ``qer.graphs.spike.assert_harness_reuse`` - the spike logic
lives in the package (so the test suite imports it directly). This wrapper builds
the ``DataLoader``, runs the spike, prints the comparison, and exits non-zero if
the wrapped graph-evaluation path fails to reproduce the native factor.

Usage:
    python -m scripts.run_graph_spike
"""

from __future__ import annotations

from qer.data.loader import DataLoader
from qer.graphs.spike import assert_harness_reuse


def main() -> dict:
    result = assert_harness_reuse(DataLoader())
    print("Harness-reuse spike PASSED")
    print(f"  factor:            {result['factor']}")
    print(f"  IC dates compared: {result['n_ic_dates']}")
    print(f"  max |IC| diff:     {result['ic_max_abs_diff']:.2e}")
    print(
        f"  decile Sharpe:     native {result['ls_sharpe_native']:.6f} "
        f"vs wrapped {result['ls_sharpe_wrapped']:.6f} "
        f"(diff {result['ls_sharpe_diff']:.2e})"
    )
    return result


if __name__ == "__main__":
    main()
