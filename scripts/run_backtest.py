"""CLI: run a full backtest for one factor (or all classical factors) and save a report.

Thin wrapper -- the backtest/costs/analytics/report logic all lives in
:mod:`qer.backtest`. Resolves the factor by name across the classical ``FACTORS`` and
graph ``GRAPH_FACTORS`` registries.

Usage:
    python -m scripts.run_backtest --factor momentum --scheme signal --spread-bps 8 --oos 2020-01-01
    python -m scripts.run_backtest --factor all              # every classical factor
"""

from __future__ import annotations

import argparse

from qer.backtest.costs import CostModel
from qer.backtest.report import build_report
from qer.config import DATA_DIR
from qer.data.loader import DataLoader
from qer.factors import all_factors
from qer.factors.graph import all_graph_factors


def _resolve(name: str):
    for f in list(all_factors()) + list(all_graph_factors()):
        if f.name == name:
            return f
    available = ", ".join(sorted(f.name for f in all_factors()))
    raise SystemExit(f"unknown factor {name!r}. Classical factors: {available}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run a backtest and save a standardised report.")
    p.add_argument("--factor", required=True, help="factor name, or 'all' for every classical factor")
    p.add_argument("--scheme", default="equal", choices=["equal", "signal", "rank"])
    p.add_argument("--freq", default="M", help="rebalance freq (W/M/Q or an integer of days)")
    p.add_argument("--spread-bps", type=float, default=8.0)
    p.add_argument("--impact-coef", type=float, default=0.1)
    p.add_argument("--borrow-bps", type=float, default=75.0)
    p.add_argument("--aum", type=float, default=5e8)
    p.add_argument("--oos", default=None, help="OOS split date, e.g. 2020-01-01")
    p.add_argument("--n-trials", type=int, default=None, help="trial count for the deflated Sharpe")
    p.add_argument("--out", default=str(DATA_DIR / "backtest"))
    p.add_argument("--fmt", default="html", choices=["html", "pdf"])
    a = p.parse_args()

    loader = DataLoader()
    cost_model = CostModel(spread_bps=a.spread_bps, impact_coef=a.impact_coef,
                           borrow_bps=a.borrow_bps, aum=a.aum)
    factors = all_factors() if a.factor == "all" else [_resolve(a.factor)]
    for factor in factors:
        path, data = build_report(
            loader, factor, out_dir=a.out, fmt=a.fmt, freq=a.freq, scheme=a.scheme,
            cost_model=cost_model, oos_split=a.oos, n_trials=a.n_trials)
        print(data.headline)
        print(f"Report: {path}")


if __name__ == "__main__":
    main()
