"""Phase 4.4: the Sharpe-vs-cost curve -- the single most convincing chart in a writeup.

Runs the (gross) backtest once, then re-nets it across a grid of assumed round-trip
spread costs and reports net Sharpe and mean net return at each. A Sharpe that falls
from 2 to 0 between 5 and 10 bps is borderline; 2 to 1.5 across that range is robust.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from qer.backtest.costs import CostModel, apply_costs
from qer.backtest.engine import Backtest
from qer.backtest.metrics import sharpe


def sharpe_vs_cost(loader, factor, bps_grid=(0, 5, 10, 15, 20, 25), *, cost_model=None,
                   periods_per_year: int = 252, **backtest_kwargs) -> pd.DataFrame:
    """Net Sharpe and mean net return as a function of assumed round-trip spread (bps).

    The gross backtest is run once; each grid point re-applies costs with
    ``spread_bps`` set to that value. ``cost_model`` defaults to a *spread-only* model
    (impact and borrow off) so the x-axis isolates the assumed round-trip spread, which
    is what the chart is about; pass a fuller :class:`CostModel` to include them.
    Returns a DataFrame indexed by ``bps``.
    """
    template = cost_model or CostModel(impact_coef=0.0, borrow_bps=0.0)
    result = Backtest(**backtest_kwargs).run(loader, factor)
    rows = []
    for bps in bps_grid:
        costed = apply_costs(result, loader, replace(template, spread_bps=float(bps)))
        rows.append({
            "bps": float(bps),
            "net_sharpe": sharpe(costed.net_returns, periods_per_year),
            "mean_net_return": float(costed.net_returns.mean()),
        })
    return pd.DataFrame(rows).set_index("bps")
