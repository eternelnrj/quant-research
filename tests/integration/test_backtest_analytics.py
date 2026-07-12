"""Integration test for Phase 4.4: the Sharpe-vs-cost curve on the harness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.backtest import sharpe_vs_cost
from qer.factors.graph.base import PanelFactor


def _factor(loader, seed=0):
    rng = np.random.default_rng(seed)
    sig = pd.DataFrame(rng.normal(size=(len(loader.close.index), len(loader.close.columns))),
                       index=loader.close.index, columns=loader.close.columns)
    return PanelFactor(sig, name="s", direction=1)


def test_sharpe_vs_cost_declines_with_spread(synthetic_loader):
    curve = sharpe_vs_cost(synthetic_loader, _factor(synthetic_loader),
                           bps_grid=(0, 5, 10, 15, 20, 25), scheme="signal")
    assert list(curve.index) == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    assert {"net_sharpe", "mean_net_return"} <= set(curve.columns)
    # mean net return is a deterministic drag: strictly decreasing in the assumed cost
    assert (curve["mean_net_return"].diff().dropna() < 0).all()
    # net Sharpe at the top of the grid is below net Sharpe at zero cost
    assert curve["net_sharpe"].iloc[-1] < curve["net_sharpe"].iloc[0]
