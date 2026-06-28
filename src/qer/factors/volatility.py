"""Low-volatility factor: trailing 60-day return volatility.

Raw factor is the trailing 60-day standard deviation of daily log returns;
direction is -1 (the low-volatility anomaly: lower-vol names tend to deliver
better risk-adjusted returns).
"""

# NEW
from __future__ import annotations

import pandas as pd

from qer.factors.base import Factor, register


def volatility_panel(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Trailing ``window``-day std of daily returns (look-ahead-safe)."""
    return returns.rolling(window, min_periods=window * 2 // 3).std(ddof=1)


class Volatility60(Factor):
    name = "volatility_60d"
    direction = -1  # high volatility => expected underperformance

    def compute_panel(self, loader) -> pd.DataFrame:
        return volatility_panel(loader.get_returns("log"), window=60)


register(Volatility60())
