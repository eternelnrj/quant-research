"""Liquidity factor: Amihud illiquidity.

Amihud (2002) illiquidity is the trailing average of |daily return| divided by
daily dollar volume - how much the price moves per dollar traded. Higher means
more illiquid; direction is +1 (the illiquidity premium: less-liquid names earn
higher expected returns). Scaled by 1e6 purely for readable magnitudes.
"""
# NEW

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.factors.base import Factor, register


def amihud_panel(
    returns: pd.DataFrame, dollar_volume: pd.DataFrame, window: int = 21
) -> pd.DataFrame:
    """Trailing-mean |return| / dollar-volume (look-ahead-safe), x 1e6."""
    dv = dollar_volume.replace(0, np.nan)
    daily = (returns.abs() / dv).replace([np.inf, -np.inf], np.nan)
    return daily.rolling(window, min_periods=window * 2 // 3).mean() * 1e6


class AmihudIlliquidity(Factor):
    name = "amihud_illiquidity"
    direction = +1  # more illiquid => higher expected return (illiquidity premium)

    def compute_panel(self, loader) -> pd.DataFrame:
        # Amihud (2002) is defined on SIMPLE daily returns. At daily frequency
        # log and simple returns are near-identical (and the factor is used
        # rank-wise), but simple returns are the faithful definition.
        return amihud_panel(loader.get_returns("simple"), loader.dollar_volume, window=21)


register(AmihudIlliquidity())
