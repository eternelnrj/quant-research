"""Short-term reversal: the most-recent-month return, expected to mean-revert.

Raw factor is the trailing 21-day log return; direction is -1 (recent winners
tend to underperform next month). Deliberately the complement of momentum_12_1,
which *skips* this most-recent month for exactly this reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.factors.base import Factor, register


def reversal_panel(prices_df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Trailing ``window``-day log return at every date (look-ahead-safe)."""
    logp = np.log(prices_df)
    return logp - logp.shift(window)


class Reversal1M(Factor):
    name = "reversal_1m"
    direction = -1  # high recent return => expected underperformance

    def compute_panel(self, loader) -> pd.DataFrame:
        return reversal_panel(loader.close, window=21)


register(Reversal1M())
