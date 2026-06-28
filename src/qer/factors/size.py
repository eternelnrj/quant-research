"""Size factor: negative log market capitalisation (the small-cap premium).

Raw factor is log market cap; direction is -1 (smaller names earn higher
expected returns). Needs shares outstanding via ``DataLoader.market_cap`` - if
no shares data has been ingested, the panel is all-NaN and the factor is simply
skipped by the zoo runner.
"""
# NEW

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.factors.base import Factor, register


class Size(Factor):
    name = "size"
    direction = -1  # smaller market cap => higher expected return

    def compute_panel(self, loader) -> pd.DataFrame:
        cap = loader.market_cap  # date x ticker
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log(cap.replace(0, np.nan))


register(Size())
