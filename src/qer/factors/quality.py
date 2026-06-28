"""Quality factor: gross profitability (gross profit / total assets).

Direction +1 (more profitable firms earn higher expected returns; Novy-Marx
2013). Needs fundamentals (``gross_profit``, ``total_assets``); skipped by the
zoo runner if unavailable.
"""

from __future__ import annotations

import numpy as np

from qer.data.fundamentals import FundamentalsLoader
from qer.factors.base import Factor, register


class Quality(Factor):
    name = "quality_gp"
    direction = +1

    def compute_panel(self, loader):
        cal = loader.close.index
        fl = FundamentalsLoader()
        gp = fl.panel("gross_profit", cal)
        ta = fl.panel("total_assets", cal)
        cols = gp.columns.intersection(ta.columns)
        with np.errstate(invalid="ignore", divide="ignore"):
            return gp[cols] / ta[cols].replace(0, np.nan)


register(Quality())
