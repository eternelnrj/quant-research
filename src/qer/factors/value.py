"""Value factor: book-to-market (book equity / market cap).

Direction +1 (cheap, high book-to-market names earn higher expected returns).
Needs fundamentals (``book_equity``) and market cap; skipped by the zoo runner
if either input is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.data.fundamentals import FundamentalsLoader
from qer.factors.base import Factor, register


class Value(Factor):
    name = "value_btm"
    direction = +1

    def compute_panel(self, loader) -> pd.DataFrame:
        cal = loader.close.index
        book = FundamentalsLoader().panel("book_equity", cal)
        cap = loader.market_cap
        cols = book.columns.intersection(cap.columns)
        with np.errstate(invalid="ignore", divide="ignore"):
            return book[cols] / cap[cols].replace(0, np.nan)


register(Value())
