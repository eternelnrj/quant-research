"""Point-in-time fundamentals access (for value and quality).

Reads a tidy parquet at ``FUNDAMENTALS_FILE`` with columns:

    ticker | available_date | field | value

``available_date`` is when the figure became *known* (period end + reporting
lag), so building a daily panel by forward-filling from ``available_date`` is
point-in-time correct by construction: at any date you see only the latest
filing already public by then. The lag is applied at ingest (see
``scripts/ingest_fundamentals.py``); this loader does not peek.

Recognised fields used by the factor zoo: ``book_equity``, ``gross_profit``,
``total_assets`` (extend freely - the loader is field-agnostic).
"""

from __future__ import annotations

import pandas as pd

from qer.config import FUNDAMENTALS_FILE


class FundamentalsLoader:
    def __init__(self, path=None):
        self.path = path if path is not None else FUNDAMENTALS_FILE
        self._raw = None

    @property
    def raw(self) -> pd.DataFrame:
        if self._raw is None:
            if not self.path.exists():
                raise FileNotFoundError(
                    f"{self.path} not found. Run `python -m scripts.ingest_fundamentals` "
                    "to enable fundamentals-based factors (value, quality)."
                )
            df = pd.read_parquet(self.path)
            df["available_date"] = pd.to_datetime(df["available_date"])
            self._raw = df
        return self._raw

    def panel(self, field: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
        """Daily ``date x ticker`` panel for one field, point-in-time correct.

        Forward-fills the latest known value onto the given daily ``calendar``.
        """
        sub = self.raw[self.raw["field"] == field]
        if sub.empty:
            raise KeyError(f"No fundamentals rows for field {field!r}.")
        wide = sub.pivot_table(
            index="available_date", columns="ticker", values="value", aggfunc="last"
        ).sort_index()
        combined = wide.index.union(calendar)
        return wide.reindex(combined).ffill().reindex(calendar)
