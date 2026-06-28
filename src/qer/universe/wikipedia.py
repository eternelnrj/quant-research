"""
Historical S&P 500 membership.

Strategy: scrape today's constituents AND the changes log from the live
Wikipedia 'List of S&P 500 companies' page, then reconstruct membership at
any past date by reversing every change that happened after that date.

Why not scrape old revisions of the page?
  - Page formatting has shifted across years; table positions and column
    layouts differ, so heuristic ticker extraction silently misses names
    (class-share tickers like BRK.B / BF.B are the most common casualties).
  - Old revisions can contain transient errors that have since been corrected
    on the live page; the curated changes log is more reliable.
  - One fetch covers every historical date.

Known limitations:
  - Tickers that were renamed mid-membership without a changes-log entry will
    appear under their current symbol, not their historical one.
  - Wikipedia is not authoritative; small errors exist.
  - The Wikipedia-to-Yahoo symbol mapping (`.` -> `-`) is a heuristic.
"""

from io import StringIO

import pandas as pd
import requests

from qer.config import RAW_DIR, S_AND_P500_CHANGES, S_AND_P500_CURRENT

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "QuantResearchProject/1.0 (educational)"}

RAW_DIR.mkdir(parents=True, exist_ok=True)

# CURRENT_CACHE = S_AND_P500_CURRENT  # RAW_DIR / "sp500_current.parquet"
# CHANGES_CACHE = S_AND_P500_CHANGES  # RAW_DIR / "sp500_changes.parquet"


def _to_yahoo(ticker: str) -> str:
    """Map Wikipedia ticker format to Yahoo Finance format: BRK.B -> BRK-B."""
    return ticker.replace(".", "-")


def _find_col(columns, *keywords):
    """Return first column name containing ALL keywords (case-insensitive)."""
    for col in columns:
        col_lower = str(col).lower()
        if all(kw in col_lower for kw in keywords):
            return col
    return None


def fetch_sp500_tables(force_refresh: bool = False):
    """
    Return (current_constituents_df, changes_df).

    Both are cached to data/raw/ after first fetch; pass force_refresh=True
    to re-download (e.g. once a quarter to pick up recent index changes).
    """

    if (not force_refresh) and S_AND_P500_CURRENT.exists() and S_AND_P500_CHANGES.exists():
        return pd.read_parquet(S_AND_P500_CURRENT), pd.read_parquet(S_AND_P500_CHANGES)

    resp = requests.get(WIKI_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))

    # --- Table 0: current constituents -------------------------------------
    current = tables[0].copy()
    current.columns = [str(c).strip() for c in current.columns]
    sym_col = _find_col(current.columns, "symbol") or _find_col(current.columns, "ticker")
    if sym_col is None:
        raise RuntimeError(
            f"No symbol/ticker column in current-constituents table. "
            f"Got columns: {current.columns.tolist()}"
        )
    current = current.rename(columns={sym_col: "ticker"})
    current["ticker"] = current["ticker"].astype(str).str.strip()

    # --- Table 1: changes log (multi-level header) -------------------------
    # Header looks like: Date | Added(Ticker, Security) | Removed(Ticker, Security) | Reason
    changes = tables[1].copy()
    flat_cols = []
    for col in changes.columns:
        if isinstance(col, tuple):
            parts = [str(p).strip().lower() for p in col if str(p).strip()]
            # Collapse repeats: ('Date', 'Date') -> 'date'
            deduped: list[str] = []
            for p in parts:
                if not deduped or deduped[-1] != p:
                    deduped.append(p)
            flat_cols.append("_".join(deduped))
        else:
            flat_cols.append(str(col).strip().lower())
    changes.columns = flat_cols

    date_col = _find_col(changes.columns, "date")
    added_col = _find_col(changes.columns, "added", "ticker") or _find_col(
        changes.columns, "added", "symbol"
    )
    removed_col = _find_col(changes.columns, "removed", "ticker") or _find_col(
        changes.columns, "removed", "symbol"
    )

    if not all([date_col, added_col, removed_col]):
        raise RuntimeError(
            f"Could not identify required columns in changes table. "
            f"Got date={date_col}, added={added_col}, removed={removed_col}. "
            f"Available: {changes.columns.tolist()}"
        )

    changes = changes[[date_col, added_col, removed_col]].copy()
    changes.columns = ["date", "added", "removed"]
    changes["date"] = pd.to_datetime(changes["date"], format="%B %d, %Y", errors="coerce")

    n_unparsed = changes["date"].isna().sum()
    assert n_unparsed == 0, (
        f"{n_unparsed} dates failed to parse with format '%B %d, %Y'. "
        f"Wikipedia may have changed its date convention. Sample: "
        f"{changes[changes['date'].isna()].head()}"
    )
    changes = changes.reset_index(drop=True)
    for col in ("added", "removed"):
        changes[col] = (
            changes[col].astype(str).str.strip().replace({"nan": "", "NaN": "", "None": ""})
        )

    current.to_parquet(S_AND_P500_CURRENT)
    changes.to_parquet(S_AND_P500_CHANGES)
    return current, changes


def get_universe(date_str: str, yahoo_format: bool = True, force_refresh: bool = False) -> list:
    """
    Return list of S&P 500 ticker symbols as of date_str (YYYY-MM-DD).

    Algorithm:
        members = today's constituents
        for each change with date > target, in reverse chronological order:
            if a ticker was ADDED after target, it was NOT a member at target
                -> remove from members
            if a ticker was REMOVED after target, it WAS a member at target
                -> add back to members

    date_str is in YYYY-MM-DD format
    yahoo_format=True maps `.` to `-` so output works directly with yfinance.
    force_refresh=True bypasses the local cache and re-fetches from Wikipedia.
    Useful when running quarterly to pick up recent index changes.
    """
    target = pd.to_datetime(date_str, format="%Y-%m-%d")
    current, changes = fetch_sp500_tables(force_refresh)

    members = set(current["ticker"])
    future_changes = changes[changes["date"] > target].sort_values("date", ascending=False)

    for _, row in future_changes.iterrows():
        if row["added"]:
            members.discard(row["added"])
        if row["removed"]:
            members.add(row["removed"])

    tickers = sorted(members)
    if yahoo_format:
        tickers = [_to_yahoo(t) for t in tickers]
    return tickers
