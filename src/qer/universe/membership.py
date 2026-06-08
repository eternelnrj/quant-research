"""
Phase 1: Build the historical S&P 500 membership table (entity-resolved).

One-off build script. Produces a DataFrame with one row per
(entity, contiguous membership interval) and saves it to parquet:

    ticker      start_date    end_date
    AAPL        1982-11-30    2099-12-31     (still in index)
    BKNG        1999-04-01    2099-12-31     (incorporates PCLN history)
    META        2013-12-23    2099-12-31     (incorporates FB history)
    LEHM        NaT           2008-09-15     (added before log, removed 2008)
    CCR         <added>       2008-07-01     (acquired; closed via TICKER_EXITS)

Entity resolution:
  Tickers that have been renamed (e.g. PCLN -> BKNG, FB -> META) are
  collapsed to their CURRENT ticker via the TICKER_RENAMES map below.
  All historical events for an old ticker are attributed to the current
  one. The rename map handles multi-step renames (e.g. WLP -> ANTM -> ELV)
  by walking the chain.

  Acquisitions (entity ceases to exist) are NOT renames. For tickers that
  were acquired during the changes-log coverage but the log didn't record
  their removal, use TICKER_EXITS to supply the acquisition date.

Conventions:
  * end_date = NaT means "still in the index".                      # MODIF
  * NaT in start_date means the addition predates the changes log.
  * Tickers absent from both TICKER_RENAMES and TICKER_EXITS, but with
    dangling-open intervals, are closed conservatively at the latest
    log date - rather than dropped - so their historical membership is
    preserved (with imprecise exit date).

Same-day events:
  * Same-day removal+addition under the same ticker (e.g. FOXA on
    2019-03-19) produces two intervals meeting at the swap date. The
    sort tiebreaker processes removals BEFORE additions on the same
    day to achieve this.

Tickers held by successive entities under the same symbol:
  * The changes log sometimes records only a removal for a ticker that
    is subsequently held by a different entity (e.g. AT&T "T" in 2005).
    For any ticker in the current set whose log-events net to "closed",
    we seed an additional "added" event at the latest observed removal.

Usage:
    python build_membership.py            # use cached Wikipedia data
    python build_membership.py --refresh  # force fresh Wikipedia fetch

Known limitations:
  * Wikipedia changes log has reliable coverage from ~2011 onwards.
    Earlier dates produce systematically low universe counts.
  * TICKER_RENAMES and TICKER_EXITS are manually curated. Add new entries
    as new corporate events happen. Warnings flag tickers needing
    classification.
"""

from collections import defaultdict

import pandas as pd

# from qer.config import DATA_DIR, SENTINEL_END
from qer.universe.renames import TICKER_EXITS, TICKER_RENAMES
from qer.universe.wikipedia import _find_col, _to_yahoo, fetch_sp500_tables

# OUTPUT_FILE = (
#    DATA_DIR / "membership.parquet"
# )  # Path("data/membership.parquet") #Path("data/processed/sp500_membership.parquet")


# ---------------------------------------------------------------------------
# Ticker rename map.
#
# Maps OLD_TICKER -> CURRENT_TICKER for cases where the same entity
# continued in the index under a new ticker. The build step walks the
# chain so multi-step renames (WLP -> ANTM -> ELV) collapse to the
# final symbol.
#
# Use this for RENAMES ONLY (same entity, new ticker). For acquisitions
# (entity ceases to exist), use TICKER_EXITS below instead.
#
# Format: Wikipedia-format tickers (BRK.B not BRK-B). Yahoo conversion is
# applied separately at output time.
# ---------------------------------------------------------------------------


def _resolve_ticker(ticker: str, renames: dict = TICKER_RENAMES) -> str:
    """
    Walk the rename chain to find the current canonical ticker.
    Handles multi-step chains (WLP -> ANTM -> ELV).
    Cycle-safe: if a cycle is somehow present in the map, stops walking.
    """
    if not ticker:
        return ticker
    seen = set()
    while ticker in renames and ticker not in seen:
        seen.add(ticker)
        ticker = renames[ticker]
    return ticker


def _event_sort_key(event):
    """
    Sort events chronologically. On the same day, process REMOVALS first,
    so that a same-day remove+add of the same ticker yields two distinct
    intervals meeting at that date (rather than a zero-length interval).
    """
    date, kind = event
    return (
        date if pd.notna(date) else pd.Timestamp.min,
        0 if kind == "removed" else 1,
    )


def _net_open_at_end(events) -> bool:
    """Walk events chronologically; return True if there is a net unclosed addition."""
    depth = 0
    for _, kind in sorted(events, key=_event_sort_key):
        depth = depth + 1 if kind == "added" else max(depth - 1, 0)
    return depth > 0


def validate(df: pd.DataFrame, current_set_yahoo: set) -> None:
    """Sanity checks. Fail loudly on anomalies.

    ``current_set_yahoo`` must be in the SAME ticker format as ``df`` (i.e.
    Yahoo format ``BRK-B`` when the table was built with ``yahoo_format=True``,
    Wikipedia format ``BRK.B`` otherwise), so the "still-in" comparison lines
    up for class-share tickers.
    """

    # No malformed intervals (start strictly >= end)
    both = df.dropna(subset=["start_date", "end_date"])
    bad = both[both["start_date"] >= both["end_date"]]
    assert bad.empty, f"Found {len(bad)} intervals with start >= end:\n{bad}"

    # Set of "still in" tickers in the table == current set from Wikipedia
    open_now = set(
        df.loc[pd.isna(df["end_date"]), "ticker"]
    )  # set(df.loc[df["end_date"] == SENTINEL_END, "ticker"])
    extra = open_now - current_set_yahoo
    missing = current_set_yahoo - open_now
    assert not extra, f"Tickers marked still-in but not in current set: {extra}"
    assert not missing, (
        f"Tickers in current set but not marked still-in: {missing}, open now: {len(open_now)}"
    )

    # Universe size at representative dates. Bound is wider than nominal
    # 500 because of multi-class shares, recycled tickers near swap dates,
    # and pre-log members with NaT starts. Dates before 2011 excluded -
    # the changes log doesn't have reliable coverage there.
    def active_at(target_str: str) -> int:
        target = pd.to_datetime(target_str)
        active = df[
            (df["start_date"].isna() | (df["start_date"] <= target))
            & (df["end_date"].isna() | (df["end_date"] > target))
        ]
        # print(len(active))
        return len(active)

    for d in ["2011-06-30", "2015-01-01", "2020-01-02", "2024-03-01"]:
        n = active_at(d)
        assert 470 <= n <= 530, f"Active universe at {d} has size {n}, expected ~500"


def build_membership_table(
    yahoo_format: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Build (ticker, start_date, end_date) membership intervals, entity-resolved.

    yahoo_format: map BRK.B -> BRK-B etc. so output is directly yfinance-usable.
    force_refresh: re-fetch Wikipedia tables instead of using the local cache.
    """
    current, changes = fetch_sp500_tables(force_refresh=force_refresh)

    # --- Apply ticker rename map to the changes log ------------------------
    # After this step, all historical events for PCLN are attributed to
    # BKNG, all FB events to META, etc.
    changes = changes.copy()
    changes["added"] = changes["added"].map(_resolve_ticker)
    changes["removed"] = changes["removed"].map(_resolve_ticker)

    # Latest date covered by the changes log; used as a conservative
    # close-at date for tickers with dangling-open intervals not in
    # TICKER_EXITS (acquisitions the log didn't capture as removals).
    log_end_date = changes["date"].max()

    # --- Find the "Date added" column in the current-constituents table ----
    date_added_col = _find_col(current.columns, "date", "added") or _find_col(
        current.columns, "date", "first"
    )
    if date_added_col is None:
        print(
            "WARNING: no 'Date added' column found in current table; "
            "current members with no event in the changes log will get NaT start."
        )

    # --- Step 1: collect every (ticker, date, kind) event from changes log
    events_by_ticker: dict[str, list[tuple]] = defaultdict(list)
    for _, row in changes.iterrows():
        if row["added"]:
            events_by_ticker[row["added"]].append((row["date"], "added"))
        if row["removed"]:
            events_by_ticker[row["removed"]].append((row["date"], "removed"))

    for ticker in list(events_by_ticker.keys()):
        unique_events = list(set(events_by_ticker[ticker]))
        events_by_ticker[ticker] = unique_events

        # Current set is rename-resolved for symmetry (in practice the current
        # table already uses current tickers so this is a no-op).
    current_set = {_resolve_ticker(t) for t in current["ticker"]}

    # Parse the current table's "Date added" column once, permissively.
    if date_added_col is not None:
        parsed_added = pd.to_datetime(current[date_added_col], errors="coerce")
        n_failed = parsed_added.isna().sum()
        n_total = len(parsed_added)
        if n_failed > 0.05 * n_total:
            raise RuntimeError(
                f"{n_failed}/{n_total} 'Date added' values failed to parse. "
                f"Wikipedia may have changed the format; sample of failed rows:\n"
                f"{current.loc[parsed_added.isna(), date_added_col].head()}"
            )
        current_added_dates = {
            _resolve_ticker(t): d for t, d in zip(current["ticker"], parsed_added)
        }
    else:
        current_added_dates = {}

    # --- Step 2: for each current member, ensure events leave them OPEN ----
    # If not, seed an "added" event. See docstring for T/AGN/FOXA details.
    for ticker in current_set:
        events = events_by_ticker[ticker]
        if _net_open_at_end(events):
            continue
        removal_dates = [d for d, k in events if k == "removed" and pd.notna(d)]
        if removal_dates:
            seed_date = max(removal_dates)
        else:
            seed_date = current_added_dates.get(ticker, pd.NaT)
        events_by_ticker[ticker].append((seed_date, "added"))

    # --- Step 3: walk each ticker's events chronologically into intervals --
    rows = []
    fallback_closes = []  # tickers closed at log_end_date (for diagnostics)
    pending_adds = []  # add this line

    for ticker, evts in events_by_ticker.items():
        evts_sorted = sorted(evts, key=_event_sort_key)
        open_start = None
        last_removal = None  # for recycled tickers (AGN-style)
        for date, kind in evts_sorted:
            if kind == "added":
                if open_start is None:
                    open_start = date
                # else: double-add without intervening removal; ignore
            else:  # removed
                interval_start = open_start if open_start is not None else last_removal
                rows.append(
                    {
                        "ticker": ticker,
                        "start_date": interval_start,
                        "end_date": date,
                    }
                )
                open_start = None
                last_removal = date

        # Close any dangling open interval.
        if open_start is not None:
            if ticker in current_set:
                end = None  # SENTINEL_END
            elif ticker in TICKER_EXITS:
                end = pd.Timestamp(TICKER_EXITS[ticker])

            elif pd.notna(open_start) and open_start >= log_end_date:
                # Added on the most recent change in the log, no removal, and
                # not yet in the current-constituents snapshot: a freshly
                # announced/effective addition the two Wikipedia tables haven't
                # reconciled. Closing at log_end_date would make a zero-length
                # interval, so skip it until the current table catches up.
                pending_adds.append(ticker)
                continue

            else:
                # No removal in log, not in current set, not in exits map.
                # Close conservatively at the last log date. The ticker
                # was a genuine member; we just don't know its exact exit.
                end = log_end_date
                fallback_closes.append(ticker)
            rows.append(
                {
                    "ticker": ticker,
                    "start_date": open_start,
                    "end_date": end,
                }
            )

    if fallback_closes:
        print(
            f"NOTE: {len(fallback_closes)} tickers closed at log_end_date "
            f"({log_end_date.date()}) because no removal was found and "
            f"they are not in TICKER_EXITS: {fallback_closes}"
        )
        print(
            "  Consider adding these to TICKER_EXITS with their actual "
            "acquisition/delisting dates for better precision."
        )

    if pending_adds:
        print(
            f"NOTE: {len(pending_adds)} freshly-added tickers skipped (added on "
            f"{log_end_date.date()}, not yet in the current-constituents table): "
            f"{pending_adds}. Re-run with --refresh once Wikipedia reconciles."
        )

    df = pd.DataFrame(rows)

    if yahoo_format:
        df["ticker"] = df["ticker"].map(_to_yahoo)
    df = df.sort_values(["ticker", "start_date"], na_position="first").reset_index(drop=True)

    return df
