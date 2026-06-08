"""CLI: rebuild the S&P 500 membership table from Wikipedia.

Builds the entity-resolved (ticker, start_date, end_date) table, validates it
against the live current-constituents set (in the same Yahoo ticker format as
the table), and persists it to the processed-data parquet.

Usage:
    python -m scripts.build_membership            # use cached Wikipedia data
    python -m scripts.build_membership --refresh  # force fresh Wikipedia fetch
"""

import argparse

from qer.config import MEMBERSHIP_FILE
from qer.universe.membership import _resolve_ticker, build_membership_table, validate
from qer.universe.wikipedia import _to_yahoo, fetch_sp500_tables

# from qer.universe.renames import

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh", action="store_true", help="Force fresh Wikipedia fetch before building."
    )
    args = parser.parse_args()

    df = build_membership_table(force_refresh=args.refresh)

    # Validate against the current set in the SAME (Yahoo) format as the table,
    # so class-share tickers (BRK.B -> BRK-B) line up. This also runs the
    # ~500-name universe-size sanity check, which only makes sense at full size.
    current, _ = fetch_sp500_tables()
    current_set_yahoo = {_to_yahoo(_resolve_ticker(t)) for t in current["ticker"]}
    validate(df, current_set_yahoo)

    MEMBERSHIP_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MEMBERSHIP_FILE)
    print(f"Wrote {len(df)} membership intervals to {MEMBERSHIP_FILE}")
