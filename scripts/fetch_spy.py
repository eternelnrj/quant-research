"""CLI: fetch SPY's price history for the data-audit benchmark check.

`qer.diagnostics.audit_data.check_spy_total_return` reads SPY from
data/raw/SPY.parquet and compares its realised annualised total return to a
plausible band - a coarse end-to-end sanity check that prices are
dividend/split adjusted and the date range is right. SPY is an ETF, not an
S&P 500 constituent, so it is never pulled by ingest_prices.py; this script
fills that gap.

Pulled with auto_adjust=False to match the project convention, so the file
keeps a separate ``Adj Close`` column (the dividend-adjusted total-return
series the check prefers) alongside the raw OHLC.

Usage:
    python -m scripts.fetch_spy
"""

import pandas as pd
import yfinance as yf

# from qer.config import END_DATE, RAW_DIR, START_DATE
from qer.config import END_DATE, RAW_DIR, SPY_FILE, START_DATE  # line 21

# SPY_FILE = RAW_DIR / "SPY.parquet"

if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = yf.download(
        "SPY",
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise SystemExit(
            "ERROR: no SPY data returned from yfinance. Check the network / rate limits and retry."
        )

    # Recent yfinance versions return MultiIndex columns even for a single
    # ticker, e.g. ('Adj Close', 'SPY'). check_spy_total_return just lowercases
    # column names, so flatten to the price-field level first.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_parquet(SPY_FILE)
    print(f"Wrote {len(df)} rows to {SPY_FILE} ({df.index[0].date()} -> {df.index[-1].date()})")
