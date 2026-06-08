"""CLI: pull per-ticker daily price history via yfinance.

Reads the membership table to know which tickers ever appeared in the index,
then downloads each one's full history to data/raw/prices/<TICKER>.parquet.

Prices are pulled with auto_adjust=False, so each file keeps a separate
``Adj Close`` column (split/dividend adjusted) alongside the raw OHLC. The
DataLoader exposes that ``adj close`` field as ``loader.close``.

Usage:
    python -m scripts.ingest_prices
"""

import pandas as pd
import yfinance as yf

from qer.config import END_DATE, MEMBERSHIP_FILE, PRICES_DIR, START_DATE

if __name__ == "__main__":
    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    membership = pd.read_parquet(MEMBERSHIP_FILE)
    all_tickers_ever_in_universe = membership["ticker"].unique().tolist()

    n = len(all_tickers_ever_in_universe)
    failed = []
    for i, ticker in enumerate(all_tickers_ever_in_universe, 1):
        try:
            df = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=False,
                progress=False,
            )
        except Exception as exc:  # network / delisted / rate-limit
            print(f"[{i}/{n}] {ticker}: download error ({exc}); skipping")
            failed.append(ticker)
            continue

        if df is None or df.empty:
            print(f"[{i}/{n}] {ticker}: no data; skipping")
            failed.append(ticker)
            continue

        df.to_parquet(PRICES_DIR / f"{ticker}.parquet")
        print(f"[{i}/{n}] {ticker}: {len(df)} rows")

    print(f"\nDone. {n - len(failed)}/{n} tickers ingested into {PRICES_DIR}")
    if failed:
        print(f"{len(failed)} failed/empty: {failed}")
