"""CLI: fetch per-ticker GICS sector metadata via yfinance.

Writes data/raw/sectors.parquet as a single-column frame indexed by ticker,
which qer.diagnostics.audit_data.load_sectors() reads to draw the sector
breakdown chart. Best-effort: tickers whose sector can't be resolved are
skipped with a warning rather than failing the whole run.

Usage:
    python -m scripts.fetch_sectors
"""

import pandas as pd
import yfinance as yf

# from qer.config import MEMBERSHIP_FILE, RAW_DIR
from qer.config import MEMBERSHIP_FILE, RAW_DIR, SECTORS_FILE  # line 15

# SECTORS_FILE = RAW_DIR / "sectors.parquet"

if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    membership = pd.read_parquet(MEMBERSHIP_FILE)
    tickers = membership["ticker"].unique().tolist()

    sectors: dict[str, str] = {}
    failed = []
    for i, ticker in enumerate(tickers, 1):
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector")
        except Exception as exc:
            sector = None
            print(f"[{i}/{len(tickers)}] {ticker}: lookup error ({exc})")
        if sector:
            sectors[ticker] = sector
        else:
            failed.append(ticker)

    series = pd.Series(sectors, name="sector").sort_index()
    series.to_frame().to_parquet(SECTORS_FILE)

    print(f"\nWrote {len(series)} sectors to {SECTORS_FILE}")
    if failed:
        print(f"{len(failed)} tickers had no sector (delisted/unknown): {failed}")
