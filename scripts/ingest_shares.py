"""CLI: ingest shares outstanding from SEC EDGAR (for market cap / size).

Downloads each S&P 500 member's XBRL ``companyfacts`` (via the shared
``qer.data.edgar`` client) and reads the cover-page shares-outstanding count
reported on every 10-K/10-Q. Writes a ``date x ticker`` parquet at
``SHARES_FILE``; ``DataLoader.market_cap`` forward-fills it onto the daily price
calendar and multiplies by the raw close.

Why EDGAR over yfinance: the filing's ``filed`` date gives point-in-time-correct
shares (consistent with the fundamentals ingest), and history goes back further
than yfinance's ``get_shares_full`` series. (yfinance is the simpler alternative
if you prefer one data source for everything.)

Concept: ``dei:EntityCommonStockSharesOutstanding`` (cover page), falling back to
``us-gaap:CommonStockSharesOutstanding``. The point-in-time row is built per
filing and de-duplicated like the fundamentals ingest (earliest disclosure of
each distinct value; restatements kept).

Known limitation: multi-class companies (e.g. GOOGL/GOOG, BRK-A/BRK-B) report
shares per class, so the cover-page figure can capture a single class rather
than the economic total - market cap for those names needs manual handling.
Validate against a few known filings before trusting the output.

Usage:
    QER_SEC_USER_AGENT="Your Name your@email.com" python -m scripts.ingest_shares
    QER_SEC_USER_AGENT="..." python -m scripts.ingest_shares --limit 5
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from qer.config import FUNDAMENTALS_DIR, FUNDAMENTALS_LAG_DAYS, MEMBERSHIP_FILE, SHARES_FILE
from qer.data.edgar import (
    available_date,
    concept_entries,
    dedup_point_in_time,
    fetch_company_facts,
    load_ticker_cik_map,
    require_user_agent,
    resolve_cik,
)

# (namespace, concept) tried in order; both are reported in "shares" units.
SHARES_CONCEPTS = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
]


def extract_shares(ticker: str, facts: dict, lag_days: int = FUNDAMENTALS_LAG_DAYS) -> list[dict]:
    """Point-in-time shares-outstanding rows: ticker | available_date | shares."""
    namespaces = facts.get("facts", {})
    lag = pd.Timedelta(days=lag_days)
    collected: list[dict] = []
    for ns, concept in SHARES_CONCEPTS:
        for e in concept_entries(namespaces.get(ns, {}), concept, "shares"):
            if "end" not in e or e.get("val") is None:
                continue
            collected.append(
                {
                    "ticker": ticker,
                    "available_date": available_date(e, lag),
                    "shares": float(e["val"]),
                    "period_end": e["end"],
                    "_end": e["end"],
                }
            )
        if collected:
            break  # first concept that yields data wins
    return dedup_point_in_time(collected, value_key="shares")


def shares_panel(rows: list[dict]) -> pd.DataFrame:
    """Collapse point-in-time shares rows into a ``date x ticker`` panel.

    When two filings share an ``available_date``, the most recent ``period_end``
    wins deterministically (the current shares figure), rather than an arbitrary
    input-order value as plain ``aggfunc="last"`` would give. The loader
    forward-fills the (sparse) result onto the daily calendar.
    """
    long = pd.DataFrame(rows)
    long["available_date"] = pd.to_datetime(long["available_date"])
    long["period_end"] = pd.to_datetime(long["period_end"])
    long = long.sort_values(["ticker", "available_date", "period_end"])
    return long.pivot_table(
        index="available_date", columns="ticker", values="shares", aggfunc="last"
    ).sort_index()


def main(limit: int | None = None, refresh: bool = False) -> pd.DataFrame:
    user_agent = require_user_agent()
    if not MEMBERSHIP_FILE.exists():
        raise SystemExit(f"{MEMBERSHIP_FILE} not found - run `make membership` first.")

    FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    SHARES_FILE.parent.mkdir(parents=True, exist_ok=True)

    tickers = sorted(pd.read_parquet(MEMBERSHIP_FILE)["ticker"].unique())
    if limit:
        tickers = tickers[:limit]

    print(f"Resolving {len(tickers)} tickers against SEC company_tickers.json ...")
    cik_map = load_ticker_cik_map(user_agent)

    all_rows: list[dict] = []
    n_ok = n_nocik = n_nofacts = n_err = 0
    for i, ticker in enumerate(tickers, 1):
        cik = resolve_cik(ticker, cik_map)
        if cik is None:
            n_nocik += 1
            continue
        try:
            facts = fetch_company_facts(cik, user_agent, FUNDAMENTALS_DIR, refresh)
        except Exception as exc:  # noqa: BLE001 - tolerate per-ticker failures
            print(f"  [{i}/{len(tickers)}] {ticker}: ERROR {type(exc).__name__}: {str(exc)[:60]}")
            n_err += 1
            continue
        if facts is None:
            n_nofacts += 1
            continue
        rows = extract_shares(ticker, facts)
        all_rows.extend(rows)
        n_ok += 1
        if i % 25 == 0:
            print(f"  [{i}/{len(tickers)}] {ticker}: cumulative {len(all_rows)} obs")

    if not all_rows:
        raise SystemExit(
            "No shares extracted - check the User-Agent and network access to data.sec.gov."
        )

    wide = shares_panel(all_rows)
    wide.to_parquet(SHARES_FILE)

    print(
        f"\nWrote shares for {wide.shape[1]} companies "
        f"({wide.shape[0]} filing dates) to {SHARES_FILE}\n"
        f"  no CIK match: {n_nocik} | no EDGAR facts: {n_nofacts} | errors: {n_err}"
    )
    return wide


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ingest shares outstanding from SEC EDGAR.")
    p.add_argument("--limit", type=int, default=None, help="only the first N tickers (for testing)")
    p.add_argument("--refresh", action="store_true", help="re-download cached companyfacts")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    main(limit=args.limit, refresh=args.refresh)
