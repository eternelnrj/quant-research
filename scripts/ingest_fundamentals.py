"""CLI: ingest fundamentals from SEC EDGAR (for value and quality factors).

Downloads each S&P 500 member's XBRL ``companyfacts`` from EDGAR (via the shared
``qer.data.edgar`` client), extracts a few US-GAAP concepts, and writes the tidy
parquet that ``FundamentalsLoader`` reads:

    ticker | available_date | field | value

Fields extracted:
  - book_equity   <- StockholdersEquity (balance-sheet stock; all filings kept)
  - total_assets  <- Assets             (balance-sheet stock; all filings kept)
  - gross_profit  <- GrossProfit, else derived as Revenues - CostOfRevenue
                     (income-statement flow; ANNUAL figures only, to avoid
                      mixing quarterly and annual magnitudes)

Known simplifications (documented deliberately): comparatives re-reported in
later filings are de-duplicated by keeping the earliest disclosure of each
distinct (period-end, value); restatements with a changed value are kept as new
point-in-time rows; quarterly flows are dropped. The gross-profit derivation
pairs revenue and cost only *within the same filing* (by accession number), so
a restated component is never mixed with an original one - a period is derivable
only when one filing tags both revenue and cost. Validate against a handful of
known 10-Ks before trusting it.

Usage:
    QER_SEC_USER_AGENT="Your Name your@email.com" python -m scripts.ingest_fundamentals
    QER_SEC_USER_AGENT="..." python -m scripts.ingest_fundamentals --limit 5
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from qer.config import (
    FUNDAMENTALS_DIR,
    FUNDAMENTALS_FILE,
    FUNDAMENTALS_LAG_DAYS,
    MEMBERSHIP_FILE,
)
from qer.data.edgar import (
    available_date,
    concept_entries,
    dedup_point_in_time,
    fetch_company_facts,
    load_ticker_cik_map,
    require_user_agent,
    resolve_cik,
)

# field -> how to read it. "stock" = balance-sheet level (keep all filings);
# "flow" = income-statement amount over a period (keep annual only).
FIELD_SPEC = {
    "book_equity": {
        "kind": "stock",
        "concepts": [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
    },
    "total_assets": {"kind": "stock", "concepts": ["Assets"]},
    "gross_profit": {
        "kind": "flow",
        "concepts": ["GrossProfit"],
        # fallbacks tried in order if GrossProfit is absent: (revenue, cost)
        "derive": [
            ("Revenues", "CostOfGoodsAndServicesSold"),
            ("RevenueFromContractWithCustomerExcludingAssessedTax", "CostOfRevenue"),
            ("Revenues", "CostOfRevenue"),
            ("SalesRevenueNet", "CostOfGoodsSold"),
        ],
    },
}


def _is_annual(entry: dict) -> bool:
    frame = entry.get("frame")
    if frame is not None:
        return "Q" not in frame  # CY2019 annual, CY2019Q3 quarterly
    return entry.get("fp") == "FY"


def extract_rows(ticker: str, facts: dict, lag_days: int = FUNDAMENTALS_LAG_DAYS) -> list[dict]:
    """Turn one company's EDGAR facts into tidy point-in-time rows."""
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    lag = pd.Timedelta(days=lag_days)
    rows: list[dict] = []

    for field, spec in FIELD_SPEC.items():
        collected: list[dict] = []
        # 1) direct concept(s)
        for concept in spec["concepts"]:
            for e in concept_entries(usgaap, concept, "USD"):
                if "end" not in e or e.get("val") is None:
                    continue
                if spec["kind"] == "flow" and not _is_annual(e):
                    continue
                collected.append(
                    {
                        "ticker": ticker,
                        "field": field,
                        "available_date": available_date(e, lag),
                        "value": float(e["val"]),
                        "_end": e["end"],
                    }
                )
            if collected:
                break  # first concept that yields data wins
        # 2) derivation fallback (gross profit = revenue - cost)
        if not collected and "derive" in spec:
            for rev_c, cost_c in spec["derive"]:
                # Pair revenue and cost ONLY within the same filing, keyed by
                # (accession number, period end). Keying by `end` alone would
                # let a restated revenue from a 10-K/A be paired with the
                # original cost from the 10-K, producing a gross-profit figure
                # that appeared in no single filing. Each filing's own
                # self-consistent figure becomes a point-in-time row; the
                # subsequent dedup keeps restatements and drops comparatives.
                rev = {
                    (e.get("accn"), e["end"]): e
                    for e in concept_entries(usgaap, rev_c, "USD")
                    if _is_annual(e) and "end" in e and e.get("val") is not None
                }
                cost = {
                    (e.get("accn"), e["end"]): e
                    for e in concept_entries(usgaap, cost_c, "USD")
                    if _is_annual(e) and "end" in e and e.get("val") is not None
                }
                for key in set(rev) & set(cost):
                    re_, ce = rev[key], cost[key]
                    if re_.get("val") is None or ce.get("val") is None:
                        continue
                    collected.append(
                        {
                            "ticker": ticker,
                            "field": field,
                            "available_date": available_date(
                                re_, lag
                            ),  # same filing -> one filed date
                            "value": float(re_["val"]) - float(ce["val"]),
                            "_end": re_["end"],
                        }
                    )
                if collected:
                    break
        rows.extend(dedup_point_in_time(collected))
    return rows


def main(limit: int | None = None, refresh: bool = False) -> pd.DataFrame:
    user_agent = require_user_agent()
    if not MEMBERSHIP_FILE.exists():
        raise SystemExit(f"{MEMBERSHIP_FILE} not found - run `make membership` first.")

    FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    FUNDAMENTALS_FILE.parent.mkdir(parents=True, exist_ok=True)

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
        all_rows.extend(extract_rows(ticker, facts))
        n_ok += 1
        if i % 25 == 0:
            print(f"  [{i}/{len(tickers)}] {ticker}: cumulative {len(all_rows)} rows")

    if not all_rows:
        raise SystemExit(
            "No fundamentals extracted - check the User-Agent and network access to data.sec.gov."
        )

    df = pd.DataFrame(all_rows)
    df["available_date"] = pd.to_datetime(df["available_date"])
    df = df.sort_values(["ticker", "field", "available_date"]).reset_index(drop=True)
    df.to_parquet(FUNDAMENTALS_FILE)

    print(
        f"\nWrote {len(df):,} rows for {n_ok} companies to {FUNDAMENTALS_FILE}\n"
        f"  no CIK match: {n_nocik} | no EDGAR facts: {n_nofacts} | errors: {n_err}\n"
        f"  fields: {dict(df['field'].value_counts())}"
    )
    return df


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ingest fundamentals from SEC EDGAR.")
    p.add_argument("--limit", type=int, default=None, help="only the first N tickers (for testing)")
    p.add_argument("--refresh", action="store_true", help="re-download cached companyfacts")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    main(limit=args.limit, refresh=args.refresh)
