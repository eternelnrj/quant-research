"""CLI: ingest 10-K Item 1 business text, embed it, and cache point-in-time embeddings.

For each universe ticker this walks its 10-K filings from EDGAR, extracts the Item 1
("Business") section, embeds it with a sentence-transformer, and writes one row per
(ticker, filing_date) to ``TEXT_EMBEDDINGS_FILE`` with columns ``ticker, filing_date,
emb_0 .. emb_{d-1}``. The Subphase 3.5 text factor reads that cache; without it, the factor
skips. A per-year coverage report is printed as the data-quality gate.

Requirements (by design this is the phase's largest data lift):
  * network access to EDGAR (SEC), via the existing ``qer.data.edgar`` client;
  * the optional ``text`` extra (sentence-transformers) for the embeddings.

Usage:
    python -m scripts.ingest_10k_text
"""

from __future__ import annotations

import pandas as pd

from qer.config import TEXT_EMBEDDINGS_FILE
from qer.data.edgar import http_get_json, load_ticker_cik_map, require_user_agent, resolve_cik
from qer.graphs.textsim import embed_texts, extract_item1

_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"


def _iter_10k_filings(cik: int, ua: str):
    """Yield (filing_date, primary_document_url) for each 10-K of a company."""
    subs = http_get_json(_SUBMISSIONS.format(cik=cik), ua)
    recent = subs.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accns = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    for form, date, acc, doc in zip(forms, dates, accns, docs):
        if form == "10-K" and doc:
            url = _ARCHIVE.format(cik=cik, acc_nodash=acc.replace("-", ""), doc=doc)
            yield pd.Timestamp(date), url


def _fetch_universe() -> list[str]:
    from qer.data.loader import DataLoader
    loader = DataLoader()
    cal = loader.close.index
    universe: set[str] = set()
    for t in cal[::63]:                      # sample the calendar quarterly
        universe.update(loader.get_universe(t))
    return sorted(universe)


def main() -> pd.DataFrame:
    import requests

    ua = require_user_agent()
    cik_map = load_ticker_cik_map(ua)
    tickers = _fetch_universe()

    records: list[dict] = []
    texts: list[str] = []
    meta: list[tuple[str, pd.Timestamp]] = []
    for ticker in tickers:
        cik = resolve_cik(ticker, cik_map)
        if cik is None:
            continue
        for filing_date, url in _iter_10k_filings(cik, ua):
            html = requests.get(url, headers={"User-Agent": ua}, timeout=30).text
            item1 = extract_item1(html)
            if len(item1) < 500:             # too little parseable text -> skip (coverage gate)
                continue
            texts.append(item1[:20000])      # cap length for the encoder
            meta.append((ticker, filing_date))

    if not texts:
        print("No 10-K Item 1 text ingested (check network / user agent).")
        return pd.DataFrame()

    embeddings = embed_texts(texts)          # (n_filings x dim), needs the 'text' extra
    dim = embeddings.shape[1]
    for (ticker, filing_date), vec in zip(meta, embeddings):
        row = {"ticker": ticker, "filing_date": filing_date}
        row.update({f"emb_{i}": float(vec[i]) for i in range(dim)})
        records.append(row)

    out = pd.DataFrame.from_records(records)
    TEXT_EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(TEXT_EMBEDDINGS_FILE)

    by_year = out.assign(year=out["filing_date"].dt.year).groupby("year")["ticker"].nunique()
    print(f"Wrote {len(out)} filing embeddings (dim {dim}) to {TEXT_EMBEDDINGS_FILE}")
    print("Coverage (distinct tickers with a filing, by year):")
    for year, n in by_year.items():
        print(f"  {year}: {n}")
    return out


if __name__ == "__main__":
    main()
