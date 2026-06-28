"""Minimal SEC EDGAR client, shared by the fundamentals and shares ingests.

Handles the bits both ingests need: the SEC-required User-Agent, ticker->CIK
resolution, fetching (and caching) a company's XBRL ``companyfacts``, and the
point-in-time helpers (use the filing's ``filed`` date; de-dup comparatives but
keep restatements). SEC requires a descriptive User-Agent and rate-limits to
<10 requests/sec - set ``QER_SEC_USER_AGENT`` to your contact info.
"""

from __future__ import annotations

import gzip
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
REQUEST_PAUSE_S = 0.13  # ~8 requests/sec, under SEC's 10/s limit


def require_user_agent() -> str:
    """Return the configured SEC User-Agent or exit with instructions."""
    ua = os.environ.get("QER_SEC_USER_AGENT", "")
    if not ua or "@" not in ua:
        raise SystemExit(
            "SEC requires a descriptive User-Agent with contact info. Set it:\n"
            '  export QER_SEC_USER_AGENT="Your Name your@email.com"'
        )
    return ua


def http_get_json(url: str, user_agent: str, retries: int = 3) -> dict:
    """GET a JSON document with the SEC-required User-Agent and gzip support."""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip"})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw.decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                raise  # caller skips
            time.sleep(1.0 * (attempt + 1))  # linear backoff
    raise last_err  # type: ignore[misc]


def load_ticker_cik_map(user_agent: str) -> dict[str, int]:
    """Map ticker -> CIK from SEC's company_tickers.json (upper-cased keys)."""
    data = http_get_json(SEC_TICKERS_URL, user_agent)
    return {str(row["ticker"]).upper(): int(row["cik_str"]) for row in data.values()}


def resolve_cik(ticker: str, cik_map: dict[str, int]) -> int | None:
    """Match a (Yahoo-format) ticker to a CIK, trying common symbol variants."""
    t = ticker.upper()
    for cand in (
        t,
        t.replace("-", "."),
        t.replace(".", "-"),
        t.replace("-", ""),
        t.replace(".", ""),
    ):
        if cand in cik_map:
            return cik_map[cand]
    return None


def fetch_company_facts(cik: int, user_agent: str, cache_dir: Path, refresh: bool) -> dict | None:
    """Download (and cache) a company's XBRL facts; None if EDGAR has none."""
    cache = cache_dir / f"CIK{cik:010d}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    try:
        facts = http_get_json(COMPANYFACTS_URL.format(cik=cik), user_agent)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    cache.write_text(json.dumps(facts))
    time.sleep(REQUEST_PAUSE_S)
    return facts


# ---------------------------------------------------------------------------
# Parsing helpers (pure)
# ---------------------------------------------------------------------------
def concept_entries(namespace_facts: dict, concept: str, unit: str | None = None) -> list[dict]:
    """Fact entries for a concept, preferring ``unit`` (e.g. 'USD', 'shares')."""
    node = namespace_facts.get(concept)
    if not node:
        return []
    units = node.get("units", {})
    if unit and unit in units:
        return units[unit]
    return next(iter(units.values()), [])


def available_date(entry: dict, lag: pd.Timedelta) -> pd.Timestamp:
    """Point-in-time date a figure became public: the filing's ``filed`` date.

    Falls back to ``end + lag`` only for the rare entry with no ``filed`` field.
    """
    filed = entry.get("filed")
    if filed:
        return pd.Timestamp(filed)
    return pd.Timestamp(entry["end"]) + lag


def dedup_point_in_time(raw_rows: list[dict], value_key: str = "value") -> list[dict]:
    """Keep the earliest disclosure of each distinct (period-end, value).

    Drops comparatives (same value re-reported later) but keeps restatements
    (a changed value filed later) as a new point-in-time observation. Rows must
    carry a private ``_end`` key; private keys are stripped from the output.

    Precondition: pass rows from a *single* (ticker, field) group. The dedup key
    is ``(_end, value)`` only - it does not include ticker or field - so the
    caller must pre-partition by those. Both current callers do: ``extract_shares``
    dedups one ticker's shares, and ``extract_rows`` dedups inside its per-field
    loop. Passing a mixed list (multiple tickers or fields at once) would
    silently collapse distinct observations that happen to share a period-end
    and value.
    """
    seen: set[tuple[str, float]] = set()
    out = []
    for r in sorted(raw_rows, key=lambda r: r["available_date"]):
        key = (r["_end"], round(float(r[value_key]), 2))
        if key in seen:
            continue
        seen.add(key)
        out.append({k: v for k, v in r.items() if not k.startswith("_")})
    return out
