"""Unit tests for the SEC EDGAR shares-outstanding ingest (pure parsing).

The live HTTP download is not exercised (network-free suite); these pin the
companyfacts -> point-in-time shares transform and confirm the pivoted output
feeds DataLoader.market_cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import qer.data.loader as loader_mod
from qer.data.loader import DataLoader
from scripts.ingest_shares import extract_shares, shares_panel


def test_extract_shares_point_in_time_and_dedup():
    facts = {
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"end": "2020-12-31", "val": 1_000_000, "filed": "2021-02-15"},
                            {
                                "end": "2020-12-31",
                                "val": 1_000_000,
                                "filed": "2022-02-15",
                            },  # comparative -> drop
                            {
                                "end": "2021-12-31",
                                "val": 1_100_000,
                                "filed": "2022-02-15",
                            },  # buyback -> new row
                        ]
                    }
                }
            }
        }
    }
    rows = sorted(extract_shares("TEST", facts), key=lambda r: r["available_date"])
    assert len(rows) == 2  # comparative dropped
    assert str(rows[0]["available_date"].date()) == "2021-02-15"  # filed date, not end+lag
    assert rows[0]["shares"] == 1_000_000
    assert rows[1]["shares"] == 1_100_000


def test_extract_shares_falls_back_to_usgaap_concept():
    facts = {
        "facts": {
            "us-gaap": {
                "CommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"end": "2020-12-31", "val": 500_000, "filed": "2021-03-01"},
                        ]
                    }
                }
            }
        }
    }
    rows = extract_shares("XYZ", facts)
    assert len(rows) == 1 and rows[0]["shares"] == 500_000


def test_shares_panel_breaks_same_date_ties_by_latest_period_end():
    """Two filings sharing an available_date must collapse to the most recent
    period-end's value, deterministically and regardless of input order."""
    rows = [
        # later period-end (current) listed FIRST to defeat naive input-order "last"
        {
            "ticker": "AAA",
            "available_date": "2021-02-15",
            "period_end": "2020-12-31",
            "shares": 950,
        },
        {
            "ticker": "AAA",
            "available_date": "2021-02-15",
            "period_end": "2020-09-30",
            "shares": 900,
        },
    ]
    panel = shares_panel(rows)
    assert panel.loc[pd.Timestamp("2021-02-15"), "AAA"] == 950  # latest period_end wins
    # order-independence: shuffling the input must not change the result
    assert shares_panel(rows[::-1]).loc[pd.Timestamp("2021-02-15"), "AAA"] == 950


def test_shares_output_feeds_market_cap(tmp_path, monkeypatch):
    prices = tmp_path / "raw" / "prices"
    prices.mkdir(parents=True)
    processed = tmp_path / "processed"
    processed.mkdir()
    membership = processed / "m.parquet"
    shares_file = tmp_path / "raw" / "shares.parquet"

    monkeypatch.setattr(loader_mod, "PRICES_DIR", prices)
    monkeypatch.setattr(loader_mod, "WIDE_DIR", tmp_path / "wide")
    monkeypatch.setattr(loader_mod, "MEMBERSHIP_FILE", membership)
    monkeypatch.setattr(loader_mod, "SHARES_FILE", shares_file)

    idx = pd.bdate_range("2021-01-01", periods=20)
    raw_close = np.linspace(100, 120, 20)
    pd.DataFrame(
        {
            "Open": raw_close,
            "High": raw_close,
            "Low": raw_close,
            "Close": raw_close,
            "Adj Close": raw_close * 0.9,
            "Volume": np.full(20, 1e6),
        },
        index=idx,
    ).to_parquet(prices / "AAA.parquet")
    pd.DataFrame(
        {"ticker": ["AAA"], "start_date": [idx[0]], "end_date": [pd.Timestamp("2099-12-31")]}
    ).to_parquet(membership)

    # sparse shares panel like fetch_shares writes (one row, ffilled by the loader)
    pd.DataFrame({"AAA": [2_000_000.0]}, index=[idx[0]]).to_parquet(shares_file)

    loader = DataLoader()
    mc = loader.market_cap["AAA"]
    # market cap uses RAW close * shares (ffilled), not adjusted close
    assert np.allclose(mc.values, raw_close * 2_000_000.0)
