"""Unit tests for the SEC EDGAR fundamentals extraction (pure parsing logic).

The live HTTP download is not exercised here (network-free suite); these tests
pin the transform from a companyfacts payload to tidy point-in-time rows, and
confirm the output schema round-trips through FundamentalsLoader.
"""

from __future__ import annotations

import pandas as pd

import qer.data.fundamentals as fund_mod
from qer.data.fundamentals import FundamentalsLoader
from scripts.ingest_fundamentals import extract_rows

FACTS = {
    "facts": {
        "us-gaap": {
            "StockholdersEquity": {
                "units": {
                    "USD": [
                        {
                            "end": "2020-12-31",
                            "val": 1000,
                            "filed": "2021-02-15",
                            "fp": "FY",
                            "frame": "CY2020",
                        },
                        {
                            "end": "2020-12-31",
                            "val": 1000,
                            "filed": "2022-02-15",
                            "fp": "FY",
                            "frame": "CY2020",
                        },  # comparative
                        {
                            "end": "2020-12-31",
                            "val": 1050,
                            "filed": "2022-06-01",
                            "fp": "FY",
                        },  # restatement
                        {
                            "end": "2021-12-31",
                            "val": 1200,
                            "filed": "2022-02-15",
                            "fp": "FY",
                            "frame": "CY2021",
                        },
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        {
                            "end": "2020-12-31",
                            "val": 5000,
                            "filed": "2021-02-15",
                            "fp": "FY",
                            "frame": "CY2020",
                        },
                    ]
                }
            },
            # no GrossProfit -> derive from Revenues - CostOfRevenue (annual only),
            # both tagged in the same filing (same accession + filed date)
            "Revenues": {
                "units": {
                    "USD": [
                        {
                            "end": "2020-12-31",
                            "val": 900,
                            "filed": "2021-02-15",
                            "accn": "acc-2020",
                            "fp": "FY",
                            "frame": "CY2020",
                        },
                        {
                            "end": "2020-09-30",
                            "val": 250,
                            "filed": "2020-11-01",
                            "accn": "acc-q3",
                            "fp": "Q3",
                            "frame": "CY2020Q3",
                        },
                    ]
                }
            },
            "CostOfRevenue": {
                "units": {
                    "USD": [
                        {
                            "end": "2020-12-31",
                            "val": 600,
                            "filed": "2021-02-15",
                            "accn": "acc-2020",
                            "fp": "FY",
                            "frame": "CY2020",
                        },
                    ]
                }
            },
        }
    }
}


def test_extract_rows_point_in_time_and_dedup():
    df = pd.DataFrame(extract_rows("TEST", FACTS))
    be = df[df.field == "book_equity"].sort_values("available_date")
    # available_date is the actual filed date, not end+lag
    assert str(be.iloc[0]["available_date"].date()) == "2021-02-15"
    # comparative (same value, later filing) dropped; restatement (new value) kept
    assert len(be) == 3
    assert 1050.0 in set(be["value"])


def test_extract_rows_derives_annual_gross_profit():
    df = pd.DataFrame(extract_rows("TEST", FACTS))
    gp = df[df.field == "gross_profit"]
    assert len(gp) == 1  # quarterly revenue excluded
    assert float(gp["value"].iloc[0]) == 300.0  # 900 - 600
    # availability is the (shared) filing date of the revenue+cost pairing
    assert str(gp.iloc[0]["available_date"].date()) == "2021-02-15"


def test_gross_profit_derivation_aligns_revenue_and_cost_by_filing():
    """A component restated in a later filing must not pair with the other
    component's original value (which would invent a never-filed gross profit)."""
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            # original 10-K
                            {
                                "end": "2020-12-31",
                                "val": 900,
                                "filed": "2021-02-15",
                                "accn": "orig",
                                "fp": "FY",
                                "frame": "CY2020",
                            },
                            # 10-K/A restates ONLY revenue, in a different accession
                            {
                                "end": "2020-12-31",
                                "val": 950,
                                "filed": "2022-03-01",
                                "accn": "amend",
                                "fp": "FY",
                            },
                        ]
                    }
                },
                "CostOfRevenue": {
                    "units": {
                        "USD": [
                            {
                                "end": "2020-12-31",
                                "val": 600,
                                "filed": "2021-02-15",
                                "accn": "orig",
                                "fp": "FY",
                                "frame": "CY2020",
                            },
                        ]
                    }
                },
            }
        }
    }
    gp = [r for r in extract_rows("TEST", facts) if r["field"] == "gross_profit"]
    values = {r["value"] for r in gp}
    # within-filing figure present; the phantom (restated 950 - original 600 = 350) absent
    assert 300.0 in values  # 900 - 600, both from the "orig" filing
    assert 350.0 not in values
    # the restated revenue has no paired cost in its own filing, so it is not derivable
    assert values == {300.0}


def test_extract_output_roundtrips_through_loader(tmp_path, monkeypatch):
    rows = extract_rows("TEST", FACTS)
    path = tmp_path / "fundamentals.parquet"
    pd.DataFrame(rows).to_parquet(path)
    monkeypatch.setattr(fund_mod, "FUNDAMENTALS_FILE", path)

    cal = pd.bdate_range("2021-01-01", "2022-12-31")
    panel = FundamentalsLoader().panel("book_equity", cal)
    # before the first filing: NaN; on/after 2021-02-15: 1000; after restatement: 1050
    assert pd.isna(panel.loc[pd.Timestamp("2021-02-01"), "TEST"])
    assert panel.loc[pd.Timestamp("2021-03-01"), "TEST"] == 1000.0
    assert panel.loc[pd.Timestamp("2022-07-01"), "TEST"] == 1050.0
