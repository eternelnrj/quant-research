# src/qer/config.py
from pathlib import Path

import pandas as pd

# Data paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

PRICES_DIR = RAW_DIR / "prices"
WIKIPEDIA_DIR = RAW_DIR / "wikipedia"

SECTORS_FILE = RAW_DIR / "sectors.parquet"
SPY_FILE = RAW_DIR / "SPY.parquet"
SHARES_FILE = RAW_DIR / "shares.parquet"  # shares outstanding (for market cap / size)

S_AND_P500_CURRENT = WIKIPEDIA_DIR / "sp500_current.parquet"
S_AND_P500_CHANGES = WIKIPEDIA_DIR / "sp500_changes.parquet"

MEMBERSHIP_FILE = PROCESSED_DIR / "sp500_membership.parquet"

# Phase 2: fundamentals (value/quality) and factor reports.
FUNDAMENTALS_DIR = RAW_DIR / "fundamentals"
FUNDAMENTALS_FILE = (
    FUNDAMENTALS_DIR / "fundamentals.parquet"
)  # tidy: ticker, available_date, field, value
FF5_FILE = RAW_DIR / "ff5.parquet"  # Fama-French 5 factors (+ RF), for exposure regressions

WIDE_DIR = DATA_DIR / "wide"
AUDIT_DIR = DATA_DIR / "audit"
FACTOR_REPORT_DIR = DATA_DIR / "factors"  # per-factor IC/portfolio report artifacts

SENTINEL_END = pd.Timestamp("2099-12-31")

# Point-in-time fundamentals lag: only use a filing once it is `available_date`
# old enough. 45-90 days is the roadmap convention; 60 is a reasonable default.
FUNDAMENTALS_LAG_DAYS = 60


# ingest prices
START_DATE = "2008-01-01"
END_DATE = "2025-12-31"
