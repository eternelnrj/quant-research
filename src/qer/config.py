# src/qer/config.py
from pathlib import Path

import pandas as pd

# Data paths
ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

PRICES_DIR = RAW_DIR / "prices"
WIKIPEDIA_DIR = RAW_DIR / "wikipedia"

SECTORS_FILE = RAW_DIR / "sectors.parquet"
SPY_FILE = RAW_DIR / "SPY.parquet"

S_AND_P500_CURRENT = WIKIPEDIA_DIR / "sp500_current.parquet"
S_AND_P500_CHANGES = WIKIPEDIA_DIR / "sp500_changes.parquet"

MEMBERSHIP_FILE = PROCESSED_DIR / "sp500_membership.parquet"


WIDE_DIR = DATA_DIR / "wide"
AUDIT_DIR = DATA_DIR / "audit"

SENTINEL_END = pd.Timestamp("2099-12-31")


# ingest prices
START_DATE = "2008-01-01"
END_DATE = "2025-12-31"
