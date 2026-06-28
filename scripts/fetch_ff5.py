"""CLI: download the Fama-French 5 daily factors (for FF5 exposure regressions).

Fetches the daily FF5 (2x3) factor file from Ken French's data library, parses
it into decimal returns, and writes the parquet that ``ff5_exposures`` and
notebook 05 read at ``config.FF5_FILE``:

    index: date (DatetimeIndex)
    columns: mkt_rf, smb, hml, rmw, cma, rf   (decimal returns, NOT percent)

Ken French publishes the factors in percent; this divides by 100 so they are on
the same scale as the daily long-short returns they are regressed against. The
sentinel missing-value codes (-99.99, -999) are mapped to NaN.

(If you'd rather not parse the raw file, ``pandas_datareader``'s "famafrench"
reader fetches the same data; this script avoids that extra dependency and uses
only the stdlib, matching the other ingest scripts.)

Usage:
    python -m scripts.fetch_ff5
"""

from __future__ import annotations

import io
import re
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from qer.config import FF5_FILE

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
_DATE_ROW = re.compile(r"^\s*(\d{8})\s*,(.*)")


def parse_ff5_csv(text: str) -> pd.DataFrame:
    """Parse Ken French's daily FF5 CSV text into a date-indexed decimal frame."""
    lines = text.splitlines()
    header = next(
        (ln for ln in lines if ln.lstrip().startswith(",") and "Mkt-RF" in ln), None
    )
    if header is None:
        raise ValueError("FF5 header row (',Mkt-RF,...') not found - format changed?")
    cols = [c.strip().lower().replace("-", "_") for c in header.split(",")[1:] if c.strip()]

    dates, rows = [], []
    for ln in lines:
        m = _DATE_ROW.match(ln)
        if not m:
            continue  # skip description/copyright lines and any annual section
        vals = [v.strip() for v in m.group(2).split(",") if v.strip() != ""]
        if len(vals) != len(cols):
            continue
        dates.append(pd.Timestamp(m.group(1)))
        rows.append([float(v) for v in vals])
    if not rows:
        raise ValueError("No daily FF5 rows parsed - format changed?")

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=cols)
    df = df.replace([-99.99, -999.0], np.nan) / 100.0  # percent -> decimal; sentinels -> NaN
    df.index.name = "date"
    return df.sort_index()


def main() -> pd.DataFrame:
    print("Downloading FF5 daily factors from Ken French's data library ...")
    with urllib.request.urlopen(FF5_URL, timeout=60) as resp:  # noqa: S310 - fixed trusted URL
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        text = z.read(name).decode("latin-1")

    df = parse_ff5_csv(text)
    FF5_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FF5_FILE)
    print(
        f"Wrote {len(df):,} daily rows "
        f"({df.index.min().date()} .. {df.index.max().date()}) to {FF5_FILE}\n"
        f"  columns: {list(df.columns)}"
    )
    return df


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
