"""Unit tests for the Fama-French 5 ingest (pure parsing).

The live download from Ken French's library is not exercised (network-free
suite); these pin the CSV->decimal-frame transform and confirm the output feeds
ff5_exposures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.diagnostics.exposures import ff5_exposures
from scripts.fetch_ff5 import parse_ff5_csv

SAMPLE = """This file was created by CMPT_ME_BEME_RETS using the CRSP database.

,Mkt-RF,SMB,HML,RMW,CMA,RF
20200102,   0.50,  -0.10,   0.20,   0.05,  -0.15,  0.006
20200103,  -0.30,   0.15,  -0.05,  -0.02,   0.10,  0.006
20200106,   1.20,   0.40,  -0.30, -99.99,   0.05,  0.006
Copyright 2020 Kenneth R. French
"""


def test_parse_ff5_columns_and_decimal_scaling():
    df = parse_ff5_csv(SAMPLE)
    assert list(df.columns) == ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    assert abs(df.iloc[0]["mkt_rf"] - 0.005) < 1e-12  # 0.50% -> 0.005
    assert abs(df.iloc[0]["rf"] - 0.00006) < 1e-12  # 0.006% -> decimal
    assert np.isnan(df.iloc[2]["rmw"])  # -99.99 sentinel -> NaN


def test_parse_ff5_skips_cruft_and_sorts_dates():
    df = parse_ff5_csv(SAMPLE)
    assert len(df) == 3  # description + copyright lines ignored
    assert list(df.index) == sorted(df.index)
    assert str(df.index[0].date()) == "2020-01-02"


def test_parse_ff5_output_feeds_ff5_exposures():
    # enough observations for a non-singular 5-factor + intercept regression
    idx = pd.bdate_range("2021-01-01", periods=120)
    rng = np.random.default_rng(0)
    pct = rng.normal(0, 0.5, (120, 6))  # in "percent" like the raw file
    header = ",Mkt-RF,SMB,HML,RMW,CMA,RF\n"
    body = "".join(
        f"{d.strftime('%Y%m%d')}," + ",".join(f"{v:.4f}" for v in row) + "\n"
        for d, row in zip(idx, pct)
    )
    df = parse_ff5_csv("header cruft\n\n" + header + body)
    ls = pd.Series(rng.normal(0, 0.01, 120), index=idx)
    res = ff5_exposures(ls, df)
    assert set(res["betas"]) == {"mkt_rf", "smb", "hml", "rmw", "cma"}
    assert res["n"] == 120 and np.isfinite(res["alpha"])
