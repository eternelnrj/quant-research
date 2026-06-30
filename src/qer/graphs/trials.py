"""Subphase 3.1: the trials ledger.

An append-only audit log -- one row per configuration *evaluated*, pass or fail
-- written to ``data/graphs/trials.parquet``. Its job is to make the true number
of trials impossible to lose: when you later report "these features survived",
the ledger is the receipt for how many you actually tried, and the grid
(:mod:`qer.graphs.grid`) is the pre-registered denominator you feed the deflated
Sharpe. :func:`unregistered_trials` cross-checks the two, surfacing any evaluation
that was run outside the pre-registered grid -- the classic p-hacking smell.
"""

from __future__ import annotations

import json

import pandas as pd

from qer.config import GRAPH_TRIALS_FILE

_VALID_STATUS = {"pass", "fail"}
LEDGER_COLUMNS = ["config_id", "feature_class", "feature", "status", "logged_at", "metrics"]


def log_trial(
    config_id: str,
    feature_class: str,
    status: str,
    feature: str = "",
    metrics: dict | None = None,
    path=GRAPH_TRIALS_FILE,
) -> None:
    """Append one evaluated configuration to the ledger.

    ``status`` must be ``"pass"`` or ``"fail"``. ``metrics`` is any small JSON-
    serialisable dict of headline numbers (IC IR, net Sharpe, deflated-Sharpe
    ratio, ...). The write is read-modify-write append; concurrent writers
    are not supported (last writer wins), which is fine for a single research loop.
    """
    if status not in _VALID_STATUS:
        raise ValueError(f"status must be one of {sorted(_VALID_STATUS)}, got {status!r}")
    row = {
        "config_id": config_id,
        "feature_class": feature_class,
        "feature": feature,
        "status": status,
        "logged_at": pd.Timestamp.now("UTC"),
        "metrics": json.dumps(metrics or {}, default=str),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        led = pd.read_parquet(path)
        led = pd.concat([led, pd.DataFrame([row])], ignore_index=True)
    else:
        led = pd.DataFrame([row], columns=LEDGER_COLUMNS)
    led.to_parquet(path)


def read_ledger(path=GRAPH_TRIALS_FILE) -> pd.DataFrame:
    """Read the full ledger (empty frame with the right columns if none yet)."""
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return pd.read_parquet(path)


def n_distinct_trials(path=GRAPH_TRIALS_FILE) -> int:
    """Number of *distinct* configurations evaluated (re-runs don't double-count)."""
    led = read_ledger(path)
    return int(led["config_id"].nunique()) if len(led) else 0


def summary(path=GRAPH_TRIALS_FILE) -> pd.DataFrame:
    """Pass/fail counts per feature class over the distinct configs evaluated."""
    led = read_ledger(path)
    if led.empty:
        return pd.DataFrame(columns=["feature_class", "pass", "fail", "total"])
    # one row per config: its latest status
    latest = led.sort_values("logged_at").drop_duplicates("config_id", keep="last")
    tab = latest.groupby(["feature_class", "status"]).size().unstack(fill_value=0).reset_index()
    for col in ("pass", "fail"):
        if col not in tab.columns:
            tab[col] = 0
    tab["total"] = tab["pass"] + tab["fail"]
    return tab[["feature_class", "pass", "fail", "total"]]


def unregistered_trials(grid: pd.DataFrame, path=GRAPH_TRIALS_FILE) -> set[str]:
    """``config_id``s in the ledger that are NOT in the pre-registered grid.

    A non-empty result means configurations were evaluated outside the grid --
    either the grid needs (deliberately) re-registering, or something was tried
    off the books. Either way it must be reconciled before trusting any
    trial-count-discounted significance number.
    """
    led = read_ledger(path)
    if led.empty:
        return set()
    return set(led["config_id"]) - set(grid["config_id"])
