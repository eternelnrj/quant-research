"""Unit tests for the Phase 3 trials ledger."""

from __future__ import annotations

import pandas as pd
import pytest

from qer.graphs import trials as T


def test_log_and_read_one_trial(tmp_path):
    path = tmp_path / "trials.parquet"
    T.log_trial("abc123", "correlation", "pass", feature="eigenvector",
                metrics={"ic_ir": 0.5}, path=path)
    led = T.read_ledger(path)
    assert len(led) == 1
    assert led.iloc[0]["config_id"] == "abc123"
    assert led.iloc[0]["status"] == "pass"


def test_invalid_status_raises(tmp_path):
    with pytest.raises(ValueError, match="status must be"):
        T.log_trial("x", "correlation", "maybe", path=tmp_path / "t.parquet")


def test_append_and_distinct_count(tmp_path):
    path = tmp_path / "trials.parquet"
    T.log_trial("a", "correlation", "pass", path=path)
    T.log_trial("b", "leadlag", "fail", path=path)
    T.log_trial("a", "correlation", "fail", path=path)  # re-eval of 'a'
    led = T.read_ledger(path)
    assert len(led) == 3                      # append-only: every evaluation kept
    assert T.n_distinct_trials(path) == 2     # but only 2 distinct configs tried


def test_summary_counts_latest_status(tmp_path):
    path = tmp_path / "trials.parquet"
    T.log_trial("a", "correlation", "fail", path=path)
    T.log_trial("a", "correlation", "pass", path=path)  # latest wins -> pass
    T.log_trial("b", "leadlag", "fail", path=path)
    summ = T.summary(path).set_index("feature_class")
    assert summ.loc["correlation", "pass"] == 1
    assert summ.loc["correlation", "fail"] == 0
    assert summ.loc["leadlag", "fail"] == 1


def test_unregistered_trials_detected(tmp_path):
    path = tmp_path / "trials.parquet"
    grid = pd.DataFrame({"config_id": ["a", "b", "c"]})
    T.log_trial("a", "correlation", "pass", path=path)   # in grid
    T.log_trial("zzz", "correlation", "pass", path=path)  # NOT in grid
    assert T.unregistered_trials(grid, path=path) == {"zzz"}


def test_empty_ledger_is_well_formed(tmp_path):
    path = tmp_path / "none.parquet"
    assert T.read_ledger(path).empty
    assert T.n_distinct_trials(path) == 0
    assert T.summary(path).empty
    assert T.unregistered_trials(pd.DataFrame({"config_id": ["a"]}), path=path) == set()
