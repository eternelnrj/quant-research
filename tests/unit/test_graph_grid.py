"""Unit tests for the Phase 3 pre-registered configuration grid."""

from __future__ import annotations

import pandas as pd
import pytest

from qer.graphs import grid as G


def test_enumerate_is_deterministic_and_counted():
    a = G.enumerate_configs()
    b = G.enumerate_configs()
    assert [c["config_id"] for c in a] == [c["config_id"] for c in b]
    # 72 centrality + 18 community + 15 lead-lag + 3 text = 108 for the default grid
    assert len(a) == 108
    counts = pd.Series([c["feature_class"] for c in a]).value_counts().to_dict()
    assert counts == {"correlation": 90, "leadlag": 15, "text": 3}


def test_config_ids_are_unique():
    ids = [c["config_id"] for c in G.enumerate_configs()]
    assert len(set(ids)) == len(ids)


def test_smaller_spec_changes_the_count():
    small = G.GridSpec(windows=(120,), sparsifiers=("mst",), centralities=("degree",),
                       centrality_transforms=("level",), communities=("louvain",),
                       leadlag_lags=(1,), leadlag_features=("in_degree",), knn=(5,),
                       text_features=("neighbour_return",))
    cfgs = G.enumerate_configs(small)
    # 1 centrality + 1 community + 1 lead-lag + 1 text
    assert len(cfgs) == 4
    assert G.n_trials(small) == 4


def test_register_then_load_roundtrip(tmp_path):
    path = tmp_path / "grid.parquet"
    written = G.register_grid(path=path)
    loaded = G.load_grid(path=path)
    assert set(written["config_id"]) == set(loaded["config_id"])
    assert G.n_trials(path) == len(written) == 108


def test_reregistering_same_spec_is_noop(tmp_path):
    path = tmp_path / "grid.parquet"
    G.register_grid(path=path)
    again = G.register_grid(path=path)  # must not raise
    assert len(again) == 108


def test_grid_drift_is_refused(tmp_path):
    path = tmp_path / "grid.parquet"
    G.register_grid(path=path)
    drifted = G.GridSpec(windows=(120, 180))  # different config set
    with pytest.raises(ValueError, match="write-once"):
        G.register_grid(drifted, path=path)
    # explicit overwrite is allowed
    out = G.register_grid(drifted, path=path, overwrite=True)
    assert len(out) < 108


def test_load_grid_missing_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="graphs-register"):
        G.load_grid(path=tmp_path / "nope.parquet")
