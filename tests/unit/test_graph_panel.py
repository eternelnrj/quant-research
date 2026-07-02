"""Unit tests for the Subphase 3.2 feature-panel engine and neutralisation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.graphs.panel import build_feature_panel, neutralize_cross_section
from qer.graphs.windows import rebalance_dates, trailing_return_matrix

WINDOW = 120


def _mean_return(rw: pd.DataFrame, universe: list) -> pd.Series:
    return rw.mean()


def test_panel_is_daily_ffilled_and_nan_before_first_snapshot(synthetic_loader):
    cal = synthetic_loader.close.index
    panel = build_feature_panel(synthetic_loader, _mean_return, window=WINDOW, freq="M")
    assert panel.index.equals(cal)                         # daily calendar
    assert set(panel.columns).issubset(set(synthetic_loader.close.columns))
    # before the first snapshot with enough history, the panel is NaN
    assert panel.loc[cal[0]].isna().all()
    # and there is at least one fully populated row later
    assert panel.notna().all(axis=1).any()


def test_panel_value_comes_from_the_most_recent_snapshot(synthetic_loader):
    cal = synthetic_loader.close.index
    panel = build_feature_panel(synthetic_loader, _mean_return, window=WINDOW, freq="M")
    snaps = [t for t in rebalance_dates(cal, freq="M") if t in cal]
    # choose two consecutive snapshots both past the warm-up
    populated = [t for t in snaps if panel.loc[t].notna().any()]
    t_i, t_next = populated[1], populated[2]
    mid = cal[(cal > t_i) & (cal < t_next)][0]

    # ffill: a day between rebuilds equals the most recent snapshot, not the next
    np.testing.assert_allclose(panel.loc[mid].to_numpy(), panel.loc[t_i].to_numpy())
    assert not np.allclose(panel.loc[mid].to_numpy(), panel.loc[t_next].to_numpy())

    # the snapshot value is exactly the window statistic computed at t_i (no lookahead)
    rw = trailing_return_matrix(synthetic_loader, t_i, window=WINDOW, min_obs=WINDOW)
    expected = rw.mean()
    got = panel.loc[t_i].reindex(expected.index)
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-9)


def test_caching_roundtrips(synthetic_loader, tmp_path):
    kw = dict(window=WINDOW, freq="M", cache_dir=tmp_path, name="meanret")
    p1 = build_feature_panel(synthetic_loader, _mean_return, **kw)   # cold: writes cache
    assert any(tmp_path.rglob("*.parquet"))                          # cache populated
    p2 = build_feature_panel(synthetic_loader, _mean_return, **kw)   # warm: reads cache
    pd.testing.assert_frame_equal(p1, p2)


# ---- neutralisation (no loader needed; build panels directly) ------------

def _panels(seed=0, n_dates=6, n_names=20):
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    names = [f"T{i:02d}" for i in range(n_names)]
    rng = np.random.default_rng(seed)
    control = pd.DataFrame(rng.normal(size=(n_dates, n_names)), index=dates, columns=names)
    return dates, names, control, rng


def test_neutralize_removes_a_pure_control():
    _, _, control, _ = _panels()
    feature = control.copy()                                # feature IS the control
    resid = neutralize_cross_section(feature, by={"size": control}, rank=True, min_names=10)
    vals = resid.to_numpy()
    assert np.nanmax(np.abs(vals)) < 1e-8                   # neutralises to ~0


def test_neutralize_preserves_an_independent_signal():
    _, _, control, rng = _panels(seed=3)
    feature = pd.DataFrame(rng.normal(size=control.shape),
                           index=control.index, columns=control.columns)
    resid = neutralize_cross_section(feature, by={"size": control}, rank=True, min_names=10)
    # an independent feature is not annihilated: residual keeps real dispersion
    assert np.nanstd(resid.to_numpy()) > 1.0


def test_neutralize_skips_thin_cross_sections():
    dates = pd.bdate_range("2020-01-01", periods=2)
    names = [f"T{i}" for i in range(5)]                     # only 5 names < min_names
    f = pd.DataFrame(np.arange(10).reshape(2, 5), index=dates, columns=names, dtype=float)
    resid = neutralize_cross_section(f, by={"c": f}, min_names=10)
    assert resid.isna().to_numpy().all()                   # too thin -> left NaN
