"""Unit tests for the Subphase 3.2 trailing-window builder and rebalance schedule."""

from __future__ import annotations

from qer.graphs.windows import rebalance_dates, trailing_return_matrix


def test_trailing_matrix_ends_at_as_of_and_respects_window(synthetic_loader):
    cal = synthetic_loader.close.index
    as_of = cal[300]
    tm = trailing_return_matrix(synthetic_loader, as_of, window=120, min_obs=120)
    assert tm.index.max() == as_of            # window ends at as_of
    assert (tm.index <= as_of).all()          # strictly backward: the look-ahead boundary
    assert len(tm) == 120
    assert tm.shape[1] == 30                   # all names have full history by date 300
    assert not tm.isna().to_numpy().any()      # min_obs=window => complete columns


def test_trailing_matrix_no_lookahead(synthetic_loader):
    cal = synthetic_loader.close.index
    as_of = cal[200]
    tm = trailing_return_matrix(synthetic_loader, as_of, window=60, min_obs=60)
    # nothing dated after as_of can appear, by construction
    assert tm.index.max() == as_of
    assert tm.index.max() < cal[-1]


def test_trailing_matrix_drops_short_history(synthetic_loader):
    # very early as_of: fewer than `window` return rows exist, so min_obs=window
    # drops every name (insufficient history) rather than zero-filling.
    cal = synthetic_loader.close.index
    as_of = cal[30]
    tm = trailing_return_matrix(synthetic_loader, as_of, window=120, min_obs=120)
    assert tm.shape[1] == 0


def test_rebalance_dates_are_month_end_trading_days(synthetic_loader):
    cal = synthetic_loader.close.index
    snaps = rebalance_dates(cal, freq="M")
    assert len(snaps) > 0
    assert snaps.is_monotonic_increasing
    assert set(snaps).issubset(set(cal))       # every snapshot is a real trading day
    # each snapshot is the last calendar day of its month
    for t in snaps:
        same_month = cal[(cal.year == t.year) & (cal.month == t.month)]
        assert t == same_month.max()
