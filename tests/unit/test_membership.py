"""
Unit tests for qer.universe.membership.

Tests cover the pure logic — _resolve_ticker, _event_sort_key,
_net_open_at_end — directly, and exercise build_membership_table by
monkeypatching fetch_sp500_tables to return canned dataframes rather
than hitting Wikipedia. This keeps tests fast (<200ms each) and
network-free.

Each tricky case from the design discussion has a dedicated test:
  * Same-day swap (FOXA pattern) produces two intervals at the swap date.
  * Recycled ticker (AGN pattern) produces intervals with NaT then date starts.
  * Pre-log member (LEHM pattern) gets NaT start, real end.
  * Successive entities under one symbol (T/AT&T pattern) gets two intervals.
  * Rename collapse (DISCA+DISCK -> WBD pattern) handles same-day duplicates.
  * Acquisition without log removal (CCR pattern) closes via TICKER_EXITS.
  * Unclassified dangling tickers fall back to log_end_date.

Run with:
    pytest tests/unit/test_membership.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from qer.universe.membership import (
    _event_sort_key,
    _net_open_at_end,
    _resolve_ticker,
    build_membership_table,
    validate,
)

# pytestmark = pytest.mark.skip(reason="Tests have not been checked yet.")


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_current(tickers, date_added=None):
    """Build a fake current-constituents DataFrame matching Wikipedia layout."""
    if date_added is None:
        date_added = ["2000-01-01"] * len(tickers)
    return pd.DataFrame(
        {
            "ticker": list(tickers),
            "Date added": date_added,
        }
    )


def _make_changes(events):
    """Build a fake changes-log DataFrame from a list of (date, added, removed) tuples.

    Use '' for absent ticker (matching the build_membership convention).
    """
    return pd.DataFrame(
        events,
        columns=["date", "added", "removed"],
    ).assign(date=lambda d: pd.to_datetime(d["date"]))


@pytest.fixture
def patch_fetch(monkeypatch):
    """Return a function that installs a fake fetch_sp500_tables.

    Usage:
        def test_something(patch_fetch):
            patch_fetch(current_df, changes_df)
            df = build_membership_table()
    """

    def _patch(current_df, changes_df):
        def fake_fetch(force_refresh=False):
            return current_df.copy(), changes_df.copy()

        monkeypatch.setattr(
            "qer.universe.membership.fetch_sp500_tables",
            fake_fetch,
        )

    return _patch


# ---------------------------------------------------------------------------
# _resolve_ticker
# ---------------------------------------------------------------------------


def test_resolve_ticker_unmapped_returns_input():
    assert _resolve_ticker("AAPL", renames={}) == "AAPL"


def test_resolve_ticker_single_step():
    assert _resolve_ticker("FB", renames={"FB": "META"}) == "META"


def test_resolve_ticker_multi_step():
    """Multi-step chain WLP -> ANTM -> ELV walks to the end."""
    renames = {"WLP": "ANTM", "ANTM": "ELV"}
    assert _resolve_ticker("WLP", renames=renames) == "ELV"
    assert _resolve_ticker("ANTM", renames=renames) == "ELV"
    assert _resolve_ticker("ELV", renames=renames) == "ELV"


def test_resolve_ticker_cycle_safe():
    """A pathological cycle in the map terminates rather than infinite-looping."""
    renames = {"A": "B", "B": "A"}  # would loop forever without cycle protection
    result = _resolve_ticker("A", renames=renames)
    # We don't care which of A or B comes out - just that it terminates.
    assert result in {"A", "B"}


def test_resolve_ticker_empty_string():
    """Empty input passes through unchanged."""
    assert _resolve_ticker("", renames={"FB": "META"}) == ""


# ---------------------------------------------------------------------------
# _event_sort_key
# ---------------------------------------------------------------------------


def test_event_sort_key_orders_by_date():
    e1 = (pd.Timestamp("2020-01-01"), "added")
    e2 = (pd.Timestamp("2020-06-01"), "added")
    assert sorted([e2, e1], key=_event_sort_key) == [e1, e2]


def test_event_sort_key_removed_first_on_same_day():
    """Same-day swap: 'removed' must come before 'added' to make
    a clean two-interval split rather than a zero-length interval."""
    date = pd.Timestamp("2020-01-01")
    added = (date, "added")
    removed = (date, "removed")
    assert sorted([added, removed], key=_event_sort_key) == [removed, added]


def test_event_sort_key_nat_dates_first():
    """NaT-dated events sort to the very front."""
    real = (pd.Timestamp("2020-01-01"), "added")
    nat = (pd.NaT, "added")
    assert sorted([real, nat], key=_event_sort_key) == [nat, real]


# ---------------------------------------------------------------------------
# _net_open_at_end
# ---------------------------------------------------------------------------


def test_net_open_at_end_empty():
    assert _net_open_at_end([]) is False


def test_net_open_at_end_added_only():
    assert _net_open_at_end([(pd.Timestamp("2020-01-01"), "added")]) is True


def test_net_open_at_end_added_then_removed():
    events = [
        (pd.Timestamp("2020-01-01"), "added"),
        (pd.Timestamp("2021-01-01"), "removed"),
    ]
    assert _net_open_at_end(events) is False


def test_net_open_at_end_two_cycles_open():
    """Added, removed, added - last event leaves it open."""
    events = [
        (pd.Timestamp("2020-01-01"), "added"),
        (pd.Timestamp("2021-01-01"), "removed"),
        (pd.Timestamp("2022-01-01"), "added"),
    ]
    assert _net_open_at_end(events) is True


def test_net_open_at_end_removed_only_returns_false():
    """A bare removal with no preceding addition nets to zero."""
    events = [(pd.Timestamp("2020-01-01"), "removed")]
    assert _net_open_at_end(events) is False


# ---------------------------------------------------------------------------
# build_membership_table - basic shape
# ---------------------------------------------------------------------------


def test_build_simple_one_current_member(patch_fetch):
    """The simplest case: one ticker in current set, no changes log
    activity. Should produce one row with SENTINEL_END end."""
    current = _make_current(["AAPL"], date_added=["1982-12-12"])
    changes = _make_changes(
        [
            # one dummy event so log_end_date is defined
            ("2020-01-01", "XYZ", "ABC"),
        ]
    )
    patch_fetch(current, changes)

    df = build_membership_table(yahoo_format=False)

    aapl = df[df["ticker"] == "AAPL"]
    assert len(aapl) == 1
    assert pd.isna(aapl.iloc[0]["end_date"])  # SENTINEL_END
    assert aapl.iloc[0]["start_date"] == pd.Timestamp("1982-12-12")


def test_build_added_then_removed(patch_fetch):
    """A ticker added and later removed by the log produces one closed
    interval. The ticker is NOT in the current set."""
    current = _make_current(["AAPL"])
    changes = _make_changes(
        [
            ("2015-01-01", "XYZ", "AAA"),  # XYZ added, AAA removed
            ("2018-06-15", "PQR", "XYZ"),  # XYZ removed by PQR
        ]
    )
    patch_fetch(current, changes)

    df = build_membership_table(yahoo_format=False)

    xyz = df[df["ticker"] == "XYZ"]
    assert len(xyz) == 1
    assert xyz.iloc[0]["start_date"] == pd.Timestamp("2015-01-01")
    assert xyz.iloc[0]["end_date"] == pd.Timestamp("2018-06-15")


# ---------------------------------------------------------------------------
# build_membership_table - edge cases from the design discussion
# ---------------------------------------------------------------------------


def test_build_same_day_swap_produces_two_intervals(patch_fetch):
    """FOXA pattern: same-day removal and addition under the same ticker
    yields two intervals meeting at the swap date, not a zero-length one."""
    current = _make_current(["FOXA"])
    changes = _make_changes(
        [
            ("2010-01-01", "FOXA", ""),  # old FOXA added pre-swap
            ("2019-03-19", "FOXA", "FOXA"),  # same-day swap: new in, old out
        ]
    )
    patch_fetch(current, changes)

    df = build_membership_table(yahoo_format=False)

    foxa = df[df["ticker"] == "FOXA"].sort_values("start_date")
    assert len(foxa) == 2
    # First interval: 2010 -> 2019-03-19
    assert foxa.iloc[0]["start_date"] == pd.Timestamp("2010-01-01")
    assert foxa.iloc[0]["end_date"] == pd.Timestamp("2019-03-19")
    # Second interval: 2019-03-19 -> SENTINEL_END
    assert foxa.iloc[1]["start_date"] == pd.Timestamp("2019-03-19")
    assert pd.isna(foxa.iloc[1]["end_date"])  # SENTINEL_END


def test_build_pre_log_member_gets_nat_start(patch_fetch):
    """LEHM pattern: a ticker that was a long-time member predating the log,
    then was removed during log coverage. Has no addition event in the log.
    Should produce a (NaT, removal_date) interval."""
    current = _make_current(["AAPL"])
    changes = _make_changes(
        [
            ("2008-09-15", "BAC", "LEHM"),  # LEHM removed; never had an add event
            # need at least one other row for log_end_date
            ("2010-01-01", "XYZ", ""),
        ]
    )
    patch_fetch(current, changes)

    df = build_membership_table(yahoo_format=False)

    lehm = df[df["ticker"] == "LEHM"]
    assert len(lehm) == 1
    assert pd.isna(lehm.iloc[0]["start_date"])
    assert lehm.iloc[0]["end_date"] == pd.Timestamp("2008-09-15")


def test_build_recycled_ticker_two_entities(patch_fetch):
    """AGN pattern: a ticker recycled across two entities. The log has
    two 'removed' events for the same ticker. We should produce two
    intervals: (NaT, first_removal) and (first_removal, second_removal)."""
    current = _make_current(["AAPL"])
    changes = _make_changes(
        [
            ("2015-03-23", "X", "AGN"),  # old Allergan removed
            ("2020-05-12", "Y", "AGN"),  # Allergan plc removed
        ]
    )
    patch_fetch(current, changes)

    df = build_membership_table(yahoo_format=False)

    agn = df[df["ticker"] == "AGN"].sort_values("end_date")
    assert len(agn) == 2
    # First interval: NaT start, first removal
    assert pd.isna(agn.iloc[0]["start_date"])
    assert agn.iloc[0]["end_date"] == pd.Timestamp("2015-03-23")
    # Second interval: starts at the first removal, ends at the second
    assert agn.iloc[1]["start_date"] == pd.Timestamp("2015-03-23")
    assert agn.iloc[1]["end_date"] == pd.Timestamp("2020-05-12")


def test_build_t_pattern_succession_under_same_ticker(monkeypatch):
    """T pattern: ticker is in the current set but log only records a
    removal (old AT&T leaving in 2005). The new entity adopted the ticker
    without an explicit add event. Should produce two intervals: the
    pre-2005 interval and a 2005-onward interval ending at SENTINEL_END."""
    current = _make_current(["T"], date_added=["1983-11-30"])
    changes = _make_changes(
        [
            ("2005-11-18", "AMZN", "T"),  # only the removal
            ("2010-01-01", "XYZ", ""),  # filler so log_end_date is post-2005
        ]
    )

    def fake_fetch(force_refresh=False):
        return current.copy(), changes.copy()

    monkeypatch.setattr(
        "qer.universe.membership.fetch_sp500_tables",
        fake_fetch,
    )

    df = build_membership_table(yahoo_format=False)

    t = df[df["ticker"] == "T"].sort_values("end_date")
    assert len(t) == 2
    # First interval: original AT&T's history through 2005-11-18
    assert t.iloc[0]["end_date"] == pd.Timestamp("2005-11-18")
    # Second interval: from the removal date through sentinel
    assert t.iloc[1]["start_date"] == pd.Timestamp("2005-11-18")
    assert pd.isna(t.iloc[1]["end_date"])  # SENTINEL_END


def test_build_acquisition_closed_via_ticker_exits(patch_fetch, monkeypatch):
    """CCR pattern: ticker is dangling-open in the log (added, never
    removed) and not in the current set. With an entry in TICKER_EXITS,
    it gets closed at the exit date."""
    current = _make_current(["AAPL"])
    changes = _make_changes(
        [
            ("2000-01-01", "CCR", ""),  # CCR added; no removal in log
            ("2010-01-01", "XYZ", ""),
        ]
    )
    patch_fetch(current, changes)

    # Inject an explicit exit date for CCR.
    monkeypatch.setattr(
        "qer.universe.membership.TICKER_EXITS",
        {"CCR": "2008-07-01"},
    )

    df = build_membership_table(yahoo_format=False)

    ccr = df[df["ticker"] == "CCR"]
    assert len(ccr) == 1
    assert ccr.iloc[0]["start_date"] == pd.Timestamp("2000-01-01")
    assert ccr.iloc[0]["end_date"] == pd.Timestamp("2008-07-01")


def test_build_unclassified_dangling_falls_back_to_log_end(patch_fetch, monkeypatch):
    """A dangling-open ticker not in current_set, not in TICKER_EXITS,
    and not a known rename gets closed at log_end_date."""
    current = _make_current(["AAPL"])
    changes = _make_changes(
        [
            ("2005-01-01", "OBSCURE", ""),
            ("2020-12-31", "XYZ", ""),  # this is log_end_date
        ]
    )
    patch_fetch(current, changes)
    monkeypatch.setattr("qer.universe.membership.TICKER_EXITS", {})

    df = build_membership_table(yahoo_format=False)

    obscure = df[df["ticker"] == "OBSCURE"]
    assert len(obscure) == 1
    assert obscure.iloc[0]["end_date"] == pd.Timestamp("2020-12-31")


def test_build_rename_collapses_into_modern_ticker(patch_fetch, monkeypatch):
    """FB -> META rename: log entries under FB are reattributed to META."""
    current = _make_current(["META"], date_added=["2013-12-23"])
    changes = _make_changes(
        [
            ("2013-12-23", "FB", ""),
            ("2020-01-01", "XYZ", ""),
        ]
    )
    patch_fetch(current, changes)
    monkeypatch.setattr(
        "qer.universe.membership.TICKER_RENAMES",
        {"FB": "META"},
    )

    df = build_membership_table(yahoo_format=False)

    # FB should NOT appear; META should have a single open interval
    # starting at FB's addition date.
    assert (df["ticker"] == "FB").sum() == 0

    meta = df[df["ticker"] == "META"]
    assert len(meta) == 1
    assert meta.iloc[0]["start_date"] == pd.Timestamp("2013-12-23")
    assert pd.isna(meta.iloc[0]["end_date"])  # SENTINEL_END


def test_build_same_day_rename_duplicates_dedup(patch_fetch, monkeypatch):
    """WBD pattern: DISCA and DISCK both rename to WBD on the same date,
    so without dedup the log would produce two 'removed' events for WBD
    on the merger date - which would cause a zero-length interval bug.
    The dedup step in build_membership_table should handle this."""
    current = _make_current(["WBD"], date_added=["2022-04-11"])
    changes = _make_changes(
        [
            ("2010-01-01", "DISCA", ""),
            ("2010-01-01", "DISCK", ""),
            ("2022-04-11", "WBD", "DISCA"),  # rename event
            ("2022-04-11", "WBD", "DISCK"),  # same date, same ticker after rename
        ]
    )
    patch_fetch(current, changes)
    monkeypatch.setattr(
        "qer.universe.membership.TICKER_RENAMES",
        {"DISCA": "WBD", "DISCK": "WBD"},
    )

    df = build_membership_table(yahoo_format=False)

    wbd = df[df["ticker"] == "WBD"].sort_values("end_date")
    # Should have one closed interval (DISCA/DISCK history) and one open
    # interval (modern WBD), not zero-length anything.
    assert len(wbd) == 2
    assert wbd.iloc[0]["end_date"] == pd.Timestamp("2022-04-11")
    assert wbd.iloc[1]["start_date"] == pd.Timestamp("2022-04-11")
    assert pd.isna(wbd.iloc[1]["end_date"])  # SENTINEL_END


def test_build_yahoo_format_converts_dot_to_dash(patch_fetch):
    """BRK.B in Wikipedia format becomes BRK-B for yfinance compatibility."""
    current = _make_current(["BRK.B"], date_added=["2010-02-16"])
    changes = _make_changes([("2020-01-01", "XYZ", "")])
    patch_fetch(current, changes)

    df = build_membership_table(yahoo_format=True)

    assert "BRK-B" in set(df["ticker"])
    assert "BRK.B" not in set(df["ticker"])


def test_build_yahoo_format_false_keeps_dots(patch_fetch):
    """When yahoo_format=False, the Wikipedia ticker convention is preserved."""
    current = _make_current(["BRK.B"], date_added=["2010-02-16"])
    changes = _make_changes([("2020-01-01", "XYZ", "")])
    patch_fetch(current, changes)

    df = build_membership_table(yahoo_format=False)

    assert "BRK.B" in set(df["ticker"])


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_passes_on_well_formed_table():
    """Sanity check: a hand-built well-formed table passes validation."""
    # Build a synthetic large table that satisfies all validate's checks.
    # 500 currently-active tickers + a handful of historical closed ones.
    n_active = 500
    rows = []
    for i in range(n_active):
        rows.append(
            {
                "ticker": f"TICK{i:04d}",
                "start_date": pd.Timestamp("2000-01-01"),
                "end_date": None,
            }
        )
    # Add a few closed intervals to make it realistic
    for i in range(5):
        rows.append(
            {
                "ticker": f"OLD{i:03d}",
                "start_date": pd.Timestamp("2000-01-01"),
                "end_date": pd.Timestamp("2015-06-15"),
            }
        )
    df = pd.DataFrame(rows)
    current_set = {f"TICK{i:04d}" for i in range(n_active)}

    # Should not raise
    validate(df, current_set)


def test_validate_rejects_malformed_interval():
    """An interval with start >= end fails validation."""
    df = pd.DataFrame(
        [
            {
                "ticker": "BAD",
                "start_date": pd.Timestamp("2020-01-01"),
                "end_date": pd.Timestamp("2019-01-01"),  # before start!
            }
        ]
    )
    with pytest.raises(AssertionError, match="intervals with start >= end"):
        validate(df, current_set_yahoo={"BAD"})


def test_validate_rejects_mismatched_current_set():
    """If a ticker is in current_set_yahoo but not marked still-in,
    validation fails."""
    df = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "start_date": pd.Timestamp("2000-01-01"),
                "end_date": pd.Timestamp("2020-01-01"),  # closed, but should be open
            }
        ]
    )
    with pytest.raises(AssertionError, match="not marked still-in"):
        validate(df, current_set_yahoo={"AAPL"})


# ---------------------------------------------------------------------------
# Regression: class-share tickers must not break in-build validation
# ---------------------------------------------------------------------------


def test_build_with_class_share_ticker_validates(patch_fetch):
    """A current member with a dotted (class-share) ticker like BRK.B builds
    and maps to its Yahoo symbol (BRK-B) while staying marked still-in.

    Regression test for the validation-format bug: validation used to compare
    the Yahoo-formatted table (BRK-B) against a Wikipedia-formatted current set
    (BRK.B), so every class share registered as "still-in but not in current
    set". Validation now lives in the CLI and is fed a Yahoo-formatted current
    set (see scripts/build_membership.py); here we assert the build itself maps
    the class share correctly.
    """
    current = _make_current(["AAPL", "BRK.B"], date_added=["1982-12-12", "2010-02-16"])
    changes = _make_changes(
        [
            ("2015-01-01", "XYZ", ""),  # XYZ added
            ("2018-06-15", "", "XYZ"),  # XYZ removed -> clean closed interval
        ]
    )
    patch_fetch(current, changes)

    df = build_membership_table(yahoo_format=True)

    # Yahoo mapping applied, and the class share is marked still-in.
    assert "BRK.B" not in set(df["ticker"])
    brk = df[df["ticker"] == "BRK-B"]
    assert len(brk) == 1
    assert pd.isna(brk.iloc[0]["end_date"])  # open / still in index


def test_validate_accepts_class_share_when_format_matches():
    """validate passes when the current set is in the SAME format as the table.

    Builds a full-size (~500) Yahoo-formatted table including a class share and
    validates against a Yahoo-formatted current set - the contract the CLI now
    honours.
    """
    rows = []
    for i in range(499):
        rows.append(
            {"ticker": f"TICK{i:04d}", "start_date": pd.Timestamp("2000-01-01"), "end_date": None}
        )
    rows.append({"ticker": "BRK-B", "start_date": pd.Timestamp("2010-02-16"), "end_date": None})
    df = pd.DataFrame(rows)
    current_set_yahoo = {f"TICK{i:04d}" for i in range(499)} | {"BRK-B"}

    validate(df, current_set_yahoo)  # must not raise


# ---------------------------------------------------------------------------
# Regression: a fresh add on the latest log date must not crash the build
# ---------------------------------------------------------------------------


def test_fresh_add_on_log_end_date_is_skipped_not_zero_length(patch_fetch):
    """A ticker added on the most recent change date, with no removal and not
    yet in the current-constituents snapshot, must be skipped - never emitted
    as a zero-length (start == end) interval.

    Regression test for the FLEX/MRVL build crash: such tickers used to be
    closed at log_end_date, which equalled their own add date, producing an
    interval validation rightly rejected.
    """
    current = _make_current(["AAPL", "MSFT"], date_added=["1982-12-12", "1994-06-01"])
    changes = _make_changes(
        [
            ("2015-01-01", "XYZ", ""),
            ("2018-06-15", "", "XYZ"),
            ("2026-06-22", "FLEX", ""),  # added on the latest log date, no removal
            ("2026-06-22", "MRVL", ""),
        ]
    )
    patch_fetch(current, changes)

    df = build_membership_table(yahoo_format=True)

    both = df.dropna(subset=["start_date", "end_date"])
    zero_len = both[both["start_date"] >= both["end_date"]]
    assert zero_len.empty, f"zero-length intervals leaked: {list(zero_len['ticker'])}"
    # The unreconciled fresh adds are skipped, not present.
    assert not ({"FLEX", "MRVL"} & set(df["ticker"]))
