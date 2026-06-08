"""
Unit tests for qer.data.loader.DataLoader.

Updated for the current DataLoader, which takes NO constructor arguments and
resolves all storage paths from module-level constants imported from
``qer.config`` (RAW_DIR, PRICES_DIR, WIDE_DIR, MEMBERSHIP_FILE). The earlier
version of this file constructed ``DataLoader(root=...)``, an API the loader no
longer exposes; every test here instead redirects those module globals at a
temporary directory via monkeypatch.

Tests are pure: no network calls, no real data dependency. Each test builds a
small, hand-crafted data layout under ``tmp_path`` and points the loader at it.

Storage layout the loader expects (and that these tests reproduce):
    <root>/processed/membership.parquet   - membership table
    <root>/raw/prices/<TICKER>.parquet    - one file per ticker
    <root>/wide/<field>.parquet           - cached wide matrices (built lazily)

Two field-naming facts the tests depend on, both true of the current loader:
  * ``loader.close`` reads the ``adj close`` field, so its on-disk cache is
    ``wide/adj close.parquet`` (note the space). Ticker files therefore carry
    an ``adj close`` column equal to ``close``.
  * ``cross_section(field="close")`` reads the plain ``close`` field.

Run with:
    pytest tests/unit/test_loader.py -v
"""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import qer.data.loader as loader_mod
from qer.data.loader import DataLoader

SENTINEL_END = pd.Timestamp("2099-12-31")


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------
# Approach: each test gets a fresh tmp_path. The `paths` fixture builds the
# data/ sub-tree there and monkeypatches the loader's module-level path
# constants at it, so a bare `DataLoader()` reads from the temp layout. Fresh
# tmp_path per test means no wide-cache bleed between tests.


@pytest.fixture
def paths(tmp_path, monkeypatch) -> SimpleNamespace:
    """Create the temp data layout and point the loader's globals at it."""
    raw = tmp_path / "raw"
    prices = raw / "prices"
    wide = tmp_path / "wide"
    membership_file = tmp_path / "processed" / "membership.parquet"
    for d in (prices, wide, membership_file.parent):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(loader_mod, "RAW_DIR", raw)
    monkeypatch.setattr(loader_mod, "PRICES_DIR", prices)
    monkeypatch.setattr(loader_mod, "WIDE_DIR", wide)
    monkeypatch.setattr(loader_mod, "MEMBERSHIP_FILE", membership_file)

    return SimpleNamespace(
        raw=raw,
        prices=prices,
        wide=wide,
        membership_file=membership_file,
        close_cache=wide / "adj close.parquet",
    )


def _write_membership(paths: SimpleNamespace, rows: list[dict]) -> None:
    """Write a membership.parquet file from a list of row dicts."""
    df = pd.DataFrame(rows)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    df.to_parquet(paths.membership_file)


def _write_ticker_file(
    paths: SimpleNamespace,
    ticker: str,
    dates: pd.DatetimeIndex,
    close: list[float] | np.ndarray,
    *,
    open_: list[float] | None = None,
    volume: list[float] | None = None,
) -> None:
    """Write a per-ticker parquet under <root>/raw/prices/<TICKER>.parquet.

    Includes an ``adj close`` column (equal to ``close``) because the loader's
    ``close`` accessor reads the adjusted field, plus the OHLCV fields the
    other accessors use.
    """
    n = len(dates)
    df = pd.DataFrame(
        {
            "open": open_ if open_ is not None else close,
            "high": close,
            "low": close,
            "close": close,
            "adj close": close,
            "volume": volume if volume is not None else [1_000_000] * n,
        },
        index=dates,
    )
    df.index.name = "date"
    df.to_parquet(paths.prices / f"{ticker}.parquet")


@pytest.fixture
def empty_loader(paths) -> DataLoader:
    """Loader with a membership file but no per-ticker files."""
    _write_membership(
        paths,
        [
            {"ticker": "AAPL", "start_date": "2010-01-01", "end_date": SENTINEL_END},
        ],
    )
    return DataLoader()


@pytest.fixture
def small_loader(paths) -> DataLoader:
    """Loader with 3 tickers, 10 trading days, varied membership."""
    dates = pd.bdate_range("2020-01-01", periods=10)

    # Ticker A: full 10 days, prices 100..109
    _write_ticker_file(paths, "TICKA", dates, list(range(100, 110)))
    # Ticker B: full 10 days, prices 50..59
    _write_ticker_file(paths, "TICKB", dates, list(range(50, 60)))
    # Ticker C: only first 5 days (delisted mid-window)
    _write_ticker_file(paths, "TICKC", dates[:5], [200, 210, 205, 220, 230])

    _write_membership(
        paths,
        [
            {"ticker": "TICKA", "start_date": "2019-01-01", "end_date": SENTINEL_END},
            {"ticker": "TICKB", "start_date": "2019-06-01", "end_date": SENTINEL_END},
            {"ticker": "TICKC", "start_date": "2019-01-01", "end_date": "2020-01-09"},
        ],
    )
    return DataLoader()


# ---------------------------------------------------------------------------
# Membership tests
# ---------------------------------------------------------------------------


def test_membership_loads_and_has_expected_columns(small_loader):
    m = small_loader.membership
    assert set(m.columns) == {"ticker", "start_date", "end_date"}
    assert len(m) == 3


def test_membership_dates_are_timestamps(small_loader):
    m = small_loader.membership
    assert pd.api.types.is_datetime64_any_dtype(m["start_date"])
    assert pd.api.types.is_datetime64_any_dtype(m["end_date"])


def test_membership_is_cached(small_loader):
    # Two calls should return the SAME object (identity, not equality)
    # because the property caches into self._membership.
    m1 = small_loader.membership
    m2 = small_loader.membership
    assert m1 is m2


def test_nat_start_filled_with_min_timestamp(paths):
    # A row with NaT start should be filled with pd.Timestamp.min so
    # date-range queries don't need NaT handling.
    _write_membership(
        paths,
        [
            {"ticker": "OLD", "start_date": pd.NaT, "end_date": "2015-01-01"},
            {"ticker": "NEW", "start_date": "2020-01-01", "end_date": SENTINEL_END},
        ],
    )
    loader = DataLoader()
    m = loader.membership

    old_row = m[m["ticker"] == "OLD"].iloc[0]
    assert old_row["start_date"] == pd.Timestamp.min
    assert not pd.isna(old_row["start_date"])


def test_nat_end_filled_with_sentinel(paths):
    # A row with a missing end date should be filled with the 2099 sentinel,
    # i.e. treated as "still in the index".
    _write_membership(
        paths,
        [
            {"ticker": "STILLIN", "start_date": "2020-01-01", "end_date": pd.NaT},
        ],
    )
    loader = DataLoader()
    row = loader.membership.iloc[0]
    assert row["end_date"] == SENTINEL_END


def test_get_universe_basic(small_loader):
    # All three tickers active on 2020-01-02
    universe = small_loader.get_universe("2020-01-02")
    assert set(universe) == {"TICKA", "TICKB", "TICKC"}


def test_get_universe_excludes_removed_tickers(small_loader):
    # TICKC was removed on 2020-01-09. The half-open convention means
    # it is NOT active ON 2020-01-09 itself.
    universe = small_loader.get_universe("2020-01-09")
    assert "TICKC" not in universe
    assert set(universe) == {"TICKA", "TICKB"}


def test_get_universe_includes_on_start_date(small_loader):
    # TICKB starts on 2019-06-01. It IS active on its start date.
    universe = small_loader.get_universe("2019-06-01")
    assert "TICKB" in universe


def test_get_universe_accepts_various_date_types(small_loader):
    """The date argument should accept str, pd.Timestamp, datetime, date."""
    as_str = small_loader.get_universe("2020-01-02")
    as_timestamp = small_loader.get_universe(pd.Timestamp("2020-01-02"))
    as_datetime = small_loader.get_universe(datetime(2020, 1, 2))
    as_date = small_loader.get_universe(date(2020, 1, 2))

    assert as_str == as_timestamp == as_datetime == as_date


def test_get_universe_returns_list(small_loader):
    universe = small_loader.get_universe("2020-01-02")
    assert isinstance(universe, list)


def test_get_universe_before_any_membership(small_loader):
    # Date before any ticker joined: empty universe.
    universe = small_loader.get_universe("1990-01-01")
    assert universe == []


def test_all_tickers_ever_is_sorted_unique_union(small_loader):
    result = small_loader.all_tickers_ever()
    assert result == ["TICKA", "TICKB", "TICKC"]
    # Sorted
    assert result == sorted(result)


# ---------------------------------------------------------------------------
# Wide-matrix tests
# ---------------------------------------------------------------------------


def test_close_returns_dataframe(small_loader):
    close = small_loader.close
    assert isinstance(close, pd.DataFrame)


def test_close_has_ticker_columns(small_loader):
    close = small_loader.close
    assert set(close.columns) == {"TICKA", "TICKB", "TICKC"}


def test_close_has_datetime_index(small_loader):
    close = small_loader.close
    assert pd.api.types.is_datetime64_any_dtype(close.index)


def test_close_values_match_input(small_loader):
    close = small_loader.close
    # TICKA had close prices 100..109 over 10 business days. The close
    # accessor reads the adjusted field, which equals close in these fixtures.
    assert close["TICKA"].iloc[0] == 100
    assert close["TICKA"].iloc[9] == 109


def test_close_outer_join_produces_nan_for_short_history(small_loader):
    """TICKC has data only for first 5 days; later days should be NaN."""
    close = small_loader.close
    assert close["TICKC"].iloc[:5].notna().all()
    assert close["TICKC"].iloc[5:].isna().all()


def test_unknown_field_raises(small_loader):
    with pytest.raises(ValueError, match="Unknown field"):
        small_loader._load_wide("not_a_real_field")


def test_missing_field_in_file_raises(paths):
    """A known field absent from a per-ticker file raises KeyError on build."""
    dates = pd.bdate_range("2020-01-01", periods=5)
    # Write a file WITHOUT an 'adj close' column, then ask for .close.
    df = pd.DataFrame(
        {"open": range(5), "close": range(5), "volume": [1] * 5},
        index=dates,
    )
    df.index.name = "date"
    df.to_parquet(paths.prices / "BAD.parquet")
    _write_membership(
        paths, [{"ticker": "BAD", "start_date": "2019-01-01", "end_date": SENTINEL_END}]
    )
    loader = DataLoader()
    with pytest.raises(KeyError, match="not found"):
        _ = loader.close


def test_no_raw_files_raises(empty_loader):
    with pytest.raises(FileNotFoundError, match="No per-ticker files"):
        _ = empty_loader.close


def test_cache_written_on_first_build(small_loader, paths):
    # Trigger first load - builds and caches. The close accessor reads the
    # 'adj close' field, so the cache file is named accordingly.
    _ = small_loader.close
    assert paths.close_cache.exists()


def test_cache_used_on_second_load(small_loader, paths):
    # First load: cold build.
    _ = small_loader.close
    # Overwrite the cache with a sentinel value, clear the in-memory cache,
    # then re-load. If the loader reads from the disk cache (instead of
    # rebuilding from per-ticker files), it'll see the sentinel.
    sentinel = pd.DataFrame(
        {"SENTINEL": [42.0]},
        index=pd.to_datetime(["2020-01-01"]),
    )
    sentinel.index.name = "date"
    paths.close_cache.unlink()
    sentinel.to_parquet(paths.close_cache)

    # Clear the in-memory cache so the loader reads from disk again.
    small_loader._wide.clear()

    wide = small_loader.close
    assert list(wide.columns) == ["SENTINEL"]
    assert wide.iloc[0, 0] == 42.0


def test_in_memory_cache_avoids_disk_read(small_loader, paths):
    """After first load, in-memory cache short-circuits disk reads."""
    wide1 = small_loader.close
    # Delete the disk cache. If in-memory cache works, next read still succeeds.
    paths.close_cache.unlink()
    wide2 = small_loader.close
    assert wide1 is wide2


def test_open_and_volume_accessors(small_loader):
    """The open and volume properties also work."""
    assert isinstance(small_loader.open, pd.DataFrame)
    assert isinstance(small_loader.volume, pd.DataFrame)
    # Volume values came from the default 1_000_000 fill.
    assert (small_loader.volume["TICKA"].dropna() == 1_000_000).all()


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def test_log_returns_match_manual_computation(small_loader):
    returns = small_loader.get_returns(kind="log")
    # TICKA: 100, 101, 102, ... so log return is log(101/100), log(102/101), ...
    expected = np.log(101 / 100)
    assert returns["TICKA"].iloc[1] == pytest.approx(expected)


def test_simple_returns_match_manual_computation(small_loader):
    returns = small_loader.get_returns(kind="simple")
    expected = (101 - 100) / 100
    assert returns["TICKA"].iloc[1] == pytest.approx(expected)


def test_log_returns_first_row_is_nan(small_loader):
    """No prior price, so first day has no return."""
    returns = small_loader.get_returns(kind="log")
    assert returns.iloc[0].isna().all()


def test_returns_have_same_shape_as_close(small_loader):
    close = small_loader.close
    returns = small_loader.get_returns()
    assert returns.shape == close.shape


def test_unknown_kind_raises(small_loader):
    with pytest.raises(ValueError, match="kind must be"):
        small_loader.get_returns(kind="bogus")


# ---------------------------------------------------------------------------
# Cross-section
# ---------------------------------------------------------------------------


def test_cross_section_returns_series_indexed_by_ticker(small_loader):
    cs = small_loader.cross_section("2020-01-02")
    assert isinstance(cs, pd.Series)
    assert set(cs.index) <= {"TICKA", "TICKB", "TICKC"}


def test_cross_section_excludes_nan_prices(small_loader):
    # 2020-01-09: TICKC has NaN price (only 5 days of history). It's also
    # excluded from the universe on this date via the membership table, so
    # even before NaN handling kicks in, it's gone. Test that the result
    # is consistent regardless.
    cs = small_loader.cross_section("2020-01-09")
    assert "TICKC" not in cs.index


def test_cross_section_unknown_date_raises(small_loader):
    with pytest.raises(KeyError, match="not in price data"):
        small_loader.cross_section("1990-01-01")


def test_cross_section_tolerates_missing_ticker_files(paths):
    """A ticker in the membership table but absent from raw files
    should be silently skipped, not crash."""
    dates = pd.bdate_range("2020-01-01", periods=5)
    _write_ticker_file(paths, "REAL", dates, [100, 101, 102, 103, 104])
    _write_membership(
        paths,
        [
            {"ticker": "REAL", "start_date": "2019-01-01", "end_date": SENTINEL_END},
            # GHOST is in the universe but has no raw file
            {"ticker": "GHOST", "start_date": "2019-01-01", "end_date": SENTINEL_END},
        ],
    )
    loader = DataLoader()
    cs = loader.cross_section("2020-01-02")
    # Should contain REAL, not GHOST, and not crash.
    assert "REAL" in cs.index
    assert "GHOST" not in cs.index


def test_cross_section_default_field_is_close(small_loader):
    cs_default = small_loader.cross_section("2020-01-02")
    cs_explicit = small_loader.cross_section("2020-01-02", field="close")
    pd.testing.assert_series_equal(cs_default, cs_explicit)


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------
# rebuild_wide_cache() now uses the module-level WIDE_DIR (previously it
# referenced a self.wide_dir attribute __init__ never set, raising
# AttributeError). These tests guard that fix.


def test_rebuild_wide_cache_removes_old_caches(small_loader, paths):
    # Populate caches.
    _ = small_loader.close
    _ = small_loader.volume
    assert paths.close_cache.exists()
    assert (paths.wide / "volume.parquet").exists()

    # Rebuild. Old files should be unlinked and replaced.
    small_loader.rebuild_wide_cache()

    # Files should still exist (rebuild also re-creates them) and the
    # in-memory cache should be re-populated.
    assert paths.close_cache.exists()
    assert "adj close" in small_loader._wide


def test_rebuild_clears_in_memory_cache_then_repopulates(small_loader):
    _ = small_loader.close
    assert "adj close" in small_loader._wide
    small_loader.rebuild_wide_cache()
    # All fields should now be in cache after rebuild.
    assert set(small_loader._wide.keys()) == set(DataLoader.FIELDS)
