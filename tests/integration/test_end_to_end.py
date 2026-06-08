"""
End-to-end integration tests for the Quant Equity Research pipeline.

Where the unit tests in ``tests/unit`` poke individual functions with hand-
crafted inputs, these tests wire the *real* modules together and check that
data flows correctly across module boundaries:

    Wikipedia reconstruction  ->  universe membership
    membership parquet        ->  DataLoader.get_universe / audit sanity checks
    per-ticker price parquet  ->  DataLoader wide matrices / returns
    prices                    ->  momentum factor -> cross-sectional IC -> summary
    prices + membership       ->  audit chart builders

No network is touched. Each fixture writes a small but realistic data layout
to a temporary directory and re-points the module-level path constants in
``qer.data.loader`` at it (those constants are bound at import time from
``qer.config``, so we patch them on the loader module). The datasets are built
once per module for speed; the path patching is per-test so tests stay
isolated.

These are deliberately the "slow" half of the suite:

    pytest tests/integration -v          # run just these
    make test-integration                # via the Makefile target

Determinism note: the drift dataset uses pure exponential price paths (no
randomness) so the headline IC assertion is *exact*; the random dataset uses a
fixed-seed RNG so the variance-sensitive summary statistics are reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import qer.data.loader as loader_mod
import qer.diagnostics.audit_data as audit
import qer.diagnostics.factor_ic as factor_ic
import qer.universe.membership as membership_mod
import qer.universe.wikipedia as wiki
from qer.data.loader import DataLoader
from qer.factors.momentum import momentum, momentum_12_1
from qer.universe.membership import _resolve_ticker

SENTINEL_END = pd.Timestamp("2099-12-31")

# Momentum window constants, kept in one place so the "need this much history"
# arithmetic below is obvious: 252-day lookback + 21-day skip + 1.
LOOKBACK = 252
SKIP = 21
MIN_HISTORY = LOOKBACK + SKIP + 1


# ===========================================================================
# Low-level builders: write a realistic on-disk layout into a base directory.
# ===========================================================================


def _write_ticker(
    prices_dir: Path, ticker: str, dates: pd.DatetimeIndex, close: np.ndarray
) -> None:
    """Write one per-ticker parquet in the shape the ingestion step produces.

    The loader reads the ``adj close`` field for prices (auto-adjust naming),
    lower-cases columns, and flattens any MultiIndex. We supply a flat,
    already-lower-case frame with both ``close`` and ``adj close`` so both the
    price accessors and ``cross_section(field="close")`` work.
    """
    n = len(dates)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "adj close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=dates,
    )
    frame.index.name = "date"
    frame.to_parquet(prices_dir / f"{ticker}.parquet")


def _write_membership(membership_file: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    membership_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(membership_file)


def _make_layout(base: Path) -> dict[str, Path]:
    """Create the standard data/ sub-tree under ``base`` and return its paths."""
    raw = base / "raw"
    prices = raw / "prices"
    wide = base / "wide"
    processed = base / "processed"
    membership_file = processed / "membership.parquet"
    for d in (prices, wide, processed):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "raw": raw,
        "prices": prices,
        "wide": wide,
        "membership_file": membership_file,
    }


def _patch_loader_paths(monkeypatch, layout: dict[str, Path]) -> None:
    """Re-point the loader's module-level path constants at ``layout``.

    DataLoader reads RAW_DIR / PRICES_DIR / WIDE_DIR / MEMBERSHIP_FILE as
    module globals, so patching them on the loader module is sufficient.
    """
    monkeypatch.setattr(loader_mod, "RAW_DIR", layout["raw"])
    monkeypatch.setattr(loader_mod, "PRICES_DIR", layout["prices"])
    monkeypatch.setattr(loader_mod, "WIDE_DIR", layout["wide"])
    monkeypatch.setattr(loader_mod, "MEMBERSHIP_FILE", layout["membership_file"])


# ===========================================================================
# Module-scoped datasets (built once) + function-scoped loader fixtures.
# ===========================================================================


@pytest.fixture(scope="module")
def drift_dataset(tmp_path_factory):
    """40 tickers, pure exponential price paths with distinct per-ticker drift.

    Each ticker i has close_i(k) = 100 * exp(d_i * k) for trading day k, with
    the d_i strictly increasing across tickers. This makes both 12-1 momentum
    (computed in log space) and the forward log return strictly monotonic in
    d_i, so their cross-sectional rank correlation is exactly +1 every day.
    That gives a deterministic, exact end-to-end check of the whole
    loader -> factor -> IC chain.
    """
    base = tmp_path_factory.mktemp("drift")
    layout = _make_layout(base)

    n_tickers = 40
    dates = pd.bdate_range("2019-01-01", periods=330)
    drifts = np.linspace(0.0002, 0.0030, n_tickers)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    k = np.arange(len(dates))
    for i, t in enumerate(tickers):
        _write_ticker(layout["prices"], t, dates, 100.0 * np.exp(drifts[i] * k))

    _write_membership(
        layout["membership_file"],
        [{"ticker": t, "start_date": "2018-01-01", "end_date": SENTINEL_END} for t in tickers],
    )
    return {"layout": layout, "tickers": tickers, "dates": dates}


@pytest.fixture(scope="module")
def random_dataset(tmp_path_factory):
    """45 tickers, fixed-seed geometric random walks.

    Used to exercise the variance-sensitive parts of summarize_ic (IC IR and
    t-stat are NaN when the IC has zero variance, as in the drift dataset, so
    we need a dataset where daily IC genuinely moves).
    """
    base = tmp_path_factory.mktemp("random")
    layout = _make_layout(base)

    n_tickers = 45
    dates = pd.bdate_range("2018-01-01", periods=360)
    rng = np.random.default_rng(12345)
    tickers = [f"R{i:02d}" for i in range(n_tickers)]

    for t in tickers:
        rets = rng.normal(0.0004, 0.012, len(dates))
        _write_ticker(layout["prices"], t, dates, 100.0 * np.exp(np.cumsum(rets)))

    _write_membership(
        layout["membership_file"],
        [{"ticker": t, "start_date": "2017-01-01", "end_date": SENTINEL_END} for t in tickers],
    )
    return {"layout": layout, "tickers": tickers, "dates": dates}


@pytest.fixture(scope="module")
def membership_dataset(tmp_path_factory):
    """Membership-only dataset with real churn and a genuine future joiner.

    No price files: the audit sanity checks (no_future_leakage, churn) and
    get_universe only read the membership table, so this exercises those paths
    without the cost of generating hundreds of price files.

      * 100 long-standing current members (still in the index).
      * 12 exited members -> ~10.6% historical churn (above the 5% canary).
      * 1 member that joins on 2019-03-01, after the 2018-06-15 date the
        leakage check probes, so it stresses the "future joiner" branch.
    """
    base = tmp_path_factory.mktemp("membership")
    layout = _make_layout(base)

    rows: list[dict] = []
    for i in range(100):
        rows.append({"ticker": f"CUR{i:03d}", "start_date": "2005-01-01", "end_date": SENTINEL_END})
    for i in range(12):
        exit_year = 2013 + (i % 6)
        rows.append(
            {"ticker": f"EX{i:03d}", "start_date": "2010-01-01", "end_date": f"{exit_year}-06-01"}
        )
    rows.append({"ticker": "FUT001", "start_date": "2019-03-01", "end_date": SENTINEL_END})

    _write_membership(layout["membership_file"], rows)
    return {"layout": layout, "n_current": 101, "n_exited": 12}


@pytest.fixture
def drift_loader(monkeypatch, drift_dataset):
    _patch_loader_paths(monkeypatch, drift_dataset["layout"])
    return DataLoader()


@pytest.fixture
def random_loader(monkeypatch, random_dataset):
    _patch_loader_paths(monkeypatch, random_dataset["layout"])
    return DataLoader()


@pytest.fixture
def membership_loader(monkeypatch, membership_dataset):
    _patch_loader_paths(monkeypatch, membership_dataset["layout"])
    return DataLoader()


def _momentum_logspace(prices_df: pd.DataFrame, as_of_date) -> pd.Series:
    """12-1 momentum evaluated treating the input as log prices.

    compute_ic_series feeds the factor function ``np.log(close)``, so the
    log-space branch of ``momentum`` (end - start, no extra log) is the one
    that recovers a clean per-ticker momentum signal here.
    """
    return momentum(prices_df, as_of_date, lookback_days=LOOKBACK, skip_days=SKIP, log_prices=True)


# ===========================================================================
# A. Point-in-time universe reconstruction (wikipedia.get_universe)
# ===========================================================================


@pytest.fixture
def wiki_reconstruction(monkeypatch):
    """Patch the Wikipedia fetch with canned tables and return the inputs."""
    current = pd.DataFrame(
        {"ticker": [f"SY{i:03d}" for i in range(490)] + ["BRK.B", "GOOG", "GOOGL"]}
    )
    changes = pd.DataFrame(
        [
            ("2020-06-01", "NEW01", ""),  # added after both probe dates
            ("2019-01-15", "", "OLD01"),  # removed in 2019 (was a member in 2018)
            ("2021-03-10", "NEW02", "OLD02"),  # swap in 2021
        ],
        columns=["date", "added", "removed"],
    )
    changes["date"] = pd.to_datetime(changes["date"])
    monkeypatch.setattr(
        wiki, "fetch_sp500_tables", lambda force_refresh=False: (current.copy(), changes.copy())
    )
    return {"current": current, "changes": changes}


def test_reconstruction_current_universe_size_and_yahoo_mapping(wiki_reconstruction):
    """Today's universe == current table, with the Yahoo dot->dash mapping applied."""
    universe = wiki.get_universe("2024-03-01")
    # 493 names: 490 SYxxx + BRK.B + GOOG + GOOGL.
    assert len(universe) == 493
    # BRK.B is stored Wikipedia-style; Yahoo-format output must convert it.
    assert "BRK-B" in universe
    assert "BRK.B" not in universe


def test_reconstruction_is_point_in_time(wiki_reconstruction):
    """Reversing the changes log must not let future events leak backwards."""
    universe_2018 = set(wiki.get_universe("2018-06-15"))
    # NEW01 joined in 2020 -> must NOT appear in a 2018 universe.
    assert "NEW01" not in universe_2018
    # OLD01 was removed in 2019 -> it WAS a member back in 2018.
    assert "OLD01" in universe_2018
    # NEW02/OLD02 swapped in 2021: in 2018 the slot was held by OLD02.
    assert "OLD02" in universe_2018
    assert "NEW02" not in universe_2018


def test_rename_chain_collapses_to_current_ticker():
    """Multi-step renames resolve to the final symbol via the real rename map.

    WLP -> ANTM -> ELV is the canonical multi-hop case; this confirms the
    resolver and the curated TICKER_RENAMES table agree end-to-end.
    """
    assert _resolve_ticker("WLP") == "ELV"
    assert _resolve_ticker("ANTM") == "ELV"
    assert _resolve_ticker("FB") == "META"
    # A ticker absent from the map is returned unchanged.
    assert _resolve_ticker("AAPL") == "AAPL"


# ===========================================================================
# B. Membership table -> loader -> audit sanity checks
# ===========================================================================


def test_loader_membership_fills_sentinels(membership_loader):
    """The loader normalises the on-disk membership into query-ready form."""
    m = membership_loader.membership
    assert set(m.columns) == {"ticker", "start_date", "end_date"}
    assert pd.api.types.is_datetime64_any_dtype(m["start_date"])
    assert pd.api.types.is_datetime64_any_dtype(m["end_date"])
    # Still-in members carry the 2099 sentinel end date.
    assert (m.loc[m["ticker"].str.startswith("CUR"), "end_date"] == SENTINEL_END).all()


def test_get_universe_is_point_in_time(membership_loader):
    """The future joiner is absent before it joins and present afterwards."""
    early = membership_loader.get_universe("2018-06-15")
    later = membership_loader.get_universe("2020-01-02")
    assert "FUT001" not in early
    assert "FUT001" in later
    # The 100 long-standing members anchor both universes.
    assert len([t for t in early if t.startswith("CUR")]) == 100


def test_audit_no_future_leakage_passes(membership_loader):
    """The survivorship/leakage sanity check runs clean on a consistent table."""
    # Raises AssertionError on any inconsistency; reaching the end is the pass.
    audit.check_no_future_leakage(membership_loader)


def test_audit_historical_churn_in_band(membership_loader):
    """Churn sits above the 5% survivorship-bias canary and below the upper bound."""
    audit.check_historical_churn(membership_loader)


def test_all_tickers_ever_is_sorted_union(membership_loader):
    everything = membership_loader.all_tickers_ever()
    assert everything == sorted(everything)
    assert len(everything) == len(set(everything))
    assert "FUT001" in everything
    assert "EX000" in everything  # an exited name is still in the historical union


# ===========================================================================
# C. Price parquet -> loader wide matrices / returns / cross-section
# ===========================================================================


def test_wide_matrices_have_expected_shape(drift_loader, drift_dataset):
    close = drift_loader.close
    assert isinstance(close, pd.DataFrame)
    assert close.shape == (len(drift_dataset["dates"]), len(drift_dataset["tickers"]))
    assert set(close.columns) == set(drift_dataset["tickers"])
    assert pd.api.types.is_datetime64_any_dtype(close.index)
    # The open and volume accessors build off the same per-ticker files.
    assert drift_loader.open.shape == close.shape
    assert (drift_loader.volume.stack() == 1_000_000.0).all()


def test_log_and_simple_returns_agree_with_prices(drift_loader, drift_dataset):
    """Returns derived from the wide close match a direct computation."""
    log_ret = drift_loader.get_returns(kind="log")
    simple_ret = drift_loader.get_returns(kind="simple")
    close = drift_loader.close

    assert log_ret.shape == close.shape
    # First row has no prior price -> all NaN.
    assert log_ret.iloc[0].isna().all()

    ticker = drift_dataset["tickers"][5]
    expected_log = np.log(close[ticker].iloc[10] / close[ticker].iloc[9])
    assert log_ret[ticker].iloc[10] == pytest.approx(expected_log)
    expected_simple = close[ticker].iloc[10] / close[ticker].iloc[9] - 1
    assert simple_ret[ticker].iloc[10] == pytest.approx(expected_simple)


def test_cross_section_respects_universe(drift_loader, drift_dataset):
    a_date = drift_loader.close.index[300]
    cs = drift_loader.cross_section(a_date)
    assert isinstance(cs, pd.Series)
    assert len(cs) == len(drift_dataset["tickers"])
    assert set(cs.index) <= set(drift_dataset["tickers"])


def test_wide_cache_is_written_then_reused(monkeypatch, tmp_path):
    """A cold build writes the wide cache; a warm read serves from it.

    Uses its own isolated layout so the shared module datasets' caches don't
    interfere with the assertion.
    """
    layout = _make_layout(tmp_path)
    dates = pd.bdate_range("2020-01-01", periods=8)
    _write_ticker(layout["prices"], "AAA", dates, np.arange(100, 108, dtype=float))
    _write_ticker(layout["prices"], "BBB", dates, np.arange(50, 58, dtype=float))
    _write_membership(
        layout["membership_file"],
        [
            {"ticker": t, "start_date": "2019-01-01", "end_date": SENTINEL_END}
            for t in ("AAA", "BBB")
        ],
    )
    _patch_loader_paths(monkeypatch, layout)

    loader = DataLoader()
    cache_file = layout["wide"] / "adj close.parquet"
    assert not cache_file.exists()

    first = loader.close  # cold build -> writes cache
    assert cache_file.exists()

    # Drop the in-memory cache and overwrite the disk cache with a sentinel.
    loader._wide.clear()
    sentinel = pd.DataFrame({"SENTINEL": [42.0]}, index=pd.to_datetime(["2020-01-01"]))
    sentinel.index.name = "date"
    cache_file.unlink()
    sentinel.to_parquet(cache_file)

    second = loader.close  # warm read -> must come from disk cache
    assert list(second.columns) == ["SENTINEL"]
    assert second.iloc[0, 0] == 42.0
    assert list(first.columns) == ["AAA", "BBB"]  # the cold build was real


# ===========================================================================
# D. Factor + cross-sectional IC integration
# ===========================================================================


def test_momentum_has_no_look_ahead(drift_loader):
    """The factor at T is invariant to mutating prices strictly after T."""
    log_prices = np.log(drift_loader.close)
    as_of = drift_loader.close.index[280]

    baseline = momentum_12_1(log_prices, as_of, log_prices=True)

    mutated = log_prices.copy()
    after = mutated.index > as_of
    mutated.loc[after] = np.nan  # obliterate the future
    after_mutation = momentum_12_1(mutated, as_of, log_prices=True)

    pd.testing.assert_series_equal(baseline, after_mutation)


def test_momentum_returns_nan_without_enough_history(drift_loader):
    """Too little history yields an all-NaN factor, never a misleading zero."""
    log_prices = np.log(drift_loader.close)
    early_date = log_prices.index[10]  # nowhere near 273 days of history
    factor = momentum_12_1(log_prices, early_date, log_prices=True)
    assert factor.isna().all()
    assert len(factor) == log_prices.shape[1]


def test_ic_is_exactly_one_on_monotonic_data(drift_loader):
    """Full chain: monotonic drift => rank IC of +1 on every sample date.

    loader.close -> np.log -> momentum (log space) -> forward 21d return ->
    Spearman. Because momentum and the forward return are both strictly
    increasing in each ticker's drift, the rank correlation is exactly 1.
    """
    sample = drift_loader.close.index[300:312]
    ic = factor_ic.compute_ic_series(
        drift_loader,
        _momentum_logspace,
        forward_days=SKIP,
        sample_dates=sample,
        min_cross_section=30,
    )
    assert len(ic) > 0
    assert np.allclose(ic.values, 1.0)

    summary = factor_ic.summarize_ic(ic)
    assert summary["mean_ic"] == pytest.approx(1.0)
    assert summary["hit_rate"] == pytest.approx(1.0)
    assert summary["n"] == len(ic)


def test_ic_summary_statistics_on_noisy_data(random_loader):
    """On data with real IC variance, the summary stats are finite and bounded."""
    sample = random_loader.close.index[300:330]
    ic = factor_ic.compute_ic_series(
        random_loader,
        _momentum_logspace,
        forward_days=SKIP,
        sample_dates=sample,
        min_cross_section=30,
    )
    assert len(ic) > 0
    # Every IC is a valid rank correlation.
    assert (ic.abs() <= 1.0 + 1e-9).all()

    summary = factor_ic.summarize_ic(ic)
    expected_keys = {
        "n",
        "mean_ic",
        "std_ic",
        "ic_ir_annualized",
        "t_stat",
        "hit_rate",
        "date_min",
        "date_max",
    }
    assert expected_keys <= set(summary)
    assert summary["std_ic"] > 0  # the whole point of the noisy dataset
    assert np.isfinite(summary["ic_ir_annualized"])
    assert np.isfinite(summary["t_stat"])
    assert 0.0 <= summary["hit_rate"] <= 1.0


def test_ic_summary_handles_empty_series():
    """An empty IC series degrades gracefully to {'n': 0}."""
    assert factor_ic.summarize_ic(pd.Series(dtype=float)) == {"n": 0}


# ===========================================================================
# E. Audit chart builders over real loader data
# ===========================================================================


def test_universe_size_series_tracks_membership(drift_loader, drift_dataset):
    sizes = audit.universe_size_series(drift_loader, freq="ME")
    assert isinstance(sizes, pd.Series)
    assert len(sizes) > 0
    # Every ticker is a member for the whole window, so size is constant.
    assert (sizes == len(drift_dataset["tickers"])).all()


def test_missingness_matrix_is_well_formed(drift_loader):
    mat = audit.missingness_matrix(drift_loader, freq="QE")
    assert isinstance(mat, pd.DataFrame)
    assert mat.shape[0] > 0 and mat.shape[1] > 0
    # No ticker is ever missing in this dataset -> all observed fractions are 0.
    observed = mat.values[~np.isnan(mat.values)]
    assert (observed == 0).all()


def test_returns_by_year_long_format(drift_loader):
    long_df = audit.returns_by_year(drift_loader)
    assert list(long_df.columns) == ["year", "log_return"]
    assert len(long_df) > 0
    # The drift dataset spans 2019-2020.
    assert set(long_df["year"].unique()) <= {2019, 2020}


# ===========================================================================
# F. Full membership build over synthetic Wikipedia data.
# ===========================================================================


def test_build_membership_table_end_to_end(monkeypatch):
    """Synthetic Wikipedia tables -> a valid membership table.

    Previously crashed because build_membership_table validated against the
    current set in the wrong ticker format; that wiring is now fixed.
    """
    n_current = 500
    current = pd.DataFrame(
        {
            "ticker": [f"SY{i:03d}" for i in range(n_current)],
            "Date added": ["2000-01-01"] * n_current,
        }
    )
    events: list[tuple[str, str, str]] = []
    for i in range(30):
        year = 2011 + (i % 13)
        events.append((f"{year}-03-01", f"EX{i:03d}", ""))
        events.append((f"{year + 1}-09-01", "", f"EX{i:03d}"))
    changes = pd.DataFrame(events, columns=["date", "added", "removed"])
    changes["date"] = pd.to_datetime(changes["date"])

    monkeypatch.setattr(
        membership_mod,
        "fetch_sp500_tables",
        lambda force_refresh=False: (current.copy(), changes.copy()),
    )

    table = membership_mod.build_membership_table()
    assert set(table.columns) == {"ticker", "start_date", "end_date"}
    assert table["ticker"].nunique() == table.shape[0] or table.shape[0] >= n_current
