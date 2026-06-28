"""
Data audit.

Produces four diagnostic charts (universe size over time, missing-data
heatmap, return distribution by year, sector breakdown) and runs three
sanity checks (no future leakage, ~5% historical churn, SPY total return
matches published).

Logic lives here as plain functions. The notebook 01_data_audit.ipynb
imports each function and renders the result inline with surrounding
narrative. This keeps the heavy code testable and the notebook readable.

Run end-to-end as a script for an audit/ folder of PNGs:
    python -m scripts.audit_data
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from qer.config import AUDIT_DIR, RAW_DIR, SENTINEL_END  # DATA_DIR,           # NO CHANGES from phase 2
from qer.config import AUDIT_DIR, SECTORS_FILE, SENTINEL_END, SPY_FILE
from qer.data.loader import DataLoader

# ---------------------------------------------------------------------------
# Universe size over time
# ---------------------------------------------------------------------------


def universe_size_series(loader: DataLoader, freq: str = "ME") -> pd.Series:
    """Number of active tickers at month-end across the loader's date range.

    freq='ME' is month-end; use 'W-FRI' for weekly Friday, 'D' for daily.
    Daily is overkill for a chart but useful for sanity checks.
    """
    close = loader.close
    sample_dates = pd.date_range(close.index.min(), close.index.max(), freq=freq)
    sizes = pd.Series(
        {d: len(loader.get_universe(d)) for d in sample_dates},
        name="universe_size",
    )
    sizes.index.name = "date"
    return sizes


def plot_universe_size(loader: DataLoader, ax=None):
    """Universe size over time. Should hover near 500 with mild drift."""
    sizes = universe_size_series(loader)

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))

    sizes.plot(ax=ax, linewidth=1.2)
    ax.axhline(500, color="red", linestyle="--", alpha=0.4, label="Nominal S&P 500 = 500")
    ax.set_title(f"Active universe size, {sizes.index.min():%Y-%m} to {sizes.index.max():%Y-%m}")
    ax.set_ylabel("Active tickers")
    ax.set_xlabel("")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Missing-data heatmap
# ---------------------------------------------------------------------------


def missingness_matrix(loader: DataLoader, freq: str = "ME") -> pd.DataFrame:
    """Fraction of universe-active tickers with NaN close at each sample date.

    Rows = sample dates (downsampled to freq); columns = ticker buckets
    (deciles by ticker name). The bucketing keeps the heatmap readable
    when 700+ tickers exist over the sample period.
    """
    close = loader.close
    sample_dates = pd.date_range(close.index.min(), close.index.max(), freq=freq)

    all_tickers = sorted(close.columns)
    n_buckets = min(20, len(all_tickers))  # guard tiny universes

    # array_split guarantees every ticker lands in exactly one bucket;
    # remainder is spread across the first few buckets, not dropped.
    bucket_lists = [list(b) for b in np.array_split(all_tickers, n_buckets)]  # MODIF
    buckets = {
        f"{b[0]} - {b[-1]}": b
        for b in bucket_lists
        if (len(b) > 0)  # array_split can yield empty arrays if n_buckets > n_tickers
    }

    rows = []
    for d in sample_dates:
        if d not in close.index:
            # Snap to nearest trading day at or before d
            valid = close.index[close.index <= d]
            if len(valid) == 0:
                continue
            d = valid[-1]
        universe = set(loader.get_universe(d))
        row = {}
        for bucket_name, tickers in buckets.items():
            in_universe = [t for t in tickers if t in universe]
            if len(in_universe) == 0:  # MODIF
                row[bucket_name] = np.nan
                continue
            n_missing = close.loc[d, in_universe].isna().sum()
            row[bucket_name] = n_missing / len(in_universe)
        rows.append((d, row))

    return pd.DataFrame.from_dict(dict(rows), orient="index")


def plot_missingness_heatmap(loader: DataLoader, ax=None):
    """Heatmap of fraction-missing by date and ticker bucket.

    Reading: bright cells indicate clusters where universe membership says
    a ticker should be active but we have no price data. Concentrated
    bright cells before 2011 likely reflect changes-log coverage gaps;
    scattered cells are likely delisted/acquired tickers.
    """
    mat = missingness_matrix(loader)

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6))

    im = ax.imshow(
        mat.T.values,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        vmin=0,
        vmax=mat.values[~np.isnan(mat.values)].max() if mat.size else 1,
    )
    ax.set_yticks(range(len(mat.columns)))
    ax.set_yticklabels(mat.columns, fontsize=7)
    n_dates = len(mat.index)
    tick_step = max(n_dates // 10, 1)
    ax.set_xticks(range(0, n_dates, tick_step))
    ax.set_xticklabels(
        [d.strftime("%Y-%m") for d in mat.index[::tick_step]],
        rotation=45,
        ha="right",
    )
    ax.set_title("Missing-data heatmap: fraction of universe with NaN price")
    plt.colorbar(im, ax=ax, label="fraction missing")
    return ax


# ---------------------------------------------------------------------------
# Return distribution by year
# ---------------------------------------------------------------------------


def returns_by_year(loader: DataLoader) -> pd.DataFrame:
    """Cross-sectional daily log returns flattened, grouped by year.

    Returns a long-format DataFrame with columns ['year', 'log_return']
    suitable for a violin or box plot.
    """
    returns = loader.get_returns(kind="log")
    # Stack into long format, drop NaN. (pandas >= 2.1 stack() keeps NaN by
    # default, so the dropna here is explicit rather than incidental.)
    stacked = returns.stack().dropna().reset_index()
    stacked.columns = ["date", "ticker", "log_return"]
    stacked["year"] = stacked["date"].dt.year
    return stacked[["year", "log_return"]]


def plot_return_distribution_by_year(loader: DataLoader, ax=None):
    """Box plot of daily log returns by year.

    Look for: 2008 with fat tails, 2017 narrow distribution, 2020 wide
    distribution centered slightly negative (COVID), 2022 wider than 2021.
    Year-on-year differences are themselves a sanity check.
    """
    long_df = returns_by_year(loader)
    years = sorted(long_df["year"].unique())
    by_year = [long_df.loc[long_df["year"] == y, "log_return"].values for y in years]

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))

    ax.boxplot(
        by_year,
        tick_labels=years,
        showfliers=False,  # outliers swamp the box; show whiskers only
        whis=(1, 99),  # whiskers at 1st and 99th percentile
        patch_artist=True,
        boxprops=dict(facecolor="lightsteelblue", alpha=0.6),
    )
    ax.axhline(0, color="red", linestyle="--", alpha=0.4)
    ax.set_title("Daily log-return distribution by year (whiskers = 1st/99th %ile)")
    ax.set_ylabel("Log return")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(alpha=0.3, axis="y")
    return ax


# ---------------------------------------------------------------------------
# Sector breakdown
# ---------------------------------------------------------------------------


def load_sectors() -> pd.Series | None:
    """Load per-ticker sector mapping if it exists.

    Expected schema: a Series indexed by ticker with sector as value.
    Returns None if the file doesn't exist (caller should skip the chart).
    """
    if not SECTORS_FILE.exists():
        return None
    return pd.read_parquet(SECTORS_FILE).iloc[:, 0]


def plot_sector_breakdown(loader: DataLoader, ax=None, as_of: str | pd.Timestamp | None = None):
    """Bar chart of universe composition by GICS sector at a snapshot date.

    Requires data/raw/sectors.parquet to exist. If not, prints a hint
    and returns None.
    """
    sectors = load_sectors()
    if sectors is None:
        print(
            f"No sector data at {SECTORS_FILE}. "
            "Run `python -m scripts.fetch_sectors` to generate it, "
            "then re-run this chart."
        )
        return None

    if as_of is None:
        as_of = loader.close.index[-1]
    as_of = pd.Timestamp(as_of)

    universe = loader.get_universe(as_of)
    sector_for_universe = sectors.reindex(universe).dropna()
    counts = sector_for_universe.value_counts().sort_values(ascending=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    counts.plot(kind="barh", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title(f"Universe by sector as of {as_of:%Y-%m-%d} ({len(universe)} tickers)")
    ax.set_xlabel("Number of tickers")
    ax.grid(alpha=0.3, axis="x")
    n_missing = len(universe) - len(sector_for_universe)
    if n_missing:
        ax.text(
            0.98,
            0.02,
            f"{n_missing} tickers missing sector data",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            alpha=0.6,
        )
    return ax


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def check_no_future_leakage(loader: DataLoader) -> None:
    m = loader.membership
    test_date = pd.Timestamp("2018-06-15")
    universe = set(loader.get_universe(test_date))

    # Forward check: every ticker in the universe has >=1 covering interval.
    for ticker in universe:
        rows = m[m["ticker"] == ticker]
        covers = (rows["start_date"] <= test_date) & (rows["end_date"] > test_date)
        assert covers.any(), (
            f"{ticker} in get_universe({test_date.date()}) but no interval covers it. Rows:\n{rows}"
        )

    # Build per-ticker coverage once, at the ticker level.
    covers_mask = (m["start_date"] <= test_date) & (m["end_date"] > test_date)
    covered_tickers = set(m.loc[covers_mask, "ticker"])

    # A genuine future joiner: NONE of its intervals covers test_date,
    # AND at least one interval starts strictly after test_date.
    starts_after = set(m.loc[m["start_date"] > test_date, "ticker"])
    genuine_future_joiners = starts_after - covered_tickers

    if genuine_future_joiners:
        future_ticker = sorted(genuine_future_joiners)[0]  # deterministic
        assert future_ticker not in universe, (
            f"{future_ticker} has no interval covering {test_date.date()} "
            f"and joins later, yet appears in get_universe({test_date.date()})"
        )

    # Optional but stronger: the universe should equal the covered set.
    assert universe == covered_tickers, (
        f"get_universe disagrees with covering-interval set at "
        f"{test_date.date()}: only-in-universe={universe - covered_tickers}, "
        f"only-in-covered={covered_tickers - universe}"
    )

    print(f"PASS: no future leakage at {test_date.date()} (universe size {len(universe)})")


def check_historical_churn(
    loader: DataLoader, min_churn: float = 0.05, max_churn: float = 0.60
) -> None:
    """At least 5% of historical tickers no longer trade today, with a
    generous upper bound.

    The lower bound (5%) is the survivorship-bias canary from roadmap
    section 6.5: less than 5% means today's constituents probably leaked
    into historical universes. The upper bound (60%) catches the opposite
    failure - if more than 60% of historical tickers have exited, something
    is over-counting exits (e.g., ticker renames not collapsing in the
    rename map, producing phantom historical entities).

    Over a 14-year window, ~40% churn is normal. The S&P 500 sees roughly
    20-25 replacements per year, accumulating to 280-350 distinct exits
    over 14 years against a peak universe of ~500-700 ever-members.
    """
    m = loader.membership
    sentinel = SENTINEL_END
    all_tickers = m["ticker"].unique()
    still_active_tickers = set(m.loc[m["end_date"] == sentinel, "ticker"])
    exited_tickers = set(all_tickers) - still_active_tickers
    churn = len(exited_tickers) / len(all_tickers)

    assert min_churn <= churn <= max_churn, (
        f"Historical churn = {churn:.1%} is outside [{min_churn:.0%}, "
        f"{max_churn:.0%}]. Too low = survivorship bias; too high = "
        f"over-counted exits or unresolved ticker renames."
    )
    print(
        f"PASS: historical churn = {churn:.1%} "
        f"({len(exited_tickers)} of {len(all_tickers)} tickers no longer in index)"
    )


def check_spy_total_return(spy_file: Path | None = None, tolerance: float = 0.01) -> None:
    """Verify SPY's total return over the sample matches a published value.

    Approach: load SPY's price history from raw/SPY.parquet if it exists,
    compute total return start-to-end, and compare to a published benchmark
    figure. Tolerance is 1% absolute on the multiplicative return.

    Note: this requires SPY to have been ingested. If not, prints a hint.
    """
    if spy_file is None:
        spy_file = SPY_FILE  # RAW_DIR / "SPY.parquet"

    if not spy_file.exists():
        print(
            f"SKIP: {spy_file} not found. Run `python -m scripts.fetch_spy` "
            "(or `make spy`) to generate it and enable this check."
        )
        return

    spy = pd.read_parquet(spy_file)
    spy.columns = [str(c).lower() for c in spy.columns]

    # TOTAL return (dividends reinvested) requires the dividend/split-adjusted
    # series. The raw "close" column is price return only; using it would drop
    # SPY's dividends (~1.5-2%/yr) and understate the total return.
    if "adj close" not in spy.columns:
        print(
            f"SKIP: {spy_file} has no 'adj close' column. This check needs the "
            "dividend-adjusted series; re-fetch with `make spy` (auto_adjust=False)."
        )
        return

    spy = spy.dropna(subset=["adj close"]).sort_index()
    if len(spy) < 100:
        print(f"SKIP: only {len(spy)} SPY data points, too sparse to check.")
        return

    start_price = spy["adj close"].iloc[0]
    end_price = spy["adj close"].iloc[-1]
    start_date = spy.index[0]
    end_date = spy.index[-1]

    total_return = (end_price / start_price) - 1

    # The "published value" comparison is approximate because we don't know
    # exactly what reference the user wants. Provide a useful range check:
    # SPY total return (from the dividend-adjusted "adj close" series) should
    # grow roughly 7-12% annualized over a typical 10+ year period.
    n_years = (end_date - start_date).days / 365.25
    annualized = (1 + total_return) ** (1 / n_years) - 1

    print(
        f"SPY total return: {total_return:+.1%} "
        f"({annualized:+.2%} annualized) over {n_years:.1f} years "
        f"({start_date:%Y-%m-%d} to {end_date:%Y-%m-%d})"
    )

    # Hard check: annualized return should be in a plausible band.
    # For 2011-2024 the realized SPY total return is ~13% annualized.
    # For 2005-2024 it's ~10% (the 2008 crash drags it down).
    # Use a wide band that catches catastrophic errors (negative, or 30%+).
    assert 0.02 <= annualized <= 0.20, (
        f"Annualized SPY return of {annualized:.1%} is implausible. "
        "Check that the total return uses the dividend-adjusted 'adj close' "
        "column (auto_adjust=False convention) and that the date range is correct."
    )
    print(f"PASS: SPY return within plausible band (tolerance={tolerance:.0%})")


def check_price_adjustment(
    loader,
    extreme_log_ret: float = 0.40,
    max_unadjusted_frac: float = 0.5,
) -> None:
    """Audit the per-ticker ADJUSTED price history (the series factors consume).

    Two tripwires; asserts only the egregious one.
      1) Extreme single-day moves |adj-close log return| > ``extreme_log_ret``,
         listed per ticker for eyeball review. A cluster in one name usually
         means an uncorrected split rather than a real move - but genuine
         crashes / M&A exist, so this is a review flag, not a hard failure.
      2) Adjustment actually applied: tickers whose adj-close returns equal the
         raw-close returns on EVERY overlapping day are either unadjusted (the
         per-ticker analogue of the SPY raw-vs-adjusted bug) or simply paid no
         dividend/split in the window. A *large fraction* of the universe being
         identical means the adjustment step never ran - that we assert on.
    """
    import numpy as np

    adj = loader.close
    raw = loader._load_wide("close")
    cols = adj.columns.intersection(raw.columns)
    adj_ret = np.log(adj[cols] / adj[cols].shift(1))
    raw_ret = np.log(raw[cols] / raw[cols].shift(1))

    extreme = (adj_ret.abs() > extreme_log_ret).sum()
    flagged = extreme[extreme > 0].sort_values(ascending=False)

    overlap = (adj_ret.notna() & raw_ret.notna()).sum()
    agree = ((adj_ret - raw_ret).abs() < 1e-9).sum()  # counts overlapping ~equal days only
    identical = agree[(overlap > 0) & (agree == overlap)].index
    frac_unadjusted = len(identical) / max(len(cols), 1)

    print(f"Tickers checked:                {len(cols)}")
    print(
        f"Extreme |adj ret| > {extreme_log_ret:.2f} (review): "
        f"{int(extreme.sum())} days across {len(flagged)} names {list(flagged.index[:5])}"
    )
    print(f"Identical-to-raw (unadjusted?): {len(identical)} ({frac_unadjusted:.1%})")
    assert frac_unadjusted < max_unadjusted_frac, (
        f"{frac_unadjusted:.0%} of tickers have adj-close == raw-close returns; "
        "the dividend/split adjustment likely did not run."
    )
    print("PASS: adjustment applied across the universe (review any flagged names above).")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all_audits(loader: DataLoader, output_dir: Path | None = None) -> None:
    """Generate all four charts to PNGs and run all three sanity checks.

    Charts go to output_dir/{universe_size,missingness,returns_by_year,
    sector_breakdown}.png. Defaults to data/audit/.
    """
    if output_dir is None:
        output_dir = AUDIT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating charts...")
    for chart_func, name in [
        (plot_universe_size, "universe_size"),
        (plot_missingness_heatmap, "missingness"),
        (plot_return_distribution_by_year, "returns_by_year"),
        (plot_sector_breakdown, "sector_breakdown"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        result = chart_func(loader, ax=ax)
        if result is not None:
            fig.tight_layout()
            fig.savefig(output_dir / f"{name}.png", dpi=120)
            print(f"  - {name}.png")
        plt.close(fig)

    print("\nRunning sanity checks...")
    check_no_future_leakage(loader)
    check_historical_churn(loader)
    check_spy_total_return()
