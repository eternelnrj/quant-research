"""
Phase 2 / Session 12: First factor IC plot.

For each trading date in the sample, compute the Spearman correlation
between the 12-1 momentum factor and the forward 21-day return across
the cross-section of active S&P 500 names. Then summarize and plot.

Reported metrics:
  - Mean IC: average of the daily cross-sectional rank correlations.
  - IC IR (information ratio): mean_IC / std_IC * sqrt(252), an
    annualized measure of how stable the signal is. Roadmap section
    7.2 says IR > 0.5 is meaningful, < 0.2 is noise.
  - t-stat of IC: mean / SE, useful for "is this signal statistically
    distinguishable from zero" rather than "is the signal economically
    interesting."

Conventions:
  - Factor is computed at end of trading day T using data through T.
  - Forward return is from close of T to close of T+21 trading days.
  - Per-date Spearman is computed only on tickers with BOTH a factor
    value AND a forward return. Tickers with NaN in either are dropped.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qer.config import AUDIT_DIR  # , DATA_DIR
from qer.data.loader import DataLoader
from qer.factors.momentum import momentum_12_1

# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _momentum_12_1_logspace(prices_df: pd.DataFrame, as_of_date) -> pd.Series:
    """12-1 momentum for a factor_func that receives LOG prices.

    ``compute_ic_series`` passes ``np.log(close)`` to the factor callable, so
    the factor must treat its input as already-log prices (``log_prices=True``)
    rather than raw levels. This wrapper makes that contract explicit.
    """
    return momentum_12_1(prices_df, as_of_date, log_prices=True)


def compute_ic_series(
    loader: DataLoader,
    factor_func,
    forward_days: int = 21,
    sample_dates: pd.DatetimeIndex | None = None,
    min_cross_section: int = 30,
) -> pd.Series:
    """Daily cross-sectional Spearman IC between factor and forward returns.

    Parameters
    ----------
    loader : DataLoader
        Provides .close (date x ticker price matrix) and .get_universe(date).
    factor_func : callable
        factor_func(prices_df, as_of_date) -> pd.Series indexed by ticker.
        For our case this is `momentum_12_1`.
    forward_days : int, default 21
        Trading days to look forward for the return computation. 21 = ~1 month.
    sample_dates : DatetimeIndex, optional
        Dates to compute IC on. Defaults to every trading day where both a
        factor and a forward return can be computed.
    min_cross_section : int, default 30
        Minimum number of (factor, return) pairs to compute Spearman.
        Below this, the date is skipped (correlation isn't stable).

    Returns
    -------
    pd.Series
        IC indexed by date. Days with insufficient cross-section are absent
        from the index entirely (NOT NaN), so summary stats are unaffected
        by silent skip dates.

    Notes
    -----
    No look-ahead: the factor at T uses only data through T, the forward
    return at T uses data from T to T+forward_days. The two are aligned
    on T - the alignment is the test the IC is measuring.
    """
    close = loader.close
    log_prices = np.log(close)

    if sample_dates is None:
        # We need: enough history before T to compute the factor, AND
        # enough forward data after T to compute the return.
        # momentum_12_1 needs 252+21=273 trading days of history.
        sample_dates = close.index[273:-forward_days]

    rows = []
    for t in sample_dates:
        # Factor at T uses only data through T.
        factor = factor_func(log_prices, t)  # factor_func(close, t)
        factor = factor.dropna()
        if len(factor) < min_cross_section:
            continue

        # Restrict to tickers in the index ON date t. Critical for the
        # IC to be meaningful - otherwise you're computing it over
        # tickers that weren't actually tradeable then.
        universe = set(loader.get_universe(t))
        factor = factor[factor.index.isin(universe)]
        if len(factor) < min_cross_section:
            continue

        # Forward return: log(P[T + forward_days] / P[T]).
        t_idx = close.index.get_loc(t)
        future_idx = t_idx + forward_days
        if future_idx >= len(close.index):
            continue
        t_future = close.index[future_idx]
        fwd = log_prices.loc[t_future] - log_prices.loc[t]

        # Align factor and forward return, drop pairs missing either side.
        common = factor.index.intersection(fwd.dropna().index)
        if len(common) < min_cross_section:
            continue

        ic, _ = spearmanr(factor.loc[common].values, fwd.loc[common].values)
        rows.append((t, ic))

    if not rows:
        return pd.Series(dtype=float, name="ic")
    series = pd.Series(dict(rows), name="ic")
    series.index.name = "date"
    return series


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def summarize_ic(ic: pd.Series, newey_west_lags: int | None = None) -> dict:
    """Compute mean IC, IC IR, t-stat, hit rate.

    Roadmap section 16.1 calibration:
        IC (daily):       0.01 - 0.04 for a known classical factor
        IC IR (ann.):     0.3 - 1.0
        t-stat of IC:     1.5 - 3.0 for a good single factor
    """
    n = len(ic)
    if n == 0:
        return {"n": 0}

    mean_ic = ic.mean()
    std_ic = ic.std(ddof=1)

    # Annualization: daily IC, ~252 obs per year. Same convention as Sharpe.
    ic_ir_annualized = mean_ic / std_ic * np.sqrt(252) if std_ic > 0 else np.nan

    # t-stat of mean IC under the null mean=0. Uses simple OLS-style SE;
    # for production use Newey-West to handle autocorrelation in IC.
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else np.nan

    # Hit rate: fraction of days where IC has the same sign as the mean.
    # For a positive-mean signal, this is fraction-positive.
    hit_rate = (np.sign(ic) == np.sign(mean_ic)).mean()

    result = {
        "n": n,
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ic_ir_annualized": ic_ir_annualized,
        "t_stat": t_stat,
        "hit_rate": hit_rate,
        "date_min": ic.index.min(),
        "date_max": ic.index.max(),
    }
    if newey_west_lags is not None:
        result["t_stat_nw"] = newey_west_tstat(ic.values, lags=newey_west_lags)
    return result


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_ic_time_series(
    ic: pd.Series,
    title: str = "12-1 Momentum: daily cross-sectional IC vs 21-day forward return",
    ax=None,
):
    """Plot IC time series with rolling mean and summary annotations.

    Reading: the raw IC bounces ±0.15 day-to-day. The 63-day rolling
    mean (~3 months) shows whether the signal is alive in any given
    regime. Persistent dips below zero indicate momentum-crash periods
    (e.g. early 2009, early 2010, early 2020).
    """
    summary = summarize_ic(ic)
    if summary["n"] == 0:
        raise ValueError("IC series is empty; nothing to plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    # Raw IC: scatter with light alpha so the cloud of daily values reads
    # as a density, not a wall of overlapping lines.
    ax.scatter(ic.index, ic.values, s=4, alpha=0.25, color="steelblue", label="Daily IC")

    # Rolling mean: the actual signal you can see despite the noise.
    rolling = ic.rolling(63, min_periods=21).mean()
    ax.plot(
        rolling.index,
        rolling.values,
        color="darkorange",
        linewidth=1.8,
        label="63-day rolling mean",
    )

    # Overall mean: horizontal reference.
    ax.axhline(
        summary["mean_ic"],
        color="black",
        linestyle="--",
        alpha=0.6,
        label=f"Mean IC = {summary['mean_ic']:.4f}",
    )
    ax.axhline(0, color="gray", linestyle="-", alpha=0.4)

    ax.set_title(title)
    ax.set_ylabel("Spearman IC")
    ax.set_xlabel("")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.3)

    # Annotation box with the headline metrics. Right-aligned, transparent
    # background, monospaced for clean number alignment.
    summary_text = (
        f"N days:      {summary['n']:>6d}\n"
        f"Mean IC:     {summary['mean_ic']:>+6.4f}\n"
        f"IC IR (ann): {summary['ic_ir_annualized']:>+6.3f}\n"
        f"t-stat:      {summary['t_stat']:>+6.2f}\n"
        f"Hit rate:    {summary['hit_rate']:>6.1%}"
    )
    ax.text(
        0.99,
        0.02,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        family="monospace",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
    )
    return ax


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_momentum_ic_analysis(
    loader: DataLoader,
    forward_days: int = 21,
    years: int = 5,
    output_dir: Path | None = None,
    show: bool = False,
) -> tuple[pd.Series, dict]:
    """Compute and plot the 12-1 momentum IC over the last `years` years.

    Saves the chart to output_dir/momentum_ic.png. Returns the IC series
    and the summary dict so callers can do further analysis.
    """
    if output_dir is None:
        output_dir = AUDIT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Restrict to "last N years" of the price data.
    close = loader.close
    start_date = close.index.max() - pd.DateOffset(years=years)
    sample_dates = close.index[
        (close.index >= start_date) & (close.index <= close.index[-forward_days - 1])
    ]
    # Also need to ensure we have factor-window history at the start.
    sample_dates = sample_dates[sample_dates >= close.index[273]]

    print(
        f"Computing IC over {len(sample_dates)} trading days "
        f"({sample_dates.min():%Y-%m-%d} to {sample_dates.max():%Y-%m-%d})..."
    )
    ic = compute_ic_series(
        loader,
        _momentum_12_1_logspace,
        forward_days=forward_days,
        sample_dates=sample_dates,
    )

    summary = summarize_ic(ic, newey_west_lags=forward_days - 1)
    print(f"\nMomentum 12-1 vs {forward_days}-day forward return")
    print(f"  N days computed:    {summary['n']}")
    print(f"  Mean IC:            {summary['mean_ic']:+.4f}")
    print(f"  Std IC:             {summary['std_ic']:.4f}")
    print(f"  IC IR (annualized): {summary['ic_ir_annualized']:+.3f}")
    print(f"  t-stat (naive):     {summary['t_stat']:+.2f}")
    # The daily ICs overlap (h-day forward return), so the naive t-stat is
    # inflated; the Newey-West (lag = h-1) value is the one to trust. IC IR is a
    # descriptive stability ratio, not a significance test, so it is left as the
    # standard mean/std*sqrt(252) and is NOT NW-adjusted.
    print(f"  t-stat (NW, lag {forward_days - 1}): {summary['t_stat_nw']:+.2f}   <- report this one")
    print(f"  Hit rate:           {summary['hit_rate']:.1%}")

    # Calibrate against roadmap expectations.
    if summary["mean_ic"] > 0.10:
        print("\nWARNING: mean IC > 0.10 - check for look-ahead.")
    elif summary["mean_ic"] < 0:
        print("\nNOTE: mean IC negative - momentum-reversal regime or bug.")

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_ic_time_series(
        ic,
        title=(
            f"12-1 Momentum: daily cross-sectional IC vs "
            f"{forward_days}-day forward return  "
            f"({sample_dates.min():%Y-%m} to {sample_dates.max():%Y-%m})"
        ),
        ax=ax,
    )
    fig.tight_layout()

    output_file = output_dir / "momentum_ic.png"
    fig.savefig(output_file, dpi=120)
    print(f"\nChart saved to {output_file}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ic, summary


# ---------------------------------------------------------------------------
# Phase 2: panel-based, multi-horizon IC + Newey-West significance
# ---------------------------------------------------------------------------

from qer.factors.base import Factor, compute_factor_panel  # noqa: E402


def newey_west_tstat(x, lags: int) -> float:
    """t-stat of the mean of ``x`` with a Newey-West (Bartlett) HAC variance.

    For daily ICs computed against an h-day forward return, consecutive ICs
    overlap, so the naive mean/std*sqrt(N) t-stat is inflated. Setting
    ``lags = h - 1`` corrects for that autocorrelation.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return np.nan
    e = x - x.mean()
    var = (e @ e) / n  # gamma_0
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1)
        var += 2.0 * w * (e[lag:] @ e[:-lag]) / n
    se = np.sqrt(var / n)
    return x.mean() / se if se > 0 else np.nan


def forward_return_panel(loader, horizon: int) -> pd.DataFrame:
    """``date x ticker`` forward log return over ``horizon`` trading days."""
    logc = np.log(loader.close)
    return logc.shift(-horizon) - logc


def _universe_mask(loader, index, columns) -> pd.DataFrame:
    """Boolean date x ticker mask of index membership on each date."""
    mask = pd.DataFrame(False, index=index, columns=columns)
    colset = set(columns)
    for t in index:
        members = [c for c in loader.get_universe(t) if c in colset]
        mask.loc[t, members] = True
    return mask


def compute_factor_ic(
    loader,
    factor: Factor,
    horizons=(1, 5, 21),
    dates=None,
    min_cross_section: int = 30,
    restrict_universe: bool = True,
) -> dict[int, pd.Series]:
    """Multi-horizon cross-sectional Spearman IC for a Factor, vectorised.

    Builds the (direction-oriented) factor panel once via
    :func:`compute_factor_panel`, then row-wise rank-correlates it with the
    forward-return panel at each horizon. Returns ``{horizon: IC series}``.
    """
    panel = compute_factor_panel(loader, factor, dates=dates, oriented=True)
    panel = panel.dropna(how="all")
    cols = panel.columns
    if restrict_universe:
        panel = panel.where(_universe_mask(loader, panel.index, cols))

    out: dict[int, pd.Series] = {}
    for h in horizons:
        fwd = forward_return_panel(loader, h).reindex(index=panel.index, columns=cols)
        ic = panel.corrwith(fwd, axis=1, method="spearman")
        n_pairs = (panel.notna() & fwd.notna()).sum(axis=1)
        ic = ic[n_pairs >= min_cross_section].dropna()
        ic.name = f"ic_{h}d"
        ic.index.name = "date"
        out[h] = ic
    return out
