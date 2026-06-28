from pathlib import Path

import numpy as np
import pandas as pd

from qer.config import (
    MEMBERSHIP_FILE,
    PRICES_DIR,
    RAW_DIR,
    SENTINEL_END,
    SHARES_FILE,
    SPY_FILE,
    WIDE_DIR,
)


class DataLoader:
    """Loads price, return, and universe data for the project.

    Storage layout (all paths resolved from ``qer.config``):
        data/processed/sp500_membership.parquet  - S&P 500 historical membership
        data/raw/prices/<TICKER>.parquet          - one file per ticker, full history
        data/wide/<field>.parquet                 - cached wide matrices, built lazily

    Price convention:
        Per-ticker files are pulled from yfinance with auto_adjust=False, so
        each file carries a separate ``Adj Close`` column (split/dividend
        adjusted) alongside the raw OHLC. ``DataLoader.close`` exposes that
        ``adj close`` field, so return computations are split/dividend safe.

        Because raw (unadjusted) OHLC is retained too, dollar volume can be
        computed from raw close * volume if needed. To switch to fully
        pre-adjusted OHLC with no separate adj-close column, re-pull
        ingestion with auto_adjust=True and point ``close`` at the ``close``
        field instead.
    """

    FIELDS = ("open", "high", "low", "close", "adj close", "volume")

    def __init__(self):  # , root="data/"):
        # self.root = Path(root)
        # self.raw_dir = self.root / "raw"
        # self.wide_dir = self.root / "wide"
        # self.wide_dir.mkdir(parents=True, exist_ok=True)

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        WIDE_DIR.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded caches
        self._membership = None
        self._wide = {}  # field -> DataFrame
        self._market_return = None
        self._shares = None

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------
    @property
    def membership(self) -> pd.DataFrame:
        """Historical S&P 500 membership table.

        Columns: ticker, start_date, end_date.
        NaT start_dates (additions predating the changes log) are filled
        with Timestamp.min so range queries don't need NaT handling.
        Currently-in-index tickers have end_date = 2099-12-31 sentinel.
        """
        if self._membership is None:
            self._membership = pd.read_parquet(MEMBERSHIP_FILE).assign(
                start_date=lambda d: d["start_date"].fillna(pd.Timestamp.min),
                end_date=lambda d: d["end_date"].fillna(pd.Timestamp(SENTINEL_END)),
            )

            # Ensure date types
            self._membership["start_date"] = pd.to_datetime(self._membership["start_date"])
            self._membership["end_date"] = pd.to_datetime(self._membership["end_date"])
        return self._membership

    def get_universe(self, date) -> list[str]:
        """Tickers in the S&P 500 on the given date.

        Note: date is when the universe is *queried*, not when fundamentals
        were available. For point-in-time fundamentals, apply a separate lag
        in Phase 1.4.

        Convention: membership is treated as half-open [start_date, end_date).
        A ticker removed on date D is NOT a member on date D. This matches
        the convention used by build_membership_table to avoid double-
        counting at same-day swap dates (e.g. FOXA on 2019-03-19).
        """
        date = pd.Timestamp(date)
        m = self.membership
        mask = (m["start_date"] <= date) & (m["end_date"] > date)
        return m.loc[mask, "ticker"].tolist()

    def all_tickers_ever(self) -> list[str]:
        """Union of all tickers that appear in membership at any point."""
        return sorted(self.membership["ticker"].unique().tolist())

    # ------------------------------------------------------------------
    # Wide-format field accessors (cached, built from per-ticker raw files)
    # ------------------------------------------------------------------
    def _load_wide(self, field: str) -> pd.DataFrame:
        """Return the date x ticker matrix for a given OHLCV field.

        Reads from cache if present; otherwise builds the matrix by reading
        every per-ticker parquet file and pivoting.
        """
        if field not in self.FIELDS:
            raise ValueError(f"Unknown field: {field}. Known: {self.FIELDS}")

        if field in self._wide:
            return self._wide[field]

        cache = WIDE_DIR / f"{field}.parquet"
        if cache.exists() and not self._wide_cache_stale(cache):
            wide = pd.read_parquet(cache)
            wide.index = pd.to_datetime(wide.index)
            self._wide[field] = wide
            return wide

        # Cold build (or stale cache): read every per-ticker raw file, take this
        # field, concat. Rebuilding on staleness keeps the per-field wide caches
        # in lock-step with the source files; otherwise re-ingesting more price
        # history could leave, say, the raw-close/volume caches behind the
        # adj-close cache, silently desyncing their date ranges. That surfaces
        # as a factor joining two fields (e.g. Amihud = |return|/dollar_volume)
        # going all-NaN on the dates present in only the newer cache.
        wide = self._build_wide(field)
        wide.to_parquet(cache)
        self._wide[field] = wide
        return wide

    @staticmethod
    def _wide_cache_stale(cache: Path) -> bool:
        """True if any per-ticker source file is newer than the wide cache.

        Wide caches are derived artifacts; if the raw per-ticker files have been
        re-ingested (extended date range, fixed adjustments) the cache must be
        rebuilt, or different fields can drift out of sync.
        """
        try:
            cache_mtime = cache.stat().st_mtime
        except OSError:
            return True
        for f in PRICES_DIR.glob("*.parquet"):
            if f.stat().st_mtime > cache_mtime:
                return True
        return False

    def _build_wide(self, field: str) -> pd.DataFrame:
        raw_files = sorted(PRICES_DIR.glob("*.parquet"))
        if not raw_files:
            raise FileNotFoundError(f"No per-ticker files in {PRICES_DIR}. Run ingestion first.")

        cols = {}
        for f in raw_files:
            ticker = f.stem
            df = pd.read_parquet(f)

            # Flatten MultiIndex columns: ('Close', 'AAPL') -> 'Close'
            # We take the FIRST level (the field name); the second level is
            # the ticker which we already know from the filename.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Normalize to lowercase
            df.columns = [str(c).lower() for c in df.columns]
            df.index = pd.to_datetime(df.index)

            if field not in df.columns:
                raise KeyError(
                    f"Field {field!r} not found in {f.name}. Available: {df.columns.tolist()}"
                )

            cols[ticker] = df[field]

        wide = pd.concat(cols, axis=1).sort_index()
        return wide

    @property
    def close(self) -> pd.DataFrame:
        """Adjusted close prices, date x ticker.

        The per-ticker files are ingested with yfinance's auto_adjust=False,
        which keeps a separate ``Adj Close`` column alongside the raw OHLC.
        This property exposes that ``adj close`` field, so split/dividend
        corrections are already baked in. Use this for return computation.
        """
        return self._load_wide("adj close")

    @property
    def open(self) -> pd.DataFrame:
        """Raw (unadjusted) open prices.

        Only ``adj close`` is split/dividend adjusted; the OHLC fields from
        auto_adjust=False are raw. Don't use this for returns - use ``close``.
        """
        return self._load_wide("open")

    @property
    def volume(self) -> pd.DataFrame:
        """Trading volume (shares)."""
        return self._load_wide("volume")

    # ------------------------------------------------------------------
    # Phase 2: derived price/volume views and market-cap inputs
    # ------------------------------------------------------------------
    @property
    def dollar_volume(self) -> pd.DataFrame:
        """Daily dollar volume = RAW close * volume, date x ticker.

        Dollar volume is the actual dollars transacted on a day: raw close price
        times shares traded. yfinance ``Volume`` is the as-reported (raw) share
        count, so it must be paired with the RAW close, not ``adj close`` -
        pairing it with the adjusted series scales each name by its own
        cumulative split/dividend factor, distorting the liquidity ranking both
        cross-sectionally and over time. Feeds the Amihud illiquidity factor.
        """
        return self._load_wide("close") * self.volume

    @property
    def market_return(self) -> pd.Series:
        """Daily log return of SPY, reindexed to the close calendar.

        Market-return proxy for the idiosyncratic-skewness beta regression.
        Reads ``SPY.parquet`` (produced by ``scripts/fetch_spy.py``).
        """
        if self._market_return is None:
            if not SPY_FILE.exists():
                raise FileNotFoundError(
                    f"{SPY_FILE} not found. Run `python -m scripts.fetch_spy` "
                    "(or `make spy`) to enable market-relative factors."
                )
            spy = pd.read_parquet(SPY_FILE)
            spy.columns = [str(c).lower() for c in spy.columns]
            if "adj close" not in spy.columns:
                raise KeyError(
                    f"{SPY_FILE} has no 'adj close' column. market_return needs "
                    "the dividend/split-adjusted series; silently using raw "
                    "'close' would drop dividends (~1.5-2%/yr for SPY) and bias "
                    "the market proxy. Re-fetch with `make spy` (auto_adjust=False)."
                )
            spy.index = pd.to_datetime(spy.index)
            px = spy["adj close"].sort_index()
            ret = np.log(px / px.shift(1))
            self._market_return = ret.reindex(self.close.index)
        return self._market_return

    def _load_shares(self) -> pd.DataFrame:
        """Shares-outstanding panel, date x ticker, forward-filled to daily."""
        if self._shares is None:
            if not SHARES_FILE.exists():
                raise FileNotFoundError(
                    f"{SHARES_FILE} not found. Shares outstanding has not been "
                    "ingested, so market cap / size are unavailable."
                )
            shares = pd.read_parquet(SHARES_FILE)
            shares.index = pd.to_datetime(shares.index)
            self._shares = shares.sort_index()
        return self._shares

    @property
    def market_cap(self) -> pd.DataFrame:
        """Market cap = RAW close * shares outstanding, date x ticker.

        Market cap is the transaction price times shares, so it uses the RAW
        close, not ``adj close`` (the adjusted series embeds future split and
        dividend factors). This assumes ``shares.parquet`` holds the actual
        historical share count; splits then cancel correctly (price halves,
        shares double). Shares are forward-filled onto the daily price calendar.
        Needs ``shares.parquet``; raises if absent (size is then skipped).
        """
        shares = self._load_shares()
        close = self._load_wide("close")
        shares_daily = shares.reindex(index=close.index).ffill()
        common = close.columns.intersection(shares_daily.columns)
        return close[common] * shares_daily[common]

    # ------------------------------------------------------------------
    # Derived: returns
    # ------------------------------------------------------------------
    def get_returns(self, kind: str = "log") -> pd.DataFrame:
        """Daily returns, date x ticker.

        kind='log' returns log returns; kind='simple' returns simple returns.
        Computed from the adjusted close (the ``adj close`` field kept by
        auto_adjust=False), so already corrected for splits and dividends.
        """
        close = self.close
        if kind == "log":
            return np.log(close / close.shift(1))
        elif kind == "simple":
            return close.pct_change()
        else:
            raise ValueError(f"kind must be 'log' or 'simple', got {kind!r}")

    # ------------------------------------------------------------------
    # Convenience: universe-filtered cross-section on a date
    # ------------------------------------------------------------------
    def cross_section(self, date, field: str = "close") -> pd.Series:
        """Field values across the current S&P 500 universe on a date.

        Returns a Series indexed by ticker. Tickers in the index that day
        but with NaN price data are dropped (delistings, missing days).
        """
        date = pd.Timestamp(date)
        universe = self.get_universe(date)
        wide = self._load_wide(field)

        if date not in wide.index:
            raise KeyError(f"Date {date} not in price data.")
        # Only keep universe tickers that actually exist in the wide matrix.
        # A ticker in the membership table but absent from the raw files
        # (e.g. ingestion failed for it) should be skipped, not crash.
        available = [t for t in universe if t in wide.columns]
        row = wide.loc[date, available].dropna()
        return row

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def rebuild_wide_cache(self):
        """Force rebuild of all cached wide matrices.

        Run this after re-ingesting raw data.
        """
        for cache in WIDE_DIR.glob("*.parquet"):
            cache.unlink()
        self._wide.clear()
        for field in self.FIELDS:
            self._load_wide(field)
