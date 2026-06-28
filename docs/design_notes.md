# Design Notes

Rationale behind the structural choices in this repository, written from what
the code actually does today rather than what the roadmap aspires to. The aim is
to be honest about the seams, so the next person (or the next phase) knows where
the bodies are buried.

## Module layout

The package is organised by concern, behind a thin `src/qer/` namespace:

- `config.py` — the single source of truth for every path and a couple of date
  constants. Nothing else hard-codes a directory. This matters more than it
  looks: the loaders bind these names at import time, which is exactly what lets
  the tests redirect storage to a temp directory by patching the constants on
  the importing module.
- `data/` — observation ingestion and serving: `loader.py` (the `DataLoader`
  for prices, returns, dollar volume, market return, market cap), `fundamentals.py`
  (the point-in-time `FundamentalsLoader`), and `edgar.py` (the shared SEC EDGAR
  client).
- `universe/` — membership reconstruction (`wikipedia.py`), interval building
  and entity resolution (`membership.py`), and the curated correction tables
  (`renames.py`).
- `factors/` — the `Factor` ABC + registry (`base.py`) and eight factors.
- `diagnostics/` — evaluation (`factor_ic.py`), long-short portfolios
  (`portfolios.py`), FF5 exposures (`exposures.py`), multiple testing
  (`multiple_testing.py`), the deflated Sharpe (`deflated_sharpe.py`), the
  factor-zoo builder (`factor_zoo.py`), and the data audit (`audit_data.py`).

The deliberate split is *logic in the package, narrative in the notebooks, thin
wrappers in scripts*. Diagnostic functions return data and figures; the numbered
notebooks import them and add prose; the `scripts/` files are CLI entry points
that build a loader, call a package function, and write/print. This is why the
factor-zoo logic lives in `diagnostics/factor_zoo.py` and `scripts/run_factor_zoo.py`
is a one-call wrapper — so a notebook can `from qer.diagnostics.factor_zoo import
build_factor_zoo_table` rather than reaching into `scripts/` (which isn't on the
import path from a notebook's working directory).

## data vs universe: two dimensions, one-way dependency

`universe/` decides *which* securities are in scope and *when*; `data/` provides
the *values* once they're in scope. The two packages never import each other —
they are coupled only through the membership parquet that `universe` writes and
the loader reads. That asymmetry is deliberate: `universe` is the lower layer
that `data` sits on top of, and isolating "who was in the index when" into one
folder concentrates all survivorship-correctness logic in one auditable place.

## DataLoader: lazy, cached, single-interface

The loader presents three layers: per-ticker raw parquet (the ingestion landing
zone), wide `date x ticker` matrices per field (built on first access by
pivoting every raw file, then cached to `data/wide/`), and derived views
(returns, universe-filtered cross-sections, market cap). The cold build is
expensive but happens once, the cache is re-derivable from raw (safe to
`clean-cache`), and laziness means a process that only needs the universe never
pays for the price pivot.

The adjust convention is now explicit and consistent: `close` (and therefore
returns) reads the `adj close` field, so splits and dividends never masquerade
as returns; but `dollar_volume` and `market_cap` multiply by the **raw** `close`,
because adjusted prices are rebased and would not match the share/volume counts
they're paired with. SPY's `market_return` uses adjusted close (total return).

## Factors: vectorised panels, no look-ahead by construction

Each factor subclasses `Factor` and implements `compute_panel(loader)`, returning
a `date x ticker` panel built in one vectorised pass with rolling/shift
operations. Look-ahead is structurally impossible: a value at date *t* depends
only on a trailing window ending at or before *t*, and insufficient history
returns NaN, never zero, so downstream code must handle "no signal" explicitly
rather than trade on a misleading flat value. The registry (`register` /
`all_factors` / `get_factor`) lets the zoo iterate every factor uniformly. Raw
panels are direction-agnostic; the harness multiplies by each factor's
`direction` so orientation lives in one place.

## Evaluation harness

`compute_factor_ic` builds a factor's panel once via `compute_factor_panel`, then
rank-correlates it against forward returns at multiple horizons (1/5/21-day),
restricting each date's cross-section to the names actually in the index that
day and dropping pairs missing either side. `summarize_ic` reports mean IC, IC
IR, hit rate, a naive t-stat, and — because overlapping forward-return windows
make the naive t-stat overstate significance — an optional Newey-West (Bartlett
HAC) t-stat. `portfolios.factor_long_short` forms decile spreads; `exposures`
regresses long-short returns on FF5 (numpy OLS point estimates, since
`statsmodels` is declared but not importable in this environment) and reads
every coefficient's t-stat - alpha and each beta - off one Newey-West HAC
sandwich `(X'X)^-1 S (X'X)^-1`, so the overlap-awareness is consistent across
the whole regression rather than HAC on alpha and plain OLS on the betas;
`multiple_testing` applies
Benjamini-Hochberg and Bonferroni across the zoo; its `pvalue_from_tstat` takes
an *explicit* reference distribution rather than guessing one (`df=None` -> the
standard normal, which is the asymptotic reference for the Newey-West `t_nw` the
zoo feeds it; a caller-supplied `df` -> Student-t for a single-sample mean
(`n-1`) or a regression coefficient (`n-k`)). And `deflated_sharpe` discounts the
best Sharpe for the number of
trials, fed an *overlap-aware* effective sample size (`n / primary_horizon`,
since the long-short returns are horizon-day forward returns sampled daily) so
it doesn't treat ~1000 correlated observations as independent. The deflated
Sharpe's selection benchmark is scaled by the empirical cross-trial variance of
the zoo's Sharpes. `factor_zoo.build_factor_zoo_table` ties these together into
one table.

`net_long_short` charges a linear turnover cost on *both* legs
(`long_short_turnover` = long-top churn + short-bottom churn); charging the long
leg alone would undercount trading cost by roughly half for a symmetric factor.
The model is deliberately trading-cost-only — short-borrow and market-impact
costs are deferred to the Phase 4 backtester.

## SEC EDGAR ingest: point-in-time by filing date

`data/edgar.py` is a small shared client (used by both the fundamentals and
shares ingests): SEC-required User-Agent handling, ticker→CIK resolution,
cached `companyfacts` fetch, and point-in-time parsing helpers. The key design
choices, all in service of not lying to the backtest:

- **`available_date` is the filing's actual `filed` date**, not an `end + lag`
  guess. The lag constant survives only as a fallback for the rare entry with no
  `filed` field.
- **`dedup_point_in_time` keeps the earliest disclosure of each distinct
  (period-end, value)** — dropping comparatives re-reported in later filings,
  but keeping genuine restatements as new dated observations.
- **Gross profit, when not tagged directly, is derived by pairing revenue and
  cost within the *same filing* (by accession number).** Keying by period-end
  alone would let a restated revenue be matched with an original cost, inventing
  a gross-profit figure that appeared in no single filing.
- **Shares outstanding collapse to one value per `available_date` by keeping the
  latest period-end** on tied filing dates, so a same-day pair resolves to the
  most current figure rather than an arbitrary input-order value.

The Fama-French 5 ingest (`scripts/fetch_ff5.py`) parses Ken French's daily file,
renames columns to `mkt_rf, smb, hml, rmw, cma, rf`, and divides by 100 so the
factors are decimal returns on the same scale as the long-short series.

## Universe reconstruction: scrape forward, replay backward

Rather than scrape old revisions of the Wikipedia page (whose formatting drifts
and silently drops class-share tickers), the design scrapes *one* current
snapshot plus the changes log, then reconstructs any past date by reversing every
change after it. `wikipedia.get_universe(date)` is the point-in-time-correct
replay; `membership.build_membership_table()` walks the same events into explicit
`(ticker, start, end)` intervals with entity resolution — renames collapsed onto
the current symbol (cycle-safe multi-step chains), acquisitions/delistings as
dated exits, same-day swaps ordered removal-before-addition, and dangling
intervals closed conservatively with a warning. The correction tables are
explicitly manual and will rot; the warnings are the maintenance signal.

## Sanity checks as first-class code

The audit module encodes checks that catch real bugs in this class of project:
no future leakage, historical churn inside a two-sided band (survivorship canary
below, over-counted-exits canary above), missingness, and a plausibility band on
SPY's annualised total return. They fail loudly with diagnostic messages, because
a quiet data bug is far more expensive than a noisy assertion.

## Reproducibility and testing

`make all` (`install → data → audit → factors`) is the reviewer-facing contract;
targets are `.PHONY` because the real data files are gitignored and mtime
tracking is brittle. Cleanup is tiered (`clean → clean-cache → clean-data →
clean-all`).

Tests are two-tiered: fast network-free unit tests poke each function with
hand-crafted inputs (including synthetic EDGAR/Ken French payloads, so the
parsing is pinned even though the live download isn't), and integration tests
wire the real modules together in a temp directory. `pyproject.toml` sets
`pythonpath = ["src", "."]` so both `qer` and `scripts` import regardless of how
pytest is launched (bare `pytest`, `uv run pytest`, `python -m pytest`, or an
IDE) — without it, `make test` and `python -m pytest` disagree, because only the
latter puts the project root on the path.

## Honest seams in the current build

Documented deliberately:

- **Multi-class shares undercount market cap.** The cover-page shares figure for
  names like GOOGL/GOOG or BRK-A/BRK-B captures a single class, so `market_cap`
  (and the `size` factor) is wrong for those names until per-class aggregation is
  added.
- **`idio_skew_60d` is an approximation.** It uses a rolling per-date beta rather
  than a single regression over the skew window, so its residuals don't come from
  one coherent market model — a vectorisation shortcut, not the Kumar/Bali
  single-window-regression definition. Cross-sectional rank agreement with the
  textbook estimator is ~0.97, so the factor signal is close but the level is not
  the textbook quantity. The docstring now states this plainly; the shortcut is
  kept deliberately, and the exact construction is the fix to make if a writeup
  cites the literature definition.
- **The deflated Sharpe leans on two estimates.** Its selection benchmark is
  scaled by the cross-trial Sharpe variance, which is estimated from only ~8
  trials and is therefore noisy; and its overlap correction uses a heuristic
  effective sample size (`n / primary_horizon`) rather than a formal
  autocorrelation-based one. Both are reasonable and conservative, not exact.
- **An "All-NaN slice encountered" RuntimeWarning** can surface (pandas-version
  dependent) from rolling factors over entirely-missing ticker windows in the
  survivorship-free panel. It is benign — those windows are correctly NaN — and
  is also a faint data-quality signal of empty columns.

## Phase 3: graph features (planned)

The notes below describe *design intent*, not shipped code — the graph layer is
the next phase. They are recorded here so the structural decisions are settled
before implementation. The governing constraint is that nothing about graph
features gets its own evaluation path.

**A graph feature is a `Factor`.** Each relational signal — a centrality, a
community label, a lead-lag in/out-degree, a text-neighbour return — produces the
same `date x ticker` panel of node-level numbers as a classical factor,
subclasses the existing `Factor` ABC, registers, and is judged by the Phase 2
harness untouched. The first task of the phase is a *harness-reuse spike*: push an
existing classical factor (e.g. momentum) through the whole graph evaluation path
with neutralisation off and assert it reproduces the Phase 2 IC/Sharpe numbers —
proving "a graph feature is a Factor" before any graph is built.

**The one structural difference: a per-rebalance loop, not a single vectorised
pass.** A centrality or community label is a property of a graph built from a
trailing window, so it cannot be written as a rolling/shift transform. The engine
loops over monthly snapshot dates, builds the graph from the window ending at *t*,
writes node features into the cross-section at *t*, then forward-fills to the
daily panel between rebuilds. Monthly (not daily) rebuilds are a
fidelity-and-clarity choice, not a compute necessity — a 120-250-day graph barely
moves day to day. Compute is explicitly *not* the binding cost at ~500 names (a
500x500 correlation is trivial, an MST near-instant, exact betweenness runs in
seconds); the real costs are 10-K text ingestion and the multiple-testing burden
of the configuration grid, so attention is spent there.

**One engine enforces correctness once** (`graphs/windows.py`, `graphs/panel.py`).
The trailing-return matrix is sliced from the point-in-time universe with a
minimum-history requirement (insufficient-history names are dropped, not
zero-filled, so a fresh listing can't contaminate a correlation), the window looks
strictly backward — the single look-ahead boundary — and the loop, the
forward-fill, the cache, and the cross-sectional neutralisation all live in one
place. A new graph feature is a pure function from a window to a per-name `Series`
and inherits look-ahead-safety, survivorship discipline, and exposure control from
the engine.

**Construction choices that matter** (recorded so they aren't relitigated).
Correlations are Ledoit-Wolf-shrunk before anything else — the cheapest defence
against high-dimensional instability — and the window starts at 120 days, not 60.
At least two sparsifiers are kept so a result isn't a one-method artefact: a hard
threshold on `|rho|` and a minimum spanning tree on the Mantegna distance
`d = sqrt(2(1-rho))` (PMFG optional). Two subtleties bite. Eigenvector centrality
assumes non-negative weights (Perron-Frobenius), but correlations are signed, so
it is computed on the MST *topology* (or on `|rho|`), never on the distance
weights the tree was built from — distance weights would rank the most
*dissimilar* nodes as most central and invert the ordering. And Louvain/Leiden
community labels are stochastic, arbitrary integers, so any cross-snapshot
"delta-community" feature must first align labels across consecutive snapshots
(greedy Jaccard / Hungarian matching on membership overlap), or most of the
apparent migration is relabelling noise.

**Controlled before believed.** A relational feature is presumed to be a
size/liquidity/sector exposure until neutralisation says otherwise, and presumed
lucky until the full trial count says otherwise. The order of operations is fixed:
raw feature -> cross-sectional neutralisation (rank-transform, then regress out log
market cap, Amihud illiquidity, and GICS sector) -> decile long-short -> gross
IC/Sharpe -> net Sharpe after a *both-leg* turnover cost (delta-centrality and
lead-lag features are high-turnover, and a feature that survives only gross of
cost is a market-structure finding, not a strategy) -> FF5 alpha -> unspanned
alpha against the full classical set -> deflated Sharpe on the full trial count. A
feature must clear every stage. Neutralisation and the spanning test are *not*
redundant: the former removes a cross-sectional characteristic tilt (a stock's
size, illiquidity, sector), the latter removes covariance with the classical
factor *returns*; a feature can pass one and fail the other, so both run.

**The spanning test, and what the bootstrap does and doesn't buy.** Unspanned
alpha is the HAC intercept of the graph long-short regressed on the eight
classical long-shorts (reusing `exposures._hac_ols_cov`); because the classical
legs are themselves collinear, the intercept is read, not the loadings. For a
single test asset that intercept t-stat *is* the spanning statistic — it equals
the Gibbons-Ross-Shanken statistic (GRS reduces to t-squared for one asset), and
the ex-post gain in maximum Sharpe from adding the feature equals its squared
appraisal ratio `alpha^2 / sigma^2(eps)`, the same quantity the t measures. The
block-bootstrap tangency view is therefore a *robustness check* against the fat
tails of one daily series (it drops the iid-normal assumption GRS makes and the
finite-sample HAC distortion), not an independent test or a power gain. Genuine
extra power comes only from a joint multi-asset GRS across a basket of graph
features, which aggregates many alphas rather than re-examining one.

**Honest seams named up front.** Lead-lag effects are weak and decay fast in
liquid large-caps — this class is the *expected* failure/decay case the done-when
criterion requires, and the FDR-vs-shuffled-null comparison is built so that "no
real edge structure here" is a presentable finding rather than a dead end. Granger
causality stays off the critical path because `statsmodels` is declared but not
importable in this environment, so the lead-lag engine is a numpy lagged
cross-correlation with a Bartlett edge test (per-(pair, lag) standard error
`~ 1/sqrt(T)`, valid when the residual series are approximately white) and BH FDR;
Granger is gated behind adding `statsmodels` as a genuine dependency. Text
ingestion is the largest new lift and the biggest data-quality unknown: a per-year
coverage report (parseable Item 1 fraction, embedding-success rate, sanity of a
few firms' nearest neighbours) is a gate that must appear on the scorecard before
any text signal is trusted, and `sentence-transformers`/`torch` live behind an
optional `text` extra so the factor skips rather than crashes when the stack is
absent. The single largest risk is self-deception via the trial count: the grid
(windows x sparsifiers x centralities x communities x lead-lag lags x kNN) is
easily hundreds of features, so it is pre-registered before forward returns are
looked at, every configuration is logged to a trials ledger
(`data/graphs/trials.parquet`) pass or fail, and the deflated Sharpe and the BH
step are fed the *total* grid size, not the count of reported winners.
