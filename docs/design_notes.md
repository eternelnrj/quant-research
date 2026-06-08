# Design Notes

Rationale behind the structural choices in this repository, written from what
the code actually does today rather than what the roadmap aspires to. The aim
is honesty about the seams, so the next person (or the next phase) knows where
the bodies are buried.

## Module layout

The package is organised by concern, behind a thin `src/qer/` namespace:

- `config.py` — the single source of truth for every path and a couple of
  date constants. 
- `data/loader.py` — the `DataLoader`, the one interface the rest of the code
  goes through for prices, returns, and the universe.
- `universe/` — membership reconstruction (`wikipedia.py`), interval building
  and entity resolution (`membership.py`), and the curated correction tables
  (`renames.py`).
- `factors/` — the factor library; momentum is the first member.
- `diagnostics/` — evaluation (`factor_ic.py`) and data audit (`audit_data.py`).

The deliberate split is *logic in the package, narrative in the notebooks*. The
diagnostic functions return data and figures; the numbered notebooks import
them and add prose. This keeps the heavy code testable and the notebooks
readable, and it is why the audit functions take a `loader` and an optional
`ax` rather than drawing to the global pyplot state.

## DataLoader: lazy, cached, single-interface

The loader presents three layers:

1. **Per-ticker raw parquet** — one file per symbol, full history. This is the
   ingestion landing zone and the unit of re-download.
2. **Wide matrices** — `date x ticker` per field, built on first access by
   reading every raw file and pivoting, then cached to `data/wide/`. Subsequent
   reads hit the disk cache; an in-memory dict short-circuits even that within
   a process.
3. **Derived views** — returns and universe-filtered cross-sections, computed
   on top of the wide matrices.

The reasons for the lazy wide-cache rather than eager materialisation: the cold
build is expensive (hundreds of file reads) but only needs to happen once, the
cache is re-derivable from raw at any time (so it is safe to `clean-cache`), and
keeping it lazy means a process that only needs the universe never pays for the
price pivot.

The price convention is stated once and now consistent across modules:
ingestion pulls with `auto_adjust=False`, so each raw file keeps a separate
`Adj Close` column alongside the raw OHLC. The `close` accessor exposes that
`adj close` field (split/dividend adjusted) and is what returns are computed
from, on purpose, so that corporate actions don't masquerade as returns. The
raw OHLC accessors (`open`, `high`, `low`) are deliberately *unadjusted* — only
`adj close` is corrected — and `cross_section` defaults to the raw `close`
field. That asymmetry between `close` (adjusted) and `cross_section`'s default
(raw) is easy to trip over when adding a new accessor, so it is called out in
each accessor's docstring.

## Universe reconstruction: scrape forward, replay backward

Rather than scrape old revisions of the Wikipedia page (whose formatting drifts
across years and silently drops class-share tickers), the design scrapes *one*
current snapshot plus the changes log, then reconstructs any past date by
reversing every change after it: undo additions, restore removals. One fetch
covers all of history, and the curated changes log is more reliable than
archived page revisions. Both tables are cached to `data/raw/wikipedia/` on
first fetch; `--refresh` forces a re-scrape.

`wikipedia.get_universe(date)` implements that replay directly and is the most
trustworthy path in the universe code — it is point-in-time correct (a name
added in 2020 never appears in a 2018 universe) and applies the Wikipedia→Yahoo
symbol mapping (`BRK.B → BRK-B`) so output is directly usable downstream.

`membership.build_membership_table()` is the richer path: it walks the same
events into explicit `(ticker, start, end)` intervals and applies entity
resolution. Entity resolution is the genuinely hard part, and the conventions
are encoded as data, not branches:

- **Renames** collapse old symbols onto the current one, walking multi-step
  chains (`WLP → ANTM → ELV`) cycle-safely.
- **Acquisitions/delistings** are exits, not renames, and get explicit dates in
  `TICKER_EXITS` for cases the changes log didn't record.
- **Same-day swaps** are handled by a sort tiebreaker that processes removals
  before additions, so a same-day remove+add yields two intervals meeting at
  the date instead of a zero-length one.
- **Dangling intervals** with no classification are closed conservatively at
  the last log date (preserving the membership, sacrificing exit precision)
  and surfaced as a warning rather than silently dropped.
- **Fresh adds the snapshot hasn't caught up on** are a special case of the
  above and treated differently: a ticker added on the most recent log date,
  with no removal and not yet present in the current-constituents snapshot, is
  *skipped* rather than closed at the last log date — closing it there would
  equal its own start date and produce an illegal zero-length interval. This is
  the two Wikipedia tables being momentarily out of sync (common right after an
  S&P announcement); the skip is surfaced as a `NOTE` and self-heals once the
  current table reconciles and the build is re-run with `--refresh`.

The builder is deliberately *pure*: it reconstructs and returns the table with
no I/O and no validation. Persisting to parquet and validating against the live
current set (in the same Yahoo ticker format, so class shares line up) are the
job of the CLI in `scripts/build_membership.py`. That separation is what lets
the builder be exercised on tiny synthetic inputs from tests and notebooks
without tripping the full-universe size check or writing to the real data
directory. These correction tables are explicitly manual and will rot; the
warnings on unclassified and pending-add tickers are the maintenance signal.

## No-look-ahead by construction

The momentum function does its window arithmetic by integer position on the
slice `prices.loc[:as_of]`, never by calendar date. Positional indexing
sidesteps weekend/holiday calendar bugs, and slicing to `as_of` up front means
the only data the function *can* see is historical — look-ahead is structurally
impossible, not merely avoided. Insufficient history returns NaN, never zero,
so downstream code is forced to handle "no signal" explicitly rather than
silently trading on a misleading flat value.

The price-convention flag, `log_prices`, says whether the input is *already* in
log space. It defaults to `False` — i.e. the common case of passing
`DataLoader.close` (raw adjusted price levels), which the function then logs
internally to produce a log-return momentum signal. Callers that have already
taken logs pass `log_prices=True`. The flag returns the same log return either
way; it only states whether the log has been taken yet.

## Evaluation harness

`factor_ic.compute_ic_series` takes a *factor function* as an argument rather
than hard-coding momentum, so every future factor plugs into the same harness
and is judged by the same IC / IR / t-stat / hit-rate summary. It restricts
each date's cross-section to the names actually in the index that day, drops
pairs missing either the factor or the forward return, and skips dates with too
thin a cross-section (rather than emitting an unstable correlation).

The harness feeds the factor function `np.log(close)`, so factors are evaluated
in log space. Momentum is plugged in through a small explicit wrapper,
`_momentum_12_1_logspace`, that calls `momentum_12_1(..., log_prices=True)` —
making the "the harness passes log prices" contract visible at the call site
rather than relying on a default. This is the seam that previously caused a
double-log (the harness logging, then a `log_prices`-defaulted momentum logging
again); the convenience path now matches intent.

## Sanity checks as first-class code

The audit module encodes three checks that have caught real bugs in this class
of project: no future leakage (the universe equals the set of covering
intervals on a probe date), historical churn inside a two-sided band
(survivorship-bias canary below, over-counted-exits canary above), and a
plausibility band on SPY's annualised total return. The SPY check reads the
dividend-adjusted `adj close` column, so it is a genuine *total*-return check;
SPY is an ETF, not an index constituent, so it is fetched by its own small
script (`scripts/fetch_spy.py` / `make spy`) rather than the constituent
ingest. The checks are written to fail loudly with diagnostic messages, because
a quiet data bug is far more expensive than a noisy assertion.

## Reproducibility via the Makefile

`make all` is the reviewer-facing contract: `install → data → audit → factors`
from a clean clone, where `data` is `membership → ingest → sectors → spy`. The
targets are intentionally all `.PHONY` — no file-based dependencies — because
the real data files are gitignored and mtime tracking is brittle across
machines. Cleanup is tiered (`clean` → `clean-cache` → `clean-data` →
`clean-all`) so the cheap, safe reset is the default and the destructive
re-download is opt-in.

The `notebooks` target is deliberately *not* part of `make all`. Notebooks are
human-read narrative, not build artifacts, and the numbered ones already call
the same `diagnostics` functions that `audit` and `factors` run headless — so
re-executing them in the build would duplicate work to no new artifact. Instead
`make notebooks` is a standalone, non-mutating smoke test: it executes the
notebooks to a discarded output (so committed files keep their cleared outputs)
purely to catch API drift, and it needs a data build first.

## Testing strategy

Two tiers, mirrored in the Makefile:

- **Unit** (`tests/unit`) — pure, fast, network-free, each function poked with
  hand-crafted inputs. This includes a momentum price-convention test (raw
  levels in, log returns out — never raw price differences), direct tests of
  `validate` against well-formed and deliberately broken tables, and regression
  tests pinning the previously-broken paths (the class-share validation format,
  the zero-length fresh-add interval, the wide-cache rebuild). The live-scrape
  universe tests skip cleanly when Wikipedia is unreachable rather than failing,
  so the suite stays offline-safe.
- **Integration** (`tests/integration`) — wires the real modules together
  across boundaries. `test_end_to_end.py` builds small realistic datasets in a
  temp directory, patches the loader's path constants at it, and exercises the
  full chains: Wikipedia replay → universe; membership parquet → loader → audit
  checks; price parquet → wide matrices → returns; prices → momentum → IC →
  summary.

The integration suite leans on two deterministic tricks. A pure
exponential-drift price set makes both momentum and the forward return strictly
monotonic in each ticker's drift, so the cross-sectional IC is *exactly* +1 —
an exact end-to-end correctness assertion, not a fuzzy "looks positive". A
fixed-seed random-walk set then provides genuine IC variance so the
variance-sensitive summary statistics (IR, t-stat) are exercised reproducibly.

## Known rough edges

Documented deliberately, in the spirit of the project's be-honest-about-the-
seams ethos. 

- **Yahoo data gaps for delisted names.** A survivorship-bias-free universe
  includes every name ever in the index since 2008, and a large fraction are
  now delisted/acquired. yfinance does not reliably serve long-dead symbols, so
  ingestion skips and reports roughly 150 of them. The pipeline degrades
  gracefully (the IC and audit code drops NaN names per cross-section), but a
  genuinely survivorship-clean backtest needs delisting-aware data from a better
  source (Stooq via pandas-datareader, or CRSP/Norgate/Sharadar). The audit's
  missingness heatmap is the tool for sizing the gap inside a given window.

- **Overlapping-window IC significance is optimistic.** `summarize_ic` computes
  the IR and t-stat as if the daily ICs were independent, but a 21-day forward
  return sampled every trading day overlaps its neighbours in 20 of 21 days, so
  the effective sample is ~N/21 and the reported significance is overstated. A
  Newey–West correction, or sampling ICs every `forward_days` to make them
  non-overlapping, would give honest significance for the daily-frequency path.

- **The factor interface is an implicit convention.** The harness assumes a
  `factor_func(prices_df, as_of_date) -> Series` callable, and the `log_prices`
  flag negotiates price representation per call. That held for one factor, but
  the flag remains a footgun and the implicit contract will not generalise to
  factors needing different inputs (fundamentals, volume, returns). A formal
  `Factor` protocol and a single decision about canonical input representation
  are the right next structural move, before the factor library grows.

- **Per-date factor computation will not scale.** `compute_ic_series` loops over
  dates and re-slices the price frame on each, which is fine for one factor over
  a few years but is the obvious bottleneck once there are many factors over the
  full universe and history. The factor layer needs to be rewritten so that instead of looping over dates and subsetting the price data, it computes the full factor values for all dates and all assets at once using vectorised matrix operations. 

- **Manual correction tables will rot.** `TICKER_RENAMES` and `TICKER_EXITS` are
  curated by hand and lag corporate actions; the unclassified-ticker and
  pending-add warnings are the only signal that they need topping up.

- **No committed sample data.** The notebooks and the `make notebooks` smoke
  test depend on a prior `make data`, which depends on live Yahoo/Wikipedia, so
  neither is runnable offline on a fresh clone. A tiny committed sample dataset
  (or promoting the integration suite's synthetic-data fixtures to a
  `make sample-data`) would make both offline- and CI-friendly.
