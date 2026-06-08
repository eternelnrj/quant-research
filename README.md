# Quant Equity Research — Cross-Sectional Long-Short with Graph Features

A systematic US large-cap long-short equity research project combining classical
cross-sectional alpha factors with graph-based features derived from
correlation, lead-lag, and text-similarity networks between stocks.

The project is being built in phases. **Today the repository contains the data
pipeline, the point-in-time universe-reconstruction layer, a first classical
factor (12-1 momentum) behind a reusable evaluation harness, and the
data-audit/diagnostics layer.** The remaining factors, the graph features, the
backtester, portfolio construction, the robustness suite, and the written report
are planned — see *Current status* and *Planned methodology* below.

## Motivation

The goal of this project is to complete an end-to-end systematic equity
research pipeline of the kind a quantitative researcher would build on the
job — data ingestion, factor construction, evaluation, backtesting, portfolio
construction, risk management, and reporting — while adding a differentiated
component grounded in graph theory.

Classical alpha factors capture asset-level information (a stock's past return,
its size, its profitability). Graph-based features capture *relational*
information between assets — which stocks lead which, which are central in the
correlation network, which cluster together in business-description space.
This relational layer is uncommon in candidate projects and connects directly
to my background in discrete geometry and graph theory.

The aim is twofold: build the practical fluency expected of a quantitative
researcher, and produce a public artefact that demonstrates that fluency
end-to-end.

## Current status

What is implemented and exercised by the test suite today:

- **Data infrastructure.** A `DataLoader` presenting per-ticker raw parquet,
  lazily-built and cached `date x ticker` wide matrices, and derived
  returns/universe views, all behind a single interface with paths resolved
  from one config module.
- **Point-in-time universe.** S&P 500 historical membership reconstructed by
  scraping one current snapshot plus the Wikipedia changes log and replaying it
  backward, with entity resolution (renames collapsed, acquisitions/delistings
  dated) and a Wikipedia→Yahoo symbol mapping.
- **First classical factor.** 12-1 month momentum, no-look-ahead by
  construction.
- **Evaluation harness.** `compute_ic_series` takes any factor function and
  produces an IC / IC IR / t-stat / hit-rate summary and time-series chart.
- **Data audit.** Universe-size, missingness, return-distribution and sector
  charts, plus loud sanity checks (no future leakage, historical churn band,
  SPY total-return plausibility).
- **Reproducibility + tests.** A `.PHONY` Makefile pipeline and a unit +
  integration pytest suite.

Not yet built: the rest of the classical factor set, all graph features, the
backtester, portfolio construction/optimisation, the robustness suite, and the
PDF report. The `src/qer/{graphs,backtest,portfolio}` packages do not exist yet.

## Planned methodology

Phase tags below mark progress: **done**, **in progress**, or **planned**.

**Universe and data (Phase 1 — done).** US large-cap equities (S&P 500),
point-in-time historical membership, daily frequency. Split- and
dividend-adjusted prices (yfinance, `auto_adjust=False`, keying off the
`adj close` field) over the currently-configured 2008-2025 window, stored as
per-ticker parquet behind a single `DataLoader`. 

**Classical factors (Phase 2 — in progress).** Six to eight cross-sectional
factors behind a common interface — 12-1 month momentum (**done**), then
short-term reversal, size, value, volatility, liquidity, quality, and
idiosyncratic skewness (**planned**). Evaluated with a shared harness; the IC /
IC IR / t-stat / hit-rate core exists today, with decile spreads, turnover,
decay curves, Fama-French 5-factor exposures, Newey-West standard errors and
multiple-testing correction planned.

**Graph features (Phase 3 — planned).** Three families, each evaluated with the
Phase 2 harness:

1. **Correlation networks** — centralities (degree, eigenvector, betweenness)
   and community membership computed on minimum-spanning trees and
   k-nearest-neighbour graphs of trailing-window return correlations.
2. **Lead-lag networks** — directed graphs from residualised
   cross-correlations at 1-5 day lags; aggregate signals from upstream
   leaders.
3. **Text-similarity networks** — k-NN graphs on sentence-transformer
   embeddings of 10-K business descriptions, as an alternative to GICS sector
   labels.

For each graph feature, incremental value is measured by FF5-adjusted alpha
and by ablation against the classical factor set.

**Backtesting (Phase 4 — planned).** Vectorised walk-forward backtester with
strict IS/OOS separation, T+1 execution, linear-plus-impact transaction costs,
explicit short-borrow costs, and capacity-aware position sizing.

**Portfolio construction (Phase 5 — planned).** Signal combination via
equal-weighted z-score sum (baseline), IC-weighted, and ridge regression.
Portfolio optimisation via `cvxpy` with Ledoit-Wolf shrinkage, sector and beta
neutralisation, turnover penalty, and volatility targeting. Equal-weight
signal portfolios retained as the benchmark.

**Robustness (Phase 6 — planned).** Deflated Sharpe accounting for trial count,
combinatorially symmetric cross-validation, parameter sensitivity analysis,
regime conditioning (VIX terciles, bull/bear), and a pre-registered one-shot
final OOS test on held-out data.

**Reporting (Phase 7 — planned).** 20-30 page PDF writeup with executive
summary, methodology, results, robustness diagnostics, and honest limitations.
One-command reproducibility from raw data via `make all`.

## Target outcome

Aspirational, for the finished pipeline: net Sharpe in the 1.0-1.5 range after
realistic transaction costs and borrow, maximum drawdown 15-25%, with
statistically and economically meaningful incremental alpha from graph features
over the classical factor base.

## Timeline

This is a compressed sprint — roughly two months of focused  work.
Dates below are targets for "done enough to move on" under the project's
Definition of Done: code runs end-to-end with one command, notebooks tell a
clear story, result is reportable, next phase can consume the outputs.

| Phase | Description                           | Target completion | Status      |
| ----- | ------------------------------------- | ----------------- | ----------- |
| 0     | Foundations and environment           | 31 May 2026       | done        |
| 1     | Data infrastructure                   | 7 June 2026       | done        |
| 2     | Classical alpha factors               | 14 June 2026      | in progress |
| 3     | Graph-based features                  | 28 June 2026      | planned     |
| 4     | Backtesting framework                 | 5 July 2026       | planned     |
| 5     | Portfolio construction and risk       | 12 July 2026      | planned     |
| 6     | Robustness, validation, stress tests  | 19 July 2026      | planned     |
| 7     | Packaging, GitHub, and written report | 26 July 2026      | planned     |

## Repository structure

```
quant-equity-research/
├── pyproject.toml
├── uv.lock
├── README.md
├── Makefile                  # one-command reproducibility
├── src/qer/                  # importable package
│   ├── config.py             # single source of truth for paths + constants
│   ├── data/                 # DataLoader: prices, returns, universe
│   ├── universe/             # S&P 500 membership reconstruction + entity resolution
│   ├── factors/              # cross-sectional alpha factors (momentum so far)
│   └── diagnostics/          # factor IC evaluation + data audit
│       # graphs/ backtest/ portfolio/ — planned (Phases 3-5), not yet created
├── scripts/                  # thin CLI entry points (ingest, build, fetch, run)
├── tests/                    # unit + integration pytest suite
├── notebooks/                # numbered, narrative notebooks
├── docs/                     # design notes, glossary
└── data/                     # gitignored (rebuilt from source)
```

## Reproducibility

```bash
git clone https://github.com/eternelnrj/quant-equity-research
cd quant-equity-research
uv sync --all-extras             # creates .venv and installs all dependencies

make data                        # membership -> ingest prices -> sectors -> SPY benchmark
make audit                       # data-audit charts and sanity checks
make factors                     # regenerate the momentum IC chart from raw data
# or run the whole chain at once:
make all                         # install -> data -> audit -> factors

uv run pytest                    # runs the test suite
uv run jupyter lab               # opens the notebooks
```

`make data` needs network access (Wikipedia + yfinance); the unit test suite is offline and deterministic.

## Status

In progress, at Phase 1 complete / Phase 2 underway (see the table above). See
`docs/design_notes.md` for the structural rationale and the honest seams in the
current build, and `docs/glossary.md` for conventions.

## License

MIT.
