# Quant Equity Research — Cross-Sectional Long-Short with Graph Features

A systematic US large-cap long-short equity research project that pairs classical
cross-sectional alpha factors with a layer of graph-based features derived
from correlation, lead-lag, and text-similarity networks between stocks.

This repository contains the data pipeline, factor library, evaluation harness,
robustness diagnostics, and (in later phases) backtesting, portfolio
construction, and a full written report.

> **Status.** Phases 0-3 — the point-in-time universe, the data pipeline, the
> eight classical factors, the evaluation harness, and the full graph-feature
> layer (correlation, lead-lag, and text-similarity networks with their
> incremental-value scorecard) — are implemented and tested. **Phases 4 and
> after are planned, not yet built**; the sections below marked *planned*
> describe design intent. See the roadmap for what exists today versus what is
> on the way.

## Motivation

The goal is an end-to-end systematic equity research pipeline of the kind a
quantitative researcher would build on the job — data ingestion, factor
construction, evaluation, backtesting, portfolio construction, risk management,
and reporting — while adding a differentiated component grounded in graph
theory.

Classical alpha factors capture asset-level information (a stock's past return,
its size, its profitability). Graph-based features capture *relational*
information between assets — which stocks lead which, which are central in the
correlation network, which cluster together in business-description space. This
relational layer is uncommon in candidate projects and connects directly to my
background in discrete geometry and graph theory.

## What is built today

Phases 0-3 are implemented and tested; Phases 4-7 are planned (see below).

**Universe and data (Phases 0-1).** US large-cap equities (S&P 500),
point-in-time historical membership reconstructed from Wikipedia's current
snapshot plus its changes log (scrape forward, replay backward), with entity
resolution for renames, acquisitions, and same-day swaps. Daily split- and
dividend-adjusted prices (yfinance). Fundamentals and shares outstanding from
SEC EDGAR, dated at each filing's actual `filed` date for point-in-time
correctness. Fama-French 5 factors from Ken French's data library. Everything is
stored as parquet behind a single lazy, cached `DataLoader`.

**Classical factors and evaluation (Phase 2).** Eight cross-sectional factors
behind a common `Factor` interface, each computed as a vectorised, look-ahead-safe
`date x ticker` panel:

| Factor              | Direction | Needs                |
| ------------------- | --------- | -------------------- |
| `momentum_12_1`     | +1        | prices               |
| `reversal_1m`       | -1        | prices               |
| `volatility_60d`    | -1        | prices               |
| `amihud_illiquidity`| +1        | prices + volume      |
| `idio_skew_60d`     | -1        | prices + market      |
| `size`              | -1        | prices + shares      |
| `value_btm`         | +1        | prices + fundamentals|
| `quality_gp`        | +1        | fundamentals         |

They are judged by one shared harness: multi-horizon information coefficient
(1/5/21-day), Newey-West (overlap-aware) IC t-statistics, decile long-short
Sharpe, Benjamini-Hochberg / Bonferroni multiple-testing correction, a deflated
Sharpe across the whole factor zoo, and Fama-French 5 exposures. The full table
is produced by `make factors`.

**Graph features (Phase 3).** Three families of *relational* signals, each a
`Factor` evaluated through the Phase 2 harness untouched and registered in a
*separate* `GRAPH_FACTORS` registry so they never enlarge the classical factor
zoo or its multiple-testing denominator:

- **Correlation networks** — Ledoit-Wolf-shrunk correlations (a numpy estimator
  verified to match `scikit-learn`) sparsified to a minimum spanning tree on the
  Mantegna distance (`|rho|`-weighted, `epsilon`-floored so a zero-variance name
  can't disconnect the backbone) and a hard-threshold graph, with degree /
  eigenvector / clustering / betweenness centralities, Louvain/Leiden communities
  and cohesion, Hungarian label alignment, and delta-centrality variants.
- **Lead-lag networks** — market-residualised lagged cross-correlations with a
  Bartlett edge test and Benjamini-Hochberg FDR over all `(i, j, lag)` pairs,
  giving directed in/out-degree (followers/leaders) and an upstream signal, plus
  a circular-shuffle honest-null density report (an independence test, with the
  residual autocorrelation reported so its reach is auditable).
- **Text-similarity networks** — a cosine k-NN graph over sentence-transformer
  embeddings of 10-K *Business* (Item 1) text, giving a neighbour-return signal,
  point-in-time via an embedding store (skips cleanly without the `text` extra).

A shared point-in-time engine (`graphs/windows.py`, `graphs/panel.py`) builds each
graph monthly from a trailing window over the point-in-time universe and
forward-fills node features to the daily panel, so look-ahead, survivorship, and
exposure-control are enforced once. Every feature is cross-sectionally neutralised
in rank space against size, illiquidity, and sector *before* testing, and the
scorecard (`diagnostics/incremental.py`, `diagnostics/graph_scorecard.py`) reports
unspanned alpha against the full eight-factor classical set (HAC t-stat;
single-asset Gibbons-Ross-Shanken `= t^2`; the appraisal-ratio tangency identity
`theta_new = theta_F + IR^2`), a block-bootstrap alpha interval, the
cluster-vs-sector confusion matrix, and a deflated Sharpe fed the *total*
configuration-grid trial count. The core is numpy/scipy; betweenness/communities
use `networkx` behind the `graphs` extra, and Granger causality is gated on
`statsmodels`. Lead-lag is the honest failure/decay candidate the done-when
criterion calls for — weak in liquid large-caps, and reported as such rather than
fished into spurious edges.

## Planned methodology (Phases 4-7)

**Backtesting (Phase 4).** Vectorised walk-forward backtester with strict
IS/OOS separation, T+1 execution, linear-plus-impact transaction costs, explicit
short-borrow costs, and capacity-aware sizing.

**Portfolio construction (Phase 5).** Signal combination (equal-weight z-score,
IC-weighted, ridge) and optimisation via `cvxpy` with Ledoit-Wolf shrinkage,
sector/beta neutralisation, turnover penalty, and volatility targeting.

**Robustness (Phase 6).** Deflated Sharpe (started in Phase 2), combinatorially
symmetric cross-validation, parameter sensitivity, regime conditioning, and a
pre-registered one-shot final OOS test.

**Reporting (Phase 7).** A focused PDF report — a concise main body with the
heavier tables and robustness detail in appendices — and one-command reproducibility
via `make all`.

## Repository structure

```
quant-equity-research/
├── pyproject.toml
├── README.md
├── Makefile                  # one-command reproducibility
├── src/qer/                  # importable package
│   ├── config.py             # single source of truth for paths/constants
│   ├── data/                 # loader, fundamentals loader, EDGAR client
│   ├── universe/             # S&P 500 membership reconstruction
│   ├── factors/              # Factor ABC + registry + 8 factors
│   │   #   graph/            — Phase 3: graph Factors + separate GRAPH_FACTORS registry
│   ├── diagnostics/          # IC, portfolios, exposures, multiple testing,
│   │                         #   deflated Sharpe, factor zoo, data audit
│   │   #   incremental.py graph_scorecard.py  — Phase 3: spanning test + scorecards
│   └── graphs/               # Phase 3 engine: windows, panel, correlation, centrality,
│                             #   leadlag, textsim, grid, trials, spike
├── scripts/                  # thin CLI entry points (ingest + run)
│   #   ingest_10k_text.py    — Phase 3: EDGAR 10-K Item 1 text, point-in-time
├── tests/                    # unit + integration pytest suites
├── notebooks/                # numbered, narrative notebooks
├── docs/                     # design notes, glossary
└── data/                     # gitignored
```

## Reproducibility

```bash
git clone https://github.com/eternelnrj/quant-equity-research
cd quant-equity-research
uv sync --all-extras          # creates .venv and installs dependencies
uv pip install -e .           # editable install so scripts/notebooks import qer

make data                     # membership + prices (+ sectors, spy)
make shares fundamentals ff5  # EDGAR shares + fundamentals, Ken French FF5
make audit                    # data-audit charts and sanity checks
make factors                  # full factor-zoo evaluation table
# or the whole chain:
make all                      # install -> data -> audit -> factors

make test                     # unit suite (uv run pytest)
uv run jupyter lab            # opens the notebooks
```

The SEC EDGAR ingests (`shares`, `fundamentals`) require a contact User-Agent,
which SEC mandates and the scripts refuse to run without:

```bash
export QER_SEC_USER_AGENT="Your Name your@email.com"
```

## Status

Phases 0-3 complete and tested; Phases 4-7 planned. See `docs/design_notes.md`
for the structural rationale and the honest seams in the current build, and
`docs/glossary.md` for conventions.

## License

MIT.
