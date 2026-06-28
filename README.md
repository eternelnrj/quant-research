# Quant Equity Research — Cross-Sectional Long-Short with Graph Features

A systematic US large-cap long-short equity research project that pairs classical
cross-sectional alpha factors with a planned layer of graph-based features derived
from correlation, lead-lag, and text-similarity networks between stocks.

This repository contains the data pipeline, factor library, evaluation harness,
robustness diagnostics, and (in later phases) backtesting, portfolio
construction, and a full written report.

> **Status.** Phases 0-2 — the point-in-time universe, the data pipeline, the
> eight classical factors, and the evaluation harness — are implemented and
> tested. **The graph features (Phase 3) and everything after are planned, not yet
> built**; the sections below marked *planned* describe design intent. See the
> roadmap for what exists today versus what is on the way.

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

Phases 0-2 are implemented and tested; Phases 3-7 are planned (see below).

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

| Factor               | Direction | Needs                 |
| -------------------- | --------- | --------------------- |
| `momentum_12_1`      | +1        | prices                |
| `reversal_1m`        | -1        | prices                |
| `volatility_60d`     | -1        | prices                |
| `amihud_illiquidity` | +1        | prices + volume       |
| `idio_skew_60d`      | -1        | prices + market       |
| `size`               | -1        | prices + shares       |
| `value_btm`          | +1        | prices + fundamentals |
| `quality_gp`         | +1        | fundamentals          |

They are judged by one shared harness: multi-horizon information coefficient
(1/5/21-day), Newey-West (overlap-aware) IC t-statistics, decile long-short
Sharpe, Benjamini-Hochberg / Bonferroni multiple-testing correction, a deflated
Sharpe across the whole factor zoo, and Fama-French 5 exposures. The full table
is produced by `make factors`.

## Planned methodology (Phases 3-7)

**Graph features (Phase 3).** Three families of *relational* signals, each a
`Factor` evaluated through the Phase 2 harness untouched: correlation-network
centralities and communities (Ledoit-Wolf-shrunk correlations sparsified to a
minimum spanning tree on the Mantegna distance, plus a hard-threshold graph);
lead-lag directed graphs from market-residualised cross-correlations with a
Bartlett edge test and Benjamini-Hochberg FDR control against a shuffled-returns
null; and text-similarity k-NN graphs on sentence-transformer embeddings of 10-K
*Business* (Item 1) descriptions. A shared point-in-time engine (`graphs/windows.py`,
`graphs/panel.py`) builds each graph monthly from a trailing window over the
point-in-time universe and forward-fills node features to the daily panel, so
look-ahead, survivorship, and exposure-control are enforced once. Every feature
is cross-sectionally neutralised against size, illiquidity, and sector *before*
testing, charged turnover on both legs, regressed for FF5-adjusted alpha and for
unspanned alpha against the full eight-factor classical set (a Gibbons-Ross-Shanken
spanning test), and discounted by a deflated Sharpe fed the total
configuration-grid trial count. 

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
│   │   #   graph/            — Phase 3: thin Factor subclasses per graph feature
│   ├── diagnostics/          # IC, portfolios, exposures, multiple testing,
│   │                         #   deflated Sharpe, factor zoo, data audit
│   │   #   incremental.py graph_scorecard.py  — Phase 3: spanning test + scorecards
│   └── graphs/               # Phase 3: windows, correlation, centrality, leadlag,
│                             #   textsim, panel (point-in-time graph engine)
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

Phases 0-2 complete and tested; Phases 3-7 planned. See `docs/design_notes.md`
for the structural rationale and the honest seams in the current build, and
`docs/glossary.md` for conventions.

## License

MIT.
