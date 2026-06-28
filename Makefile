# Makefile for the Quant Equity Research project.
#
# Conventions:
#   * Phony targets only - no file-based dependencies, because the actual
#     data files are gitignored and present_files mtime tracking is brittle
#     across machines.
#   * uv is the canonical environment manager. All Python commands go through
#     `uv run` so the right venv is used without manual activation.
#   * Targets are arranged in dependency order: install -> data -> audit
#     -> factors. `make all` runs the full chain.
#   * Help is the default target so a bare `make` shows what's available.

# ---------------------------------------------------------------------------
# Variables - override on the command line, e.g. `make test PYTEST_ARGS=-x`
# ---------------------------------------------------------------------------

PYTHON := uv run python
PYTEST := uv run pytest
PYTEST_ARGS ?= -q

# ---------------------------------------------------------------------------
# .PHONY declarations - every target here is a command, not a file
# ---------------------------------------------------------------------------

.PHONY: help install \
        membership ingest sectors spy data membership-refresh \
        audit factors notebooks shares fundamentals ff5 \
        test test-unit test-integration test-fast test-cov \
        lint format typecheck check \
        clean clean-data clean-cache clean-all \
        all

# ---------------------------------------------------------------------------
# Help is the default - `make` with no args shows what's available
# ---------------------------------------------------------------------------

help:
	@echo "Quant Equity Research - Make targets"
	@echo ""
	@echo "Setup:"
	@echo "  install         Install package + dev dependencies via uv"
	@echo ""
	@echo "Data pipeline (Phase 1):"
	@echo "  membership      Rebuild S&P 500 membership table from Wikipedia"
	@echo "  ingest          Pull per-ticker price data via yfinance"
	@echo "  sectors         Pull per-ticker sector metadata via yfinance"
	@echo "  spy             Pull SPY benchmark history (for the audit check)"
	@echo "  data            Run all data-build steps (membership + ingest + sectors + spy)"
	@echo ""
	@echo "Analysis:"
	@echo "  audit           Run data-audit charts and sanity checks"
	@echo "  factors         Compute and plot factor IC analyses"
	@echo "  notebooks       Execute the narrative notebooks as a smoke test (needs data)"
	@echo ""
	@echo "Quality:"
	@echo "  test            Run all tests"
	@echo "  test-unit       Run only unit tests (fast)"
	@echo "  test-integration  Run only integration tests (slow)"
	@echo "  test-cov        Run tests with coverage report"
	@echo "  lint            Run ruff linter (read-only check)"
	@echo "  format          Run ruff + black to fix formatting"
	@echo "  typecheck       Run mypy"
	@echo "  check           Run lint + typecheck + test (pre-commit gate)"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean           Remove Python bytecode and build artifacts"
	@echo "  clean-cache     Remove derived data caches (wide matrices, etc.)"
	@echo "  clean-data      Remove ALL data including raw downloads (destructive)"
	@echo "  clean-all       Remove everything: data + caches + build artifacts"
	@echo ""
	@echo "End-to-end:"
	@echo "  all             install -> data -> audit -> factors"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install:
	uv sync --all-extras
	@echo "Installed. Activate with: source .venv/bin/activate (or use \`uv run <cmd>\`)"

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
# Each step is independently runnable. `data` orchestrates the full chain.
# Pass --refresh to membership when you want to re-pull Wikipedia.

membership:
	$(PYTHON) -m scripts.build_membership

membership-refresh:
	$(PYTHON) -m scripts.build_membership --refresh

ingest:
	$(PYTHON) -m scripts.ingest_prices

sectors:
	$(PYTHON) -m scripts.fetch_sectors

spy:
	$(PYTHON) -m scripts.fetch_spy

# Full data build. membership must run before ingest because ingest reads
# the membership table to know which tickers to fetch.
data: membership ingest sectors spy

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

audit:
	$(PYTHON) -m scripts.audit_data

factors:
	$(PYTHON) -m scripts.run_factor_zoo

# Optional, network-dependent data for the data-gated factors (size,
# value, quality). Both are scaffolds - see the scripts. Not in `data`.
shares:
	$(PYTHON) -m scripts.ingest_shares

fundamentals:
	$(PYTHON) -m scripts.ingest_fundamentals

ff5:
	$(PYTHON) -m scripts.fetch_ff5

# Smoke-test that the narrative notebooks still execute end-to-end against the
# current code (catches API drift, e.g. a renamed function). NOT part of `all`:
# notebooks are human-read narrative, not build artifacts, and this needs the
# data present (run `make data` first). Non-mutating - executes to a discarded
# output so the committed notebooks keep their cleared outputs. The active uv
# venv supplies the kernel, so a registered `qr-env` kernel is not required.
NOTEBOOKS := notebooks/01_data_audit.ipynb notebooks/02_first_factor_momentum.ipynb
# Dedicated .PHONY (in addition to the consolidated one above): the target name
# collides with the notebooks/ directory, so without this Make would consider
# `notebooks` "up to date" and skip the recipe.
.PHONY: notebooks
notebooks:
	@for nb in $(NOTEBOOKS); do \
		echo "Executing $$nb ..."; \
		$(PYTHON) -m jupyter nbconvert --to notebook --execute --stdout \
			--ExecutePreprocessor.kernel_name=python3 "$$nb" > /dev/null || exit 1; \
	done
	@echo "Notebooks executed cleanly (committed files unchanged)."

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
# Unit tests are fast (<5s) and should pass on every commit.
# Integration tests are slow (may hit network) and run on demand.

test: test-unit

test-unit:
	$(PYTEST) tests/unit $(PYTEST_ARGS)

test-integration:
	$(PYTEST) tests/integration $(PYTEST_ARGS)

test-fast:
	$(PYTEST) tests/unit $(PYTEST_ARGS) -x --ff

test-cov:
	$(PYTEST) tests/unit --cov=qer --cov-report=term-missing $(PYTEST_ARGS)

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff check --fix src tests scripts
	uv run black src tests scripts

typecheck:
	uv run mypy src

# The pre-commit gate: anything that should pass before pushing.
check: lint typecheck test-unit

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
# Three escalating levels:
#   clean       - just Python bytecode and build artifacts (cheap, safe)
#   clean-cache - + derived data (wide matrices). Re-derivable from raw.
#   clean-data  - + raw data. Will re-download from Wikipedia/yfinance.
#   clean-all   - everything.

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build dist *.egg-info src/*.egg-info

clean-cache:
	rm -rf data/processed data/wide data/audit

clean-data: clean-cache
	rm -rf data/raw

clean-all: clean clean-data

# ---------------------------------------------------------------------------
# End-to-end: the reviewer-facing target
# ---------------------------------------------------------------------------
# `make all` on a fresh clone should produce all artifacts: data, charts,
# factor results. This is what gets quoted in the README's reproducibility
# section. If `all` ever stops working, the project is broken.

all: install data audit factors
	@echo ""
	@echo "Build complete. Outputs:"
	@echo "  data/processed/membership.parquet"
	@echo "  data/raw/<ticker>.parquet (per ticker)"
	@echo "  data/audit/*.png"
	@echo ""
	@echo "Next: open notebooks/01_data_audit.ipynb for the narrative."
