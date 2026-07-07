"""Integration test for Subphase 3.5: the text-graph factor through the harness.

Uses a synthetic :class:`EmbeddingStore` (two planted clusters over the fixture tickers), builds
a :class:`TextGraphFactor` directly -- not via the registry, since there is no ingested cache --
and runs it through the Phase-2 harness. Also confirms the factor does not auto-register when no
embeddings cache exists (the "skip, don't crash" contract).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import qer.factors.graph as GF
from qer.diagnostics.factor_ic import compute_factor_ic
from qer.diagnostics.portfolios import factor_long_short
from qer.factors.graph.text_factors import TextGraphFactor
from qer.graphs.textsim import EmbeddingStore

DIM = 8


def _planted_store(tickers, seed=0):
    """One 10-K per ticker (dated before the fixture calendar); two clusters in embedding space."""
    rng = np.random.default_rng(seed)
    recs = []
    for i, tk in enumerate(tickers):
        center = np.zeros(DIM)
        center[0 if i < len(tickers) // 2 else 1] = 3.0
        emb = center + rng.normal(0, 0.3, DIM)
        row = {"ticker": tk, "filing_date": "2018-06-01"}
        row.update({f"emb_{j}": float(emb[j]) for j in range(DIM)})
        recs.append(row)
    return EmbeddingStore.from_frame(pd.DataFrame(recs))


def test_text_factor_not_registered_without_cache():
    # no TEXT_EMBEDDINGS_FILE in a base install -> no text factor in the registry
    assert not any("text" in f.name for f in GF.all_graph_factors())


def test_text_factor_builds_and_scores(synthetic_loader):
    tickers = list(synthetic_loader.close.columns)
    store = _planted_store(tickers)
    factor = TextGraphFactor("text_neighbour_return_test", store, k=5, lookback=21, min_names=10)

    panel = factor.compute_panel(synthetic_loader)
    assert panel.index.equals(synthetic_loader.close.index)     # daily, causal grid
    populated = panel.dropna(how="all")
    assert len(populated) > 0
    assert populated.iloc[-1].dropna().std() > 0                # non-trivial cross-section

    ic = compute_factor_ic(synthetic_loader, factor, horizons=(1, 5, 21))
    assert set(ic.keys()) == {1, 5, 21}
    assert np.isfinite(ic[21].dropna().to_numpy()).all()

    ls = factor_long_short(synthetic_loader, factor, n_buckets=5, horizon=21)
    assert np.isfinite(ls.dropna().to_numpy()).all()


def test_text_factor_is_point_in_time(synthetic_loader):
    # a store whose only filing post-dates the whole calendar -> no coverage -> empty panel
    tickers = list(synthetic_loader.close.columns)
    recs = [{"ticker": tk, "filing_date": "2099-01-01",
             **{f"emb_{j}": 1.0 for j in range(DIM)}} for tk in tickers]
    future_store = EmbeddingStore.from_frame(pd.DataFrame(recs))
    panel = TextGraphFactor("text_future", future_store, k=5, min_names=10).compute_panel(synthetic_loader)
    assert panel.dropna(how="all").empty                        # nothing is known in-sample yet
