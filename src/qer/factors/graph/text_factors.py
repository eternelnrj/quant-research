"""Subphase 3.5: text-similarity network factor (feature class C).

One feature -- ``text_neighbour_return`` -- an industry-momentum signal on a data-driven graph:
each firm's return is predicted by the recent returns of its textual neighbours. Unlike the
correlation and lead-lag factors, the graph here comes from *external, point-in-time* 10-K
embeddings rather than the return window, so this factor uses its own ``compute_panel`` (which
knows the rebalance date ``t``) instead of the generic snapshot loop.

The factor registers only when an embedding cache has been ingested (``TEXT_EMBEDDINGS_FILE``);
absent that -- e.g. a base install without the ``text`` extra or EDGAR ingestion -- it simply
does not appear in ``GRAPH_FACTORS``, i.e. it skips rather than crashes.
"""

from __future__ import annotations

import pandas as pd

from qer.factors.base import Factor
from qer.factors.graph.base import register_graph_factor
from qer.graphs.textsim import EmbeddingStore, knn_graph, neighbour_return_signal
from qer.graphs.windows import rebalance_dates


class TextGraphFactor(Factor):
    """Neighbour-return signal on a point-in-time cosine-kNN text graph.

    Inherits :class:`Factor` (the harness/registry interface), NOT :class:`GraphFactor`:
    ``GraphFactor`` is the specialisation for graphs built from the *return window* via a
    ``snapshot_fn`` + ``build_feature_panel``, whereas this factor's graph comes from external
    point-in-time embeddings and needs its own ``compute_panel`` and constructor. It registers
    and evaluates like any other ``Factor`` -- ``register_graph_factor`` and the harness are
    polymorphic over the ``Factor`` interface, with no subclass check.
    """

    def __init__(self, name: str, store: EmbeddingStore, *, k: int = 20, freq: str = "M",
                 lookback: int = 21, min_sim: float = 0.0, min_names: int = 10,
                 kind: str = "log", direction: int = 1):
        self.name = name
        self.direction = int(direction)
        self._store = store
        self._k = k
        self._freq = freq
        self._lookback = lookback
        self._min_sim = min_sim
        self._min_names = min_names
        self._kind = kind

    def compute_panel(self, loader) -> pd.DataFrame:
        cal = loader.close.index
        rets = loader.get_returns(self._kind)
        rows: dict[pd.Timestamp, pd.Series] = {}
        for t in rebalance_dates(cal, freq=self._freq):
            universe = loader.get_universe(t)
            emb = self._store.as_of(t, universe)                 # point-in-time embeddings
            if emb is None or len(emb) < self._min_names:
                continue
            adj = knn_graph(emb, k=self._k, min_sim=self._min_sim)
            recent = rets.loc[:t].iloc[-self._lookback:]         # returns up to t (no look-ahead)
            s = neighbour_return_signal(adj, recent, lookback=self._lookback)
            if len(s) == 0:
                continue
            rows[t] = s
        if not rows:
            return pd.DataFrame(index=cal)
        panel = pd.DataFrame(rows).T
        panel.index = pd.DatetimeIndex(panel.index)
        return panel.sort_index().reindex(cal).ffill()


def _register_class_c(window_k: int = 20) -> None:
    from qer.config import TEXT_EMBEDDINGS_FILE

    if not TEXT_EMBEDDINGS_FILE.exists():
        return  # no ingested 10-K embeddings -> the text factor skips (design: skip, not crash)
    store = EmbeddingStore.from_frame(pd.read_parquet(TEXT_EMBEDDINGS_FILE))
    register_graph_factor(
        TextGraphFactor(f"text_neighbour_return_k{window_k}", store, k=window_k, lookback=21)
    )


_register_class_c()
