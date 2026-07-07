"""Subphase 3.5: text-similarity networks from 10-K business descriptions.

A data-driven analogue of fixed industry codes (a TNIC, after Hoberg-Phillips): firms whose
Item 1 ("Business") text is similar are linked, and the recent returns of a firm's textual
neighbours become a predictor -- an industry-momentum signal on a graph the data chose.

The pipeline, per rebalance date ``t``:

1. take each firm's most recent 10-K filed on or before ``t`` (point-in-time), via
   :class:`EmbeddingStore`;
2. build a cosine k-nearest-neighbour graph over the embeddings (:func:`knn_graph`);
3. average each firm's neighbours' recent returns, weighted by textual similarity
   (:func:`neighbour_return_signal`).

Dependency split: the graph maths (cosine, kNN, neighbour signal, coverage, the point-in-time
store, Item-1 parsing) is numpy/pandas only and fully testable. Turning raw text into vectors
(:func:`embed_texts`) needs the optional ``text`` extra (sentence-transformers); if it is
absent, and no embedding cache has been ingested, the text factor simply does not register --
it skips rather than crashes.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Similarity and the kNN graph (numpy only)
# ---------------------------------------------------------------------------

def _unit_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0, norms, 1.0)


def cosine_similarity_matrix(embeddings) -> pd.DataFrame:
    """Pairwise cosine similarity of row-embeddings, in [-1, 1] (unit diagonal)."""
    emb = pd.DataFrame(embeddings)
    Xn = _unit_rows(emb.to_numpy(dtype=float))
    S = np.clip(Xn @ Xn.T, -1.0, 1.0)
    return pd.DataFrame(S, index=emb.index, columns=emb.index)


def knn_graph(embeddings, k: int = 20, min_sim: float = 0.0) -> pd.DataFrame:
    """Directed cosine k-nearest-neighbour graph (row ``i`` = ``i``'s ``k`` closest firms).

    ``A[i, j]`` is the cosine similarity of ``i`` and its neighbour ``j`` (``0`` otherwise,
    and ``0`` on the diagonal); neighbours below ``min_sim`` are dropped. Row-directed so the
    neighbour-return signal reads each firm's own peer set.
    """
    emb = pd.DataFrame(embeddings)
    names = emb.index
    Xn = _unit_rows(emb.to_numpy(dtype=float))
    n = Xn.shape[0]
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)                 # never a neighbour of itself
    A = np.zeros((n, n))
    kk = min(k, n - 1)
    if kk >= 1:
        top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]     # k largest per row
        rows = np.repeat(np.arange(n), kk)
        cols = top.ravel()
        vals = S[rows, cols]
        keep = vals >= min_sim
        A[rows[keep], cols[keep]] = np.clip(vals[keep], -1.0, 1.0)
    return pd.DataFrame(A, index=names, columns=names)


def neighbour_return_signal(adj, returns_window, lookback: int = 21) -> pd.Series:
    """Similarity-weighted mean of each firm's neighbours' recent returns.

    For firm ``i``: ``sum_j A[i,j] rbar_j / sum_j A[i,j]``, where ``rbar_j`` is neighbour
    ``j``'s mean return over the last ``lookback`` days. Restricted to firms present in both
    the graph and the return window; firms with no usable neighbour get NaN.
    """
    A_df = pd.DataFrame(adj)
    rw = pd.DataFrame(returns_window)
    common = A_df.index.intersection(rw.columns)
    if len(common) < 2:
        return pd.Series(dtype=float)
    A = A_df.loc[common, common].to_numpy(dtype=float)
    rbar = rw[common].iloc[-lookback:].mean(axis=0).to_numpy()
    w = A.sum(axis=1)
    num = A @ rbar
    sig = np.where(w > 0, num / np.where(w > 0, w, 1.0), np.nan)
    return pd.Series(sig, index=common, name="text_neighbour_return")


# ---------------------------------------------------------------------------
# Point-in-time embedding store
# ---------------------------------------------------------------------------

class EmbeddingStore:
    """Per-firm 10-K embeddings, queryable as of any date (most recent filing <= date)."""

    def __init__(self, filings: dict[str, list[tuple[pd.Timestamp, np.ndarray]]]):
        self._filings = {
            tk: sorted(recs, key=lambda dr: dr[0]) for tk, recs in filings.items()
        }

    @classmethod
    def from_frame(cls, df: pd.DataFrame, ticker_col: str = "ticker",
                   date_col: str = "filing_date") -> "EmbeddingStore":
        """Build from a tidy frame with ``ticker``, ``filing_date`` and ``emb_*`` columns."""
        emb_cols = sorted((c for c in df.columns if str(c).startswith("emb_")),
                          key=lambda c: int(str(c).split("_")[1]))
        filings: dict[str, list[tuple[pd.Timestamp, np.ndarray]]] = {}
        for tk, g in df.groupby(ticker_col):
            recs = [
                (pd.Timestamp(row[date_col]), g.loc[idx, emb_cols].to_numpy(dtype=float))
                for idx, row in g.iterrows()
            ]
            filings[str(tk)] = recs
        return cls(filings)

    def as_of(self, date, universe=None) -> pd.DataFrame | None:
        """Most-recent-filing embedding for each covered firm, as a ``(firms x dim)`` frame."""
        date = pd.Timestamp(date)
        firms = list(universe) if universe is not None else list(self._filings)
        rows: dict[str, np.ndarray] = {}
        for tk in firms:
            recs = self._filings.get(str(tk))
            if not recs:
                continue
            past = [emb for d, emb in recs if d <= date]
            if past:
                rows[str(tk)] = past[-1]          # filings are sorted; last <= date
        if not rows:
            return None
        return pd.DataFrame.from_dict(rows, orient="index")


def coverage_report(store: EmbeddingStore, checkpoints) -> pd.DataFrame:
    """Per-checkpoint text coverage: the data-quality gate for the text graph.

    ``checkpoints`` is an iterable of ``(date, universe)``. A graph on 60% coverage is a
    different, biased object than one on 95%, so this fraction is a first-class output.
    """
    rows = []
    for date, universe in checkpoints:
        emb = store.as_of(date, universe)
        n_cov = 0 if emb is None else len(emb)
        n_uni = len(list(universe))
        rows.append({
            "date": pd.Timestamp(date),
            "n_universe": n_uni,
            "n_covered": n_cov,
            "coverage": (n_cov / n_uni) if n_uni else 0.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Item 1 parsing (numpy/regex only)
# ---------------------------------------------------------------------------

def extract_item1(text: str) -> str:
    """Best-effort extraction of the Item 1 ("Business") section from 10-K text.

    Returns the text between the first ``Item 1.`` heading and the next ``Item 1A``/``Item 2``
    heading (case-insensitive), stripped of tags/whitespace; empty string if not found.
    """
    plain = re.sub(r"<[^>]+>", " ", text)                 # strip any HTML tags
    plain = re.sub(r"\s+", " ", plain)
    start = re.search(r"item\s*1\.?\s+business", plain, flags=re.IGNORECASE)
    if not start:
        return ""
    tail = plain[start.end():]
    end = re.search(r"item\s*(1a|2)\b", tail, flags=re.IGNORECASE)
    body = tail[: end.start()] if end else tail
    return body.strip()


# ---------------------------------------------------------------------------
# Embedding (optional: the 'text' extra)
# ---------------------------------------------------------------------------

def embed_texts(texts, model_name: str = "all-MiniLM-L6-v2",
                batch_size: int = 32) -> np.ndarray:  # pragma: no cover - needs the extra
    """Sentence-transformer embeddings (L2-normalised). Requires the ``text`` extra."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "embed_texts needs sentence-transformers; install the 'text' extra "
            "(pip install -e '.[text]')."
        ) from exc
    model = SentenceTransformer(model_name)
    emb = model.encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                       normalize_embeddings=True)
    return np.asarray(emb, dtype=float)
