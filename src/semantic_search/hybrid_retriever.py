"""Hybrid retrieval: fuses BM25 lexical scores with vector cosine similarity.

Two fusion strategies are supported:

- "weighted": min-max normalize each ranker's scores over the candidate pool,
  then combine as `alpha * bm25_norm + (1 - alpha) * vector_norm`.
- "rrf": reciprocal rank fusion, combining `1 / (k + rank)` across the two
  rankers' orderings. RRF is scale-free and does not require score
  normalization, which makes it more robust when BM25 and cosine similarity
  scores live on very different numeric ranges.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic_search.bm25 import BM25Index
from semantic_search.corpus import Document
from semantic_search.embeddings import Embedder
from semantic_search.vector_store import VectorStore


@dataclass
class RetrievalResult:
    doc_id: str
    score: float
    bm25_score: float
    vector_score: float


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        # Degenerate case: a single candidate, or a tie across all of them.
        # Normalizing to 1.0 (rather than 0.0) preserves the fact that these
        # are the *best* (only) scored candidates, so they still outrank
        # candidates absent from `scores` entirely (which default to 0.0 via
        # dict.get elsewhere) instead of collapsing to an uninformative tie.
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    """Combines a BM25 index and a vector store into one ranked result list."""

    def __init__(
        self,
        documents: list[Document],
        bm25_index: BM25Index,
        vector_store: VectorStore,
        embedder: Embedder,
    ):
        self.documents = documents
        self.doc_by_id = {d.id: d for d in documents}
        self.bm25_index = bm25_index
        self.vector_store = vector_store
        self.embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_pool: int = 50,
        alpha: float = 0.5,
        method: str = "weighted",
    ) -> list[RetrievalResult]:
        """Retrieve and fuse results.

        `alpha` weights BM25 vs. vector similarity for the "weighted" method
        (1.0 = BM25 only, 0.0 = vector only). Ignored for "rrf".
        """
        bm25_hits = self.bm25_index.search(query, top_k=candidate_pool)
        bm25_scores = {h.doc_id: h.score for h in bm25_hits}

        query_vec = self.embedder.encode([query])[0]
        vector_hits = self.vector_store.search(query_vec, top_k=candidate_pool)
        vector_scores = {doc_id: sim for doc_id, sim in vector_hits}

        candidate_ids = set(bm25_scores) | set(vector_scores)
        if not candidate_ids:
            return []

        if method == "rrf":
            fused = self._rrf(bm25_scores, vector_scores)
        elif method == "weighted":
            fused = self._weighted(bm25_scores, vector_scores, candidate_ids, alpha)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

        ranked = sorted(candidate_ids, key=lambda doc_id: fused[doc_id], reverse=True)[:top_k]
        return [
            RetrievalResult(
                doc_id=doc_id,
                score=fused[doc_id],
                bm25_score=bm25_scores.get(doc_id, 0.0),
                vector_score=vector_scores.get(doc_id, 0.0),
            )
            for doc_id in ranked
        ]

    @staticmethod
    def _weighted(
        bm25_scores: dict[str, float],
        vector_scores: dict[str, float],
        candidate_ids: set[str],
        alpha: float,
    ) -> dict[str, float]:
        bm25_norm = _min_max_normalize(bm25_scores)
        vector_norm = _min_max_normalize(vector_scores)
        return {
            doc_id: alpha * bm25_norm.get(doc_id, 0.0) + (1 - alpha) * vector_norm.get(doc_id, 0.0)
            for doc_id in candidate_ids
        }

    @staticmethod
    def _rrf(bm25_scores: dict[str, float], vector_scores: dict[str, float], k: int = 60) -> dict[str, float]:
        def ranks(scores: dict[str, float]) -> dict[str, int]:
            ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            return {doc_id: rank for rank, (doc_id, _) in enumerate(ordered, start=1)}

        bm25_ranks = ranks(bm25_scores)
        vector_ranks = ranks(vector_scores)
        candidate_ids = set(bm25_scores) | set(vector_scores)
        fused: dict[str, float] = {}
        for doc_id in candidate_ids:
            score = 0.0
            if doc_id in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[doc_id])
            if doc_id in vector_ranks:
                score += 1.0 / (k + vector_ranks[doc_id])
            fused[doc_id] = score
        return fused
