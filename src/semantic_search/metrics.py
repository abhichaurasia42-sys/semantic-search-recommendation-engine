"""Ranking evaluation metrics: NDCG@k and Recall@k."""
from __future__ import annotations

import math


def dcg_at_k(relevances: list[int], k: int) -> float:
    """relevances: graded relevance of the ranked list, in ranked order."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2**rel - 1) / math.log2(i + 2)  # i is 0-indexed -> rank i+1
    return dcg


def ndcg_at_k(ranked_doc_ids: list[str], relevance_by_doc: dict[str, int], k: int = 10) -> float:
    """NDCG@k for a single query.

    `ranked_doc_ids` is the system's ranking (best first). `relevance_by_doc`
    maps doc_id -> graded relevance (missing = 0, i.e. non-relevant).
    """
    gains = [relevance_by_doc.get(doc_id, 0) for doc_id in ranked_doc_ids[:k]]
    actual_dcg = dcg_at_k(gains, k)

    ideal_gains = sorted(relevance_by_doc.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal_gains, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def recall_at_k(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int = 20) -> float:
    """Fraction of all known-relevant docs that appear in the top k of the ranking."""
    if not relevant_doc_ids:
        return 0.0
    top_k = set(ranked_doc_ids[:k])
    hits = len(top_k & relevant_doc_ids)
    return hits / len(relevant_doc_ids)
