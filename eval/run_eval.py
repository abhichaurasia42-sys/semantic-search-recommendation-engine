#!/usr/bin/env python
"""Runs the retrieval/ranking evaluation and prints real, measured metrics.

Computes:
  - NDCG@10 for the hybrid (BM25 + vector) retriever
  - Recall@20 for BM25-only vs. BM25 + scikit-learn re-ranker

against eval/eval_set.jsonl, using the bundled data/corpus.jsonl. All numbers
printed are actually computed on this run, not hard-coded.

Run:
    python eval/run_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_search.bm25 import BM25Index  # noqa: E402
from semantic_search.config import (  # noqa: E402
    CORPUS_PATH,
    EVAL_SET_PATH,
    RERANKER_MODEL_PATH,
    TFIDF_VECTORIZER_PATH,
)
from semantic_search.corpus import load_corpus  # noqa: E402
from semantic_search.embeddings import Embedder  # noqa: E402
from semantic_search.hybrid_retriever import HybridRetriever  # noqa: E402
from semantic_search.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from semantic_search.reranker import Reranker, RerankFeaturizer  # noqa: E402
from semantic_search.vector_store import SQLiteVectorStore  # noqa: E402

NDCG_K = 10
RECALL_K = 20
CANDIDATE_POOL = 50


def load_eval_set(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    t0 = time.time()
    docs = load_corpus(CORPUS_PATH)
    doc_text_by_id = {d.id: d.full_text for d in docs}
    eval_rows = load_eval_set(EVAL_SET_PATH)
    print(f"Loaded {len(docs)} documents and {len(eval_rows)} eval queries.")

    bm25 = BM25Index(docs)

    embedder = Embedder()
    print(f"Embedding backend: {'FALLBACK hashed BoW' if embedder.using_fallback else embedder.model_name}")
    doc_embeddings = embedder.encode([d.full_text for d in docs])
    store = SQLiteVectorStore(":memory:")
    store.upsert([d.id for d in docs], doc_embeddings)

    hybrid = HybridRetriever(docs, bm25, store, embedder)

    have_reranker = RERANKER_MODEL_PATH.exists() and TFIDF_VECTORIZER_PATH.exists()
    reranker = None
    featurizer = None
    if have_reranker:
        reranker = Reranker.load(RERANKER_MODEL_PATH)
        import joblib

        featurizer = joblib.load(TFIDF_VECTORIZER_PATH)
    else:
        print(
            "WARNING: no trained re-ranker found at "
            f"{RERANKER_MODEL_PATH}. Run scripts/train_reranker.py first. "
            "Skipping BM25+reranker recall@20."
        )

    ndcg_scores = []
    bm25_recalls = []
    reranked_recalls = []

    for row in eval_rows:
        query = row["query"]
        relevance_by_doc = {j["doc_id"]: j["relevance"] for j in row["judgments"]}
        relevant_doc_ids = {doc_id for doc_id, rel in relevance_by_doc.items() if rel >= 1}

        # --- NDCG@10 for the hybrid retriever ---
        hybrid_results = hybrid.search(query, top_k=NDCG_K, candidate_pool=CANDIDATE_POOL, method="weighted", alpha=0.5)
        hybrid_ranked_ids = [r.doc_id for r in hybrid_results]
        ndcg_scores.append(ndcg_at_k(hybrid_ranked_ids, relevance_by_doc, k=NDCG_K))

        # --- Recall@20: BM25 baseline ---
        bm25_hits = bm25.search(query, top_k=CANDIDATE_POOL)
        bm25_ranked_ids = [h.doc_id for h in bm25_hits]
        bm25_recalls.append(recall_at_k(bm25_ranked_ids, relevant_doc_ids, k=RECALL_K))

        # --- Recall@20: BM25 + re-ranker ---
        if reranker is not None:
            candidate_rows = [
                (query, h.doc_id, doc_text_by_id[h.doc_id], h.score)
                for h in bm25_hits
                if h.doc_id in doc_text_by_id
            ]
            if candidate_rows:
                X = featurizer.transform_many(candidate_rows)
                scores = reranker.score(X)
                order = sorted(
                    zip([r[1] for r in candidate_rows], scores), key=lambda t: t[1], reverse=True
                )
                reranked_ids = [doc_id for doc_id, _ in order]
            else:
                reranked_ids = []
            reranked_recalls.append(recall_at_k(reranked_ids, relevant_doc_ids, k=RECALL_K))

    n = len(eval_rows)
    mean_ndcg = sum(ndcg_scores) / n
    mean_bm25_recall = sum(bm25_recalls) / n

    print()
    print("=" * 60)
    print("RESULTS (measured on this run, bundled sample corpus + eval set)")
    print("=" * 60)
    print(f"Corpus size:            {len(docs)} documents")
    print(f"Eval queries:           {n}")
    print(f"Eval judgment pairs:    {sum(len(r['judgments']) for r in eval_rows)}")
    print(f"Hybrid retriever NDCG@{NDCG_K}:  {mean_ndcg:.4f}")
    print(f"BM25-only Recall@{RECALL_K}:      {mean_bm25_recall:.4f}")
    if reranker is not None:
        mean_reranked_recall = sum(reranked_recalls) / n
        delta = mean_reranked_recall - mean_bm25_recall
        print(f"BM25+Reranker Recall@{RECALL_K}: {mean_reranked_recall:.4f}  (delta vs BM25-only: {delta:+.4f})")
    print("=" * 60)
    print(f"Eval run took {time.time() - t0:.1f}s")

    results = {
        "corpus_size": len(docs),
        "n_eval_queries": n,
        "n_judgment_pairs": sum(len(r["judgments"]) for r in eval_rows),
        "hybrid_ndcg_at_10": mean_ndcg,
        "bm25_recall_at_20": mean_bm25_recall,
    }
    if reranker is not None:
        results["bm25_plus_reranker_recall_at_20"] = sum(reranked_recalls) / n
    results_path = Path(__file__).resolve().parent / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
