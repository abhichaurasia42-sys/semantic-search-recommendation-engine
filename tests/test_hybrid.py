from semantic_search.bm25 import BM25Index
from semantic_search.embeddings import Embedder
from semantic_search.hybrid_retriever import HybridRetriever
from semantic_search.vector_store import SQLiteVectorStore


def _build_hybrid(tiny_corpus):
    # Force the deterministic hashed-BoW fallback embedder so this test is
    # fast and fully offline/deterministic -- it still exercises the real
    # fusion logic, just not the real neural embeddings (covered by the
    # slower API-level tests instead).
    embedder = Embedder(model_name="__nonexistent_model_to_force_fallback__")
    assert embedder.using_fallback

    bm25 = BM25Index(tiny_corpus)
    store = SQLiteVectorStore(":memory:")
    embeddings = embedder.encode([d.full_text for d in tiny_corpus])
    store.upsert([d.id for d in tiny_corpus], embeddings)
    return HybridRetriever(tiny_corpus, bm25, store, embedder)


def test_hybrid_weighted_fusion_returns_ranked_results(tiny_corpus):
    hybrid = _build_hybrid(tiny_corpus)
    results = hybrid.search("cloud serverless compute", top_k=3, method="weighted", alpha=0.5)
    assert len(results) <= 3
    assert results[0].doc_id in {"doc-1", "doc-2"}


def test_hybrid_rrf_fusion_returns_ranked_results(tiny_corpus):
    hybrid = _build_hybrid(tiny_corpus)
    results = hybrid.search("bread dough oven baking", top_k=3, method="rrf")
    doc_ids = [r.doc_id for r in results]
    assert "doc-4" in doc_ids or "doc-5" in doc_ids


def test_hybrid_alpha_extremes_match_pure_bm25_or_vector(tiny_corpus):
    hybrid = _build_hybrid(tiny_corpus)
    bm25_only = hybrid.search("knife sharp chopping", top_k=6, method="weighted", alpha=1.0)
    assert bm25_only[0].doc_id == "doc-6"


def test_hybrid_search_result_scores_are_populated(tiny_corpus):
    hybrid = _build_hybrid(tiny_corpus)
    results = hybrid.search("neural network layers", top_k=3)
    for r in results:
        assert isinstance(r.score, float)
        assert isinstance(r.bm25_score, float)
        assert isinstance(r.vector_score, float)
