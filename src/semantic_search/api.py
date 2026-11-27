"""FastAPI app exposing /search, /recommend, /health.

On startup, builds the BM25 index and vector store from the bundled corpus
(data/corpus.jsonl) using the local, zero-infra SQLite vector store by
default. Set `SEMANTIC_SEARCH_PG_DSN` to point `/health` and retrieval at a
real pgvector-backed Postgres instead (see vector_store.PgVectorStore and
docker-compose.yml).
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from semantic_search.bm25 import BM25Index
from semantic_search.config import CORPUS_PATH, RERANKER_MODEL_PATH, TFIDF_VECTORIZER_PATH
from semantic_search.corpus import load_corpus
from semantic_search.embeddings import Embedder, cosine_similarity_matrix
from semantic_search.hybrid_retriever import HybridRetriever
from semantic_search.reranker import Reranker
from semantic_search.vector_store import PgVectorStore, SQLiteVectorStore, VectorStore

logger = logging.getLogger(__name__)


class AppState:
    def __init__(self):
        self.documents = []
        self.doc_by_id = {}
        self.bm25: BM25Index | None = None
        self.embedder: Embedder | None = None
        self.vector_store: VectorStore | None = None
        self.hybrid: HybridRetriever | None = None
        self.reranker: Reranker | None = None
        self.featurizer = None
        self.doc_embeddings = None

    def build(self) -> None:
        self.documents = load_corpus(CORPUS_PATH)
        self.doc_by_id = {d.id: d for d in self.documents}
        self.bm25 = BM25Index(self.documents)
        self.embedder = Embedder()

        self.doc_embeddings = self.embedder.encode([d.full_text for d in self.documents])

        pg_dsn = os.environ.get("SEMANTIC_SEARCH_PG_DSN")
        if pg_dsn:
            logger.info("Using PgVectorStore backend")
            self.vector_store = PgVectorStore(pg_dsn, dim=self.embedder.dimension)
        else:
            logger.info("Using local SQLiteVectorStore backend (no infra required)")
            self.vector_store = SQLiteVectorStore(":memory:")
        self.vector_store.upsert([d.id for d in self.documents], self.doc_embeddings)

        self.hybrid = HybridRetriever(self.documents, self.bm25, self.vector_store, self.embedder)

        if RERANKER_MODEL_PATH.exists() and TFIDF_VECTORIZER_PATH.exists():
            self.reranker = Reranker.load(RERANKER_MODEL_PATH)
            self.featurizer = joblib.load(TFIDF_VECTORIZER_PATH)
            logger.info("Loaded trained re-ranker")
        else:
            logger.warning("No trained re-ranker found; /search?use_reranker=true will 400")


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.build()
    yield


app = FastAPI(
    title="Semantic Search & Recommendation Engine",
    description="Hybrid BM25 + vector search with a scikit-learn re-ranker.",
    version="0.1.0",
    lifespan=lifespan,
)


class SearchResultItem(BaseModel):
    doc_id: str
    title: str
    snippet: str
    score: float
    bm25_score: float
    vector_score: float


class SearchResponse(BaseModel):
    query: str
    used_reranker: bool
    results: list[SearchResultItem]


class RecommendResultItem(BaseModel):
    doc_id: str
    title: str
    snippet: str
    similarity: float


class RecommendResponse(BaseModel):
    source_doc_id: str
    results: list[RecommendResultItem]


class HealthResponse(BaseModel):
    status: str
    corpus_size: int
    embedding_backend: str
    reranker_loaded: bool
    vector_store: str


def _snippet(text: str, max_len: int = 160) -> str:
    return text if len(text) <= max_len else text[:max_len].rsplit(" ", 1)[0] + "..."


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ok = state.bm25 is not None and state.vector_store is not None and state.vector_store.health_check()
    return HealthResponse(
        status="ok" if ok else "degraded",
        corpus_size=len(state.documents),
        embedding_backend="fallback-hashed-bow" if (state.embedder and state.embedder.using_fallback) else "sentence-transformers",
        reranker_loaded=state.reranker is not None,
        vector_store=type(state.vector_store).__name__ if state.vector_store else "none",
    )


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Search query text"),
    top_k: int = Query(10, ge=1, le=100),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="BM25 weight in weighted fusion"),
    method: str = Query("weighted", pattern="^(weighted|rrf)$"),
    use_reranker: bool = Query(False, description="Re-rank the fused candidates with the trained scikit-learn model"),
) -> SearchResponse:
    if state.hybrid is None:
        raise HTTPException(status_code=503, detail="Index not built yet")

    candidate_pool = max(top_k * 5, 50)
    fused = state.hybrid.search(q, top_k=candidate_pool, candidate_pool=candidate_pool, alpha=alpha, method=method)

    used_reranker = False
    if use_reranker:
        if state.reranker is None or state.featurizer is None:
            raise HTTPException(status_code=400, detail="No trained re-ranker available. Run scripts/train_reranker.py first.")
        rows = [
            (q, r.doc_id, state.doc_by_id[r.doc_id].full_text, r.bm25_score)
            for r in fused
        ]
        if rows:
            X = state.featurizer.transform_many(rows)
            scores = state.reranker.score(X)
            fused = [
                r for r, _ in sorted(zip(fused, scores), key=lambda t: t[1], reverse=True)
            ]
        used_reranker = True

    top = fused[:top_k]
    results = [
        SearchResultItem(
            doc_id=r.doc_id,
            title=state.doc_by_id[r.doc_id].title,
            snippet=_snippet(state.doc_by_id[r.doc_id].text),
            score=r.score,
            bm25_score=r.bm25_score,
            vector_score=r.vector_score,
        )
        for r in top
    ]
    return SearchResponse(query=q, used_reranker=used_reranker, results=results)


@app.get("/recommend", response_model=RecommendResponse)
def recommend(doc_id: str = Query(...), top_k: int = Query(10, ge=1, le=100)) -> RecommendResponse:
    """Content-based recommendation: nearest neighbors of `doc_id` in embedding space."""
    if state.embedder is None or state.doc_embeddings is None:
        raise HTTPException(status_code=503, detail="Index not built yet")
    if doc_id not in state.doc_by_id:
        raise HTTPException(status_code=404, detail=f"Unknown doc_id: {doc_id}")

    doc_ids = [d.id for d in state.documents]
    pos = doc_ids.index(doc_id)
    query_vec = state.doc_embeddings[pos : pos + 1]
    sims = cosine_similarity_matrix(query_vec, state.doc_embeddings)[0]

    ranked = sorted(
        ((doc_ids[i], float(sims[i])) for i in range(len(doc_ids)) if doc_ids[i] != doc_id),
        key=lambda t: t[1],
        reverse=True,
    )[:top_k]

    results = [
        RecommendResultItem(
            doc_id=d_id,
            title=state.doc_by_id[d_id].title,
            snippet=_snippet(state.doc_by_id[d_id].text),
            similarity=sim,
        )
        for d_id, sim in ranked
    ]
    return RecommendResponse(source_doc_id=doc_id, results=results)
