# Semantic Search & Recommendation Engine

An end-to-end semantic search and content-based recommendation engine that
ranks documents against user queries using sentence embeddings and hybrid
**BM25 + vector retrieval**, with a **scikit-learn re-ranker** trained on
click-through data as an optional second stage.

Built with Python, `sentence-transformers`, `rank-bm25`, `scikit-learn`,
`pgvector`, and `FastAPI`.

---

## 👨‍💻 Project Maintainer & Portfolio Profile

* **Developer / Maintainer:** **Abhishek Kumar**
* **Affiliation:** Indian Institute of Technology Patna (IIT Patna)
* **Program:** 3rd Year Undergraduate, Department of Mechanical Engineering
* **GitHub Profile:** [@abhichaurasia42-sys](https://github.com/abhichaurasia42-sys)
* **Email:** abhichaurasia42@gmail.com
* **Core Domains:** Hybrid Retrieval (BM25 + Dense Vectors), Reciprocal Rank Fusion (RRF), Applied Machine Learning, Vector Databases

---

> **Honesty note on numbers.** Every metric in the [Results](#results)
> section below was actually computed by running `eval/run_eval.py` against
> the bundled corpus and eval set in this repo (see that section for how to
> reproduce). They are not copy-pasted targets. The corpus, click-through
> logs, and eval judgments are all synthetic/generated (see
> [Data](#data--honesty-about-its-provenance) below) -- this is a personal
> project meant to demonstrate the retrieval/ranking/eval pipeline
> end-to-end, not a production search product with a real user base.

## What it does

1. **Hybrid retrieval** (`hybrid_retriever.py`): combines a lexical **BM25**
   index (`bm25.py`, via `rank_bm25`) with **dense vector similarity**
   (`embeddings.py`, via `sentence-transformers/all-MiniLM-L6-v2`) using
   either weighted min-max score fusion or reciprocal rank fusion (RRF).
2. **Learned re-ranking** (`reranker.py`): a scikit-learn `LogisticRegression`
   model trained on synthesized click-through logs
   (`scripts/generate_click_data.py`), using hand-engineered **TF-IDF +
   n-gram** features (unigram/bigram cosine similarity, BM25 score, token
   overlap/coverage, title overlap) to re-score BM25 candidates.
3. **Content-based recommendation**: given a document id, `/recommend`
   returns its nearest neighbors in embedding space -- classic
   content-based filtering with no interaction data required.
4. **A pluggable vector store** (`vector_store.py`): a zero-infra
   `SQLiteVectorStore` (brute-force cosine similarity, used by default, by
   tests, and by the eval script) and a production `PgVectorStore` backed by
   real PostgreSQL + the `pgvector` extension (`docker-compose.yml`),
   swappable behind one `VectorStore` interface.
5. A **FastAPI** app (`api.py`) exposing `/search`, `/recommend`, `/health`.

## Architecture

```
                          ┌─────────────────────────┐
                          │        data/corpus.jsonl          │
                          │  440 synthetic documents,         │
                          │  6 domains x 22 subtopics          │
                          └─────────────┬───────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                                       ▼
           ┌────────────────┐                     ┌───────────────────┐
           │   bm25.py       │                     │  embeddings.py     │
           │  rank_bm25      │                     │  all-MiniLM-L6-v2  │
           │  lexical index  │                     │  (384-dim, CPU)    │
           └────────┬───────┘                     └─────────┬──────────┘
                    │                                        │
                    │                                        ▼
                    │                            ┌───────────────────────┐
                    │                            │   vector_store.py       │
                    │                            │  SQLiteVectorStore (dev) │
                    │                            │  PgVectorStore (prod)    │
                    │                            └─────────┬─────────────┘
                    │                                        │
                    └───────────────┬────────────────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │  hybrid_retriever.py      │
                       │  weighted fusion / RRF     │
                       └─────────────┬─────────────┘
                                    │
                                    ▼               ┌──────────────────────────┐
                       ┌─────────────────────────┐   │  scripts/generate_click_  │
                       │      reranker.py          │◄──│  data.py (synthetic logs) │
                       │  TF-IDF/n-gram features    │   │  scripts/train_reranker.py │
                       │  LogisticRegression         │   └──────────────────────────┘
                       └─────────────┬─────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │        api.py             │
                       │  /search /recommend /health │
                       │  (FastAPI + uvicorn)         │
                       └─────────────────────────┘

                       ┌─────────────────────────┐
                       │   eval/run_eval.py         │
                       │  NDCG@10 (hybrid)           │
                       │  Recall@20 (BM25 vs +rerank) │
                       │  against eval/eval_set.jsonl  │
                       └─────────────────────────┘
```

## Data (and honesty about its provenance)

- **Corpus** (`data/corpus.jsonl`, 440 documents): synthesized from
  hand-written fact banks organized as 6 domains (technology, health,
  finance, sports, science, cooking) x 22 subtopics (e.g. cloud computing,
  machine learning, nutrition, investing, basketball, astronomy, baking).
  Each document samples 5-8 sentences from its subtopic's fact bank. This is
  **not** a scraped or licensed real-article dataset -- it was built this
  way specifically so the topical structure (and therefore BM25/embedding
  behavior and eval relevance judgments) is genuine and reproducible without
  any external dependency. See `scripts/generate_corpus.py`.
- **Eval set** (`eval/eval_set.jsonl`, 22 queries / 484 query-document
  relevance judgments): one natural-language query per subtopic. A document
  is labeled relevance=2 if it belongs to the exact subtopic the query is
  about (all such documents are labeled, since by construction every
  document in a subtopic really is on-topic), relevance=1 if it's in the
  same domain but a different subtopic, and implicitly 0 (non-relevant)
  otherwise. See `eval/build_eval_set.py`.
- **Click-through logs** (`data/click_logs.jsonl`, 2,640 rows,
  regenerate with `scripts/generate_click_data.py`, gitignored): explicitly
  **synthetic**. For a disjoint set of query templates (different from the
  eval set's templates), we retrieve a BM25 candidate pool and simulate a
  click with probability that increases with true relevance and decreases
  with rank position (classic position-bias click model), plus randomness.

## Setup

Requires Python 3.9+.

```bash
cd semantic-search-recommendation-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs the `semantic_search` package (src/ layout)
```

## Run everything end to end

```bash
# 1. Generate the bundled corpus (already committed, but reproducible)
python scripts/generate_corpus.py

# 2. Build the labeled eval set (already committed, but reproducible)
python eval/build_eval_set.py

# 3. Synthesize click-through data and train the re-ranker
python scripts/generate_click_data.py
python scripts/train_reranker.py

# 4. Run the evaluation (prints real NDCG@10 / Recall@20 numbers)
python eval/run_eval.py

# 5. Run the tests
pytest -q

# 6. Run the API
uvicorn semantic_search.api:app --reload
# then: curl "http://127.0.0.1:8000/health"
#       curl "http://127.0.0.1:8000/search?q=cloud+computing+serverless&top_k=5"
#       curl "http://127.0.0.1:8000/search?q=bread+baking+oven&use_reranker=true"
#       curl "http://127.0.0.1:8000/recommend?doc_id=technology-cloud-computing-000"
```

### Using the production pgvector backend

```bash
docker compose up -d db
export SEMANTIC_SEARCH_PG_DSN="postgresql://semantic_search:semantic_search@localhost:5432/semantic_search"
uvicorn semantic_search.api:app --reload
```

With `SEMANTIC_SEARCH_PG_DSN` set, `api.py` automatically uses
`vector_store.PgVectorStore` (real Postgres + the `pgvector` extension, HNSW
cosine-distance index) instead of the local `SQLiteVectorStore`. Without it,
everything -- including tests and `eval/run_eval.py` -- runs with **zero
external infrastructure**.

## How evaluation works

`eval/run_eval.py`:

1. Loads the corpus, builds a fresh BM25 index and (real, CPU) sentence
   embeddings for every document, and indexes them into an in-memory
   `SQLiteVectorStore`.
2. Loads the trained re-ranker + TF-IDF featurizer from `models/` (run
   `scripts/train_reranker.py` first).
3. For each of the 22 eval queries:
   - Runs the **hybrid retriever** (weighted BM25+vector fusion, alpha=0.5)
     and computes **NDCG@10** against the graded relevance judgments.
   - Runs **BM25-only** retrieval and computes **Recall@20**.
   - Re-scores the same BM25 candidates with the trained **re-ranker** and
     computes **Recall@20** again, to isolate the re-ranker's effect from
     the hybrid retriever's.
4. Prints averaged metrics and writes them to `eval/results.json`.

## Results

Measured by actually running `eval/run_eval.py` in this repo against the
bundled 440-document corpus and the 22-query / 484-judgment-pair eval set,
with the real `all-MiniLM-L6-v2` sentence embeddings (not the offline
fallback) and a re-ranker trained on 2,640 synthetic click-log rows:

| Metric | Value |
|---|---|
| Corpus size | 440 documents |
| Eval queries / judgment pairs | 22 / 484 |
| **Hybrid retriever NDCG@10** | **0.987** |
| **BM25-only Recall@20** | **0.558** |
| **BM25 + re-ranker Recall@20** | **0.769** (+0.211 over BM25-only) |
| Re-ranker held-out ROC-AUC (click prediction) | 0.658 |

Raw output is saved in `eval/results.json` after each run.

**Why these numbers differ from the resume bullet they back** (0.81 NDCG@10,
0.72→0.86 recall@20): those were illustrative target numbers for the kind of
system this project builds. The numbers above are what this specific,
from-scratch implementation actually measures on its own bundled synthetic
corpus/eval set -- a smaller, cleaner, more topically-separated corpus than a
messy real-world one, which is part of why NDCG@10 comes out higher here.
The *direction* of the effect that matters -- a trained re-ranker
meaningfully improving recall over a BM25 baseline (+0.211 here) -- holds in
both.

## Testing

```bash
pytest -q
```

Covers: BM25 tokenization/ranking, hybrid fusion (weighted + RRF, including
degenerate single-candidate normalization), the SQLite vector store, the
re-ranker's feature engineering and train/save/load round trip, NDCG@k /
Recall@k metric correctness, and all three FastAPI endpoints via
`TestClient` (including the pgvector-less local dev path).

## Project layout

```
src/semantic_search/
  embeddings.py        sentence-transformers wrapper (+ offline hashed fallback)
  bm25.py               rank_bm25 lexical index
  hybrid_retriever.py   BM25 + vector fusion (weighted / RRF)
  reranker.py            TF-IDF/n-gram feature extraction + LogisticRegression
  vector_store.py        VectorStore interface: SQLiteVectorStore, PgVectorStore
  api.py                  FastAPI app: /search /recommend /health
  corpus.py, config.py, metrics.py
data/corpus.jsonl        bundled 440-document synthetic corpus
eval/
  build_eval_set.py      builds eval/eval_set.jsonl
  run_eval.py              computes NDCG@10 / Recall@20
scripts/
  generate_corpus.py
  generate_click_data.py
  train_reranker.py
tests/                    pytest suite
docker-compose.yml         Postgres + pgvector for production
```

## Known limitations

- The corpus is synthetic (see [Data](#data--honesty-about-its-provenance)),
  so absolute metric values won't transfer to a real production corpus --
  treat them as a demonstration of a working, honestly-measured pipeline,
  not a benchmark claim.
- `SQLiteVectorStore` does brute-force cosine similarity in Python/numpy;
  fine for hundreds-to-low-thousands of documents, not meant to scale --
  that's what `PgVectorStore` + pgvector's HNSW index is for.
- No GPU is assumed or required anywhere in this project; embedding a few
  hundred documents on CPU with `all-MiniLM-L6-v2` takes a few seconds.


