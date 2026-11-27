"""Central paths and constants used across the package."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
EVAL_DIR = ROOT_DIR / "eval"
MODELS_DIR = ROOT_DIR / "models"

CORPUS_PATH = DATA_DIR / "corpus.jsonl"
CLICK_LOG_PATH = DATA_DIR / "click_logs.jsonl"
EVAL_SET_PATH = EVAL_DIR / "eval_set.jsonl"

RERANKER_MODEL_PATH = MODELS_DIR / "reranker.joblib"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
EMBEDDINGS_CACHE_PATH = MODELS_DIR / "doc_embeddings.npy"
EMBEDDINGS_CACHE_IDS_PATH = MODELS_DIR / "doc_embeddings_ids.json"

# Sentence-transformers model. Small + fast, runs fine on CPU.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Local dev/test fallback store (no infra required).
LOCAL_SQLITE_PATH = DATA_DIR / "vector_store.db"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
