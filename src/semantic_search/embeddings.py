"""Sentence embedding backend.

Uses `sentence-transformers` with a small, fast CPU-friendly model
(all-MiniLM-L6-v2, 384-dim). The model is downloaded once and cached by
huggingface_hub under ~/.cache; it is NOT committed to this repo (see
.gitignore) since it is a large binary artifact that can always be
re-downloaded or re-fetched from the pip package cache.

If the model cannot be loaded (e.g. no network on first run in a fully
offline environment), `Embedder` falls back to a deterministic hashed
bag-of-words embedding so the rest of the pipeline still runs end to end.
This fallback is clearly logged -- it is NOT a silent substitute, and the
README documents that real evaluation numbers assume the real model loads.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Sequence

import numpy as np

from semantic_search.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

_FALLBACK_DIM = 384


class Embedder:
    """Thin wrapper around a sentence-transformers model with a safe fallback."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._using_fallback = False
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._using_fallback = False
            logger.info("Loaded sentence-transformers model %s", self.model_name)
        except Exception as exc:  # pragma: no cover - exercised only when offline
            logger.warning(
                "Falling back to hashed bag-of-words embeddings because "
                "sentence-transformers model failed to load: %s",
                exc,
            )
            self._model = None
            self._using_fallback = True

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    @property
    def dimension(self) -> int:
        if self._model is not None:
            return int(self._model.get_sentence_embedding_dimension())
        return _FALLBACK_DIM

    def encode(self, texts: Sequence[str], batch_size: int = 64, show_progress_bar: bool = False) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)
        return np.stack([self._hashed_embedding(t) for t in texts])

    def _hashed_embedding(self, text: str) -> np.ndarray:
        """Deterministic hashed bag-of-words vector, L2-normalized."""
        vec = np.zeros(_FALLBACK_DIM, dtype=np.float32)
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % _FALLBACK_DIM] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


def cosine_similarity_matrix(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity between each query vector and each doc vector.

    Assumes vectors may not be normalized; normalizes defensively.
    """
    q = query_vecs / np.clip(np.linalg.norm(query_vecs, axis=1, keepdims=True), 1e-8, None)
    d = doc_vecs / np.clip(np.linalg.norm(doc_vecs, axis=1, keepdims=True), 1e-8, None)
    return q @ d.T
