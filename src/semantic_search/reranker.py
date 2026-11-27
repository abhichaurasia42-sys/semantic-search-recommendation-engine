"""Learned re-ranker trained on (synthetic) click-through data.

Features are hand-engineered from TF-IDF + n-gram statistics and the BM25
score, following the "TF-IDF and n-gram features" spec: no sentence
embeddings are used here on purpose, so this stage is a cheap, classic
scikit-learn model that re-scores whatever candidates the (more expensive)
hybrid retriever/BM25 stage already narrowed down.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from semantic_search.bm25 import tokenize
from semantic_search.corpus import Document


@dataclass
class RerankFeatures:
    names = (
        "tfidf_unigram_cosine",
        "tfidf_bigram_cosine",
        "bm25_score_log",
        "jaccard_overlap",
        "query_term_coverage",
        "title_overlap",
        "query_len",
        "doc_len_log",
    )


class RerankFeaturizer:
    """Fits TF-IDF vectorizers over the corpus and turns (query, doc) pairs
    into fixed-length numeric feature vectors for the re-ranker.
    """

    def __init__(self):
        self.unigram_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
        self.bigram_vectorizer = TfidfVectorizer(ngram_range=(2, 2), min_df=1)
        self._doc_unigram = None
        self._doc_bigram = None
        self._doc_ids: list[str] = []
        self._doc_id_to_pos: dict[str, int] = {}
        self._doc_titles: dict[str, str] = {}

    def fit(self, documents: list[Document]) -> "RerankFeaturizer":
        texts = [d.full_text for d in documents]
        self._doc_unigram = self.unigram_vectorizer.fit_transform(texts)
        self._doc_bigram = self.bigram_vectorizer.fit_transform(texts)
        self._doc_ids = [d.id for d in documents]
        self._doc_id_to_pos = {doc_id: i for i, doc_id in enumerate(self._doc_ids)}
        self._doc_titles = {d.id: d.title for d in documents}
        return self

    def transform_one(self, query: str, doc_id: str, doc_text: str, bm25_score: float) -> np.ndarray:
        q_uni = self.unigram_vectorizer.transform([query])
        q_bi = self.bigram_vectorizer.transform([query])
        pos = self._doc_id_to_pos.get(doc_id)
        if pos is not None:
            d_uni = self._doc_unigram[pos]
            d_bi = self._doc_bigram[pos]
        else:
            d_uni = self.unigram_vectorizer.transform([doc_text])
            d_bi = self.bigram_vectorizer.transform([doc_text])

        uni_cos = _sparse_cosine(q_uni, d_uni)
        bi_cos = _sparse_cosine(q_bi, d_bi)

        q_tokens = set(tokenize(query))
        d_tokens = set(tokenize(doc_text))
        jaccard = len(q_tokens & d_tokens) / max(len(q_tokens | d_tokens), 1)
        coverage = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
        title_tokens = set(tokenize(self._doc_titles.get(doc_id, "")))
        title_overlap = len(q_tokens & title_tokens) / max(len(q_tokens), 1)

        return np.array(
            [
                uni_cos,
                bi_cos,
                np.log1p(max(bm25_score, 0.0)),
                jaccard,
                coverage,
                title_overlap,
                len(q_tokens),
                np.log1p(len(d_tokens)),
            ],
            dtype=np.float32,
        )

    def transform_many(self, rows: list[tuple[str, str, str, float]]) -> np.ndarray:
        """rows: list of (query, doc_id, doc_text, bm25_score)."""
        return np.stack([self.transform_one(*row) for row in rows])


def _sparse_cosine(a, b) -> float:
    num = (a.multiply(b)).sum()
    denom = np.sqrt(a.multiply(a).sum()) * np.sqrt(b.multiply(b).sum())
    if denom == 0:
        return 0.0
    return float(num / denom)


class Reranker:
    """Logistic-regression re-ranker predicting P(click | query, doc)."""

    def __init__(self, model: LogisticRegression | None = None):
        self.model = model or LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Reranker":
        self.model.fit(X, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Returns P(relevant/clicked) for each row, usable as a rank score."""
        return self.model.predict_proba(X)[:, 1]

    def save(self, path) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path) -> "Reranker":
        return cls(model=joblib.load(path))
