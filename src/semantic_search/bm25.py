"""BM25 lexical retrieval over the document corpus, backed by `rank_bm25`."""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from semantic_search.corpus import Document

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ScoredDoc:
    doc_id: str
    score: float


class BM25Index:
    """A BM25 index over a fixed list of documents, addressable by doc id."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.doc_ids = [d.id for d in documents]
        self._id_to_pos = {d.id: i for i, d in enumerate(documents)}
        self._tokenized = [tokenize(d.full_text) for d in documents]
        self._bm25 = BM25Okapi(self._tokenized)

    def __len__(self) -> int:
        return len(self.documents)

    def search(self, query: str, top_k: int = 20) -> list[ScoredDoc]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked_idx = scores.argsort()[::-1][:top_k]
        return [ScoredDoc(doc_id=self.doc_ids[i], score=float(scores[i])) for i in ranked_idx if scores[i] > 0]

    def score_all(self, query: str) -> dict[str, float]:
        """Returns BM25 score for every document (0.0 if no term overlap)."""
        tokens = tokenize(query)
        if not tokens:
            return {doc_id: 0.0 for doc_id in self.doc_ids}
        scores = self._bm25.get_scores(tokens)
        return {self.doc_ids[i]: float(scores[i]) for i in range(len(self.doc_ids))}
