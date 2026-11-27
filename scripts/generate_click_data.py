#!/usr/bin/env python
"""Synthesizes click-through training data for the scikit-learn re-ranker.

This is explicitly SYNTHETIC data (labeled as such in the README): we do not
have a real search engine's query logs, so we simulate one. For each
generated query we:
  1. Retrieve a candidate pool with the plain BM25 index (mimicking what a
     first-pass retriever would actually surface to a user).
  2. Assign each candidate a "true" relevance grade from corpus metadata
     (same rule as eval/build_eval_set.py: 2 = same subtopic, 1 = same
     domain, 0 = otherwise) -- but note the QUERY TEMPLATES here are
     disjoint from eval/build_eval_set.py's templates, so we are not just
     reusing eval queries as training data.
  3. Simulate a click with probability that increases with relevance and
     decreases with rank position (position bias), plus random noise, the
     way real click-through logs behave.

Run:
    python scripts/generate_click_data.py
Writes data/click_logs.jsonl: {"query": ..., "doc_id": ..., "rank": ...,
"bm25_score": ..., "clicked": 0/1}
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_search.bm25 import BM25Index  # noqa: E402
from semantic_search.config import CLICK_LOG_PATH, CORPUS_PATH  # noqa: E402
from semantic_search.corpus import load_corpus  # noqa: E402

RANDOM_SEED = 7
CANDIDATE_POOL = 30

# Disjoint from eval/build_eval_set.py's EVAL_QUERY_TEMPLATES on purpose.
CLICK_QUERY_TEMPLATES = [
    "how does {title} work",
    "best practices for {title}",
    "{title} tips for beginners",
    "guide to {title}",
]

# Base click probability by true relevance grade, before position decay.
BASE_CLICK_PROB = {2: 0.85, 1: 0.35, 0: 0.05}


def position_decay(rank: int) -> float:
    # Classic 1/log2(rank+1) style position bias, rank is 1-indexed.
    import math

    return 1.0 / math.log2(rank + 1.5)


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    docs = load_corpus(CORPUS_PATH)
    bm25 = BM25Index(docs)

    by_subtopic: dict[tuple[str, str], list] = {}
    subtopic_title: dict[tuple[str, str], str] = {}
    for d in docs:
        key = (d.domain, d.subtopic)
        by_subtopic.setdefault(key, []).append(d)
        if key not in subtopic_title:
            subtopic_title[key] = d.title.rsplit(" Notes #", 1)[0]

    rows = []
    for (domain, subtopic) in sorted(by_subtopic):
        title = subtopic_title[(domain, subtopic)]
        for template in CLICK_QUERY_TEMPLATES:
            query = template.format(title=title)
            hits = bm25.search(query, top_k=CANDIDATE_POOL)
            if not hits:
                continue
            for rank, hit in enumerate(hits, start=1):
                doc = next(d for d in docs if d.id == hit.doc_id)
                if doc.subtopic == subtopic and doc.domain == domain:
                    true_relevance = 2
                elif doc.domain == domain:
                    true_relevance = 1
                else:
                    true_relevance = 0
                prob = BASE_CLICK_PROB[true_relevance] * position_decay(rank)
                prob = min(max(prob, 0.0), 0.98)
                clicked = 1 if rng.random() < prob else 0
                rows.append(
                    {
                        "query": query,
                        "doc_id": hit.doc_id,
                        "rank": rank,
                        "bm25_score": hit.score,
                        "true_relevance": true_relevance,
                        "clicked": clicked,
                    }
                )

    CLICK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLICK_LOG_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_clicked = sum(r["clicked"] for r in rows)
    print(f"Wrote {len(rows)} click-log rows ({n_clicked} clicks, {len(rows) - n_clicked} non-clicks) to {CLICK_LOG_PATH}")


if __name__ == "__main__":
    main()
