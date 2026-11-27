#!/usr/bin/env python
"""Builds the labeled eval set: query -> graded-relevance document judgments.

Relevance grades come honestly from the corpus's own generation metadata
(domain/subtopic), not from hand-picking or the numbers we want to see:
    - relevance 2: document belongs to the exact subtopic the query is about.
      ALL documents in that subtopic are labeled (not a sample) -- by
      construction every document in a subtopic is genuinely on-topic for a
      subtopic-level query, so under-labeling them would unfairly penalize a
      retriever for correctly surfacing an on-topic document that happened
      not to be in an arbitrary sample.
    - relevance 1: document belongs to a *different* subtopic in the *same*
      domain (topically adjacent, partially relevant). Sampled (bounded) to
      keep the eval set at a reasonable, realistic scale.
    - relevance 0 (implicit): everything else -- not stored, treated as
      non-relevant by the eval metrics

Query text is generated from a fixed set of natural-language templates
applied to each subtopic's title. A DIFFERENT, disjoint set of templates is
used by scripts/generate_click_data.py for the re-ranker's training data, so
the eval queries are not literally reused as training signal.

Run:
    python eval/build_eval_set.py
Writes eval/eval_set.jsonl, one JSON object per query:
    {"qid": "...", "query": "...", "domain": "...", "subtopic": "...",
     "judgments": [{"doc_id": "...", "relevance": 2}, ...]}
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_search.config import CORPUS_PATH, EVAL_SET_PATH  # noqa: E402
from semantic_search.corpus import load_corpus  # noqa: E402

RANDOM_SEED = 123

# Templates reserved for EVAL queries only (see generate_click_data.py for the
# disjoint set used to build training click logs).
EVAL_QUERY_TEMPLATES = [
    "What is {title} and why does it matter?",
]

N_SAME_DOMAIN_PARTIAL = 2


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    docs = load_corpus(CORPUS_PATH)

    by_subtopic: dict[tuple[str, str], list] = {}
    by_domain: dict[str, list] = {}
    subtopic_title: dict[tuple[str, str], str] = {}
    for d in docs:
        key = (d.domain, d.subtopic)
        by_subtopic.setdefault(key, []).append(d)
        by_domain.setdefault(d.domain, []).append(d)
        if key not in subtopic_title:
            subtopic_title[key] = d.title.rsplit(" Notes #", 1)[0]

    eval_rows = []
    qid = 0
    for (domain, subtopic), docs_in_subtopic in sorted(by_subtopic.items()):
        title = subtopic_title[(domain, subtopic)]
        for template in EVAL_QUERY_TEMPLATES:
            query = template.format(title=title)
            qid += 1

            # Every document in this subtopic is genuinely relevant to a
            # subtopic-level query -- label all of them, not a sample.
            relevant_sample = docs_in_subtopic

            other_domain_docs = [d for d in by_domain[domain] if d.subtopic != subtopic]
            rng.shuffle(other_domain_docs)
            partial_sample = other_domain_docs[:N_SAME_DOMAIN_PARTIAL]

            judgments = [{"doc_id": d.id, "relevance": 2} for d in relevant_sample]
            judgments += [{"doc_id": d.id, "relevance": 1} for d in partial_sample]

            eval_rows.append(
                {
                    "qid": f"q{qid:04d}",
                    "query": query,
                    "domain": domain,
                    "subtopic": subtopic,
                    "judgments": judgments,
                }
            )

    EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_SET_PATH.open("w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_pairs = sum(len(r["judgments"]) for r in eval_rows)
    print(f"Wrote {len(eval_rows)} queries / {n_pairs} query-document judgment pairs to {EVAL_SET_PATH}")


if __name__ == "__main__":
    main()
