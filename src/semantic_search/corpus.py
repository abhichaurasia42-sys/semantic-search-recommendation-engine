"""Loading/saving the bundled document corpus.

Each document is a JSON object on its own line (JSONL):
    {"id": "tech-cloud-003", "domain": "technology", "subtopic": "cloud-computing",
     "title": "...", "text": "..."}

`domain`/`subtopic` are generation metadata used to build the eval set's graded
relevance judgments honestly (see eval/build_eval_set.py) -- they are NOT used
by the retrieval/ranking code itself, which only ever sees `id`/`title`/`text`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Document:
    id: str
    domain: str
    subtopic: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}"


def write_corpus(docs: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.__dict__, ensure_ascii=False) + "\n")


def iter_corpus(path: Path) -> Iterator[Document]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield Document(**obj)


def load_corpus(path: Path) -> list[Document]:
    return list(iter_corpus(path))
