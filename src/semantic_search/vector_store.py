"""Vector store abstraction.

Two implementations share the same `VectorStore` interface:

- `SQLiteVectorStore`: zero-infra local/dev/test backend. Stores embeddings as
  raw float32 blobs in SQLite and does brute-force cosine similarity search
  in-process with numpy. This is what tests, the eval script, and local API
  runs use by default -- no external services required.
- `PgVectorStore`: production backend using PostgreSQL + the `pgvector`
  extension (see docker-compose.yml). Uses `<=>` cosine distance and an
  approximate nearest-neighbor index for real-world scale. Requires
  `psycopg[binary]` and a running Postgres instance; only imported/instantiated
  when actually selected, so the rest of the project has no hard dependency
  on Postgres being available.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


class VectorStore(ABC):
    """Common interface for vector similarity search backends."""

    @abstractmethod
    def upsert(self, ids: Sequence[str], embeddings: np.ndarray, metadatas: Sequence[dict] | None = None) -> None:
        """Insert or replace embeddings for the given document ids."""

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        """Returns [(doc_id, cosine_similarity)] sorted descending by similarity."""

    @abstractmethod
    def count(self) -> int:
        """Number of vectors currently stored."""

    @abstractmethod
    def health_check(self) -> bool:
        """Returns True if the backend is reachable and usable."""


class SQLiteVectorStore(VectorStore):
    """Local, dependency-free vector store for dev/tests.

    Not meant to scale past a few tens of thousands of vectors -- similarity
    search is brute-force numpy matrix multiplication after loading all
    vectors into memory, which is perfectly fine for a corpus of a few
    hundred to a couple thousand documents.
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        # check_same_thread=False + a lock: FastAPI/Starlette runs sync
        # request handlers in a worker thread pool, so a store built once at
        # app startup must remain usable from whichever thread later
        # `search()` calls land on.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                doc_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                metadata TEXT
            )
            """
        )
        self._conn.commit()

    def upsert(self, ids: Sequence[str], embeddings: np.ndarray, metadatas: Sequence[dict] | None = None) -> None:
        if len(ids) != embeddings.shape[0]:
            raise ValueError("ids and embeddings must have the same length")
        metadatas = metadatas or [{} for _ in ids]
        rows = []
        for doc_id, vec, meta in zip(ids, embeddings, metadatas):
            vec = np.asarray(vec, dtype=np.float32)
            rows.append((doc_id, vec.tobytes(), vec.shape[0], json.dumps(meta)))
        with self._lock:
            self._conn.executemany(
                "INSERT INTO vectors (doc_id, embedding, dim, metadata) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(doc_id) DO UPDATE SET embedding=excluded.embedding, dim=excluded.dim, metadata=excluded.metadata",
                rows,
            )
            self._conn.commit()

    def _load_all(self) -> tuple[list[str], np.ndarray]:
        with self._lock:
            cur = self._conn.execute("SELECT doc_id, embedding, dim FROM vectors")
            rows = cur.fetchall()
        ids: list[str] = []
        vecs: list[np.ndarray] = []
        for doc_id, blob, dim in rows:
            ids.append(doc_id)
            vecs.append(np.frombuffer(blob, dtype=np.float32, count=dim))
        if not vecs:
            return [], np.zeros((0, 0), dtype=np.float32)
        return ids, np.stack(vecs)

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        ids, matrix = self._load_all()
        if not ids:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / max(np.linalg.norm(q), 1e-8)
        norms = np.clip(np.linalg.norm(matrix, axis=1), 1e-8, None)
        sims = (matrix @ q) / norms
        order = np.argsort(sims)[::-1][:top_k]
        return [(ids[i], float(sims[i])) for i in order]

    def count(self) -> int:
        with self._lock:
            (n,) = self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
        return int(n)

    def health_check(self) -> bool:
        try:
            with self._lock:
                self._conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def close(self) -> None:
        self._conn.close()


class PgVectorStore(VectorStore):
    """Production vector store backed by PostgreSQL + the pgvector extension.

    See docker-compose.yml for a local pgvector-enabled Postgres you can run
    with `docker compose up -d db`. Connection is lazy: importing this module
    or the package does not require psycopg or a live database -- only
    instantiating `PgVectorStore` does.
    """

    def __init__(self, dsn: str, table: str = "document_embeddings", dim: int = 384):
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PgVectorStore requires the 'psycopg[binary]' and 'pgvector' packages. "
                "Install them with `pip install psycopg[binary] pgvector`."
            ) from exc
        self._psycopg = psycopg
        self.dsn = dsn
        self.table = table
        self.dim = dim
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    doc_id TEXT PRIMARY KEY,
                    embedding vector({self.dim}),
                    metadata JSONB
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx "
                f"ON {self.table} USING hnsw (embedding vector_cosine_ops)"
            )

    def upsert(self, ids: Sequence[str], embeddings: np.ndarray, metadatas: Sequence[dict] | None = None) -> None:
        metadatas = metadatas or [{} for _ in ids]
        with self._conn.cursor() as cur:
            for doc_id, vec, meta in zip(ids, embeddings, metadatas):
                vec_literal = "[" + ",".join(f"{x:.8f}" for x in vec.tolist()) + "]"
                cur.execute(
                    f"""
                    INSERT INTO {self.table} (doc_id, embedding, metadata)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE
                        SET embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata
                    """,
                    (doc_id, vec_literal, json.dumps(meta)),
                )

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        vec_literal = "[" + ",".join(f"{x:.8f}" for x in query_embedding.tolist()) + "]"
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT doc_id, 1 - (embedding <=> %s) AS cosine_sim
                FROM {self.table}
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (vec_literal, vec_literal, top_k),
            )
            return [(row[0], float(row[1])) for row in cur.fetchall()]

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            (n,) = cur.fetchone()
            return int(n)

    def health_check(self) -> bool:
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False


def build_default_store(db_path: str | Path = ":memory:") -> VectorStore:
    """Convenience factory: the zero-infra local store used by tests/dev/eval."""
    return SQLiteVectorStore(db_path)
