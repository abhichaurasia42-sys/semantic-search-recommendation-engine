import numpy as np

from semantic_search.vector_store import SQLiteVectorStore


def test_upsert_and_search_returns_most_similar_first():
    store = SQLiteVectorStore(":memory:")
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    store.upsert(["a", "b", "c"], embeddings)

    results = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=3)
    ids = [doc_id for doc_id, _ in results]
    assert ids[0] == "a"
    assert ids[1] == "b"
    assert store.count() == 3


def test_upsert_is_idempotent_on_same_id():
    store = SQLiteVectorStore(":memory:")
    store.upsert(["a"], np.array([[1.0, 0.0]], dtype=np.float32))
    store.upsert(["a"], np.array([[0.0, 1.0]], dtype=np.float32))
    assert store.count() == 1
    results = store.search(np.array([0.0, 1.0], dtype=np.float32), top_k=1)
    assert results[0][0] == "a"
    assert results[0][1] > 0.99


def test_health_check_true_for_open_connection():
    store = SQLiteVectorStore(":memory:")
    assert store.health_check() is True
