import pytest
from fastapi.testclient import TestClient

from semantic_search.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["corpus_size"] > 0


def test_search_endpoint_returns_results(client):
    resp = client.get("/search", params={"q": "cloud computing serverless", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "cloud computing serverless"
    assert 0 < len(body["results"]) <= 5
    first = body["results"][0]
    assert set(first.keys()) >= {"doc_id", "title", "snippet", "score"}


def test_search_endpoint_with_rrf_method(client):
    resp = client.get("/search", params={"q": "bread baking oven", "method": "rrf", "top_k": 3})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) <= 3


def test_search_endpoint_rejects_empty_query(client):
    resp = client.get("/search", params={"q": ""})
    assert resp.status_code == 422


def test_search_with_reranker_if_available(client):
    health = client.get("/health").json()
    resp = client.get("/search", params={"q": "machine learning models", "use_reranker": True})
    if health["reranker_loaded"]:
        assert resp.status_code == 200
        assert resp.json()["used_reranker"] is True
    else:
        assert resp.status_code == 400


def test_recommend_endpoint_returns_similar_docs(client):
    search_resp = client.get("/search", params={"q": "cloud computing", "top_k": 1})
    doc_id = search_resp.json()["results"][0]["doc_id"]

    resp = client.get("/recommend", params={"doc_id": doc_id, "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source_doc_id"] == doc_id
    assert len(body["results"]) <= 5
    assert all(r["doc_id"] != doc_id for r in body["results"])


def test_recommend_endpoint_404_for_unknown_doc():
    with TestClient(app) as c:
        resp = c.get("/recommend", params={"doc_id": "does-not-exist"})
        assert resp.status_code == 404
