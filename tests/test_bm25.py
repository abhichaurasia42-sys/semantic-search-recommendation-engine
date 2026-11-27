from semantic_search.bm25 import BM25Index, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Cloud-Computing, Basics!") == ["cloud", "computing", "basics"]


def test_bm25_ranks_relevant_doc_first(tiny_corpus):
    index = BM25Index(tiny_corpus)
    hits = index.search("serverless functions billed execution", top_k=3)
    assert hits, "expected at least one hit"
    assert hits[0].doc_id == "doc-2"


def test_bm25_returns_empty_for_no_overlap(tiny_corpus):
    index = BM25Index(tiny_corpus)
    hits = index.search("zzzznonexistenttoken", top_k=5)
    assert hits == []


def test_bm25_score_all_covers_every_document(tiny_corpus):
    index = BM25Index(tiny_corpus)
    scores = index.score_all("bread oven yeast")
    assert set(scores.keys()) == {d.id for d in tiny_corpus}
    assert scores["doc-4"] > scores["doc-3"]
