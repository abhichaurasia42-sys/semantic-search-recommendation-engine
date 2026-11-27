from semantic_search.metrics import ndcg_at_k, recall_at_k


def test_ndcg_perfect_ranking_is_one():
    relevance = {"a": 2, "b": 1, "c": 0}
    ranked = ["a", "b", "c"]
    assert ndcg_at_k(ranked, relevance, k=3) == 1.0


def test_ndcg_worst_ranking_is_less_than_one():
    relevance = {"a": 2, "b": 1, "c": 0}
    ranked = ["c", "b", "a"]
    score = ndcg_at_k(ranked, relevance, k=3)
    assert 0.0 <= score < 1.0


def test_ndcg_no_relevant_docs_is_zero():
    assert ndcg_at_k(["a", "b"], {}, k=10) == 0.0


def test_recall_at_k_counts_fraction_of_relevant_found():
    relevant = {"a", "b", "c", "d"}
    ranked = ["a", "x", "b", "y", "z"]
    # 2 of 4 relevant docs found within top 5
    assert recall_at_k(ranked, relevant, k=5) == 0.5


def test_recall_at_k_full_recall_when_all_relevant_present():
    relevant = {"a", "b"}
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, relevant, k=3) == 1.0


def test_recall_at_k_empty_relevant_set_is_zero():
    assert recall_at_k(["a", "b"], set(), k=5) == 0.0
