import numpy as np

from semantic_search.reranker import Reranker, RerankFeaturizer


def test_featurizer_produces_fixed_length_vector(tiny_corpus):
    featurizer = RerankFeaturizer().fit(tiny_corpus)
    vec = featurizer.transform_one(
        query="cloud computing storage",
        doc_id="doc-1",
        doc_text=tiny_corpus[0].full_text,
        bm25_score=3.2,
    )
    assert vec.shape == (8,)
    assert np.isfinite(vec).all()


def test_featurizer_relevant_pair_scores_higher_than_irrelevant(tiny_corpus):
    featurizer = RerankFeaturizer().fit(tiny_corpus)
    relevant_vec = featurizer.transform_one(
        "cloud computing storage internet", "doc-1", tiny_corpus[0].full_text, bm25_score=4.0
    )
    irrelevant_vec = featurizer.transform_one(
        "cloud computing storage internet", "doc-6", tiny_corpus[5].full_text, bm25_score=0.0
    )
    # unigram tfidf cosine (feature 0) should favor the on-topic document
    assert relevant_vec[0] > irrelevant_vec[0]


def test_reranker_trains_and_predicts_reasonable_scores():
    rng = np.random.RandomState(0)
    # Two well-separated synthetic clusters: "relevant" pairs have high
    # feature values, "irrelevant" pairs have low ones.
    X_pos = rng.normal(loc=3.0, scale=0.3, size=(30, 8))
    X_neg = rng.normal(loc=-3.0, scale=0.3, size=(30, 8))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 30 + [0] * 30)

    reranker = Reranker()
    reranker.fit(X, y)
    scores = reranker.score(X)

    assert scores.shape == (60,)
    assert scores[:30].mean() > scores[30:].mean()


def test_reranker_save_and_load_round_trip(tmp_path):
    rng = np.random.RandomState(1)
    X = rng.normal(size=(20, 8))
    y = (X[:, 0] > 0).astype(int)
    reranker = Reranker().fit(X, y)

    path = tmp_path / "reranker.joblib"
    reranker.save(path)
    loaded = Reranker.load(path)

    np.testing.assert_allclose(reranker.score(X), loaded.score(X))
