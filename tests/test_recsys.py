import numpy as np

from recsys.data import leave_n_out, synthetic
from recsys.metrics import evaluate, ndcg_at_k, precision_at_k, recall_at_k
from recsys.models import ImplicitALS, ItemKNN, Popularity


def test_metrics_known_values():
    recs = np.array([3, 1, 7])
    rel = {1, 9}
    assert precision_at_k(recs, rel, 3) == 1 / 3
    assert recall_at_k(recs, rel, 3) == 0.5
    expected = (1 / np.log2(3)) / (1 + 1 / np.log2(3))
    assert np.isclose(ndcg_at_k(recs, rel, 3), expected)
    assert ndcg_at_k(np.array([1, 9]), rel, 2) == 1.0


def test_split_moves_interactions_without_overlap():
    data = synthetic(n_users=50, n_items=80, seed=3)
    train, test = leave_n_out(data, n=5)
    assert ((train > 0) & (test > 0)).sum() == 0
    assert (train + test == data.matrix).all()
    assert ((test > 0).sum(axis=1) <= 5).all()


def test_recommend_excludes_seen_items():
    data = synthetic(n_users=40, n_items=60, seed=1)
    train, _ = leave_n_out(data)
    m = Popularity().fit(train)
    recs = m.recommend(np.arange(40), k=5)
    for u, row in enumerate(recs):
        assert not (train[u, row] > 0).any()


def test_als_loss_decreases():
    data = synthetic(n_users=80, n_items=60, seed=2)
    m = ImplicitALS(factors=8, iterations=6).fit(data.matrix)
    assert m.loss_history[-1] < m.loss_history[0]


def test_personalised_models_beat_popularity():
    train, test = leave_n_out(synthetic(seed=0))
    pop = evaluate(Popularity().fit(train), test)["ndcg@10"]
    knn = evaluate(ItemKNN().fit(train), test)["ndcg@10"]
    als = evaluate(ImplicitALS(factors=32, reg=10, alpha=1, iterations=10).fit(train), test)["ndcg@10"]
    assert knn > 2 * pop
    assert als > 2 * pop


def test_similar_items_same_genre_structure():
    data = synthetic(n_users=300, n_items=100, seed=4)
    m = ImplicitALS(factors=16, iterations=8).fit(data.matrix)
    sims = m.similar_items(0, k=5)
    assert 0 not in sims and len(sims) == 5
