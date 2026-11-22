"""Ranking metrics for top-k recommendation."""

from __future__ import annotations

import numpy as np


def precision_at_k(recs: np.ndarray, relevant: set, k: int) -> float:
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs: np.ndarray, relevant: set, k: int) -> float:
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recs: np.ndarray, relevant: set, k: int) -> float:
    dcg = sum(1 / np.log2(i + 2) for i, it in enumerate(recs[:k]) if it in relevant)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def evaluate(model, test: np.ndarray, k: int = 10) -> dict:
    users = np.flatnonzero(test.sum(axis=1) > 0)
    recs = model.recommend(users, k)
    p, r, n = [], [], []
    for row, u in zip(recs, users):
        rel = set(np.flatnonzero(test[u]))
        p.append(precision_at_k(row, rel, k))
        r.append(recall_at_k(row, rel, k))
        n.append(ndcg_at_k(row, rel, k))
    cov = len(np.unique(recs)) / test.shape[1]
    return {f"precision@{k}": float(np.mean(p)), f"recall@{k}": float(np.mean(r)),
            f"ndcg@{k}": float(np.mean(n)), "catalog_coverage": float(cov)}


def tune_als(train: np.ndarray, grid: dict, k: int = 10, holdout: int = 3, seed: int = 1):
    """Grid-search ImplicitALS on a validation split carved out of ``train``.

    The test set is never used for tuning, so reported test metrics are unbiased.
    Returns (best_params, best_val_ndcg).
    """
    from itertools import product

    from .data import Interactions, leave_n_out
    from .models import ImplicitALS

    inner_train, val = leave_n_out(Interactions(train, [], []), n=holdout, seed=seed)
    best, best_score = None, -1.0
    keys = list(grid)
    for values in product(*(grid[key] for key in keys)):
        params = dict(zip(keys, values))
        score = evaluate(ImplicitALS(**params).fit(inner_train), val, k)[f"ndcg@{k}"]
        if score > best_score:
            best, best_score = params, score
    return best, best_score
