"""Recommenders: popularity baseline, item-kNN, and implicit-feedback ALS."""

from __future__ import annotations

import numpy as np


class Recommender:
    def fit(self, train: np.ndarray) -> "Recommender":
        raise NotImplementedError

    def score(self, users: np.ndarray) -> np.ndarray:
        """Scores for every item, shape (len(users), n_items)."""
        raise NotImplementedError

    def recommend(self, users: np.ndarray, k: int = 10, exclude_seen: bool = True) -> np.ndarray:
        s = self.score(users).astype(float)
        if exclude_seen:
            s[self.train[users] > 0] = -np.inf
        top = np.argpartition(-s, kth=min(k, s.shape[1] - 1), axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(s, top, axis=1), axis=1)
        return np.take_along_axis(top, order, axis=1)


class Popularity(Recommender):
    """Recommend the most-interacted items to everyone."""

    def fit(self, train):
        self.train = train
        self.pop = (train > 0).sum(axis=0).astype(float)
        return self

    def score(self, users):
        return np.tile(self.pop, (len(users), 1))


class ItemKNN(Recommender):
    """Item-item cosine similarity, keeping the top ``neighbors`` per item."""

    def __init__(self, neighbors: int = 50):
        self.neighbors = neighbors

    def fit(self, train):
        self.train = train
        x = (train > 0).astype(float)
        norms = np.linalg.norm(x, axis=0) + 1e-9
        sim = (x.T @ x) / np.outer(norms, norms)
        np.fill_diagonal(sim, 0)
        if self.neighbors < sim.shape[0]:
            cut = np.partition(sim, -self.neighbors, axis=1)[:, -self.neighbors][:, None]
            sim[sim < cut] = 0
        self.sim = sim
        return self

    def score(self, users):
        return (self.train[users] > 0).astype(float) @ self.sim


class ImplicitALS(Recommender):
    """Alternating Least Squares for implicit feedback (Hu, Koren & Volinsky, 2008).

    Preference p_ui = 1 if the user interacted with the item, else 0.
    Confidence c_ui = 1 + alpha * r_ui, so repeated interactions count more.
    Minimises  sum_ui c_ui (p_ui - x_u . y_i)^2 + reg (|X|^2 + |Y|^2)
    by solving a ridge regression per user, then per item, in turn.
    """

    def __init__(self, factors: int = 32, reg: float = 0.1, alpha: float = 10.0,
                 iterations: int = 15, seed: int = 0):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed = iterations, seed
        self.loss_history: list[float] = []

    def _solve(self, fixed: np.ndarray, conf: np.ndarray, pref: np.ndarray) -> np.ndarray:
        # For each row r:  (F^T C_r F + reg I) x_r = F^T C_r p_r
        # computed as F^T F + F^T (C_r - I) F to reuse the shared Gram matrix.
        ftf = fixed.T @ fixed
        eye = self.reg * np.eye(self.factors)
        out = np.empty((conf.shape[0], self.factors))
        for r in range(conf.shape[0]):
            nz = np.flatnonzero(pref[r])
            f_nz, c_nz = fixed[nz], conf[r, nz]
            a = ftf + (f_nz.T * (c_nz - 1)) @ f_nz + eye
            b = (f_nz.T * c_nz) @ np.ones(len(nz))
            out[r] = np.linalg.solve(a, b)
        return out

    def _loss(self, conf, pref) -> float:
        pred = self.user_factors @ self.item_factors.T
        fit = float((conf * (pref - pred) ** 2).sum())
        return fit + self.reg * float((self.user_factors ** 2).sum() + (self.item_factors ** 2).sum())

    def fit(self, train):
        self.train = train
        rng = np.random.default_rng(self.seed)
        n_users, n_items = train.shape
        pref = (train > 0).astype(float)
        conf = 1 + self.alpha * train
        self.user_factors = rng.normal(0, 0.01, (n_users, self.factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, self.factors))
        self.loss_history = []
        for _ in range(self.iterations):
            self.user_factors = self._solve(self.item_factors, conf, pref)
            self.item_factors = self._solve(self.user_factors, conf.T, pref.T)
            self.loss_history.append(self._loss(conf, pref))
        return self

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T

    def similar_items(self, item: int, k: int = 10) -> np.ndarray:
        v = self.item_factors
        sims = v @ v[item] / (np.linalg.norm(v, axis=1) * np.linalg.norm(v[item]) + 1e-9)
        sims[item] = -np.inf
        return np.argsort(-sims)[:k]
