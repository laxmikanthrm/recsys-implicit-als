"""Interaction data: synthetic generator, CSV loader, and train/test split."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np


@dataclass
class Interactions:
    matrix: np.ndarray            # users x items, interaction counts (0 = none)
    user_ids: list
    item_ids: list

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape


def synthetic(n_users: int = 600, n_items: int = 300, n_genres: int = 8,
              interactions_per_user: int = 25, seed: int = 0) -> Interactions:
    """Users mostly interact with items from 1-2 favourite genres, plus some
    globally popular items. This gives the data real latent structure that a
    good model should recover, and a popularity bias a naive model can exploit.
    """
    rng = np.random.default_rng(seed)
    item_genre = rng.integers(0, n_genres, n_items)
    popularity = rng.zipf(1.6, n_items).astype(float)
    popularity = np.minimum(popularity, 50) / 50
    m = np.zeros((n_users, n_items))
    for u in range(n_users):
        favs = rng.choice(n_genres, size=rng.integers(1, 3), replace=False)
        w = 0.15 * popularity + np.isin(item_genre, favs) * 1.0 + 0.01
        w /= w.sum()
        items = rng.choice(n_items, size=interactions_per_user, replace=False, p=w)
        m[u, items] = rng.integers(1, 5, size=len(items))
    return Interactions(m, list(range(n_users)), list(range(n_items)))


def from_csv(path: str, user_col: str = "user_id", item_col: str = "item_id",
             value_col: str | None = "rating", min_value: float = 0) -> Interactions:
    """Load a CSV of (user, item[, value]) rows, e.g. MovieLens ratings."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = float(r[value_col]) if value_col else 1.0
            if v > min_value:
                rows.append((r[user_col], r[item_col], v))
    users = sorted({u for u, _, _ in rows})
    items = sorted({i for _, i, _ in rows})
    ui, ii = {u: k for k, u in enumerate(users)}, {i: k for k, i in enumerate(items)}
    m = np.zeros((len(users), len(items)))
    for u, i, v in rows:
        m[ui[u], ii[i]] = v
    return Interactions(m, users, items)


def leave_n_out(data: Interactions, n: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Hold out ``n`` random interactions per user (users with > 2n interactions)."""
    rng = np.random.default_rng(seed)
    train, test = data.matrix.copy(), np.zeros_like(data.matrix)
    for u in range(train.shape[0]):
        items = np.flatnonzero(train[u])
        if len(items) <= 2 * n:
            continue
        held = rng.choice(items, size=n, replace=False)
        test[u, held] = train[u, held]
        train[u, held] = 0
    return train, test
