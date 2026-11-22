# recsys-implicit-als

![CI](https://github.com/laxmikanthrm/recsys-implicit-als/actions/workflows/ci.yml/badge.svg)

A recommender system for **implicit feedback** (clicks, views, purchases)
built from scratch with NumPy. It implements weighted Alternating Least Squares
(Hu, Koren & Volinsky, 2008) and benchmarks it against popularity and item-kNN
baselines with standard ranking metrics.

## Results

600 users x 300 items, 5 held-out interactions per user, ALS hyperparameters
tuned on a separate validation split (the test set is never used for tuning):

| Model | Precision@10 | Recall@10 | NDCG@10 | Catalog coverage |
|---|---|---|---|---|
| Popularity | 0.032 | 0.064 | 0.049 | 6% |
| Item-kNN (cosine, 50 neighbours) | 0.120 | 0.240 | 0.185 | 90% |
| **Implicit ALS** (32 factors) | **0.122** | **0.244** | **0.185** | **99%** |

Both personalised models beat popularity by about **3.8x on NDCG@10**. ALS matches
item-kNN on accuracy while recommending from almost the entire catalogue,
which matters for long-tail items, and it produces dense item embeddings that
can be reused for "similar items".

Reproduce with `python -m recsys --tune`.

## How ALS works here

- Preference `p_ui = 1` if user *u* interacted with item *i*, else 0.
- Confidence `c_ui = 1 + alpha * r_ui`: repeated interactions count more, and
  missing ones still count a little (they're weak negatives, not unknowns).
- Minimise `sum c_ui (p_ui - x_u . y_i)^2 + reg (|X|^2 + |Y|^2)`.
- Fix item factors and solve a ridge regression for each user, then swap.
  The shared Gram matrix `YᵀY` is computed once per sweep, so each user's
  solve only touches the items they interacted with.

See `recsys/models.py` (`ImplicitALS`). Training loss is tracked per iteration
and a test checks it decreases.

## Quick start

```bash
pip install -e ".[dev]"
python -m recsys            # synthetic data, default params
python -m recsys --tune     # tune ALS on a validation split first
pytest -q
```

Run on real data, e.g. [MovieLens](https://grouplens.org/datasets/movielens/)
(treating ratings of 4+ as positive interactions):

```bash
python -m recsys --csv ratings.csv --user-col userId --item-col movieId \
                 --value-col rating --min-rating 3.5 --tune
```

## Project layout

```
recsys/data.py      # synthetic generator, CSV loader, leave-n-out split
recsys/models.py    # Popularity, ItemKNN, ImplicitALS
recsys/metrics.py   # precision/recall/NDCG@k, coverage, validation tuning
recsys/__main__.py  # benchmark CLI
tests/              # pytest suite
```

## Next steps

- Sparse matrices + conjugate-gradient solver to scale to millions of users.
- Add BPR (Bayesian Personalized Ranking) for comparison.
- Serve top-k recommendations through an API with approximate nearest-neighbour search.

## License

MIT
