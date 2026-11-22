"""Train and compare recommenders.

    python -m recsys                       # synthetic data
    python -m recsys --csv ratings.csv     # your own (user_id,item_id,rating) data
"""

from __future__ import annotations

import argparse
import sys
import time

from .data import from_csv, leave_n_out, synthetic
from .metrics import evaluate, tune_als
from .models import ImplicitALS, ItemKNN, Popularity


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="recsys", description="Compare recommenders.")
    p.add_argument("--csv", help="CSV with user_id,item_id,rating columns")
    p.add_argument("--user-col", default="user_id")
    p.add_argument("--item-col", default="item_id")
    p.add_argument("--value-col", default="rating")
    p.add_argument("--min-rating", type=float, default=0,
                   help="treat ratings <= this as no interaction")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--factors", type=int, default=32)
    p.add_argument("--holdout", type=int, default=5)
    p.add_argument("--tune", action="store_true",
                   help="grid-search ALS hyperparameters on a validation split")
    a = p.parse_args(argv)

    data = (from_csv(a.csv, a.user_col, a.item_col, a.value_col, a.min_rating)
            if a.csv else synthetic())
    train, test = leave_n_out(data, n=a.holdout)
    print(f"{data.shape[0]} users x {data.shape[1]} items, "
          f"{int((train > 0).sum())} train / {int((test > 0).sum())} test interactions\n")

    als_params = {"factors": a.factors}
    if a.tune:
        grid = {"factors": [8, 16, 32], "reg": [0.1, 1.0, 10.0], "alpha": [1.0, 5.0, 20.0],
                "iterations": [10]}
        als_params, val = tune_als(train, grid, a.k)
        print(f"Tuned ALS on validation split: {als_params} (val NDCG@{a.k}={val:.3f})\n")

    models = {
        "Popularity": Popularity(),
        "ItemKNN": ItemKNN(neighbors=50),
        "ImplicitALS": ImplicitALS(**als_params),
    }
    header = f"{'model':<12} {'P@'+str(a.k):>7} {'R@'+str(a.k):>7} {'NDCG@'+str(a.k):>8} {'coverage':>9} {'fit (s)':>8}"
    print(header)
    print("-" * len(header))
    for name, m in models.items():
        t = time.perf_counter()
        m.fit(train)
        secs = time.perf_counter() - t
        r = evaluate(m, test, a.k)
        print(f"{name:<12} {r[f'precision@{a.k}']:>7.3f} {r[f'recall@{a.k}']:>7.3f} "
              f"{r[f'ndcg@{a.k}']:>8.3f} {r['catalog_coverage']:>9.2f} {secs:>8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
