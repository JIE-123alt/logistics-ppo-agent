"""Two baselines: a random policy and a base-stock rule tuned on the TRAIN scenarios only.

Both see only what the PPO agent sees (the normalised observation), so the comparison is fair.
Run:  python run_all.py --only baselines [--quick]
"""
import itertools

import numpy as np

from src.env import N_SIZES, unpack_obs
from src.evaluate import evaluate
from src.utils import get_dirs, get_splits, load_json, load_yaml, save_json


class RandomPolicy:
    """Picks any of the 8 actions with equal chance. Re-seeded every episode for repeatability."""

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return int(self.rng.integers(0, 2 * N_SIZES))


class BaseStockPolicy:
    """Order up to a target level S = cover_weeks x forecast.

    gap = S - (stock on hand + stock on the way). Order the allowed size closest to the gap.
    Use Supplier B when the reliability signal of A is below `threshold`, otherwise Supplier A.
    """

    def __init__(self, cfg, cover_weeks, threshold):
        self.cfg, self.cover_weeks, self.threshold = cfg, cover_weeks, threshold

    def __call__(self, obs):
        o = unpack_obs(obs, self.cfg)
        position = o["inventory"] + o["transit_a"] + o["transit_b"]
        gap = max(0.0, self.cover_weeks * o["forecast"] - position)
        size_index = int(np.argmin([abs(s - gap) for s in self.cfg["order_sizes"]]))
        supplier = 1 if o["rel_signal"] < self.threshold else 0
        return supplier * N_SIZES + size_index


def tune_base_stock(cfg, train_seeds, episodes=2):
    """Grid search on TRAIN scenarios only. Returns (best params, table of all tries)."""
    covers = [1, 2, 3, 4, 5, 6]
    thresholds = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]   # 0.0 = never B, 1.01 = always B
    table = []
    for cover, thr in itertools.product(covers, thresholds):
        # ep_offset=50 so tuning episodes differ from the episodes used later to report train cost
        policy = BaseStockPolicy(cfg, cover, thr)
        cost = float(evaluate(policy, cfg, train_seeds, episodes, ep_offset=50)["total_cost"].mean())
        table.append({"cover_weeks": cover, "threshold": thr, "train_cost": cost})
    best = min(table, key=lambda r: r["train_cost"])
    return {"cover_weeks": best["cover_weeks"], "threshold": best["threshold"]}, table


def load_base_stock(cfg, results_dir):
    """Build the tuned base-stock policy from the saved parameters."""
    p = load_json(results_dir / "baseline_params.json")
    return BaseStockPolicy(cfg, p["cover_weeks"], p["threshold"])


def run(quick):
    cfg = load_yaml("config/env.yaml")
    results_dir, _ = get_dirs(quick)
    train = get_splits(cfg)["train"][:10] if quick else get_splits(cfg)["train"]
    best, table = tune_base_stock(cfg, train, episodes=1 if quick else 2)
    best["tuned_on"] = f"{len(train)} train scenarios"
    save_json(best, results_dir / "baseline_params.json")
    save_json(table, results_dir / "baseline_tuning_table.json")
    print("Best base-stock parameters (tuned on train only):", best)
