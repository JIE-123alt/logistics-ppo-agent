"""Two stress tests: forecast noise std and order-API failure probability.

The final model (chosen on validation) is compared with the baselines on the VALIDATION
scenarios (so the test set stays untouched). Baselines are NOT re-tuned: same settings as before.
Run:  python -m src.stress_test [--quick]
"""
import argparse

import pandas as pd
from stable_baselines3 import PPO

from src.baselines import RandomPolicy, load_base_stock
from src.evaluate import evaluate, ppo_policy
from src.utils import get_dirs, get_splits, load_json, load_yaml, with_changes

TESTS = {
    "forecast_noise_std": [0.05, 0.15, 0.30, 0.50],
    "api_failure_prob": [0.0, 0.05, 0.20, 0.40],
}


def run(quick):
    cfg = load_yaml("config/env.yaml")
    results_dir, runs_dir = get_dirs(quick)
    final = load_json(results_dir / "final_model.json")
    model = PPO.load(runs_dir / f"{final['run']}_seed{final['seed']}" / "best_model", device="cpu")
    policies = {"ppo": ppo_policy(model), "base_stock": load_base_stock(cfg, results_dir),
                "random": RandomPolicy()}
    seeds = get_splits(cfg)["val"]
    n_eps = 1 if quick else 3
    rows = []
    for setting, levels in TESTS.items():
        for level in levels:
            stressed = with_changes(cfg, "uncertainty", **{setting: level})
            for name, policy in policies.items():
                df = evaluate(policy, stressed, seeds, n_eps)
                rows.append({"test": setting, "level": level, "policy": name,
                             "total_cost": df["total_cost"].mean(), "total_cost_std": df["total_cost"].std(),
                             "fill_rate": df["fill_rate"].mean(), "fallback_rate": df["fallback_rate"].mean(),
                             "loop_trap_rate": df["loop_trap"].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(results_dir / "stress_results.csv", index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    run(parser.parse_args().quick)
