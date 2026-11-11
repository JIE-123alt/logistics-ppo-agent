"""Final evaluation. Uses the SAME evaluate() function for every policy and every split.

 * train and validation: random, base-stock, and the best checkpoint of runs A, B and C
 * test: ONLY run C (3 seeds), random and base-stock, in ONE call (the test set is used once)

Writes  results/episodes_all.csv (one row per episode)  and  results/run_summary.csv.
Run:  python -m src.final_eval [--quick]
"""
import argparse
import logging

import pandas as pd
from stable_baselines3 import PPO

from src.baselines import RandomPolicy, load_base_stock
from src.evaluate import evaluate, ppo_policy, use_test_set
from src.train import training_pool
from src.utils import get_dirs, get_splits, load_yaml, save_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
RUNS = ["config/run_a_underfit.yaml", "config/run_b_overfit.yaml", "config/run_c_final.yaml"]
PPO_INDICATORS = {"train/entropy_loss": "entropy", "train/approx_kl": "approx_kl",
                  "train/clip_fraction": "clip_fraction", "train/explained_variance": "explained_variance"}


def collect(df, policy, run, seed, split):
    df = df.copy()
    df.insert(0, "policy", policy), df.insert(1, "run", run)
    df.insert(2, "seed", seed), df.insert(3, "split", split)
    return df


def ppo_indicators(run_dir):
    """Average of the last 10 logged updates from SB3's progress.csv (entropy = -entropy_loss)."""
    log = pd.read_csv(run_dir / "progress.csv").tail(10)
    out = {}
    for col, name in PPO_INDICATORS.items():
        if col in log:
            value = log[col].mean()
            out[name] = -value if name == "entropy" else value
    return out


def run(quick):
    cfg = load_yaml("config/env.yaml")
    results_dir, runs_dir = get_dirs(quick)
    splits = get_splits(cfg)
    n_eps = 2 if quick else 5
    frames, extra = [], {}
    base_stock = load_base_stock(cfg, results_dir)
    baselines = {"random": RandomPolicy(), "base_stock": base_stock}

    for name, policy in baselines.items():                    # baselines: train + validation
        for split in ["train", "val"]:
            frames.append(collect(evaluate(policy, cfg, splits[split], n_eps), name, "-", 0, split))
        # the 3 scenarios Run B memorises: a fair yardstick for how hard those 3 scenarios are
        frames.append(collect(evaluate(policy, cfg, splits["train"][:3], n_eps), name, "-", 0, "train3"))

    for path in RUNS:                                         # PPO runs: train + validation
        run_cfg = load_yaml(path)
        pool = training_pool(run_cfg, splits)
        for seed in run_cfg["seeds"]:
            name = f"{run_cfg['run_name']}_seed{seed}"
            model = PPO.load(runs_dir / name / "best_model", device="cpu")
            policy = ppo_policy(model)
            for split, seeds in [("train", pool), ("val", splits["val"])]:
                frames.append(collect(evaluate(policy, cfg, seeds, n_eps), "ppo", run_cfg["run_name"], seed, split))
            extra[(run_cfg["run_name"], seed)] = ppo_indicators(runs_dir / name)

    # ---- the ONE use of the test set: run C (all seeds) + the two baselines ----
    count = use_test_set(results_dir, "final evaluation (run C seeds, random, base-stock)")
    for name, policy in baselines.items():
        frames.append(collect(evaluate(policy, cfg, splits["test"], n_eps), name, "-", 0, "test"))
    run_c = load_yaml(RUNS[2])
    for seed in run_c["seeds"]:
        model = PPO.load(runs_dir / f"{run_c['run_name']}_seed{seed}" / "best_model", device="cpu")
        frames.append(collect(evaluate(ppo_policy(model), cfg, splits["test"], n_eps), "ppo", run_c["run_name"], seed, "test"))

    episodes = pd.concat(frames, ignore_index=True)
    episodes.to_csv(results_dir / "episodes_all.csv", index=False)
    summary = (episodes.groupby(["policy", "run", "seed", "split"])
               .agg(total_cost=("total_cost", "mean"), total_cost_std=("total_cost", "std"),
                    cost_per_unit=("cost_per_unit", "mean"), fill_rate=("fill_rate", "mean"),
                    stockout_freq=("stockout_freq", "mean"), fallback_rate=("fallback_rate", "mean"),
                    loop_trap_rate=("loop_trap", "mean"), episodes=("total_cost", "size"))
               .reset_index())
    for key, ind in extra.items():                            # add PPO logger indicators
        for col, value in ind.items():
            summary.loc[(summary["run"] == key[0]) & (summary["seed"] == key[1]), col] = value
    summary.to_csv(results_dir / "run_summary.csv", index=False)

    val_c = summary[(summary["run"] == run_c["run_name"]) & (summary["split"] == "val")]
    best_seed = int(val_c.loc[val_c["total_cost"].idxmin(), "seed"])   # chosen on VALIDATION only
    save_json({"run": run_c["run_name"], "seed": best_seed, "test_used_count": count},
              results_dir / "final_model.json")
    print(summary[["policy", "run", "seed", "split", "total_cost", "fill_rate"]].round(3).to_string())
    print("Final model (chosen on validation): seed", best_seed, "| test set used", count, "time(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    run(parser.parse_args().quick)
