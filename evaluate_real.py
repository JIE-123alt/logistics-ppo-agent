"""Evaluate synthetic-trained PPO checkpoints on held-out real-demand products.

Run after training Run C, tuning the synthetic base-stock baseline, and preparing the real panel:
    python evaluate_real.py
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from src.baselines import RandomPolicy, load_base_stock
from src.evaluate import ppo_policy
from src.real_env import make_windows, run_policy
from src.utils import ROOT, load_run, load_yaml


def load_test_windows(cfg, eval_cfg, panel_dir=None):
    panel_dir = Path(panel_dir or ROOT / eval_cfg["real"]["panel_dir"])
    panel_path, split_path = panel_dir / "real_panel.csv", panel_dir / "split.json"
    if not panel_path.exists() or not split_path.exists():
        raise FileNotFoundError(
            f"Missing real test panel under {panel_dir}. Run prepare_real_panel.py first."
        )

    split = json.loads(split_path.read_text(encoding="utf-8"))
    products = split.get("test")
    if not isinstance(products, list) or not products:
        raise ValueError(f"{split_path} must contain a non-empty 'test' product list.")
    calibration_weeks = int(split["calibration_weeks"])
    if calibration_weeks != eval_cfg["real"]["calibration_weeks"]:
        raise ValueError("Panel calibration_weeks does not match config/real_eval.yaml.")

    long = pd.read_csv(panel_path, dtype={"product": str})
    panel = long.pivot(index="week_ending", columns="product", values="demand").sort_index()
    return make_windows(
        panel,
        products,
        cfg,
        stride=int(eval_cfg["real"]["eval_stride"]),
        calibration_weeks=calibration_weeks,
    )


def check_test_lock(results_dir, allow_reuse):
    lock_path = results_dir / "test_lock_real.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {"uses": []}
    if lock["uses"] and not allow_reuse:
        raise RuntimeError(
            f"Real test split has already been used {len(lock['uses'])} time(s). "
            "Pass --allow-test-reuse only if you intend to rerun it."
        )
    return lock_path, lock


def policies_to_evaluate(cfg, run_cfg):
    results_dir = ROOT / "results"
    baseline_path = results_dir / "baseline_params.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            "Missing synthetic-tuned results/baseline_params.json. "
            "Run the synthetic baseline-tuning step before real-data evaluation."
        )
    policies = [
        {"policy": "random", "seed": None, "training_data": "none",
         "agent": RandomPolicy()},
        {"policy": "synthetic_base_stock", "seed": None, "training_data": "synthetic",
         "agent": load_base_stock(cfg, results_dir)},
    ]
    for seed in run_cfg["seeds"]:
        name = f"{run_cfg['run_name']}_seed{seed}"
        model_path = ROOT / "runs" / name / "best_model.zip"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing synthetic-trained checkpoint {model_path}. Train Run C first."
            )
        policies.append({
            "policy": "ppo",
            "seed": seed,
            "training_data": "synthetic",
            "agent": ppo_policy(PPO.load(model_path, device="cpu")),
        })
    return policies


def evaluate_test(cfg, eval_cfg, windows, allow_reuse=False):
    if not windows:
        raise ValueError("No real test windows were generated.")

    results_dir = ROOT / "results_real"
    results_dir.mkdir(parents=True, exist_ok=True)
    lock_path, lock = check_test_lock(results_dir, allow_reuse)
    episodes_path = results_dir / "real_test_episodes.csv"
    summary_path = results_dir / "real_test_summary.csv"
    if not allow_reuse and (episodes_path.exists() or summary_path.exists()):
        raise FileExistsError(
            f"Real test output already exists under {results_dir}; "
            "use --allow-test-reuse to replace it."
        )

    run_cfg = load_run("run_c_final")
    policies = policies_to_evaluate(cfg, run_cfg)
    episode_frames = []
    for item in policies:
        stats = run_policy(
            item["agent"],
            cfg,
            windows,
            repeats=int(eval_cfg["real"]["test_repeats"]),
            base_seed=70000,
        )
        stats.insert(0, "policy", item["policy"])
        stats.insert(1, "seed", item["seed"])
        stats.insert(2, "training_data", item["training_data"])
        episode_frames.append(stats)

    episodes = pd.concat(episode_frames, ignore_index=True)
    summary = (
        episodes.groupby(["policy", "seed", "training_data"], dropna=False)
        .agg(
            mean_total_cost=("total_cost", "mean"),
            std_total_cost=("total_cost", "std"),
            mean_cost_per_unit=("cost_per_unit", "mean"),
            mean_fill_rate=("fill_rate", "mean"),
            mean_stockout_freq=("stockout_freq", "mean"),
            episodes=("total_cost", "size"),
            products=("product", "nunique"),
        )
        .reset_index()
    )
    episodes.to_csv(episodes_path, index=False)
    summary.to_csv(summary_path, index=False)

    lock["uses"].append({
        "time": datetime.now().isoformat(timespec="seconds"),
        "products": len({window["product"] for window in windows}),
        "windows": len(windows),
        "explicit_reuse": bool(lock["uses"]),
    })
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved real test episodes to {episodes_path}")
    print(f"Saved real test summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/real_eval.yaml")
    parser.add_argument(
        "--allow-test-reuse",
        action="store_true",
        help="explicitly rerun the held-out real test split",
    )
    args = parser.parse_args()

    cfg = load_yaml("config/env.yaml")
    eval_cfg = load_yaml(args.config)
    windows = load_test_windows(cfg, eval_cfg)
    print(f"Loaded {len(windows)} held-out real-demand windows for evaluation.")
    evaluate_test(cfg, eval_cfg, windows, allow_reuse=args.allow_test_reuse)


if __name__ == "__main__":
    main()
