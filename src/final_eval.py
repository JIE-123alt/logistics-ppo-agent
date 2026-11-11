"""
 * train and validation: random, base-stock, and the best checkpoint of runs A, B and C
 * test: ONLY run C (3 seeds), random and base-stock, in ONE call (the test set is used once)

Then it checks the pass/fail rules for Run C (limits come from config/run_c_final.yaml, written BEFORE any run).
Writes  results/episodes_all.csv (one row per episode), results/run_summary.csv and results/pass_fail.csv.
Run:  python run_all.py --only eval [--quick]
"""
import logging

import pandas as pd
from stable_baselines3 import PPO

from src.baselines import RandomPolicy, load_base_stock
from src.evaluate import evaluate, ppo_policy, use_test_set
from src.train import training_pool
from src.utils import RUN_NAMES, get_dirs, get_splits, load_run, load_yaml, save_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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


def values(summary, policy, run, split, col="total_cost"):
    rows = summary[(summary["policy"] == policy) & (summary["run"] == run) & (summary["split"] == split)]
    return rows.sort_values("seed")[col].tolist()


def curve_change(log, fraction):
    """Relative improvement of the validation cost in the last window versus the window before."""
    n = len(log)
    w = max(1, int(round(n * fraction)))
    if n < 2 * w:
        return None
    last = log["val_cost"].tail(w).mean()
    before = log["val_cost"].iloc[n - 2 * w:n - w].mean()
    return (before - last) / before


def check_run_c(summary, runs_dir, run_cfg):
    """Return a list of rows: criterion, measured value, limit, PASS/FAIL."""
    c, name = run_cfg["pass_criteria"], run_cfg["run_name"]
    val, train = values(summary, "ppo", name, "val"), values(summary, "ppo", name, "train")
    test = values(summary, "ppo", name, "test")
    random_val = values(summary, "random", "-", "val")[0]
    mean = lambda x: sum(x) / len(x)
    rows = []

    def add(group, text, measured, limit, ok):
        rows.append({"group": group, "criterion": text, "measured": measured, "limit": limit,
                     "result": "PASS" if ok else "FAIL"})

    improvement = (random_val - mean(val)) / random_val
    add("(a) not underfit", "validation cost better than random by", f"{improvement:.1%}",
        f">= {c['min_improvement_vs_random']:.0%}", improvement >= c["min_improvement_vs_random"])
    changes = []
    for seed in run_cfg["seeds"]:
        log = pd.read_csv(runs_dir / f"{name}_seed{seed}" / "eval_log.csv")
        changes.append(curve_change(log, c["flat_window_fraction"]))
    if None in changes:
        add("(a) not underfit", "learning curve flat (last 20% vs previous 20%)", "too few checkpoints",
            f"<= {c['flat_tolerance']:.0%} gain", False)
    else:
        worst = max(changes)
        add("(a) not underfit", "learning curve flat: biggest gain in last 20% of training", f"{worst:.1%}",
            f"<= {c['flat_tolerance']:.0%}", worst <= c["flat_tolerance"])

    gap = mean([(v - t) / t for v, t in zip(val, train)])
    add("(b) not overfit", "train vs validation cost gap (val - train) / train", f"{gap:.1%}",
        f"<= {c['max_train_val_gap']:.0%}", gap <= c["max_train_val_gap"])
    vt = mean([abs(te - v) / v for te, v in zip(test, val)])
    add("(b) not overfit", "test cost vs validation cost, abs(test - val) / val", f"{vt:.1%}",
        f"<= {c['max_val_test_gap']:.0%}", vt <= c["max_val_test_gap"])

    for label, vals in [("validation", val), ("test", test)]:
        spread = (max(vals) - min(vals)) / mean(vals)
        add("(c) stable", f"3 seeds, {label} cost: (max - min) / mean", f"{spread:.1%}",
            f"<= {c['max_seed_spread']:.0%}", spread <= c["max_seed_spread"])
    return rows


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

    for run_name in RUN_NAMES:                                # PPO runs: train + validation
        run_cfg = load_run(run_name)
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
    run_c = load_run("run_c_final")
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
    pass_fail = pd.DataFrame(check_run_c(summary, runs_dir, run_c))
    pass_fail.to_csv(results_dir / "pass_fail.csv", index=False)

    val_c = summary[(summary["run"] == run_c["run_name"]) & (summary["split"] == "val")]
    best_seed = int(val_c.loc[val_c["total_cost"].idxmin(), "seed"])   # chosen on VALIDATION only
    save_json({"run": run_c["run_name"], "seed": best_seed, "test_used_count": count},
              results_dir / "final_model.json")
    print(summary[["policy", "run", "seed", "split", "total_cost", "fill_rate"]].round(3).to_string())
    print("Final model (chosen on validation): seed", best_seed, "| test set used", count, "time(s)")
    print(pass_fail[["group", "criterion", "measured", "limit", "result"]].to_string(index=False))
