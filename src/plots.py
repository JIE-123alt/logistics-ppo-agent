"""Make all PNG plots from the saved CSV files (nothing is typed in by hand).
Run:  python run_all.py --only plots [--quick]
"""

import matplotlib
matplotlib.use("Agg")                       # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

from src.final_eval import values
from src.utils import RUN_NAMES, get_dirs, load_run


def save(fig, results_dir, name):
    fig.tight_layout()
    fig.savefig(results_dir / name, dpi=130)
    plt.close(fig)


def learning_curves(results_dir, runs_dir):
    """Train (dashed) and validation (solid) cost against training steps, one panel per run."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, name in zip(axes, RUN_NAMES):
        for seed in load_run(name)["seeds"]:
            log = pd.read_csv(runs_dir / f"{name}_seed{seed}" / "eval_log.csv")
            line, = ax.plot(log["step"], log["val_cost"], label=f"val seed {seed}")
            ax.plot(log["step"], log["train_cost"], "--", color=line.get_color(), label=f"train seed {seed}")
        ax.set(title=name, xlabel="training steps", ylabel="mean total cost per episode")
        ax.legend(fontsize=7)
    save(fig, results_dir, "learning_curves.png")


def mean_std(summary, policy, run, split):
    vals = pd.Series(values(summary, policy, run, split))              # one cost per seed
    return vals.mean(), (vals.std() if len(vals) > 1 else 0.0)


def grouped_bars(ax, summary, labels, splits):
    """For each (policy, run, name) in labels draw one bar per split, side by side."""
    for i, split in enumerate(splits):
        vals = [mean_std(summary, p, r, split) for p, r, _ in labels]
        xs = [j + (i - 0.5) * 0.38 for j in range(len(labels))]
        ax.bar(xs, [v[0] for v in vals], 0.38, yerr=[v[1] for v in vals], capsize=3, label=split)
    ax.set_xticks(range(len(labels)), [name for _, _, name in labels])
    ax.legend()


def train_vs_validation(results_dir, summary):
    labels = [("random", "-", "random"), ("base_stock", "-", "base-stock"), ("ppo", "run_a_underfit", "Run A"),
              ("ppo", "run_b_overfit", "Run B"), ("ppo", "run_c_final", "Run C")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped_bars(ax, summary, labels, ["train", "val"])
    ax.set(ylabel="mean total cost per episode", title="Train vs validation cost (lower is better)")
    save(fig, results_dir, "train_vs_validation.png")


def run_comparison(results_dir, summary):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = ["Run A\n(underfit demo)", "Run B\n(overfit demo)", "Run C\n(final)"]
    vals = [mean_std(summary, "ppo", r, "val") for r in RUN_NAMES]
    ax.bar(names, [v[0] for v in vals], yerr=[v[1] for v in vals], capsize=4, color=["#c77", "#ca6", "#6a8"])
    for label, policy in [("random", "random"), ("base-stock", "base_stock")]:
        ax.axhline(mean_std(summary, policy, "-", "val")[0], ls="--", color="gray")
        ax.text(2.45, mean_std(summary, policy, "-", "val")[0], label, va="bottom", ha="right", fontsize=8)
    ax.set(ylabel="validation cost", title="Runs A / B / C on validation scenarios")
    save(fig, results_dir, "run_abc_comparison.png")


def bar_vs_baselines(results_dir, summary):
    labels = [("random", "-", "random"), ("base_stock", "-", "base-stock"), ("ppo", "run_c_final", "PPO (Run C)")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grouped_bars(ax, summary, labels, ["val", "test"])
    ax.set(ylabel="mean total cost per episode", title="PPO vs baselines (error bar = std over 3 seeds)")
    save(fig, results_dir, "bar_vs_baselines.png")


def stress_curves(results_dir):
    stress = pd.read_csv(results_dir / "stress_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (test, df) in zip(axes, stress.groupby("test", sort=False)):
        for policy, d in df.groupby("policy", sort=False):
            ax.plot(d["level"], d["total_cost"], marker="o", label=policy)
        ax.set(title=f"Stress test: {test}", xlabel=test, ylabel="mean total cost (validation scenarios)")
        ax.legend()
    save(fig, results_dir, "stress_curves.png")


def ppo_indicators(results_dir, runs_dir):
    """Entropy, approx_kl, clip_fraction, explained_variance for Run C (from SB3's progress.csv)."""
    cols = {"train/entropy_loss": "entropy (= -entropy_loss)", "train/approx_kl": "approx_kl",
            "train/clip_fraction": "clip_fraction", "train/explained_variance": "explained_variance"}
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for seed in load_run("run_c_final")["seeds"]:
        log = pd.read_csv(runs_dir / f"run_c_final_seed{seed}" / "progress.csv")
        for ax, (col, title) in zip(axes, cols.items()):
            y = -log[col] if col == "train/entropy_loss" else log[col]
            ax.plot(log["time/total_timesteps"], y, label=f"seed {seed}")
            ax.set(title=title, xlabel="training steps")
    axes[0].legend()
    save(fig, results_dir, "ppo_indicators.png")


def run(quick):
    results_dir, runs_dir = get_dirs(quick)
    summary = pd.read_csv(results_dir / "run_summary.csv")
    learning_curves(results_dir, runs_dir)
    train_vs_validation(results_dir, summary)
    run_comparison(results_dir, summary)
    bar_vs_baselines(results_dir, summary)
    stress_curves(results_dir)
    ppo_indicators(results_dir, runs_dir)
    print("Plots saved in", results_dir)
