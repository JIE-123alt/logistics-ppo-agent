r"""Makes the 3 report figures. Save this file in the repo's results\ folder and run from the repo folder:
    python results\make_plots.py
Reads:  results\run_summary.csv, results\stress_results.csv, runs\run_c_final_seed{0,1,2}\eval_log.csv
Writes: results\fig_validation_curves.png, fig_test_cost.png, fig_stress.png
Needs:  pip install pandas matplotlib
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent      # ...\results
ROOT = HERE.parent                          # ...\logistics-ppo-agent
OUT = HERE
FINAL_SEED = 1                              # the seed in results\final_model.json
EVAL_LOGS = {s: ROOT / "runs" / f"run_c_final_seed{s}" / "eval_log.csv" for s in (0, 1, 2)}

for p in [HERE / "run_summary.csv", HERE / "stress_results.csv", *EVAL_LOGS.values()]:
    if not p.exists():
        raise SystemExit(f"Missing file: {p}\nCheck the folder names, or edit the paths at the top of this script.")

summary = pd.read_csv(HERE / "run_summary.csv")
def cost(policy, split, run=None, seed=0):
    r = summary[(summary.policy == policy) & (summary.split == split)]
    if run:
        r = r[(r.run == run) & (r.seed == seed)]
    return float(r.total_cost.iloc[0])

# Figure 1: validation cost vs training steps (dots = checkpoint kept as best_model)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for seed, f in EVAL_LOGS.items():
    e = pd.read_csv(f)
    line, = ax.plot(e.step, e.val_cost, label=f"PPO seed {seed}", lw=1.4)
    b = e.loc[e.val_cost.idxmin()]
    ax.plot(b.step, b.val_cost, "o", color=line.get_color())
ax.axhline(cost("random", "val"), color="grey", ls="--", label="Random (val)")
ax.axhline(cost("base_stock", "val"), color="black", ls=":", label="Base-stock (val)")
ax.set_ylim(13000, 21500)
ax.set_xlabel("Training steps"); ax.set_ylabel("Validation cost (52-week total)")
ax.set_title("Validation cost during training (dots = selected checkpoints)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_validation_curves.png", dpi=200)

# Figure 2: test cost
labels, vals = ["Random", "Base-stock"], [cost("random", "test"), cost("base_stock", "test")]
for s in (0, 1, 2):
    labels.append(f"PPO seed {s}" + (" (final)" if s == FINAL_SEED else ""))
    vals.append(cost("ppo", "test", "run_c_final", s))
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(labels, vals, color=["#999", "#333", "#6a9fd8", "#1f5fa8", "#6a9fd8"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 200, f"{v:,.0f}", ha="center", fontsize=8)
ax.set_ylabel("Test cost (52-week total)"); ax.set_title("Test-set cost by policy")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right"); fig.tight_layout()
fig.savefig(OUT / "fig_test_cost.png", dpi=200)

# Figure 3: stress tests (validation scenarios)
st = pd.read_csv(HERE / "stress_results.csv")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, (name, title) in zip(axes, [("forecast_noise_std", "Forecast noise (std)"),
                                    ("api_failure_prob", "Order-API failure probability")]):
    for pol, style in [("ppo", "-o"), ("base_stock", "--s"), ("random", ":^")]:
        d = st[(st.test == name) & (st.policy == pol)]
        ax.plot(d.level, d.total_cost, style, label=pol)
    ax.set_xlabel(title); ax.grid(alpha=0.3)
axes[0].set_ylabel("Validation cost"); axes[0].legend(fontsize=8)
fig.suptitle("Stress tests (validation scenarios)"); fig.tight_layout()
fig.savefig(OUT / "fig_stress.png", dpi=200)
print("Saved 3 figures in", OUT)
