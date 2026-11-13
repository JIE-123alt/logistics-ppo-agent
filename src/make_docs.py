"""Write FACT_SHEET.md, RESULTS_SUMMARY.md and VIVA_NOTES.md. Every number comes from a file
written by a real run (CSV / JSON / YAML); nothing is typed in by hand.
Run:  python run_all.py --only docs [--quick]
"""
from datetime import date

import pandas as pd

from src.final_eval import values
from src.utils import RUN_NAMES, ROOT, get_dirs, get_docs_dir, load_json, load_run, load_yaml


def table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def mean_std(vals):
    s = pd.Series(vals)
    return f"{s.mean():,.0f} ± {s.std():,.0f}" if len(vals) > 1 else f"{s.mean():,.0f}"


def headline_rows(summary):
    rows = []
    for label, policy, run in [("random", "random", "-"), ("base-stock", "base_stock", "-"), ("PPO Run A (underfit demo)", "ppo", RUN_NAMES[0]),
                               ("PPO Run B (overfit demo)", "ppo", RUN_NAMES[1]), ("PPO Run C (final)", "ppo", RUN_NAMES[2])]:
        for split in ["train", "val", "test"]:
            d = summary[(summary["policy"] == policy) & (summary["run"] == run) & (summary["split"] == split)]
            if d.empty:
                continue
            rows.append([label, split, mean_std(d["total_cost"].tolist()), f"{d['fill_rate'].mean():.3f}",
                         f"{d['stockout_freq'].mean():.3f}", f"{d['fallback_rate'].mean():.4f}",
                         f"{d['loop_trap_rate'].mean():.3f}", f"{d['cost_per_unit'].mean():.2f}"])
    return rows


def what_runs_showed(summary):
    """Sentences chosen by simple rules from the numbers (no hand-written conclusions)."""
    def v(policy, run, split):                      # mean over seeds (Run A, B and the baselines have 1 seed)
        vals = values(summary, policy, run, split)
        return sum(vals) / len(vals)

    rand_val, bs_val = v("random", "-", "val"), v("base_stock", "-", "val")
    out = []
    a_val = v("ppo", RUN_NAMES[0], "val")
    imp = (rand_val - a_val) / rand_val
    out.append(f"**Run A (underfit demo):** validation cost {a_val:,.0f} versus random {rand_val:,.0f} "
               f"(improvement over random: {imp:+.1%}). " +
               ("It stays below the 20% improvement bar, which is consistent with underfitting." if imp < 0.20 else
                "It still beat random by 20% or more, so this demo did NOT show clear underfitting."))
    b_train, b_val = v("ppo", RUN_NAMES[1], "train"), v("ppo", RUN_NAMES[1], "val")
    bs3, bsv = v("base_stock", "-", "train3"), bs_val
    gap_b = (b_val - b_train) / b_train
    rel_train, rel_val = b_train / bs3, b_val / bsv
    out.append(f"**Run B (overfit demo):** cost on its 3 training scenarios {b_train:,.0f}, on validation {b_val:,.0f} "
               f"(gap {gap_b:+.1%}). Measured against base-stock on the same scenarios it scores {rel_train:.2f}x on its "
               f"3 training scenarios and {rel_val:.2f}x on validation. " +
               ("The gap is above 10% and the relative score gets worse on unseen scenarios: overfitting is visible."
                if gap_b > 0.10 and rel_val > rel_train else
                "No clear overfitting gap appeared in this run (we report this honestly)."))
    c_val, c_train = v("ppo", RUN_NAMES[2], "val"), v("ppo", RUN_NAMES[2], "train")
    out.append(f"**Run C (final, mean of seeds):** train {c_train:,.0f}, validation {c_val:,.0f}; base-stock validation "
               f"{bs_val:,.0f}; random {rand_val:,.0f}. " +
               ("PPO beat the tuned base-stock rule on validation." if c_val < bs_val else
                "PPO did NOT beat the tuned base-stock rule on validation (the heuristic is better or equal)."))
    return out


def stress_table(results_dir):
    df = pd.read_csv(results_dir / "stress_results.csv")
    rows = []
    for (test, level), d in df.groupby(["test", "level"], sort=False):
        cost = {r.policy: f"{r.total_cost:,.0f}" for r in d.itertuples()}
        rows.append([test, level, cost["ppo"], cost["base_stock"], cost["random"]])
    return table(rows, ["stress test", "level", "PPO cost", "base-stock cost", "random cost"])


def results_summary(quick, results_dir, summary):
    criteria = pd.read_csv(results_dir / "pass_fail.csv").to_dict("records")   # written by final_eval
    lock = load_json(results_dir / "test_lock.json")
    lines = ["# Results summary (auto-generated)", "",
             f"Generated {date.today()} from the CSV files in `{results_dir.name}/`. Do not edit by hand."]
    if quick:
        lines += ["", "> **QUICK MODE: these numbers come from a very short smoke-test run. They are NOT the final results.**"]
    lines += ["", f"Test set used {lock['count']} time(s) (rule: once).", "", "## Headline results (mean total cost per episode, ± std)", "",
              table(headline_rows(summary), ["policy", "split", "total cost", "fill rate", "stockout freq", "fallback rate",
                                             "loop-trap rate", "cost per unit demand"]),
              "", "Run C rows show mean ± std across seeds. Run B's 'train' = its own 3 scenarios.", "",
              "## What the runs showed", ""] + [f"- {s}" for s in what_runs_showed(summary)]
    rows = [[c["group"], c["criterion"], c["measured"], c["limit"], c["result"]] for c in criteria]
    lines += ["", "## Pass / fail (criteria fixed in `config/run_c_final.yaml` before running; our own choice)", "",
              table(rows, ["group", "check", "measured", "limit", "result"])]
    c = summary[(summary["run"] == RUN_NAMES[2]) & (summary["split"] == "val")]
    ind = [[col, f"{c[col].mean():.4f}"] for col in ["entropy", "approx_kl", "clip_fraction", "explained_variance"] if col in c]
    lines += ["", "## PPO indicators, Run C (mean of the last 10 updates, mean over seeds)", "", table(ind, ["indicator", "value"]),
              "", "## Stress tests (validation scenarios, same settings for all policies)", "", stress_table(results_dir)]
    speed = results_dir / "speed_test.json"
    if speed.exists():
        lines += ["", "## Speed test (this computer)", "", "```", json_text(load_json(speed)), "```"]
    return "\n".join(lines) + "\n"


def json_text(d):
    return "\n".join(f"{k}: {v}" for k, v in d.items())


def fact_sheet(quick, results_dir, runs_dir):
    env, run_c = load_yaml("config/env.yaml"), load_run("run_c_final")
    info = load_json(runs_dir / "run_c_final_seed0" / "run_info.json")
    hp = [[k, v] for k, v in run_c["train"].items() if k not in ("quick_timesteps",)]
    hp.append(["timesteps actually used (this run)", info["timesteps"]])
    obs = ["inventory / 1000", "in-transit stock from A / 1000", "in-transit stock from B / 1000",
           "noisy forecast of next week's demand / 200", "noisy reliability signal of A (0..1)", "week / 52",
           "last_order_failed flag (0 or 1)"]
    sizes = env["order_sizes"]
    actions = [[a, f"{'AB'[a // 4]}", sizes[a % 4]] for a in range(8)]
    text = ["# Fact sheet (auto-generated from the YAML configs and run_info files)", "",
            "## Observation (7 numbers, normalised)", ""] + [f"{i + 1}. {o}" for i, o in enumerate(obs)]
    text += ["", "## Action: ONE Discrete(8) = supplier x order size", "", table(actions, ["action", "supplier", "order size"]),
             "", "(Actions 0 and 4 both mean 'order nothing'.)", "", "## Reward", "",
             f"reward = -(weekly cost) / {env['costs']['reward_scale']}. Weekly cost = holding {env['costs']['holding']} per unit left "
             f"+ lost sale {env['costs']['lost_sale']} per unit + fixed order cost {env['fixed_order_cost']} per accepted order "
             "+ unit cost of the goods that arrive that week (pay on delivery). Raw cost is saved in `info['cost']`. "
             "Critical ratio 5/(5+1) is about 83%: an approximate guide to how much safety stock is worth holding.",
             "", "## Environment settings (config/env.yaml)", "",
             table([[k, v] for k, v in {**env["uncertainty"], **env["safety"]}.items()], ["setting", "value"]),
             "", "## Final hyperparameters (config/run_c_final.yaml)", "", table(hp, ["hyperparameter", "value"]),
             "", "## Seeds, scenarios and versions", "",
             f"- PPO seeds: {run_c['seeds']}; scenario seeds: train {env['splits']['train']}, val {env['splits']['val']}, test {env['splits']['test']}",
             "- versions: " + ", ".join(f"{k} {v}" for k, v in info["versions"].items())]
    return "\n".join(text) + "\n"


def run(quick):
    results_dir, runs_dir = get_dirs(quick)
    docs = get_docs_dir(quick)
    summary = pd.read_csv(results_dir / "run_summary.csv")
    (docs / "RESULTS_SUMMARY.md").write_text(results_summary(quick, results_dir, summary), encoding="utf-8")
    (docs / "FACT_SHEET.md").write_text(fact_sheet(quick, results_dir, runs_dir), encoding="utf-8")
    template = (ROOT / "docs" / "templates" / "VIVA_NOTES.template.md").read_text(encoding="utf-8")
    runs_text = "\n".join(f"- {s}" for s in what_runs_showed(summary))
    (docs / "VIVA_NOTES.md").write_text(template.replace("{{RUNS_SECTION}}", runs_text), encoding="utf-8")
    print("Docs written to", docs)
