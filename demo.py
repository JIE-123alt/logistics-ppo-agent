"""Live demo of the trained agent, shown in the terminal. Put this file in the project root (next to run_all.py).

    python demo.py                    final model, first validation scenario, one week every 0.3 seconds
    python demo.py --delay 0          no waiting
    python demo.py --scenario 2005    another validation scenario (2000 to 2019)
    python demo.py --seed 2           a different Run C seed (default: the one chosen on validation)
    python demo.py --quick            use the smoke-test models in runs_quick/ (to rehearse)

  REAL DATASET (made first with prepare_real_datasets.py):
    python demo.py --real data/processed/real_dataset.csv             52 weeks of real demand
    python demo.py --real data/processed/real_dataset.csv --start 20  another 52-week window of the file

Everything shown is calculated live from the saved model. The synthetic demo uses validation scenarios
only, so the test set stays untouched. The real-dataset demo never touches the test set either: the agent
is NOT retrained on real data, it only meets it for the first time.
Needs the trained model from:  python run_all.py
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.agent import decide
from src.baselines import RandomPolicy, load_base_stock
from src.env import SupplyChainEnv, make_obs, unpack_obs
from src.evaluate import evaluate, ppo_policy
from src.utils import get_dirs, get_splits, load_json, load_yaml

LINE = "=" * 112


class RealEnv(SupplyChainEnv):
    """The normal environment, but the demand is a real series instead of a synthetic scenario.
    Costs, delivery rules, supplier failures and disruptions are exactly those of SupplyChainEnv (still simulated)."""

    def __init__(self, cfg, scenario_seeds, real):
        super().__init__(cfg, scenario_seeds)
        self.real = real

    def _draw_randomness(self):
        # the forecast is built from a smoothed version of the real series (like a forecaster would)
        self.scn = {"expected": self.real["expected"], "noise": 0.0, "base": float(self.real["actual"].mean())}
        super()._draw_randomness()                       # lead times, failures, disruptions, forecast noise
        self.demand = self.real["actual"].astype(float)  # what customers really wanted


def make_env(cfg, scenario, real):
    return RealEnv(cfg, [scenario], real) if real else SupplyChainEnv(cfg, [scenario])


def load_model(results_dir, runs_dir, seed):
    if seed is None:                                           # the seed chosen on validation
        marker = results_dir / "final_model.json"
        seed = load_json(marker)["seed"] if marker.exists() else 0
    path = runs_dir / f"run_c_final_seed{seed}" / "best_model.zip"
    if not path.exists():
        sys.exit(f"No trained model found at {path}\nTrain first with:  python run_all.py")
    return PPO.load(path, device="cpu"), seed


# ---------- real dataset ----------

def load_real_series(path, cfg):
    """Read the weekly file and rescale it to the size the agent trained on (average week = 100 units)."""
    if not Path(path).exists():
        sys.exit(f"File not found: {path}\nMake it first with:  python prepare_real_datasets.py --help")
    df = pd.read_csv(path)
    column = "demand" if "demand" in df.columns else df.columns[-1]
    series = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
    need = cfg["weeks"] + 1                                    # +1: the forecast looks one week ahead
    if len(series) < need:
        sys.exit(f"The file has {len(series)} weeks, the demo needs at least {need}.")
    d = cfg["demand"]
    target = (d["base_min"] + d["base_max"]) / 2
    return series * (target / series.mean()), target / series.mean(), target


def cut_window(scaled, start, cfg):
    weeks = cfg["weeks"]
    if start < 0 or start + weeks + 1 > len(scaled):
        sys.exit(f"--start {start} does not fit: allowed values are 0 to {len(scaled) - weeks - 1}.")
    window = scaled[start:start + weeks + 1]
    expected = pd.Series(window).rolling(5, center=True, min_periods=1).mean().to_numpy()
    return {"expected": expected, "actual": np.rint(window[:weeks])}


def print_profile(path, real, start, scale, target, n_weeks_file, weeks):
    actual, expected = real["actual"], real["expected"][:weeks]
    noise = float(np.std(actual - expected) / max(actual.mean(), 1e-9))
    peak = float(actual.max() / max(actual.mean(), 1e-9))
    print("REAL DATASET PROFILE")
    print(f"  file: {path}  ({n_weeks_file} weeks; this demo uses weeks {start} to {start + weeks - 1})")
    print(f"  rescaled by x{scale:.4f} so the average week = {target:.0f} units (the size the agent trained on)")
    print(f"  demand after scaling: mean {actual.mean():.0f}, min {actual.min():.0f}, max {actual.max():.0f}; "
          f"busiest week = {peak:.1f} x the average (training scenarios: about 1.1 to 1.4 x)")
    print(f"  week-to-week noise: {noise:.0%} of the mean (training scenarios: about 5% to 20%)")
    if noise > 0.25 or peak > 1.6:
        print("  WARNING: this data is rougher than anything the agent trained on, so expect weaker results.")
    print()


def run_episode(policy, cfg, real, seed):
    """One 52-week episode on the real window, without printing."""
    env = RealEnv(cfg, [0], real)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    obs, _ = env.reset(seed=seed, options={"scenario_seed": 0})
    done = False
    while not done:
        obs, _, done, _, _ = env.step(policy(obs))
    return env.episode_stats()


# ---------- one scenario ----------

def play(model, cfg, scenario, delay, real=None, seed=None):
    """The agent plays one 52-week scenario. Every decision is printed as it happens."""
    env = make_env(cfg, scenario, real)
    obs, _ = env.reset(seed=scenario * 100 if seed is None else seed, options={"scenario_seed": scenario})
    print("week | stock | on the way A / B | forecast | A signal | agent decides                    | sure |"
          " demand | lost |  cost | note")
    print("-" * 112)
    orders, done = {"A": 0, "B": 0, "no order": 0}, False
    while not done:
        seen = unpack_obs(obs, cfg)
        answer = decide(model, cfg, obs)
        obs, _, done, _, info = env.step(answer["action"])
        orders[answer["supplier"] if answer["order_size"] else "no order"] += 1
        notes = [text for flag, text in [(info["lost"] > 0, "STOCKOUT"), (info["order_failed"], "ORDER FAILED"),
                                         (info["fallback_used"], "FALLBACK ORDER"),
                                         (seen["rel_signal"] < 0.5, "A looks unreliable")] if flag]
        print(f"{info['week'] + 1:>4} | {seen['inventory']:>5.0f} | {seen['transit_a']:>6.0f} / {seen['transit_b']:<6.0f}  |"
              f" {seen['forecast']:>8.0f} | {seen['rel_signal']:>8.2f} | {answer['decision']:<32} | {answer['confidence']:>4.0%} |"
              f" {info['demand']:>6.0f} | {info['lost']:>4.0f} | {info['cost']:>5.0f} | {', '.join(notes)}")
        time.sleep(delay)
    return env.episode_stats(), orders


def compare(model, cfg, scenario, agent_stats, results_dir, validation):
    """Same scenario and same luck for the agent, the tuned base-stock rule and a random policy."""
    others = {"base-stock rule": load_base_stock(cfg, results_dir), "random policy": RandomPolicy()}
    policies = {"agent (PPO)": ppo_policy(model), **others}
    print(f"\n{LINE}\nHOW GOOD IS IT?  (lower cost is better)\n{LINE}")
    print(f"{'':<17}| this scenario ({scenario}) {'':<10}| average of the {len(validation)} validation scenarios (3 episodes each)")
    print(f"{'':<17}| {'cost':>8} | {'fill rate':>9}      | {'cost':>8} | {'fill rate':>9}")
    for name, policy in policies.items():
        mine = agent_stats if name.startswith("agent") else evaluate(policy, cfg, [scenario], 1).iloc[0]
        average = evaluate(policy, cfg, validation, 3)
        print(f"{name:<17}| {mine['total_cost']:>8.0f} | {mine['fill_rate']:>9.2f}      | "
              f"{average['total_cost'].mean():>8.0f} | {average['fill_rate'].mean():>9.2f}")
    print("fill rate = share of customer demand that was served")


def compare_real(model, cfg, scaled, start, agent_stats, results_dir):
    """Same real window and same luck for the three policies, then the average over many windows of the file."""
    policies = {"agent (PPO)": ppo_policy(model), "base-stock rule": load_base_stock(cfg, results_dir),
                "random policy": RandomPolicy()}
    possible = len(scaled) - cfg["weeks"]                      # number of valid start weeks
    starts = list(range(0, possible, max(1, -(-possible // 20))))   # at most about 20 windows, spread over the file
    costs = {name: [] for name in policies}
    fills = {name: [] for name in policies}
    for s in starts:
        window = cut_window(scaled, s, cfg)
        for name, policy in policies.items():
            stats = run_episode(policy, cfg, window, s)
            costs[name].append(stats["total_cost"])
            fills[name].append(stats["fill_rate"])
    print(f"\n{LINE}\nHOW GOOD IS IT ON REAL DATA?  (lower cost is better)\n{LINE}")
    print(f"{'':<17}| this window (from week {start}) {'':<2}| average of {len(starts)} windows of the file")
    print(f"{'':<17}| {'cost':>8} | {'fill rate':>9}      | {'cost':>8} | {'fill rate':>9}")
    for name, policy in policies.items():
        mine = agent_stats if name.startswith("agent") else run_episode(policy, cfg, cut_window(scaled, start, cfg), start)
        print(f"{name:<17}| {mine['total_cost']:>8.0f} | {mine['fill_rate']:>9.2f}      | "
              f"{np.mean(costs[name]):>8.0f} | {np.mean(fills[name]):>9.2f}")
    wins = sum(a < b for a, b in zip(costs["agent (PPO)"], costs["base-stock rule"]))
    print(f"The agent was cheaper than the base-stock rule in {wins} of {len(starts)} windows.")
    print("fill rate = share of customer demand that was served. The windows overlap, so they are not independent tests.")


def ask_the_agent(model, cfg):
    """Does the agent react to the situation? Same question, different situations."""
    print(f"\n{LINE}\nASK THE AGENT: what should I order in these situations?\n{LINE}")
    base = dict(stock=200, transit_a=0, transit_b=0, forecast=100, rel_signal=0.8, week=10)
    situations = [("plenty of stock, more on the way", dict(base, stock=450, transit_a=100)),
                  ("low stock, nothing on the way", dict(base, stock=30)),
                  ("low stock, Supplier A looks unreliable", dict(base, stock=30, rel_signal=0.2)),
                  ("low stock, the last order failed", dict(base, stock=30, last_failed=1))]
    for title, situation in situations:
        answer = decide(model, cfg, make_obs(cfg, **situation))
        print(f"\n{title}\n  stock {situation['stock']}, forecast {situation['forecast']}, "
              f"A signal {situation['rel_signal']}")
        print(f"  -> {answer['decision']}  (sure: {answer['confidence']:.0%}; "
              f"second choice: {answer['alternative']} {answer['alternative_probability']:.0%})")
        print(f"     risk: {answer['risk_flag']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, help="scenario seed (default: first validation scenario)")
    parser.add_argument("--seed", type=int, help="Run C seed (default: the one chosen on validation)")
    parser.add_argument("--delay", type=float, default=0.3, help="seconds between weeks")
    parser.add_argument("--quick", action="store_true", help="use runs_quick/ and results_quick/")
    parser.add_argument("--real", help="weekly real dataset made by prepare_real_datasets.py (instead of a synthetic scenario)")
    parser.add_argument("--start", type=int, default=0, help="with --real: first week of the 52-week window")
    args = parser.parse_args()

    cfg = load_yaml("config/env.yaml")
    results_dir, runs_dir = get_dirs(args.quick)
    splits = get_splits(cfg)
    real = scaled = None
    if args.real:
        scaled, scale, target = load_real_series(args.real, cfg)
        real = cut_window(scaled, args.start, cfg)
        scenario, episode_seed = 0, args.start                  # scenario 0 is only a placeholder name
    else:
        scenario = args.scenario if args.scenario is not None else splits["val"][0]
        episode_seed = None
        if scenario in splits["test"]:
            sys.exit("That is a test scenario. The test set is used once only, so pick a validation scenario (2000 to 2019).")
    model, seed = load_model(results_dir, runs_dir, args.seed)

    print(f"{LINE}\nSUPPLY CHAIN AGENT DEMO\n{LINE}")
    sa, sb, costs = cfg["supplier_a"], cfg["supplier_b"], cfg["costs"]
    sizes = cfg["order_sizes"]
    print(f"Each week the agent decides how much to order ({', '.join(map(str, sizes[:-1]))} or {sizes[-1]} units) and from which supplier.")
    print(f"Supplier A: cost {sa['unit_cost']} per unit, {sa['lead_min']}-{sa['lead_max']} weeks, on time {sa['reliability']:.0%}.  "
          f"Supplier B: cost {sb['unit_cost']} per unit, {sb['lead_min']}-{sb['lead_max']} weeks, on time {sb['reliability']:.0%}.")
    print(f"Lost sales cost {costs['lost_sale']} per unit, holding stock costs {costs['holding']} per unit. "
          + ("Demand is REAL data; delays, disruptions and order failures are still simulated." if real else
             "Demand, delays, disruptions and order failures are random."))
    if real:
        print(f"Model: PPO, Run C seed {seed}, trained on synthetic demand only (not retrained on this data).\n")
        print_profile(args.real, real, args.start, scale, target, len(scaled), cfg["weeks"])
    else:
        print(f"Model: PPO, Run C seed {seed}.  Scenario {scenario} (a validation scenario the agent never trained on).\n")

    stats, orders = play(model, cfg, scenario, args.delay, real, episode_seed)
    print(f"\n{LINE}\nRESULT OF THIS {'REAL-DATA WINDOW' if real else 'SCENARIO'}\n{LINE}")
    print(f"total cost {stats['total_cost']:.0f} | fill rate {stats['fill_rate']:.2f} | "
          f"stockout weeks {stats['stockout_freq'] * cfg['weeks']:.0f} of {cfg['weeks']} | "
          f"orders: {orders['A']} from A, {orders['B']} from B, {orders['no order']} weeks with no order")
    if real:
        compare_real(model, cfg, scaled, args.start, stats, results_dir)
        print("\nNOTE: the agent and the base-stock rule were both set up on synthetic demand. This shows how they cope with")
        print("data they have never seen. It is not a retrained model, and the supply side is still simulated.")
    else:
        compare(model, cfg, scenario, stats, results_dir, splits["val"])
    ask_the_agent(model, cfg)


if __name__ == "__main__":
    main()