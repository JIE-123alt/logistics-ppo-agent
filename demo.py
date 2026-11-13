"""Live demo: watch the trained agent run one 52-week episode, week by week.

Run from the project folder (needs results/final_model.json and the model in runs/):

  python demo.py                                   # a TEST scenario: PPO vs base-stock vs random
  python demo.py --scenario 3005 --weeks 52        # another scenario, print all 52 weeks
  python demo.py --pause                           # press Enter to move to the next week
  python demo.py --both                            # also print the base-stock rule's weeks
  python demo.py --real data/processed/real_dataset.csv           # REAL dataset (weekly demand)
  python demo.py --real data/processed/real_dataset.csv --start 8 # another 52-week window

It only reads files. Nothing in results/ or runs/ is changed, and the test-set lock is not touched.
A real dataset is a zero-shot test: the agent was trained on synthetic demand only and is NOT retrained.
"""
import argparse
import time

import numpy as np
import pandas as pd

from src.baselines import RandomPolicy, load_base_stock
from src.env import N_SIZES, SupplyChainEnv, unpack_obs
from src.evaluate import ppo_policy
from src.utils import ROOT, get_splits, load_json, load_yaml

SUPPLIER = "AB"
A_STATUS = {0: "ok", 1: "delayed", 2: "DOWN"}


class DemoEnv(SupplyChainEnv):
    """The normal environment, plus (1) a note of the order placed this week and (2) an optional
    real dataset (weekly demand series). Costs, delivery rules and randomness are exactly those of SupplyChainEnv."""

    def __init__(self, cfg, scenario_seeds, real=None):
        super().__init__(cfg, scenario_seeds)
        self.real, self.last_order = real, None

    def _place_order(self, t, supplier, qty):
        super()._place_order(t, supplier, qty)
        self.last_order = {"supplier": supplier, "qty": qty, "arrives": self.pipeline[-1][0]}

    def _draw_randomness(self):
        if self.real is not None:                      # forecast is built from the smoothed real series
            self.scn = {"expected": self.real["expected"], "noise": 0.0, "base": float(self.real["actual"].mean())}
        super()._draw_randomness()                     # lead times, failures, disruptions, forecast noise
        if self.real is not None:                      # the demand the customers really had
            self.demand = self.real["actual"].astype(float)


# ---------- loading ----------

def load_agent():
    from stable_baselines3 import PPO                  # imported here so --help works without it
    final = load_json(ROOT / "results" / "final_model.json")
    path = ROOT / "runs" / f"{final['run']}_seed{final['seed']}" / "best_model"
    if not path.with_suffix(".zip").exists():
        raise SystemExit(f"Model not found: {path}.zip  (run the training first)")
    model = PPO.load(path, device="cpu")
    return f"PPO ({final['run']}, seed {final['seed']})", ppo_policy(model)


def load_real(path, cfg, start):
    """Read a weekly real-dataset file, rescale it to the size the agent trained on, cut one 52-week window."""
    df = pd.read_csv(path)
    column = "demand" if "demand" in df.columns else df.columns[-1]
    series = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
    weeks, need = cfg["weeks"], cfg["weeks"] + 1       # +1: the forecast looks one week ahead
    if start < 0 or len(series) < start + need:
        raise SystemExit(f"Need {need} weeks from --start {start}, the file has {len(series)} weeks "
                         f"(allowed --start: 0 to {max(0, len(series) - need)}).")
    d = cfg["demand"]
    target_mean = (d["base_min"] + d["base_max"]) / 2  # the agent trained on demand averaging about 100
    scale = target_mean / series.mean()
    window = series[start:start + need] * scale
    expected = pd.Series(window).rolling(5, center=True, min_periods=1).mean().to_numpy()
    actual = np.rint(window[:weeks])
    noise_share = float(np.std(actual - expected[:weeks]) / max(actual.mean(), 1e-9))
    print("REAL DATASET PROFILE")
    print(f"  file: {path}  ({len(series)} weeks, using weeks {start} to {start + weeks})")
    print(f"  rescaled by x{scale:.4f} so the average week = {target_mean:.0f} units (the size the agent trained on)")
    print(f"  weekly demand after scaling: mean {actual.mean():.0f}, min {actual.min():.0f}, max {actual.max():.0f}")
    print(f"  week-to-week noise: {noise_share:.0%} of the mean (training scenarios: about 5% to 20%)")
    if noise_share > 0.25:
        print("  WARNING: much noisier than anything the agent trained on, so expect weaker results.")
    print()
    return {"expected": expected, "actual": actual}


# ---------- one episode ----------

def decision_text(env, info, action):
    """Plain-English description of what happened to this week's order."""
    wanted_qty, wanted_sup = env.sizes[action % N_SIZES], SUPPLIER[action // N_SIZES]
    order = env.last_order
    if info["fallback_used"]:
        if order:
            return f"FALLBACK: safety rule orders {order['qty']} from B (arrives wk {order['arrives']})"
        return "FALLBACK: safety rule triggered, nothing needed"
    if info["order_failed"]:
        return f"ORDER FAILED: {wanted_qty} from {wanted_sup} was rejected"
    if order:
        return f"order {order['qty']} from {SUPPLIER[order['supplier']]}, arrives wk {order['arrives']}"
    return "no order"


def play(name, policy, cfg, scenario, seed, real, show_weeks, pause, delay):
    """Run one 52-week episode. Same scenario and same seed for every policy = same demand and luck."""
    env = DemoEnv(cfg, [scenario], real)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    obs, _ = env.reset(seed=seed, options={"scenario_seed": scenario})
    if show_weeks:
        print(f"=== {name} ===")
        print(f"{'Wk':>3} {'Fcst':>5} {'SigA':>5} {'A(truth)':>8} {'Stock':>6} {'OnWay':>6} "
              f"{'Demand':>6} {'Lost':>5} {'Cost':>7}  Decision")
    done = False
    while not done:
        o, t = unpack_obs(obs, cfg), env.t
        a_truth = A_STATUS[int(env.disruption[t])]
        action = int(policy(obs))
        env.last_order = None
        obs, _, done, _, info = env.step(action)
        if t < show_weeks:
            print(f"{t:>3} {o['forecast']:>5.0f} {o['rel_signal']:>5.2f} {a_truth:>8} {o['inventory']:>6.0f} "
                  f"{o['transit_a'] + o['transit_b']:>6.0f} {info['demand']:>6.0f} {info['lost']:>5.0f} "
                  f"{info['cost']:>7.0f}  {decision_text(env, info, action)}")
            if pause:
                input()
            elif delay:
                time.sleep(delay)
    if show_weeks:
        shown = min(show_weeks, cfg["weeks"])
        if shown < cfg["weeks"]:
            print(f"... (first {shown} of {cfg['weeks']} weeks shown; the totals below cover all {cfg['weeks']})")
        print()
    stats = env.episode_stats()
    stats["policy"] = name
    stats["stockout_weeks"] = round(stats["stockout_freq"] * cfg["weeks"])
    stats["fallback_weeks"] = round(stats["fallback_rate"] * cfg["weeks"])
    return stats


def print_legend():
    print("HOW TO READ THE TABLE (one row = one week)")
    print("  Fcst    next week's forecast the agent sees (noisy)")
    print("  SigA    reliability signal of cheap Supplier A (about 0.8 normal, low = trouble)")
    print("  A(truth) what is really happening to A: ok / delayed / DOWN (the agent cannot see this)")
    print("  Stock   units on hand at the start of the week;  OnWay = units ordered, not yet arrived")
    print("  Demand  what customers wanted;  Lost = demand it could not serve")
    print("  Cost    holding + lost sales + order fee + cost of goods that arrive next week")
    print("  Decision what the agent chose (A = cheap but slow/unreliable, B = dearer but fast/reliable)")
    print()


def print_summary(results, weeks):
    df = pd.DataFrame(results)[["policy", "total_cost", "fill_rate", "stockout_weeks",
                                "failed_orders", "fallback_weeks"]]
    df["fill_rate"] = (df["fill_rate"] * 100).round(1).astype(str) + "%"
    df["total_cost"] = df["total_cost"].round(0).astype(int)
    df.columns = ["Policy", f"Total cost ({weeks} wks)", "Demand served", "Stockout wks",
                  "Failed orders", "Fallback wks"]
    print("RESULT OF THIS EPISODE (lower cost is better)")
    print(df.to_string(index=False))
    cost = {r["policy"]: r["total_cost"] for r in results}
    agent = next(k for k in cost if k.startswith("PPO"))
    print()
    for other, label in [("random", "the random policy"), ("base-stock (tuned rule)", "the tuned base-stock rule")]:
        change = (cost[other] - cost[agent]) / cost[other] * 100
        word = "cheaper" if change >= 0 else "more expensive"
        print(f"  The agent was {abs(change):.1f}% {word} than {label} on this episode.")
    print("\nThis is ONE episode, so it is an example, not proof. The averages over all validation and")
    print("test scenarios (3 seeds, 5 episodes each) are in results/run_summary.csv.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", type=int, help="scenario seed (default: first test scenario)")
    p.add_argument("--seed", type=int, help="episode seed for the random events (default: scenario x 100)")
    p.add_argument("--weeks", type=int, default=16, help="how many weeks to print (default 16, max 52)")
    p.add_argument("--pause", action="store_true", help="wait for Enter after each week")
    p.add_argument("--delay", type=float, default=0.0, help="seconds to wait after each printed week")
    p.add_argument("--both", action="store_true", help="also print the base-stock rule's weeks")
    p.add_argument("--real", help="weekly dataset file made by prepare_real_datasets.py")
    p.add_argument("--start", type=int, default=0, help="first week of the 52-week window in the real file")
    args = p.parse_args()

    cfg = load_yaml("config/env.yaml")
    weeks_shown = max(0, min(args.weeks, cfg["weeks"]))
    real = load_real(args.real, cfg, args.start) if args.real else None
    scenario = args.scenario if args.scenario is not None else get_splits(cfg)["test"][0]
    seed = args.seed if args.seed is not None else (0 if real else scenario * 100)

    agent_name, agent = load_agent()
    policies = [(agent_name, agent, weeks_shown),
                ("base-stock (tuned rule)", load_base_stock(cfg, ROOT / "results"), weeks_shown if args.both else 0),
                ("random", RandomPolicy(), 0)]

    source = f"REAL dataset ({args.real}, window starting week {args.start})" if real else f"synthetic scenario {scenario}"
    print(f"DEMO: {source}, episode seed {seed}, {cfg['weeks']} weeks. Same demand and same luck for every policy.\n")
    if weeks_shown:
        print_legend()
    results = [play(name, policy, cfg, scenario, seed, real, shown, args.pause, args.delay)
               for name, policy, shown in policies]
    print_summary(results, cfg["weeks"])
    if real:
        print("\nNOTE: the agent was trained on synthetic demand only. This is how it copes with data it has")
        print("never seen, not a retrained model, and the base-stock rule was also tuned on synthetic data.")


if __name__ == "__main__":
    main()