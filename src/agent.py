"""The agent's answers in plain words.

decide() turns ONE situation into an answer: what to order and from which supplier, how sure the agent
is, the main alternative and a risk flag. run() lets the final model play the train and validation
scenarios, saves every weekly decision to results/decisions.csv and prints one scenario in the terminal.
The safety fallback after 3 failed orders is applied by the environment, so decide() gives the model's own choice.
Run:  python run_all.py --only decisions [--quick]
"""
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from src.env import N_SIZES, SupplyChainEnv, unpack_obs
from src.utils import get_dirs, get_splits, load_json, load_yaml

TRAIN_SCENARIOS_SHOWN = 5          # first 5 train scenarios + all validation scenarios (never the test set)


def action_text(action, cfg):
    qty = cfg["order_sizes"][action % N_SIZES]
    return "Order nothing this week" if qty == 0 else f"Order {qty} units from Supplier {'AB'[action // N_SIZES]}"


def risk_flags(o, confidence):
    flags = []
    if o["inventory"] + o["transit_a"] + o["transit_b"] < o["forecast"]:
        flags.append("stock on hand and on the way is below next week's forecast (stockout risk)")
    if o["rel_signal"] < 0.5:
        flags.append(f"Supplier A looks unreliable (signal {o['rel_signal']:.2f})")
    if o["last_failed"] > 0.5:
        flags.append("the last order failed (order system problem?)")
    if confidence < 0.5:
        flags.append(f"the agent is not very sure (top action only {confidence:.0%})")
    return "; ".join(flags) if flags else "none flagged"


def decide(model, cfg, obs):
    """Answer for one observation (the 7 normalised numbers; make_obs() builds them from real numbers)."""
    with torch.no_grad():
        probs = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0]).distribution.probs.cpu().numpy()[0]
    best, second = (int(a) for a in np.argsort(-probs)[:2])
    size = cfg["order_sizes"][best % N_SIZES]
    return {"action": best, "decision": action_text(best, cfg), "order_size": size,
            "supplier": "AB"[best // N_SIZES] if size > 0 else "-",
            "confidence": float(probs[best]), "alternative": action_text(second, cfg),
            "alternative_probability": float(probs[second]),
            "risk_flag": risk_flags(unpack_obs(obs, cfg), float(probs[best])), "probabilities": probs}


def run(quick):
    cfg = load_yaml("config/env.yaml")
    results_dir, runs_dir = get_dirs(quick)
    final = load_json(results_dir / "final_model.json")
    model = PPO.load(runs_dir / f"{final['run']}_seed{final['seed']}" / "best_model", device="cpu")
    splits = get_splits(cfg)
    chosen = [("train", s) for s in splits["train"][:TRAIN_SCENARIOS_SHOWN]] + [("val", s) for s in splits["val"]]
    rows = []
    for split, scenario in chosen:
        env = SupplyChainEnv(cfg, [scenario])
        obs, _ = env.reset(seed=scenario * 100, options={"scenario_seed": scenario})   # same episode as evaluation
        done = False
        while not done:
            seen, week = unpack_obs(obs, cfg), env.t
            answer = decide(model, cfg, obs)
            obs, _, done, _, info = env.step(answer["action"])
            rows.append({"split": split, "scenario": scenario, "week": week, "stock": seen["inventory"],
                         "on_way_A": seen["transit_a"], "on_way_B": seen["transit_b"], "forecast": seen["forecast"],
                         "A_signal": seen["rel_signal"], "last_order_failed": int(seen["last_failed"]),
                         "decision": answer["decision"], "confidence": answer["confidence"],
                         "alternative": answer["alternative"], "risk_flag": answer["risk_flag"],
                         "fallback_used": int(info["fallback_used"]), "order_failed": int(info["order_failed"]),
                         "demand": info["demand"], "lost": info["lost"], "cost": info["cost"]})
    table = pd.DataFrame(rows).round(2)
    table.to_csv(results_dir / "decisions.csv", index=False)
    first = splits["val"][0]
    show = table[(table["split"] == "val") & (table["scenario"] == first)].copy()
    show["confidence"] = (show["confidence"] * 100).round().astype(int).astype(str) + "%"
    print(f"Decisions of {final['run']} seed {final['seed']} on validation scenario {first}:")
    print(show[["week", "stock", "on_way_A", "on_way_B", "forecast", "A_signal", "decision", "confidence",
                "demand", "lost", "cost"]].to_string(index=False))
    print(f"All {len(table)} weekly decisions saved to {results_dir / 'decisions.csv'}")
