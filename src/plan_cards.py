"""Plan cards: 3 example decisions of the final model, with confidence, alternative and risk flag.

The card text is an APPROXIMATION: simple rules we wrote to describe the situation. It is not
what the neural network 'thought'. Examples come from a VALIDATION scenario (test stays untouched).
Run:  python run_all.py --only cards [--quick]
"""

import numpy as np
from stable_baselines3 import PPO

from src.agent import action_text, decide
from src.env import SupplyChainEnv, unpack_obs
from src.evaluate import ppo_policy
from src.utils import get_dirs, get_docs_dir, get_splits, load_json, load_yaml, save_json


def play_episode(model, cfg, scenario_seed):
    """Play one validation episode and remember every (week, observation)."""
    env = SupplyChainEnv(cfg, [scenario_seed])
    obs, _ = env.reset(seed=scenario_seed * 100, options={"scenario_seed": scenario_seed})
    policy, states, done = ppo_policy(model), [], False
    while not done:
        states.append((env.t, obs))
        obs, _, done, _, _ = env.step(policy(obs))
    return states


def run(quick):
    cfg = load_yaml("config/env.yaml")
    results_dir, runs_dir = get_dirs(quick)
    final = load_json(results_dir / "final_model.json")
    model = PPO.load(runs_dir / f"{final['run']}_seed{final['seed']}" / "best_model", device="cpu")
    scenario = get_splits(cfg)["val"][0]
    states = play_episode(model, cfg, scenario)
    unpacked = [unpack_obs(o, cfg) for _, o in states]
    stock_ratio = [u["inventory"] / max(u["forecast"], 1) for u in unpacked]
    picks = [("lowest stock compared with the forecast", int(np.argmin(stock_ratio))),
             ("lowest reliability signal for Supplier A", int(np.argmin([u["rel_signal"] for u in unpacked]))),
             ("a routine mid-year week", 26)]
    cards, used = [], set()
    for title, i in picks:
        while i in used:
            i += 1                                           # keep the three weeks different
        used.add(i)
        week, obs = states[i]
        answer = decide(model, cfg, obs)
        o = unpacked[i]
        cards.append({
            "example": title, "scenario_seed": scenario, "week": int(week),
            "situation": {k: round(float(v), 2) for k, v in o.items()},
            "chosen_action": answer["decision"], "confidence": round(answer["confidence"], 3),
            "main_alternative": answer["alternative"], "alternative_probability": round(answer["alternative_probability"], 3),
            "action_probabilities": {action_text(a, cfg) + f" [action {a}]": round(float(p), 3) for a, p in enumerate(answer["probabilities"])},
            "risk_flag": answer["risk_flag"],
        })
    out = {"note": "Explanations are approximations written by simple rules, not the network's reasoning. "
                   "Confidence = the probability the policy gives to its top action.",
           "model": f"{final['run']}_seed{final['seed']} (best checkpoint)", "cards": cards}
    save_json(out, get_docs_dir(quick) / "PLAN_CARD_SAMPLES.json")
    for c in cards:
        print(f"week {c['week']}: {c['chosen_action']} (confidence {c['confidence']}) | risk: {c['risk_flag']}")
