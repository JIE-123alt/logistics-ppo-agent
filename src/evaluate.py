"""ONE evaluate function used for PPO, the baselines, train, validation and test.

A policy is anything you can call as  policy(obs) -> action  (an int from 0 to 7).
The same scenario seeds and the same episode seeds are used for every policy, so the
comparison is fair (every policy sees the same demand and the same luck).
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.env import SupplyChainEnv

log = logging.getLogger("evaluate")


def evaluate(policy, cfg, scenario_seeds, episodes_per_scenario=5, ep_offset=0):
    """Run the policy on every scenario. Returns one row per episode (a DataFrame)."""
    env = SupplyChainEnv(cfg, scenario_seeds)
    rows = []
    for s in scenario_seeds:
        for k in range(episodes_per_scenario):
            episode_seed = s * 100 + ep_offset + k             # same for every policy
            if hasattr(policy, "reset"):
                policy.reset(episode_seed)                      # only the random policy needs this
            obs, _ = env.reset(seed=episode_seed, options={"scenario_seed": s})
            done = False
            while not done:
                obs, _, done, _, _ = env.step(policy(obs))
            row = env.episode_stats()
            row["episode"] = k
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(df):
    """Average the metrics of the episodes (plus the standard deviation of total cost)."""
    out = df.drop(columns=["scenario", "episode"]).mean().to_dict()
    out["total_cost_std"] = float(df["total_cost"].std())
    return out


def ppo_policy(model):
    """Wrap a Stable-Baselines3 model as a deterministic policy."""
    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return policy


def cost_of(policy, cfg, seeds, episodes_per_scenario=5):
    """Mean total cost of a policy on a list of scenarios."""
    return float(evaluate(policy, cfg, seeds, episodes_per_scenario)["total_cost"].mean())


def use_test_set(results_dir, who):
    """Call this before every test evaluation. It records the use and warns on a second one."""
    lock = Path(results_dir) / "test_lock.json"
    record = json.loads(lock.read_text(encoding="utf-8")) if lock.exists() else {"uses": []}
    if record["uses"]:
        log.warning("WARNING: the TEST set was already used (%d time(s) before). "
                    "Using it again breaks the rule 'test only once'.", len(record["uses"]))
    record["uses"].append({"time": datetime.now().isoformat(timespec="seconds"), "who": who})
    record["count"] = len(record["uses"])
    lock.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record["count"]
