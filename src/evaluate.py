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


def ppo_policy(model):
    """Wrap a Stable-Baselines3 model as a deterministic policy."""
    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return policy


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
