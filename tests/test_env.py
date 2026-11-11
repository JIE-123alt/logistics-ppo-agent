"""Tests for the environment. Run with:  pytest -q"""
import numpy as np
from gymnasium.utils.env_checker import check_env

from src.baselines import BaseStockPolicy, RandomPolicy
from src.env import SupplyChainEnv
from src.evaluate import evaluate
from src.utils import get_splits, load_yaml, with_changes

CFG = load_yaml("config/env.yaml")
SPLITS = get_splits(CFG)


def run_episode(cfg, actions, seed=5, scenario=1000):
    """Play a list of actions (cycled) and return the list of step infos."""
    env = SupplyChainEnv(cfg, [scenario])
    env.reset(seed=seed, options={"scenario_seed": scenario})
    infos, done, i = [], False, 0
    while not done:
        _, _, done, _, info = env.step(actions[i % len(actions)])
        infos.append(info)
        i += 1
    return infos, env


def test_gymnasium_check_env():
    env = SupplyChainEnv(CFG, SPLITS["train"])
    check_env(env, skip_render_check=True)


def test_same_seed_gives_same_episode():
    a, _ = run_episode(CFG, [3, 6, 0, 5])
    b, _ = run_episode(CFG, [3, 6, 0, 5])
    assert [x["cost"] for x in a] == [x["cost"] for x in b]


def test_demand_does_not_depend_on_actions():
    """Common random numbers: two different policies face the same demand."""
    a, _ = run_episode(CFG, [0])
    b, _ = run_episode(CFG, [7])
    assert [x["demand"] for x in a] == [x["demand"] for x in b]


def test_cost_arithmetic_without_orders():
    """With no orders, week-1 cost = holding on leftover stock + lost-sale cost."""
    infos, _ = run_episode(CFG, [0])
    d = infos[0]["demand"]
    sold = min(200, d)
    expected = 1.0 * (200 - sold) + 5.0 * (d - sold)
    assert abs(infos[0]["cost"] - expected) < 1e-9


def test_episode_has_52_steps():
    infos, _ = run_episode(CFG, [4])
    assert len(infos) == 52


def test_fallback_after_three_failed_orders():
    """If the order system always fails, the 4th try is replaced by the fallback order."""
    cfg = with_changes(CFG, "uncertainty", api_failure_prob=1.0, disruption_prob=0.0)
    infos, env = run_episode(cfg, [7])          # always: order 200 from Supplier B
    assert [i["order_failed"] for i in infos[:3]] == [True, True, True]
    assert infos[3]["fallback_used"] and not infos[3]["order_failed"]
    assert env.stats["loop_trap"] >= 1          # the agent was still trying a 4th time


def test_splits_do_not_overlap():
    train, val, test = set(SPLITS["train"]), set(SPLITS["val"]), set(SPLITS["test"])
    assert not (train & val) and not (train & test) and not (val & test)
    assert (len(train), len(val), len(test)) == (50, 20, 20)


def test_evaluate_is_repeatable():
    policy = BaseStockPolicy(CFG, 3, 0.6)
    one = evaluate(policy, CFG, SPLITS["val"][:3], 2)
    two = evaluate(policy, CFG, SPLITS["val"][:3], 2)
    assert np.allclose(one["total_cost"], two["total_cost"])
    r1 = evaluate(RandomPolicy(), CFG, SPLITS["val"][:3], 2)
    r2 = evaluate(RandomPolicy(), CFG, SPLITS["val"][:3], 2)
    assert np.allclose(r1["total_cost"], r2["total_cost"])