"""A tiny end-to-end check: PPO trains for a few steps, saves a model, and evaluate() can use it."""
import shutil

from stable_baselines3 import PPO

from src.agent import decide
from src.env import make_obs
from src.evaluate import evaluate, ppo_policy
from src.train import train_one
from src.utils import ROOT, get_splits, load_run, load_yaml


def test_tiny_training_run_and_evaluation():
    cfg = load_yaml("config/env.yaml")
    run_cfg = load_run("run_c_final")
    run_cfg["run_name"] = "pytest_tiny"
    run_cfg["train"]["quick_timesteps"] = 2048          # one PPO update only
    run_cfg["eval"]["quick_every_steps"] = 2048
    name = train_one(run_cfg, cfg, seed=0, quick=True)
    run_dir = ROOT / "runs_quick" / name
    assert (run_dir / "best_model.zip").exists() and (run_dir / "eval_log.csv").exists()
    model = PPO.load(run_dir / "best_model", device="cpu")
    df = evaluate(ppo_policy(model), cfg, get_splits(cfg)["val"][:2], 1)
    assert len(df) == 2 and (df["total_cost"] > 0).all()
    answer = decide(model, cfg, make_obs(cfg, stock=120, transit_a=0, transit_b=100, forecast=95, rel_signal=0.8, week=10))
    assert 0 < answer["confidence"] <= 1 and len(answer["probabilities"]) == 8 and answer["decision"]
    shutil.rmtree(run_dir)
    (ROOT / "results_quick" / f"run_info_{name}.json").unlink()
