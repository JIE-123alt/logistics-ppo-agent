"""A tiny end-to-end check: PPO trains for a few steps, saves a model, and evaluate() can use it."""
import shutil

from stable_baselines3 import PPO

from src.evaluate import evaluate, ppo_policy
from src.train import train_one
from src.utils import ROOT, get_splits, load_yaml


def test_tiny_training_run_and_evaluation():
    cfg = load_yaml("config/env.yaml")
    run_cfg = load_yaml("config/run_c_final.yaml")
    run_cfg["run_name"] = "pytest_tiny"
    run_cfg["train"]["quick_timesteps"] = 2048          # one PPO update only
    run_cfg["eval"]["quick_every_steps"] = 2048
    name = train_one(run_cfg, cfg, seed=0, quick=True)
    run_dir = ROOT / "runs_quick" / name
    assert (run_dir / "best_model.zip").exists() and (run_dir / "eval_log.csv").exists()
    model = PPO.load(run_dir / "best_model", device="cpu")
    df = evaluate(ppo_policy(model), cfg, get_splits(cfg)["val"][:2], 1)
    assert len(df) == 2 and (df["total_cost"] > 0).all()
    shutil.rmtree(run_dir)
    (ROOT / "results_quick" / f"run_info_{name}.json").unlink()