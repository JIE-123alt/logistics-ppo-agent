"""Run the whole pipeline.   python run_all.py --quick   (a few minutes, smoke test)
                            python run_all.py           (about 60-90 minutes, set by --budget-minutes)
                            python run_all.py --only plots   (one stage on its own)

Stages, in order: speed (full run only), baselines, train (runs A, B, C), eval, stress, plots, cards, docs.
The last three only make reports (src/plots.py, plan_cards.py, make_docs.py); they are skipped if those files are missing.
Quick mode writes to results_quick/ and runs_quick/ and starts clean.
"""
import argparse
import importlib
import shutil
import time

from src import baselines, final_eval, stress_test
from src.train import speed_test, train_one
from src.utils import ROOT, RUN_NAMES, get_dirs, load_json, load_run, load_yaml

STAGES = ["speed", "baselines", "train", "eval", "stress", "plots", "cards", "docs"]
REPORT = {"plots": "plots", "cards": "plan_cards", "docs": "make_docs"}   # stage -> file in src/


def report_stage(stage, quick):
    """Plots, plan cards and docs only turn the results into reports. They can be left out of the repo."""
    module_name = f"src.{REPORT[stage]}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:                  # a real missing package, not our optional file
            raise
        print(f"skipped: src/{REPORT[stage]}.py is not in this repo")
        return
    module.run(quick)


def train_all(quick, scale):
    cfg = load_yaml("config/env.yaml")
    for name in RUN_NAMES:
        run_cfg = load_run(name)
        run_scale = 1.0 if name == "run_a_underfit" else scale        # run A is short on purpose
        for seed in run_cfg["seeds"]:
            print(f"--- {name} seed {seed}")
            train_one(run_cfg, cfg, seed, quick, run_scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", choices=STAGES, help="run just this stage")
    parser.add_argument("--budget-minutes", type=float, default=75)
    parser.add_argument("--skip-speed-test", action="store_true")
    args = parser.parse_args()
    quick, started = args.quick, time.time()
    todo = [args.only] if args.only else STAGES

    if quick and not args.only:                        # quick mode is a smoke test: start clean
        for folder in ["results_quick", "runs_quick"]:
            shutil.rmtree(ROOT / folder, ignore_errors=True)

    scale = 1.0                                        # how much to scale the training steps
    if "speed" in todo and not quick and not args.skip_speed_test:
        print("\n===== speed =====")
        scale = speed_test(args.budget_minutes)
    elif args.only and not quick:                      # e.g. --only train: reuse the last speed test
        saved = get_dirs(False)[0] / "speed_test.json"
        scale = load_json(saved)["timesteps_scale"] if saved.exists() else 1.0

    steps = {"baselines": lambda: baselines.run(quick), "train": lambda: train_all(quick, scale),
             "eval": lambda: final_eval.run(quick), "stress": lambda: stress_test.run(quick)}
    steps.update({stage: (lambda stage=stage: report_stage(stage, quick)) for stage in REPORT})
    for stage in todo:
        if stage != "speed":
            print(f"\n===== {stage} =====")
            begin = time.time()
            steps[stage]()
            print(f"----- {stage}: {time.time() - begin:.0f} s -----")
    print(f"\nALL DONE in {(time.time() - started) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
