"""Focused tests for held-out real-demand evaluation."""

import json
import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from evaluate_real import load_test_windows
from src.real_env import RealWindowEnv, make_windows
from src.utils import load_yaml


CFG = load_yaml("config/env.yaml")


def panel(values):
    return pd.DataFrame({"sku": np.asarray(values, dtype=float)})


def test_real_windows_use_causal_forecast_and_calibration_scale():
    values = np.arange(1, 80, dtype=float)
    before = make_windows(panel(values), ["sku"], CFG, stride=1, calibration_weeks=12)[0]
    changed = values.copy()
    changed[12:] *= 100
    after = make_windows(panel(changed), ["sku"], CFG, stride=1, calibration_weeks=12)[0]

    assert before["start"] == after["start"] == 12
    assert before["expected"][1] == after["expected"][1]
    assert before["actual"][0] != after["actual"][0]


def test_real_window_environment_passes_gymnasium_checks():
    values = np.full(70, 100.0)
    windows = make_windows(panel(values), ["sku"], CFG, stride=1, calibration_weeks=12)
    check_env(RealWindowEnv(CFG, windows), skip_render_check=True)


def test_real_evaluator_uses_only_products_in_test_split(tmp_path):
    values = np.arange(1, 81, dtype=float)
    rows = pd.concat([
        pd.DataFrame({"week_ending": np.arange(len(values)), "product": "train-sku",
                      "demand": values}),
        pd.DataFrame({"week_ending": np.arange(len(values)), "product": "test-sku",
                      "demand": values}),
    ])
    rows.to_csv(tmp_path / "real_panel.csv", index=False)
    (tmp_path / "split.json").write_text(json.dumps({
        "calibration_weeks": 12,
        "train": ["train-sku"],
        "val": [],
        "test": ["test-sku"],
    }), encoding="utf-8")

    windows = load_test_windows(
        CFG,
        {"real": {"calibration_weeks": 12, "eval_stride": 13}},
        panel_dir=tmp_path,
    )

    assert windows
    assert {window["product"] for window in windows} == {"test-sku"}
