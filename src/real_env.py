"""Real-demand episode helpers for evaluation of policies trained elsewhere."""

import numpy as np
import pandas as pd

from src.env import SupplyChainEnv


def make_windows(panel, products, cfg, stride, calibration_weeks):
    """Create scaled episodes with a causal forecast using only earlier weeks."""
    weeks = cfg["weeks"]
    target_mean = (cfg["demand"]["base_min"] + cfg["demand"]["base_max"]) / 2
    windows = []
    for product in products:
        if product not in panel.columns:
            raise ValueError(f"Product {product!r} is missing from the weekly panel.")
        raw = panel[product].to_numpy(dtype=float)
        calibration_mean = raw[:calibration_weeks].mean()
        if calibration_mean <= 0:
            raise ValueError(f"Product {product!r} has no positive calibration demand.")
        series = raw * (target_mean / calibration_mean)
        for start in range(calibration_weeks, len(series) - weeks, stride):
            actual = np.rint(series[start:start + weeks])
            expected = np.empty(weeks + 1, dtype=float)
            for offset in range(weeks + 1):
                end = start + offset - 1
                history = series[max(0, end - 4):end]
                if history.size == 0:
                    raise ValueError("A real-demand window has insufficient history for its forecast.")
                expected[offset] = history.mean()
            windows.append({
                "product": str(product),
                "start": int(start),
                "expected": expected,
                "actual": actual,
            })
    if not windows:
        raise ValueError("No real-demand windows were generated; check panel length and stride.")
    return windows


class RealWindowEnv(SupplyChainEnv):
    """Reuse the supply-side simulator while sourcing customer demand from a real window."""

    def __init__(self, cfg, windows, rng_seed=0):
        if not windows:
            raise ValueError("RealWindowEnv requires at least one demand window.")
        super().__init__(cfg, range(len(windows)))
        self.windows = windows
        self.real = windows[0]
        self._picker = np.random.default_rng(rng_seed)

    def reset(self, seed=None, options=None):
        options = dict(options or {})
        if seed is not None:
            self._picker = np.random.default_rng(seed)
        if "scenario_seed" in options:
            idx = int(options["scenario_seed"])
        else:
            idx = int(self._picker.integers(len(self.windows)))
        if not 0 <= idx < len(self.windows):
            raise ValueError(f"Window index {idx} is outside 0..{len(self.windows) - 1}.")
        self.real = self.windows[idx]
        options["scenario_seed"] = idx
        return super().reset(seed=seed, options=options)

    def _draw_randomness(self):
        self.scn = {
            "expected": self.real["expected"],
            "noise": 0.0,
            "base": float(self.real["actual"].mean()),
        }
        super()._draw_randomness()
        self.demand = self.real["actual"].astype(float)


def run_policy(policy, cfg, windows, repeats=1, base_seed=0):
    env = RealWindowEnv(cfg, windows)
    rows = []
    try:
        for idx, window in enumerate(windows):
            for repeat in range(repeats):
                episode_seed = base_seed + idx * 100 + repeat
                if hasattr(policy, "reset"):
                    policy.reset(episode_seed)
                obs, _ = env.reset(seed=episode_seed, options={"scenario_seed": idx})
                terminated = truncated = False
                while not (terminated or truncated):
                    obs, _, terminated, truncated, _ = env.step(policy(obs))
                row = dict(env.episode_stats())
                row.update(window=idx, product=window["product"], start=window["start"])
                rows.append(row)
    finally:
        env.close()
    return pd.DataFrame(rows)
