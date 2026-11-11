"""
Each week the agent picks ONE action from 8:  action = supplier * 4 + size_index
  supplier: 0 = A (cheap, slow, unreliable), 1 = B (dearer, fast, reliable)
  size_index: 0..3 -> the 4 order sizes in config/env.yaml   (action 0 and 4 both mean "order nothing")
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

N_SIZES = 4


def make_scenario(seed, cfg):
    """Draw the demand pattern of one scenario. The same seed always gives the same pattern."""
    d = cfg["demand"]
    if d["source"] != "synthetic":
        raise ValueError("Only synthetic demand is built so far (Stage 2 would add real demand).")
    rng = np.random.default_rng(seed)
    base = rng.uniform(d["base_min"], d["base_max"])
    amp = rng.uniform(d["season_min"], d["season_max"])
    noise = rng.uniform(d["noise_min"], d["noise_max"])
    phase = rng.uniform(0, 2 * np.pi)
    weeks = np.arange(cfg["weeks"] + 1)                      # one extra week for next-week forecasts
    expected = base * (1 + amp * np.sin(2 * np.pi * weeks / d["season_length_weeks"] + phase))
    return {"expected": expected, "noise": noise, "base": base}


def unpack_obs(obs, cfg):
    """Turn the normalised observation back into real numbers (used by baselines and plan cards)."""
    cap = cfg["inventory_cap"]
    return {
        "inventory": obs[0] * cap, "transit_a": obs[1] * cap, "transit_b": obs[2] * cap,
        "forecast": obs[3] * cfg["normalise"]["forecast_scale"],
        "rel_signal": obs[4], "week": obs[5] * cfg["weeks"], "last_failed": obs[6],
    }


class SupplyChainEnv(gym.Env):
    def __init__(self, cfg, scenario_seeds):
        super().__init__()
        self.cfg = cfg
        self.scenario_seeds = list(scenario_seeds)
        self.weeks = cfg["weeks"]
        self.sizes = cfg["order_sizes"]
        assert len(self.sizes) == N_SIZES, f"order_sizes must have exactly {N_SIZES} entries"
        self.obs_max = cfg["normalise"]["obs_max"]
        self.suppliers = [cfg["supplier_a"], cfg["supplier_b"]]
        self.action_space = spaces.Discrete(2 * N_SIZES)
        self.observation_space = spaces.Box(0.0, self.obs_max, shape=(7,), dtype=np.float32)

    # ---------- start of an episode ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)                              # seeds self.np_random
        options = options or {}
        s = options.get("scenario_seed")
        if s is None:                                         # training: pick a random scenario
            s = int(self.np_random.choice(self.scenario_seeds))
        self.scenario_seed = s
        self.scn = make_scenario(s, self.cfg)
        self._draw_randomness()
        self.t = 0
        self.inventory = float(self.cfg["start_inventory"])
        self.pipeline = []                                    # orders on the way: [arrival_week, qty, supplier]
        self.fail_streak = 0                                  # failed orders in a row
        self.last_failed = 0
        self.stats = dict(cost=0.0, demand=0.0, sold=0.0, stockout_weeks=0, fallback=0,
                          loop_trap=0, failed=0)
        return self._obs(), {}

    def _draw_randomness(self):
        """Draw ALL random numbers for the whole episode now. Because they do not depend on the
        agent's actions, every policy faces exactly the same demand and the same luck."""
        r, u, n = self.np_random, self.cfg["uncertainty"], self.weeks
        z = r.normal(size=(3, n))                             # demand, forecast, signal noise
        self.z_forecast, self.z_signal = z[1], z[2]
        self.demand = np.rint(np.maximum(0, self.scn["expected"][:n] + self.scn["noise"] * z[0]))
        self.u_lead, self.u_rel, self.u_api = r.random(n), r.random(n), r.random(n)
        self.disruption = np.zeros(n, dtype=int)              # 0 none, 1 A delayed, 2 A unavailable
        t = 0
        while t < n:
            if r.random() < u["disruption_prob"]:
                length = int(r.integers(u["disruption_min_weeks"], u["disruption_max_weeks"] + 1))
                self.disruption[t:t + length] = 1 + int(r.random() < u["disruption_unavailable_share"])
                t += length
            else:
                t += 1

    # ---------- what the agent sees ----------
    def _forecast(self):
        t, std = min(self.t, self.weeks - 1), self.cfg["uncertainty"]["forecast_noise_std"]
        return max(0.0, self.scn["expected"][t + 1] * (1 + std * self.z_forecast[t]))

    def _signal(self):
        u, t = self.cfg["uncertainty"], min(self.t, self.weeks - 1)
        centre = [u["signal_normal"], u["signal_delayed"], u["signal_unavailable"]][self.disruption[t]]
        return float(np.clip(centre + u["signal_noise_std"] * self.z_signal[t], 0, 1))

    def _transit(self, supplier):
        return sum(q for _, q, s in self.pipeline if s == supplier)

    def _obs(self):
        cap = self.cfg["inventory_cap"]
        obs = [self.inventory / cap, self._transit(0) / cap, self._transit(1) / cap,
               self._forecast() / self.cfg["normalise"]["forecast_scale"],
               self._signal(), self.t / self.weeks, float(self.last_failed)]
        return np.clip(np.array(obs, dtype=np.float32), 0.0, self.obs_max)

    # ---------- one week ----------
    def step(self, action):
        action = int(action)
        supplier, qty = action // N_SIZES, self.sizes[action % N_SIZES]
        wanted_order = qty > 0
        t, info = self.t, {"fallback_used": False, "order_failed": False}
        fixed = 0.0

        if self.fail_streak >= self.cfg["safety"]["fallback_after_failures"]:
            supplier, qty = 1, self._fallback_qty()           # safety net: reliable supplier B
            info["fallback_used"] = True
            self.stats["fallback"] += 1
            self.stats["loop_trap"] += int(wanted_order)      # agent was still retrying (4th try)
            self.fail_streak = 0
            failed = False
        else:
            failed = qty > 0 and self._order_fails(t, supplier)

        if qty > 0 and failed:
            self.fail_streak += 1
            self.stats["failed"] += 1
            info["order_failed"] = True
            qty = 0
        if qty > 0:                                           # accepted order (or fallback order)
            self._place_order(t, supplier, qty)
            fixed = self.cfg["fixed_order_cost"]
            if not info["fallback_used"]:
                self.fail_streak = 0
        self.last_failed = int(failed)

        # demand happens, then we pay holding and lost-sale costs
        d = self.demand[t]
        sold = min(self.inventory, d)
        lost = d - sold
        self.inventory -= sold
        c = self.cfg["costs"]
        cost = c["holding"] * self.inventory + c["lost_sale"] * lost + fixed

        # next week starts: deliveries arrive and we pay for them (pay on delivery)
        self.t += 1
        if self.t < self.weeks:
            cost += self._receive()
        self.stats["cost"] += cost
        self.stats["demand"] += d
        self.stats["sold"] += sold
        self.stats["stockout_weeks"] += int(lost > 0)
        info.update(cost=cost, demand=d, lost=lost, week=t)
        done = self.t >= self.weeks                           # week/52 is in the observation,
        return self._obs(), -cost / c["reward_scale"], done, False, info  # so end = "terminated"

    def _order_fails(self, t, supplier):
        api_down = self.u_api[t] < self.cfg["uncertainty"]["api_failure_prob"]
        a_unavailable = supplier == 0 and self.disruption[t] == 2
        return bool(api_down or a_unavailable)

    def _place_order(self, t, supplier, qty):
        sup, u = self.suppliers[supplier], self.cfg["uncertainty"]
        lead = sup["lead_min"] + int(self.u_lead[t] * (sup["lead_max"] - sup["lead_min"] + 1))
        if self.u_rel[t] >= sup["reliability"]:               # supplier is unreliable this time
            lead += sup["delay_weeks"]
        if supplier == 0 and self.disruption[t] == 1:         # A is delayed by a disruption
            lead += u["disruption_extra_weeks"]
        self.pipeline.append([t + lead, qty, supplier])

    def _receive(self):
        """Move arrived orders into inventory. Returns the purchase cost of what arrived."""
        arrived = [o for o in self.pipeline if o[0] <= self.t]
        self.pipeline = [o for o in self.pipeline if o[0] > self.t]
        got = sum(q for _, q, _ in arrived)
        stored = min(got, self.cfg["inventory_cap"] - self.inventory)
        self.inventory += stored
        return sum(q * self.suppliers[s]["unit_cost"] for _, q, s in arrived)

    def _fallback_qty(self):
        """Fallback base-stock rule: order up to (cover weeks x forecast) from Supplier B."""
        target = self.cfg["safety"]["fallback_cover_weeks"] * self._forecast()
        gap = target - self.inventory - self._transit(0) - self._transit(1)
        fits = [s for s in self.sizes if s <= gap]
        return max(fits) if fits else 0

    def episode_stats(self):
        """Numbers for the finished episode (used by evaluate.py)."""
        s = self.stats
        return {
            "scenario": self.scenario_seed, "total_cost": s["cost"],
            "cost_per_unit": s["cost"] / max(s["demand"], 1),
            "fill_rate": s["sold"] / max(s["demand"], 1),
            "stockout_freq": s["stockout_weeks"] / self.weeks,
            "fallback_rate": s["fallback"] / self.weeks,
            "loop_trap": int(s["loop_trap"] > 0), "failed_orders": s["failed"],
        }
