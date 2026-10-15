"""Gym actions exercise simulation physics directly, without implying safety approval."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .common import SiteAction
from .site import SiteSimulator


class SiteEnv(gym.Env[NDArray[np.float64], NDArray[np.float64]]):
    def __init__(self, simulator: SiteSimulator) -> None:
        self.metadata = {"render_modes": []}
        self.simulator = simulator
        c = simulator.config
        self.action_space = spaces.Box(
            np.array(
                [0, 0, -c.battery.maximum_charge_power_mw, 0, -c.grid_export_limit_mw],
                dtype=np.float64,
            ),
            np.array(
                [
                    c.compute.max_power_mw,
                    c.thermal.cooling_max_power_mw,
                    c.battery.maximum_discharge_power_mw,
                    c.generator.maximum_power_mw,
                    c.grid_import_limit_mw,
                ],
                dtype=np.float64,
            ),
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64)

    def _observation(self) -> NDArray[np.float64]:
        o = self.simulator.observation()
        return np.array(
            [
                o.workload_demand_compute_units_per_h,
                o.facility_temperature_c,
                o.battery_energy_mwh,
                o.generator_power_mw,
                o.electricity_price_usd_per_mwh,
                o.ambient_temperature_c,
                o.deferrable_backlog_compute_units,
            ],
            dtype=np.float64,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        super().reset(seed=seed)
        _, info = self.simulator.reset(seed=seed, options=options)
        return self._observation(), info

    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        if not self.action_space.contains(action):
            raise ValueError("action outside finite configured space")
        _, reward, terminated, truncated, info = self.simulator.advance(SiteAction(*action))
        return self._observation(), reward, terminated, truncated, info
