"""Learned dispatch policy (behaviour cloning) behind the standard ``PolicyProtocol``.

What it is
----------
A fast regressor that imitates the MILP planner when that planner is given PERFECT
HINDSIGHT forecasts, while seeing only what a deployed policy would see: the current
observation and the ML forecast. Labels come from the oracle; features never do.

What it is not
--------------
Not a safety mechanism and not a proven improvement over the MILP baseline. Behaviour
cloning compounds errors off the demonstrated state distribution (no DAgger/RL
fine-tuning yet), so measure it with the unchanged evaluator before trusting it.
Its proposals still go through the dispatcher and safety shield like any other policy.
The grid setpoint is derived from the power balance rather than learned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from energy_decision_layer.contracts import (
    ForecastBundleV1,
    ProposedControlActionV1,
    SiteObservationV1,
)
from energy_decision_layer.evaluation.runner import evaluate
from energy_decision_layer.optimizer import MilpPolicy
from energy_decision_layer.optimizer.planner import proposal
from energy_decision_layer.sim import SiteConfig
from energy_decision_layer.testing.doubles import OfflineHindsightForecastDouble

from .forecaster import MLForecastProvider, QuantileForecastModel

POLICY_NAME = "energy-ml-behavior-cloning"
FORECAST_LOOKAHEAD = 6

FEATURE_NAMES: tuple[str, ...] = (
    "demand_rate",
    "price",
    "ambient_c",
    "facility_temperature_c",
    "battery_soc",
    "battery_energy_mwh",
    "generator_power_mw",
    "generator_on",
    "generator_available",
    "generator_on_steps",
    "generator_off_steps",
    "deferrable_backlog",
    "urgent_backlog",
    "grid_available",
    "grid_import_limit_mw",
    "compute_capacity_rate",
    "tod_sin",
    "tod_cos",
    "f_price_mean",
    "f_price_max",
    "f_price_min",
    "f_demand_mean",
    "f_demand_max",
    "f_ambient_max",
)
TARGET_NAMES: tuple[str, ...] = (
    "compute_power_mw",
    "cooling_power_mw",
    "battery_power_mw",
    "generator_power_mw",
)


def observation_features(
    o: SiteObservationV1, forecast: ForecastBundleV1 | None = None
) -> np.ndarray:
    """Features available at decision time. Missing forecast falls back to persistence."""
    seconds = (
        o.timestamp_utc.hour * 3600 + o.timestamp_utc.minute * 60 + o.timestamp_utc.second
    )
    angle = 2 * np.pi * seconds / 86400.0
    prices = [o.electricity_price_usd_per_mwh]
    demands = [o.workload_demand_compute_units_per_h]
    ambients = [o.ambient_temperature_c]
    if forecast is not None:
        window = forecast.points[:FORECAST_LOOKAHEAD]
        prices = [p.electricity_price_usd_per_mwh for p in window]
        demands = [p.workload_demand_compute_units_per_h for p in window]
        ambients = [p.ambient_temperature_c for p in window]
    urgent = sum(q.remaining_compute_units for q in o.queued_workloads if q.steps_to_deadline <= 2)
    values = {
        "demand_rate": o.workload_demand_compute_units_per_h,
        "price": o.electricity_price_usd_per_mwh,
        "ambient_c": o.ambient_temperature_c,
        "facility_temperature_c": o.facility_temperature_c,
        "battery_soc": o.battery_state_of_charge,
        "battery_energy_mwh": o.battery_energy_mwh,
        "generator_power_mw": o.generator_power_mw,
        "generator_on": float(o.generator_is_on),
        "generator_available": float(o.generator_available),
        "generator_on_steps": o.generator_consecutive_on_steps,
        "generator_off_steps": o.generator_consecutive_off_steps,
        "deferrable_backlog": o.deferrable_backlog_compute_units,
        "urgent_backlog": urgent,
        "grid_available": float(o.grid_available),
        "grid_import_limit_mw": o.grid_import_limit_mw,
        "compute_capacity_rate": o.active_compute_capacity_compute_units_per_h,
        "tod_sin": np.sin(angle),
        "tod_cos": np.cos(angle),
        "f_price_mean": float(np.mean(prices)),
        "f_price_max": float(np.max(prices)),
        "f_price_min": float(np.min(prices)),
        "f_demand_mean": float(np.mean(demands)),
        "f_demand_max": float(np.max(demands)),
        "f_ambient_max": float(np.max(ambients)),
    }
    return np.array([values[name] for name in FEATURE_NAMES], dtype=float)


@dataclass
class LearnedPolicy:
    """``PolicyProtocol`` implementation wrapping a fitted multi-output regressor."""

    model: Any
    config: SiteConfig

    def propose_action(
        self, observation: SiteObservationV1, forecast: ForecastBundleV1 | None = None
    ) -> ProposedControlActionV1:
        o, c = observation, self.config
        raw = self.model.predict(observation_features(o, forecast).reshape(1, -1))[0]
        compute = float(np.clip(raw[0], 0.0, c.compute.max_power_mw))
        cooling = float(np.clip(raw[1], 0.0, c.thermal.cooling_max_power_mw))
        battery = float(
            np.clip(raw[2], -c.battery.maximum_charge_power_mw, c.battery.maximum_discharge_power_mw)
        )
        generator = float(np.clip(raw[3], 0.0, c.generator.maximum_power_mw))
        if not o.generator_available or generator < 0.5 * c.generator.minimum_power_mw:
            generator = 0.0
        elif generator < c.generator.minimum_power_mw:
            generator = c.generator.minimum_power_mw
        if o.grid_available:
            grid = compute + cooling + c.auxiliary_power_mw - battery - generator
            grid = float(np.clip(grid, -o.grid_export_limit_mw, o.grid_import_limit_mw))
        else:
            grid = 0.0
        return proposal(o, (compute, cooling, battery, generator, grid), POLICY_NAME)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @staticmethod
    def load(path: str | Path) -> LearnedPolicy:
        policy = joblib.load(path)
        if not isinstance(policy, LearnedPolicy):
            raise TypeError(f"{path} does not contain a LearnedPolicy")
        return policy


class _OracleRecorder:
    """Acts as the oracle MILP but records (deployable features, oracle action) pairs."""

    def __init__(self, oracle_forecaster: OfflineHindsightForecastDouble, rows: list[dict]) -> None:
        self.milp = MilpPolicy(allow_offline_hindsight=True, use_prediction_intervals=False)
        self.oracle_forecaster, self.rows = oracle_forecaster, rows
        self.last_result = None

    def propose_action(
        self, observation: SiteObservationV1, forecast: ForecastBundleV1 | None = None
    ) -> ProposedControlActionV1:
        horizon = len(forecast.points) if forecast is not None else 1
        hindsight = self.oracle_forecaster.forecast(
            observation, horizon_steps=horizon, cadence_seconds=observation.timestep_seconds
        )
        action = self.milp.propose_action(observation, hindsight)
        self.last_result = self.milp.last_result
        row = dict(zip(FEATURE_NAMES, observation_features(observation, forecast), strict=True))
        row.update({name: getattr(action, name) for name in TARGET_NAMES})
        self.rows.append(row)
        return action


def collect_demonstrations(
    model: QuantileForecastModel,
    episodes: list[tuple[pd.DataFrame, pd.DataFrame]],
    *,
    episode_start,
) -> pd.DataFrame:
    """Run the oracle through the unchanged simulator, one (context, episode) pair per run.

    ``context`` is the past data that primes the forecaster (lags only); ``episode``
    supplies the simulated exogenous inputs. Features use the ML forecast; labels the oracle.
    """
    from .synthetic import to_site_inputs

    rows: list[dict] = []
    truncated = 0
    for context, episode in episodes:
        inputs = to_site_inputs(episode)
        provider = MLForecastProvider(model)
        provider.prime(context)
        oracle = OfflineHindsightForecastDouble(tuple(inputs), episode_start)
        episode_rows: list[dict] = []
        result = evaluate(
            "demonstrations",
            inputs,
            lambda: _OracleRecorder(oracle, episode_rows),
            forecaster=provider,
            policy_label="oracle-recorder",
        )
        if result.failed:
            # The shield fails closed (no executed action) when no feasible coupled
            # action exists. The last recorded proposal was never executed, so it is
            # not a valid label; keep only the executed prefix of the episode.
            if not episode_rows:
                raise RuntimeError(f"demonstration run failed: {result.failure_reason}")
            episode_rows.pop()
            truncated += 1
        rows.extend(episode_rows)
    frame = pd.DataFrame(rows)
    frame.attrs["truncated_episodes"] = truncated
    return frame


def fit_policy(demos: pd.DataFrame, config: SiteConfig | None = None, seed: int = 0) -> LearnedPolicy:
    if len(demos) < 200:
        raise ValueError("too few demonstrations to fit a policy")
    model = ExtraTreesRegressor(
        n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=seed
    ).fit(demos[list(FEATURE_NAMES)].to_numpy(), demos[list(TARGET_NAMES)].to_numpy())
    return LearnedPolicy(model, config or SiteConfig())
