"""Simulator-trained parametric dispatch with cross-entropy policy search.

This small learned-control approach is appropriate for the deterministic,
low-dimensional simulator: every candidate is scored on the real simulator and
passes through the independent shield. It makes no reinforcement-learning or
baseline superiority claim. Inference emits proposals, never executable tokens.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from energy_decision_layer.contracts import (
    ForecastBundleV1,
    ProposedControlActionV1,
    SiteObservationV1,
)
from energy_decision_layer.optimizer.planner import RulePolicy, proposal
from energy_decision_layer.safety import SafetyShield
from energy_decision_layer.sim import SiteConfig, SiteSimulator
from energy_decision_layer.sim.common import SiteInputs

LOWER = np.array([10.0, 5.0, 0.0, 0.2, 0.5])
UPPER = np.array([180.0, 100.0, 5.0, 0.8, 1.5])
DEFAULT = np.array([80.0, 35.0, 2.0, 0.5, 1.0])


class LearnedPolicy:
    """Five learned parameters: price centre/spread, battery gain/SOC, cooling gain."""

    def __init__(
        self, config: SiteConfig | None = None, parameters: Sequence[float] | None = None
    ) -> None:
        self.config = config or SiteConfig()
        self.parameters = np.array(parameters if parameters is not None else DEFAULT, dtype=float)
        if (
            self.parameters.shape != (5,)
            or not np.isfinite(self.parameters).all()
            or np.any(self.parameters < LOWER)
            or np.any(self.parameters > UPPER)
        ):
            raise ValueError("Policy parameters must be five finite values within training bounds")
        self.version = "untrained-default"
        self.training_metadata: dict[str, Any] = {}

    def propose_action(
        self, observation: SiteObservationV1, forecast: ForecastBundleV1 | None = None
    ) -> ProposedControlActionV1:
        o, c = observation, self.config
        base = RulePolicy(c).propose_action(o, forecast)
        centre, spread, gain, target_soc, cooling_gain = self.parameters
        price = o.electricity_price_usd_per_mwh
        # Current price is known; only supplied predictions may influence later prices.
        if forecast is not None and len(forecast.points) > 1:
            centre = 0.5 * centre + 0.5 * float(
                np.mean([p.electricity_price_usd_per_mwh for p in forecast.points])
            )
        battery = gain * float(np.tanh((price - centre) / spread))
        battery += 2.0 * (o.battery_state_of_charge - target_soc)
        battery = float(
            np.clip(
                battery, -c.battery.maximum_charge_power_mw, c.battery.maximum_discharge_power_mw
            )
        )
        cooling = min(c.thermal.cooling_max_power_mw, base.cooling_power_mw * cooling_gain)
        load = base.compute_power_mw + cooling + c.auxiliary_power_mw
        fuel_price = (
            c.generator.fuel_cost_per_mwh_thermal / c.generator.electrical_efficiency
            + c.generator.variable_cost_per_mwh_electric
        )
        generator = base.generator_power_mw
        if o.generator_available and price > fuel_price + spread:
            generator = min(c.generator.maximum_power_mw, max(0.0, load - battery))
        grid = (
            float(
                np.clip(load - battery - generator, -o.grid_export_limit_mw, o.grid_import_limit_mw)
            )
            if o.grid_available
            else 0.0
        )
        # A grid restriction requests reserve; dynamic projection remains shield-owned.
        battery = max(battery, load - grid - generator)
        p = proposal(
            o, (base.compute_power_mw, cooling, battery, generator, grid), "learned-cem-dispatch"
        )
        return p.model_copy(
            update={
                "policy_version": self.version,
                "rationale": (
                    "Simulator-trained"
                    if self.version != "untrained-default"
                    else "Untrained default"
                )
                + " price/SOC/cooling controller; requires safety approval",
            }
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        if target.suffix != ".json":
            target.mkdir(parents=True, exist_ok=True)
            target = target / "policy.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "version": self.version,
                    "parameters": self.parameters.tolist(),
                    "site_config": asdict(self.config),
                    "metadata": self.training_metadata,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    @classmethod
    def load(cls, path: str | Path, config: SiteConfig | None = None) -> LearnedPolicy:
        from energy_decision_layer.sim.battery import BatteryConfig
        from energy_decision_layer.sim.compute import ComputeConfig
        from energy_decision_layer.sim.generator import GeneratorConfig
        from energy_decision_layer.sim.thermal import ThermalConfig

        source = Path(path)
        data = json.loads(
            (source if source.suffix == ".json" else source / "policy.json").read_text()
        )
        stored = data["site_config"]
        stored_config = SiteConfig(
            **{
                **stored,
                "battery": BatteryConfig(**stored["battery"]),
                "compute": ComputeConfig(**stored["compute"]),
                "generator": GeneratorConfig(**stored["generator"]),
                "thermal": ThermalConfig(**stored["thermal"]),
            }
        )
        if config is not None and asdict(config) != asdict(stored_config):
            raise ValueError("Policy checkpoint site configuration does not match runtime")
        policy = cls(config or stored_config, data["parameters"])
        policy.version, policy.training_metadata = data["version"], data["metadata"]
        return policy


def policy_score(
    policy: LearnedPolicy, inputs: Sequence[SiteInputs], *, seed: int = 42
) -> dict[str, float | int]:
    """Environment reward plus explicit service, thermal, grid and terminal reserve terms."""
    sim = SiteSimulator(inputs, policy.config)
    o, _ = sim.reset(seed=seed)
    shield = SafetyShield(policy.config)
    total, cost, compute, misses, wear, fuel, thermal, unserved = (0.0,) * 8
    initial_energy = o.battery_energy_mwh
    rejected = interventions = steps = 0
    for _ in inputs:
        response = shield.evaluate(policy.propose_action(o), o)
        if response.executed_action is None:
            rejected += 1
            total += 1e8
            break
        interventions += int(bool(response.executed_action.modified_fields))
        grid_import_cap, grid_export_cap = o.grid_import_limit_mw, o.grid_export_limit_mw
        o, reward, terminated, truncated, info = sim.step(response.executed_action)
        total -= reward  # electricity, fuel/start costs, battery wear and missed compute
        cost += float(info["operating_cost"])
        compute += float(info["completed_compute_units"])
        misses += float(info["deadline_missed_compute_units"])
        wear += float(info["battery_degradation_cost"])
        fuel += float(info["generator_fuel_mwh"])
        thermal += float(info["thermal_violation"])
        unserved += float(info["unserved_energy_mwh"])
        total += 1e6 * (float(info["thermal_violation"]) + float(info["unserved_energy_mwh"]))
        total += 1e6 * max(0.0, abs(float(info["power_balance_residual_mw"])) - 1e-7)
        # Compare interval feedback to the SAME interval's grid telemetry, not next-state limits.
        total += 1e6 * (
            max(0.0, float(info["grid_power_mw"]) - grid_import_cap)
            + max(0.0, -float(info["grid_power_mw"]) - grid_export_cap)
        )
        steps += 1
        if terminated or truncated:
            break
    # Avoid learning to spend free initial battery inventory or defer all compute beyond episode.
    replacement_price = max(0.0, float(np.mean([i.grid_import_price_per_mwh for i in inputs])))
    total += max(0.0, initial_energy - o.battery_energy_mwh) * replacement_price
    total += o.deferrable_backlog_compute_units * policy.config.deadline_penalty_per_unit
    return {
        "objective": total,
        "operating_cost": cost,
        "compute_completed": compute,
        "missed_compute": misses,
        "battery_wear_cost": wear,
        "generator_fuel_mwh": fuel,
        "thermal_violations": thermal,
        "unserved_energy_mwh": unserved,
        "shield_rejections": rejected,
        "shield_interventions": interventions,
        "steps": steps,
    }


def train_policy(
    inputs: Sequence[SiteInputs],
    output: str | Path,
    *,
    config: SiteConfig | None = None,
    seed: int = 42,
    iterations: int = 3,
    population: int = 8,
) -> dict[str, Any]:
    if len(inputs) < 6 or iterations < 1 or population < 4:
        raise ValueError("Need >=6 intervals, >=1 iteration, and >=4 candidates")
    rng = np.random.default_rng(seed)
    c = config or SiteConfig()
    cutoff = max(3, int(len(inputs) * 0.75))
    train, validation = inputs[:cutoff], inputs[cutoff:]
    mean, scale = DEFAULT.copy(), (UPPER - LOWER) / 4
    best = LearnedPolicy(c, mean.tolist())
    initial = policy_score(best, train, seed=seed)
    best_score = float(initial["objective"])
    trace = []
    for iteration in range(iterations):
        candidates = [
            best.parameters.copy(),
            *list(np.clip(rng.normal(mean, scale, size=(population - 1, 5)), LOWER, UPPER)),
        ]
        scored = []
        for parameters in candidates:
            policy = LearnedPolicy(c, parameters.tolist())
            score = float(policy_score(policy, train, seed=seed)["objective"])
            scored.append((score, parameters))
            if score < best_score:
                best_score, best = score, policy
        elite = np.array(
            [p for _, p in sorted(scored, key=lambda v: v[0])[: max(2, population // 4)]]
        )
        mean = np.mean(elite, axis=0)
        scale = np.maximum(np.std(elite, axis=0), (UPPER - LOWER) * 0.02)
        trace.append({"iteration": iteration, "best_training_objective": best_score})
    import hashlib

    best.version = (
        "cem-" + hashlib.sha256(best.parameters.tobytes() + str(seed).encode()).hexdigest()[:12]
    )
    report: dict[str, Any] = {
        "method": "cross-entropy simulator policy search",
        "seed": seed,
        "data_kind": "synthetic simulator trajectories",
        "training_intervals": len(train),
        "validation_intervals": len(validation),
        "iterations": iterations,
        "population": population,
        "trace": trace,
        "parameters": best.parameters.tolist(),
        "initial_training": initial,
        "training": policy_score(best, train, seed=seed),
        "validation": policy_score(best, validation, seed=seed),
        "safety": "Every training step uses SafetyShield then SiteSimulator.step",
        "limitations": (
            "No production calibration or baseline superiority claim; see cross-policy report"
        ),
    }
    best.training_metadata = report
    best.save(output)
    out = Path(output)
    (out.parent if out.suffix == ".json" else out).joinpath("training-report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report
