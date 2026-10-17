"""Comparable shielded ML experiments with explicit oracle and data limitations."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from energy_decision_layer.contracts import (
    ForecastBundleV1,
    ForecastPointV1,
    ForecastProviderProtocol,
    ForecastSignal,
    PolicyProtocol,
    PredictionIntervalV1,
    ProposedControlActionV1,
    SafetyResponseV1,
    SiteObservationV1,
)
from energy_decision_layer.optimizer import MilpPolicy, RulePolicy
from energy_decision_layer.safety import SafetyShield
from energy_decision_layer.sim import SiteConfig, SiteInputs
from energy_decision_layer.sim.battery import BatteryConfig
from energy_decision_layer.testing.doubles import OfflineHindsightForecastDouble

from .runner import evaluate, scenarios, write_reports


def policy_factory(policy: PolicyProtocol) -> Callable[[], PolicyProtocol]:
    return lambda: policy


class ExperimentLogger:
    """Capture auditable decisions and exact interval feedback for persisted reports."""

    def __init__(self) -> None:
        self.decisions: list[dict[str, Any]] = []
        self.outcomes: list[dict[str, Any]] = []

    def record_decision(
        self, proposal: ProposedControlActionV1, response: SafetyResponseV1
    ) -> None:
        self.decisions.append(
            {
                "proposal": proposal.model_dump(mode="json"),
                "safety": response.model_dump(mode="json"),
            }
        )

    def record_outcome(
        self,
        proposal: ProposedControlActionV1,
        observation: SiteObservationV1,
        diagnostics: Mapping[str, Any],
    ) -> None:
        info = dict(diagnostics)
        info["violations"] = [asdict(v) for v in info["violations"]]
        self.outcomes.append(
            {
                "action_id": str(proposal.action_id),
                "observation": observation.model_dump(mode="json"),
                "metrics": info,
            }
        )


class ForecastPerturbation:
    """Controlled prospective bias: current known inputs are never corrupted.

    All three signals use the same signed fraction. Intervals move with the
    point estimate so this experiment measures forecast bias, not interval width.
    """

    def __init__(self, provider: ForecastProviderProtocol, fraction: float = 0.0) -> None:
        if not -1.0 < fraction < 10.0:
            raise ValueError("forecast error fraction must lie in (-1, 10)")
        self.provider, self.fraction = provider, fraction

    def forecast(
        self, observation: SiteObservationV1, *, horizon_steps: int, cadence_seconds: float
    ) -> ForecastBundleV1:
        base = self.provider.forecast(
            observation, horizon_steps=horizon_steps, cadence_seconds=cadence_seconds
        )
        if not self.fraction:
            return base
        points = []
        for index, point in enumerate(base.points):
            if index == 0:
                points.append(point)
                continue
            changes = {
                ForecastSignal.WORKLOAD_DEMAND: max(
                    0.0, point.workload_demand_compute_units_per_h * (1 + self.fraction)
                ),
                ForecastSignal.ELECTRICITY_PRICE: point.electricity_price_usd_per_mwh
                + abs(point.electricity_price_usd_per_mwh) * self.fraction,
                ForecastSignal.AMBIENT_TEMPERATURE: max(
                    -273.15, point.ambient_temperature_c + 10 * self.fraction
                ),
            }
            payload = point.model_dump()
            intervals = []
            for band in point.prediction_intervals:
                delta = changes.get(band.signal, getattr(point, band.signal.value)) - getattr(
                    point, band.signal.value
                )
                lower, upper = band.lower_value + delta, band.upper_value + delta
                if band.signal is ForecastSignal.WORKLOAD_DEMAND:
                    lower, upper = max(0.0, lower), max(0.0, upper)
                if band.signal is ForecastSignal.AMBIENT_TEMPERATURE:
                    lower, upper = max(-273.15, lower), max(-273.15, upper)
                intervals.append(
                    PredictionIntervalV1(
                        signal=band.signal,
                        lower_value=lower,
                        upper_value=upper,
                        coverage_probability=band.coverage_probability,
                    )
                )
            payload.update({signal.value: value for signal, value in changes.items()})
            payload["prediction_intervals"] = tuple(intervals)
            points.append(ForecastPointV1.model_validate(payload))
        data = base.model_dump()
        data.update(points=tuple(points), model_name=f"bias-{self.fraction:+.0%}-{base.model_name}")
        return ForecastBundleV1.model_validate(data)


def operational_details(
    logger: ExperimentLogger, inputs: Sequence[SiteInputs], config: SiteConfig
) -> dict[str, Any]:
    thermal = grid = 0
    fuel = 0.0
    for index, outcome in enumerate(logger.outcomes):
        m, x = outcome["metrics"], inputs[index]
        thermal += int(m["thermal_violation"])
        limit = (
            min(
                config.grid_import_limit_mw,
                x.grid_import_limit_mw
                if x.grid_import_limit_mw is not None
                else config.grid_import_limit_mw,
            )
            if x.grid_available
            else 0.0
        )
        export = (
            min(
                config.grid_export_limit_mw,
                x.grid_export_limit_mw
                if x.grid_export_limit_mw is not None
                else config.grid_export_limit_mw,
            )
            if x.grid_available
            else 0.0
        )
        grid += int(not -export - 1e-7 <= m["grid_power_mw"] <= limit + 1e-7)
        fuel += m["generator_fuel_mwh"]
    return {
        "thermal_limit_violations": thermal,
        "grid_limit_violations": grid,
        "generator_fuel_mwh_thermal": fuel,
        "decision_count": len(logger.decisions),
        "executed_count": len(logger.outcomes),
    }


def analytic_regret(policies: Mapping[str, PolicyProtocol], *, seed: int = 7) -> dict[str, Any]:
    """An independently soluble restricted fixture with a valid comparable optimum.

    One interval, nondeferrable demand, no storage/onsite generation, positive
    price, and enough thermal headroom to require no cooling. Each additional
    served unit costs $0.075 < the $100 missed-unit penalty. Therefore serving
    all work at minimum compute power and zero cooling is the exact optimum.
    This does not claim an optimum for the general rolling-horizon benchmark.
    """
    c = SiteConfig(
        battery=BatteryConfig(maximum_charge_power_mw=0.0, maximum_discharge_power_mw=0.0)
    )
    x = SiteInputs(nondeferrable_compute_units=40.0, generator_available=False)
    dt = c.timestep_seconds / 3600
    compute = c.compute.idle_power_mw + (
        x.nondeferrable_compute_units / dt
    ) / c.compute.max_useful_compute_units_per_hour * (
        c.compute.max_power_mw - c.compute.idle_power_mw
    )
    next_temperature = c.thermal.initial_temperature_c + dt / (
        c.thermal.thermal_capacitance_mwh_per_c
    ) * (
        c.thermal.compute_heat_fraction * compute
        - c.thermal.envelope_conductance_mw_per_c
        * (c.thermal.initial_temperature_c - x.ambient_temperature_c)
    )
    assert c.thermal.minimum_temperature_c <= next_temperature <= c.thermal.maximum_temperature_c
    optimum = (compute + c.auxiliary_power_mw) * dt * x.grid_import_price_per_mwh
    rows = []
    for name, policy in policies.items():
        result = evaluate(
            "analytic-one-step",
            [x],
            policy_factory(policy),
            config=c,
            seed=seed,
            shield=SafetyShield(c),
            policy_label=name,
        )
        rows.append(
            {
                "policy": name,
                "failed": result.failed,
                "objective_cost": result.objective_cost,
                "regret_usd": None if result.failed else result.objective_cost - optimum,
            }
        )
    return {
        "description": "Exact one-step restricted no-storage/no-generator optimum",
        "optimum_objective_usd": optimum,
        "config": asdict(c),
        "seed": seed,
        "results": rows,
    }


def run_ml_evaluation(
    model_directory: str | Path, output: str | Path, *, steps: int = 24, seed: int = 7
) -> dict[str, Any]:
    from energy_decision_layer.adapters import MockSiteAdapter
    from energy_decision_layer.ml import ForecastModel, ForecastProvider, LearnedPolicy

    target, models = Path(output), Path(model_directory)
    target.mkdir(parents=True, exist_ok=True)
    model = ForecastModel.load(models / "forecast")
    learned = LearnedPolicy.load(models / "policy.json")
    c = SiteConfig()
    results = []
    extra: list[dict[str, Any]] = []
    traces = target / "traces"
    traces.mkdir(exist_ok=True)
    for name, inputs in scenarios(steps).items():
        for policy_name in (
            "rule",
            "baseline",
            "learned",
            "baseline-bias-low",
            "baseline-bias-high",
            "OFFLINE-ROLLING-HINDSIGHT",
        ):
            logger = ExperimentLogger()
            policy: PolicyProtocol
            if policy_name == "rule":
                policy = RulePolicy(c)
            elif policy_name == "learned":
                policy = LearnedPolicy.load(models / "policy.json", config=c)
            else:
                policy = MilpPolicy(c, allow_offline_hindsight=policy_name.startswith("OFFLINE"))
            provider: ForecastProviderProtocol = ForecastProvider(model)
            if policy_name.startswith("OFFLINE"):
                provider = OfflineHindsightForecastDouble(
                    inputs, datetime(2026, 1, 1, tzinfo=timezone.utc)
                )
            elif policy_name.endswith("-low"):
                provider = ForecastPerturbation(provider, -0.25)
            elif policy_name.endswith("-high"):
                provider = ForecastPerturbation(provider, 0.25)
            metrics = evaluate(
                name,
                inputs,
                policy_factory(policy),
                config=c,
                seed=seed,
                forecaster=provider,
                policy_label=policy_name,
                shield=SafetyShield(c),
                decision_logger=logger,
                execution_adapter=MockSiteAdapter,
            )
            results.append(metrics)
            extra.append(
                {"scenario": name, "policy": policy_name, **operational_details(logger, inputs, c)}
            )
            (traces / f"{name}-{policy_name}.json").write_text(
                json.dumps(
                    {"decisions": logger.decisions, "outcomes": logger.outcomes},
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
    write_reports(results, target)
    regret = analytic_regret(
        {"rule": RulePolicy(), "baseline": MilpPolicy(), "learned": learned}, seed=seed
    )
    report = {
        "seed": seed,
        "synthetic": True,
        "steps": steps,
        "config": asdict(c),
        "forecast_training": model.report,
        "additional_metrics": extra,
        "analytic_regret": regret,
        "oracle_scope": "Rolling hindsight MILP; not a proven full-episode optimum",
        "failed_runs": sum(r.failed for r in results),
        "safety_stopped_runs": sum(r.failed and r.shield_rejections > 0 for r in results),
        "unexpected_failures": sum(r.failed and r.shield_rejections == 0 for r in results),
    }
    (target / "ml-evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = target / "summary.md"
    summary.write_text(
        summary.read_text() + "\nForecast-error experiments perturb future workload, "
        "price, and temperature by ±25% (temperature ±2.5°C). Known current inputs "
        "remain unchanged. Every run uses the same simulator, seed, initial state, "
        "costs, shield, and mock adapter.\n\nThe offline rolling oracle is a "
        "benchmark, not a proven global optimum. `ml-evaluation.json` separately "
        "reports exact regret for an independently soluble one-step fixture. "
        "Compare served compute and terminal battery/temperature with costs; "
        "lower cost alone does not establish policy superiority.\n"
        "\nFailed runs remain visible. In the compound no-grid/no-generator scenario, "
        "battery reserve eventually makes even auxiliary supply infeasible; the shield "
        "then rejects and stops dispatch. `safety_stopped_runs` counts these explicit "
        "rejections separately from `unexpected_failures`. A safe stop is not successful "
        "completion of the scenario.\n"
    )
    return report
