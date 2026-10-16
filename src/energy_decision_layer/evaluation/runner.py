"""Identical-input comparison; failures are visible and do not become zero-cost wins."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

from energy_decision_layer.contracts import (
    DecisionLoggerProtocol,
    ForecastProviderProtocol,
    PolicyProtocol,
    SafetyShieldProtocol,
)
from energy_decision_layer.optimizer.dispatcher import (
    Dispatcher,
    DispatchOutcomeLoggingFailed,
    DispatchStopped,
)
from energy_decision_layer.optimizer.planner import RulePolicy
from energy_decision_layer.sim import SiteConfig, SiteInputs, SiteSimulator
from energy_decision_layer.testing.doubles import (
    DevelopmentAcceptAllShield,
    InMemoryDecisionLogger,
    PersistenceForecastDouble,
)


@dataclass(frozen=True)
class Metrics:
    scenario: str
    policy: str
    steps: int
    total_operating_cost: float
    objective_cost: float
    compute_throughput: float
    cost_per_compute: float | None
    completion_rate: float
    deadline_violations: float
    electricity_mwh: float
    peak_grid_mw: float
    cooling_energy_mwh: float
    battery_throughput_mwh: float
    battery_degradation_cost: float
    generator_energy_mwh: float
    unserved_energy_mwh: float
    physical_constraint_events: int
    actuator_adjustments: int
    shield_interventions: int
    shield_rejections: int
    simulator_failures: int
    outcome_audit_failures: int
    shield_name: str
    failed: bool
    failure_reason: str | None
    policy_latency_seconds: float
    optimization_runtime_seconds: float
    terminal_battery_energy_mwh: float
    terminal_temperature_c: float
    pending_compute_units: float
    battery_equivalent_cycles: float
    fallback_steps: int
    wall_seconds: float


def scenarios(steps: int = 24) -> dict[str, tuple[SiteInputs, ...]]:
    if steps < 1:
        raise ValueError("steps must be positive")
    normal = tuple(SiteInputs(nondeferrable_compute_units=40.0) for _ in range(steps))
    return {
        "normal": normal,
        "price_spike": tuple(
            replace(x, grid_import_price_per_mwh=600.0 if i >= steps // 2 else 30.0)
            for i, x in enumerate(normal)
        ),
        "heat_wave": tuple(replace(x, ambient_temperature_c=45.0) for x in normal),
        "demand_spike": tuple(
            replace(x, nondeferrable_compute_units=100.0 if i == steps // 2 else 40.0)
            for i, x in enumerate(normal)
        ),
        "grid_outage": tuple(
            replace(x, grid_available=i < steps // 2) for i, x in enumerate(normal)
        ),
        "grid_constraint": tuple(replace(x, grid_import_limit_mw=3.0) for x in normal),
        "generator_failure": tuple(
            replace(x, generator_available=False, grid_import_limit_mw=3.0) for x in normal
        ),
        "compound": tuple(
            replace(x, ambient_temperature_c=45.0, grid_available=False, generator_available=False)
            for x in normal
        ),
    }


def evaluate(
    name: str,
    inputs: Sequence[SiteInputs],
    factory: Callable[[], PolicyProtocol],
    config: SiteConfig | None = None,
    seed: int = 7,
    forecaster: ForecastProviderProtocol | None = None,
    policy_label: str | None = None,
    shield: SafetyShieldProtocol | None = None,
    decision_logger: DecisionLoggerProtocol | None = None,
) -> Metrics:
    sim = SiteSimulator(inputs, config)
    o, _ = sim.reset(seed=seed)
    policy = factory()
    logger = decision_logger if decision_logger is not None else InMemoryDecisionLogger()
    dispatcher = Dispatcher(
        policy,
        shield if shield is not None else DevelopmentAcceptAllShield(),
        logger,
        RulePolicy(sim.config),
        forecaster if forecaster is not None else PersistenceForecastDouble(),
    )
    records = []
    started = perf_counter()
    error = None
    sim_failures = rejections = audit_failures = 0
    optimization_runtime = 0.0
    while not sim.done:
        try:
            dispatcher.horizon_steps = min(12, len(inputs) - sim.index)
            o, _, terminated, _, info = dispatcher.dispatch(o, sim)
            records.append(info)
            if terminated:
                error = "simulator terminated before scenario end"
                sim_failures += 1
            result = getattr(policy, "last_result", None)
            if result is not None:
                optimization_runtime += result.runtime_seconds
        except DispatchOutcomeLoggingFailed as exc:
            o, _, terminated, _, info = exc.result
            records.append(info)
            result = getattr(policy, "last_result", None)
            if result is not None:
                optimization_runtime += result.runtime_seconds
            sim_failures += int(terminated)
            audit_failures += 1
            error = str(exc)
            break
        except DispatchStopped as exc:
            error = str(exc)
            rejections = int(
                dispatcher.last_response is not None
                and dispatcher.last_response.executed_action is None
            )
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            sim_failures += 1
            break

    def total(key: str) -> float:
        return float(sum(r[key] for r in records))

    cost, compute = total("operating_cost"), total("completed_compute_units")
    arrived = (
        sum(
            x.nondeferrable_compute_units + x.deferrable_compute_units
            for x in inputs[: len(records)]
        )
        + sim.config.compute.initial_deferrable_compute_units
    )
    return Metrics(
        name,
        policy_label or type(policy).__name__,
        len(records),
        cost,
        total("objective_cost"),
        compute,
        cost / compute if compute else None,
        compute / arrived if arrived else 1.0,
        total("deadline_missed_compute_units"),
        total("electricity_mwh"),
        max((r["grid_power_mw"] for r in records), default=0.0),
        total("cooling_energy_mwh"),
        total("battery_throughput_mwh"),
        total("battery_degradation_cost"),
        total("generator_energy_mwh"),
        total("unserved_energy_mwh"),
        sum(int(r["thermal_violation"]) + int(r["unserved_energy_mwh"] > 1e-8) for r in records),
        sum(
            sum(
                v.code not in {"high_temperature", "low_temperature", "deadline_missed"}
                for v in r["violations"]
            )
            for r in records
        ),
        int(total("shield_intervention")),
        rejections,
        sim_failures,
        audit_failures,
        dispatcher.last_response.shield_name
        if dispatcher.last_response
        else type(dispatcher.shield).__name__,
        error is not None,
        error,
        total("policy_latency_seconds"),
        optimization_runtime,
        sim.bs.energy_mwh,
        sim.ts.indoor_temperature_c,
        sim.cs.deferrable_backlog_compute_units,
        sim.bs.equivalent_full_cycles,
        sum(r["fallback_reason"] is not None for r in records),
        perf_counter() - started,
    )


def write_reports(results: Sequence[Metrics], directory: str | Path) -> None:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    (target / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    if rows:
        with (target / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Offline synthetic benchmark",
        "",
        "Shield names are recorded in metrics.json/CSV. "
        "DEVELOPMENT doubles provide no safety assurance.",
        "",
        "| Scenario | Policy | Cost | Compute | Missed work | Failed |",
        "|---|---|---:|---:|---:|---|",
    ]
    lines.extend(
        f"| {r.scenario} | {r.policy} | {r.total_operating_cost:.2f} | "
        f"{r.compute_throughput:.2f} | {r.deadline_violations:.2f} | {r.failed} |"
        for r in results
    )
    (target / "summary.md").write_text("\n".join(lines) + "\n")
