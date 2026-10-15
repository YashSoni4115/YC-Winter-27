"""Deterministic single-bus site; local physics is independent of policy and safety."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from energy_decision_layer.contracts import (
    ExecutedControlActionV1,
    GridStressStatus,
    SiteObservationV1,
)
from energy_decision_layer.contracts.telemetry import QueuedWorkloadV1

from .battery import BatteryConfig, BatterySystem
from .common import SiteAction, SiteInputs, require_nonnegative, require_positive
from .compute import ComputeConfig, ComputeSystem
from .generator import GeneratorConfig, GeneratorSystem
from .thermal import ThermalConfig, ThermalSystem


@dataclass(frozen=True)
class SiteConfig:
    site_id: str = "synthetic-site"
    timestep_seconds: float = 300.0
    grid_import_limit_mw: float = 15.0
    grid_export_limit_mw: float = 0.0
    auxiliary_power_mw: float = 0.2
    deadline_penalty_per_unit: float = 100.0
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)

    def __post_init__(self) -> None:
        require_positive("timestep_seconds", self.timestep_seconds)
        if (
            self.thermal.envelope_conductance_mw_per_c * self.timestep_seconds / 3600
            > self.thermal.thermal_capacitance_mwh_per_c
        ):
            raise ValueError("timestep exceeds stable thermal integration interval")
        for name in (
            "grid_import_limit_mw",
            "grid_export_limit_mw",
            "auxiliary_power_mw",
            "deadline_penalty_per_unit",
        ):
            require_nonnegative(name, getattr(self, name))


class SiteSimulator:
    def __init__(
        self,
        inputs: Sequence[SiteInputs],
        config: SiteConfig | None = None,
        start: datetime | None = None,
    ) -> None:
        if not inputs:
            raise ValueError("at least one input interval is required")
        self.inputs = tuple(inputs)
        self.config = config or SiteConfig()
        self.start = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
        if self.start.tzinfo is None or self.start.utcoffset() is None:
            raise ValueError("start must be timezone-aware")
        self.start = self.start.astimezone(timezone.utc)
        self.battery = BatterySystem(self.config.battery)
        self.compute = ComputeSystem(self.config.compute)
        self.generator = GeneratorSystem(self.config.generator)
        self.thermal = ThermalSystem(self.config.thermal)
        self.reset()

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[SiteObservationV1, dict[str, Any]]:
        if options:
            raise ValueError("reset options are not supported")
        self.episode_id = uuid4()
        self.index = 0
        self.bs = self.battery.initial_state()
        self.cs = self.compute.initial_state()
        self.gs = self.generator.initial_state()
        self.ts = self.thermal.initial_state()
        self.last_grid = 0.0
        self.last_auxiliary = 0.0
        self.last_unserved = 0.0
        self.done = False
        return self.observation(), {"seed": seed, "deterministic": True}

    @property
    def current_inputs(self) -> SiteInputs:
        return self.inputs[min(self.index, len(self.inputs) - 1)]

    def observation(self) -> SiteObservationV1:
        x, c = self.current_inputs, self.config
        dt = c.timestep_seconds / 3600
        return SiteObservationV1(
            site_id=c.site_id,
            state_id=f"episode:{self.episode_id}:{self.index}",
            timestamp_utc=self.start + timedelta(seconds=self.index * c.timestep_seconds),
            step_index=self.index,
            timestep_seconds=c.timestep_seconds,
            workload_demand_compute_units_per_h=(
                x.nondeferrable_compute_units + x.deferrable_compute_units
            )
            / dt,
            active_compute_capacity_compute_units_per_h=min(
                c.compute.max_useful_compute_units_per_hour,
                x.active_compute_capacity_compute_units_per_hour
                if x.active_compute_capacity_compute_units_per_hour is not None
                else c.compute.max_useful_compute_units_per_hour,
            ),
            critical_backlog_compute_units=0.0,
            queued_workloads=tuple(
                QueuedWorkloadV1(
                    remaining_compute_units=b.remaining_compute_units,
                    steps_to_deadline=b.steps_to_deadline,
                )
                for b in self.cs.deferrable_batches
            ),
            incoming_deferrable_fraction=x.deferrable_compute_units
            / max(1e-12, x.nondeferrable_compute_units + x.deferrable_compute_units),
            incoming_deferrable_deadline_steps=x.deferrable_deadline_steps,
            deferrable_backlog_compute_units=self.cs.deferrable_backlog_compute_units,
            completed_compute_units=self.cs.cumulative_completed_compute_units,
            deadline_violated_compute_units=self.cs.cumulative_deadline_missed_compute_units,
            compute_power_mw=self.cs.previous_power_mw,
            cooling_power_mw=self.ts.cooling_power_mw,
            auxiliary_power_mw=self.last_auxiliary,
            facility_temperature_c=self.ts.indoor_temperature_c,
            ambient_temperature_c=x.ambient_temperature_c,
            battery_energy_mwh=self.bs.energy_mwh,
            battery_capacity_mwh=c.battery.energy_capacity_mwh,
            battery_state_of_charge=self.bs.soc_fraction(c.battery),
            battery_power_mw=self.bs.power_mw,
            generator_power_mw=self.gs.power_mw if x.generator_available else 0.0,
            generator_is_on=self.gs.is_on and x.generator_available,
            generator_available=x.generator_available,
            generator_consecutive_on_steps=self.gs.consecutive_on_steps,
            generator_consecutive_off_steps=self.gs.consecutive_off_steps,
            grid_power_mw=self.last_grid if x.grid_available else 0.0,
            grid_available=x.grid_available,
            grid_stress_status=GridStressStatus.OUTAGE
            if not x.grid_available
            else GridStressStatus.ELEVATED
            if x.grid_stress
            else GridStressStatus.NORMAL,
            grid_import_limit_mw=min(
                c.grid_import_limit_mw,
                x.grid_import_limit_mw
                if x.grid_import_limit_mw is not None
                else c.grid_import_limit_mw,
            )
            if x.grid_available
            else 0.0,
            grid_export_limit_mw=min(
                c.grid_export_limit_mw,
                x.grid_export_limit_mw
                if x.grid_export_limit_mw is not None
                else c.grid_export_limit_mw,
            )
            if x.grid_available
            else 0.0,
            electricity_price_usd_per_mwh=x.grid_import_price_per_mwh,
            unserved_power_mw=self.last_unserved,
        )

    def step(
        self, action: ExecutedControlActionV1
    ) -> tuple[SiteObservationV1, float, bool, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("episode finished; call reset")
        if not isinstance(action, ExecutedControlActionV1):
            raise TypeError("step requires a shield-authorized ExecutedControlActionV1")
        action = ExecutedControlActionV1.model_validate_json(action.model_dump_json())
        obs = self.observation()
        end = obs.timestamp_utc + timedelta(seconds=self.config.timestep_seconds)
        if (
            action.site_id != obs.site_id
            or action.based_on_state_id != obs.state_id
            or not action.effective_at_utc <= obs.timestamp_utc < end <= action.expires_at_utc
        ):
            raise ValueError("action has wrong site, stale state, or insufficient validity")
        if action.renewable_curtailment_mw:
            raise ValueError("renewables are not configured in this topology")
        return self.advance(
            SiteAction(**{k: getattr(action, k) for k in SiteAction.__dataclass_fields__})
        )

    def advance(
        self, action: SiteAction
    ) -> tuple[SiteObservationV1, float, bool, bool, dict[str, Any]]:
        """Physics-only entry point for local tests/Gym; never a live actuation API."""
        if self.done:
            raise RuntimeError("episode finished; call reset")
        c, x, obs = self.config, self.current_inputs, self.observation()
        dt = c.timestep_seconds / 3600
        g = self.generator.step(
            self.gs, action.generator_power_mw, dt, available=x.generator_available
        )
        bp, _ = self.battery.constrain_power(self.bs, action.battery_power_mw, dt)
        grid = min(obs.grid_import_limit_mw, max(-obs.grid_export_limit_mw, action.grid_power_mw))
        # Exports and charging cannot consume unavailable bus energy.
        grid = max(grid, -max(0.0, g.actual_power_mw + max(0.0, bp)))
        supply = grid + g.actual_power_mw + max(0.0, bp)
        b = self.battery.step(
            self.bs, action.battery_power_mw, dt, minimum_bus_power_mw=-max(0.0, supply)
        )
        available = max(0.0, grid + g.actual_power_mw + b.actual_power_mw)
        auxiliary = min(c.auxiliary_power_mw, available)
        available -= auxiliary
        cooling, _ = self.thermal.constrain_power(self.ts, action.cooling_power_mw, dt)
        delivered_cooling = min(cooling, available)
        available -= delivered_cooling
        compute, _ = self.compute.constrain_power(action.compute_power_mw)
        fleet_cap = c.compute.max_power_mw
        if x.active_compute_capacity_compute_units_per_hour is not None:
            fleet_cap = c.compute.idle_power_mw + min(
                1.0,
                x.active_compute_capacity_compute_units_per_hour
                / c.compute.max_useful_compute_units_per_hour,
            ) * (c.compute.max_power_mw - c.compute.idle_power_mw)
        delivered_compute = min(compute, available, fleet_cap)
        comp = self.compute.step(
            self.cs,
            action.compute_power_mw,
            dt,
            x.nondeferrable_compute_units,
            x.deferrable_compute_units,
            x.deferrable_deadline_steps,
            maximum_delivered_power_mw=delivered_compute,
        )
        thermal = self.thermal.step(
            self.ts,
            action.cooling_power_mw,
            comp.actual_power_mw,
            x.ambient_temperature_c,
            dt,
            maximum_delivered_cooling_power_mw=delivered_cooling,
        )
        load = auxiliary + thermal.actual_cooling_power_mw + comp.actual_power_mw
        net_supply = grid + g.actual_power_mw + b.actual_power_mw
        spill = max(0.0, net_supply - load)
        residual = net_supply - load - spill
        unserved = (
            c.auxiliary_power_mw
            - auxiliary
            + cooling
            - delivered_cooling
            + compute
            - delivered_compute
        )
        grid_cost = dt * (
            max(0.0, grid) * x.grid_import_price_per_mwh
            + min(0.0, grid) * x.grid_export_price_per_mwh
        )
        operating_cost = (
            grid_cost + g.operating_cost + g.startup_cost + g.shutdown_cost + b.degradation_cost
        )
        penalty = comp.deadline_missed_compute_units * c.deadline_penalty_per_unit
        violations = b.violations + g.violations + comp.violations + thermal.violations
        self.bs, self.gs, self.cs, self.ts = b.state, g.state, comp.state, thermal.state
        self.last_grid, self.last_auxiliary, self.last_unserved = grid, auxiliary, unserved
        self.index += 1
        terminated = not (-273.15 <= self.ts.indoor_temperature_c <= 80.0)
        truncated = self.index >= len(self.inputs) and not terminated
        self.done = terminated or truncated
        info = dict(
            operating_cost=operating_cost,
            objective_cost=operating_cost + penalty,
            completed_compute_units=comp.completed_compute_units,
            deadline_missed_compute_units=comp.deadline_missed_compute_units,
            electricity_mwh=max(grid, 0.0) * dt,
            grid_power_mw=grid,
            cooling_energy_mwh=thermal.actual_cooling_power_mw * dt,
            generator_energy_mwh=g.generated_energy_mwh,
            generator_fuel_mwh=g.fuel_energy_mwh_thermal,
            battery_throughput_mwh=b.cell_throughput_mwh,
            battery_degradation_cost=b.degradation_cost,
            battery_conversion_loss_mwh=b.conversion_loss_mwh,
            power_balance_residual_mw=residual,
            unserved_energy_mwh=unserved * dt,
            spilled_energy_mwh=spill * dt,
            violations=violations,
            thermal_violation=any(v.component == "thermal" for v in violations),
            requested_action=action.as_dict(),
            applied_action=SiteAction(
                comp.actual_power_mw,
                thermal.actual_cooling_power_mw,
                b.actual_power_mw,
                g.actual_power_mw,
                grid,
            ).as_dict(),
        )
        return self.observation(), -(operating_cost + penalty), terminated, truncated, info
