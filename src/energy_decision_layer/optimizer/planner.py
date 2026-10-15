"""Receding-horizon MILP baseline (SciPy/HiGHS); no production safety logic."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from math import exp
from time import perf_counter
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp

from energy_decision_layer.contracts import (
    ForecastBundleV1,
    ProposedControlActionV1,
    SiteObservationV1,
)
from energy_decision_layer.sim.site import SiteConfig


def proposal(
    o: SiteObservationV1, values: tuple[float, float, float, float, float], name: str
) -> ProposedControlActionV1:
    return ProposedControlActionV1(
        action_id=uuid4(),
        site_id=o.site_id,
        based_on_state_id=o.state_id,
        created_at_utc=o.timestamp_utc,
        effective_at_utc=o.timestamp_utc,
        expires_at_utc=o.timestamp_utc + timedelta(seconds=o.timestep_seconds),
        compute_power_mw=max(0.0, float(values[0])),
        cooling_power_mw=max(0.0, float(values[1])),
        battery_power_mw=float(values[2]),
        generator_power_mw=max(0.0, float(values[3])),
        grid_power_mw=float(values[4]),
        policy_name=name,
        policy_version="0.1.0",
    )


class RulePolicy:
    def __init__(self, config: SiteConfig | None = None) -> None:
        self.config = config or SiteConfig()

    def propose_action(
        self, observation: SiteObservationV1, forecast: ForecastBundleV1 | None = None
    ) -> ProposedControlActionV1:
        o, c = observation, self.config
        demand = (
            o.workload_demand_compute_units_per_h
            + (o.deferrable_backlog_compute_units + o.critical_backlog_compute_units)
            * 3600
            / o.timestep_seconds
        )
        compute = (
            min(
                c.compute.max_power_mw,
                c.compute.idle_power_mw
                + demand
                / c.compute.max_useful_compute_units_per_hour
                * (c.compute.max_power_mw - c.compute.idle_power_mw),
            )
            if demand
            else 0.0
        )
        cop = max(
            c.thermal.cooling_minimum_cop,
            min(
                c.thermal.cooling_maximum_cop,
                c.thermal.cooling_nominal_cop
                + (c.thermal.cop_reference_ambient_temperature_c - o.ambient_temperature_c)
                * c.thermal.cop_temperature_slope_per_c,
            ),
        )
        cooling = min(
            c.thermal.cooling_max_power_mw,
            max(
                0.0,
                (
                    compute * c.thermal.compute_heat_fraction
                    + (o.facility_temperature_c - c.thermal.initial_temperature_c)
                    * c.thermal.thermal_capacitance_mwh_per_c
                )
                / cop,
            ),
        )
        load = compute + cooling + c.auxiliary_power_mw
        grid = min(load, o.grid_import_limit_mw)
        battery = min(
            max(0.0, load - grid),
            c.battery.maximum_discharge_power_mw,
            max(0.0, o.battery_energy_mwh - c.battery.minimum_energy_mwh)
            * c.battery.discharge_efficiency
            * 3600
            / o.timestep_seconds,
        )
        generator = (
            min(c.generator.maximum_power_mw, max(0.0, load - grid - battery))
            if o.generator_available
            else 0.0
        )
        return proposal(o, (compute, cooling, battery, generator, grid), "rule-baseline")


@dataclass(frozen=True)
class PlanResult:
    status: str
    action: ProposedControlActionV1 | None
    runtime_seconds: float
    message: str
    objective: float | None = None


class PlanningFailure(RuntimeError):
    pass


class MilpPolicy:
    """Aggregate demand defaults to due each interval; explicit deferrable fractions and
    queued deadlines enable earliest-deadline workload shifting without inventing SLAs.

    Includes signed battery flow with exclusive charge/discharge, terminal reserve,
    generator commitment/start costs and ramps, thermal dynamics and cooling lag.
    Shutdown jumps use the subsystem's explicit commitment semantics. Minimum up/down
    durations use the observation's commitment history.
    """

    def __init__(
        self,
        config: SiteConfig | None = None,
        time_limit_seconds: float = 5.0,
        *,
        allow_offline_hindsight: bool = False,
        use_prediction_intervals: bool = True,
    ) -> None:
        self.config = config or SiteConfig()
        if not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
            raise ValueError("time limit must be finite and positive")
        self.use_prediction_intervals = use_prediction_intervals
        self.allow_offline_hindsight = allow_offline_hindsight
        self.time_limit_seconds = time_limit_seconds
        self.last_result: PlanResult | None = None

    def propose_action(
        self, observation: SiteObservationV1, forecast: ForecastBundleV1 | None = None
    ) -> ProposedControlActionV1:
        self.last_result = self.plan(observation, forecast)
        if self.last_result.action is None:
            raise PlanningFailure(f"{self.last_result.status}: {self.last_result.message}")
        return self.last_result.action

    def plan(self, o: SiteObservationV1, forecast: ForecastBundleV1 | None = None) -> PlanResult:
        started = perf_counter()

        def failure(status: str, message: str) -> PlanResult:
            return PlanResult(status, None, perf_counter() - started, message)

        c = self.config
        if forecast is not None:
            if (
                forecast.site_id != o.site_id
                or forecast.issued_at_utc > o.timestamp_utc
                or forecast.valid_from_utc != o.timestamp_utc
                or forecast.cadence_seconds != o.timestep_seconds
                or (forecast.is_perfect_hindsight and not self.allow_offline_hindsight)
            ):
                return failure(
                    "invalid_forecast",
                    "site, issuance, cadence, or horizon mismatch; hindsight is offline only",
                )
            points = forecast.points
        else:
            from energy_decision_layer.contracts import ForecastPointV1

            points = (
                ForecastPointV1(
                    target_timestamp_utc=o.timestamp_utc,
                    workload_demand_compute_units_per_h=o.workload_demand_compute_units_per_h,
                    deferrable_fraction=o.incoming_deferrable_fraction,
                    deferrable_deadline_steps=o.incoming_deferrable_deadline_steps,
                    electricity_price_usd_per_mwh=o.electricity_price_usd_per_mwh,
                    ambient_temperature_c=o.ambient_temperature_c,
                    grid_available=o.grid_available,
                    grid_import_limit_mw=o.grid_import_limit_mw,
                ),
            )
        if self.use_prediction_intervals and any(point.prediction_intervals for point in points):
            from energy_decision_layer.contracts import ForecastSignal

            conservative_points = []
            for point in points:
                payload = point.model_dump()
                for band in point.prediction_intervals:
                    if band.signal in {
                        ForecastSignal.WORKLOAD_DEMAND,
                        ForecastSignal.ELECTRICITY_PRICE,
                        ForecastSignal.AMBIENT_TEMPERATURE,
                    }:
                        payload[band.signal.value] = band.upper_value
                    elif band.signal in {
                        ForecastSignal.GRID_IMPORT_LIMIT,
                        ForecastSignal.ACTIVE_COMPUTE_CAPACITY,
                    }:
                        payload[band.signal.value] = max(0.0, band.lower_value)
                conservative_points.append(type(point).model_validate(payload))
            points = tuple(conservative_points)
        if len(points) > 288:
            return failure("invalid_forecast", "horizon exceeds supported 288 intervals")
        n, width = len(points), 14
        size = n * width
        cost = np.zeros(size)
        low, high = np.zeros(size), np.full(size, np.inf)
        integer = np.zeros(size)
        rows: list[NDArray[np.float64]] = []
        lower: list[float] = []
        upper: list[float] = []

        def add(coefficients: Mapping[int, float], lo: float, hi: float) -> None:
            row = np.zeros(size)
            for k, value in coefficients.items():
                row[k] = value
            rows.append(row)
            lower.append(lo)
            upper.append(hi)

        dt = o.timestep_seconds / 3600
        cc, bc, gc, tc = c.compute, c.battery, c.generator, c.thermal
        rate = cc.max_useful_compute_units_per_hour / (cc.max_power_mw - cc.idle_power_mw)
        alpha = (
            1.0
            if tc.cooling_response_time_hours == 0
            else 1 - exp(-dt / tc.cooling_response_time_hours)
        )
        arrivals = np.zeros(n)
        due = np.zeros(n)
        backlog = o.deferrable_backlog_compute_units
        arrivals[0] += backlog + o.critical_backlog_compute_units
        due[0] += o.critical_backlog_compute_units
        if o.queued_workloads:
            if abs(sum(b.remaining_compute_units for b in o.queued_workloads) - backlog) > 1e-6:
                return failure("invalid_state", "queue total differs from backlog")
            for batch in o.queued_workloads:
                if batch.steps_to_deadline <= n:
                    due[batch.steps_to_deadline - 1] += batch.remaining_compute_units
        else:
            due[0] += backlog  # Legacy aggregate backlog is conservatively due now.
        for t, point in enumerate(points):
            work = point.workload_demand_compute_units_per_h * dt
            arrivals[t] += work
            due[t] += work * (1 - point.deferrable_fraction)
            deadline = t + point.deferrable_deadline_steps - 1
            if deadline < n:
                due[deadline] += work * point.deferrable_fraction
        for t, p in enumerate(points):
            base = t * width
            (
                cp,
                cool,
                charge,
                discharge,
                grid,
                gen,
                energy,
                on,
                mode,
                short,
                temp,
                start,
                served,
                compute_on,
            ) = range(base, base + width)
            prev = base - width
            cap_rate = min(
                cc.max_useful_compute_units_per_hour,
                p.active_compute_capacity_compute_units_per_h
                if p.active_compute_capacity_compute_units_per_h is not None
                else cc.max_useful_compute_units_per_hour,
            )
            if t == 0:
                cap_rate = min(cap_rate, o.active_compute_capacity_compute_units_per_h)
            high[cp] = cc.idle_power_mw + cap_rate / rate
            high[short] = due[t]
            high[cool] = tc.cooling_max_power_mw
            high[charge], high[discharge] = (
                bc.maximum_charge_power_mw,
                bc.maximum_discharge_power_mw,
            )
            high[grid] = (
                min(
                    c.grid_import_limit_mw,
                    p.grid_import_limit_mw
                    if p.grid_import_limit_mw is not None
                    else c.grid_import_limit_mw,
                )
                if p.grid_available
                else 0
            )
            if t == 0:
                high[grid] = min(high[grid], o.grid_import_limit_mw if o.grid_available else 0.0)
            high[gen] = gc.maximum_power_mw if o.generator_available else 0
            low[energy], high[energy] = bc.minimum_energy_mwh, bc.maximum_energy_mwh
            low[temp], high[temp] = tc.minimum_temperature_c, tc.maximum_temperature_c
            integer[[on, mode, start, compute_on]] = 1
            high[[on, mode, start, compute_on]] = 1
            cost[grid] = p.electricity_price_usd_per_mwh * dt
            cost[gen] = (
                gc.fuel_cost_per_mwh_thermal / gc.electrical_efficiency
                + gc.variable_cost_per_mwh_electric
            ) * dt
            cost[charge] = bc.degradation_cost_per_mwh_throughput * bc.charge_efficiency * dt
            cost[discharge] = bc.degradation_cost_per_mwh_throughput / bc.discharge_efficiency * dt
            cost[short] = c.deadline_penalty_per_unit
            cost[start] = gc.startup_cost + gc.shutdown_cost
            cost[on] -= gc.shutdown_cost
            if t:
                cost[prev + 7] += gc.shutdown_cost
            # Balance includes fixed auxiliary; export/renewables conservatively excluded.
            add(
                {grid: 1, gen: 1, discharge: 1, charge: -1, cp: -1, cool: -1},
                c.auxiliary_power_mw,
                c.auxiliary_power_mw,
            )
            add({cp: 1, compute_on: -high[cp]}, -np.inf, 0)
            add({cp: 1, compute_on: -cc.idle_power_mw}, 0, np.inf)
            add({served: 1, cp: -rate * dt, compute_on: cc.idle_power_mw * rate * dt}, -np.inf, 0)
            work_balance = {k * width + offset: 1.0 for k in range(t + 1) for offset in (9, 12)}
            add(work_balance, float(sum(due[: t + 1])), float(sum(arrivals[: t + 1])))
            add({charge: 1, mode: -bc.maximum_charge_power_mw}, -np.inf, 0)
            add(
                {discharge: 1, mode: bc.maximum_discharge_power_mw},
                -np.inf,
                bc.maximum_discharge_power_mw,
            )
            erow = {
                energy: 1,
                charge: -dt * bc.charge_efficiency,
                discharge: dt / bc.discharge_efficiency,
            }
            if t:
                erow[prev + 6] = -1
            add(erow, o.battery_energy_mwh if not t else 0, o.battery_energy_mwh if not t else 0)
            brow = {discharge: 1, charge: -1}
            if t:
                brow[prev + 3], brow[prev + 2] = -1, 1
            bp = o.battery_power_mw if not t else 0
            add(brow, bp - bc.ramp_rate_mw_per_hour * dt, bp + bc.ramp_rate_mw_per_hour * dt)
            add({gen: 1, on: -gc.maximum_power_mw}, -np.inf, 0)
            add({gen: 1, on: -gc.minimum_power_mw}, 0, np.inf)
            srow = {start: 1, on: -1}
            if t:
                srow[prev + 7] = 1
            add(srow, -float(o.generator_is_on) if not t else 0, np.inf)
            # Exact startup indicator; gratuitous starts must not relax ramp bounds.
            add({start: 1, on: -1}, -np.inf, 0)
            previous_on = float(o.generator_is_on) if not t else 0.0
            startup_upper = {start: 1.0}
            stop_nonnegative = {on: 1.0, start: -1.0}
            if t:
                startup_upper[prev + 7] = 1.0
                stop_nonnegative[prev + 7] = -1.0
            add(startup_upper, -np.inf, 1 - previous_on)
            add(stop_nonnegative, -np.inf, previous_on)
            for k in range(t, min(n, t + gc.minimum_up_steps)):
                add({start: 1, k * width + 7: -1}, -np.inf, 0)
            # stop = previous_on + startup - on; enforce minimum downtime.
            for k in range(t, min(n, t + gc.minimum_down_steps)):
                downrow: dict[int, float] = {start: 1.0}
                downrow[on] = downrow.get(on, 0.0) - 1
                downrow[k * width + 7] = downrow.get(k * width + 7, 0.0) + 1
                if t:
                    downrow[prev + 7] = downrow.get(prev + 7, 0.0) + 1
                add(downrow, -np.inf, 1 - previous_on)
            if o.generator_available:
                if o.generator_is_on and t < gc.minimum_up_steps - o.generator_consecutive_on_steps:
                    low[on] = 1
                if (
                    not o.generator_is_on
                    and t < gc.minimum_down_steps - o.generator_consecutive_off_steps
                ):
                    high[on] = 0
            grow = {gen: 1, start: -gc.minimum_power_mw}
            if t:
                grow[prev + 5] = -1
            add(
                grow,
                -np.inf,
                gc.ramp_up_rate_mw_per_hour * dt + (o.generator_power_mw if not t else 0),
            )
            down = {gen: -1, on: gc.maximum_power_mw}
            if t:
                down[prev + 5] = 1
            add(
                down,
                -np.inf,
                gc.maximum_power_mw
                + gc.ramp_down_rate_mw_per_hour * dt
                - (o.generator_power_mw if not t else 0),
            )
            # Cooling actual endpoint, same discrete lag/ramp approximation as simulator.
            crow: dict[int, float] = {cool: 1.0}
            if t:
                crow[prev + 1] = -(1 - alpha)
            initial_cooling = (1 - alpha) * o.cooling_power_mw if not t else 0
            add(crow, initial_cooling, initial_cooling + alpha * tc.cooling_max_power_mw)
            ramp = {cool: 1}
            if t:
                ramp[prev + 1] = -1
            initial_cooling = o.cooling_power_mw if not t else 0
            add(
                ramp,
                initial_cooling - tc.cooling_ramp_rate_mw_per_hour * dt,
                initial_cooling + tc.cooling_ramp_rate_mw_per_hour * dt,
            )
            cop = max(
                tc.cooling_minimum_cop,
                min(
                    tc.cooling_maximum_cop,
                    tc.cooling_nominal_cop
                    + (tc.cop_reference_ambient_temperature_c - p.ambient_temperature_c)
                    * tc.cop_temperature_slope_per_c,
                ),
            )
            factor = dt / tc.thermal_capacitance_mwh_per_c
            tr = {temp: 1, cp: -factor * tc.compute_heat_fraction, cool: factor * cop}
            decay = 1 - factor * tc.envelope_conductance_mw_per_c
            if t:
                tr[prev + 10] = -decay
            rhs = factor * tc.envelope_conductance_mw_per_c * p.ambient_temperature_c
            if not t:
                rhs += decay * o.facility_temperature_c
            add(tr, rhs, rhs)
        low[(n - 1) * width + 6] = max(
            bc.minimum_energy_mwh, bc.initial_soc_fraction * bc.energy_capacity_mwh
        )
        try:
            result = milp(
                cost,
                integrality=integer,
                bounds=Bounds(low, high),
                constraints=LinearConstraint(np.array(rows), np.array(lower), np.array(upper)),
                options={"time_limit": self.time_limit_seconds},
            )
        except (ValueError, RuntimeError) as exc:
            return failure("solver_error", str(exc))
        if result.status != 0 or result.x is None:
            return failure(
                {1: "timeout", 2: "infeasible", 3: "unbounded"}.get(result.status, "solver_error"),
                str(result.message),
            )
        z = result.x
        # Independently check the returned solution, including mocked/corrupt solver outputs.
        if np.shape(z) != (size,):
            return failure("invalid_solution", "solver output has an unexpected shape")
        activity = np.array(rows) @ z
        if (
            not np.all(np.isfinite(z))
            or np.any(z < low - 1e-6)
            or np.any(z > high + 1e-6)
            or np.any(activity < np.array(lower) - 1e-6)
            or np.any(activity > np.array(upper) + 1e-6)
            or np.any(np.abs(z[integer == 1] - np.round(z[integer == 1])) > 1e-6)
        ):
            return failure(
                "invalid_solution", "solver output failed independent feasibility checks"
            )
        target = (z[1] - (1 - alpha) * o.cooling_power_mw) / alpha
        action = proposal(o, (z[0], target, z[3] - z[2], z[5], z[4]), "milp-baseline")
        return PlanResult(
            "optimal", action, perf_counter() - started, str(result.message), float(result.fun)
        )
