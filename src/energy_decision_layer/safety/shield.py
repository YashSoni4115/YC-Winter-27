"""Coupled one-step projection using the simulator's authoritative physics.

This is a simulation safety envelope, not certified plant protection. Cooling
commands and realized cooling differ because of the existing response lag.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import numpy as np
from scipy.optimize import linprog

from energy_decision_layer.contracts import (
    ExecutedControlActionV1,
    ProposedControlActionV1,
    SafetyDecisionStatus,
    SafetyReasonCode,
    SafetyResponseV1,
    SiteObservationV1,
)
from energy_decision_layer.contracts.common import (
    ConstraintKind,
    ConstraintMetadataV1,
    DataQualityFlag,
)
from energy_decision_layer.sim import SiteConfig
from energy_decision_layer.sim.battery import BatteryState, BatterySystem
from energy_decision_layer.sim.generator import GeneratorState, GeneratorSystem
from energy_decision_layer.sim.thermal import ThermalState, ThermalSystem

FIELDS = (
    "compute_power_mw",
    "cooling_power_mw",
    "battery_power_mw",
    "generator_power_mw",
    "grid_power_mw",
    "renewable_curtailment_mw",
)


class SafetyShield:
    name = "coupled-physics-shield"
    version = "1.0.0"

    def __init__(self, config: SiteConfig | None = None) -> None:
        self.config = config or SiteConfig()

    def evaluate(
        self,
        proposal: ProposedControlActionV1,
        observation: SiteObservationV1,
        *,
        now: datetime | None = None,
    ) -> SafetyResponseV1:
        o, p, c = observation, proposal, self.config
        response_id = uuid4()

        def reject(code: SafetyReasonCode, message: str) -> SafetyResponseV1:
            action_id = getattr(p, "action_id", None)
            timestamp = getattr(o, "timestamp_utc", None)
            return SafetyResponseV1(
                response_id=response_id,
                action_id=action_id if isinstance(action_id, UUID) else uuid4(),
                site_id=c.site_id,
                decided_at_utc=(
                    timestamp
                    if isinstance(timestamp, datetime) and timestamp.tzinfo is not None
                    else datetime.now(timezone.utc)
                ),
                shield_name=self.name,
                shield_version=self.version,
                status=SafetyDecisionStatus.REJECTED,
                reason_codes=(code,),
                reason_message=message,
            )

        try:
            o = SiteObservationV1.model_validate_json(o.model_dump_json())
        except (ValueError, TypeError, AttributeError):
            return reject(
                SafetyReasonCode.MISSING_TELEMETRY, "Invalid or missing essential telemetry"
            )
        try:
            p = ProposedControlActionV1.model_validate_json(p.model_dump_json())
        except (ValueError, TypeError, AttributeError):
            return reject(SafetyReasonCode.INVALID_ACTION, "Invalid/nonfinite action")
        end = o.timestamp_utc + timedelta(seconds=o.timestep_seconds)
        if (
            p.site_id != o.site_id
            or o.site_id != c.site_id
            or p.based_on_state_id != o.state_id
            or p.created_at_utc > o.timestamp_utc
            or o.timestep_seconds != c.timestep_seconds
            or (
                now is not None
                and (
                    now.tzinfo is None
                    or now.utcoffset() is None
                    or not 0 <= (now - o.timestamp_utc).total_seconds() <= o.timestep_seconds
                )
            )
        ):
            return reject(SafetyReasonCode.STALE_STATE, "Site, state, clock, or cadence mismatch")
        if p.effective_at_utc > o.timestamp_utc:
            return reject(SafetyReasonCode.ACTION_NOT_YET_VALID, "Action is not effective yet")
        if p.expires_at_utc < end:
            return reject(
                SafetyReasonCode.ACTION_EXPIRED, "Validity must cover the entire interval"
            )
        if any(
            flag in {DataQualityFlag.MISSING, DataQualityFlag.SUSPECT}
            for flag in o.quality_flags.values()
        ):
            return reject(
                SafetyReasonCode.MISSING_TELEMETRY, "Telemetry has missing/suspect readings"
            )
        if (
            not math.isclose(o.battery_capacity_mwh, c.battery.energy_capacity_mwh)
            or not c.battery.minimum_energy_mwh - 1e-8
            <= o.battery_energy_mwh
            <= c.battery.maximum_energy_mwh + 1e-8
            or not c.thermal.minimum_temperature_c
            <= o.facility_temperature_c
            <= c.thermal.maximum_temperature_c
            or not -c.battery.maximum_charge_power_mw
            <= o.battery_power_mw
            <= c.battery.maximum_discharge_power_mw
            or not 0.0 <= o.cooling_power_mw <= c.thermal.cooling_max_power_mw
            or not 0.0 <= o.generator_power_mw <= c.generator.maximum_power_mw
        ):
            return reject(
                SafetyReasonCode.MISSING_TELEMETRY, "State lies outside configured envelope"
            )
        try:
            return self._project(p, o, response_id)
        except Exception:
            return reject(
                SafetyReasonCode.SHIELD_INTERNAL_ERROR, "Safety calculation failed closed"
            )

    def _project(
        self, p: ProposedControlActionV1, o: SiteObservationV1, response_id: object
    ) -> SafetyResponseV1:
        # A UUID annotation is used here rather than accepting an executable token.
        assert isinstance(response_id, UUID)
        c, dt = self.config, o.timestep_seconds / 3600
        battery, thermal, generator = (
            BatterySystem(c.battery),
            ThermalSystem(c.thermal),
            GeneratorSystem(c.generator),
        )
        bs = BatteryState(o.battery_energy_mwh, o.battery_power_mw)
        ts = ThermalState(o.facility_temperature_c, o.cooling_power_mw)
        gs = GeneratorState(
            o.generator_is_on,
            o.generator_power_mw,
            o.generator_consecutive_on_steps,
            o.generator_consecutive_off_steps,
        )
        gen = generator.step(gs, p.generator_power_mw, dt, available=o.generator_available)
        g = gen.actual_power_mw
        blo = battery.constrain_power(bs, -c.battery.maximum_charge_power_mw, dt)[0]
        bhi = battery.constrain_power(bs, c.battery.maximum_discharge_power_mw, dt)[0]
        clo = thermal.constrain_power(ts, 0.0, dt)[0]
        chi = thermal.constrain_power(ts, c.thermal.cooling_max_power_mw, dt)[0]
        fleet = min(
            c.compute.max_useful_compute_units_per_hour,
            o.active_compute_capacity_compute_units_per_h,
        )
        compute_cap = (
            (
                c.compute.idle_power_mw
                + fleet
                / c.compute.max_useful_compute_units_per_hour
                * (c.compute.max_power_mw - c.compute.idle_power_mw)
            )
            if fleet
            else 0.0
        )
        gridlo = -min(c.grid_export_limit_mw, o.grid_export_limit_mw) if o.grid_available else 0.0
        gridhi = min(c.grid_import_limit_mw, o.grid_import_limit_mw) if o.grid_available else 0.0
        # x = [compute, realized cooling, battery, grid, four absolute deviations].
        target = np.array(
            [
                p.compute_power_mw,
                thermal.constrain_power(ts, p.cooling_power_mw, dt)[0],
                p.battery_power_mw,
                p.grid_power_mw,
            ]
        )
        cop = thermal.cooling_cop(o.ambient_temperature_c)
        k = dt / c.thermal.thermal_capacitance_mwh_per_c
        passive = c.thermal.envelope_conductance_mw_per_c * (
            o.facility_temperature_c - o.ambient_temperature_c
        )
        temp0 = o.facility_temperature_c - k * passive
        heat = np.array([k * c.thermal.compute_heat_fraction, -k * cop, 0.0, 0.0])
        rows = [
            np.r_[np.array([1.0, 1.0, -1.0, -1.0]), np.zeros(4)],
            np.r_[heat, np.zeros(4)],
            np.r_[-heat, np.zeros(4)],
        ]
        rhs = [
            g - c.auxiliary_power_mw,
            c.thermal.maximum_temperature_c - temp0,
            temp0 - c.thermal.minimum_temperature_c,
        ]
        for i in range(4):
            row = np.zeros(8)
            row[i], row[4 + i] = 1.0, -1.0
            rows.extend([row, np.r_[-row[:4], row[4:]]])
            rhs.extend([target[i], -target[i]])
        result = linprog(
            np.r_[np.zeros(4), [100.0, 20.0, 2.0, 1.0]],
            A_ub=np.array(rows),
            b_ub=np.array(rhs),
            bounds=[(0, compute_cap), (clo, chi), (blo, bhi), (gridlo, gridhi)] + [(0, None)] * 4,
            method="highs",
        )
        if not result.success or result.x is None or not np.isfinite(result.x).all():
            return SafetyResponseV1(
                response_id=response_id,
                action_id=p.action_id,
                site_id=p.site_id,
                decided_at_utc=o.timestamp_utc,
                shield_name=self.name,
                shield_version=self.version,
                status=SafetyDecisionStatus.REJECTED,
                reason_codes=(SafetyReasonCode.POWER_BALANCE, SafetyReasonCode.TEMPERATURE_LIMIT),
                reason_message=(
                    "No feasible coupled electrical/thermal action; local protection required"
                ),
            )
        compute, cooling, b, grid = (float(v) for v in result.x[:4])
        alpha = (
            1.0
            if c.thermal.cooling_response_time_hours <= 1e-9
            else 1 - math.exp(-dt / c.thermal.cooling_response_time_hours)
        )
        command = min(
            c.thermal.cooling_max_power_mw,
            max(0.0, o.cooling_power_mw + (cooling - o.cooling_power_mw) / alpha),
        )
        # Preserve an unchanged valid command, including one hitting the ramp plateau.
        if abs(cooling - target[1]) < 1e-9 and p.cooling_power_mw <= c.thermal.cooling_max_power_mw:
            command = p.cooling_power_mw
        values = dict(zip(FIELDS, (compute, command, b, g, grid, 0.0), strict=True))
        for key in FIELDS:
            if abs(values[key] - getattr(p, key)) <= 1e-10:
                values[key] = float(getattr(p, key))
        # Independent authoritative transition checks AFTER modification.
        tb = battery.step(bs, values["battery_power_mw"], dt)
        tg = generator.step(gs, values["generator_power_mw"], dt, available=o.generator_available)
        tt = thermal.step(ts, values["cooling_power_mw"], compute, o.ambient_temperature_c, dt)
        supply = grid + tg.actual_power_mw + tb.actual_power_mw
        load = compute + tt.actual_cooling_power_mw + c.auxiliary_power_mw
        next_temp = tt.state.indoor_temperature_c
        if (
            abs(tb.actual_power_mw - b) > 1e-7
            or abs(tg.actual_power_mw - g) > 1e-7
            or abs(tt.actual_cooling_power_mw - cooling) > 1e-7
            or supply < load - 1e-7
            or not c.thermal.minimum_temperature_c - 1e-7
            <= next_temp
            <= c.thermal.maximum_temperature_c + 1e-7
        ):
            raise RuntimeError("Projected action failed authoritative coupled recheck")
        changed = tuple(k for k in FIELDS if abs(values[k] - getattr(p, k)) > 1e-12)
        final_checks = (
            ConstraintMetadataV1(
                constraint_id="battery.energy",
                kind=ConstraintKind.BATTERY_SOC,
                description="Next stored energy",
                unit="MWh",
                lower_bound=c.battery.minimum_energy_mwh,
                upper_bound=c.battery.maximum_energy_mwh,
                observed_value=tb.state.energy_mwh,
                satisfied=True,
            ),
            ConstraintMetadataV1(
                constraint_id="thermal.next",
                kind=ConstraintKind.TEMPERATURE,
                description="Next facility temperature including cooling lag",
                unit="degC",
                lower_bound=c.thermal.minimum_temperature_c,
                upper_bound=c.thermal.maximum_temperature_c,
                observed_value=next_temp,
                satisfied=True,
            ),
            ConstraintMetadataV1(
                constraint_id="bus.balance",
                kind=ConstraintKind.POWER_BALANCE,
                description="Supply minus loads; positive surplus uses modeled spill sink",
                unit="MW",
                lower_bound=0.0,
                observed_value=supply - load,
                satisfied=True,
            ),
        )
        requested_battery = battery.step(bs, p.battery_power_mw, dt)
        requested_thermal = thermal.step(
            ts,
            p.cooling_power_mw,
            min(compute_cap, p.compute_power_mw),
            o.ambient_temperature_c,
            dt,
        )
        requested_balance = (
            p.grid_power_mw
            + g
            + requested_battery.actual_power_mw
            - p.compute_power_mw
            - requested_thermal.actual_cooling_power_mw
            - c.auxiliary_power_mw
        )
        checks_list = []
        reason_list = []

        def check(
            identifier: str,
            kind: ConstraintKind,
            value: float,
            unit: str,
            lower: float | None,
            upper: float | None,
            code: SafetyReasonCode,
        ) -> None:
            valid = (lower is None or value >= lower - 1e-8) and (
                upper is None or value <= upper + 1e-8
            )
            checks_list.append(
                ConstraintMetadataV1(
                    constraint_id="proposal." + identifier,
                    kind=kind,
                    description="Original proposal constraint",
                    unit=unit,
                    lower_bound=lower,
                    upper_bound=upper,
                    observed_value=value,
                    satisfied=valid,
                )
            )
            if not valid:
                reason_list.append(code)

        check(
            "compute",
            ConstraintKind.COMPUTE_CAPACITY,
            p.compute_power_mw,
            "MW",
            0.0,
            compute_cap,
            SafetyReasonCode.COMPUTE_CAPACITY_LIMIT,
        )
        check(
            "cooling",
            ConstraintKind.COOLING_CAPACITY,
            p.cooling_power_mw,
            "MW",
            0.0,
            c.thermal.cooling_max_power_mw,
            SafetyReasonCode.COOLING_CAPACITY_LIMIT,
        )
        check(
            "battery.rating",
            ConstraintKind.BATTERY_POWER,
            p.battery_power_mw,
            "MW",
            -c.battery.maximum_charge_power_mw,
            c.battery.maximum_discharge_power_mw,
            SafetyReasonCode.BATTERY_POWER_LIMIT,
        )
        raw_energy = (
            o.battery_energy_mwh - p.battery_power_mw * dt / c.battery.discharge_efficiency
            if p.battery_power_mw >= 0
            else o.battery_energy_mwh - p.battery_power_mw * dt * c.battery.charge_efficiency
        )
        check(
            "battery.energy",
            ConstraintKind.BATTERY_SOC,
            raw_energy,
            "MWh",
            c.battery.minimum_energy_mwh,
            c.battery.maximum_energy_mwh,
            SafetyReasonCode.BATTERY_SOC_LIMIT,
        )
        check(
            "battery.ramp",
            ConstraintKind.BATTERY_POWER,
            p.battery_power_mw,
            "MW",
            o.battery_power_mw - c.battery.ramp_rate_mw_per_hour * dt,
            o.battery_power_mw + c.battery.ramp_rate_mw_per_hour * dt,
            SafetyReasonCode.BATTERY_POWER_LIMIT,
        )
        check(
            "grid.import",
            ConstraintKind.GRID_IMPORT,
            p.grid_power_mw,
            "MW",
            None,
            gridhi,
            SafetyReasonCode.GRID_IMPORT_LIMIT,
        )
        check(
            "grid.export",
            ConstraintKind.GRID_EXPORT,
            p.grid_power_mw,
            "MW",
            gridlo,
            None,
            SafetyReasonCode.GRID_EXPORT_LIMIT,
        )
        if not o.grid_available and abs(p.grid_power_mw) > 1e-8:
            reason_list.append(SafetyReasonCode.GRID_UNAVAILABLE)
        check(
            "generator",
            ConstraintKind.GENERATOR_POWER,
            p.generator_power_mw,
            "MW",
            0.0,
            c.generator.maximum_power_mw if o.generator_available else 0.0,
            SafetyReasonCode.GENERATOR_POWER_LIMIT,
        )
        if abs(g - p.generator_power_mw) > 1e-8:
            reason_list.append(
                SafetyReasonCode.GENERATOR_RAMP_LIMIT
                if any(
                    "ramp" in v.code or "time" in v.code or v.code == "startup_output_limit"
                    for v in gen.violations
                )
                else SafetyReasonCode.GENERATOR_POWER_LIMIT
            )
        check(
            "thermal.next",
            ConstraintKind.TEMPERATURE,
            requested_thermal.state.indoor_temperature_c,
            "degC",
            c.thermal.minimum_temperature_c,
            c.thermal.maximum_temperature_c,
            SafetyReasonCode.TEMPERATURE_LIMIT,
        )
        check(
            "bus.balance",
            ConstraintKind.POWER_BALANCE,
            requested_balance,
            "MW",
            0.0,
            None,
            SafetyReasonCode.POWER_BALANCE,
        )
        check(
            "renewables",
            ConstraintKind.OTHER,
            p.renewable_curtailment_mw,
            "MW",
            0.0,
            0.0,
            SafetyReasonCode.INVALID_ACTION,
        )
        if changed and not reason_list:
            reason_list.append(SafetyReasonCode.POWER_BALANCE)
        codes = tuple(dict.fromkeys(reason_list)) if changed else ()
        status = SafetyDecisionStatus.MODIFIED if changed else SafetyDecisionStatus.ACCEPTED
        action = ExecutedControlActionV1(
            execution_id=uuid4(),
            source_action=p,
            safety_response_id=response_id,
            authorization_status=status,
            authorized_at_utc=o.timestamp_utc,
            effective_at_utc=p.effective_at_utc,
            expires_at_utc=p.expires_at_utc,
            modified_fields=changed,
            compute_power_mw=values["compute_power_mw"],
            cooling_power_mw=values["cooling_power_mw"],
            battery_power_mw=values["battery_power_mw"],
            generator_power_mw=values["generator_power_mw"],
            grid_power_mw=values["grid_power_mw"],
            renewable_curtailment_mw=values["renewable_curtailment_mw"],
        )
        return SafetyResponseV1(
            response_id=response_id,
            action_id=p.action_id,
            site_id=p.site_id,
            decided_at_utc=o.timestamp_utc,
            shield_name=self.name,
            shield_version=self.version,
            status=status,
            reason_codes=codes or (SafetyReasonCode.WITHIN_CONSTRAINTS,),
            reason_message=(
                "Projected coupled constraints; changed: " + ", ".join(changed)
                if changed
                else "All configured one-step constraints satisfied"
            ),
            evaluated_constraints=tuple(checks_list) + final_checks,
            executed_action=action,
        )
