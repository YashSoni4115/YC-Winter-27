"""Cooling-plant dynamics and a lumped data-hall thermal model."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .common import (
    NUMERICAL_EPSILON,
    ConstraintViolation,
    clip,
    require_finite,
    require_nonnegative,
    require_positive,
)


@dataclass(frozen=True)
class ThermalConfig:
    initial_temperature_c: float = 24.0
    minimum_temperature_c: float = 15.0
    maximum_temperature_c: float = 32.0
    thermal_capacitance_mwh_per_c: float = 4.0
    envelope_conductance_mw_per_c: float = 0.08
    compute_heat_fraction: float = 0.97
    cooling_max_power_mw: float = 4.0
    cooling_nominal_cop: float = 4.0
    cooling_minimum_cop: float = 1.2
    cooling_maximum_cop: float = 7.0
    cop_reference_ambient_temperature_c: float = 25.0
    cop_temperature_slope_per_c: float = 0.06
    cooling_ramp_rate_mw_per_hour: float = 20.0
    cooling_response_time_hours: float = 0.05
    initial_cooling_power_mw: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "initial_temperature_c",
            "minimum_temperature_c",
            "maximum_temperature_c",
            "cop_reference_ambient_temperature_c",
        ):
            require_finite(name, getattr(self, name))
        if self.minimum_temperature_c >= self.maximum_temperature_c:
            raise ValueError("minimum_temperature_c must be below maximum_temperature_c")
        if not (
            self.minimum_temperature_c <= self.initial_temperature_c <= self.maximum_temperature_c
        ):
            raise ValueError("initial_temperature_c must be inside thermal limits")
        require_positive("thermal_capacitance_mwh_per_c", self.thermal_capacitance_mwh_per_c)
        require_nonnegative(
            "envelope_conductance_mw_per_c",
            self.envelope_conductance_mw_per_c,
        )
        if not 0.0 <= self.compute_heat_fraction <= 1.0:
            raise ValueError("compute_heat_fraction must lie in [0, 1]")
        require_nonnegative("cooling_max_power_mw", self.cooling_max_power_mw)
        require_positive("cooling_nominal_cop", self.cooling_nominal_cop)
        require_positive("cooling_minimum_cop", self.cooling_minimum_cop)
        require_positive("cooling_maximum_cop", self.cooling_maximum_cop)
        if self.cooling_minimum_cop > self.cooling_maximum_cop:
            raise ValueError("cooling_minimum_cop cannot exceed cooling_maximum_cop")
        require_nonnegative("cop_temperature_slope_per_c", self.cop_temperature_slope_per_c)
        require_positive("cooling_ramp_rate_mw_per_hour", self.cooling_ramp_rate_mw_per_hour)
        require_nonnegative("cooling_response_time_hours", self.cooling_response_time_hours)
        if not 0.0 <= self.initial_cooling_power_mw <= self.cooling_max_power_mw:
            raise ValueError("initial_cooling_power_mw must be within cooling limits")


@dataclass(frozen=True)
class ThermalState:
    indoor_temperature_c: float
    cooling_power_mw: float
    cumulative_thermal_violation_c_hours: float = 0.0


@dataclass(frozen=True)
class ThermalStepResult:
    state: ThermalState
    requested_cooling_power_mw: float
    actual_cooling_power_mw: float
    cooling_cop: float
    compute_heat_mw: float
    cooling_heat_removed_mw: float
    passive_heat_removed_mw: float
    net_thermal_power_mw: float
    stored_thermal_energy_change_mwh: float
    violations: tuple[ConstraintViolation, ...]


class ThermalSystem:
    def __init__(self, config: ThermalConfig) -> None:
        self.config = config

    def initial_state(self) -> ThermalState:
        return ThermalState(
            indoor_temperature_c=self.config.initial_temperature_c,
            cooling_power_mw=self.config.initial_cooling_power_mw,
        )

    def cooling_cop(self, ambient_temperature_c: float) -> float:
        require_finite("ambient_temperature_c", ambient_temperature_c)
        unconstrained = (
            self.config.cooling_nominal_cop
            + (self.config.cop_reference_ambient_temperature_c - ambient_temperature_c)
            * self.config.cop_temperature_slope_per_c
        )
        return clip(
            unconstrained,
            self.config.cooling_minimum_cop,
            self.config.cooling_maximum_cop,
        )

    def constrain_power(
        self,
        state: ThermalState,
        requested_power_mw: float,
        duration_hours: float,
    ) -> tuple[float, tuple[ConstraintViolation, ...]]:
        require_finite("requested_power_mw", requested_power_mw)
        require_positive("duration_hours", duration_hours)
        violations: list[ConstraintViolation] = []
        target = clip(requested_power_mw, 0.0, self.config.cooling_max_power_mw)
        if abs(target - requested_power_mw) > NUMERICAL_EPSILON:
            violations.append(
                ConstraintViolation(
                    component="cooling",
                    code="power_limit",
                    message="Cooling setpoint was clipped to plant limits.",
                    requested_value=requested_power_mw,
                    actual_value=target,
                    unit="MW",
                )
            )

        if self.config.cooling_response_time_hours <= NUMERICAL_EPSILON:
            response_target = target
        else:
            response_fraction = 1.0 - math.exp(
                -duration_hours / self.config.cooling_response_time_hours
            )
            response_target = state.cooling_power_mw + response_fraction * (
                target - state.cooling_power_mw
            )
            if abs(response_target - target) > NUMERICAL_EPSILON:
                violations.append(
                    ConstraintViolation(
                        component="cooling",
                        code="response_lag",
                        message="Cooling response lag prevented immediate setpoint tracking.",
                        severity="info",
                        requested_value=target,
                        actual_value=response_target,
                        unit="MW",
                    )
                )

        ramp_delta = self.config.cooling_ramp_rate_mw_per_hour * duration_hours
        actual = clip(
            response_target,
            max(0.0, state.cooling_power_mw - ramp_delta),
            min(self.config.cooling_max_power_mw, state.cooling_power_mw + ramp_delta),
        )
        if abs(actual - response_target) > NUMERICAL_EPSILON:
            violations.append(
                ConstraintViolation(
                    component="cooling",
                    code="ramp_limit",
                    message="Cooling response was constrained by the plant ramp rate.",
                    requested_value=response_target,
                    actual_value=actual,
                    unit="MW",
                )
            )
        return actual, tuple(violations)

    def step(
        self,
        state: ThermalState,
        requested_cooling_power_mw: float,
        compute_power_mw: float,
        ambient_temperature_c: float,
        duration_hours: float,
        *,
        maximum_delivered_cooling_power_mw: float | None = None,
    ) -> ThermalStepResult:
        require_nonnegative("compute_power_mw", compute_power_mw)
        require_finite("ambient_temperature_c", ambient_temperature_c)
        require_positive("duration_hours", duration_hours)
        actual, initial_violations = self.constrain_power(
            state, requested_cooling_power_mw, duration_hours
        )
        violations = list(initial_violations)
        if maximum_delivered_cooling_power_mw is not None:
            require_nonnegative(
                "maximum_delivered_cooling_power_mw",
                maximum_delivered_cooling_power_mw,
            )
            capped = min(actual, maximum_delivered_cooling_power_mw)
            if capped < actual - NUMERICAL_EPSILON:
                violations.append(
                    ConstraintViolation(
                        component="cooling",
                        code="critical_load_shed",
                        message=(
                            "Cooling power was forcibly shed because available supply "
                            "was insufficient."
                        ),
                        severity="critical",
                        requested_value=actual,
                        actual_value=capped,
                        unit="MW",
                    )
                )
            actual = capped

        cop = self.cooling_cop(ambient_temperature_c)
        compute_heat = compute_power_mw * self.config.compute_heat_fraction
        cooling_removed = actual * cop
        passive_removed = self.config.envelope_conductance_mw_per_c * (
            state.indoor_temperature_c - ambient_temperature_c
        )
        net_thermal_power = compute_heat - cooling_removed - passive_removed
        stored_energy_change = net_thermal_power * duration_hours
        next_temperature = state.indoor_temperature_c + (
            stored_energy_change / self.config.thermal_capacitance_mwh_per_c
        )

        violation_degrees = max(0.0, next_temperature - self.config.maximum_temperature_c) + max(
            0.0, self.config.minimum_temperature_c - next_temperature
        )
        if next_temperature > self.config.maximum_temperature_c + NUMERICAL_EPSILON:
            violations.append(
                ConstraintViolation(
                    component="thermal",
                    code="high_temperature",
                    message="Data-hall temperature exceeded its configured maximum.",
                    severity="critical",
                    requested_value=self.config.maximum_temperature_c,
                    actual_value=next_temperature,
                    unit="degC",
                )
            )
        elif next_temperature < self.config.minimum_temperature_c - NUMERICAL_EPSILON:
            violations.append(
                ConstraintViolation(
                    component="thermal",
                    code="low_temperature",
                    message="Data-hall temperature fell below its configured minimum.",
                    severity="warning",
                    requested_value=self.config.minimum_temperature_c,
                    actual_value=next_temperature,
                    unit="degC",
                )
            )

        next_state = ThermalState(
            indoor_temperature_c=next_temperature,
            cooling_power_mw=actual,
            cumulative_thermal_violation_c_hours=(
                state.cumulative_thermal_violation_c_hours + violation_degrees * duration_hours
            ),
        )
        return ThermalStepResult(
            state=next_state,
            requested_cooling_power_mw=requested_cooling_power_mw,
            actual_cooling_power_mw=actual,
            cooling_cop=cop,
            compute_heat_mw=compute_heat,
            cooling_heat_removed_mw=cooling_removed,
            passive_heat_removed_mw=passive_removed,
            net_thermal_power_mw=net_thermal_power,
            stored_thermal_energy_change_mwh=stored_energy_change,
            violations=tuple(violations),
        )
