"""Dispatchable generator commitment, ramping, and fuel model."""

from __future__ import annotations

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
class GeneratorConfig:
    minimum_power_mw: float = 1.0
    maximum_power_mw: float = 8.0
    ramp_up_rate_mw_per_hour: float = 8.0
    ramp_down_rate_mw_per_hour: float = 8.0
    electrical_efficiency: float = 0.40
    fuel_cost_per_mwh_thermal: float = 30.0
    variable_cost_per_mwh_electric: float = 4.0
    startup_cost: float = 40.0
    shutdown_cost: float = 5.0
    minimum_up_steps: int = 1
    minimum_down_steps: int = 1
    initial_on: bool = False
    initial_power_mw: float = 0.0
    initial_elapsed_steps: int | None = None

    def __post_init__(self) -> None:
        require_nonnegative("minimum_power_mw", self.minimum_power_mw)
        require_positive("maximum_power_mw", self.maximum_power_mw)
        if self.minimum_power_mw > self.maximum_power_mw:
            raise ValueError("minimum_power_mw cannot exceed maximum_power_mw")
        require_positive("ramp_up_rate_mw_per_hour", self.ramp_up_rate_mw_per_hour)
        require_positive("ramp_down_rate_mw_per_hour", self.ramp_down_rate_mw_per_hour)
        if not 0.0 < self.electrical_efficiency <= 1.0:
            raise ValueError("electrical_efficiency must lie in (0, 1]")
        for name in (
            "fuel_cost_per_mwh_thermal",
            "variable_cost_per_mwh_electric",
            "startup_cost",
            "shutdown_cost",
        ):
            require_nonnegative(name, getattr(self, name))
        if self.minimum_up_steps < 0 or self.minimum_down_steps < 0:
            raise ValueError("minimum up/down steps must be nonnegative")
        if self.initial_elapsed_steps is not None and self.initial_elapsed_steps < 0:
            raise ValueError("initial_elapsed_steps must be nonnegative")
        if self.initial_on:
            if not self.minimum_power_mw <= self.initial_power_mw <= self.maximum_power_mw:
                raise ValueError("an initially-on generator must start inside output limits")
        elif abs(self.initial_power_mw) > NUMERICAL_EPSILON:
            raise ValueError("an initially-off generator must have zero power")


@dataclass(frozen=True)
class GeneratorState:
    is_on: bool
    power_mw: float
    consecutive_on_steps: int
    consecutive_off_steps: int
    cumulative_fuel_energy_mwh_thermal: float = 0.0
    cumulative_generated_energy_mwh: float = 0.0
    startup_count: int = 0


@dataclass(frozen=True)
class GeneratorStepResult:
    state: GeneratorState
    requested_power_mw: float
    actual_power_mw: float
    fuel_energy_mwh_thermal: float
    generated_energy_mwh: float
    operating_cost: float
    startup_cost: float
    shutdown_cost: float
    started: bool
    stopped: bool
    violations: tuple[ConstraintViolation, ...]


class GeneratorSystem:
    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config

    def initial_state(self) -> GeneratorState:
        default_elapsed = (
            self.config.minimum_up_steps
            if self.config.initial_on
            else self.config.minimum_down_steps
        )
        elapsed = (
            default_elapsed
            if self.config.initial_elapsed_steps is None
            else self.config.initial_elapsed_steps
        )
        return GeneratorState(
            is_on=self.config.initial_on,
            power_mw=self.config.initial_power_mw,
            consecutive_on_steps=elapsed if self.config.initial_on else 0,
            consecutive_off_steps=0 if self.config.initial_on else elapsed,
        )

    def step(
        self,
        state: GeneratorState,
        requested_power_mw: float,
        duration_hours: float,
        *,
        available: bool,
    ) -> GeneratorStepResult:
        require_finite("requested_power_mw", requested_power_mw)
        require_positive("duration_hours", duration_hours)
        requested_nonnegative = max(0.0, requested_power_mw)
        violations: list[ConstraintViolation] = []
        if requested_power_mw < -NUMERICAL_EPSILON:
            violations.append(
                ConstraintViolation(
                    component="generator",
                    code="negative_power",
                    message="Negative generator power was replaced with zero.",
                    requested_value=requested_power_mw,
                    actual_value=0.0,
                    unit="MW",
                )
            )

        started = False
        stopped = False
        next_on = state.is_on
        actual = 0.0

        if not available:
            stopped = state.is_on
            next_on = False
            actual = 0.0
            if state.is_on or requested_nonnegative > NUMERICAL_EPSILON:
                violations.append(
                    ConstraintViolation(
                        component="generator",
                        code="unavailable",
                        message="Generator output is zero because the unit is unavailable.",
                        severity="critical" if state.is_on else "warning",
                        requested_value=requested_nonnegative,
                        actual_value=0.0,
                        unit="MW",
                    )
                )
        elif state.is_on:
            if requested_nonnegative <= NUMERICAL_EPSILON:
                if state.consecutive_on_steps < self.config.minimum_up_steps:
                    next_on = True
                    actual = max(
                        self.config.minimum_power_mw,
                        state.power_mw - self.config.ramp_down_rate_mw_per_hour * duration_hours,
                    )
                    violations.append(
                        ConstraintViolation(
                            component="generator",
                            code="minimum_up_time",
                            message="Shutdown was blocked by minimum up-time.",
                            requested_value=0.0,
                            actual_value=actual,
                            unit="MW",
                        )
                    )
                else:
                    next_on = False
                    stopped = True
                    actual = 0.0
            else:
                target = clip(
                    requested_nonnegative,
                    self.config.minimum_power_mw,
                    self.config.maximum_power_mw,
                )
                if abs(target - requested_nonnegative) > NUMERICAL_EPSILON:
                    violations.append(
                        ConstraintViolation(
                            component="generator",
                            code="output_limit",
                            message="Generator request was clipped to committed output limits.",
                            requested_value=requested_nonnegative,
                            actual_value=target,
                            unit="MW",
                        )
                    )
                lower = max(
                    self.config.minimum_power_mw,
                    state.power_mw - self.config.ramp_down_rate_mw_per_hour * duration_hours,
                )
                upper = min(
                    self.config.maximum_power_mw,
                    state.power_mw + self.config.ramp_up_rate_mw_per_hour * duration_hours,
                )
                actual = clip(target, lower, upper)
                if abs(actual - target) > NUMERICAL_EPSILON:
                    violations.append(
                        ConstraintViolation(
                            component="generator",
                            code="ramp_limit",
                            message="Generator output was constrained by its ramp rate.",
                            requested_value=target,
                            actual_value=actual,
                            unit="MW",
                        )
                    )
        elif requested_nonnegative > NUMERICAL_EPSILON:
            if state.consecutive_off_steps < self.config.minimum_down_steps:
                violations.append(
                    ConstraintViolation(
                        component="generator",
                        code="minimum_down_time",
                        message="Startup was blocked by minimum down-time.",
                        requested_value=requested_nonnegative,
                        actual_value=0.0,
                        unit="MW",
                    )
                )
                next_on = False
                actual = 0.0
            else:
                next_on = True
                started = True
                target = clip(
                    requested_nonnegative,
                    self.config.minimum_power_mw,
                    self.config.maximum_power_mw,
                )
                # Startup enters at minimum stable output, then may move by one
                # interval's normal up-ramp.  The commitment jump itself is not
                # an in-service ramp and is therefore explicit here.
                actual = min(
                    target,
                    self.config.minimum_power_mw
                    + self.config.ramp_up_rate_mw_per_hour * duration_hours,
                )
                if abs(actual - requested_nonnegative) > NUMERICAL_EPSILON:
                    violations.append(
                        ConstraintViolation(
                            component="generator",
                            code="startup_output_limit",
                            message="Startup output was constrained by stable output and ramps.",
                            requested_value=requested_nonnegative,
                            actual_value=actual,
                            unit="MW",
                        )
                    )

        if next_on:
            next_on_steps = state.consecutive_on_steps + 1 if state.is_on else 1
            next_off_steps = 0
        else:
            next_on_steps = 0
            next_off_steps = state.consecutive_off_steps + 1 if not state.is_on else 1

        generated_energy = actual * duration_hours
        fuel_energy = generated_energy / self.config.electrical_efficiency
        operating_cost = (
            fuel_energy * self.config.fuel_cost_per_mwh_thermal
            + generated_energy * self.config.variable_cost_per_mwh_electric
        )
        applied_startup_cost = self.config.startup_cost if started else 0.0
        applied_shutdown_cost = self.config.shutdown_cost if stopped else 0.0
        next_state = GeneratorState(
            is_on=next_on,
            power_mw=actual,
            consecutive_on_steps=next_on_steps,
            consecutive_off_steps=next_off_steps,
            cumulative_fuel_energy_mwh_thermal=(
                state.cumulative_fuel_energy_mwh_thermal + fuel_energy
            ),
            cumulative_generated_energy_mwh=(
                state.cumulative_generated_energy_mwh + generated_energy
            ),
            startup_count=state.startup_count + int(started),
        )
        return GeneratorStepResult(
            state=next_state,
            requested_power_mw=requested_power_mw,
            actual_power_mw=actual,
            fuel_energy_mwh_thermal=fuel_energy,
            generated_energy_mwh=generated_energy,
            operating_cost=operating_cost,
            startup_cost=applied_startup_cost,
            shutdown_cost=applied_shutdown_cost,
            started=started,
            stopped=stopped,
            violations=tuple(violations),
        )
