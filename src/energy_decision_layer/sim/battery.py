"""Battery energy, efficiency, ramping, and degradation accounting."""

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
class BatteryConfig:
    energy_capacity_mwh: float = 20.0
    minimum_soc_fraction: float = 0.1
    maximum_soc_fraction: float = 0.95
    initial_soc_fraction: float = 0.5
    maximum_charge_power_mw: float = 5.0
    maximum_discharge_power_mw: float = 5.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    ramp_rate_mw_per_hour: float = 40.0
    degradation_cost_per_mwh_throughput: float = 3.0
    initial_power_mw: float = 0.0

    def __post_init__(self) -> None:
        require_positive("energy_capacity_mwh", self.energy_capacity_mwh)
        for name in (
            "minimum_soc_fraction",
            "maximum_soc_fraction",
            "initial_soc_fraction",
            "charge_efficiency",
            "discharge_efficiency",
        ):
            require_finite(name, getattr(self, name))
        if not 0.0 <= self.minimum_soc_fraction < self.maximum_soc_fraction <= 1.0:
            raise ValueError("battery SOC limits must satisfy 0 <= min < max <= 1")
        if not (
            self.minimum_soc_fraction <= self.initial_soc_fraction <= self.maximum_soc_fraction
        ):
            raise ValueError("initial_soc_fraction must lie inside configured limits")
        if not 0.0 < self.charge_efficiency <= 1.0:
            raise ValueError("charge_efficiency must lie in (0, 1]")
        if not 0.0 < self.discharge_efficiency <= 1.0:
            raise ValueError("discharge_efficiency must lie in (0, 1]")
        require_nonnegative("maximum_charge_power_mw", self.maximum_charge_power_mw)
        require_nonnegative("maximum_discharge_power_mw", self.maximum_discharge_power_mw)
        require_positive("ramp_rate_mw_per_hour", self.ramp_rate_mw_per_hour)
        require_nonnegative(
            "degradation_cost_per_mwh_throughput",
            self.degradation_cost_per_mwh_throughput,
        )
        if not (
            -self.maximum_charge_power_mw
            <= self.initial_power_mw
            <= self.maximum_discharge_power_mw
        ):
            raise ValueError("initial_power_mw must lie inside battery power limits")

    @property
    def minimum_energy_mwh(self) -> float:
        return self.minimum_soc_fraction * self.energy_capacity_mwh

    @property
    def maximum_energy_mwh(self) -> float:
        return self.maximum_soc_fraction * self.energy_capacity_mwh

    @property
    def usable_energy_mwh(self) -> float:
        return self.maximum_energy_mwh - self.minimum_energy_mwh


@dataclass(frozen=True)
class BatteryState:
    energy_mwh: float
    power_mw: float = 0.0
    cumulative_cell_throughput_mwh: float = 0.0
    equivalent_full_cycles: float = 0.0

    def soc_fraction(self, config: BatteryConfig) -> float:
        return self.energy_mwh / config.energy_capacity_mwh


@dataclass(frozen=True)
class BatteryStepResult:
    state: BatteryState
    requested_power_mw: float
    actual_power_mw: float
    stored_energy_change_mwh: float
    cell_throughput_mwh: float
    conversion_loss_mwh: float
    degradation_cost: float
    violations: tuple[ConstraintViolation, ...]


class BatterySystem:
    """A one-bus battery model.

    Positive bus power is discharge; negative bus power is charge.  Charge
    and discharge cannot occur simultaneously because the action is scalar.
    """

    def __init__(self, config: BatteryConfig) -> None:
        self.config = config

    def initial_state(self) -> BatteryState:
        return BatteryState(
            energy_mwh=(self.config.initial_soc_fraction * self.config.energy_capacity_mwh),
            power_mw=self.config.initial_power_mw,
        )

    def constrain_power(
        self,
        state: BatteryState,
        requested_power_mw: float,
        duration_hours: float,
    ) -> tuple[float, tuple[ConstraintViolation, ...]]:
        require_finite("requested_power_mw", requested_power_mw)
        require_positive("duration_hours", duration_hours)
        require_finite("state.energy_mwh", state.energy_mwh)
        require_finite("state.power_mw", state.power_mw)
        if not self.config.minimum_energy_mwh <= state.energy_mwh <= self.config.maximum_energy_mwh:
            raise ValueError("battery state energy outside configured limits")
        violations: list[ConstraintViolation] = []

        rated = clip(
            requested_power_mw,
            -self.config.maximum_charge_power_mw,
            self.config.maximum_discharge_power_mw,
        )
        if abs(rated - requested_power_mw) > NUMERICAL_EPSILON:
            violations.append(
                ConstraintViolation(
                    component="battery",
                    code="power_limit",
                    message="Battery setpoint was clipped to its power rating.",
                    requested_value=requested_power_mw,
                    actual_value=rated,
                    unit="MW",
                )
            )

        delta_limit = self.config.ramp_rate_mw_per_hour * duration_hours
        ramped = clip(
            rated,
            state.power_mw - delta_limit,
            state.power_mw + delta_limit,
        )
        if abs(ramped - rated) > NUMERICAL_EPSILON:
            violations.append(
                ConstraintViolation(
                    component="battery",
                    code="ramp_limit",
                    message="Battery setpoint was constrained by its ramp rate.",
                    requested_value=rated,
                    actual_value=ramped,
                    unit="MW",
                )
            )

        available_discharge_mwh = max(0.0, state.energy_mwh - self.config.minimum_energy_mwh)
        maximum_discharge_from_energy_mw = (
            available_discharge_mwh * self.config.discharge_efficiency / duration_hours
        )
        available_charge_room_mwh = max(0.0, self.config.maximum_energy_mwh - state.energy_mwh)
        maximum_charge_from_energy_mw = (
            available_charge_room_mwh / self.config.charge_efficiency / duration_hours
        )
        energy_limited = clip(
            ramped,
            -maximum_charge_from_energy_mw,
            maximum_discharge_from_energy_mw,
        )
        if abs(energy_limited - ramped) > NUMERICAL_EPSILON:
            limit_name = "minimum SOC" if ramped > 0.0 else "maximum SOC"
            violations.append(
                ConstraintViolation(
                    component="battery",
                    code="soc_limit",
                    message=f"Battery power was reduced to preserve {limit_name}.",
                    requested_value=ramped,
                    actual_value=energy_limited,
                    unit="MW",
                )
            )
        return energy_limited, tuple(violations)

    def step(
        self,
        state: BatteryState,
        requested_power_mw: float,
        duration_hours: float,
        *,
        minimum_bus_power_mw: float | None = None,
    ) -> BatteryStepResult:
        """Advance battery state.

        ``minimum_bus_power_mw`` is used only by the integrated site when
        insufficient supply forces charging toward zero.  It represents an
        emergency bus-balance override, is recorded explicitly, and never
        permits extra discharge or an SOC violation.
        """

        actual, initial_violations = self.constrain_power(state, requested_power_mw, duration_hours)
        violations = list(initial_violations)
        if minimum_bus_power_mw is not None:
            require_finite("minimum_bus_power_mw", minimum_bus_power_mw)
            if minimum_bus_power_mw > 0.0 + NUMERICAL_EPSILON:
                raise ValueError("minimum_bus_power_mw cannot force discharge")
            overridden = max(actual, minimum_bus_power_mw)
            if overridden > actual + NUMERICAL_EPSILON:
                violations.append(
                    ConstraintViolation(
                        component="battery",
                        code="charge_curtailed_for_balance",
                        message=(
                            "Battery charging was curtailed because site supply was insufficient."
                        ),
                        severity="critical",
                        requested_value=actual,
                        actual_value=overridden,
                        unit="MW",
                    )
                )
                actual = overridden

        if actual >= 0.0:
            cell_withdrawal = actual * duration_hours / self.config.discharge_efficiency
            stored_change = -cell_withdrawal
            conversion_loss = cell_withdrawal - actual * duration_hours
            cell_throughput = cell_withdrawal
        else:
            bus_charge_energy = -actual * duration_hours
            stored_charge = bus_charge_energy * self.config.charge_efficiency
            stored_change = stored_charge
            conversion_loss = bus_charge_energy - stored_charge
            cell_throughput = stored_charge

        next_energy = clip(
            state.energy_mwh + stored_change,
            self.config.minimum_energy_mwh,
            self.config.maximum_energy_mwh,
        )
        # Any clip larger than floating-point noise indicates an internal bug,
        # because constrain_power already enforces the same energy bounds.
        expected_energy = state.energy_mwh + stored_change
        if abs(next_energy - expected_energy) > 1e-7:
            raise RuntimeError("battery energy transition escaped configured SOC bounds")

        cumulative_throughput = state.cumulative_cell_throughput_mwh + cell_throughput
        equivalent_cycles = (
            cumulative_throughput / (2.0 * self.config.usable_energy_mwh)
            if self.config.usable_energy_mwh > NUMERICAL_EPSILON
            else 0.0
        )
        degradation_cost = cell_throughput * self.config.degradation_cost_per_mwh_throughput
        next_state = BatteryState(
            energy_mwh=next_energy,
            power_mw=actual,
            cumulative_cell_throughput_mwh=cumulative_throughput,
            equivalent_full_cycles=equivalent_cycles,
        )
        return BatteryStepResult(
            state=next_state,
            requested_power_mw=requested_power_mw,
            actual_power_mw=actual,
            stored_energy_change_mwh=stored_change,
            cell_throughput_mwh=cell_throughput,
            conversion_loss_mwh=conversion_loss,
            degradation_cost=degradation_cost,
            violations=tuple(violations),
        )
