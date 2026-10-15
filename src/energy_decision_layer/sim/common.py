"""Shared simulator value objects and validation helpers.

Power is expressed in MW, energy in MWh, elapsed time in hours, temperature
in degrees Celsius, and workload in abstract useful-compute units.  At the
site bus, positive battery power means discharge and positive grid power
means import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

NUMERICAL_EPSILON = 1e-9


def require_finite(name: str, value: float) -> None:
    """Raise ``ValueError`` when a public numerical input is not finite."""

    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")


def require_nonnegative(name: str, value: float) -> None:
    require_finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")


def require_positive(name: str, value: float) -> None:
    require_finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


@dataclass(frozen=True)
class ConstraintViolation:
    """A visible adjustment or violated operating constraint.

    ``severity`` is one of ``info``, ``warning``, or ``critical``.  Local
    limit enforcement is normally a warning; actual loss of critical load or
    a thermal excursion is critical.
    """

    component: str
    code: str
    message: str
    severity: str = "warning"
    requested_value: float | None = None
    actual_value: float | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if self.severity not in {"info", "warning", "critical"}:
            raise ValueError(f"invalid violation severity: {self.severity!r}")
        if self.requested_value is not None:
            require_finite("requested_value", self.requested_value)
        if self.actual_value is not None:
            require_finite("actual_value", self.actual_value)


@dataclass(frozen=True)
class SiteAction:
    """Requested or realized site setpoints.

    Sign conventions:

    * ``battery_power_mw > 0`` discharges into the site bus; a negative value
      charges the battery.
    * ``grid_power_mw > 0`` imports from the grid; a negative value exports.
    * all other powers are nonnegative electrical consumption or production.
    """

    compute_power_mw: float
    cooling_power_mw: float
    battery_power_mw: float
    generator_power_mw: float
    grid_power_mw: float

    def __post_init__(self) -> None:
        for field_name, value in self.as_dict().items():
            require_finite(field_name, value)

    def as_dict(self) -> dict[str, float]:
        return {
            "compute_power_mw": float(self.compute_power_mw),
            "cooling_power_mw": float(self.cooling_power_mw),
            "battery_power_mw": float(self.battery_power_mw),
            "generator_power_mw": float(self.generator_power_mw),
            "grid_power_mw": float(self.grid_power_mw),
        }


@dataclass(frozen=True)
class SiteInputs:
    """Exogenous values known for one simulation interval."""

    active_compute_capacity_compute_units_per_hour: float | None = None
    ambient_temperature_c: float = 25.0
    nondeferrable_compute_units: float = 0.0
    deferrable_compute_units: float = 0.0
    deferrable_deadline_steps: int = 4
    grid_available: bool = True
    grid_stress: bool = False
    grid_import_limit_mw: float | None = None
    grid_export_limit_mw: float | None = None
    grid_import_price_per_mwh: float = 75.0
    grid_export_price_per_mwh: float = 25.0
    generator_available: bool = True

    def __post_init__(self) -> None:
        require_finite("ambient_temperature_c", self.ambient_temperature_c)
        if self.ambient_temperature_c < -273.15:
            raise ValueError("ambient temperature is below absolute zero")
        if self.active_compute_capacity_compute_units_per_hour is not None:
            require_nonnegative(
                "active_compute_capacity_compute_units_per_hour",
                self.active_compute_capacity_compute_units_per_hour,
            )
        for name in ("grid_available", "grid_stress", "generator_available"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        require_nonnegative("nondeferrable_compute_units", self.nondeferrable_compute_units)
        require_nonnegative("deferrable_compute_units", self.deferrable_compute_units)
        if self.deferrable_deadline_steps < 1:
            raise ValueError("deferrable_deadline_steps must be at least one")
        if self.grid_import_limit_mw is not None:
            require_nonnegative("grid_import_limit_mw", self.grid_import_limit_mw)
        if self.grid_export_limit_mw is not None:
            require_nonnegative("grid_export_limit_mw", self.grid_export_limit_mw)
        require_finite("grid_import_price_per_mwh", self.grid_import_price_per_mwh)
        require_finite("grid_export_price_per_mwh", self.grid_export_price_per_mwh)


@dataclass(frozen=True)
class PowerBalance:
    """Electrical bus accounting for a completed interval.

    ``supply_mw`` and ``served_load_mw`` contain only energy that physically
    crossed the bus.  Unserved requested load is reported separately and is
    never counted as fictitious supply.
    """

    supply_mw: float
    served_load_mw: float
    spilled_power_mw: float
    unserved_power_mw: float
    battery_charge_curtailed_mw: float
    error_mw: float


@dataclass(frozen=True)
class CostBreakdown:
    grid_cost: float = 0.0
    generator_cost: float = 0.0
    generator_startup_cost: float = 0.0
    generator_shutdown_cost: float = 0.0
    battery_degradation_cost: float = 0.0
    deadline_penalty: float = 0.0
    unserved_energy_penalty: float = 0.0
    thermal_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.grid_cost
            + self.generator_cost
            + self.generator_startup_cost
            + self.generator_shutdown_cost
            + self.battery_degradation_cost
            + self.deadline_penalty
            + self.unserved_energy_penalty
            + self.thermal_penalty
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "grid_cost": self.grid_cost,
            "generator_cost": self.generator_cost,
            "generator_startup_cost": self.generator_startup_cost,
            "generator_shutdown_cost": self.generator_shutdown_cost,
            "battery_degradation_cost": self.battery_degradation_cost,
            "deadline_penalty": self.deadline_penalty,
            "unserved_energy_penalty": self.unserved_energy_penalty,
            "thermal_penalty": self.thermal_penalty,
            "total": self.total,
        }
