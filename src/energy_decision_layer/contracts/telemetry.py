"""Telemetry and complete simulator observation contracts."""

from __future__ import annotations

from math import isclose
from typing import ClassVar

from pydantic import Field, model_validator

from .common import (
    ContractModel,
    DataQualityFlag,
    DataQualityIssueV1,
    FiniteFloat,
    Fraction,
    GridStressStatus,
    Identifier,
    NonNegativeFloat,
    PositiveFloat,
    ProvenanceV1,
    UtcDatetime,
)


class TelemetryV1(ContractModel):
    """A possibly incomplete raw telemetry sample.

    Missing sensor readings are represented as ``None``.  If a field is listed
    in ``quality_flags`` as ``missing``, its value must be ``None``; conversely,
    a present value cannot be labelled missing.  Complete control state is
    represented by :class:`SiteObservationV1`, not this ingestion boundary.
    """

    site_id: Identifier
    state_id: Identifier
    timestamp_utc: UtcDatetime
    sequence_number: int = Field(ge=0)

    workload_demand_compute_units_per_h: NonNegativeFloat | None = None
    active_compute_capacity_compute_units_per_h: NonNegativeFloat | None = None
    critical_backlog_compute_units: NonNegativeFloat | None = None
    deferrable_backlog_compute_units: NonNegativeFloat | None = None
    compute_power_mw: NonNegativeFloat | None = None
    cooling_power_mw: NonNegativeFloat | None = None
    auxiliary_power_mw: NonNegativeFloat | None = None
    facility_temperature_c: FiniteFloat | None = Field(default=None, ge=-273.15)
    ambient_temperature_c: FiniteFloat | None = Field(default=None, ge=-273.15)
    battery_energy_mwh: NonNegativeFloat | None = None
    battery_state_of_charge: Fraction | None = None
    battery_power_mw: FiniteFloat | None = None
    generator_power_mw: NonNegativeFloat | None = None
    generator_is_on: bool | None = None
    generator_available: bool | None = None
    grid_power_mw: FiniteFloat | None = None
    grid_available: bool | None = None
    grid_import_limit_mw: NonNegativeFloat | None = None
    grid_export_limit_mw: NonNegativeFloat | None = None
    electricity_price_usd_per_mwh: FiniteFloat | None = None
    grid_carbon_intensity_kg_co2e_per_mwh: NonNegativeFloat | None = None
    grid_stress_status: GridStressStatus = GridStressStatus.UNKNOWN
    renewable_available_power_mw: NonNegativeFloat | None = None
    renewable_used_power_mw: NonNegativeFloat | None = None

    provenance: ProvenanceV1
    quality_flags: dict[str, DataQualityFlag] = Field(default_factory=dict)
    quality_issues: tuple[DataQualityIssueV1, ...] = ()

    _NON_MEASUREMENT_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "site_id",
            "state_id",
            "timestamp_utc",
            "sequence_number",
            "provenance",
            "quality_flags",
            "quality_issues",
        }
    )

    @model_validator(mode="after")
    def quality_flags_reference_measurements(self) -> TelemetryV1:
        measurement_fields = set(type(self).model_fields) - self._NON_MEASUREMENT_FIELDS
        for field_name, flag in self.quality_flags.items():
            if field_name not in measurement_fields:
                raise ValueError(f"quality_flags contains unknown measurement field {field_name!r}")
            value = getattr(self, field_name)
            if flag is DataQualityFlag.MISSING and value is not None:
                raise ValueError(f"{field_name} is present but its quality flag is 'missing'")
            if flag is not DataQualityFlag.MISSING and value is None:
                raise ValueError(f"{field_name} is absent but its quality flag is {flag.value!r}")
        for issue in self.quality_issues:
            if issue.field_name not in measurement_fields:
                raise ValueError(
                    f"quality issue references unknown measurement field {issue.field_name!r}"
                )
        return self


class QueuedWorkloadV1(ContractModel):
    remaining_compute_units: NonNegativeFloat
    steps_to_deadline: int = Field(ge=1)


class SiteObservationV1(ContractModel):
    """Complete state supplied to planners, policies, and the Gym adapter."""

    site_id: Identifier
    state_id: Identifier
    timestamp_utc: UtcDatetime
    step_index: int = Field(ge=0)
    timestep_seconds: PositiveFloat

    workload_demand_compute_units_per_h: NonNegativeFloat
    active_compute_capacity_compute_units_per_h: NonNegativeFloat
    critical_backlog_compute_units: NonNegativeFloat
    deferrable_backlog_compute_units: NonNegativeFloat
    queued_workloads: tuple[QueuedWorkloadV1, ...] = ()
    incoming_deferrable_fraction: Fraction = 0.0
    incoming_deferrable_deadline_steps: int = Field(default=1, ge=1)
    completed_compute_units: NonNegativeFloat
    deadline_violated_compute_units: NonNegativeFloat = 0.0

    compute_power_mw: NonNegativeFloat
    cooling_power_mw: NonNegativeFloat
    auxiliary_power_mw: NonNegativeFloat
    facility_temperature_c: FiniteFloat = Field(ge=-273.15)
    ambient_temperature_c: FiniteFloat = Field(ge=-273.15)

    battery_energy_mwh: NonNegativeFloat
    battery_capacity_mwh: NonNegativeFloat
    battery_state_of_charge: Fraction
    battery_power_mw: FiniteFloat

    generator_power_mw: NonNegativeFloat
    generator_is_on: bool
    generator_available: bool
    generator_consecutive_on_steps: int = Field(default=0, ge=0)
    generator_consecutive_off_steps: int = Field(default=0, ge=0)

    grid_power_mw: FiniteFloat
    grid_available: bool
    grid_import_limit_mw: NonNegativeFloat
    grid_export_limit_mw: NonNegativeFloat
    electricity_price_usd_per_mwh: FiniteFloat
    grid_carbon_intensity_kg_co2e_per_mwh: NonNegativeFloat | None = None
    grid_stress_status: GridStressStatus = GridStressStatus.NORMAL

    renewable_available_power_mw: NonNegativeFloat = 0.0
    renewable_used_power_mw: NonNegativeFloat = 0.0
    curtailed_power_mw: NonNegativeFloat = 0.0
    unserved_power_mw: NonNegativeFloat = 0.0
    power_balance_residual_mw: FiniteFloat = 0.0

    quality_flags: dict[str, DataQualityFlag] = Field(default_factory=dict)

    @model_validator(mode="after")
    def state_is_internally_consistent(self) -> SiteObservationV1:
        tolerance = 1e-7
        if self.battery_energy_mwh > self.battery_capacity_mwh + tolerance:
            raise ValueError("battery_energy_mwh cannot exceed battery_capacity_mwh")
        expected_soc = (
            self.battery_energy_mwh / self.battery_capacity_mwh
            if self.battery_capacity_mwh > 0.0
            else 0.0
        )
        if not isclose(
            self.battery_state_of_charge,
            expected_soc,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError(
                "battery_state_of_charge must equal battery_energy_mwh / battery_capacity_mwh"
            )
        if not self.grid_available and abs(self.grid_power_mw) > tolerance:
            raise ValueError("grid_power_mw must be zero while the grid is unavailable")
        if not self.generator_available and (
            self.generator_is_on or self.generator_power_mw > tolerance
        ):
            raise ValueError("an unavailable generator must be off and produce zero power")
        if not self.generator_is_on and self.generator_power_mw > tolerance:
            raise ValueError("generator_power_mw must be zero while generator_is_on is false")
        if self.renewable_used_power_mw > self.renewable_available_power_mw + tolerance:
            raise ValueError("renewable_used_power_mw cannot exceed renewable_available_power_mw")
        expected_curtailment = self.renewable_available_power_mw - self.renewable_used_power_mw
        if not isclose(
            self.curtailed_power_mw,
            expected_curtailment,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError("curtailed_power_mw must equal available minus used renewable power")
        return self


# Ergonomic aliases point to the sole supported version; serialized payloads
# still carry and validate schema_version="1.0.0".
Telemetry = TelemetryV1
SiteObservation = SiteObservationV1
