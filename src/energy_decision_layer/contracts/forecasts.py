"""Leakage-resistant forecast exchange contracts."""

from __future__ import annotations

import itertools
from enum import Enum
from math import isclose
from uuid import UUID

from pydantic import Field, model_validator

from .common import (
    ContractModel,
    FiniteFloat,
    Fraction,
    GridStressStatus,
    Identifier,
    NonEmptyString,
    NonNegativeFloat,
    PositiveFloat,
    ProvenanceV1,
    UtcDatetime,
)


class ForecastSignal(str, Enum):
    WORKLOAD_DEMAND = "workload_demand_compute_units_per_h"
    ACTIVE_COMPUTE_CAPACITY = "active_compute_capacity_compute_units_per_h"
    ELECTRICITY_PRICE = "electricity_price_usd_per_mwh"
    AMBIENT_TEMPERATURE = "ambient_temperature_c"
    GRID_IMPORT_LIMIT = "grid_import_limit_mw"
    GRID_EXPORT_LIMIT = "grid_export_limit_mw"
    GRID_CARBON_INTENSITY = "grid_carbon_intensity_kg_co2e_per_mwh"
    RENEWABLE_AVAILABILITY = "renewable_available_power_mw"


class PredictionIntervalV1(ContractModel):
    signal: ForecastSignal
    lower_value: FiniteFloat
    upper_value: FiniteFloat
    coverage_probability: Fraction

    @model_validator(mode="after")
    def interval_is_ordered(self) -> PredictionIntervalV1:
        if self.lower_value > self.upper_value:
            raise ValueError("lower_value cannot exceed upper_value")
        return self


class ForecastPointV1(ContractModel):
    target_timestamp_utc: UtcDatetime
    workload_demand_compute_units_per_h: NonNegativeFloat
    deferrable_fraction: Fraction = 0.0
    deferrable_deadline_steps: int = Field(default=1, ge=1)
    active_compute_capacity_compute_units_per_h: NonNegativeFloat | None = None
    electricity_price_usd_per_mwh: FiniteFloat
    ambient_temperature_c: FiniteFloat = Field(ge=-273.15)
    grid_available: bool = True
    grid_import_limit_mw: NonNegativeFloat | None = None
    grid_export_limit_mw: NonNegativeFloat | None = None
    grid_carbon_intensity_kg_co2e_per_mwh: NonNegativeFloat | None = None
    grid_stress_status: GridStressStatus = GridStressStatus.NORMAL
    renewable_available_power_mw: NonNegativeFloat = 0.0
    prediction_intervals: tuple[PredictionIntervalV1, ...] = ()

    @model_validator(mode="after")
    def prediction_intervals_are_unique_and_contain_points(self) -> ForecastPointV1:
        seen: set[ForecastSignal] = set()
        for interval in self.prediction_intervals:
            if interval.signal in seen:
                raise ValueError(f"duplicate prediction interval for {interval.signal.value}")
            seen.add(interval.signal)
            point_value = getattr(self, interval.signal.value)
            if point_value is None:
                raise ValueError(f"prediction interval provided for absent {interval.signal.value}")
            if not interval.lower_value <= point_value <= interval.upper_value:
                raise ValueError(
                    f"point forecast for {interval.signal.value} lies outside its interval"
                )
        return self


class ForecastBundleV1(ContractModel):
    """Forecast values that were available at one explicit issuance time."""

    forecast_id: UUID
    site_id: Identifier
    issued_at_utc: UtcDatetime
    valid_from_utc: UtcDatetime
    valid_until_utc: UtcDatetime
    cadence_seconds: PositiveFloat
    model_name: NonEmptyString
    model_version: NonEmptyString
    points: tuple[ForecastPointV1, ...] = Field(min_length=1)
    provenance: ProvenanceV1
    is_perfect_hindsight: bool = False

    @model_validator(mode="after")
    def horizon_is_ordered_and_leakage_safe(self) -> ForecastBundleV1:
        if self.valid_from_utc < self.issued_at_utc:
            raise ValueError("valid_from_utc cannot precede issued_at_utc")
        if self.valid_until_utc < self.valid_from_utc:
            raise ValueError("valid_until_utc cannot precede valid_from_utc")

        timestamps = [point.target_timestamp_utc for point in self.points]
        if timestamps[0] != self.valid_from_utc:
            raise ValueError("valid_from_utc must equal the first target timestamp")
        if timestamps[-1] != self.valid_until_utc:
            raise ValueError("valid_until_utc must equal the last target timestamp")
        if any(timestamp < self.issued_at_utc for timestamp in timestamps):
            raise ValueError("forecast target timestamps cannot precede issuance")
        if any(right <= left for left, right in itertools.pairwise(timestamps)):
            raise ValueError("forecast target timestamps must be strictly increasing")

        for left, right in itertools.pairwise(timestamps):
            actual_seconds = (right - left).total_seconds()
            if not isclose(
                actual_seconds,
                self.cadence_seconds,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("forecast target timestamps must follow cadence_seconds exactly")
        return self


ForecastBundle = ForecastBundleV1
