"""Shared primitives for version 1 of the public integration contracts.

The contracts deliberately use unit-bearing field names instead of a separate
unit registry.  This keeps JSON payloads unambiguous and avoids implicit unit
conversion at service boundaries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
SchemaVersion = Literal["1.0.0"]


def _as_utc(value: datetime) -> datetime:
    """Reject naive datetimes and normalize aware datetimes to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_as_utc)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=512),
]
Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    ),
]


class ContractModel(BaseModel):
    """Base class for immutable, strict, forward-incompatible boundaries."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
        str_strip_whitespace=True,
    )

    schema_version: SchemaVersion = SCHEMA_VERSION


class DataQualityFlag(str, Enum):
    """Per-value quality state used by telemetry and data pipelines."""

    GOOD = "good"
    ESTIMATED = "estimated"
    IMPUTED = "imputed"
    MISSING = "missing"
    SUSPECT = "suspect"
    SYNTHETIC = "synthetic"


class DataQualityReason(str, Enum):
    """Machine-readable reasons behind a non-good quality flag."""

    SOURCE_MISSING = "source_missing"
    DUPLICATE = "duplicate"
    OUTLIER = "outlier"
    STALE = "stale"
    INVALID_UNIT = "invalid_unit"
    IMPOSSIBLE_VALUE = "impossible_value"
    TIMEZONE_AMBIGUOUS = "timezone_ambiguous"
    IMPUTED = "imputed"
    SOURCE_ERROR = "source_error"
    OTHER = "other"


class GridStressStatus(str, Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    EMERGENCY = "emergency"
    OUTAGE = "outage"
    UNKNOWN = "unknown"


class ConstraintKind(str, Enum):
    BATTERY_SOC = "battery_soc"
    BATTERY_POWER = "battery_power"
    GENERATOR_POWER = "generator_power"
    GENERATOR_RAMP = "generator_ramp"
    GRID_IMPORT = "grid_import"
    GRID_EXPORT = "grid_export"
    GRID_AVAILABILITY = "grid_availability"
    COMPUTE_CAPACITY = "compute_capacity"
    COOLING_CAPACITY = "cooling_capacity"
    TEMPERATURE = "temperature"
    POWER_BALANCE = "power_balance"
    WORKLOAD_DEADLINE = "workload_deadline"
    ACTION_VALIDITY = "action_validity"
    TELEMETRY_QUALITY = "telemetry_quality"
    FORECAST_FRESHNESS = "forecast_freshness"
    OTHER = "other"


class ConstraintSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class SafetyDecisionStatus(str, Enum):
    ACCEPTED = "accepted"
    MODIFIED = "modified"
    REJECTED = "rejected"


class SafetyReasonCode(str, Enum):
    WITHIN_CONSTRAINTS = "within_constraints"
    BATTERY_SOC_LIMIT = "battery_soc_limit"
    BATTERY_POWER_LIMIT = "battery_power_limit"
    GENERATOR_POWER_LIMIT = "generator_power_limit"
    GENERATOR_RAMP_LIMIT = "generator_ramp_limit"
    GRID_IMPORT_LIMIT = "grid_import_limit"
    GRID_EXPORT_LIMIT = "grid_export_limit"
    GRID_UNAVAILABLE = "grid_unavailable"
    COMPUTE_CAPACITY_LIMIT = "compute_capacity_limit"
    COOLING_CAPACITY_LIMIT = "cooling_capacity_limit"
    TEMPERATURE_LIMIT = "temperature_limit"
    POWER_BALANCE = "power_balance"
    STALE_STATE = "stale_state"
    ACTION_NOT_YET_VALID = "action_not_yet_valid"
    ACTION_EXPIRED = "action_expired"
    INVALID_ACTION = "invalid_action"
    MISSING_TELEMETRY = "missing_telemetry"
    COMMUNICATION_FAILURE = "communication_failure"
    SHIELD_INTERNAL_ERROR = "shield_internal_error"
    OTHER = "other"


class ProvenanceV1(ContractModel):
    """Origin metadata retained with external or synthetic observations."""

    source_name: NonEmptyString
    source_uri: NonEmptyString | None = None
    source_record_id: NonEmptyString | None = None
    source_timestamp_utc: UtcDatetime | None = None
    source_timezone: NonEmptyString | None = None
    ingested_at_utc: UtcDatetime
    license_identifier: NonEmptyString | None = None
    is_synthetic: bool = False

    @field_validator("source_timezone")
    @classmethod
    def source_timezone_must_be_iana(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("source_timezone must be an IANA timezone name") from exc
        return value


class DataQualityIssueV1(ContractModel):
    field_name: Identifier
    flag: DataQualityFlag
    reason: DataQualityReason
    message: NonEmptyString | None = None

    @model_validator(mode="after")
    def good_values_do_not_have_issue_reasons(self) -> DataQualityIssueV1:
        if self.flag is DataQualityFlag.GOOD:
            raise ValueError("quality issues cannot use the 'good' flag")
        return self


class ConstraintMetadataV1(ContractModel):
    """A constraint considered by a planner or checked by a shield."""

    constraint_id: Identifier
    kind: ConstraintKind
    description: NonEmptyString
    unit: NonEmptyString | None = None
    lower_bound: FiniteFloat | None = None
    upper_bound: FiniteFloat | None = None
    observed_value: FiniteFloat | None = None
    satisfied: bool | None = None
    severity: ConstraintSeverity = ConstraintSeverity.CRITICAL

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> ConstraintMetadataV1:
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("lower_bound cannot exceed upper_bound")
        for name in ("lower_bound", "upper_bound", "observed_value"):
            value = getattr(self, name)
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite")
        return self
