"""Versioned data-boundary models.

The gold interval model deliberately keeps physical observations separate from
provenance. Joined datasets can have a different source per column, so the
``lineage_id`` refers to a sidecar lineage manifest rather than claiming one
source for the whole row.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from enum import Enum, IntFlag
from typing import Any, Literal
from urllib.parse import parse_qsl, urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DATA_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be normalized to UTC")
    return value.astimezone(timezone.utc)


class StrictModel(BaseModel):
    """Boundary base model that rejects misspelled or future fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


class RedistributionStatus(str, Enum):
    """Whether downloaded bytes or derivatives may be included in releases."""

    ALLOWED = "allowed"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


class SignalSemantics(str, Enum):
    """Aggregation semantics needed to resample a signal correctly."""

    INTERVAL_TOTAL = "interval_total"
    INTERVAL_MEAN = "interval_mean"
    INSTANTANEOUS = "instantaneous"
    STATE = "state"
    EVENT = "event"


class QualityFlag(IntFlag):
    """Stable bit assignments for row/field quality annotations."""

    NONE = 0
    SOURCE_MISSING = 1 << 0
    IMPUTED_FORWARD = 1 << 1
    INTERPOLATED = 1 << 2
    DUPLICATE_RESOLVED = 1 << 3
    OUTLIER_STATISTICAL = 1 << 4
    OUT_OF_PHYSICAL_RANGE = 1 << 5
    SOURCE_QC_FAILED = 1 << 6
    TIMEZONE_AMBIGUOUS = 1 << 7
    SYNTHETIC = 1 << 8
    DERIVED = 1 << 9


class SourceTerms(StrictModel):
    """Legal/access metadata captured before a source is ingested."""

    terms_url: str = Field(min_length=1)
    retrieved_at_utc: datetime
    terms_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    redistribution_status: RedistributionStatus
    required_attribution: str = ""
    notes: str = ""

    _utc = field_validator("retrieved_at_utc")(_require_utc)


class RawArtifactManifest(StrictModel):
    """Immutable metadata for exact bytes returned by a source."""

    schema_version: Literal["1.0.0"] = DATA_SCHEMA_VERSION
    source_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    product_id: str = Field(min_length=1)
    canonical_url: str = Field(min_length=1)
    retrieved_at_utc: datetime
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    content_type: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    request_parameters: dict[str, str] = Field(default_factory=dict)
    etag: str | None = None
    last_modified: str | None = None
    terms: SourceTerms

    _utc = field_validator("retrieved_at_utc")(_require_utc)

    @field_validator("source_id")
    @classmethod
    def reject_relative_components(cls, value: str) -> str:
        if value in {".", ".."}:
            raise ValueError("source_id cannot be a relative path component")
        return value

    @field_validator("canonical_url")
    @classmethod
    def reject_embedded_credentials(cls, value: str) -> str:
        parts = urlparse(value)
        if (
            parts.username
            or parts.password
            or any(
                any(
                    word in key.lower()
                    for word in ("key", "token", "secret", "password", "authorization")
                )
                for key, _ in parse_qsl(parts.query)
            )
        ):
            raise ValueError("canonical_url must not contain credentials")
        return value

    @field_validator("request_parameters")
    @classmethod
    def reject_secret_like_parameters(cls, value: dict[str, str]) -> dict[str, str]:
        forbidden_fragments = ("key", "token", "secret", "password", "authorization")
        for name in value:
            if any(fragment in name.lower() for fragment in forbidden_fragments):
                raise ValueError(f"request parameter {name!r} may contain a secret")
        return dict(sorted(value.items()))


class SimulationIntervalV1(StrictModel):
    """Canonical simulation-ready interval in UTC.

    Intervals are left-closed and right-open: ``[start, end)``. A nullable field
    means the source did not provide a trustworthy value; zero is never used as
    an implicit missing-value marker.
    """

    schema_version: Literal["1.0.0"] = DATA_SCHEMA_VERSION
    site_id: str = Field(min_length=1)
    interval_start_utc: datetime
    interval_end_utc: datetime
    original_timestamp: str | None = None
    source_timezone: str | None = None
    workload_demand_compute_units: float | None = Field(default=None, ge=0)
    active_compute_capacity_compute_units_per_hour: float | None = Field(default=None, ge=0)
    grid_import_price_per_mwh: float | None = None
    price_currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    ambient_temperature_c: float | None = None
    grid_stress_status: Literal["normal", "watch", "emergency", "curtailment", "unknown"] = (
        "unknown"
    )
    grid_available: bool | None = None
    grid_carbon_intensity_kgco2e_per_mwh: float | None = Field(default=None, ge=0)
    cooling_thermal_load_mwth: float | None = Field(default=None, ge=0)
    quality_mask: int = Field(default=0, ge=0, le=(1 << 64) - 1)
    lineage_id: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _start_utc = field_validator("interval_start_utc")(_require_utc)
    _end_utc = field_validator("interval_end_utc")(_require_utc)

    @field_validator(
        "workload_demand_compute_units",
        "active_compute_capacity_compute_units_per_hour",
        "grid_import_price_per_mwh",
        "ambient_temperature_c",
        "grid_carbon_intensity_kgco2e_per_mwh",
        "cooling_thermal_load_mwth",
    )
    @classmethod
    def finite_optional_values(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("numeric values must be finite")
        return value

    @model_validator(mode="after")
    def validate_interval_and_currency(self) -> SimulationIntervalV1:
        if self.interval_end_utc <= self.interval_start_utc:
            raise ValueError("interval_end_utc must be after interval_start_utc")
        if (self.grid_import_price_per_mwh is None) != (self.price_currency is None):
            raise ValueError("grid price and price_currency must be provided together")
        return self

    @property
    def duration_hours(self) -> float:
        return (self.interval_end_utc - self.interval_start_utc).total_seconds() / 3600.0
