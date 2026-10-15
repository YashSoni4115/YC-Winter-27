"""Proposed and shield-authorized control action contracts."""

from __future__ import annotations

from math import isclose
from uuid import UUID

from pydantic import model_validator

from .common import (
    ConstraintMetadataV1,
    ContractModel,
    FiniteFloat,
    Identifier,
    NonEmptyString,
    NonNegativeFloat,
    SafetyDecisionStatus,
    SafetyReasonCode,
    UtcDatetime,
)

_SETPOINT_FIELDS = (
    "compute_power_mw",
    "cooling_power_mw",
    "battery_power_mw",
    "generator_power_mw",
    "grid_power_mw",
    "renewable_curtailment_mw",
)


class ProposedControlActionV1(ContractModel):
    """A policy request, not permission to actuate equipment.

    Battery power is positive for discharge and negative for charge. Grid power
    is positive for import and negative for export. Loads and generation are
    non-negative. A proposal may be infeasible; it must pass through a safety
    implementation before it can be applied.
    """

    action_id: UUID
    site_id: Identifier
    based_on_state_id: Identifier
    created_at_utc: UtcDatetime
    effective_at_utc: UtcDatetime
    expires_at_utc: UtcDatetime

    compute_power_mw: NonNegativeFloat
    cooling_power_mw: NonNegativeFloat
    battery_power_mw: FiniteFloat
    generator_power_mw: NonNegativeFloat
    grid_power_mw: FiniteFloat
    renewable_curtailment_mw: NonNegativeFloat = 0.0

    policy_name: NonEmptyString
    policy_version: NonEmptyString
    planner_run_id: UUID | None = None
    is_fallback: bool = False
    fallback_reason: SafetyReasonCode | None = None
    constraint_metadata: tuple[ConstraintMetadataV1, ...] = ()
    rationale: NonEmptyString | None = None

    @model_validator(mode="after")
    def validity_window_and_fallback_are_consistent(self) -> ProposedControlActionV1:
        if self.effective_at_utc < self.created_at_utc:
            raise ValueError("effective_at_utc cannot precede created_at_utc")
        if self.expires_at_utc <= self.effective_at_utc:
            raise ValueError("expires_at_utc must be after effective_at_utc")
        if self.is_fallback != (self.fallback_reason is not None):
            raise ValueError("fallback_reason must be present exactly when is_fallback is true")
        return self


class ExecutedControlActionV1(ContractModel):
    """Exact setpoints authorized for application after a safety decision.

    ``source_action`` retains the complete request so modifications are
    auditable.  The actual setpoint fields are the only values a simulator or
    controller may apply.  Physical actuator feedback, if available, belongs
    in subsequent telemetry rather than being inferred from this command.
    """

    execution_id: UUID
    source_action: ProposedControlActionV1
    safety_response_id: UUID
    authorization_status: SafetyDecisionStatus
    authorized_at_utc: UtcDatetime
    effective_at_utc: UtcDatetime
    expires_at_utc: UtcDatetime

    compute_power_mw: NonNegativeFloat
    cooling_power_mw: NonNegativeFloat
    battery_power_mw: FiniteFloat
    generator_power_mw: NonNegativeFloat
    grid_power_mw: FiniteFloat
    renewable_curtailment_mw: NonNegativeFloat = 0.0

    modified_fields: tuple[str, ...] = ()

    @property
    def source_action_id(self) -> UUID:
        return self.source_action.action_id

    @property
    def site_id(self) -> str:
        return self.source_action.site_id

    @property
    def based_on_state_id(self) -> str:
        return self.source_action.based_on_state_id

    @model_validator(mode="after")
    def authorization_is_consistent(self) -> ExecutedControlActionV1:
        if self.authorization_status is SafetyDecisionStatus.REJECTED:
            raise ValueError("a rejected proposal cannot produce an executed action")
        if self.authorized_at_utc < self.source_action.created_at_utc:
            raise ValueError("authorization cannot precede proposal creation")
        if self.authorized_at_utc > self.effective_at_utc:
            raise ValueError("authorized_at_utc cannot follow effective_at_utc")
        if self.effective_at_utc < self.source_action.effective_at_utc:
            raise ValueError("executed action cannot begin before the proposal")
        if self.expires_at_utc > self.source_action.expires_at_utc:
            raise ValueError("executed action cannot outlive the proposal")
        if self.expires_at_utc <= self.effective_at_utc:
            raise ValueError("expires_at_utc must be after effective_at_utc")

        if len(set(self.modified_fields)) != len(self.modified_fields):
            raise ValueError("modified_fields cannot contain duplicates")
        invalid_fields = set(self.modified_fields) - set(_SETPOINT_FIELDS)
        if invalid_fields:
            raise ValueError(
                "modified_fields contains non-setpoint fields: " + ", ".join(sorted(invalid_fields))
            )

        differing_fields = {
            name
            for name in _SETPOINT_FIELDS
            if not isclose(
                float(getattr(self, name)),
                float(getattr(self.source_action, name)),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        }
        if set(self.modified_fields) != differing_fields:
            raise ValueError(
                "modified_fields must exactly identify setpoints changed from source_action"
            )
        if self.authorization_status is SafetyDecisionStatus.ACCEPTED and differing_fields:
            raise ValueError("accepted actions cannot change proposed setpoints")
        if self.authorization_status is SafetyDecisionStatus.MODIFIED and not differing_fields:
            raise ValueError("modified actions must change at least one proposed setpoint")
        return self


ProposedControlAction = ProposedControlActionV1
ExecutedControlAction = ExecutedControlActionV1
