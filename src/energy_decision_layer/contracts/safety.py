"""Safety-shield decision contract with enforced status semantics."""

from __future__ import annotations

from uuid import UUID

from pydantic import model_validator

from .actions import ExecutedControlActionV1
from .common import (
    ConstraintMetadataV1,
    ContractModel,
    Identifier,
    NonEmptyString,
    SafetyDecisionStatus,
    SafetyReasonCode,
    UtcDatetime,
)


class SafetyResponseV1(ContractModel):
    """Result returned by a production shield or an explicit development double.

    Accepted and modified responses carry the sole action permitted for
    application. Rejected responses never carry an executable action. This
    model describes the integration boundary; it does not implement safety.
    """

    response_id: UUID
    action_id: UUID
    site_id: Identifier
    decided_at_utc: UtcDatetime
    shield_name: NonEmptyString
    shield_version: NonEmptyString
    status: SafetyDecisionStatus
    reason_codes: tuple[SafetyReasonCode, ...] = ()
    reason_message: NonEmptyString | None = None
    evaluated_constraints: tuple[ConstraintMetadataV1, ...] = ()
    executed_action: ExecutedControlActionV1 | None = None

    @model_validator(mode="after")
    def status_payload_is_consistent(self) -> SafetyResponseV1:
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("reason_codes cannot contain duplicates")

        if self.status is SafetyDecisionStatus.REJECTED:
            if self.executed_action is not None:
                raise ValueError("rejected responses cannot contain an executed_action")
            if not self.reason_codes:
                raise ValueError("rejected responses require at least one reason code")
            return self

        if self.executed_action is None:
            raise ValueError("accepted and modified responses require executed_action")
        if self.executed_action.authorized_at_utc != self.decided_at_utc:
            raise ValueError("authorization time must equal decision time")
        if self.executed_action.authorization_status is not self.status:
            raise ValueError("executed_action.authorization_status must match response status")
        if self.executed_action.source_action_id != self.action_id:
            raise ValueError("executed_action source action does not match action_id")
        if self.executed_action.site_id != self.site_id:
            raise ValueError("executed_action site does not match site_id")
        if self.executed_action.safety_response_id != self.response_id:
            raise ValueError("executed_action safety_response_id does not match response_id")

        if self.status is SafetyDecisionStatus.ACCEPTED:
            invalid_reasons = set(self.reason_codes) - {SafetyReasonCode.WITHIN_CONSTRAINTS}
            if invalid_reasons:
                raise ValueError("accepted responses may only use the within_constraints reason")
        elif not self.reason_codes:
            raise ValueError("modified responses require at least one reason code")
        return self


SafetyResponse = SafetyResponseV1
