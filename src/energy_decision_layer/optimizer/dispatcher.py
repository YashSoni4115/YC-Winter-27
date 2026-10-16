"""Fail-closed orchestration; every fallback goes through the injected shield."""

from __future__ import annotations

from datetime import timedelta
from time import perf_counter
from typing import Any

from energy_decision_layer.contracts import (
    DecisionLoggerProtocol,
    ForecastBundleV1,
    ForecastProviderProtocol,
    PolicyProtocol,
    SafetyReasonCode,
    SafetyResponseV1,
    SafetyShieldProtocol,
    SiteObservationV1,
    SiteSimulatorProtocol,
)


class DispatchStopped(RuntimeError):
    """No action was applied; caller must hand control to existing local controllers."""


class DispatchOutcomeLoggingFailed(RuntimeError):
    """The action WAS applied; retain its result so audits can retry without re-actuation."""

    def __init__(self, result: tuple[SiteObservationV1, float, bool, bool, dict[str, Any]]) -> None:
        super().__init__("action applied, but outcome logging failed; do not replay the action")
        self.result = result


class Dispatcher:
    def __init__(
        self,
        policy: PolicyProtocol,
        shield: SafetyShieldProtocol,
        logger: DecisionLoggerProtocol,
        fallback: PolicyProtocol,
        forecaster: ForecastProviderProtocol | None = None,
        horizon_steps: int = 12,
    ) -> None:
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        self.last_response: SafetyResponseV1 | None = None
        self.policy, self.shield, self.logger = policy, shield, logger
        self.fallback, self.forecaster, self.horizon_steps = fallback, forecaster, horizon_steps

    def dispatch(
        self, observation: SiteObservationV1, simulator: SiteSimulatorProtocol
    ) -> tuple[SiteObservationV1, float, bool, bool, dict[str, Any]]:
        self.last_response = None
        started = perf_counter()
        fallback_reason = None
        try:
            forecast = (
                self.forecaster.forecast(
                    observation,
                    horizon_steps=self.horizon_steps,
                    cadence_seconds=observation.timestep_seconds,
                )
                if self.forecaster is not None
                else None
            )
            if forecast is not None:
                forecast = ForecastBundleV1.model_validate_json(forecast.model_dump_json())
                if (
                    forecast.site_id != observation.site_id
                    or forecast.issued_at_utc > observation.timestamp_utc
                    or forecast.valid_from_utc != observation.timestamp_utc
                    or forecast.cadence_seconds != observation.timestep_seconds
                ):
                    raise ValueError("forecast does not match current site, issuance or cadence")
            action = self.policy.propose_action(observation, forecast)
            action = type(action).model_validate_json(action.model_dump_json())
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            try:
                candidate = self.fallback.propose_action(observation, None)
                payload = candidate.model_dump()
                payload.update(
                    is_fallback=True,
                    fallback_reason=SafetyReasonCode.OTHER,
                    rationale=fallback_reason[:512],
                )
                action = type(candidate).model_validate(payload)
            except Exception as fallback_exc:
                raise DispatchStopped("primary and fallback policy failed") from fallback_exc
        latency = perf_counter() - started
        end = observation.timestamp_utc + timedelta(seconds=observation.timestep_seconds)
        if (
            action.site_id != observation.site_id
            or action.based_on_state_id != observation.state_id
            or action.created_at_utc > observation.timestamp_utc
            or not action.effective_at_utc
            <= observation.timestamp_utc
            < end
            <= action.expires_at_utc
        ):
            raise DispatchStopped("invalid or stale proposal")
        try:
            raw = self.shield.evaluate(action, observation)
            # Revalidate even if an integration used unsafe Pydantic model_construct/copy.
            response = SafetyResponseV1.model_validate_json(raw.model_dump_json())
            if (
                response.action_id != action.action_id
                or response.site_id != observation.site_id
                or response.decided_at_utc != observation.timestamp_utc
            ):
                raise ValueError("shield response does not match current request")
            if (
                response.executed_action is not None
                and response.executed_action.source_action != action
            ):
                raise ValueError("shield substituted the source proposal")
            self.last_response = response
            self.logger.record_decision(action, response)
        except Exception as exc:
            raise DispatchStopped("shield or audit exchange failed; no action applied") from exc
        if response.executed_action is None:
            raise DispatchStopped("shield rejected proposal; no action applied")
        obs, reward, terminated, truncated, details = simulator.step(response.executed_action)
        info = dict(details)
        info.update(
            policy_latency_seconds=latency,
            shield_intervention=response.status.value != "accepted",
            fallback_reason=fallback_reason,
            shield_name=response.shield_name,
        )
        try:
            self.logger.record_outcome(action, obs, info)
        except Exception as exc:
            raise DispatchOutcomeLoggingFailed((obs, reward, terminated, truncated, info)) from exc
        return obs, reward, terminated, truncated, info
