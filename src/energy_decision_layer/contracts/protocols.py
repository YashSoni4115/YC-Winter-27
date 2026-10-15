"""Structural interfaces owned jointly across project workstreams.

These protocols define production integration shapes but provide no production
forecasting, safety, policy, simulator, or logging implementation. Test and
development doubles must be explicitly named as such in their own modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .actions import ExecutedControlActionV1, ProposedControlActionV1
from .forecasts import ForecastBundleV1
from .safety import SafetyResponseV1
from .telemetry import SiteObservationV1


@runtime_checkable
class ForecastProviderProtocol(Protocol):
    """Interface implemented by the ML teammate's forecast provider."""

    def forecast(
        self,
        observation: SiteObservationV1,
        *,
        horizon_steps: int,
        cadence_seconds: float,
    ) -> ForecastBundleV1:
        """Return a forecast issued no later than the observation timestamp."""
        ...


@runtime_checkable
class PolicyProtocol(Protocol):
    """Common interface for rule, optimization, learned, and RL policies."""

    def propose_action(
        self,
        observation: SiteObservationV1,
        forecast: ForecastBundleV1 | None = None,
    ) -> ProposedControlActionV1:
        """Return an unauthorised proposal based only on available information."""
        ...


@runtime_checkable
class SafetyShieldProtocol(Protocol):
    """Interface owned by the production safety/backend workstream."""

    def evaluate(
        self,
        proposal: ProposedControlActionV1,
        observation: SiteObservationV1,
    ) -> SafetyResponseV1:
        """Accept, modify, or reject a proposed action."""
        ...


@runtime_checkable
class SiteSimulatorProtocol(Protocol):
    """Typed simulator boundary; a Gym wrapper may adapt these return values."""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[SiteObservationV1, Mapping[str, Any]]: ...

    def step(
        self,
        action: ExecutedControlActionV1,
    ) -> tuple[SiteObservationV1, float, bool, bool, Mapping[str, Any]]: ...


@runtime_checkable
class DecisionLoggerProtocol(Protocol):
    """Backend-owned audit sink interface; persistence is not implemented here."""

    def record_decision(
        self,
        proposal: ProposedControlActionV1,
        response: SafetyResponseV1,
    ) -> None: ...

    def record_outcome(
        self,
        proposal: ProposedControlActionV1,
        observation: SiteObservationV1,
        diagnostics: Mapping[str, Any],
    ) -> None:
        """Record applied action/physics feedback after execution, linked by proposal UUID."""
        ...
