"""Small integration adapters; forecasting and learned controllers remain ML-owned."""

from __future__ import annotations

from collections.abc import Callable

from energy_decision_layer.contracts import (
    ForecastBundleV1,
    ProposedControlActionV1,
    SiteObservationV1,
)
from energy_decision_layer.optimizer.planner import proposal


class FixedPolicy:
    """Fixed operating schedule for controlled offline comparisons."""

    def __init__(
        self, setpoints: tuple[float, float, float, float, float] = (6.0, 1.5, 0.0, 0.0, 7.7)
    ) -> None:
        self.setpoints = setpoints

    def propose_action(
        self, observation: SiteObservationV1, forecast: ForecastBundleV1 | None = None
    ) -> ProposedControlActionV1:
        return proposal(observation, self.setpoints, "fixed-schedule")


class CallablePolicyAdapter:
    """Wrap a teammate policy callable without implementing or training their model."""

    def __init__(
        self,
        policy: Callable[[SiteObservationV1, ForecastBundleV1 | None], ProposedControlActionV1],
    ) -> None:
        self.policy = policy

    def propose_action(
        self, observation: SiteObservationV1, forecast: ForecastBundleV1 | None = None
    ) -> ProposedControlActionV1:
        result = self.policy(observation, forecast)
        return ProposedControlActionV1.model_validate_json(result.model_dump_json())
