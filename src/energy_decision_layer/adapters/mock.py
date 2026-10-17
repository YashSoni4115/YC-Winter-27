"""Atomic simulated actuation through a second independent shield check."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from energy_decision_layer.contracts import ExecutedControlActionV1, SiteObservationV1
from energy_decision_layer.safety import SafetyShield
from energy_decision_layer.sim import SiteSimulator


class AdapterError(RuntimeError):
    """A controller refused the batch before any simulated equipment moved."""


class ControllerProtocol(Protocol):
    def prepare(self, action: ExecutedControlActionV1) -> None: ...


class MockController:
    """Equipment-specific preparation; the shared simulator commits the whole bus."""

    def __init__(self, component: str) -> None:
        if component not in {"compute", "cooling", "battery", "generator", "grid"}:
            raise ValueError("unknown equipment component")
        self.component = component
        self.available = True

    def prepare(self, action: ExecutedControlActionV1) -> None:
        if not self.available:
            raise AdapterError(f"{self.component} controller communication unavailable")
        ExecutedControlActionV1.model_validate_json(action.model_dump_json())


class MockSiteAdapter:
    """Only executes the setpoints independently reproduced by the shared shield."""

    def __init__(self, simulator: SiteSimulator, shield: SafetyShield | None = None) -> None:
        self.simulator = simulator
        self.shield = shield or SafetyShield(simulator.config)
        self.controllers: dict[str, ControllerProtocol] = {
            name: MockController(name)
            for name in ("compute", "cooling", "battery", "generator", "grid")
        }

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[SiteObservationV1, dict[str, Any]]:
        return self.simulator.reset(seed=seed, options=options)

    def step(
        self, action: ExecutedControlActionV1
    ) -> tuple[SiteObservationV1, float, bool, bool, dict[str, Any]]:
        checked = self.shield.evaluate(action.source_action, self.simulator.observation())
        permitted = checked.executed_action
        fields = (
            "compute_power_mw",
            "cooling_power_mw",
            "battery_power_mw",
            "generator_power_mw",
            "grid_power_mw",
            "renewable_curtailment_mw",
        )
        if permitted is None or any(
            abs(getattr(action, name) - getattr(permitted, name)) > 1e-8 for name in fields
        ):
            raise AdapterError("action differs from independently shield-authorized setpoints")
        for controller in self.controllers.values():
            controller.prepare(action)
        return self.simulator.step(action)
