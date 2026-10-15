"""Offline teammate integration smoke; replace factories with your own implementations."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from energy_decision_layer.contracts import (
    DecisionLoggerProtocol,
    ForecastProviderProtocol,
    PolicyProtocol,
    SafetyShieldProtocol,
)
from energy_decision_layer.evaluation.runner import evaluate
from energy_decision_layer.sim import SiteInputs


def load_factory(specification: str) -> Callable[[], Any]:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("factory must use importable.module:factory format")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"{specification} is not callable")
    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--policy", default="energy_decision_layer.optimizer:MilpPolicy")
    doubles = "energy_decision_layer.testing.doubles:"
    parser.add_argument("--forecaster", default=doubles + "PersistenceForecastDouble")
    parser.add_argument("--shield", default=doubles + "DevelopmentAcceptAllShield")
    parser.add_argument("--logger", default=doubles + "InMemoryDecisionLogger")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    policy = load_factory(args.policy)()
    forecaster = load_factory(args.forecaster)()
    shield = load_factory(args.shield)()
    logger = load_factory(args.logger)()
    for component, contract in (
        (policy, PolicyProtocol),
        (forecaster, ForecastProviderProtocol),
        (shield, SafetyShieldProtocol),
        (logger, DecisionLoggerProtocol),
    ):
        if not isinstance(component, contract):
            raise TypeError(f"{type(component).__name__} does not implement {contract.__name__}")
    result = evaluate(
        "teammate-offline-smoke",
        [SiteInputs(nondeferrable_compute_units=40.0)] * args.steps,
        lambda: policy,
        forecaster=forecaster,
        shield=shield,
        decision_logger=logger,
    )
    print(json.dumps(asdict(result), indent=2, allow_nan=False))
    if result.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
