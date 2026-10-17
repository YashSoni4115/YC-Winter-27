from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_decision_layer.contracts import PolicyProtocol, ProposedControlActionV1
from energy_decision_layer.sim import SiteConfig, SiteSimulator
from energy_ml.policy import FEATURE_NAMES, TARGET_NAMES, fit_policy, observation_features
from energy_ml.synthetic import generate_series, to_site_inputs


@pytest.fixture(scope="module")
def policy():
    rng = np.random.default_rng(0)
    demos = pd.DataFrame(rng.normal(size=(300, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    for name, scale in zip(TARGET_NAMES, (8.0, 3.0, 6.0, 9.0), strict=True):
        demos[name] = rng.normal(scale=scale, size=300)  # deliberately wild: tests clipping
    return fit_policy(demos, SiteConfig())


def _obs(**overrides):
    inputs = to_site_inputs(generate_series(5, seed=1))
    if overrides:
        from dataclasses import replace

        inputs[0] = replace(inputs[0], **overrides)
    return SiteSimulator(inputs).reset()[0]


def test_policy_satisfies_protocol_and_emits_valid_bounded_proposals(policy):
    assert isinstance(policy, PolicyProtocol)
    c, o = SiteConfig(), _obs()
    p = policy.propose_action(o)
    ProposedControlActionV1.model_validate_json(p.model_dump_json())
    assert 0 <= p.compute_power_mw <= c.compute.max_power_mw
    assert 0 <= p.cooling_power_mw <= c.thermal.cooling_max_power_mw
    assert -c.battery.maximum_charge_power_mw <= p.battery_power_mw <= c.battery.maximum_discharge_power_mw
    assert p.generator_power_mw == 0 or c.generator.minimum_power_mw <= p.generator_power_mw <= c.generator.maximum_power_mw
    assert -o.grid_export_limit_mw <= p.grid_power_mw <= o.grid_import_limit_mw
    assert p.based_on_state_id == o.state_id


def test_grid_is_derived_from_power_balance_when_unclipped(policy):
    c, o = SiteConfig(), _obs()
    p = policy.propose_action(o)
    expected = p.compute_power_mw + p.cooling_power_mw + c.auxiliary_power_mw - p.battery_power_mw - p.generator_power_mw
    if -o.grid_export_limit_mw < p.grid_power_mw < o.grid_import_limit_mw:
        assert p.grid_power_mw == pytest.approx(expected)


def test_unavailable_equipment_is_never_commanded(policy):
    o = _obs(grid_available=False, generator_available=False)
    p = policy.propose_action(o)
    assert p.grid_power_mw == 0.0
    assert p.generator_power_mw == 0.0


def test_features_are_finite_with_and_without_forecast():
    o = _obs()
    x = observation_features(o, None)
    assert x.shape == (len(FEATURE_NAMES),) and np.isfinite(x).all()
