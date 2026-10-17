from datetime import timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from energy_decision_layer.contracts import SafetyDecisionStatus, SafetyReasonCode
from energy_decision_layer.contracts.common import DataQualityFlag
from energy_decision_layer.optimizer.planner import RulePolicy, proposal
from energy_decision_layer.safety import SafetyShield
from energy_decision_layer.sim import SiteConfig, SiteInputs, SiteSimulator
from energy_decision_layer.sim.battery import BatteryConfig
from energy_decision_layer.sim.thermal import ThermalConfig


def site(config=None, inputs=None):
    return SiteSimulator(inputs or [SiteInputs(nondeferrable_compute_units=30)] * 6, config)


def test_safe_action_and_coupled_diagnostics():
    sim = site()
    o = sim.observation()
    shield = SafetyShield(sim.config)
    p = RulePolicy(sim.config).propose_action(o)
    response = shield.evaluate(p, o)
    assert response.status is SafetyDecisionStatus.ACCEPTED
    assert response.executed_action.source_action == p
    assert all(c.satisfied for c in response.evaluated_constraints)
    sim.step(response.executed_action)
    bad = proposal(sim.observation(), (90, 90, 90, 90, 90), "unsafe")
    modified = shield.evaluate(bad, sim.observation())
    assert modified.status is SafetyDecisionStatus.MODIFIED
    assert SafetyReasonCode.GRID_IMPORT_LIMIT in modified.reason_codes
    assert SafetyReasonCode.BATTERY_POWER_LIMIT in modified.reason_codes
    assert any(c.satisfied is False for c in modified.evaluated_constraints)
    assert modified.executed_action.source_action == bad
    _, _, _, _, info = sim.step(modified.executed_action)
    assert not info["thermal_violation"]
    assert info["unserved_energy_mwh"] < 1e-7


@pytest.mark.parametrize(
    "field,value",
    [("battery_power_mw", float("nan")), ("grid_power_mw", float("inf")), ("cooling_power_mw", -1)],
)
def test_malformed_actions_fail_closed(field, value):
    sim = site()
    p = RulePolicy().propose_action(sim.observation()).model_copy(update={field: value})
    result = SafetyShield().evaluate(p, sim.observation())
    assert result.status is SafetyDecisionStatus.REJECTED
    assert result.reason_codes == (SafetyReasonCode.INVALID_ACTION,)
    assert result.executed_action is None


def test_missing_and_invalid_observation_fail_closed():
    sim = site()
    o = sim.observation()
    p = RulePolicy().propose_action(o)
    for bad in [
        o.model_copy(update={"battery_state_of_charge": 0.99}),
        o.model_copy(update={"quality_flags": {"facility_temperature_c": DataQualityFlag.MISSING}}),
        o.model_copy(update={"facility_temperature_c": 50.0}),
        o.model_copy(update={"battery_power_mw": 50.0}),
        o.model_copy(update={"cooling_power_mw": 50.0}),
        o.model_copy(update={"generator_power_mw": 50.0, "generator_is_on": True}),
        None,
    ]:
        result = SafetyShield().evaluate(p, bad)
        assert result.status is SafetyDecisionStatus.REJECTED
        assert result.reason_codes == (SafetyReasonCode.MISSING_TELEMETRY,)
    result = SafetyShield().evaluate(None, o)
    assert result.reason_codes == (SafetyReasonCode.INVALID_ACTION,)


def test_clock_validity_and_stale_state():
    o = site().observation()
    p = RulePolicy().propose_action(o)
    shield = SafetyShield()
    for changed in [
        p.model_copy(update={"based_on_state_id": "old"}),
        p.model_copy(update={"site_id": "different"}),
    ]:
        assert shield.evaluate(changed, o).reason_codes == (SafetyReasonCode.STALE_STATE,)
    assert (
        shield.evaluate(p, o, now=o.timestamp_utc + timedelta(hours=1)).status.value == "rejected"
    )
    assert (
        shield.evaluate(p, o, now=o.timestamp_utc.replace(tzinfo=None)).status.value == "rejected"
    )
    future = p.model_copy(update={"effective_at_utc": o.timestamp_utc + timedelta(seconds=1)})
    assert shield.evaluate(future, o).reason_codes == (SafetyReasonCode.ACTION_NOT_YET_VALID,)
    expired = p.model_copy(update={"expires_at_utc": o.timestamp_utc + timedelta(seconds=1)})
    assert shield.evaluate(expired, o).reason_codes == (SafetyReasonCode.ACTION_EXPIRED,)


def test_infeasible_grid_outage_rejects_instead_of_unsafe_execution():
    config = SiteConfig(battery=BatteryConfig(initial_soc_fraction=0.1))
    sim = site(config, [SiteInputs(grid_available=False, generator_available=False)])
    o = sim.observation()
    result = SafetyShield(config).evaluate(RulePolicy(config).propose_action(o), o)
    assert result.status is SafetyDecisionStatus.REJECTED
    assert result.executed_action is None


def test_soc_reserve_generator_availability_and_heat():
    config = SiteConfig(
        battery=BatteryConfig(initial_soc_fraction=0.1),
        thermal=ThermalConfig(initial_temperature_c=32),
    )
    sim = site(config, [SiteInputs(generator_available=False, ambient_temperature_c=50)])
    o = sim.observation()
    p = proposal(o, (10, 0, 5, 10, 20), "unsafe")
    response = SafetyShield(config).evaluate(p, o)
    assert response.executed_action is not None
    assert response.executed_action.generator_power_mw == 0
    assert response.executed_action.battery_power_mw <= 1e-8
    assert SafetyReasonCode.BATTERY_SOC_LIMIT in response.reason_codes
    assert SafetyReasonCode.TEMPERATURE_LIMIT in response.reason_codes
    assert not sim.step(response.executed_action)[4]["thermal_violation"]


@settings(max_examples=65, deadline=None)
@given(
    st.tuples(
        st.floats(0, 50), st.floats(0, 20), st.floats(-20, 20), st.floats(0, 30), st.floats(-20, 40)
    )
)
def test_projection_preserves_all_coupled_physical_bounds(values):
    sim = site()
    o = sim.observation()
    response = SafetyShield(sim.config).evaluate(proposal(o, values, "property"), o)
    if response.executed_action is None:
        assert response.status is SafetyDecisionStatus.REJECTED
        return
    action = response.executed_action
    next_o, _, _, _, info = sim.step(action)
    assert sim.config.battery.minimum_soc_fraction - 1e-8 <= next_o.battery_state_of_charge
    assert next_o.battery_state_of_charge <= sim.config.battery.maximum_soc_fraction + 1e-8
    assert sim.config.thermal.minimum_temperature_c - 1e-8 <= next_o.facility_temperature_c
    assert next_o.facility_temperature_c <= sim.config.thermal.maximum_temperature_c + 1e-8
    assert next_o.grid_power_mw <= o.grid_import_limit_mw + 1e-8
    assert info["unserved_energy_mwh"] < 1e-7
    assert abs(info["power_balance_residual_mw"]) < 1e-7


def test_internal_failure_is_rejected(monkeypatch):
    import energy_decision_layer.safety.shield as module

    monkeypatch.setattr(module, "linprog", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    o = site().observation()
    result = SafetyShield().evaluate(RulePolicy().propose_action(o), o)
    assert result.reason_codes == (SafetyReasonCode.SHIELD_INTERNAL_ERROR,)
