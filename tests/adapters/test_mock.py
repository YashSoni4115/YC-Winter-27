import pytest

from energy_decision_layer.adapters import AdapterError, MockController, MockSiteAdapter
from energy_decision_layer.optimizer import RulePolicy
from energy_decision_layer.safety import SafetyShield
from energy_decision_layer.sim import SiteInputs, SiteSimulator


def prepared():
    sim = SiteSimulator([SiteInputs(nondeferrable_compute_units=20)] * 3)
    adapter = MockSiteAdapter(sim)
    o, _ = adapter.reset(seed=2)
    action = SafetyShield(sim.config).evaluate(RulePolicy().propose_action(o), o).executed_action
    return sim, adapter, action


def test_mock_batch_executes_and_feedback_is_real():
    sim, adapter, action = prepared()
    observation, reward, _, _, info = adapter.step(action)
    assert observation.step_index == 1
    assert observation == sim.observation()
    assert info["completed_compute_units"] == pytest.approx(20)
    assert reward <= 0


def test_forged_authorized_action_cannot_bypass_shield():
    sim, adapter, action = prepared()
    forged = action.model_copy(update={"grid_power_mw": 500})
    with pytest.raises(AdapterError, match="shield-authorized"):
        adapter.step(forged)
    assert sim.index == 0


def test_controller_failure_is_atomic():
    sim, adapter, action = prepared()
    adapter.controllers["battery"].available = False
    with pytest.raises(AdapterError, match="communication"):
        adapter.step(action)
    assert sim.index == 0
    with pytest.raises(ValueError, match="unknown equipment"):
        MockController("hardware")
