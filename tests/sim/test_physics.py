from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from energy_decision_layer.sim import SiteAction, SiteInputs, SiteSimulator
from energy_decision_layer.sim.battery import BatteryConfig, BatterySystem
from energy_decision_layer.sim.compute import ComputeConfig, ComputeSystem
from energy_decision_layer.sim.env import SiteEnv
from energy_decision_layer.sim.generator import GeneratorConfig, GeneratorSystem
from energy_decision_layer.sim.thermal import ThermalConfig, ThermalSystem


@pytest.mark.parametrize("dt", [1 / 60, 1 / 12, 0.25, 1.0])
@pytest.mark.parametrize("power", [-100, -5, 0, 5, 100])
def test_battery_conserves_energy_and_bounds(dt, power):
    b = BatterySystem(BatteryConfig())
    s = b.initial_state()
    for _ in range(30):
        r = b.step(s, power, dt)
        assert (
            b.config.minimum_energy_mwh - 1e-9
            <= r.state.energy_mwh
            <= b.config.maximum_energy_mwh + 1e-9
        )
        assert r.state.energy_mwh - s.energy_mwh == pytest.approx(r.stored_energy_change_mwh)
        assert -r.actual_power_mw * dt == pytest.approx(
            r.stored_energy_change_mwh + r.conversion_loss_mwh
        )
        assert r.conversion_loss_mwh >= 0
        s = r.state


def test_workload_deadline_and_queue_accounting():
    model = ComputeSystem(ComputeConfig())
    s = model.initial_state()
    a = model.step(s, 0, 1 / 12, 10, 20, 2)
    assert a.deadline_missed_compute_units == 10
    assert a.state.deferrable_backlog_compute_units == 20
    b = model.step(a.state, 0, 1 / 12, 0, 0, 2)
    assert b.deadline_missed_compute_units == 20
    assert b.state.deferrable_backlog_compute_units == 0
    assert b.state.cumulative_deadline_missed_compute_units == 30


def test_thermal_energy_and_heat_direction():
    m = ThermalSystem(ThermalConfig(cooling_response_time_hours=0))
    s = m.initial_state()
    r = m.step(s, 0, 10, 30, 0.25)
    assert r.state.indoor_temperature_c > s.indoor_temperature_c
    assert (
        r.state.indoor_temperature_c - s.indoor_temperature_c
    ) * m.config.thermal_capacitance_mwh_per_c == pytest.approx(r.stored_thermal_energy_change_mwh)
    assert m.cooling_cop(45) < m.cooling_cop(25)
    r = m.step(s, 4, 0, 20, 0.25)
    assert r.state.indoor_temperature_c < s.indoor_temperature_c


def test_generator_commitment_failure_and_ramp():
    m = GeneratorSystem(GeneratorConfig(minimum_up_steps=3, minimum_down_steps=2))
    s = m.initial_state()
    r = m.step(s, 8, 1 / 12, available=True)
    assert r.started
    assert r.actual_power_mw == pytest.approx(1 + 8 / 12)
    assert r.fuel_energy_mwh_thermal * m.config.electrical_efficiency == pytest.approx(
        r.generated_energy_mwh
    )
    blocked = m.step(r.state, 0, 1 / 12, available=True)
    assert blocked.state.is_on
    failed = m.step(blocked.state, 8, 1 / 12, available=False)
    assert not failed.state.is_on
    assert failed.actual_power_mw == 0


@pytest.mark.parametrize("seed", range(8))
def test_randomized_site_bus_balance_and_battery_invariants(seed):
    rng = np.random.default_rng(seed)
    inputs = [
        SiteInputs(
            grid_available=i % 3 != 0,
            generator_available=i % 5 != 0,
            ambient_temperature_c=40,
            nondeferrable_compute_units=100,
        )
        for i in range(80)
    ]
    sim = SiteSimulator(inputs)
    while not sim.done:
        action = SiteAction(*rng.uniform([-2, -1, -20, -1, -5], [20, 8, 20, 20, 30]))
        _, reward, _, _, info = sim.advance(action)
        assert np.isfinite(reward)
        assert abs(info["power_balance_residual_mw"]) < 1e-8
        a = info["applied_action"]
        assert a["grid_power_mw"] + a["generator_power_mw"] + a["battery_power_mw"] >= -1e-8
        assert (
            sim.config.battery.minimum_energy_mwh - 1e-8
            <= sim.bs.energy_mwh
            <= sim.config.battery.maximum_energy_mwh + 1e-8
        )


def test_gym_api_and_determinism():
    env = SiteEnv(SiteSimulator([SiteInputs(nondeferrable_compute_units=40)] * 12))
    check_env(env, skip_render_check=True)
    a, _ = env.reset(seed=4)
    step1 = env.step(np.array([5, 1, 0, 0, 6], dtype=float))
    b, _ = env.reset(seed=4)
    step2 = env.step(np.array([5, 1, 0, 0, 6], dtype=float))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(step1[0], step2[0])
    assert step1[1:] == step2[1:]


def test_outage_has_no_fictitious_energy():
    sim = SiteSimulator([SiteInputs(grid_available=False, generator_available=False)])
    _, _, _, truncated, info = sim.advance(SiteAction(10, 4, -5, 8, 15))
    assert truncated
    assert info["electricity_mwh"] == 0
    assert info["unserved_energy_mwh"] > 0
    assert info["applied_action"]["battery_power_mw"] == 0
    assert info["power_balance_residual_mw"] == 0
    with pytest.raises(RuntimeError):
        sim.advance(SiteAction(0, 0, 0, 0, 0))


@pytest.mark.parametrize("field", ["energy_capacity_mwh", "charge_efficiency", "initial_power_mw"])
def test_invalid_battery_configuration(field):
    with pytest.raises(ValueError):
        replace(BatteryConfig(), **{field: float("nan")})
