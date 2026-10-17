from dataclasses import replace

import pytest

from energy_decision_layer.evaluation.integrated import (
    ExperimentLogger,
    ForecastPerturbation,
    analytic_regret,
    operational_details,
)
from energy_decision_layer.evaluation.runner import evaluate, scenarios
from energy_decision_layer.optimizer import MilpPolicy, RulePolicy
from energy_decision_layer.safety import SafetyShield
from energy_decision_layer.sim import SiteConfig, SiteSimulator
from energy_decision_layer.testing.doubles import PersistenceForecastDouble


def test_bias_experiments_keep_known_current_values_and_valid_intervals():
    o, _ = SiteSimulator(scenarios(4)["normal"]).reset(seed=7)
    base = PersistenceForecastDouble()
    for bias in (-0.25, 0.0, 0.25):
        f = ForecastPerturbation(base, bias).forecast(o, horizon_steps=3, cadence_seconds=300.0)
        assert f.points[0].electricity_price_usd_per_mwh == o.electricity_price_usd_per_mwh
        assert f.points[1].workload_demand_compute_units_per_h == pytest.approx(
            o.workload_demand_compute_units_per_h * (1 + bias)
        )
        assert f.points[1].ambient_temperature_c == pytest.approx(
            o.ambient_temperature_c + bias * 10
        )
    with pytest.raises(ValueError):
        ForecastPerturbation(base, -1.0)


def test_exact_restricted_hindsight_optimum_is_independently_verified():
    report = analytic_regret({"rule": RulePolicy(), "milp": MilpPolicy()})
    assert report["optimum_objective_usd"] == pytest.approx(37.5)
    rows = {r["policy"]: r for r in report["results"]}
    assert not rows["milp"]["failed"]
    assert rows["milp"]["regret_usd"] == pytest.approx(0, abs=1e-5)
    assert rows["rule"]["regret_usd"] >= 0


def test_real_safety_and_adapter_flow_reuses_existing_evaluation():
    from energy_decision_layer.adapters import MockSiteAdapter

    inputs = scenarios(6)["price_spike"]
    logger = ExperimentLogger()
    result = evaluate(
        "integrated",
        inputs,
        MilpPolicy,
        shield=SafetyShield(),
        decision_logger=logger,
        execution_adapter=MockSiteAdapter,
    )
    assert not result.failed
    assert result.steps == 6
    assert result.shield_name == SafetyShield.name
    details = operational_details(logger, inputs, SiteConfig())
    assert details["decision_count"] == details["executed_count"] == 6
    assert details["thermal_limit_violations"] == details["grid_limit_violations"] == 0
    assert all(
        o["metrics"]["power_balance_residual_mw"] == pytest.approx(0) for o in logger.outcomes
    )


def test_operational_metrics_detect_actual_violations():
    logger = ExperimentLogger()
    logger.outcomes = [
        {"metrics": {"thermal_violation": True, "grid_power_mw": 5.0, "generator_fuel_mwh": 1.5}}
    ]
    inputs = [replace(scenarios(1)["normal"][0], grid_available=False)]
    details = operational_details(logger, inputs, SiteConfig())
    assert details["thermal_limit_violations"] == details["grid_limit_violations"] == 1
    assert details["generator_fuel_mwh_thermal"] == 1.5
