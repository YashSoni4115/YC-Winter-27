"""ML contracts, chronological leakage boundaries and real shield integration."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from energy_decision_layer.ml import (
    DynamicsSurrogate,
    ForecastModel,
    ForecastProvider,
    LearnedPolicy,
    features,
    policy_score,
    synthetic_intervals,
    train_policy,
    train_surrogates,
)
from energy_decision_layer.optimizer.planner import MilpPolicy
from energy_decision_layer.safety import SafetyShield
from energy_decision_layer.sim import SiteConfig, SiteSimulator
from energy_decision_layer.sim.common import SiteInputs


@pytest.fixture(scope="module")
def rows():
    return synthetic_intervals(400)


@pytest.fixture(scope="module")
def model(rows):
    return ForecastModel.fit(rows, seed=4, n_estimators=8)


def test_feature_engineering_only_reads_past_and_handles_missing(rows):
    history = np.array([[1.0, 23.0, 50.0], [np.nan, 24.0, 55.0]])
    first = features(history, rows[2].interval_start_utc, 288)
    assert first.shape == (52,)
    assert first[4] == 1.0 and first[5] == 1.0
    assert first[20] == 24.0
    assert np.isfinite(first).all()
    assert np.array_equal(history, np.array([[1, 23, 50], [np.nan, 24, 55]]), equal_nan=True)


def test_forecast_chronological_fit_baselines_and_artifact(model, rows, tmp_path):
    assert model.training_end < rows[-1].interval_start_utc
    assert model.report["data_kind"] == "synthetic"
    for signal in model.report["signals"].values():
        metrics = signal["test"]
        assert metrics["rmse"] >= metrics["mae"] >= 0
        assert 0 <= metrics["coverage"] <= 1
        assert metrics["mean_interval_width"] >= 0
        assert "seasonal_naive" in metrics and "persistence" in metrics
    model.save(tmp_path)
    restored = ForecastModel.load(tmp_path)
    assert restored.version == model.version
    history = np.array([[500, 23, 50], [550, 24, 60]])
    expected = model.predict(history, rows[-1].interval_end_utc)[0]
    assert np.array_equal(expected, restored.predict(history, rows[-1].interval_end_utc)[0])
    with pytest.raises(ValueError, match="after forecast"):
        model.predict(history, rows[0].interval_start_utc)


def test_holdout_targets_never_change_fitted_models(rows):
    changed = tuple(
        r.model_copy(update={"ambient_temperature_c": 120.0}) if i >= 320 else r
        for i, r in enumerate(rows)
    )
    a = ForecastModel.fit(rows, n_estimators=4)
    b = ForecastModel.fit(changed, n_estimators=4)
    history = np.array([[500, 23, 50], [550, 24, 60]])
    assert np.array_equal(
        a.predict(history, rows[-1].interval_end_utc)[0],
        b.predict(history, rows[-1].interval_end_utc)[0],
    )
    assert a.report["signals"] != b.report["signals"]


def test_forecast_provider_consumes_etl_to_planner(model, rows):
    start = rows[-1].interval_end_utc
    sim = SiteSimulator([SiteInputs(nondeferrable_compute_units=40)] * 4, start=start)
    o = sim.observation()
    # Poisoned future fixture rows cannot enter inference history.
    future = rows[-1].model_copy(
        update={
            "interval_start_utc": start + timedelta(days=10),
            "interval_end_utc": start + timedelta(days=10, minutes=5),
            "ambient_temperature_c": 200.0,
        }
    )
    provider = ForecastProvider(model, (*rows, future))
    forecast = provider.forecast(o, horizon_steps=3, cadence_seconds=300.0)
    clean = ForecastProvider(model, rows).forecast(o, horizon_steps=3, cadence_seconds=300.0)
    assert provider.last_status == "trained"
    assert forecast.points == clean.points
    assert forecast.points[0].workload_demand_compute_units_per_h == 480
    assert forecast.model_version == model.version
    assert forecast.points[1].prediction_intervals
    p = MilpPolicy().propose_action(o, forecast)
    response = SafetyShield().evaluate(p, o)
    assert response.executed_action is not None
    next_observation, _, _, _, info = sim.step(response.executed_action)
    assert next_observation.step_index == 1
    assert abs(info["power_balance_residual_mw"]) < 1e-7


def test_fallback_missing_history_and_inference_failure(model, rows, monkeypatch):
    sim = SiteSimulator([SiteInputs()] * 3, start=rows[-1].interval_end_utc)
    o = sim.observation()
    provider = ForecastProvider()
    forecast = provider.forecast(o, horizon_steps=3, cadence_seconds=300)
    assert "fallback" in forecast.model_name
    assert forecast.points[1].prediction_intervals[0].upper_value > 0
    provider = ForecastProvider(model, rows)

    def fail(*args, **kwargs):
        raise RuntimeError("model failed")

    monkeypatch.setattr(model, "predict", fail)
    forecast = provider.forecast(o, horizon_steps=3, cadence_seconds=300)
    assert "inference failure" in forecast.model_name
    assert len(forecast.points) == 3
    with pytest.raises(ValueError):
        provider.forecast(o, horizon_steps=0, cadence_seconds=300)
    with pytest.raises(ValueError):
        provider.forecast(o, horizon_steps=3, cadence_seconds=60)


def test_training_rejects_wrong_currency_cadence_and_insufficient(rows):
    with pytest.raises(ValueError, match="60"):
        ForecastModel.fit(rows[:10])
    wrong = tuple(r.model_copy(update={"price_currency": "CAD"}) for r in rows)
    with pytest.raises(ValueError, match="USD"):
        ForecastModel.fit(wrong)
    with pytest.raises(ValueError, match="contiguous"):
        ForecastModel.fit((*rows[:100], *rows[101:]))


def test_policy_training_reproducible_checkpoint_and_real_safety(tmp_path):
    inputs = tuple(
        SiteInputs(
            nondeferrable_compute_units=35 + i % 3,
            grid_import_price_per_mwh=30.0 if i < 5 else 140.0,
            ambient_temperature_c=25.0 + i,
            grid_import_limit_mw=5.0,
        )
        for i in range(12)
    )
    first = train_policy(inputs, tmp_path / "a.json", seed=9, iterations=1, population=4)
    second = train_policy(inputs, tmp_path / "b.json", seed=9, iterations=1, population=4)
    assert first == second
    assert first["training"]["shield_rejections"] == 0
    policy = LearnedPolicy.load(tmp_path / "a.json")
    sim = SiteSimulator(inputs)
    o = sim.observation()
    proposal = policy.propose_action(o)
    assert proposal.based_on_state_id == o.state_id
    assert proposal.policy_version.startswith("cem-")
    shield = SafetyShield()
    response = shield.evaluate(proposal, o)
    assert response.executed_action is not None
    sim.step(response.executed_action)
    assert policy_score(policy, inputs)["steps"] == len(inputs)
    with pytest.raises(ValueError, match="configuration"):
        LearnedPolicy.load(tmp_path / "a.json", config=SiteConfig(grid_import_limit_mw=10))
    with pytest.raises(ValueError):
        LearnedPolicy(parameters=[np.nan] * 5)


def test_surrogate_training_predictions_and_physical_bounds(tmp_path):
    inputs = tuple(
        SiteInputs(
            nondeferrable_compute_units=30 + i % 8,
            grid_import_price_per_mwh=30.0 + i * 10,
            ambient_temperature_c=25 + i,
        )
        for i in range(12)
    )
    report = train_surrogates(inputs, tmp_path, seed=4, episodes=6)
    assert report["advisory_only"] is True
    assert report["samples"]["test"] > 0
    surrogate = DynamicsSurrogate.load(tmp_path)
    sim = SiteSimulator(inputs)
    o = sim.observation()
    p = LearnedPolicy().propose_action(o)
    prediction = surrogate.predict(o, p)
    assert prediction["advisory_only"]
    assert set(prediction["predictions"]) == {
        "facility_temperature_c",
        "cooling_power_mw",
        "battery_degradation_cost_usd",
        "generator_power_mw",
    }
    for value in prediction["predictions"].values():
        assert value["lower"] <= value["value"] <= value["upper"]
    assert 0 <= prediction["predictions"]["cooling_power_mw"]["value"] <= 4
    assert prediction["predictions"]["battery_degradation_cost_usd"]["value"] >= 0
    outage = o.model_copy(update={"generator_available": False})
    assert surrogate.predict(outage, p)["predictions"]["generator_power_mw"]["value"] == 0


def test_missing_labels_quality_and_calibration_cutoff(rows, tmp_path):
    missing = tuple(
        r.model_copy(update={"ambient_temperature_c": None, "grid_stress_status": "unknown"})
        if i >= 240
        else r.model_copy(update={"grid_stress_status": "unknown"})
        for i, r in enumerate(rows)
    )
    m = ForecastModel.fit(missing, n_estimators=4)
    ambient = m.report["signals"]["ambient_temperature_c"]
    assert ambient["test"]["status"].startswith("unavailable")
    assert ambient["validation"]["status"].startswith("unavailable")
    assert "unavailable" in m.report["grid_stress"]["status"]
    m.save(tmp_path)  # allow_nan=False must serialize even wholly missing holdout labels.
    obs = SiteSimulator([SiteInputs()], start=m.training_end).observation()
    forecast = ForecastProvider(m, rows).forecast(obs, horizon_steps=3, cadence_seconds=300)
    assert "fallback" in forecast.model_name  # validation calibration is still in future.
    assert m.available_after > m.training_end
    with pytest.raises(ValueError, match="after forecast"):
        m.predict(np.zeros((3, 3)), m.training_end)


def test_stress_classifier_probability_and_labelled_status(rows, model):
    assert model.classifier is not None
    assert model.report["grid_stress"]["status"] == "trained"
    assert 0 <= model.report["grid_stress"]["brier"] <= 1
    provider = ForecastProvider(model, rows)
    obs = SiteSimulator([SiteInputs()], start=rows[-1].interval_end_utc).observation()
    bundle = provider.forecast(obs, horizon_steps=4, cadence_seconds=300)
    assert provider.last_stress_probabilities[0] is None  # current is telemetry nowcast
    for probability, point in zip(
        provider.last_stress_probabilities[1:], bundle.points[1:], strict=True
    ):
        assert probability is not None and 0 <= probability <= 1
        assert point.grid_stress_status.value == ("elevated" if probability >= 0.5 else "normal")


def test_invalid_training_and_surrogate_inputs(tmp_path):
    from energy_decision_layer.ml.forecasting import regression_metrics

    assert "unavailable" in regression_metrics(np.empty(0), np.empty(0))["status"]
    with pytest.raises(ValueError):
        synthetic_intervals(0)
    with pytest.raises(ValueError):
        train_policy([SiteInputs()] * 3, tmp_path)
    with pytest.raises(ValueError):
        train_surrogates([SiteInputs()] * 3, tmp_path)
    surrogate = DynamicsSurrogate()
    sim = SiteSimulator([SiteInputs()])
    o = sim.observation()
    assert "Untrained default" in LearnedPolicy().propose_action(o).rationale
    with pytest.raises(ValueError, match="not trained"):
        surrogate.predict(o, LearnedPolicy().propose_action(o))


def test_multihorizon_evaluation_runs_recursive_heldout_forecasts(model, rows):
    report = model.report["multi_horizon_test"]
    assert report["issuance_origins"] > 0
    assert set(report["horizons"]) == {"1", "3", "12"}
    for horizon in report["horizons"].values():
        for signal in horizon.values():
            assert signal["mae"] >= 0
            assert 0 <= signal["coverage"] <= 1
            assert signal["persistence"]["mae"] >= 0
    # Fitting did not claim that the early validation/calibration period is held-out inference.
    early = model.evaluate_horizons(rows[:100])
    assert "unavailable" in early["status"]
    with pytest.raises(ValueError):
        model.evaluate_horizons(rows, horizons=(0,))
