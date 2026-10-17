from __future__ import annotations

import numpy as np
import pytest

from energy_decision_layer.contracts import ForecastBundleV1, ForecastProviderProtocol
from energy_decision_layer.evaluation.runner import evaluate
from energy_decision_layer.optimizer import RulePolicy
from energy_decision_layer.sim import SiteSimulator
from energy_ml.forecaster import MLForecastProvider, QuantileForecastModel
from energy_ml.synthetic import AMBIENT, PRICE, SIGNALS, WORKLOAD, generate_series, to_site_inputs


@pytest.fixture(scope="module")
def model() -> QuantileForecastModel:
    return QuantileForecastModel(max_iter=40).fit(generate_series(8 * 288, seed=11))


def _observation(index: int = 0):
    sim = SiteSimulator(to_site_inputs(generate_series(60, seed=2)))
    return sim.reset()[0]


def test_provider_satisfies_protocol_and_emits_valid_bundle(model):
    provider = MLForecastProvider(model)
    assert isinstance(provider, ForecastProviderProtocol)
    o = _observation()
    bundle = provider.forecast(o, horizon_steps=12, cadence_seconds=300.0)
    # Round trip through strict JSON validation, as a service client would.
    ForecastBundleV1.model_validate_json(bundle.model_dump_json())
    assert len(bundle.points) == 12
    assert bundle.issued_at_utc == o.timestamp_utc
    assert bundle.valid_from_utc == o.timestamp_utc
    assert bundle.provenance.is_synthetic is True
    assert bundle.is_perfect_hindsight is False
    first = bundle.points[0]  # current interval is observed, so exact and band-free
    assert first.workload_demand_compute_units_per_h == o.workload_demand_compute_units_per_h
    assert first.prediction_intervals == ()
    for point in bundle.points[1:]:
        assert {b.signal.value for b in point.prediction_intervals} == set(SIGNALS)


def test_rejects_untrained_horizon_and_cadence(model):
    provider = MLForecastProvider(model)
    o = _observation()
    with pytest.raises(ValueError, match="exceeds the trained horizon"):
        provider.forecast(o, horizon_steps=13, cadence_seconds=300.0)
    with pytest.raises(ValueError, match="cadence"):
        provider.forecast(o, horizon_steps=4, cadence_seconds=600.0)
    with pytest.raises(ValueError):
        provider.forecast(o, horizon_steps=0, cadence_seconds=300.0)


def test_single_step_horizon_needs_no_model_call(model):
    bundle = MLForecastProvider(model).forecast(_observation(), horizon_steps=1, cadence_seconds=300.0)
    assert len(bundle.points) == 1


def test_unfitted_model_is_rejected():
    with pytest.raises(ValueError, match="not been fitted"):
        MLForecastProvider(QuantileForecastModel())


def test_training_rejects_irregular_or_short_data():
    frame = generate_series(8 * 288, seed=0)
    with pytest.raises(ValueError, match="regular index"):
        QuantileForecastModel().fit(frame.iloc[[*range(10), *range(12, 8 * 288)]])
    with pytest.raises(ValueError, match="need >= 2 days"):
        QuantileForecastModel().fit(generate_series(300, seed=0))
    with pytest.raises(ValueError, match="missing"):
        bad = frame.copy()
        bad.iloc[5, 0] = np.nan
        QuantileForecastModel().fit(bad)


def test_future_inputs_cannot_change_a_forecast(model):
    """Two scenarios identical up to step K but different afterwards must forecast identically at K."""
    k, n = 15, 40
    base = generate_series(n, seed=5)
    altered = base.copy()
    altered.iloc[k + 5 :, :] = generate_series(n, seed=99).iloc[k + 5 :, :].to_numpy()

    def forecasts(frame):
        seen = {}
        provider = MLForecastProvider(model)

        class Recorder:
            def forecast(self, observation, *, horizon_steps, cadence_seconds):
                bundle = provider.forecast(
                    observation, horizon_steps=horizon_steps, cadence_seconds=cadence_seconds
                )
                seen[observation.step_index] = bundle
                return bundle

        evaluate("leak-check", to_site_inputs(frame), lambda: RulePolicy(), forecaster=Recorder())
        return seen

    a, b = forecasts(base), forecasts(altered)
    # Guard against a vacuous test: once the scenarios diverge, forecasts must differ too.
    assert a[k + 6].model_dump(exclude={"forecast_id"}) != b[k + 6].model_dump(
        exclude={"forecast_id"}
    ), "scenarios never diverged, so this leakage test has no power"
    for step in range(k + 1):
        da = a[step].model_dump(exclude={"forecast_id"})
        db = b[step].model_dump(exclude={"forecast_id"})
        assert da == db, f"forecast at step {step} depends on future inputs"


def test_history_resets_on_new_episode(model):
    provider = MLForecastProvider(model)
    sim = SiteSimulator(to_site_inputs(generate_series(60, seed=2)))
    o0 = sim.reset()[0]
    provider.forecast(o0, horizon_steps=4, cadence_seconds=300.0)
    first = provider.forecast(o0, horizon_steps=4, cadence_seconds=300.0)  # repeat call, same interval
    o1 = sim.reset()[0]  # new episode, timestamps restart
    again = provider.forecast(o1, horizon_steps=4, cadence_seconds=300.0)
    assert [p.workload_demand_compute_units_per_h for p in first.points] == [
        p.workload_demand_compute_units_per_h for p in again.points
    ]


def test_intervals_are_calibrated_out_of_sample():
    """Empirical coverage on a later, unseen period should be near the nominal 80%."""
    series = generate_series(14 * 288, seed=21)
    train, test = series.iloc[: 10 * 288], series.iloc[10 * 288 - 288 :]  # 1 day overlap for lags
    m = QuantileForecastModel(max_iter=60).fit(train)
    for signal in (WORKLOAD, PRICE, AMBIENT):
        covered = []
        for h in (1, 6, 11):
            lo, pt, hi = m.predict_vectorized(signal, test[signal].to_numpy(), test.index, h)
            y = test[signal].to_numpy()[h:]
            covered.append(np.mean((y >= lo[:-h]) & (y <= hi[:-h])))
        assert 0.65 <= float(np.mean(covered)) <= 0.97, (signal, covered)
