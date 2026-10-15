import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from energy_decision_layer.contracts import (
    ProposedControlActionV1,
    SafetyResponseV1,
    SiteObservationV1,
)
from energy_decision_layer.optimizer import RulePolicy
from energy_decision_layer.sim import SiteInputs, SiteSimulator
from energy_decision_layer.testing.doubles import (
    DevelopmentAcceptAllShield,
    PersistenceForecastDouble,
)


def test_contracts_json_round_trip():
    o, _ = SiteSimulator([SiteInputs()]).reset()
    p = RulePolicy().propose_action(o)
    r = DevelopmentAcceptAllShield().evaluate(p, o)
    f = PersistenceForecastDouble().forecast(o, horizon_steps=3, cadence_seconds=300.0)
    for obj in (o, p, r, r.executed_action, f):
        assert type(obj).model_validate_json(obj.model_dump_json()) == obj


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": "2.0.0"},
        {"compute_power_mw": float("nan")},
        {"cooling_power_mw": -1.0},
        {"surprise": 1},
        {"created_at_utc": datetime(2026, 1, 1)},
        {"is_fallback": True},
    ],
)
def test_invalid_proposals_rejected(change):
    o, _ = SiteSimulator([SiteInputs()]).reset()
    payload = RulePolicy().propose_action(o).model_dump()
    payload.update(change)
    with pytest.raises(ValidationError):
        ProposedControlActionV1.model_validate(payload)


def test_expired_stale_and_unapproved_actions_rejected():
    sim = SiteSimulator([SiteInputs()] * 3)
    o, _ = sim.reset()
    p = RulePolicy().propose_action(o)
    with pytest.raises(TypeError):
        sim.step(p)
    authorized = DevelopmentAcceptAllShield().evaluate(p, o).executed_action
    sim.step(authorized)
    with pytest.raises(ValueError, match="stale"):
        sim.step(authorized)
    assert sim.index == 1


def test_modified_and_rejected_status_consistency():
    o, _ = SiteSimulator([SiteInputs()]).reset()
    p = RulePolicy().propose_action(o)
    r = DevelopmentAcceptAllShield().evaluate(p, o)
    payload = r.model_dump(mode="json")
    payload["status"] = "rejected"
    with pytest.raises(ValidationError):
        SafetyResponseV1.model_validate_json(json.dumps(payload))
    payload = r.executed_action.model_dump(mode="json")
    payload["compute_power_mw"] += 1
    with pytest.raises(ValidationError):
        type(r.executed_action).model_validate_json(json.dumps(payload))


def test_committed_schemas_match_models():
    from energy_decision_layer import contracts
    from energy_decision_layer.data.models import SimulationIntervalV1

    models = [
        contracts.TelemetryV1,
        contracts.SiteObservationV1,
        contracts.ForecastBundleV1,
        contracts.ProposedControlActionV1,
        contracts.ExecutedControlActionV1,
        contracts.SafetyResponseV1,
        SimulationIntervalV1,
    ]
    for model in models:
        assert (
            json.loads(Path(f"docs/schemas/{model.__name__}.json").read_text())
            == model.model_json_schema()
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"battery_energy_mwh": 30.0},
        {"battery_state_of_charge": 0.7},
        {"grid_available": False, "grid_power_mw": 1.0},
        {"generator_available": False, "generator_power_mw": 1.0},
        {"generator_is_on": False, "generator_power_mw": 1.0},
        {"renewable_used_power_mw": 1.0},
        {"curtailed_power_mw": 1.0},
    ],
)
def test_inconsistent_observation_rejected(updates):
    o, _ = SiteSimulator([SiteInputs()]).reset()
    payload = o.model_dump()
    payload.update(updates)
    with pytest.raises(ValidationError):
        SiteObservationV1.model_validate(payload)


def test_forecast_uncertainty_and_issuance_rejection():
    from energy_decision_layer.contracts import (
        ForecastPointV1,
        ForecastSignal,
        PredictionIntervalV1,
    )

    o, _ = SiteSimulator([SiteInputs()]).reset()
    f = PersistenceForecastDouble().forecast(o, horizon_steps=3, cadence_seconds=300.0)
    p = f.model_dump()
    p["issued_at_utc"] = o.timestamp_utc + timedelta(seconds=1)
    with pytest.raises(ValidationError):
        type(f).model_validate(p)
    interval = PredictionIntervalV1(
        signal=ForecastSignal.ELECTRICITY_PRICE,
        lower_value=0.0,
        upper_value=100.0,
        coverage_probability=0.9,
    )
    point = f.points[0].model_dump()
    point["prediction_intervals"] = (interval, interval)
    with pytest.raises(ValidationError, match="duplicate"):
        ForecastPointV1.model_validate(point)
    with pytest.raises(ValidationError):
        PredictionIntervalV1(
            signal=ForecastSignal.ELECTRICITY_PRICE,
            lower_value=100.0,
            upper_value=0.0,
            coverage_probability=0.9,
        )


def test_raw_telemetry_quality_flags_are_consistent():
    from energy_decision_layer.contracts import DataQualityFlag, ProvenanceV1, TelemetryV1

    o, _ = SiteSimulator([SiteInputs()]).reset()
    base = dict(
        site_id=o.site_id,
        state_id=o.state_id,
        timestamp_utc=o.timestamp_utc,
        sequence_number=0,
        provenance=ProvenanceV1(source_name="test", ingested_at_utc=o.timestamp_utc),
    )
    good = TelemetryV1(
        **base, compute_power_mw=1.0, quality_flags={"compute_power_mw": DataQualityFlag.GOOD}
    )
    assert good.compute_power_mw == 1
    for flags in ({"unknown": DataQualityFlag.GOOD}, {"compute_power_mw": DataQualityFlag.MISSING}):
        with pytest.raises(ValidationError):
            TelemetryV1(**base, compute_power_mw=1.0, quality_flags=flags)


def test_action_cannot_be_replayed_after_reset_or_in_another_simulator():
    sim = SiteSimulator([SiteInputs(nondeferrable_compute_units=40.0)])
    original, _ = sim.reset(seed=7)
    proposal = RulePolicy().propose_action(original)
    executed = DevelopmentAcceptAllShield().evaluate(proposal, original).executed_action
    reset_observation, _ = sim.reset(seed=7)
    other = SiteSimulator(sim.inputs)
    assert original.state_id != reset_observation.state_id != other.observation().state_id
    # Physics is reproducible; execution identities are intentionally fresh.
    assert original.battery_energy_mwh == reset_observation.battery_energy_mwh
    for target in (sim, other):
        with pytest.raises(ValueError, match="stale"):
            target.step(executed)
        assert target.index == 0
