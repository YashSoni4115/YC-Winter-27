import json

import pytest

from energy_decision_layer.__main__ import main
from energy_decision_layer.evaluation.integrated import run_ml_evaluation
from energy_decision_layer.workflows import (
    create_runtime,
    demo_inputs,
    generate_dataset,
    read_intervals,
    run_demo,
    train_components,
)


@pytest.fixture(scope="module")
def trained_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("integrated-models")
    config = root / "train.json"
    config.write_text(
        json.dumps({"forecast": {"n_estimators": 8}, "policy": {"iterations": 1, "population": 4}})
    )
    report = train_components("train-all", root / "models", steps=120, config=config)
    assert set(report) == {"forecast", "policy", "surrogates"}
    return root / "models"


def test_normalized_fixture_can_train_and_replay_without_schema_changes(tmp_path):
    data = tmp_path / "normalized.jsonl"
    rows = generate_dataset(data, steps=120, seed=7)
    assert read_intervals(data) == rows
    model_dir = tmp_path / "models"
    train_components("train-forecast", model_dir, data=data)
    runtime = create_runtime(model_dir, tmp_path / "run.sqlite3", data=data)
    try:
        assert runtime.current.timestamp_utc == rows[0].interval_start_utc
        assert runtime.current.workload_demand_compute_units_per_h == pytest.approx(
            rows[0].workload_demand_compute_units / rows[0].duration_hours
        )
        result = runtime.dispatch("baseline", idempotency_key="normalized-step")
        assert result["executed"]
        assert runtime.current.step_index == 1
        assert "learned" not in runtime.policies
    finally:
        runtime.close()
    empty = tmp_path / "empty.jsonl"
    empty.touch()
    with pytest.raises(ValueError, match="empty"):
        read_intervals(empty)
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"unsupported": {}}')
    with pytest.raises(ValueError, match="unknown"):
        train_components("train-all", tmp_path / "bad-model", config=invalid)
    with pytest.raises(ValueError, match="at least four"):
        demo_inputs(1)


def test_complete_demo_includes_real_unsafe_projection_charge_discharge_and_audit(
    trained_models, tmp_path
):
    result = run_demo(trained_models, tmp_path / "demo", steps=8)
    assert result["steps"] == 8
    assert result["simulation_only"]
    assert result["initial_recommendation"]["surrogate"]["advisory_only"]
    assert "predictions" in result["initial_recommendation"]["surrogate"]
    dispatches = result["dispatches"]
    assert all(item["executed"] for item in dispatches)
    assert dispatches[0]["outcome"]["applied_action"]["battery_power_mw"] < 0
    assert dispatches[1]["outcome"]["applied_action"]["battery_power_mw"] > 0
    assert dispatches[1]["safety"]["status"] == "modified"
    assert dispatches[1]["proposal"]["generator_power_mw"] == 100
    assert dispatches[1]["outcome"]["applied_action"]["generator_power_mw"] < 8
    assert result["state_api"]["step_index"] == 8
    assert len(result["decisions_api"]["items"]) == 8
    assert (tmp_path / "demo" / "decisions.sqlite3").is_file()
    with pytest.raises(ValueError, match="already contains"):
        run_demo(trained_models, tmp_path / "demo", steps=8)


def test_real_comparisons_persist_forecasts_shield_outcomes_and_exact_regret(
    trained_models, tmp_path
):
    result = run_ml_evaluation(trained_models, tmp_path / "evaluation", steps=2)
    assert result["failed_runs"] == 0
    assert len(result["additional_metrics"]) == 48
    assert all(r["grid_limit_violations"] == 0 for r in result["additional_metrics"])
    assert len(list((tmp_path / "evaluation" / "traces").glob("*.json"))) == 48
    rows = json.loads((tmp_path / "evaluation" / "metrics.json").read_text())
    assert len(rows) == 48
    assert all(row["shield_name"] == "coupled-physics-shield" for row in rows)
    assert "not a proven global optimum" in (tmp_path / "evaluation" / "summary.md").read_text()


def test_new_cli_generates_reproducible_normalized_data(monkeypatch, tmp_path, capsys):
    target = tmp_path / "fixture.jsonl"
    monkeypatch.setattr(
        "sys.argv", ["edl", "synthetic", "--steps", "60", "--seed", "7", "--output", str(target)]
    )
    main()
    assert len(read_intervals(target)) == 60
    assert "labelled synthetic" in capsys.readouterr().out
