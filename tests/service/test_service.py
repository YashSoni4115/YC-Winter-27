import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from energy_decision_layer.optimizer.planner import RulePolicy, proposal
from energy_decision_layer.service.api import create_app
from energy_decision_layer.service.runtime import SimulationRuntime, demo_simulator
from energy_decision_layer.service.store import encode


@pytest.fixture
def runtime():
    r = SimulationRuntime(
        db_path=":memory:", operator_token="test-operator", dispatcher_token="test-dispatch"
    )
    yield r
    r.close()


@pytest.fixture
def client(runtime):
    with TestClient(create_app(runtime)) as c:
        yield c


def headers(key="request-1", operator=False):
    return {
        "Authorization": "Bearer test-operator" if operator else "Bearer test-dispatch",
        "Idempotency-Key": key,
    }


def test_dashboard_and_all_read_endpoints(client):
    for path in [
        "/",
        "/static/app.js",
        "/static/style.css",
        "/health",
        "/ready",
        "/api/state",
        "/api/telemetry",
        "/api/forecasts",
        "/api/policies",
        "/api/metrics",
        "/api/decisions",
        "/api/forecast-status",
        "/api/evaluation",
        "/openapi.json",
    ]:
        assert client.get(path).status_code == 200, path
    assert "SIMULATION ONLY" in client.get("/").text
    assert client.get("/api/recommendations/learned").status_code == 404
    assert client.get("/api/policies").json()["can_dispatch"] is False
    assert client.get("/api/telemetry?limit=0").status_code == 422


def test_end_to_end_baseline_forecast_shield_adapter_telemetry_audit(client, runtime):
    forecast = client.get("/api/forecasts").json()
    assert len(forecast["points"]) == 12
    recommended = client.get("/api/recommendations/baseline").json()
    assert not recommended["executed"]
    assert recommended["safety"]["executed_action"] is not None
    for i in range(4):
        response = client.post(
            "/api/dispatch", json={"policy": "baseline"}, headers=headers(str(i))
        )
        assert response.status_code == 200, response.text
        assert response.json()["executed"]
    assert client.get("/api/state").json()["step_index"] == 4
    assert len(client.get("/api/decisions").json()["items"]) == 4
    assert len(client.get("/api/telemetry").json()["items"]) == 5
    log = client.get("/api/decisions").json()["items"][0]
    assert log["forecast"]["model_version"]
    assert log["proposal"]["action_id"] == log["response"]["action_id"]
    assert log["status"] == "executed"
    assert runtime.metrics()["policies"]["milp-baseline"]["executed_steps"] == 4


def test_auth_roles_and_readonly_defaults(client, runtime):
    assert client.post("/api/dispatch", json={"policy": "rule"}).status_code == 401
    assert (
        client.post(
            "/api/dispatch",
            json={"policy": "rule"},
            headers={"Authorization": "Bearer incorrect", "Idempotency-Key": "denied"},
        ).status_code
        == 401
    )
    p = RulePolicy().propose_action(runtime.current).model_dump(mode="json")
    assert client.post("/api/manual-overrides", json=p, headers=headers()).status_code == 403
    assert (
        client.post(
            "/api/dispatch", json={"policy": "rule", "extra": True}, headers=headers()
        ).status_code
        == 422
    )
    assert runtime.current.step_index == 0
    with pytest.raises(ValueError, match="distinct"):
        SimulationRuntime(db_path=":memory:", operator_token="same", dispatcher_token="same")


def test_unsafe_manual_proposal_modified_and_stale_rejected(client, runtime):
    p = proposal(runtime.current, (90, 20, 30, 30, 30), "manual-operator").model_dump(mode="json")
    validation = client.post("/api/safety/validate", json=p)
    assert validation.status_code == 200, validation.text
    assert validation.json()["status"] == "modified"
    assert runtime.current.step_index == 0
    result = client.post("/api/manual-overrides", json=p, headers=headers("manual", True))
    assert result.status_code == 200, result.text
    assert result.json()["executed"] is True
    assert result.json()["safety"]["executed_action"]["grid_power_mw"] <= 15
    stale = client.post("/api/actions", json=p, headers=headers("stale", True)).json()
    assert stale["executed"] is False
    assert runtime.current.step_index == 1
    invalid = {**p, "battery_power_mw": None}
    assert client.post("/api/safety/validate", json=invalid).status_code == 422


def test_idempotency_cannot_duplicate_or_substitute_actions(client, runtime):
    first = client.post("/api/dispatch", json={"policy": "rule"}, headers=headers())
    duplicate = client.post("/api/dispatch", json={"policy": "rule"}, headers=headers())
    assert duplicate.json() == first.json()
    assert runtime.current.step_index == 1
    collision = client.post("/api/dispatch", json={"policy": "baseline"}, headers=headers())
    assert collision.status_code == 409
    assert (
        client.post(
            "/api/dispatch", json={"policy": "learned"}, headers=headers("learned")
        ).status_code
        == 409
    )
    assert (
        client.post("/api/policies/select", json={"policy": "rule"}, headers=headers()).json()[
            "selected"
        ]
        == "rule"
    )
    assert (
        client.post(
            "/api/policies/select", json={"policy": "learned"}, headers=headers()
        ).status_code
        == 404
    )


def test_restart_restores_physics_queue_and_cached_result(tmp_path):
    path = tmp_path / "audit.db"
    first = SimulationRuntime(db_path=path)
    result = first.dispatch("rule", idempotency_key="persisted")
    state = first.current
    first.close()
    second = SimulationRuntime(db_path=path)
    assert second.current == state
    assert second.dispatch("rule", idempotency_key="persisted") == result
    assert second.dispatch("baseline", idempotency_key="next")["executed"]
    assert len(second.store.history("decisions")) == 2
    second.close()
    with pytest.raises(ValueError, match="scenario differs"):
        SimulationRuntime(simulator=demo_simulator(2), db_path=path)


def test_pending_restart_blocks_all_automatic_replay(tmp_path):
    path = tmp_path / "audit.db"
    runtime = SimulationRuntime(db_path=path)
    runtime.store.reserve("uncertain", "digest")
    runtime.close()
    restarted = SimulationRuntime(db_path=path)
    with TestClient(create_app(restarted)) as c:
        assert c.get("/ready").status_code == 503
        assert c.get("/health").json()["ready"] is False
    with pytest.raises(RuntimeError, match="unresolved"):
        restarted.dispatch("rule", idempotency_key="new")
    restarted.close()


def test_single_runtime_ownership_and_threaded_deduplication(tmp_path, runtime):
    path = tmp_path / "one.db"
    first = SimulationRuntime(db_path=path)
    with pytest.raises(RuntimeError, match="already owned"):
        SimulationRuntime(db_path=path)
    alias = tmp_path / "alias.db"
    alias.symlink_to(path)
    with pytest.raises(RuntimeError, match="already owned"):
        SimulationRuntime(db_path=alias)
    first.close()

    def run(_):
        return runtime.dispatch("rule", idempotency_key="parallel")

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, range(3)))
    assert results[0] == results[1] == results[2]
    assert runtime.current.step_index == 1


def test_adapter_failure_logged_without_execution(runtime):
    runtime.adapter.controllers["cooling"].available = False
    result = runtime.dispatch("rule", idempotency_key="failed")
    assert not result["executed"]
    assert runtime.current.step_index == 0
    assert runtime.store.history("decisions")[0]["status"] == "execution_failed"
    assert not runtime.store.unresolved()


def test_database_migration_and_newer_schema_releases_lock(tmp_path):
    import energy_decision_layer.service.store as module
    from energy_decision_layer.service.store import DecisionStore

    path = tmp_path / "version.db"
    connection = sqlite3.connect(path)
    migration = Path(module.__file__).with_name("migrations") / "001_initial.sql"
    connection.executescript(migration.read_text())
    connection.close()
    store = DecisionStore(path)
    assert store.connection.execute("PRAGMA user_version").fetchone()[0] == 2
    store.connection.execute("PRAGMA user_version=3")
    store.close()
    with pytest.raises(ValueError, match="newer"):
        DecisionStore(path)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version=2")
    connection.close()
    recovered = DecisionStore(path)
    recovered.close()


def test_forecast_failure_falls_back_but_preserves_manual_request(runtime):
    class BrokenForecast:
        def forecast(self, *args, **kwargs):
            raise ValueError("injected inference failure")

    runtime.forecaster = BrokenForecast()
    p = proposal(runtime.current, (4, 1, 0, 0, 5.2), "manual-explicit")
    manual = runtime.dispatch("manual", proposal=p, idempotency_key="manual-failed-forecast")
    assert manual["executed"]
    assert manual["proposal"]["action_id"] == str(p.action_id)
    assert manual["proposal"]["policy_name"] == "manual-explicit"
    baseline = runtime.dispatch("baseline", idempotency_key="baseline-failed-forecast")
    assert baseline["executed"]
    assert baseline["proposal"]["is_fallback"]
    assert runtime.store.history("decisions")[0]["forecast"] is None


def test_failed_outcome_commit_retains_pending_and_blocks_reexecution(runtime, monkeypatch):
    from energy_decision_layer.optimizer.dispatcher import DispatchOutcomeLoggingFailed

    def fail(*args):
        raise OSError("disk failed during checkpoint")

    monkeypatch.setattr(runtime.store, "add_telemetry", fail)
    with pytest.raises(DispatchOutcomeLoggingFailed):
        runtime.dispatch("rule", idempotency_key="uncertain")
    assert runtime.current.step_index == 1
    assert runtime.store.unresolved()
    with pytest.raises(RuntimeError, match="unresolved execution"):
        runtime.dispatch("rule", idempotency_key="uncertain")
    assert runtime.current.step_index == 1


def test_runtime_injected_learned_policy_is_shielded(runtime):
    class UnsafePolicy:
        def propose_action(self, o, forecast=None):
            return proposal(o, (80, 20, 20, 20, 20), "learned-test")

    runtime.policies["learned"] = UnsafePolicy()
    result = runtime.dispatch("learned", idempotency_key="learned-shielded")
    assert result["executed"]
    assert result["safety"]["status"] == "modified"
    assert result["safety"]["executed_action"]["compute_power_mw"] <= 10


def test_failed_forecast_and_recommendation_http_errors(client, runtime, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("inference failed")

    monkeypatch.setattr(runtime.forecaster, "forecast", fail)
    assert client.get("/api/forecasts").status_code == 503
    assert client.get("/api/recommendations/baseline").status_code == 503


def test_advisory_surrogate_never_overrides_physical_shield(client, runtime):
    class Advisory:
        def predict(self, observation, action):
            return {"advisory_only": True, "predictions": {"temperature": {"value": -100}}}

    runtime.surrogate = Advisory()
    recommendation = client.get("/api/recommendations/rule").json()
    assert recommendation["surrogate"]["advisory_only"]
    assert recommendation["safety"]["executed_action"] is not None
    runtime.surrogate = object()
    result = client.get("/api/recommendations/rule").json()
    assert result["surrogate"]["status"] == "inference_unavailable"
    assert result["safety"]["status"] == recommendation["safety"]["status"]


def test_scenario_completion_and_store_validation(tmp_path):
    runtime = SimulationRuntime(simulator=demo_simulator(1), db_path=":memory:")
    assert runtime.dispatch("rule", idempotency_key="finish")["executed"]
    with pytest.raises(ValueError, match="finished"):
        runtime.dispatch("rule", idempotency_key="extra")
    assert runtime.dispatch("rule", idempotency_key="finish")["executed"]
    with pytest.raises(ValueError, match="invalid history"):
        runtime.store.history("commands")
    with pytest.raises(TypeError):
        encode(object())
    runtime.close()


def test_evaluation_report_and_unsupported_physical_mode(client, monkeypatch, tmp_path):
    monkeypatch.setenv("EDL_EVALUATION_DIR", str(tmp_path))
    (tmp_path / "metrics.json").write_text('[{"policy":"rule"}]')
    assert client.get("/api/evaluation").json()["metrics.json"][0]["policy"] == "rule"
    (tmp_path / "metrics.json").write_text("invalid")
    assert client.get("/api/evaluation").status_code == 503
    monkeypatch.setenv("EDL_CONTROL_MODE", "physical")
    with pytest.raises(ValueError, match="only simulation"):
        create_app()
