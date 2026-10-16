import json

import pytest

from energy_decision_layer.__main__ import main
from energy_decision_layer.evaluation.policies import CallablePolicyAdapter, FixedPolicy
from energy_decision_layer.evaluation.runner import evaluate, scenarios, write_reports
from energy_decision_layer.optimizer import MilpPolicy, RulePolicy
from energy_decision_layer.sim import SiteInputs, SiteSimulator
from energy_decision_layer.testing.doubles import OfflineHindsightForecastDouble


@pytest.mark.parametrize("name", list(scenarios(4)))
@pytest.mark.parametrize("factory", [RulePolicy, MilpPolicy, FixedPolicy])
def test_all_scenarios_and_policy_adapters(name, factory):
    inputs = scenarios(4)[name]
    result = evaluate(name, inputs, factory)
    assert not result.failed
    assert result.steps == 4
    assert 0 <= result.completion_rate <= 1 + 1e-8
    assert result.cost_per_compute is None or result.compute_throughput > 0
    assert result.simulator_failures == 0


def test_comparison_reproducible_except_timings(tmp_path):
    from dataclasses import asdict

    a = evaluate("normal", scenarios(4)["normal"], RulePolicy)
    b = evaluate("normal", scenarios(4)["normal"], RulePolicy)
    da, db = asdict(a), asdict(b)
    for key in ("wall_seconds", "policy_latency_seconds", "optimization_runtime_seconds"):
        da.pop(key)
        db.pop(key)
    assert da == db
    write_reports([a, b], tmp_path)
    assert len(json.loads((tmp_path / "metrics.json").read_text())) == 2
    assert "DEVELOPMENT" in (tmp_path / "summary.md").read_text()
    write_reports([], tmp_path / "empty")


def test_learned_callable_and_offline_hindsight_contracts():
    inputs = scenarios(4)["normal"]
    o, _ = SiteSimulator(inputs).reset()
    adapter = CallablePolicyAdapter(
        lambda observation, forecast: RulePolicy().propose_action(observation, forecast)
    )
    assert adapter.propose_action(o).based_on_state_id == o.state_id
    oracle = OfflineHindsightForecastDouble(inputs, o.timestamp_utc)
    forecast = oracle.forecast(o, horizon_steps=4, cadence_seconds=300.0)
    assert forecast.is_perfect_hindsight
    assert MilpPolicy().plan(o, forecast).status == "invalid_forecast"
    assert MilpPolicy(allow_offline_hindsight=True).plan(o, forecast).status == "optimal"
    result = evaluate(
        "oracle",
        inputs,
        lambda: MilpPolicy(allow_offline_hindsight=True),
        forecaster=oracle,
        policy_label="OFFLINE-HINDSIGHT",
    )
    assert not result.failed
    assert result.policy == "OFFLINE-HINDSIGHT"


@pytest.mark.parametrize("command", ["schemas", "etl", "benchmark"])
def test_cli_entrypoints(monkeypatch, tmp_path, command):
    args = ["edl", command]
    if command == "etl":
        args += ["examples/synthetic.csv", "--synthetic"]
    if command == "benchmark":
        args += ["--steps", "2"]
    args += ["--output", str(tmp_path)]
    monkeypatch.setattr("sys.argv", args)
    main()
    assert list(tmp_path.iterdir())


def test_failed_policy_and_fallback_are_not_zero_cost_success(monkeypatch):
    from energy_decision_layer.optimizer.dispatcher import DispatchStopped

    def fail(*args):
        raise DispatchStopped("deliberate communication failure")

    monkeypatch.setattr("energy_decision_layer.evaluation.runner.Dispatcher.dispatch", fail)
    result = evaluate("failure", [SiteInputs()], RulePolicy)
    assert result.failed
    assert result.failure_reason
    assert result.steps == 0


def test_applied_step_is_scored_when_outcome_logging_fails():
    from energy_decision_layer.testing.doubles import InMemoryDecisionLogger

    class BrokenAudit(InMemoryDecisionLogger):
        def record_outcome(self, *args):
            raise OSError("audit disk unavailable")

    result = evaluate(
        "audit-failure",
        [SiteInputs(nondeferrable_compute_units=40.0)],
        RulePolicy,
        decision_logger=BrokenAudit(),
    )
    assert result.failed
    assert result.steps == 1
    assert result.compute_throughput == pytest.approx(40.0)
    assert result.total_operating_cost > 0
    assert result.simulator_failures == 0
    assert result.outcome_audit_failures == 1
    assert "action applied" in result.failure_reason


def test_falsy_injected_components_are_used_instead_of_development_doubles(tmp_path):
    from uuid import uuid4

    from energy_decision_layer.contracts import (
        SafetyDecisionStatus,
        SafetyReasonCode,
        SafetyResponseV1,
    )
    from energy_decision_layer.testing.doubles import (
        InMemoryDecisionLogger,
        PersistenceForecastDouble,
    )

    class FalsyRejectShield:
        def __bool__(self):
            return False

        def evaluate(self, proposal, observation):
            return SafetyResponseV1(
                response_id=uuid4(),
                action_id=proposal.action_id,
                site_id=observation.site_id,
                decided_at_utc=observation.timestamp_utc,
                shield_name="test-injected-reject",
                shield_version="1",
                status=SafetyDecisionStatus.REJECTED,
                reason_codes=(SafetyReasonCode.INVALID_ACTION,),
            )

    class FalsyLogger(InMemoryDecisionLogger):
        def __bool__(self):
            return False

    class FalsyForecast(PersistenceForecastDouble):
        calls = 0

        def __bool__(self):
            return False

        def forecast(self, *args, **kwargs):
            self.calls += 1
            return super().forecast(*args, **kwargs)

    forecaster, logger = FalsyForecast(), FalsyLogger()
    result = evaluate(
        "injection",
        [SiteInputs()],
        RulePolicy,
        forecaster=forecaster,
        shield=FalsyRejectShield(),
        decision_logger=logger,
    )
    assert result.failed and result.shield_rejections == 1 and result.steps == 0
    assert forecaster.calls == 1 and len(logger.records) == 1
    assert result.shield_name == "test-injected-reject"
    write_reports([result], tmp_path)
    assert (
        json.loads((tmp_path / "metrics.json").read_text())[0]["shield_name"] == result.shield_name
    )


def test_teammate_smoke_example_runs_from_checkout():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "examples/team_smoke.py", "--steps", "3"],
        capture_output=True,
        text=True,
        check=True,
    )
    metrics = json.loads(result.stdout)
    assert metrics["steps"] == 3 and not metrics["failed"]
    assert "DEVELOPMENT" in metrics["shield_name"]
