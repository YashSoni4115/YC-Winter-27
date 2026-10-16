from types import SimpleNamespace

import numpy as np
import pytest

from energy_decision_layer.contracts import SafetyDecisionStatus, SafetyReasonCode, SafetyResponseV1
from energy_decision_layer.optimizer import MilpPolicy, RulePolicy
from energy_decision_layer.optimizer.dispatcher import Dispatcher, DispatchStopped
from energy_decision_layer.sim import SiteConfig, SiteInputs, SiteSimulator
from energy_decision_layer.testing.doubles import (
    DevelopmentAcceptAllShield,
    InMemoryDecisionLogger,
    PersistenceForecastDouble,
)


def test_milp_meets_known_demand_and_physics():
    sim = SiteSimulator([SiteInputs(nondeferrable_compute_units=40)] * 12)
    o, _ = sim.reset()
    policy = MilpPolicy(sim.config)
    logger = InMemoryDecisionLogger()
    dispatcher = Dispatcher(
        policy, DevelopmentAcceptAllShield(), logger, RulePolicy(), PersistenceForecastDouble()
    )
    while not sim.done:
        o, _, _, _, info = dispatcher.dispatch(o, sim)
        assert policy.last_result.status == "optimal"
        assert info["deadline_missed_compute_units"] < 1e-5
        assert abs(info["power_balance_residual_mw"]) < 1e-8
        # Predicted applied cooling equals physical lagged/ramped output.
        assert all(v.code != "load_shed" for v in info["violations"])
    assert len(logger.records) == 12


def test_infeasibility_and_invalid_forecast_are_explicit():
    sim = SiteSimulator([SiteInputs(grid_available=False, generator_available=False)])
    o, _ = sim.reset()
    policy = MilpPolicy()
    assert policy.plan(o).status == "infeasible"
    forecast = PersistenceForecastDouble().forecast(o, horizon_steps=2, cadence_seconds=600.0)
    assert policy.plan(o, forecast).status == "invalid_forecast"
    assert policy.plan(o, forecast).action is None


@pytest.mark.parametrize(
    "status,expected", [(1, "timeout"), (2, "infeasible"), (3, "unbounded"), (4, "solver_error")]
)
def test_solver_statuses(monkeypatch, status, expected):
    monkeypatch.setattr(
        "energy_decision_layer.optimizer.planner.milp",
        lambda *a, **kw: SimpleNamespace(status=status, x=None, message="injected"),
    )
    o, _ = SiteSimulator([SiteInputs()]).reset()
    assert MilpPolicy().plan(o).status == expected


def test_invalid_incumbent_rejected(monkeypatch):
    monkeypatch.setattr(
        "energy_decision_layer.optimizer.planner.milp",
        lambda *a, **kw: SimpleNamespace(
            status=0, x=np.full(14, np.nan), message="injected", fun=0
        ),
    )
    o, _ = SiteSimulator([SiteInputs()]).reset()
    assert MilpPolicy().plan(o).status == "invalid_solution"


class BrokenPolicy:
    def propose_action(self, *args):
        raise RuntimeError("deliberate failure")


class BrokenShield:
    def evaluate(self, *args):
        raise ConnectionError("disconnected")


class RejectShield:
    def evaluate(self, proposal, observation):
        from uuid import uuid4

        return SafetyResponseV1(
            response_id=uuid4(),
            action_id=proposal.action_id,
            site_id=observation.site_id,
            decided_at_utc=observation.timestamp_utc,
            shield_name="test-reject",
            shield_version="1",
            status=SafetyDecisionStatus.REJECTED,
            reason_codes=(SafetyReasonCode.INVALID_ACTION,),
        )


def test_fallback_passes_through_shield_and_logs():
    sim = SiteSimulator([SiteInputs()])
    o, _ = sim.reset()
    logger = InMemoryDecisionLogger()
    d = Dispatcher(BrokenPolicy(), DevelopmentAcceptAllShield(), logger, RulePolicy())
    d.dispatch(o, sim)
    assert logger.records[0][0].is_fallback
    assert sim.index == 1


@pytest.mark.parametrize("shield", [BrokenShield(), RejectShield()])
def test_shield_failure_never_advances(shield):
    sim = SiteSimulator([SiteInputs()])
    o, _ = sim.reset()
    d = Dispatcher(RulePolicy(), shield, InMemoryDecisionLogger(), RulePolicy())
    with pytest.raises(DispatchStopped):
        d.dispatch(o, sim)
    assert sim.index == 0


def test_minimum_up_time_does_not_bypass_down_ramp():
    from energy_decision_layer.sim.generator import GeneratorConfig, GeneratorSystem

    m = GeneratorSystem(
        GeneratorConfig(
            initial_on=True, initial_power_mw=8, initial_elapsed_steps=0, minimum_up_steps=3
        )
    )
    r = m.step(m.initial_state(), 0, 1 / 12, available=True)
    assert r.actual_power_mw == pytest.approx(8 - 8 / 12)
    assert r.state.is_on


def test_milp_supports_extended_commitment_history():
    from energy_decision_layer.sim.generator import GeneratorConfig

    c = SiteConfig(
        generator=GeneratorConfig(
            initial_on=True,
            initial_power_mw=3,
            initial_elapsed_steps=0,
            minimum_up_steps=4,
            minimum_down_steps=3,
        )
    )
    sim = SiteSimulator([SiteInputs(nondeferrable_compute_units=20)] * 8, c)
    o, _ = sim.reset()
    forecast = PersistenceForecastDouble().forecast(o, horizon_steps=8, cadence_seconds=300.0)
    result = MilpPolicy(c).plan(o, forecast)
    assert result.status == "optimal"
    assert result.action.generator_power_mw >= c.generator.minimum_power_mw
    assert (
        abs(result.action.generator_power_mw - o.generator_power_mw)
        <= c.generator.ramp_down_rate_mw_per_hour / 12 + 1e-8
    )


def test_deferrable_work_shifts_to_cheaper_interval():
    from energy_decision_layer.testing.doubles import OfflineHindsightForecastDouble

    inputs = (
        SiteInputs(
            nondeferrable_compute_units=0,
            deferrable_compute_units=30,
            deferrable_deadline_steps=2,
            grid_import_price_per_mwh=500,
        ),
        SiteInputs(nondeferrable_compute_units=0, grid_import_price_per_mwh=10),
    )
    sim = SiteSimulator(inputs)
    o, _ = sim.reset()
    oracle = OfflineHindsightForecastDouble(inputs, o.timestamp_utc)
    policy = MilpPolicy(allow_offline_hindsight=True)
    d = Dispatcher(
        policy,
        DevelopmentAcceptAllShield(),
        InMemoryDecisionLogger(),
        RulePolicy(),
        oracle,
        horizon_steps=2,
    )
    o, _, _, _, first = d.dispatch(o, sim)
    assert first["completed_compute_units"] < 1e-5
    assert o.deferrable_backlog_compute_units == pytest.approx(30)
    d.horizon_steps = 1
    _, _, _, _, second = d.dispatch(o, sim)
    assert second["completed_compute_units"] == pytest.approx(30)
    assert second["deadline_missed_compute_units"] == pytest.approx(0)


def test_single_interval_matches_independent_analytic_lower_bound():
    # 40 units / (100 units per MW-hour * 1/12 hour) + 1 MW idle = 5.8 MW.
    # Terminal reserve forbids net battery discharge. Generation is dearer than grid.
    # Thermal headroom permits zero cooling here, hence minimum grid is 6 MW.
    sim = SiteSimulator([SiteInputs(nondeferrable_compute_units=40)])
    o, _ = sim.reset()
    result = MilpPolicy().plan(o)
    assert result.status == "optimal"
    assert result.action.compute_power_mw == pytest.approx(5.8)
    assert result.action.grid_power_mw == pytest.approx(6.0)
    assert result.objective == pytest.approx(6 * 75 / 12)


def test_shield_modified_action_is_the_only_applied_action():
    class ModifyShield:
        def evaluate(self, proposal, observation):
            response = DevelopmentAcceptAllShield().evaluate(proposal, observation)
            payload = response.model_dump(mode="json")
            payload["status"] = "modified"
            payload["reason_codes"] = ["compute_capacity_limit"]
            payload["executed_action"]["authorization_status"] = "modified"
            payload["executed_action"]["compute_power_mw"] = 0.0
            payload["executed_action"]["modified_fields"] = ["compute_power_mw"]
            return SafetyResponseV1.model_validate_json(__import__("json").dumps(payload))

    sim = SiteSimulator([SiteInputs(nondeferrable_compute_units=40)])
    o, _ = sim.reset()
    logger = InMemoryDecisionLogger()
    _, _, _, _, info = Dispatcher(RulePolicy(), ModifyShield(), logger, RulePolicy()).dispatch(
        o, sim
    )
    assert info["applied_action"]["compute_power_mw"] == 0
    assert info["shield_intervention"]
    assert info["deadline_missed_compute_units"] == 40


def test_forged_shield_source_and_logger_failures_stop_before_execution():
    class SubstituteShield:
        def evaluate(self, proposal, observation):
            different = proposal.model_copy(update={"policy_name": "substituted"})
            return DevelopmentAcceptAllShield().evaluate(different, observation)

    class BrokenLogger:
        def record_decision(self, *args):
            raise OSError("audit storage unavailable")

    for shield, logger in (
        (SubstituteShield(), InMemoryDecisionLogger()),
        (DevelopmentAcceptAllShield(), BrokenLogger()),
    ):
        sim = SiteSimulator([SiteInputs()])
        o, _ = sim.reset()
        with pytest.raises(DispatchStopped):
            Dispatcher(RulePolicy(), shield, logger, RulePolicy()).dispatch(o, sim)
        assert sim.index == 0


def test_prediction_intervals_change_reserved_compute_capacity():
    from energy_decision_layer.contracts import (
        ForecastBundleV1,
        ForecastPointV1,
        ForecastSignal,
        PredictionIntervalV1,
    )

    sim = SiteSimulator([SiteInputs(nondeferrable_compute_units=20)])
    o, _ = sim.reset()
    forecast = PersistenceForecastDouble().forecast(o, horizon_steps=1, cadence_seconds=300.0)
    point_payload = forecast.points[0].model_dump()
    point_payload["prediction_intervals"] = (
        PredictionIntervalV1(
            signal=ForecastSignal.WORKLOAD_DEMAND,
            lower_value=120.0,
            upper_value=480.0,
            coverage_probability=0.9,
        ),
    )
    payload = forecast.model_dump()
    payload["points"] = (ForecastPointV1.model_validate(point_payload),)
    forecast = ForecastBundleV1.model_validate(payload)
    nominal = MilpPolicy(use_prediction_intervals=False).plan(o, forecast)
    conservative = MilpPolicy().plan(o, forecast)
    assert nominal.status == conservative.status == "optimal"
    assert conservative.action.compute_power_mw > nominal.action.compute_power_mw


def test_terminal_reserve_does_not_ratchet_after_charging():
    from energy_decision_layer.contracts import SiteObservationV1

    sim = SiteSimulator([SiteInputs(nondeferrable_compute_units=40, grid_import_price_per_mwh=500)])
    o, _ = sim.reset()
    payload = o.model_dump()
    payload.update(battery_energy_mwh=11.0, battery_state_of_charge=0.55)
    charged = SiteObservationV1.model_validate(payload)
    result = MilpPolicy().plan(charged)
    assert result.status == "optimal"
    assert result.action.battery_power_mw > 1.0


def test_executed_outcome_is_linked_and_post_execution_audit_error_is_explicit():
    from energy_decision_layer.optimizer.dispatcher import DispatchOutcomeLoggingFailed

    logger = InMemoryDecisionLogger()
    sim = SiteSimulator([SiteInputs()])
    o, _ = sim.reset()
    Dispatcher(RulePolicy(), DevelopmentAcceptAllShield(), logger, RulePolicy()).dispatch(o, sim)
    assert logger.outcomes[0]["action_id"] == str(logger.records[0][0].action_id)
    assert "applied_action" in logger.outcomes[0]["diagnostics"]

    class BrokenOutcomeLogger(InMemoryDecisionLogger):
        def record_outcome(self, *args):
            raise OSError("disk unavailable")

    sim.reset()
    with pytest.raises(DispatchOutcomeLoggingFailed, match="action applied"):
        Dispatcher(
            RulePolicy(), DevelopmentAcceptAllShield(), BrokenOutcomeLogger(), RulePolicy()
        ).dispatch(sim.observation(), sim)
    assert sim.index == 1


def test_current_outage_overrides_optimistic_forecast_and_connection_rating():
    sim = SiteSimulator([SiteInputs(grid_available=False, generator_available=False)])
    o, _ = sim.reset()
    # A disconnected connection can still have a nonzero contractual nameplate rating.
    o = o.model_copy(update={"grid_import_limit_mw": 15.0})
    f = PersistenceForecastDouble().forecast(o, horizon_steps=1, cadence_seconds=300.0)
    point = f.points[0].model_copy(update={"grid_available": True, "grid_import_limit_mw": 15.0})
    f = f.model_copy(update={"points": (point,)})
    assert MilpPolicy().plan(o, f).status == "infeasible"


def test_critical_backlog_is_not_ignored_by_baselines():
    sim = SiteSimulator([SiteInputs()])
    o, _ = sim.reset()
    o = o.model_copy(update={"critical_backlog_compute_units": 40.0})
    assert RulePolicy().propose_action(o).compute_power_mw == pytest.approx(5.8)
    result = MilpPolicy().plan(o)
    assert result.status == "optimal"
    assert result.action.compute_power_mw == pytest.approx(5.8)


def test_wrong_solver_solution_shape_returns_explicit_failure(monkeypatch):
    monkeypatch.setattr(
        "energy_decision_layer.optimizer.planner.milp",
        lambda *args, **kwargs: SimpleNamespace(status=0, x=np.zeros(2), message="bad shape"),
    )
    o, _ = SiteSimulator([SiteInputs()]).reset()
    assert MilpPolicy().plan(o).status == "invalid_solution"


def test_dispatcher_validates_forecast_even_if_policy_ignores_it():
    class WrongSiteForecast(PersistenceForecastDouble):
        def forecast(self, *args, **kwargs):
            return super().forecast(*args, **kwargs).model_copy(update={"site_id": "wrong-site"})

    logger = InMemoryDecisionLogger()
    sim = SiteSimulator([SiteInputs()])
    o, _ = sim.reset()
    Dispatcher(
        RulePolicy(), DevelopmentAcceptAllShield(), logger, RulePolicy(), WrongSiteForecast()
    ).dispatch(o, sim)
    assert logger.records[0][0].is_fallback
    assert "forecast does not match" in logger.records[0][0].rationale
