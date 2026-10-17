"""One integrated, simulation-only decision loop with durable audit checkpoints."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from energy_decision_layer.adapters import AdapterError, MockSiteAdapter
from energy_decision_layer.contracts import (
    ForecastBundleV1,
    ForecastProviderProtocol,
    PolicyProtocol,
    ProposedControlActionV1,
    SafetyResponseV1,
    SiteObservationV1,
)
from energy_decision_layer.optimizer import MilpPolicy, RulePolicy
from energy_decision_layer.optimizer.dispatcher import (
    Dispatcher,
    DispatchOutcomeLoggingFailed,
    DispatchStopped,
)
from energy_decision_layer.safety import SafetyShield
from energy_decision_layer.sim import SiteConfig, SiteInputs, SiteSimulator
from energy_decision_layer.sim.battery import BatteryState
from energy_decision_layer.sim.compute import ComputeState, WorkloadBatch
from energy_decision_layer.sim.generator import GeneratorState
from energy_decision_layer.sim.thermal import ThermalState

from .store import DecisionStore, encode


def demo_simulator(steps: int = 288, seed: int = 42) -> SiteSimulator:
    """Explicit synthetic varying-price, warm-weather, constrained-grid scenario."""
    rng = np.random.default_rng(seed)
    inputs = []
    for index in range(steps):
        cycle = np.sin(index * 2 * np.pi / 72)
        inputs.append(
            SiteInputs(
                nondeferrable_compute_units=float(25 + 8 * cycle + rng.uniform(0, 3)),
                deferrable_compute_units=8.0,
                deferrable_deadline_steps=6,
                ambient_temperature_c=float(27 + 9 * (index / max(1, steps - 1))),
                grid_import_price_per_mwh=float(70 + 55 * cycle),
                grid_import_limit_mw=4.0 if index % 72 in range(35, 45) else 15.0,
                grid_stress=index % 72 in range(35, 45),
            )
        )
    return SiteSimulator(inputs, SiteConfig())


class _ProposalPolicy:
    def __init__(self, action: ProposedControlActionV1) -> None:
        self.action = action

    def propose_action(
        self, observation: SiteObservationV1, forecast: ForecastBundleV1 | None = None
    ) -> ProposedControlActionV1:
        return self.action


class _AuditLogger:
    def __init__(
        self,
        runtime: SimulationRuntime,
        key: str,
        forecast: ForecastBundleV1 | None,
        forecast_latency_seconds: float = 0.0,
    ) -> None:
        self.runtime, self.key, self.forecast = runtime, key, forecast
        self.proposal: ProposedControlActionV1 | None = None
        self.response: SafetyResponseV1 | None = None
        self.forecast_latency_seconds = forecast_latency_seconds

    def record_decision(
        self, proposal: ProposedControlActionV1, response: SafetyResponseV1
    ) -> None:
        self.proposal, self.response = proposal, response
        store = self.runtime.store
        with store.connection:
            store.connection.execute(
                "INSERT INTO decisions(command_key,action_id,timestamp_utc,policy,state_json,"
                "forecast_json,proposal_json,response_json,status) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    self.key,
                    str(proposal.action_id),
                    proposal.created_at_utc.isoformat(),
                    proposal.policy_name,
                    encode(self.runtime.current),
                    encode(self.forecast) if self.forecast else None,
                    encode(proposal),
                    encode(response),
                    "authorized" if response.executed_action else "rejected",
                ),
            )

    def record_outcome(
        self, proposal: ProposedControlActionV1, observation: SiteObservationV1, diagnostics: Any
    ) -> None:
        runtime, store = self.runtime, self.runtime.store
        runtime.current = observation
        outcome = json.loads(encode(diagnostics))
        outcome["forecast_latency_seconds"] = self.forecast_latency_seconds
        result = {
            "executed": True,
            "proposal": proposal.model_dump(mode="json"),
            "safety": self.response.model_dump(mode="json") if self.response else None,
            "state": observation.model_dump(mode="json"),
            "outcome": outcome,
        }
        with store.connection:
            store.connection.execute(
                "UPDATE decisions SET outcome_json=?,status='executed' "
                "WHERE action_id=? AND command_key=?",
                (encode(outcome), str(proposal.action_id), self.key),
            )
            store.add_telemetry(observation)
            store.set_metadata("checkpoint", runtime._checkpoint())
            store.finish(self.key, result)


class SimulationRuntime:
    """All dispatch paths share Dispatcher → SafetyShield → MockSiteAdapter."""

    simulation_only = True

    def __init__(
        self,
        simulator: SiteSimulator | None = None,
        db_path: str | Path = "artifacts/decisions.sqlite3",
        forecaster: ForecastProviderProtocol | None = None,
        learned_policy: PolicyProtocol | None = None,
        surrogate: Any = None,
        operator_token: str | None = None,
        dispatcher_token: str | None = None,
    ) -> None:
        if operator_token and operator_token == dispatcher_token:
            raise ValueError("operator and dispatcher tokens must be distinct")
        self.operator_token, self.dispatcher_token = operator_token, dispatcher_token
        self.simulator = simulator or demo_simulator()
        self.shield = SafetyShield(self.simulator.config)
        self.adapter = MockSiteAdapter(self.simulator, self.shield)
        if forecaster is None:
            from energy_decision_layer.ml import ForecastProvider

            forecaster = ForecastProvider()
        self.forecaster = forecaster
        self.surrogate = surrogate
        self.policies: dict[str, PolicyProtocol] = {
            "rule": RulePolicy(self.simulator.config),
            "baseline": MilpPolicy(self.simulator.config),
        }
        if learned_policy is not None:
            self.policies["learned"] = learned_policy
        self.selected_policy = "baseline"
        self.lock = threading.RLock()
        self.store = DecisionStore(db_path)
        identity = json.loads(
            encode(
                {
                    "config": asdict(self.simulator.config),
                    "inputs": [asdict(x) for x in self.simulator.inputs],
                    "start": self.simulator.start,
                }
            )
        )
        previous = self.store.metadata("scenario")
        if previous is not None and identity != previous:
            self.store.close()
            raise ValueError("database scenario differs; choose a new database path")
        if previous is None:
            with self.store.connection:
                self.store.set_metadata("scenario", identity)
        checkpoint = self.store.metadata("checkpoint")
        if checkpoint:
            self._restore(checkpoint)
        self.current = self.simulator.observation()
        with self.store.connection:
            self.store.add_telemetry(self.current)
            self.store.set_metadata("checkpoint", self._checkpoint())

    def close(self) -> None:
        self.store.close()

    def _checkpoint(self) -> dict[str, Any]:
        sim = self.simulator
        return {
            "episode_id": str(sim.episode_id),
            "index": sim.index,
            "done": sim.done,
            "battery": asdict(sim.bs),
            "compute": asdict(sim.cs),
            "generator": asdict(sim.gs),
            "thermal": asdict(sim.ts),
            "last_grid": sim.last_grid,
            "last_auxiliary": sim.last_auxiliary,
            "last_unserved": sim.last_unserved,
        }

    def _restore(self, checkpoint: dict[str, Any]) -> None:
        from uuid import UUID

        sim = self.simulator
        sim.episode_id = UUID(checkpoint["episode_id"])
        sim.index, sim.done = checkpoint["index"], checkpoint["done"]
        sim.bs = BatteryState(**checkpoint["battery"])
        compute = checkpoint["compute"]
        compute["deferrable_batches"] = tuple(
            WorkloadBatch(**item) for item in compute["deferrable_batches"]
        )
        sim.cs = ComputeState(**compute)
        sim.gs = GeneratorState(**checkpoint["generator"])
        sim.ts = ThermalState(**checkpoint["thermal"])
        sim.last_grid = checkpoint["last_grid"]
        sim.last_auxiliary = checkpoint["last_auxiliary"]
        sim.last_unserved = checkpoint["last_unserved"]

    def forecasts(self) -> ForecastBundleV1:
        with self.lock:
            return self.forecaster.forecast(
                self.current,
                horizon_steps=12,
                cadence_seconds=self.current.timestep_seconds,
            )

    def recommend(self, policy: str) -> dict[str, Any]:
        with self.lock:
            if policy not in self.policies:
                raise ValueError(f"policy {policy!r} unavailable")
            forecast = self.forecasts()
            proposed = self.policies[policy].propose_action(self.current, forecast)
            advisory = None
            if self.surrogate is not None:
                try:
                    advisory = self.surrogate.predict(self.current, proposed)
                except Exception:
                    advisory = {"advisory_only": True, "status": "inference_unavailable"}
            return {
                "proposal": proposed,
                "safety": self.shield.evaluate(proposed, self.current),
                "forecast": forecast,
                "executed": False,
                "surrogate": advisory,
            }

    def dispatch(
        self,
        policy: str = "baseline",
        *,
        proposal: ProposedControlActionV1 | None = None,
        idempotency_key: str,
    ) -> dict[str, Any]:
        with self.lock:
            request = {"policy": policy, "proposal": proposal}
            digest = hashlib.sha256(encode(request).encode()).hexdigest()
            existing = self.store.connection.execute(
                "SELECT 1 FROM commands WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if self.store.unresolved() and existing is None:
                raise RuntimeError("unresolved prior dispatch; inspect audit before recovery")
            if self.simulator.done and existing is None:
                raise ValueError("simulation scenario finished; start a new database/scenario")
            if policy not in self.policies and proposal is None:
                raise ValueError(f"policy {policy!r} unavailable")
            cached = self.store.reserve(idempotency_key, digest)
            if cached is not None:
                return cached
            forecast_started = perf_counter()
            try:
                forecast = self.forecasts()
            except Exception:
                forecast = None
            logger = _AuditLogger(
                self, idempotency_key, forecast, perf_counter() - forecast_started
            )
            if proposal is not None:
                # Manual requests retain their exact action even when forecasts fail.
                # Audit stale/invalid proposals before Dispatcher's clock preflight.
                preflight = self.shield.evaluate(proposal, self.current)
                if preflight.executed_action is None:
                    logger.record_decision(proposal, preflight)
                    rejected = {
                        "executed": False,
                        "proposal": proposal.model_dump(mode="json"),
                        "safety": preflight.model_dump(mode="json"),
                        "state": self.current.model_dump(mode="json"),
                        "outcome": None,
                        "error": "shield rejected manual proposal; no action applied",
                    }
                    with self.store.connection:
                        self.store.finish(idempotency_key, rejected)
                    return rejected

            class CapturedForecast:
                def forecast(
                    self,
                    observation: SiteObservationV1,
                    *,
                    horizon_steps: int,
                    cadence_seconds: float,
                ) -> ForecastBundleV1:
                    if forecast is None:
                        raise ValueError("forecast inference failed")
                    return forecast

            chosen = _ProposalPolicy(proposal) if proposal else self.policies[policy]
            dispatcher = Dispatcher(
                chosen,
                self.shield,
                logger,
                self.policies["rule"],
                None if proposal is not None else CapturedForecast(),
            )
            try:
                dispatcher.dispatch(self.current, self.adapter)
            except DispatchOutcomeLoggingFailed:
                # Pending is intentionally retained: retrying might duplicate an applied action.
                self.current = self.simulator.observation()
                raise
            except (DispatchStopped, AdapterError, ValueError) as exc:
                result = {
                    "executed": False,
                    "error": str(exc),
                    "proposal": logger.proposal.model_dump(mode="json")
                    if logger.proposal
                    else (proposal.model_dump(mode="json") if proposal else None),
                    "safety": logger.response.model_dump(mode="json") if logger.response else None,
                    "state": self.current.model_dump(mode="json"),
                    "outcome": None,
                }
                with self.store.connection:
                    self.store.connection.execute(
                        "UPDATE decisions SET status='execution_failed' "
                        "WHERE command_key=? AND status='authorized'",
                        (idempotency_key,),
                    )
                    self.store.finish(idempotency_key, result)
                return result
            row = self.store.connection.execute(
                "SELECT result_json FROM commands WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            return json.loads(row[0])  # type: ignore[no-any-return]

    def metrics(self) -> dict[str, Any]:
        rows = self.store.connection.execute(
            "SELECT policy,status,response_json,outcome_json FROM decisions"
        ).fetchall()
        groups: dict[str, dict[str, Any]] = {}
        for row in rows:
            group = groups.setdefault(
                row["policy"],
                {
                    "decisions": 0,
                    "executed_steps": 0,
                    "rejected": 0,
                    "modified": 0,
                    "operating_cost_usd": 0.0,
                    "completed_compute_units": 0.0,
                    "deadline_missed_compute_units": 0.0,
                    "battery_throughput_mwh": 0.0,
                    "generator_energy_mwh": 0.0,
                    "thermal_violations": 0,
                    "constraint_violations": 0,
                    "inference_latency_seconds": 0.0,
                },
            )
            group["decisions"] += 1
            response = json.loads(row["response_json"])
            if response["status"] in {"modified", "rejected"}:
                group[response["status"]] += 1
            if not row["outcome_json"]:
                continue
            outcome = json.loads(row["outcome_json"])
            group["executed_steps"] += 1
            for key in (
                "completed_compute_units",
                "deadline_missed_compute_units",
                "battery_throughput_mwh",
                "generator_energy_mwh",
            ):
                group[key] += outcome.get(key, 0.0)
            group["operating_cost_usd"] += outcome.get("operating_cost", 0.0)
            group["thermal_violations"] += int(outcome.get("thermal_violation", False))
            group["constraint_violations"] += sum(
                event.get("severity") != "info" and event.get("code") != "deadline_missed"
                for event in outcome.get("violations", [])
            )
            group["inference_latency_seconds"] += outcome.get(
                "policy_latency_seconds", 0.0
            ) + outcome.get("forecast_latency_seconds", 0.0)
        for group in groups.values():
            completed = group["completed_compute_units"]
            group["cost_per_completed_compute_usd"] = (
                group["operating_cost_usd"] / completed if completed else None
            )
        return {
            "policies": groups,
            "currency": "USD",
            "simulation_only": True,
            "comparison_note": (
                "Operational history; use seeded offline evaluation for fair comparisons"
            ),
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
        }
