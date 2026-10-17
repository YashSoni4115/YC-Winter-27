"""Typed local HTTP API. Read-only by default; all execution is simulated."""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from energy_decision_layer.contracts import (
    ForecastBundleV1,
    ProposedControlActionV1,
    SafetyResponseV1,
    SiteObservationV1,
)

from .runtime import SimulationRuntime


class APIAction(ProposedControlActionV1):
    """HTTP JSON coercion, followed by strict canonical contract revalidation."""

    model_config = ConfigDict(extra="forbid", strict=False, frozen=True, allow_inf_nan=False)

    def canonical(self) -> ProposedControlActionV1:
        return ProposedControlActionV1.model_validate_json(self.model_dump_json())


class DispatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy: Literal["rule", "baseline", "learned"] = "baseline"


class DispatchResult(BaseModel):
    executed: bool
    proposal: dict[str, Any] | None = None
    safety: dict[str, Any] | None = None
    state: SiteObservationV1
    outcome: dict[str, Any] | None = None
    error: str | None = None


class HistoryResponse(BaseModel):
    items: list[dict[str, Any]]
    simulation_only: bool = True


class RecommendationResponse(BaseModel):
    proposal: ProposedControlActionV1
    safety: SafetyResponseV1
    forecast: ForecastBundleV1
    executed: bool = False
    surrogate: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    simulation_only: bool = True
    ready: bool
    reason: str | None = None


def create_app(runtime: SimulationRuntime | None = None) -> FastAPI:
    if os.environ.get("EDL_CONTROL_MODE", "simulation") != "simulation":
        raise ValueError("only simulation control mode is supported")
    active = runtime or SimulationRuntime(
        db_path=os.environ.get("EDL_DB_PATH", "artifacts/decisions.sqlite3"),
        dispatcher_token=os.environ.get("EDL_DISPATCHER_TOKEN"),
        operator_token=os.environ.get("EDL_OPERATOR_TOKEN"),
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if runtime is None:
                active.close()

    app = FastAPI(title="Energy Decision Layer — simulation", version="0.1.0", lifespan=lifespan)
    app.state.runtime = active

    def role(authorization: str | None = Header(default=None)) -> str:
        supplied = authorization.removeprefix("Bearer ") if authorization else ""
        if supplied and authorization and authorization.startswith("Bearer "):
            if active.operator_token and secrets.compare_digest(supplied, active.operator_token):
                return "operator"
            if active.dispatcher_token and secrets.compare_digest(
                supplied, active.dispatcher_token
            ):
                return "dispatcher"
        return "observer"

    def dispatch_role(current_role: str = Depends(role)) -> str:
        if current_role == "observer":
            raise HTTPException(401, "Dispatch requires a configured bearer token")
        return current_role

    def operator_role(current_role: str = Depends(role)) -> str:
        if current_role != "operator":
            raise HTTPException(403, "Manual override requires the operator bearer token")
        return current_role

    def execute(
        policy: str, key: str, proposal: ProposedControlActionV1 | None = None
    ) -> dict[str, Any]:
        try:
            result = active.dispatch(policy, proposal=proposal, idempotency_key=key)
            return {
                **result,
                "state": SiteObservationV1.model_validate_json(json.dumps(result["state"])),
            }
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", ready=not active.store.unresolved())

    @app.get("/ready", response_model=HealthResponse)
    def ready() -> HealthResponse:
        if active.store.unresolved():
            raise HTTPException(503, "Unresolved prior dispatch requires audit recovery")
        active.store.connection.execute("SELECT 1").fetchone()
        return HealthResponse(status="ok", ready=True)

    @app.get("/api/state", response_model=SiteObservationV1)
    def state() -> SiteObservationV1:
        with active.lock:
            return active.current

    @app.get("/api/telemetry", response_model=HistoryResponse)
    def telemetry(limit: int = Query(default=100, ge=1, le=1000)) -> HistoryResponse:
        with active.lock:
            return HistoryResponse(items=active.store.history("telemetry", limit))

    @app.get("/api/forecasts", response_model=ForecastBundleV1)
    def forecasts() -> ForecastBundleV1:
        try:
            return active.forecasts()
        except Exception as exc:
            raise HTTPException(503, "Forecast inference unavailable") from exc

    @app.get("/api/recommendations/{policy}", response_model=RecommendationResponse)
    def recommendation(policy: str) -> dict[str, Any]:
        if policy not in active.policies:
            raise HTTPException(404, "Policy checkpoint unavailable")
        try:
            return active.recommend(policy)
        except Exception as exc:
            raise HTTPException(503, "Policy inference unavailable") from exc

    @app.post("/api/safety/validate", response_model=SafetyResponseV1)
    def validate(action: APIAction) -> SafetyResponseV1:
        with active.lock:
            return active.shield.evaluate(action.canonical(), active.current)

    @app.post("/api/dispatch", response_model=DispatchResult)
    def dispatch(
        request: DispatchRequest,
        key: str = Header(alias="Idempotency-Key", min_length=1, max_length=128),
        current_role: str = Depends(dispatch_role),
    ) -> dict[str, Any]:
        return execute(request.policy, key)

    @app.post("/api/actions", response_model=DispatchResult)
    @app.post("/api/manual-overrides", response_model=DispatchResult)
    def manual(
        action: APIAction,
        key: str = Header(alias="Idempotency-Key", min_length=1, max_length=128),
        current_role: str = Depends(operator_role),
    ) -> dict[str, Any]:
        return execute("manual", key, action.canonical())

    @app.get("/api/decisions", response_model=HistoryResponse)
    def decisions(limit: int = Query(default=100, ge=1, le=1000)) -> HistoryResponse:
        with active.lock:
            return HistoryResponse(items=active.store.history("decisions", limit))

    @app.get("/api/metrics")
    def metrics() -> dict[str, Any]:
        with active.lock:
            return active.metrics()

    @app.get("/api/forecast-status")
    def forecast_status() -> dict[str, Any]:
        return {
            "status": getattr(active.forecaster, "last_status", "unavailable"),
            "grid_stress_probabilities": getattr(
                active.forecaster, "last_stress_probabilities", []
            ),
        }

    @app.get("/api/evaluation")
    def evaluation() -> dict[str, Any]:
        directory = Path(os.environ.get("EDL_EVALUATION_DIR", "artifacts/ml-evaluation"))
        reports = {}
        for name in ("metrics.json", "ml-evaluation.json"):
            path = directory / name
            if path.is_file():
                try:
                    reports[name] = json.loads(path.read_text())
                except (ValueError, OSError) as exc:
                    raise HTTPException(503, "Evaluation report is unreadable") from exc
        return reports or {
            "status": "not_run",
            "message": "Run the documented ML evaluation command",
        }

    @app.get("/api/policies")
    def policies(current_role: str = Depends(role)) -> dict[str, Any]:
        return {
            "available": list(active.policies),
            "selected": active.selected_policy,
            "role": current_role,
            "simulation_only": True,
            "can_dispatch": current_role != "observer" and not active.simulator.done,
            "can_override": current_role == "operator" and not active.simulator.done,
            "scenario_finished": active.simulator.done,
            "units": {
                "power": "MW",
                "energy": "MWh",
                "price": "USD/MWh",
                "temperature": "°C",
                "timestamps": "UTC",
            },
        }

    @app.post("/api/policies/select")
    def select(
        request: DispatchRequest, current_role: str = Depends(dispatch_role)
    ) -> dict[str, Any]:
        if request.policy not in active.policies:
            raise HTTPException(404, "Policy checkpoint unavailable")
        active.selected_policy = request.policy
        return {"selected": request.policy, "executed": False}

    static = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(static / "index.html")

    return app
