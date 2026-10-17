"""Reproducible local training, serving, and full-system demonstration workflows."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from energy_decision_layer.data.models import SimulationIntervalV1
from energy_decision_layer.data.pipeline import to_site_inputs
from energy_decision_layer.sim import SiteConfig, SiteInputs, SiteSimulator


def read_intervals(path: Path) -> list[SimulationIntervalV1]:
    rows = [
        SimulationIntervalV1.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("normalized interval dataset is empty")
    return rows


def generate_dataset(
    output: Path, *, steps: int = 2016, seed: int = 7
) -> list[SimulationIntervalV1]:
    from energy_decision_layer.ml import synthetic_intervals

    rows = list(synthetic_intervals(steps=steps, seed=seed))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(row.model_dump_json() for row in rows) + "\n")
    return rows


def train_components(
    command: str,
    output: Path,
    *,
    data: Path | None = None,
    steps: int = 2016,
    seed: int = 7,
    config: Path | None = None,
) -> dict[str, Any]:
    from energy_decision_layer.ml import (
        synthetic_intervals,
        train_forecast,
        train_policy,
        train_surrogates,
    )

    settings = json.loads(config.read_text()) if config is not None else {}
    unknown = set(settings) - {"forecast", "policy", "surrogates"}
    if unknown:
        raise ValueError(f"unknown training configuration keys: {sorted(unknown)}")
    rows = (
        read_intervals(data)
        if data is not None
        else list(synthetic_intervals(steps=steps, seed=seed))
    )
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if command in {"train-all", "train-forecast"}:
        results["forecast"] = train_forecast(
            rows, output / "forecast", seed=seed, **settings.get("forecast", {})
        )
    if command in {"train-all", "train-policy", "train-surrogates"}:
        intervals = to_site_inputs(rows)
        c = SiteConfig(site_id=rows[0].site_id, timestep_seconds=rows[0].duration_hours * 3600)
        if command in {"train-all", "train-policy"}:
            results["policy"] = train_policy(
                intervals[:288],
                output / "policy.json",
                config=c,
                seed=seed,
                **settings.get("policy", {}),
            )
        if command in {"train-all", "train-surrogates"}:
            results["surrogates"] = train_surrogates(
                intervals[:576],
                output / "surrogates",
                config=c,
                seed=seed,
                **settings.get("surrogates", {}),
            )
    (output / f"{command}-report.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    return results


def demo_inputs(steps: int = 24) -> tuple[SiteInputs, ...]:
    if steps < 4:
        raise ValueError("demo requires at least four steps")
    return tuple(
        SiteInputs(
            nondeferrable_compute_units=30.0 + 15.0 * (1 + math.sin(i / 3)),
            ambient_temperature_c=25.0 + 18.0 * i / max(1, steps - 1),
            grid_import_price_per_mwh=25.0 if i < steps // 2 else 250.0,
            grid_import_limit_mw=15.0 if i < steps // 2 else 4.0,
            grid_stress=i >= steps // 2,
        )
        for i in range(steps)
    )


def create_runtime(
    models: Path, database: Path, *, steps: int = 288, data: Path | None = None
) -> Any:
    from energy_decision_layer.ml import (
        DynamicsSurrogate,
        ForecastModel,
        ForecastProvider,
        LearnedPolicy,
    )
    from energy_decision_layer.service.runtime import SimulationRuntime

    if data is not None:
        rows = read_intervals(data)
        sim = SiteSimulator(
            to_site_inputs(rows),
            SiteConfig(site_id=rows[0].site_id, timestep_seconds=rows[0].duration_hours * 3600),
            start=rows[0].interval_start_utc,
        )
    else:
        sim = SiteSimulator(demo_inputs(steps))
    model = ForecastModel.load(models / "forecast") if (models / "forecast").exists() else None
    learned = (
        LearnedPolicy.load(models / "policy.json", config=sim.config)
        if (models / "policy.json").exists()
        else None
    )
    surrogate = (
        DynamicsSurrogate.load(models / "surrogates") if (models / "surrogates").exists() else None
    )
    return SimulationRuntime(
        simulator=sim,
        db_path=str(database),
        forecaster=ForecastProvider(model),
        learned_policy=learned,
        surrogate=surrogate,
        operator_token=os.environ.get("EDL_OPERATOR_TOKEN"),
        dispatcher_token=os.environ.get("EDL_DISPATCHER_TOKEN"),
    )


def run_demo(models: Path, output: Path, *, steps: int = 24, seed: int = 7) -> dict[str, Any]:
    """Run actual service/SQLite/adapters; intentionally include an unsafe request."""
    from fastapi.testclient import TestClient

    from energy_decision_layer.optimizer.planner import proposal
    from energy_decision_layer.service.api import create_app

    if not (models / "forecast").exists() or not (models / "policy.json").exists():
        train_components("train-all", models, steps=2016, seed=seed)
    output.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(models, output / "decisions.sqlite3", steps=steps)
    # CLI demonstration acts as an in-process operator; public HTTP commands still
    # require externally configured tokens. This never enables physical control.
    executed = []
    if runtime.current.step_index != 0:
        runtime.close()
        raise ValueError("demo database already contains a run; select a new --output directory")
    try:
        with TestClient(create_app(runtime)) as client:
            preview = client.get("/api/recommendations/baseline")
            preview.raise_for_status()
            initial_recommendation = preview.json()
        charge = proposal(runtime.current, (5.5, 1.4, -2.0, 0.0, 9.1), "demo-charge")
        executed.append(
            runtime.dispatch(policy="manual", proposal=charge, idempotency_key="demo-charge")
        )
        unsafe = proposal(runtime.current, (100.0, 100.0, 100.0, 100.0, 100.0), "demo-unsafe")
        executed.append(
            runtime.dispatch(policy="manual", proposal=unsafe, idempotency_key="demo-unsafe")
        )
        while not runtime.simulator.done:
            label = "learned" if runtime.current.step_index % 2 else "baseline"
            result = runtime.dispatch(
                policy=label, idempotency_key=f"demo-step-{runtime.current.step_index}"
            )
            executed.append(result)
            if not result["executed"]:
                raise RuntimeError(f"demo dispatch stopped: {result}")
        with TestClient(create_app(runtime)) as client:
            for endpoint in ("/", "/api/state", "/api/telemetry", "/api/decisions", "/api/metrics"):
                response = client.get(endpoint)
                response.raise_for_status()
            current = client.get("/api/state").json()
            metrics = client.get("/api/metrics").json()
            decisions = client.get("/api/decisions").json()
        if not executed[1]["executed"] or executed[1]["safety"]["status"] != "modified":
            raise RuntimeError("unsafe demo proposal must be visibly modified before execution")
        report = {
            "synthetic": True,
            "seed": seed,
            "simulation_only": True,
            "steps": runtime.current.step_index,
            "state_api": current,
            "metrics_api": metrics,
            "decisions_api": decisions,
            "dispatches": executed,
            "initial_recommendation": initial_recommendation,
        }
        (output / "demo.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
    finally:
        runtime.close()
