"""Reproducible local timings, without hardware-dependent CI assertions."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from time import perf_counter

from energy_decision_layer.data.models import SignalSemantics
from energy_decision_layer.data.resampling import IntervalSample, resample_intervals
from energy_decision_layer.optimizer import MilpPolicy
from energy_decision_layer.sim import SiteAction, SiteInputs, SiteSimulator
from energy_decision_layer.testing.doubles import PersistenceForecastDouble


def profile() -> dict[str, object]:
    intervals = 10_000
    sim = SiteSimulator([SiteInputs(nondeferrable_compute_units=30.0)] * intervals)
    started = perf_counter()
    for _ in range(intervals):
        sim.advance(SiteAction(5.0, 1.25, 0.0, 0.0, 6.45))
    simulation_seconds = perf_counter() - started
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    samples = [
        IntervalSample(
            origin + timedelta(minutes=i * 5), origin + timedelta(minutes=(i + 1) * 5), 10.0
        )
        for i in range(intervals)
    ]
    started = perf_counter()
    output = resample_intervals(
        samples,
        start_utc=origin,
        end_utc=origin + timedelta(minutes=intervals * 5),
        interval=timedelta(minutes=5),
        semantics=SignalSemantics.INTERVAL_TOTAL,
    )
    etl_seconds = perf_counter() - started
    assert sum(r.value for r in output) == 100_000
    o, _ = sim.reset(seed=7)
    forecast = PersistenceForecastDouble().forecast(o, horizon_steps=12, cadence_seconds=300.0)
    planner = MilpPolicy()
    runs = [planner.plan(o, forecast) for _ in range(10)]
    assert all(r.status == "optimal" for r in runs)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {p: version(p) for p in ("numpy", "scipy", "pandas", "gymnasium", "pydantic")},
        "simulator_intervals": intervals,
        "simulator_seconds": simulation_seconds,
        "simulator_steps_per_second": intervals / simulation_seconds,
        "resampling_input_intervals": intervals,
        "resampling_seconds": etl_seconds,
        "planner_horizon_steps": 12,
        "planner_first_run_seconds": runs[0].runtime_seconds,
        "planner_mean_seconds": sum(r.runtime_seconds for r in runs) / len(runs),
        "planner_max_seconds": max(r.runtime_seconds for r in runs),
    }


if __name__ == "__main__":
    print(json.dumps(profile(), indent=2))
