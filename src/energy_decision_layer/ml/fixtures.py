"""Explicitly synthetic normalized fixture generation; never real site telemetry."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np

from energy_decision_layer.data.models import QualityFlag, SimulationIntervalV1


def synthetic_intervals(
    steps: int = 2016,
    *,
    seed: int = 42,
    site_id: str = "synthetic-site",
    cadence_seconds: float = 300.0,
    start: datetime | None = None,
) -> tuple[SimulationIntervalV1, ...]:
    if steps < 1 or cadence_seconds <= 0:
        raise ValueError("Positive interval count and cadence required")
    rng = np.random.default_rng(seed)
    start = start or datetime(2025, 12, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(steps):
        t = start + timedelta(seconds=i * cadence_seconds)
        phase = 2 * np.pi * (t.hour + t.minute / 60) / 24
        demand = float(max(0.0, 470 + 160 * np.sin(phase) + rng.normal(0, 12)))
        temperature = float(25 + 7 * np.sin(phase - 1) + rng.normal(0, 0.25))
        price = float(65 + 30 * np.sin(phase - 0.5) + rng.normal(0, 3))
        stress: Literal["watch", "normal"] = "watch" if np.sin(phase - 0.5) > 0.85 else "normal"
        rows.append(
            SimulationIntervalV1(
                site_id=site_id,
                interval_start_utc=t,
                interval_end_utc=t + timedelta(seconds=cadence_seconds),
                workload_demand_compute_units=demand * cadence_seconds / 3600,
                active_compute_capacity_compute_units_per_hour=900.0,
                ambient_temperature_c=temperature,
                grid_import_price_per_mwh=price,
                price_currency="USD",
                grid_stress_status=stress,
                grid_available=True,
                quality_mask=int(QualityFlag.SYNTHETIC),
                lineage_id=f"synthetic-seed-{seed}",
                metadata={"is_synthetic": True, "fixture_seed": seed},
            )
        )
    return tuple(rows)
