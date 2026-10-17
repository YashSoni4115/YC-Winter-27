"""SYNTHETIC exogenous series for developing and testing the ML workstream.

Nothing here is observed data. The series have diurnal and weekly structure,
autocorrelated noise and rare bursts/price spikes so that forecasting is a
non-trivial problem, but the numbers are invented. Swap in the ETL output from
``energy_decision_layer.data`` (see ``load_etl_dataset``) before drawing any
real-world conclusion.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from energy_decision_layer.sim import SiteInputs

WORKLOAD = "workload_demand_compute_units_per_h"
PRICE = "electricity_price_usd_per_mwh"
AMBIENT = "ambient_temperature_c"
SIGNALS: tuple[str, ...] = (WORKLOAD, PRICE, AMBIENT)

DEFAULT_START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _ar1(rng: np.random.Generator, n: int, rho: float, sd: float) -> np.ndarray:
    """Stationary AR(1) noise with marginal standard deviation ``sd``."""
    innovations = rng.normal(0.0, sd * np.sqrt(1.0 - rho**2), n)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sd)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + innovations[i]
    return out


def generate_series(
    n_steps: int,
    *,
    cadence_seconds: float = 300.0,
    seed: int = 0,
    start: datetime = DEFAULT_START,
) -> pd.DataFrame:
    """Return a UTC-indexed frame with the three forecast signals in contract units.

    ``workload_demand_compute_units_per_h`` is a rate (the observation/forecast
    convention). Per-interval totals are ``rate * cadence_seconds / 3600``.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n_steps, freq=pd.Timedelta(seconds=cadence_seconds))
    hour = (index.hour + index.minute / 60.0).to_numpy()
    weekday = (index.dayofweek < 5).astype(float)
    dt_h = cadence_seconds / 3600.0

    # Workload (units per interval at 5 min; the simulated site tops out near 75).
    base = 42.0 + 14.0 * np.sin(2 * np.pi * (hour - 9.0) / 24.0) + 4.0 * weekday
    base = base * (cadence_seconds / 300.0)
    burst = np.zeros(n_steps)
    for i in np.flatnonzero(rng.random(n_steps) < 0.002):
        burst[i : i + 6] += rng.uniform(10.0, 22.0) * (cadence_seconds / 300.0)
    workload_total = np.clip(
        base + _ar1(rng, n_steps, 0.95, 3.0) + burst, 5.0, 72.0 * cadence_seconds / 300.0
    )

    # Price (USD/MWh): evening peak, noise, rare short spikes.
    spike = np.zeros(n_steps)
    for i in np.flatnonzero(rng.random(n_steps) < 0.004):
        spike[i : i + 3] += rng.uniform(150.0, 400.0)
    price = np.maximum(
        55.0 + 25.0 * np.sin(2 * np.pi * (hour - 12.0) / 24.0) + _ar1(rng, n_steps, 0.9, 6.0) + spike,
        -20.0,
    )

    # Ambient temperature (degC): afternoon peak plus slow day-level weather drift.
    day = np.floor(np.arange(n_steps) * cadence_seconds / 86400.0).astype(int)
    day_offset = np.repeat(_ar1(rng, int(day.max()) + 1, 0.7, 3.0), np.bincount(day))
    ambient = (
        17.0
        + 7.0 * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
        + day_offset
        + _ar1(rng, n_steps, 0.98, 0.4)
    )

    return pd.DataFrame(
        {
            WORKLOAD: workload_total / dt_h,
            PRICE: price,
            AMBIENT: np.clip(ambient, -30.0, 45.0),
        },
        index=index.rename("timestamp_utc"),
    )


def to_site_inputs(
    frame: pd.DataFrame,
    *,
    cadence_seconds: float = 300.0,
    deferrable_fraction: float = 0.3,
    deferrable_deadline_steps: int = 12,
) -> list[SiteInputs]:
    """Convert signal rows to simulator inputs.

    The deferrable split and deadline are an explicit SYNTHETIC assumption: public
    workload traces do not carry SLAs (see docs/DATA_SOURCES.md).
    """
    if not 0.0 <= deferrable_fraction <= 1.0:
        raise ValueError("deferrable_fraction must be in [0, 1]")
    dt_h = cadence_seconds / 3600.0
    inputs = []
    for rate, price, ambient in zip(
        frame[WORKLOAD], frame[PRICE], frame[AMBIENT], strict=True
    ):
        total = float(rate) * dt_h
        inputs.append(
            SiteInputs(
                ambient_temperature_c=float(ambient),
                nondeferrable_compute_units=total * (1.0 - deferrable_fraction),
                deferrable_compute_units=total * deferrable_fraction,
                deferrable_deadline_steps=deferrable_deadline_steps,
                grid_import_price_per_mwh=float(price),
            )
        )
    return inputs
