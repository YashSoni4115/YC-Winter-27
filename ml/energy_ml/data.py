"""Load Devansh's ETL output (``SimulationIntervalV1``) as a training frame.

Unit handling follows docs/INTEGRATION.md: ETL interval workload is a TOTAL, the
forecast/observation workload is a RATE (divide by interval hours), and non-USD
prices need an explicit exchange rate. Missing values are rejected, not imputed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from energy_decision_layer.data.models import SimulationIntervalV1

from .synthetic import AMBIENT, PRICE, WORKLOAD


def load_etl_dataset(
    intervals_jsonl: str | Path, *, usd_per_currency: float | None = None
) -> pd.DataFrame:
    """Read ``<dataset>/intervals.jsonl`` into a UTC-indexed frame in contract units."""
    lines = [ln for ln in Path(intervals_jsonl).read_text().splitlines() if ln.strip()]
    rows = [SimulationIntervalV1.model_validate_json(ln) for ln in lines]
    if not rows:
        raise ValueError("dataset contains no intervals")
    records = []
    for i, r in enumerate(rows):
        if i and rows[i - 1].interval_end_utc != r.interval_start_utc:
            raise ValueError("gap or overlap in intervals; resample or repair upstream")
        if r.duration_hours != rows[0].duration_hours:
            raise ValueError("cadence changes within the dataset")
        if (
            r.workload_demand_compute_units is None
            or r.grid_import_price_per_mwh is None
            or r.ambient_temperature_c is None
        ):
            raise ValueError("missing value; impute explicitly before training")
        price = r.grid_import_price_per_mwh
        if r.price_currency != "USD":
            if usd_per_currency is None or not np.isfinite(usd_per_currency) or usd_per_currency <= 0:
                raise ValueError("non-USD price requires an explicit positive usd_per_currency")
            price *= usd_per_currency
        records.append(
            {
                "timestamp_utc": r.interval_start_utc,
                WORKLOAD: r.workload_demand_compute_units / r.duration_hours,
                PRICE: price,
                AMBIENT: r.ambient_temperature_c,
            }
        )
    frame = pd.DataFrame.from_records(records).set_index("timestamp_utc")
    frame.index = pd.DatetimeIndex(frame.index).tz_convert("UTC")
    return frame
