"""Offline-first configurable CSV ETL. Source-specific mappings must be explicit."""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from energy_decision_layer.data.models import QualityFlag, SimulationIntervalV1
from energy_decision_layer.sim.common import SiteInputs


@dataclass(frozen=True)
class CsvMapping:
    timestamp: str = "timestamp"
    workload: str = "workload_compute_units"
    price: str = "price_per_mwh"
    temperature: str = "temperature_c"
    source_timezone: str = "UTC"
    currency: str = "USD"
    interval_seconds: float = 300.0
    temperature_unit: str = "C"
    workload_scale: float = 1.0
    synthetic: bool = False


def _utc(value: str, zone: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        tz = ZoneInfo(zone)
        first, second = timestamp.replace(tzinfo=tz, fold=0), timestamp.replace(tzinfo=tz, fold=1)
        if first.utcoffset() != second.utcoffset():
            raise ValueError("ambiguous or nonexistent local time; supply explicit UTC offset")
        timestamp = first
        if timestamp.astimezone(timezone.utc).astimezone(tz).replace(
            tzinfo=None
        ) != timestamp.replace(tzinfo=None):
            raise ValueError("nonexistent local time")
    return timestamp.astimezone(timezone.utc)


def parse_csv(
    payload: bytes, *, site_id: str, mapping: CsvMapping | None = None
) -> tuple[list[SimulationIntervalV1], dict[str, object]]:
    import math

    mapping = mapping or CsvMapping()

    if not math.isfinite(mapping.interval_seconds) or mapping.interval_seconds <= 0:
        raise ValueError("interval_seconds must be finite and positive")
    if not math.isfinite(mapping.workload_scale) or mapping.workload_scale <= 0:
        raise ValueError("workload_scale must be finite and positive")
    if mapping.temperature_unit not in {"C", "F"}:
        raise ValueError("temperature unit must be C or F")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    required = {mapping.timestamp, mapping.workload, mapping.price, mapping.temperature}
    if reader.fieldnames is None or not required <= set(reader.fieldnames):
        raise ValueError(f"CSV missing columns: {sorted(required - set(reader.fieldnames or []))}")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("CSV contains duplicate column names")
    lineage = hashlib.sha256(payload).hexdigest()
    output = []
    for number, row in enumerate(reader, start=2):
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("row field count does not match CSV header")
            start = _utc(row[mapping.timestamp], mapping.source_timezone)

            def optional(column: str, source: dict[str, str] = row) -> float | None:
                value = source[column].strip()
                return float(value) if value else None

            demand, price, temperature = (
                optional(mapping.workload),
                optional(mapping.price),
                optional(mapping.temperature),
            )
            if temperature is not None and mapping.temperature_unit == "F":
                temperature = (temperature - 32) * 5 / 9
            mask = int(QualityFlag.SYNTHETIC) if mapping.synthetic else 0
            if any(v is None for v in (demand, price, temperature)):
                mask |= int(QualityFlag.SOURCE_MISSING)
            output.append(
                SimulationIntervalV1(
                    site_id=site_id,
                    interval_start_utc=start,
                    interval_end_utc=start + timedelta(seconds=mapping.interval_seconds),
                    original_timestamp=row[mapping.timestamp],
                    source_timezone=mapping.source_timezone,
                    workload_demand_compute_units=demand * mapping.workload_scale
                    if demand is not None
                    else None,
                    grid_import_price_per_mwh=price,
                    price_currency=mapping.currency if price is not None else None,
                    ambient_temperature_c=temperature,
                    grid_available=True,
                    quality_mask=mask,
                    lineage_id=lineage,
                    metadata={"parser": "mapped-csv-v1", "synthetic": mapping.synthetic},
                )
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"CSV row {number}: {exc}") from exc
    if not output:
        raise ValueError("CSV contains no observations")
    output.sort(key=lambda r: r.interval_start_utc)
    for previous, current in itertools.pairwise(output):
        if current.interval_start_utc < previous.interval_end_utc:
            raise ValueError("duplicate or overlapping source intervals")
    missing = sum(bool(r.quality_mask & int(QualityFlag.SOURCE_MISSING)) for r in output)
    gaps = sum(a.interval_end_utc != b.interval_start_utc for a, b in itertools.pairwise(output))
    outliers = sum(
        r.ambient_temperature_c is not None and not -80 <= r.ambient_temperature_c <= 65
        for r in output
    )
    return output, {
        "schema_version": "1.0.0",
        "rows": len(output),
        "missing_rows": missing,
        "gap_count": gaps,
        "temperature_outliers": outliers,
        "sha256": lineage,
        "synthetic": mapping.synthetic,
    }


def to_site_inputs(
    rows: list[SimulationIntervalV1], *, usd_per_currency: float | None = None
) -> tuple[SiteInputs, ...]:
    import math

    if not rows:
        raise ValueError("no intervals")
    if usd_per_currency is not None and (
        not math.isfinite(usd_per_currency) or usd_per_currency <= 0
    ):
        raise ValueError("FX conversion must be finite and positive")
    result = []
    for i, row in enumerate(rows):
        if row.site_id != rows[0].site_id or row.duration_hours != rows[0].duration_hours:
            raise ValueError("site or cadence differs")
        if i and rows[i - 1].interval_end_utc != row.interval_start_utc:
            raise ValueError("gap or overlap in simulation intervals")
        if (
            row.workload_demand_compute_units is None
            or row.grid_import_price_per_mwh is None
            or row.ambient_temperature_c is None
            or row.grid_available is None
        ):
            raise ValueError("missing mandatory simulation value; explicit imputation required")
        if not -80 <= row.ambient_temperature_c <= 65:
            raise ValueError("temperature outlier requires review")
        if row.price_currency != "USD" and usd_per_currency is None:
            raise ValueError("explicit USD exchange rate required; no implicit CAD/USD conversion")
        result.append(
            SiteInputs(
                nondeferrable_compute_units=row.workload_demand_compute_units,
                grid_import_price_per_mwh=row.grid_import_price_per_mwh
                * ((usd_per_currency or 1.0) if row.price_currency != "USD" else 1.0),
                ambient_temperature_c=row.ambient_temperature_c,
                grid_available=row.grid_available,
                active_compute_capacity_compute_units_per_hour=row.active_compute_capacity_compute_units_per_hour,
                grid_stress=row.grid_stress_status in {"watch", "emergency", "curtailment"},
            )
        )
    return tuple(result)


def chronological_split(
    rows: list[SimulationIntervalV1], cutoff: datetime
) -> tuple[list[SimulationIntervalV1], list[SimulationIntervalV1]]:
    if cutoff.tzinfo is None:
        raise ValueError("cutoff must be aware")
    # Intervals straddling the cutoff are excluded from both sets.
    return (
        [r for r in rows if r.interval_end_utc <= cutoff],
        [r for r in rows if r.interval_start_utc >= cutoff],
    )


def write_dataset(
    rows: list[SimulationIntervalV1], report: dict[str, object], directory: str | Path
) -> None:
    import os
    import shutil
    import tempfile

    import pandas as pd

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    content = "\n".join(r.model_dump_json() for r in rows) + "\n"
    version = hashlib.sha256(("dataset-v1-json-metadata\n" + content).encode()).hexdigest()
    parquet_rows = [
        dict(r.model_dump(mode="json"), metadata=json.dumps(r.metadata, sort_keys=True))
        for r in rows
    ]
    target = directory / version
    report_content = json.dumps(report, indent=2) + "\n"
    if target.exists():
        if (target / "intervals.jsonl").read_text() != content or (
            target / "quality.json"
        ).read_text() != report_content:
            raise ValueError("existing dataset version has conflicting or corrupted contents")
        frame = pd.read_parquet(target / "intervals.parquet")
        try:
            pd.testing.assert_frame_equal(frame, pd.DataFrame(parquet_rows), check_dtype=False)
        except AssertionError as exc:
            raise ValueError("existing Parquet content differs from validated rows") from exc
        return
    temporary = Path(tempfile.mkdtemp(prefix=".dataset-", dir=directory))
    try:
        (temporary / "intervals.jsonl").write_text(content)
        pd.DataFrame(parquet_rows).to_parquet(temporary / "intervals.parquet", index=False)
        (temporary / "quality.json").write_text(report_content)
        os.rename(temporary, target)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
