from datetime import datetime, timezone
from pathlib import Path

import pytest

from energy_decision_layer.data.pipeline import (
    CsvMapping,
    chronological_split,
    parse_csv,
    to_site_inputs,
    write_dataset,
)


def test_offline_etl_parquet_and_reproducibility(tmp_path):
    payload = Path("examples/synthetic.csv").read_bytes()
    rows, report = parse_csv(payload, site_id="fixture", mapping=CsvMapping(synthetic=True))
    assert report["missing_rows"] == 0
    assert len(to_site_inputs(rows)) == 4
    write_dataset(rows, report, tmp_path)
    write_dataset(rows, report, tmp_path)
    assert len(list(tmp_path.glob("*/intervals.parquet"))) == 1
    train, test = chronological_split(rows, datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc))
    assert len(train) == len(test) == 2
    assert max(r.interval_end_utc for r in train) <= min(r.interval_start_utc for r in test)


@pytest.mark.parametrize("timestamp", ["2026-11-01T01:30:00", "2026-03-08T02:30:00"])
def test_dst_ambiguity_rejected(timestamp):
    payload = (
        f"timestamp,workload_compute_units,price_per_mwh,temperature_c\n{timestamp},40,20,25\n"
    ).encode()
    with pytest.raises(ValueError, match="ambiguous or nonexistent"):
        parse_csv(payload, site_id="fixture", mapping=CsvMapping(source_timezone="America/Toronto"))


def test_duplicate_missing_and_currency_are_explicit():
    payload = Path("examples/synthetic.csv").read_bytes()
    with pytest.raises(ValueError, match="duplicate"):
        parse_csv(payload + payload.splitlines()[1] + b"\n", site_id="fixture")
    rows, _ = parse_csv(payload, site_id="fixture", mapping=CsvMapping(currency="CAD"))
    with pytest.raises(ValueError, match="exchange rate"):
        to_site_inputs(rows)
    assert to_site_inputs(rows, usd_per_currency=0.7)[0].grid_import_price_per_mwh == 21
    payload = payload.replace(b",40,30,25", b",,30,25")
    rows, report = parse_csv(payload, site_id="fixture")
    assert report["missing_rows"] == 1
    with pytest.raises(ValueError, match="missing mandatory"):
        to_site_inputs(rows)


def test_processed_dataset_detects_corruption(tmp_path):
    payload = Path("examples/synthetic.csv").read_bytes()
    rows, report = parse_csv(payload, site_id="test", mapping=CsvMapping(synthetic=True))
    write_dataset(rows, report, tmp_path)
    path = next(tmp_path.glob("*/intervals.jsonl"))
    path.write_text("corrupted")
    with pytest.raises(ValueError, match="corrupted"):
        write_dataset(rows, report, tmp_path)


def test_raw_to_simulation_uses_source_cadence_and_active_capacity():
    from energy_decision_layer.sim import SiteAction, SiteConfig, SiteSimulator

    rows, _ = parse_csv(Path("examples/synthetic.csv").read_bytes(), site_id="test")
    rows = [
        r.model_copy(update={"active_compute_capacity_compute_units_per_hour": 100.0}) for r in rows
    ]
    sim = SiteSimulator(
        to_site_inputs(rows),
        SiteConfig(site_id="test", timestep_seconds=rows[0].duration_hours * 3600),
        start=rows[0].interval_start_utc,
    )
    o, _ = sim.reset()
    assert o.active_compute_capacity_compute_units_per_h == 100
    _, _, _, _, info = sim.advance(SiteAction(10, 1, 0, 0, 15))
    assert info["completed_compute_units"] <= 100 / 12 + 1e-8


@pytest.mark.parametrize(
    "row", ["2026-01-01T00:00:00Z,40,30", "2026-01-01T00:00:00Z,40,30,25,extra"]
)
def test_malformed_csv_field_counts_report_source_row(row):
    payload = (
        "timestamp,workload_compute_units,price_per_mwh,temperature_c\n" + row + "\n"
    ).encode()
    with pytest.raises(ValueError, match="CSV row 2: row field count"):
        parse_csv(payload, site_id="test")


def test_duplicate_csv_headers_are_rejected():
    payload = b"timestamp,workload_compute_units,price_per_mwh,temperature_c,temperature_c\n"
    with pytest.raises(ValueError, match="duplicate column"):
        parse_csv(payload, site_id="test")
