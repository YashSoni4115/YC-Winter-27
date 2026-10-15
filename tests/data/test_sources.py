from datetime import datetime, timedelta, timezone

import pytest

from energy_decision_layer.data.sources import parse_ieso_zonal_xml, parse_noaa_temperatures


def xml(interval="1", price="42.0", hour="1"):
    # Synthetic schema-shaped data, no redistributed source observations.
    return (
        f'<Document docID="RealtimeOntarioZonalPrice" xmlns="http://www.ieso.ca/schema">'
        f"<DocBody><DeliveryDate>2026-07-01</DeliveryDate><DeliveryHour>{hour}</DeliveryHour>"
        f"<ZonalPrice><Interval>{interval}</Interval><LmpCap>{price}</LmpCap></ZonalPrice>"
        "</DocBody></Document>"
    ).encode()


def test_ieso_uses_fixed_est_hour_ending_even_in_summer():
    [r] = parse_ieso_zonal_xml(xml())
    assert r.start_utc == datetime(2026, 7, 1, 5, tzinfo=timezone.utc)
    assert r.duration_seconds == 300
    assert r.value == 42


@pytest.mark.parametrize(
    "payload",
    [xml("13"), xml(price="nan"), xml(hour="25"), b"<wrong/>", b"<!DOCTYPE x><Document/>"],
)
def test_malformed_ieso_rejected(payload):
    with pytest.raises(ValueError):
        parse_ieso_zonal_xml(payload)


def test_noaa_quality_and_units():
    payload = (
        b'STATION,DATE,TMP\nabc,2026-01-01T00:00:00,"+0123,1"\n'
        b'abc,2026-01-01T01:00:00,"+9999,9"\n'
        b'other,2026-01-01T00:00:00,"+0010,1"\n'
    )
    [r] = parse_noaa_temperatures(payload, station="abc")
    assert r.value == 12.3
    assert r.start_utc.utcoffset() == timedelta(0)
    with pytest.raises(ValueError, match="no trustworthy"):
        parse_noaa_temperatures(payload, station="unknown")


def test_google_concurrent_task_rates_are_summed_and_work_conserved():
    from energy_decision_layer.data.models import QualityFlag, SignalSemantics
    from energy_decision_layer.data.resampling import resample_intervals
    from energy_decision_layer.data.sources import parse_google_task_usage

    def row(start, end, task, cpu):
        return ",".join([str(start), str(end), "job", task, "machine", str(cpu)] + [""] * 14)

    payload = (row(0, 3_600_000_000, "a", 0.5) + "\n" + row(0, 1_800_000_000, "b", 0.25)).encode()
    origin = datetime(2011, 5, 1, tzinfo=timezone.utc)
    values = parse_google_task_usage(
        payload, trace_origin_utc=origin, compute_units_per_cpu_hour=100
    )
    assert sum(v.value for v in values) == pytest.approx(62.5)
    assert all(v.quality_mask & int(QualityFlag.SYNTHETIC) for v in values)
    result = resample_intervals(
        values,
        start_utc=origin,
        end_utc=origin + timedelta(hours=1),
        interval=timedelta(minutes=5),
        semantics=SignalSemantics.INTERVAL_TOTAL,
    )
    assert sum(v.value for v in result) == pytest.approx(62.5)
    with pytest.raises(ValueError, match="duplicate"):
        parse_google_task_usage(
            payload + b"\n" + payload.splitlines()[0],
            trace_origin_utc=origin,
            compute_units_per_cpu_hour=100,
        )


def test_public_fetch_retains_exact_bytes_and_rejects_partial_download(monkeypatch, tmp_path):
    from email.message import Message
    from io import BytesIO

    from energy_decision_layer.data.artifacts import RawArtifactStore
    from energy_decision_layer.data.models import RedistributionStatus, SourceTerms
    from energy_decision_layer.data.sources import fetch_public

    class Response(BytesIO):
        headers = Message()
        headers["Content-Type"] = "text/csv"

        def geturl(self):
            return "https://www.ncei.noaa.gov/test"

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: Response(b"sample"))
    terms = SourceTerms(
        terms_url="urn:test:synthetic",
        retrieved_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        terms_sha256="a" * 64,
        redistribution_status=RedistributionStatus.UNKNOWN,
    )
    store = RawArtifactStore(tmp_path)
    manifest = fetch_public(
        "https://www.ncei.noaa.gov/test",
        store=store,
        source_id="noaa.test",
        product_id="test",
        terms=terms,
    )
    assert manifest.size_bytes == 6
    assert next(iter(tmp_path.glob("**/payload.bin"))).read_bytes() == b"sample"
    with pytest.raises(ValueError, match="byte limit"):
        fetch_public(
            "https://www.ncei.noaa.gov/test",
            store=store,
            source_id="oversize",
            product_id="test",
            terms=terms,
            max_bytes=3,
        )
    assert not (tmp_path / "oversize").exists()
    with pytest.raises(ValueError, match="HTTPS"):
        fetch_public(
            "http://example.test/test",
            store=store,
            source_id="test",
            product_id="test",
            terms=terms,
        )
