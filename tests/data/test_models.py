from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from energy_decision_layer.data.models import (
    QualityFlag,
    RawArtifactManifest,
    RedistributionStatus,
    SimulationIntervalV1,
    SourceTerms,
)

UTC = timezone.utc


def terms() -> SourceTerms:
    return SourceTerms(
        terms_url="https://example.test/terms",
        retrieved_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
        terms_sha256="a" * 64,
        redistribution_status=RedistributionStatus.ALLOWED,
    )


def test_simulation_interval_validates_duration_and_price_currency_pair() -> None:
    interval = SimulationIntervalV1(
        site_id="site-a",
        interval_start_utc=datetime(2026, 1, 1, tzinfo=UTC),
        interval_end_utc=datetime(2026, 1, 1, 0, 15, tzinfo=UTC),
        grid_import_price_per_mwh=-5.0,
        price_currency="CAD",
        quality_mask=int(QualityFlag.SYNTHETIC),
        lineage_id="fixture-v1",
    )
    assert interval.duration_hours == pytest.approx(0.25)

    payload = interval.model_dump(mode="json")
    assert SimulationIntervalV1.model_validate(payload) == interval


@pytest.mark.parametrize(
    "updates",
    [
        {"interval_end_utc": datetime(2026, 1, 1, tzinfo=UTC)},
        {"grid_import_price_per_mwh": 10.0, "price_currency": None},
        {"ambient_temperature_c": float("nan")},
        {"unexpected_field": 1},
    ],
)
def test_simulation_interval_rejects_invalid_payloads(updates: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "site_id": "site-a",
        "interval_start_utc": datetime(2026, 1, 1, tzinfo=UTC),
        "interval_end_utc": datetime(2026, 1, 1, 1, tzinfo=UTC),
        "lineage_id": "fixture-v1",
    }
    payload.update(updates)
    with pytest.raises(ValidationError):
        SimulationIntervalV1.model_validate(payload)


def test_non_utc_timestamp_is_rejected_even_when_timezone_aware() -> None:
    with pytest.raises(ValidationError, match="normalized to UTC"):
        SimulationIntervalV1(
            site_id="site-a",
            interval_start_utc=datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=-5))),
            interval_end_utc=datetime(2026, 1, 1, 1, tzinfo=timezone(timedelta(hours=-5))),
            lineage_id="fixture-v1",
        )


def test_manifest_rejects_secret_like_request_parameters() -> None:
    with pytest.raises(ValidationError, match="may contain a secret"):
        RawArtifactManifest(
            source_id="example",
            product_id="prices",
            canonical_url="https://example.test/data",
            retrieved_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
            content_sha256="b" * 64,
            size_bytes=3,
            content_type="text/csv",
            parser_version="1",
            request_parameters={"api_key": "must-not-be-stored"},
            terms=terms(),
        )
