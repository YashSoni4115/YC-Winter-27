from datetime import datetime, timedelta, timezone

import pytest

from energy_decision_layer.data.artifacts import ArtifactIntegrityError, RawArtifactStore
from energy_decision_layer.data.models import RedistributionStatus, SourceTerms

UTC = timezone.utc


def terms() -> SourceTerms:
    return SourceTerms(
        terms_url="https://example.test/terms",
        retrieved_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
        terms_sha256="a" * 64,
        redistribution_status=RedistributionStatus.ALLOWED,
        required_attribution="Synthetic Example Source",
    )


def put(store: RawArtifactStore, *, retrieved_at: datetime | None = None):
    return store.put(
        b"timestamp,value\n2026-01-01T00:00:00Z,1\n",
        source_id="synthetic.example",
        product_id="example-v1",
        canonical_url="https://example.test/data.csv",
        retrieved_at_utc=retrieved_at or datetime(2026, 1, 2, tzinfo=UTC),
        content_type="text/csv",
        parser_version="1.0.0",
        terms=terms(),
        request_parameters={"day": "2026-01-01"},
    )


def test_artifact_store_is_content_addressed_and_idempotent(tmp_path) -> None:
    store = RawArtifactStore(tmp_path)
    first = put(store)
    second = put(store)
    assert second == first

    manifest_path = tmp_path / "synthetic.example" / first.content_sha256 / "manifest.json"
    assert store.verify(manifest_path) == first
    assert manifest_path.with_name("payload.bin").read_bytes().startswith(b"timestamp")


def test_artifact_store_rejects_provenance_rewrite_for_same_bytes(tmp_path) -> None:
    store = RawArtifactStore(tmp_path)
    put(store)
    with pytest.raises(ArtifactIntegrityError, match="different provenance"):
        put(store, retrieved_at=datetime(2026, 1, 2, tzinfo=UTC) + timedelta(seconds=1))


def test_artifact_store_detects_corruption(tmp_path) -> None:
    store = RawArtifactStore(tmp_path)
    manifest = put(store)
    manifest_path = tmp_path / "synthetic.example" / manifest.content_sha256 / "manifest.json"
    manifest_path.with_name("payload.bin").write_bytes(b"corrupt")
    with pytest.raises(ArtifactIntegrityError, match="checksum or size mismatch"):
        store.verify(manifest_path)


def test_concurrent_raw_writers_publish_complete_artifact(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    store = RawArtifactStore(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: put(store), range(12)))
    assert all(r == results[0] for r in results)
    assert len(list(tmp_path.glob("**/manifest.json"))) == 1
    assert not list(tmp_path.glob("**/.staging-*"))
