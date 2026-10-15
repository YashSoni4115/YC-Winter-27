"""Immutable, content-addressed storage for exact source responses."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from energy_decision_layer.data.models import RawArtifactManifest, SourceTerms


class ArtifactIntegrityError(RuntimeError):
    """Raised when stored bytes or metadata no longer match their digest."""


class RawArtifactStore:
    """Persist exact bytes and their manifest without destructive overwrites.

    Layout is ``root/source_id/<digest>/payload.bin`` plus ``manifest.json``.
    Re-storing identical content is idempotent. A conflicting manifest for the
    same content is rejected so provenance cannot be silently rewritten.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def put(
        self,
        payload: bytes,
        *,
        source_id: str,
        product_id: str,
        canonical_url: str,
        retrieved_at_utc: datetime,
        content_type: str,
        parser_version: str,
        terms: SourceTerms,
        request_parameters: dict[str, str] | None = None,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> RawArtifactManifest:
        digest = hashlib.sha256(payload).hexdigest()
        manifest = RawArtifactManifest(
            source_id=source_id,
            product_id=product_id,
            canonical_url=canonical_url,
            retrieved_at_utc=retrieved_at_utc,
            content_sha256=digest,
            size_bytes=len(payload),
            content_type=content_type,
            parser_version=parser_version,
            request_parameters=request_parameters or {},
            etag=etag,
            last_modified=last_modified,
            terms=terms,
        )
        directory = self.root / source_id / digest
        manifest_path = directory / "manifest.json"
        directory.parent.mkdir(parents=True, exist_ok=True)
        serialized = manifest.model_dump_json(indent=2).encode("utf-8") + b"\n"
        if directory.exists():
            return self._existing(manifest_path, manifest)
        # Publish the complete pair with one directory rename. No reader sees
        # partially written payload/manifest pairs, including concurrent writers.
        temporary = Path(tempfile.mkdtemp(prefix=".staging-", dir=directory.parent))
        try:
            self._atomic_write(temporary / "payload.bin", payload)
            self._atomic_write(temporary / "manifest.json", serialized)
            try:
                os.rename(temporary, directory)
            except OSError:
                if directory.exists():
                    return self._existing(manifest_path, manifest)
                raise
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        self.verify(manifest_path)
        return manifest

    def _existing(self, path: Path, expected: RawArtifactManifest) -> RawArtifactManifest:
        existing = self.verify(path)
        if existing != expected:
            raise ArtifactIntegrityError(
                "identical content was previously stored with different provenance metadata"
            )
        return existing

    def verify(self, manifest_path: str | Path) -> RawArtifactManifest:
        manifest_file = Path(manifest_path)
        try:
            manifest = RawArtifactManifest.model_validate_json(manifest_file.read_bytes())
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(f"invalid manifest {manifest_file}: {exc}") from exc
        payload_path = manifest_file.with_name("payload.bin")
        try:
            payload = payload_path.read_bytes()
        except OSError as exc:
            raise ArtifactIntegrityError(f"missing payload {payload_path}") from exc
        digest = hashlib.sha256(payload).hexdigest()
        if digest != manifest.content_sha256 or len(payload) != manifest.size_bytes:
            raise ArtifactIntegrityError(f"checksum or size mismatch for {payload_path}")
        return manifest

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name)
