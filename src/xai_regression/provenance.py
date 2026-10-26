"""Content fingerprints for uploaded local comparison artifacts."""

from __future__ import annotations

from hashlib import sha256


def artifact_fingerprint(filename: str, payload: bytes) -> dict[str, str | int]:
    """Return stable metadata that identifies exact uploaded bytes."""
    digest = sha256(payload).hexdigest()
    return {
        "filename": filename,
        "sha256": digest,
        "short_sha256": digest[:12],
        "bytes": len(payload),
    }
