"""Trusted checkpoint validation and strict state-dict loading."""

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Optional

import torch

from .errors import CheckpointError


ALLOWED_SOURCES = {
    "official_repository",
    "author_direct",
    "author_team_direct",
    "advisor_direct",
    "verified_mirror",
}
_MANIFEST_KEYS = {
    "schema_version",
    "filename",
    "source_type",
    "source_reference",
    "retrieved_at",
    "provenance_evidence",
    "expected_sha256",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CheckpointManifest:
    filename: str
    source_type: str
    source_reference: str
    retrieved_at: str
    provenance_evidence: str
    expected_sha256: str


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    size_bytes: int
    sha256: str
    manifest: CheckpointManifest
    checkpoint_verified: bool


@dataclass(frozen=True)
class CheckpointBundle:
    state_dict: Mapping[str, torch.Tensor]
    info: CheckpointInfo


def default_manifest_path(checkpoint: Path) -> Path:
    return Path(str(checkpoint) + ".json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path, checkpoint: Path) -> CheckpointManifest:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if type(data) is not dict or set(data) != _MANIFEST_KEYS:
            raise ValueError("manifest fields do not match schema version 1")
        if data["schema_version"] != 1:
            raise ValueError("schema_version must be 1")
        manifest = CheckpointManifest(
            filename=data["filename"],
            source_type=data["source_type"],
            source_reference=data["source_reference"],
            retrieved_at=data["retrieved_at"],
            provenance_evidence=data["provenance_evidence"],
            expected_sha256=data["expected_sha256"],
        )
        if any(type(value) is not str for value in vars(manifest).values()):
            raise ValueError("manifest text fields must be strings")
        date.fromisoformat(manifest.retrieved_at)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise CheckpointError("invalid checkpoint manifest {}: {}".format(path, exc)) from exc

    if manifest.filename != checkpoint.name:
        raise CheckpointError("manifest filename does not match checkpoint")
    if manifest.source_type not in ALLOWED_SOURCES:
        raise CheckpointError("untrusted source_type: {}".format(manifest.source_type))
    if not manifest.source_reference.strip() or not manifest.provenance_evidence.strip():
        raise CheckpointError("source_reference and provenance_evidence are required")
    if not _SHA256_RE.fullmatch(manifest.expected_sha256):
        raise CheckpointError("expected_sha256 must be 64 lowercase hexadecimal characters")
    if manifest.expected_sha256 == "0" * 64:
        raise CheckpointError("expected_sha256 must not be the all-zero sentinel")
    return manifest


def _normalize_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    normalized = OrderedDict()
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise CheckpointError("invalid state_dict key: {!r}".format(key))
        normalized_key = key[7:] if key.startswith("module.") else key
        if normalized_key in normalized:
            raise CheckpointError("invalid or duplicate state_dict key: {}".format(normalized_key))
        if not isinstance(value, torch.Tensor):
            raise CheckpointError("invalid state_dict tensor: {}".format(normalized_key))
        normalized[normalized_key] = value
    return normalized


def load_checkpoint_bundle(
    checkpoint_path: Path, manifest_path: Optional[Path] = None
) -> CheckpointBundle:
    checkpoint = Path(checkpoint_path)
    manifest_file = (
        Path(manifest_path) if manifest_path is not None else default_manifest_path(checkpoint)
    )
    if not checkpoint.is_file():
        raise CheckpointError("checkpoint file is missing: {}".format(checkpoint))

    manifest = _read_manifest(manifest_file, checkpoint)
    size = checkpoint.stat().st_size
    if size == 0:
        raise CheckpointError("checkpoint is empty")
    with checkpoint.open("rb") as handle:
        head = handle.read(512).lower()
    if b"<html" in head or b"<!doctype html" in head:
        raise CheckpointError("checkpoint is an HTML download page")

    actual_sha256 = _sha256_file(checkpoint)
    if actual_sha256 != manifest.expected_sha256:
        raise CheckpointError(
            "SHA-256 mismatch: expected {}, got {}".format(
                manifest.expected_sha256, actual_sha256
            )
        )
    try:
        payload = torch.load(str(checkpoint), map_location="cpu")
    except Exception as exc:
        raise CheckpointError(
            "trusted checkpoint could not be deserialized: {}".format(exc)
        ) from exc
    if type(payload) is not dict or not isinstance(payload.get("state_dict"), Mapping):
        raise CheckpointError("official checkpoint must contain a top-level state_dict mapping")

    info = CheckpointInfo(
        checkpoint.resolve(), size, actual_sha256, manifest, checkpoint_verified=True
    )
    return CheckpointBundle(_normalize_state_dict(payload["state_dict"]), info)


def load_weights_strict(model: torch.nn.Module, bundle: CheckpointBundle) -> CheckpointInfo:
    model_state = model.state_dict()
    checkpoint_state = bundle.state_dict
    missing = sorted(set(model_state) - set(checkpoint_state))
    unexpected = sorted(set(checkpoint_state) - set(model_state))
    mismatched = [
        (key, tuple(checkpoint_state[key].shape), tuple(model_state[key].shape))
        for key in sorted(set(model_state) & set(checkpoint_state))
        if tuple(checkpoint_state[key].shape) != tuple(model_state[key].shape)
    ]
    if missing or unexpected or mismatched:
        raise CheckpointError(
            "strict state_dict mismatch; missing={}; unexpected={}; shape mismatch={}".format(
                missing[:10], unexpected[:10], mismatched[:10]
            )
        )
    try:
        model.load_state_dict(checkpoint_state, strict=True)
    except RuntimeError as exc:
        raise CheckpointError("strict state_dict load failed: {}".format(exc)) from exc
    return bundle.info
