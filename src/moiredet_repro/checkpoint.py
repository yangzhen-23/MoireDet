"""Trusted checkpoint validation and strict state-dict loading."""

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
import hashlib
import io
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
_TRUST_STORE_KEYS = {"schema_version", "checkpoints"}
_TRUSTED_CHECKPOINT_KEYS = _MANIFEST_KEYS - {"schema_version"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TRUST_STORE_PATH = Path(__file__).resolve().parents[2] / "configs" / "trusted_checkpoints.json"


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


def _read_checkpoint_snapshot(path: Path) -> bytes:
    """Return the only checkpoint bytes that will be verified and deserialized."""
    try:
        snapshot = path.read_bytes()
    except OSError as exc:
        raise CheckpointError("checkpoint file is missing or unreadable: {}".format(path)) from exc
    if not snapshot:
        raise CheckpointError("checkpoint is empty")
    head = snapshot[:512].lower()
    if b"<html" in head or b"<!doctype html" in head:
        raise CheckpointError("checkpoint is an HTML download page")
    return snapshot


def _sha256_bytes(snapshot: bytes) -> str:
    return hashlib.sha256(snapshot).hexdigest()


def _read_manifest(path: Path, checkpoint: Path) -> CheckpointManifest:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if type(data) is not dict or set(data) != _MANIFEST_KEYS:
            raise ValueError("manifest fields do not match schema version 1")
        if data["schema_version"] != 1:
            raise ValueError("schema_version must be 1")
        manifest = _manifest_from_trust_record(data)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise CheckpointError("invalid checkpoint manifest {}: {}".format(path, exc)) from exc

    if manifest.filename != checkpoint.name:
        raise CheckpointError("manifest filename does not match checkpoint")
    return manifest


def _manifest_from_trust_record(data) -> CheckpointManifest:
    """Validate provenance fields shared by sidecars and repository pins."""
    manifest = CheckpointManifest(
        filename=data["filename"],
        source_type=data["source_type"],
        source_reference=data["source_reference"],
        retrieved_at=data["retrieved_at"],
        provenance_evidence=data["provenance_evidence"],
        expected_sha256=data["expected_sha256"],
    )
    if any(type(value) is not str for value in vars(manifest).values()):
        raise ValueError("checkpoint provenance fields must be strings")
    if not manifest.filename.strip():
        raise ValueError("filename is required")
    date.fromisoformat(manifest.retrieved_at)
    if manifest.source_type not in ALLOWED_SOURCES:
        raise ValueError("untrusted source_type: {}".format(manifest.source_type))
    if not manifest.source_reference.strip() or not manifest.provenance_evidence.strip():
        raise ValueError("source_reference and provenance_evidence are required")
    if not _SHA256_RE.fullmatch(manifest.expected_sha256):
        raise ValueError("expected_sha256 must be 64 lowercase hexadecimal characters")
    if manifest.expected_sha256 == "0" * 64:
        raise ValueError("expected_sha256 must not be the all-zero sentinel")
    return manifest


def _read_trusted_checkpoints(path: Path):
    """Read only repository-controlled checkpoint pins, never user sidecars."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if type(data) is not dict or set(data) != _TRUST_STORE_KEYS:
            raise ValueError("trusted store fields do not match schema version 1")
        if data["schema_version"] != 1:
            raise ValueError("schema_version must be 1")
        if type(data["checkpoints"]) is not list:
            raise ValueError("checkpoints must be a list")
        trusted = []
        for entry in data["checkpoints"]:
            if type(entry) is not dict or set(entry) != _TRUSTED_CHECKPOINT_KEYS:
                raise ValueError("trusted checkpoint fields do not match schema version 1")
            trusted.append(_manifest_from_trust_record(entry))
        if len(set(trusted)) != len(trusted):
            raise ValueError("trusted checkpoint entries must be unique")
        return tuple(trusted)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise CheckpointError("invalid trusted checkpoint store {}: {}".format(path, exc)) from exc


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
    manifest = _read_manifest(manifest_file, checkpoint)
    trusted = _read_trusted_checkpoints(_TRUST_STORE_PATH)
    if manifest not in trusted:
        raise CheckpointError(
            "checkpoint manifest is not authorized by repository trusted checkpoint store"
        )
    snapshot = _read_checkpoint_snapshot(checkpoint)
    actual_sha256 = _sha256_bytes(snapshot)
    if actual_sha256 != manifest.expected_sha256:
        raise CheckpointError(
            "SHA-256 mismatch: expected {}, got {}".format(
                manifest.expected_sha256, actual_sha256
            )
        )
    try:
        payload = torch.load(io.BytesIO(snapshot), map_location="cpu")
    except Exception as exc:
        raise CheckpointError(
            "trusted checkpoint could not be deserialized: {}".format(exc)
        ) from exc
    if type(payload) is not dict or not isinstance(payload.get("state_dict"), Mapping):
        raise CheckpointError("official checkpoint must contain a top-level state_dict mapping")

    info = CheckpointInfo(
        checkpoint.resolve(), len(snapshot), actual_sha256, manifest, checkpoint_verified=True
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
