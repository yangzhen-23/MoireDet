import hashlib
import io
import json
from pathlib import Path

import pytest
import torch

import moiredet_repro.checkpoint as checkpoint_module
from moiredet_repro.checkpoint import (
    default_manifest_path,
    load_checkpoint_bundle,
    load_weights_strict,
)
from moiredet_repro.errors import CheckpointError


class DeserializationTrap:
    """A harmless pickle payload that makes deserialization observable."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (_write_deserialization_marker, (str(self.marker),))


def _write_deserialization_marker(marker):
    Path(marker).write_text("torch.load ran", encoding="utf-8")
    return torch.ones(1)


def write_manifest(path, checkpoint, sha256, **overrides):
    data = {
        "schema_version": 1,
        "filename": checkpoint.name,
        "source_type": "advisor_direct",
        "source_reference": "advisor file transfer",
        "retrieved_at": "2026-07-27",
        "provenance_evidence": "advisor provided the named official checkpoint",
        "expected_sha256": sha256,
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_checkpoint_and_manifest(tmp_path, payload, manifest_overrides=None):
    checkpoint = tmp_path / "model.pth"
    torch.save(payload, checkpoint)
    manifest = Path(str(checkpoint) + ".json")
    write_manifest(
        manifest,
        checkpoint,
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        **(manifest_overrides or {})
    )
    return checkpoint, manifest


def write_trusted_store(path, *manifests):
    checkpoints = []
    for manifest in manifests:
        entry = json.loads(Path(manifest).read_text(encoding="utf-8"))
        entry.pop("schema_version")
        checkpoints.append(entry)
    path.write_text(
        json.dumps({"schema_version": 1, "checkpoints": checkpoints}), encoding="utf-8"
    )


def trust_manifest(monkeypatch, manifest, **overrides):
    store = Path(manifest).with_name("trusted_checkpoints.json")
    if overrides:
        entry = json.loads(Path(manifest).read_text(encoding="utf-8"))
        entry.update(overrides)
        temporary_manifest = store.with_name("trusted-entry.json")
        temporary_manifest.write_text(json.dumps(entry), encoding="utf-8")
        write_trusted_store(store, temporary_manifest)
    else:
        write_trusted_store(store, manifest)
    monkeypatch.setattr(checkpoint_module, "_TRUST_STORE_PATH", store, raising=False)
    return store


def test_default_manifest_appends_json():
    assert default_manifest_path(Path("model.pth")) == Path("model.pth.json")


def test_matching_hash_loads_wrapped_state_dict_and_strips_prefix(tmp_path, monkeypatch):
    checkpoint, manifest = write_checkpoint_and_manifest(
        tmp_path,
        {"state_dict": {"module.weight": torch.ones(1, 1), "module.bias": torch.zeros(1)}},
    )
    trust_manifest(monkeypatch, manifest)

    bundle = load_checkpoint_bundle(checkpoint, manifest)

    assert set(bundle.state_dict) == {"weight", "bias"}
    assert bundle.info.sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert bundle.info.checkpoint_verified is True


def test_deserialization_uses_the_verified_byte_snapshot_when_path_is_replaced(
    tmp_path, monkeypatch
):
    checkpoint, manifest = write_checkpoint_and_manifest(
        tmp_path, {"state_dict": {"weight": torch.tensor([1.0])}}
    )
    expected_bytes = checkpoint.read_bytes()
    replacement = tmp_path / "replacement.pth"
    torch.save({"state_dict": {"weight": torch.tensor([2.0])}}, replacement)
    real_torch_load = torch.load
    received_sources = []
    trust_manifest(monkeypatch, manifest)

    def replace_path_before_load(source, *args, **kwargs):
        checkpoint.write_bytes(replacement.read_bytes())
        received_sources.append(source)
        return real_torch_load(source, *args, **kwargs)

    monkeypatch.setattr(checkpoint_module.torch, "load", replace_path_before_load)

    bundle = load_checkpoint_bundle(checkpoint, manifest)

    assert len(received_sources) == 1
    assert isinstance(received_sources[0], io.BytesIO)
    assert hashlib.sha256(received_sources[0].getvalue()).hexdigest() == hashlib.sha256(
        expected_bytes
    ).hexdigest()
    assert bundle.state_dict["weight"].item() == 1.0


@pytest.mark.parametrize("payload", [b"", b"<!doctype html><html>download failed</html>"])
def test_empty_or_html_is_rejected_before_torch_load(tmp_path, monkeypatch, payload):
    checkpoint = tmp_path / "bad.pth"
    checkpoint.write_bytes(payload)
    manifest = Path(str(checkpoint) + ".json")
    write_manifest(manifest, checkpoint, hashlib.sha256(payload).hexdigest())
    trust_manifest(monkeypatch, manifest)

    with pytest.raises(CheckpointError, match="empty|HTML"):
        load_checkpoint_bundle(checkpoint, manifest)


def test_manifest_requires_a_trusted_complete_provenance_record(tmp_path):
    checkpoint, manifest = write_checkpoint_and_manifest(
        tmp_path,
        {"state_dict": {"weight": torch.ones(1)}},
        {"source_type": "unverified_upload", "provenance_evidence": ""},
    )

    with pytest.raises(CheckpointError, match="untrusted source_type"):
        load_checkpoint_bundle(checkpoint, manifest)


@pytest.mark.parametrize("expected_sha256", ["0" * 64, "A" * 64, "not-a-sha256"])
def test_manifest_rejects_sentinel_or_non_normalized_hashes(tmp_path, expected_sha256):
    checkpoint, manifest = write_checkpoint_and_manifest(
        tmp_path,
        {"state_dict": {"weight": torch.ones(1)}},
        {"expected_sha256": expected_sha256},
    )

    with pytest.raises(CheckpointError, match="expected_sha256"):
        load_checkpoint_bundle(checkpoint, manifest)


def test_bare_state_dict_is_rejected(tmp_path, monkeypatch):
    checkpoint, manifest = write_checkpoint_and_manifest(tmp_path, {"weight": torch.ones(1)})
    trust_manifest(monkeypatch, manifest)

    with pytest.raises(CheckpointError, match="state_dict"):
        load_checkpoint_bundle(checkpoint, manifest)


def test_prefix_normalization_rejects_colliding_keys(tmp_path, monkeypatch):
    checkpoint, manifest = write_checkpoint_and_manifest(
        tmp_path,
        {"state_dict": {"weight": torch.ones(1), "module.weight": torch.ones(1)}},
    )
    trust_manifest(monkeypatch, manifest)

    with pytest.raises(CheckpointError, match="duplicate"):
        load_checkpoint_bundle(checkpoint, manifest)


def test_strict_load_reports_shape_mismatch(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 1)
    checkpoint, manifest = write_checkpoint_and_manifest(
        tmp_path,
        {"state_dict": {"weight": torch.ones(1, 3), "bias": torch.zeros(1)}},
    )
    trust_manifest(monkeypatch, manifest)

    with pytest.raises(CheckpointError, match="shape mismatch.*weight"):
        load_weights_strict(model, load_checkpoint_bundle(checkpoint, manifest))


def test_strict_load_reports_missing_and_unexpected_keys(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 1)
    checkpoint, manifest = write_checkpoint_and_manifest(
        tmp_path,
        {"state_dict": {"weight": torch.ones(1, 2), "extra": torch.zeros(1)}},
    )
    trust_manifest(monkeypatch, manifest)

    with pytest.raises(CheckpointError, match="missing=.*bias.*unexpected=.*extra"):
        load_weights_strict(model, load_checkpoint_bundle(checkpoint, manifest))


def test_self_authored_valid_sidecar_cannot_deserialize_without_repository_pin(tmp_path):
    """Removing repository authorization must block pickle execution before torch.load."""
    marker = tmp_path / "deserialized.txt"
    checkpoint, _ = write_checkpoint_and_manifest(
        tmp_path, {"state_dict": {"weight": DeserializationTrap(marker)}}
    )

    with pytest.raises(CheckpointError, match="repository trusted checkpoint"):
        load_checkpoint_bundle(checkpoint)

    assert not marker.exists()


def test_sidecar_must_exactly_match_repository_pinned_provenance(tmp_path, monkeypatch):
    """Changing any sidecar provenance value must revoke the repository pin."""
    checkpoint, manifest = write_checkpoint_and_manifest(
        tmp_path, {"state_dict": {"weight": torch.ones(1)}}
    )
    trust_manifest(
        monkeypatch,
        manifest,
        provenance_evidence="a different reviewed provenance record",
    )
    monkeypatch.setattr(
        checkpoint_module.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("torch.load must not run for an unpinned sidecar"),
    )

    with pytest.raises(CheckpointError, match="repository trusted checkpoint"):
        load_checkpoint_bundle(checkpoint, manifest)


def test_malformed_repository_store_blocks_deserialization(tmp_path, monkeypatch):
    """Invalid repository metadata cannot fail open to a self-hashed sidecar."""
    checkpoint, manifest = write_checkpoint_and_manifest(
        tmp_path, {"state_dict": {"weight": torch.ones(1)}}
    )
    malformed_store = tmp_path / "trusted_checkpoints.json"
    malformed_store.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(checkpoint_module, "_TRUST_STORE_PATH", malformed_store, raising=False)
    monkeypatch.setattr(
        checkpoint_module.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("torch.load must not run for malformed trust metadata"),
    )

    with pytest.raises(CheckpointError, match="trusted checkpoint store"):
        load_checkpoint_bundle(checkpoint, manifest)


def test_repository_pinned_hash_mismatch_blocks_deserialization(tmp_path, monkeypatch):
    """A reviewed record still cannot deserialize bytes with a different digest."""
    checkpoint, manifest = write_checkpoint_and_manifest(
        tmp_path,
        {"state_dict": {"weight": torch.ones(1)}},
        {"expected_sha256": "a" * 64},
    )
    trust_manifest(monkeypatch, manifest)
    monkeypatch.setattr(
        checkpoint_module.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("torch.load must not run for a hash mismatch"),
    )

    with pytest.raises(CheckpointError, match="SHA-256 mismatch"):
        load_checkpoint_bundle(checkpoint, manifest)
