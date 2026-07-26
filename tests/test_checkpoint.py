import hashlib
import json
from pathlib import Path

import pytest
import torch

from moiredet_repro.checkpoint import (
    default_manifest_path,
    load_checkpoint_bundle,
    load_weights_strict,
)
from moiredet_repro.errors import CheckpointError


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


def test_default_manifest_appends_json():
    assert default_manifest_path(Path("model.pth")) == Path("model.pth.json")


def test_matching_hash_loads_wrapped_state_dict_and_strips_prefix(tmp_path):
    checkpoint, manifest = write_checkpoint_and_manifest(
        tmp_path,
        {"state_dict": {"module.weight": torch.ones(1, 1), "module.bias": torch.zeros(1)}},
    )

    bundle = load_checkpoint_bundle(checkpoint, manifest)

    assert set(bundle.state_dict) == {"weight", "bias"}
    assert bundle.info.sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert bundle.info.checkpoint_verified is True


@pytest.mark.parametrize("payload", [b"", b"<!doctype html><html>download failed</html>"])
def test_empty_or_html_is_rejected_before_torch_load(tmp_path, payload):
    checkpoint = tmp_path / "bad.pth"
    checkpoint.write_bytes(payload)
    manifest = Path(str(checkpoint) + ".json")
    write_manifest(manifest, checkpoint, hashlib.sha256(payload).hexdigest())

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


def test_bare_state_dict_is_rejected(tmp_path):
    checkpoint, manifest = write_checkpoint_and_manifest(tmp_path, {"weight": torch.ones(1)})

    with pytest.raises(CheckpointError, match="state_dict"):
        load_checkpoint_bundle(checkpoint, manifest)


def test_prefix_normalization_rejects_colliding_keys(tmp_path):
    checkpoint, manifest = write_checkpoint_and_manifest(
        tmp_path,
        {"state_dict": {"weight": torch.ones(1), "module.weight": torch.ones(1)}},
    )

    with pytest.raises(CheckpointError, match="duplicate"):
        load_checkpoint_bundle(checkpoint, manifest)


def test_strict_load_reports_shape_mismatch(tmp_path):
    model = torch.nn.Linear(2, 1)
    checkpoint, manifest = write_checkpoint_and_manifest(
        tmp_path,
        {"state_dict": {"weight": torch.ones(1, 3), "bias": torch.zeros(1)}},
    )

    with pytest.raises(CheckpointError, match="shape mismatch.*weight"):
        load_weights_strict(model, load_checkpoint_bundle(checkpoint, manifest))


def test_strict_load_reports_missing_and_unexpected_keys(tmp_path):
    model = torch.nn.Linear(2, 1)
    checkpoint, manifest = write_checkpoint_and_manifest(
        tmp_path,
        {"state_dict": {"weight": torch.ones(1, 2), "extra": torch.zeros(1)}},
    )

    with pytest.raises(CheckpointError, match="missing=.*bias.*unexpected=.*extra"):
        load_weights_strict(model, load_checkpoint_bundle(checkpoint, manifest))
