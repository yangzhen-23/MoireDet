import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from moiredet_repro.checkpoint import CheckpointInfo, CheckpointManifest
from moiredet_repro.config import load_config
from moiredet_repro.errors import OutputError
from moiredet_repro.inference import PerformanceStats
from moiredet_repro.metadata import build_run_metadata
from moiredet_repro.rendering import (
    make_comparison,
    normalize_for_display,
    preflight_output_dir,
    render_moire_map,
    write_output_bundle,
)


def test_minmax_rounding_and_constant_protection():
    pred = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
    image, stats = normalize_for_display(pred, epsilon=1e-12)

    assert image.dtype == np.uint8
    assert image.tolist() == [[0, 128], [191, 255]]
    assert stats.minimum == 0.0
    assert stats.maximum == 1.0
    assert stats.dynamic_range == 1.0

    constant, _ = render_moire_map(np.ones((320, 320), np.float32), 9, 5, 1e-12)
    assert constant.shape == (5, 9)
    assert np.count_nonzero(constant) == 0


def test_map_and_comparison_restore_original_dimensions():
    original = np.full((5, 9, 3), 17, dtype=np.uint8)
    moire, _ = render_moire_map(
        np.arange(320 * 320, dtype=np.float32).reshape(320, 320), 9, 5, 1e-12
    )
    comparison = make_comparison(original, moire)

    assert moire.shape == (5, 9)
    assert comparison.shape == (5, 18, 3)
    assert np.array_equal(comparison[:, :9], original)
    assert np.array_equal(comparison[:, 9:, 0], moire)


@pytest.mark.parametrize(
    "name", ["prediction.npy", "moire_map.png", "comparison.png", "run.json"]
)
def test_preflight_refuses_any_existing_target(tmp_path, name):
    old = tmp_path / name
    old.write_bytes(b"keep")

    with pytest.raises(OutputError, match="refusing to overwrite"):
        preflight_output_dir(tmp_path)

    assert old.read_bytes() == b"keep"


def test_bundle_preserves_raw_prediction_and_schema(tmp_path):
    prediction = np.linspace(-1, 1, 320 * 320, dtype=np.float32).reshape(320, 320)
    original = np.zeros((5, 9, 3), dtype=np.uint8)
    metadata = {
        "schema_version": 1,
        "outputs": {
            "prediction": "prediction.npy",
            "moire_map": "moire_map.png",
            "comparison": "comparison.png",
            "run": "run.json",
        },
    }

    write_output_bundle(tmp_path, original, prediction, metadata, 1e-12)

    assert np.array_equal(np.load(str(tmp_path / "prediction.npy")), prediction)
    assert cv2.imdecode(np.fromfile(str(tmp_path / "moire_map.png"), np.uint8), 0).shape == (5, 9)
    assert json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))["schema_version"] == 1


def test_bundle_removes_artifacts_created_before_png_write_failure(tmp_path, monkeypatch):
    prediction = np.zeros((320, 320), dtype=np.float32)
    original = np.zeros((5, 9, 3), dtype=np.uint8)

    def fail_encoding(extension, image):
        return False, None

    monkeypatch.setattr(cv2, "imencode", fail_encoding)
    with pytest.raises(OutputError, match="OpenCV could not encode"):
        write_output_bundle(tmp_path, original, prediction, {"schema_version": 1}, 1e-12)

    assert not any((tmp_path / name).exists() for name in ("prediction.npy", "moire_map.png", "comparison.png", "run.json"))


def test_fixed_metadata_schema_has_every_required_key():
    manifest = CheckpointManifest(
        filename="model.pth",
        source_type="official_repository",
        source_reference="https://example.invalid/model.pth",
        retrieved_at="2026-07-27",
        provenance_evidence="synthetic unit-test fixture",
        expected_sha256="1" * 64,
    )
    info = CheckpointInfo(Path("model.pth").resolve(), 123, "1" * 64, manifest, True)
    fake_prepared = SimpleNamespace(width=9, height=5)
    fake_service = SimpleNamespace(
        checkpoint_info=info,
        last_performance=PerformanceStats(1.25, None, None),
        device=torch.device("cpu"),
    )
    prediction = np.zeros((320, 320), dtype=np.float32)

    metadata = build_run_metadata(
        Path("sample.png"),
        fake_prepared,
        load_config(),
        fake_service,
        prediction,
        "auto",
    )

    assert set(metadata) == {
        "schema_version",
        "created_at_utc",
        "input",
        "model",
        "runtime",
        "preprocessing",
        "prediction",
        "performance",
        "outputs",
    }
    assert set(metadata["model"]["checkpoint"]) == {
        "path",
        "filename",
        "size_bytes",
        "sha256",
        "source_type",
        "source_reference",
        "retrieved_at",
        "provenance_evidence",
        "checkpoint_verified",
    }
    assert metadata["prediction"]["shape"] == [320, 320]
    assert metadata["outputs"] == {
        "prediction": "prediction.npy",
        "moire_map": "moire_map.png",
        "comparison": "comparison.png",
        "run": "run.json",
    }
