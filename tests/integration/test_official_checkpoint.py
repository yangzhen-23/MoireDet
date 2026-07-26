"""Acceptance gate for a trusted official MoireDet checkpoint.

These tests are intentionally skipped until a reviewer supplies both a trusted
checkpoint and its provenance manifest.  A configured artifact is never
treated as absent: validation and strict loading errors must fail the test.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from moiredet_repro.checkpoint import load_checkpoint_bundle, load_weights_strict
from moiredet_repro.cli import main as cli_main
from moiredet_repro.config import load_config
from moiredet_repro.inference import MoireDetInference, set_determinism
from moiredet_repro.preprocessing import prepare_image
from moiredet_repro.upstream_adapter import build_official_model


pytestmark = [pytest.mark.checkpoint, pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "MoireDet" / "script" / "00002423.png"
OUTPUT_NAMES = {"prediction.npy", "moire_map.png", "comparison.png", "run.json"}


def trusted_paths():
    """Return configured trusted artifacts, or visibly skip this external gate."""
    checkpoint = os.environ.get("MOIREDET_CHECKPOINT")
    manifest = os.environ.get("MOIREDET_CHECKPOINT_MANIFEST")
    if not checkpoint or not manifest:
        pytest.skip(
            "trusted checkpoint gate requires both MOIREDET_CHECKPOINT and "
            "MOIREDET_CHECKPOINT_MANIFEST"
        )
    return Path(checkpoint), Path(manifest)


def require_rtx_4060():
    """Keep formal CUDA acceptance tied to the requested review hardware."""
    if not torch.cuda.is_available():
        pytest.skip("formal checkpoint acceptance requires an available RTX 4060 CUDA runtime")
    name = torch.cuda.get_device_name(torch.device("cuda:0"))
    if "RTX 4060" not in name.upper():
        pytest.skip("formal checkpoint acceptance requires RTX 4060; found {}".format(name))


def assert_output_bundle(output, source):
    """Validate all four artifacts and their cross-artifact metadata contract."""
    assert {path.name for path in output.iterdir()} == OUTPUT_NAMES

    prediction = np.load(str(output / "prediction.npy"), allow_pickle=False)
    assert prediction.shape == (320, 320)
    assert prediction.dtype == np.float32
    assert np.isfinite(prediction).all()
    assert np.ptp(prediction) > 1.0e-8

    source_image = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
    moire_map = cv2.imdecode(
        np.fromfile(str(output / "moire_map.png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE
    )
    comparison = cv2.imdecode(
        np.fromfile(str(output / "comparison.png"), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert source_image is not None
    assert moire_map is not None and moire_map.shape == source_image.shape[:2]
    assert comparison is not None and comparison.shape == (
        source_image.shape[0], source_image.shape[1] * 2, 3
    )
    assert np.array_equal(comparison[:, : source_image.shape[1]], source_image)
    assert np.ptp(moire_map) > 0

    metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert metadata["input"]["path"] == str(source.resolve())
    assert metadata["model"]["name"] == "TripleBranchWithSpecificConv"
    assert metadata["model"]["checkpoint"]["checkpoint_verified"] is True
    assert metadata["runtime"]["device_resolved"].startswith("cuda")
    assert metadata["prediction"] == {
        "shape": [320, 320],
        "dtype": "float32",
        "min": pytest.approx(float(prediction.min())),
        "max": pytest.approx(float(prediction.max())),
        "dynamic_range": pytest.approx(float(np.ptp(prediction))),
    }
    assert metadata["performance"]["forward_ms"] > 0
    assert metadata["performance"]["peak_memory_allocated_bytes"] > 0
    assert metadata["performance"]["peak_memory_reserved_bytes"] > 0
    assert metadata["outputs"] == {
        "prediction": "prediction.npy",
        "moire_map": "moire_map.png",
        "comparison": "comparison.png",
        "run": "run.json",
    }


def test_official_checkpoint_strict_loads_on_cpu():
    checkpoint, manifest = trusted_paths()
    info = load_weights_strict(
        build_official_model(load_config()), load_checkpoint_bundle(checkpoint, manifest)
    )
    assert info.checkpoint_verified is True
    assert info.sha256 == info.manifest.expected_sha256


@pytest.mark.gpu
def test_official_sample_cuda_outputs_and_repeatability(tmp_path):
    checkpoint, manifest = trusted_paths()
    require_rtx_4060()

    output = tmp_path / "sample"
    code = cli_main(
        [
            "infer",
            "--input",
            str(SAMPLE),
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-manifest",
            str(manifest),
            "--output",
            str(output),
            "--device",
            "cuda",
        ]
    )
    assert code == 0
    assert_output_bundle(output, SAMPLE)

    set_determinism(2)
    service = MoireDetInference.from_checkpoint(checkpoint, manifest, load_config(), "cuda")
    prepared = prepare_image(SAMPLE, load_config(), service.device)
    first = service.predict_tensor(prepared.tensor)
    second = service.predict_tensor(prepared.tensor)
    np.testing.assert_allclose(first, second, rtol=1.0e-5, atol=1.0e-6)


@pytest.mark.gpu
def test_user_image_cuda_output(tmp_path):
    checkpoint, manifest = trusted_paths()
    user_image = os.environ.get("MOIREDET_USER_IMAGE")
    if not user_image:
        pytest.skip("optional user-image gate requires MOIREDET_USER_IMAGE")
    require_rtx_4060()

    source = Path(user_image)
    output = tmp_path / "user"
    code = cli_main(
        [
            "infer",
            "--input",
            str(source),
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-manifest",
            str(manifest),
            "--output",
            str(output),
            "--device",
            "cuda",
        ]
    )
    assert code == 0
    assert_output_bundle(output, source)


@pytest.mark.gpu
@pytest.mark.benchmark
@pytest.mark.slow
def test_fixed_cuda_benchmark_contract():
    checkpoint, manifest = trusted_paths()
    require_rtx_4060()
    service = MoireDetInference.from_checkpoint(checkpoint, manifest, load_config(), "cuda")
    prepared = prepare_image(SAMPLE, load_config(), service.device)
    stats = service.benchmark_tensor(prepared.tensor, warmup=5, iterations=20)
    assert (stats.warmup, stats.iterations) == (5, 20)
    assert stats.median_ms > 0
    assert stats.p95_ms >= stats.median_ms
    assert stats.peak_memory_allocated_bytes > 0
    assert stats.peak_memory_reserved_bytes > 0
