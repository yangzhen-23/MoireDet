"""Deterministic, display-only rendering of MoireDet prediction maps."""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Dict, Tuple

import cv2
import numpy as np

from .errors import OutputError


TARGETS = ("prediction.npy", "moire_map.png", "comparison.png", "run.json")


@dataclass(frozen=True)
class PredictionStats:
    minimum: float
    maximum: float
    dynamic_range: float


def preflight_output_dir(path: Path) -> Path:
    """Reserve a new output leaf while checking that its parent is writable."""
    output = Path(path)
    if output.exists():
        raise OutputError("refusing to overwrite existing output directory: {}".format(output))
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        handle, probe_name = tempfile.mkstemp(prefix=".moiredet-write-probe-", dir=str(output.parent))
        os.close(handle)
        Path(probe_name).unlink()
    except OSError as exc:
        raise OutputError("output directory is not writable: {}".format(output)) from exc
    return output


def normalize_for_display(prediction: np.ndarray, epsilon: float) -> Tuple[np.ndarray, PredictionStats]:
    """Map a finite float32 prediction to a uint8 image without changing its input."""
    if (
        not isinstance(prediction, np.ndarray)
        or prediction.ndim != 2
        or prediction.dtype != np.float32
        or not np.isfinite(prediction).all()
    ):
        raise OutputError("prediction must be a finite 2D float32 array")
    if not isinstance(epsilon, float) or not math.isfinite(epsilon) or epsilon < 0:
        raise OutputError("display normalization epsilon must be a finite non-negative float")

    minimum = float(prediction.min())
    maximum = float(prediction.max())
    dynamic_range = maximum - minimum
    if dynamic_range <= epsilon:
        normalized = np.zeros_like(prediction)
    else:
        normalized = (prediction - minimum) * (255.0 / dynamic_range)
    rendered = np.rint(np.clip(normalized, 0.0, 255.0)).astype(np.uint8)
    return rendered, PredictionStats(minimum, maximum, dynamic_range)


def render_moire_map(
    prediction: np.ndarray, width: int, height: int, epsilon: float
) -> Tuple[np.ndarray, PredictionStats]:
    """Render a 320x320 prediction at the original image dimensions."""
    if prediction.shape != (320, 320):
        raise OutputError("prediction must have shape 320x320")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise OutputError("rendered map dimensions must be positive integers")
    normalized, stats = normalize_for_display(prediction, epsilon)
    resized = cv2.resize(
        normalized.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    )
    return np.rint(np.clip(resized, 0.0, 255.0)).astype(np.uint8), stats


def make_comparison(original_bgr: np.ndarray, moire_map: np.ndarray) -> np.ndarray:
    """Place the original BGR image beside the grayscale display map as BGR."""
    if (
        not isinstance(original_bgr, np.ndarray)
        or original_bgr.ndim != 3
        or original_bgr.shape[2] != 3
        or original_bgr.dtype != np.uint8
    ):
        raise OutputError("original must be a uint8 3-channel BGR image")
    if moire_map.ndim != 2 or moire_map.dtype != np.uint8:
        raise OutputError("rendered map must be a uint8 grayscale image")
    if original_bgr.shape[:2] != moire_map.shape:
        raise OutputError("original and rendered map dimensions differ")
    return np.concatenate([original_bgr, cv2.cvtColor(moire_map, cv2.COLOR_GRAY2BGR)], axis=1)


def _write_exclusive(path: Path, writer) -> None:
    """Write one artifact only inside this invocation's private staging directory."""
    try:
        with path.open("xb") as handle:
            writer(handle)
    except FileExistsError as exc:
        raise OutputError("staging output already exists: {}".format(path.name)) from exc
    except Exception as exc:
        if isinstance(exc, OutputError):
            raise
        raise OutputError("could not write {}".format(path.name)) from exc


def _write_bytes_exclusive(path: Path, data: bytes) -> None:
    def write_all(handle):
        written = handle.write(data)
        if written != len(data):
            raise OSError("incomplete write")

    _write_exclusive(path, write_all)


def _write_png(path: Path, image: np.ndarray) -> None:
    try:
        ok, encoded = cv2.imencode(".png", image)
    except cv2.error as exc:
        raise OutputError("OpenCV could not encode {}".format(path.name)) from exc
    if not ok:
        raise OutputError("OpenCV could not encode {}".format(path.name))
    _write_bytes_exclusive(path, encoded.tobytes())
    try:
        decoded = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except (OSError, ValueError, cv2.error) as exc:
        raise OutputError("invalid {}".format(path.name)) from exc
    if decoded is None or decoded.shape != image.shape or not np.array_equal(decoded, image):
        raise OutputError("invalid {}".format(path.name))


def _write_npy(path: Path, prediction: np.ndarray) -> None:
    _write_exclusive(path, lambda handle: np.save(handle, prediction, allow_pickle=False))
    try:
        restored = np.load(str(path), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise OutputError("invalid prediction.npy") from exc
    if (
        restored.dtype != np.float32
        or restored.shape != (320, 320)
        or not np.array_equal(restored, prediction)
    ):
        raise OutputError("invalid prediction.npy")


def _write_json(path: Path, metadata: Dict) -> None:
    try:
        serialized = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise OutputError("could not serialize {}".format(path.name)) from exc
    _write_bytes_exclusive(path, serialized.encode("utf-8"))
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise OutputError("invalid run.json") from exc


def _create_stage(output: Path) -> Path:
    try:
        return Path(tempfile.mkdtemp(prefix=".{}-staging-".format(output.name), dir=str(output.parent)))
    except OSError as exc:
        raise OutputError("could not create private staging directory") from exc


def _cleanup_stage(stage: Path) -> None:
    try:
        shutil.rmtree(str(stage))
    except OSError:
        pass


def _publish_stage(stage: Path, output: Path) -> None:
    try:
        stage.rename(output)
    except OSError as exc:
        raise OutputError(
            "refusing to overwrite output directory during publication: {}".format(output)
        ) from exc


def write_output_bundle(
    output_dir: Path,
    original_bgr: np.ndarray,
    prediction: np.ndarray,
    metadata: Dict,
    epsilon: float,
) -> Dict[str, Path]:
    """Publish a complete fixed bundle to a new output directory in one rename."""
    output = preflight_output_dir(output_dir)
    height, width = original_bgr.shape[:2]
    moire_map, stats = render_moire_map(prediction, width, height, epsilon)
    comparison = make_comparison(original_bgr, moire_map)
    if "prediction" in metadata:
        metadata["prediction"].update(
            {"min": stats.minimum, "max": stats.maximum, "dynamic_range": stats.dynamic_range}
        )

    stage = _create_stage(output)
    try:
        _write_npy(stage / "prediction.npy", prediction)
        _write_png(stage / "moire_map.png", moire_map)
        _write_png(stage / "comparison.png", comparison)
        _write_json(stage / "run.json", metadata)
        if {item.name for item in stage.iterdir()} != set(TARGETS):
            raise OutputError("staging directory does not contain the fixed output bundle")
        _publish_stage(stage, output)
    except OutputError:
        _cleanup_stage(stage)
        raise
    except Exception as exc:
        _cleanup_stage(stage)
        raise OutputError("could not build complete output bundle") from exc
    return {name: (output / name).resolve() for name in TARGETS}
