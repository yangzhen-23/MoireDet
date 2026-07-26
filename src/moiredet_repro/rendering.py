"""Deterministic, display-only rendering of MoireDet prediction maps."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
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
    """Return a writable output directory only when no target would be replaced."""
    output = Path(path)
    conflicts = [name for name in TARGETS if (output / name).exists()]
    if conflicts:
        raise OutputError(
            "refusing to overwrite existing output(s): {}".format(", ".join(conflicts))
        )
    try:
        output.mkdir(parents=True, exist_ok=True)
        probe = output / ".write-probe"
        probe.write_bytes(b"")
        probe.unlink()
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


def _write_png(path: Path, image: np.ndarray) -> None:
    try:
        ok, encoded = cv2.imencode(".png", image)
    except cv2.error as exc:
        raise OutputError("OpenCV could not encode {}".format(path.name)) from exc
    if not ok:
        raise OutputError("OpenCV could not encode {}".format(path.name))
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise OutputError("could not write {}".format(path.name)) from exc


def write_output_bundle(
    output_dir: Path,
    original_bgr: np.ndarray,
    prediction: np.ndarray,
    metadata: Dict,
    epsilon: float,
) -> Dict[str, Path]:
    """Write all fixed output artifacts, deleting artifacts created if writing fails."""
    output = preflight_output_dir(output_dir)
    height, width = original_bgr.shape[:2]
    moire_map, stats = render_moire_map(prediction, width, height, epsilon)
    comparison = make_comparison(original_bgr, moire_map)
    if "prediction" in metadata:
        metadata["prediction"].update(
            {"min": stats.minimum, "max": stats.maximum, "dynamic_range": stats.dynamic_range}
        )

    created = []
    try:
        prediction_path = output / "prediction.npy"
        created.append(prediction_path)
        with prediction_path.open("xb") as handle:
            np.save(handle, prediction, allow_pickle=False)

        map_path = output / "moire_map.png"
        created.append(map_path)
        _write_png(map_path, moire_map)

        comparison_path = output / "comparison.png"
        created.append(comparison_path)
        _write_png(comparison_path, comparison)

        run_path = output / "run.json"
        created.append(run_path)
        with run_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
    except (OutputError, OSError, TypeError, ValueError) as exc:
        for artifact in reversed(created):
            try:
                artifact.unlink()
            except OSError:
                pass
        if isinstance(exc, OutputError):
            raise
        raise OutputError("could not write complete output bundle") from exc
    return {name: (output / name).resolve() for name in TARGETS}
