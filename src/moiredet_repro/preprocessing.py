from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch

from .config import InferenceConfig
from .errors import InputImageError


@dataclass(frozen=True)
class PreparedImage:
    original_bgr: np.ndarray
    tensor: torch.Tensor
    width: int
    height: int


def load_bgr_image(path: Union[str, Path]) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise InputImageError("missing input image: {}".format(source))
    try:
        encoded = np.fromfile(str(source), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except (OSError, ValueError, cv2.error) as exc:
        raise InputImageError(
            "corrupt or unreadable input image: {}".format(source)
        ) from exc
    if image is None:
        raise InputImageError("corrupt or unreadable input image: {}".format(source))
    if image.ndim != 3 or image.shape[2] != 3:
        raise InputImageError(
            "input must be a 3-channel BGR image; got shape {}".format(image.shape)
        )
    return np.ascontiguousarray(image)


def preprocess_bgr(
    image: np.ndarray, config: InferenceConfig, device: torch.device
) -> torch.Tensor:
    if image.ndim != 3 or image.shape[2] != 3:
        raise InputImageError("in-memory input must be a 3-channel BGR image")

    resized = cv2.resize(image, config.input_size, interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(
        np.ascontiguousarray(resized.transpose(2, 0, 1))
    ).float()
    tensor.div_(255.0)
    mean = torch.tensor(config.mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(config.std, dtype=torch.float32).view(3, 1, 1)
    tensor = tensor.sub(mean).div(std).unsqueeze(0)
    if tuple(tensor.shape) != (1, 3, 320, 320) or not torch.isfinite(tensor).all():
        raise InputImageError("preprocessed tensor contract failed")
    return tensor.to(device).contiguous()


def prepare_image(
    path: Union[str, Path], config: InferenceConfig, device: torch.device
) -> PreparedImage:
    original = load_bgr_image(path)
    height, width = original.shape[:2]
    return PreparedImage(
        original_bgr=original,
        tensor=preprocess_bgr(original, config, device),
        width=width,
        height=height,
    )
