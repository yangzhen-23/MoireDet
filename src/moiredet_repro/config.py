from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from .errors import ConfigurationError


OFFICIAL_ARGS = {
    "backbone": "resnet18",
    "fpem_repeat": 2,
    "pretrained": True,
    "segmentation_head": "FPEM_FFM",
    "is_dct": False,
    "is_light": True,
}


@dataclass(frozen=True)
class InferenceConfig:
    model_name: str
    model_args: Dict[str, Any]
    input_size: Tuple[int, int]
    channel_order: str
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    batch_size: int
    constant_epsilon: float
    minimum_dynamic_range: float


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "inference.yaml"


def load_config(path: Optional[Path] = None) -> InferenceConfig:
    source = Path(path) if path is not None else default_config_path()
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        model, prep = data["model"], data["preprocessing"]
        config = InferenceConfig(
            model_name=str(model["name"]),
            model_args=dict(model["args"]),
            input_size=tuple(prep["input_size"]),
            channel_order=str(prep["channel_order"]),
            mean=tuple(float(value) for value in prep["mean"]),
            std=tuple(float(value) for value in prep["std"]),
            batch_size=int(prep["batch_size"]),
            constant_epsilon=float(data["rendering"]["constant_epsilon"]),
            minimum_dynamic_range=float(data["integration"]["minimum_dynamic_range"]),
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            "Invalid inference config {}: {}".format(source, exc)
        ) from exc

    if config.model_name != "TripleBranchWithSpecificConv" or config.model_args != OFFICIAL_ARGS:
        raise ConfigurationError("Model contract differs from official sample_code.json")
    if (config.input_size, config.channel_order, config.batch_size) != ((320, 320), "BGR", 1):
        raise ConfigurationError("Preprocessing must remain BGR, 320x320, batch size 1")
    if config.mean != (0.485, 0.456, 0.406) or config.std != (0.229, 0.224, 0.225):
        raise ConfigurationError(
            "ImageNet normalization constants differ from the official sample"
        )
    return config
