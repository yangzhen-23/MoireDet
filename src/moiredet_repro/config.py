from dataclasses import dataclass
import math
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

_TOP_LEVEL_KEYS = {"model", "preprocessing", "rendering", "integration"}
_MODEL_KEYS = {"name", "args"}
_PREPROCESSING_KEYS = {"input_size", "channel_order", "mean", "std", "batch_size"}
_RENDERING_KEYS = {"constant_epsilon"}
_INTEGRATION_KEYS = {"minimum_dynamic_range"}


def require_mapping(value: Any, name: str, keys: set) -> Dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("{} must be a mapping".format(name))
    if set(value) != keys:
        raise ValueError("{} has unexpected or missing keys".format(name))
    return value


def require_exact_type(value: Any, expected: type, name: str) -> None:
    if type(value) is not expected:
        raise ValueError("{} must be a {}".format(name, expected.__name__))


def require_float_vector(value: Any, name: str, length: int) -> Tuple[float, ...]:
    if type(value) is not list or len(value) != length:
        raise ValueError("{} must be a {}-element list".format(name, length))
    if any(type(item) is not float or not math.isfinite(item) for item in value):
        raise ValueError("{} must contain finite floats".format(name))
    return tuple(value)


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
        data = require_mapping(data, "root", _TOP_LEVEL_KEYS)
        model = require_mapping(data["model"], "model", _MODEL_KEYS)
        args = require_mapping(model["args"], "model.args", set(OFFICIAL_ARGS))
        prep = require_mapping(data["preprocessing"], "preprocessing", _PREPROCESSING_KEYS)
        rendering = require_mapping(data["rendering"], "rendering", _RENDERING_KEYS)
        integration = require_mapping(data["integration"], "integration", _INTEGRATION_KEYS)
        require_exact_type(model["name"], str, "model.name")
        require_exact_type(args["backbone"], str, "model.args.backbone")
        require_exact_type(args["fpem_repeat"], int, "model.args.fpem_repeat")
        require_exact_type(args["pretrained"], bool, "model.args.pretrained")
        require_exact_type(args["segmentation_head"], str, "model.args.segmentation_head")
        require_exact_type(args["is_dct"], bool, "model.args.is_dct")
        require_exact_type(args["is_light"], bool, "model.args.is_light")
        if type(prep["input_size"]) is not list or len(prep["input_size"]) != 2:
            raise ValueError("preprocessing.input_size must be a two-element list")
        if any(type(value) is not int for value in prep["input_size"]):
            raise ValueError("preprocessing.input_size must contain integers")
        require_exact_type(prep["channel_order"], str, "preprocessing.channel_order")
        require_exact_type(prep["batch_size"], int, "preprocessing.batch_size")
        mean = require_float_vector(prep["mean"], "preprocessing.mean", 3)
        std = require_float_vector(prep["std"], "preprocessing.std", 3)
        require_exact_type(rendering["constant_epsilon"], float, "rendering.constant_epsilon")
        require_exact_type(
            integration["minimum_dynamic_range"],
            float,
            "integration.minimum_dynamic_range",
        )
        if not math.isfinite(rendering["constant_epsilon"]):
            raise ValueError("rendering.constant_epsilon must be finite")
        if not math.isfinite(integration["minimum_dynamic_range"]):
            raise ValueError("integration.minimum_dynamic_range must be finite")
        config = InferenceConfig(
            model_name=model["name"],
            model_args=dict(args),
            input_size=tuple(prep["input_size"]),
            channel_order=prep["channel_order"],
            mean=mean,
            std=std,
            batch_size=prep["batch_size"],
            constant_epsilon=rendering["constant_epsilon"],
            minimum_dynamic_range=integration["minimum_dynamic_range"],
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
    if config.constant_epsilon != 1.0e-12:
        raise ConfigurationError("Rendering epsilon differs from the fixed contract")
    if config.minimum_dynamic_range != 1.0e-8:
        raise ConfigurationError("Minimum dynamic range differs from the fixed contract")
    return config
