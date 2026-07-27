from pathlib import Path

import pytest
import yaml

from moiredet_repro.config import ConfigurationError, default_config_path, load_config
from moiredet_repro.upstream_adapter import build_official_model, resolve_upstream


def test_inference_config_matches_official_contract():
    config = load_config()
    assert config.model_name == "TripleBranchWithSpecificConv"
    assert config.model_args == {
        "backbone": "resnet18",
        "fpem_repeat": 2,
        "pretrained": True,
        "segmentation_head": "FPEM_FFM",
        "is_dct": False,
        "is_light": True,
    }
    assert config.input_size == (320, 320)
    assert config.channel_order == "BGR"
    assert config.mean == (0.485, 0.456, 0.406)
    assert config.std == (0.229, 0.224, 0.225)
    assert config.batch_size == 1


def write_config(tmp_path, data):
    path = tmp_path / "inference.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def official_config_data():
    return yaml.safe_load(default_config_path().read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model", "args", "pretrained"), 1),
        (("model", "args", "is_dct"), 0),
        (("preprocessing", "batch_size"), True),
        (("preprocessing", "input_size"), [320.0, 320.0]),
    ],
)
def test_inference_config_rejects_numeric_and_boolean_type_coercion(
    tmp_path, path, value
):
    data = official_config_data()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ConfigurationError):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("rendering", "constant_epsilon"), 1.0e-9),
        (("rendering", "constant_epsilon"), float("inf")),
        (("integration", "minimum_dynamic_range"), 1.0e-7),
        (("integration", "minimum_dynamic_range"), float("nan")),
    ],
)
def test_inference_config_rejects_drifted_or_non_finite_constants(
    tmp_path, path, value
):
    data = official_config_data()
    data[path[0]][path[1]] = value

    with pytest.raises(ConfigurationError):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize("contents", ["model: [", "[]", "null"])
def test_inference_config_rejects_malformed_or_non_mapping_yaml(tmp_path, contents):
    path = tmp_path / "inference.yaml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_config(path)


@pytest.mark.parametrize("mutation", ["missing_model", "wrong_model", "wrong_args"])
def test_inference_config_rejects_missing_or_wrong_mapping_keys(tmp_path, mutation):
    data = official_config_data()
    if mutation == "missing_model":
        del data["model"]
    elif mutation == "wrong_model":
        data["model"] = []
    else:
        data["model"]["args"] = []

    with pytest.raises(ConfigurationError):
        load_config(write_config(tmp_path, data))


def test_upstream_paths_are_resolved_independently_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = resolve_upstream()
    assert paths.repo_root.name == "MoireDet"
    assert (paths.repo_root / "lib").is_dir()
    assert (paths.performer_root / "performer_pytorch" / "__init__.py").is_file()


def test_model_factory_passes_only_official_arguments():
    captured = {}

    class SpyModel:
        def __init__(self, args):
            captured.update(args)

    model = build_official_model(load_config(), model_class=SpyModel)
    assert isinstance(model, SpyModel)
    assert captured == load_config().model_args
    assert "ouput_channel" not in captured


@pytest.mark.integration
def test_actual_official_model_builds_without_download(monkeypatch):
    import torch.hub

    monkeypatch.setattr(
        torch.hub,
        "load_state_dict_from_url",
        lambda *args, **kwargs: pytest.fail("network download attempted"),
    )
    model = build_official_model(load_config())
    assert type(model).__name__ == "TripleBranchWithSpecificConv"
