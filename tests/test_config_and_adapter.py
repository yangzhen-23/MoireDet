from pathlib import Path

import pytest

from moiredet_repro.config import load_config
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
