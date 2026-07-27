from pathlib import Path


def test_upstream_baseline_and_sample_are_pinned():
    root = Path(__file__).resolve().parents[1]
    assert (root / "MoireDet" / "script" / "00002423.png").is_file()
    assert (root / "MoireDet" / "script" / "performer_pytorch" / "__init__.py").is_file()


def test_target_model_builds_without_legacy_import_or_network(monkeypatch):
    import importlib
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "MoireDet"), str(root / "MoireDet" / "script")]
    from lib.models import model as official_model

    resnet = importlib.import_module("lib.models.modules.resnet")

    def fail_if_downloaded(*args, **kwargs):
        raise AssertionError("target construction attempted an online download")

    monkeypatch.setattr(resnet, "load_state_dict_from_url", fail_if_downloaded)
    instance = official_model.TripleBranchWithSpecificConv(
        {
            "backbone": "resnet18",
            "fpem_repeat": 2,
            "pretrained": True,
            "segmentation_head": "FPEM_FFM",
            "is_dct": False,
            "is_light": True,
        }
    )
    assert isinstance(instance, official_model.TripleBranchWithSpecificConv)
