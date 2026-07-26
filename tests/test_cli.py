from pathlib import Path
import subprocess
import sys

import pytest

import moiredet_repro.cli as cli


def test_parser_defaults_to_auto_and_sidecar_manifest():
    args = cli.build_parser().parse_args(
        ["infer", "--input", "a.png", "--checkpoint", "model.pth", "--output", "out"]
    )
    assert args.device == "auto"
    assert args.checkpoint_manifest is None


def test_module_help_runs_from_repository_root():
    result = subprocess.run(
        [sys.executable, "-m", "moiredet_repro.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "infer" in result.stdout


def test_output_conflict_stops_before_model_build(tmp_path, monkeypatch):
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"unused because preflight must run first")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"unused")
    output = tmp_path / "out"
    output.mkdir()
    (output / "run.json").write_text("keep")

    def fail_if_model_is_built(**kwargs):
        raise AssertionError("model construction ran before output preflight")

    monkeypatch.setattr(cli.MoireDetInference, "from_checkpoint", fail_if_model_is_built)
    code = cli.main(
        [
            "infer",
            "--input",
            str(input_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ]
    )
    assert code == 2
    assert (output / "run.json").read_text() == "keep"


def test_fake_service_end_to_end_writes_and_reports_all_outputs(tmp_path, monkeypatch, capsys):
    import numpy as np
    import torch

    from moiredet_repro.checkpoint import CheckpointInfo, CheckpointManifest
    from moiredet_repro.inference import PerformanceStats
    from moiredet_repro.preprocessing import PreparedImage

    input_path = tmp_path / "中文输入.png"
    input_path.write_bytes(b"decode is replaced by the prepared-image fixture")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint loading is replaced by the fake service")
    output = tmp_path / "out"
    manifest = CheckpointManifest(
        filename="model.pth",
        source_type="official_repository",
        source_reference="https://example.invalid/model.pth",
        retrieved_at="2026-07-27",
        provenance_evidence="synthetic CLI fixture",
        expected_sha256="2" * 64,
    )

    class FakeService:
        device = torch.device("cpu")
        checkpoint_info = CheckpointInfo(
            checkpoint.resolve(), checkpoint.stat().st_size, "2" * 64, manifest, True
        )
        last_performance = PerformanceStats(1.25, None, None)

        def predict_tensor(self, tensor):
            return np.linspace(0, 1, 320 * 320, dtype=np.float32).reshape(320, 320)

    prepared = PreparedImage(
        np.zeros((5, 9, 3), dtype=np.uint8), torch.zeros(1, 3, 320, 320), 9, 5
    )
    monkeypatch.setattr(cli, "prepare_image", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(cli.MoireDetInference, "from_checkpoint", lambda **kwargs: FakeService())

    code = cli.main(
        [
            "infer",
            "--input",
            str(input_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    for name in ("prediction.npy", "moire_map.png", "comparison.png", "run.json"):
        path = output / name
        assert path.is_file()
        assert str(path.resolve()) in captured.out
