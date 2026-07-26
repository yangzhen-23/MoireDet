"""Black-box contracts for the repository script entry points."""

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_script(script, *args, cwd=None):
    return subprocess.run(
        [PYTHON, str(ROOT / "scripts" / script), *args],
        cwd=str(ROOT if cwd is None else cwd),
        capture_output=True,
        text=True,
    )


def test_environment_help_is_lightweight_and_exposes_device_control(tmp_path):
    result = run_script("verify_environment.py", "--help", cwd=tmp_path)
    assert result.returncode == 0
    assert "--device" in result.stdout
    assert result.stderr == ""


def test_benchmark_help_is_lightweight_and_exposes_checkpoint_contract(tmp_path):
    result = run_script("benchmark_inference.py", "--help", cwd=tmp_path)
    assert result.returncode == 0
    assert "--checkpoint" in result.stdout
    assert "--checkpoint-manifest" in result.stdout
    assert result.stderr == ""


def test_environment_report_is_json_and_never_attempts_checkpoint_loading(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import verify_environment
    finally:
        sys.path.pop(0)

    monkeypatch.setattr(verify_environment.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(verify_environment, "build_official_model", lambda config: object())

    class FakeService:
        def __init__(self, model, device, checkpoint_info):
            assert checkpoint_info is None

        def predict_tensor(self, tensor):
            return tensor.new_zeros((320, 320)).numpy()

    monkeypatch.setattr(verify_environment, "MoireDetInference", FakeService)
    monkeypatch.setattr(verify_environment, "load_config", lambda: object())
    report, exit_code = verify_environment.collect_report("cpu")
    assert exit_code == 0
    assert report["checkpoint_integration"] == "not_run"
    assert report["cpu_random_forward_shape"] == [320, 320]
    assert report["cuda_random_forward_shape"] is None
    assert json.loads(json.dumps(report))["torch"]


def test_environment_forward_suppresses_upstream_noise(monkeypatch, capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import verify_environment
    finally:
        sys.path.pop(0)

    class FakeService:
        def __init__(self, model, device, checkpoint_info):
            print("upstream noise")

        def predict_tensor(self, tensor):
            print("more upstream noise")
            return tensor.new_zeros((320, 320)).numpy()

    monkeypatch.setattr(verify_environment, "load_config", lambda: object())
    monkeypatch.setattr(verify_environment, "build_official_model", lambda config: object())
    monkeypatch.setattr(verify_environment, "MoireDetInference", FakeService)
    assert verify_environment._run_forward(verify_environment.torch.device("cpu")) == [320, 320]
    assert capsys.readouterr().out == ""
