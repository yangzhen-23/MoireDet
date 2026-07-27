"""Black-box contracts for the repository script entry points."""

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import urllib.request

import pytest


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_script(script, *args, cwd=None):
    return subprocess.run(
        [PYTHON, str(ROOT / "scripts" / script), *args],
        cwd=str(ROOT if cwd is None else cwd),
        capture_output=True,
        text=True,
    )


def load_benchmark_module(monkeypatch):
    """Load the CLI module without its CUDA runtime dependencies."""
    inference = types.ModuleType("moiredet_repro.inference")
    inference.MoireDetInference = object
    inference.set_determinism = lambda seed: None
    preprocessing = types.ModuleType("moiredet_repro.preprocessing")
    preprocessing.prepare_image = lambda *args: None
    monkeypatch.setitem(sys.modules, "moiredet_repro.inference", inference)
    monkeypatch.setitem(sys.modules, "moiredet_repro.preprocessing", preprocessing)
    module_name = "benchmark_inference_test_{}".format(id(monkeypatch))
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts" / "benchmark_inference.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


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


def test_benchmark_refuses_output_created_after_preflight(monkeypatch, tmp_path, capsys):
    """A writer racing the final publication must retain its exact bytes."""
    benchmark_inference = load_benchmark_module(monkeypatch)

    output = tmp_path / "reports" / "benchmark.json"
    concurrent_bytes = b'{"writer":"concurrent"}\n'

    @dataclass
    class BenchmarkResult:
        latency_ms: float = 1.0

    class FakeService:
        device = object()
        checkpoint_info = type(
            "CheckpointInfo",
            (),
            {"sha256": "a" * 64, "checkpoint_verified": True},
        )()

        @classmethod
        def from_checkpoint(cls, *args):
            return cls()

        def benchmark_tensor(self, tensor, warmup, iterations):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(concurrent_bytes)
            return BenchmarkResult()

    monkeypatch.setattr(benchmark_inference, "load_config", lambda: object())
    monkeypatch.setattr(benchmark_inference, "set_determinism", lambda seed: None)
    monkeypatch.setattr(benchmark_inference, "MoireDetInference", FakeService)
    monkeypatch.setattr(
        benchmark_inference,
        "prepare_image",
        lambda input_path, config, device: type("Prepared", (), {"tensor": object()})(),
    )

    with pytest.raises(SystemExit) as raised:
        benchmark_inference.main(
            [
                "--input",
                str(tmp_path / "input.png"),
                "--checkpoint",
                str(tmp_path / "checkpoint.pt"),
                "--checkpoint-manifest",
                str(tmp_path / "checkpoint.json"),
                "--output",
                str(output),
            ]
        )

    assert raised.value.code == 2
    assert output.read_bytes() == concurrent_bytes
    assert "--output must be a fresh, nonexistent file" in capsys.readouterr().err


def test_benchmark_writer_creates_complete_readable_finite_json(monkeypatch, tmp_path):
    benchmark_inference = load_benchmark_module(monkeypatch)

    output = tmp_path / "reports" / "benchmark.json"
    payload = {"latency_ms": 1.25, "checkpoint_verified": True}
    benchmark_inference.write_fresh_json(output, payload)

    raw = output.read_bytes()
    assert json.loads(raw.decode("utf-8")) == payload


def test_benchmark_writer_preserves_existing_output_directory(monkeypatch, tmp_path):
    benchmark_inference = load_benchmark_module(monkeypatch)

    output = tmp_path / "benchmark.json"
    output.mkdir()

    with pytest.raises(benchmark_inference.BenchmarkOutputError):
        benchmark_inference.write_fresh_json(output, {"latency_ms": 1.25})

    assert output.is_dir()


def test_benchmark_writer_rejects_nonfinite_json(monkeypatch, tmp_path):
    benchmark_inference = load_benchmark_module(monkeypatch)

    output = tmp_path / "benchmark.json"
    with pytest.raises(benchmark_inference.BenchmarkOutputError):
        benchmark_inference.write_fresh_json(output, {"latency_ms": float("nan")})

    assert not output.exists()


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


def test_environment_cpu_forward_does_not_deserialize_or_access_network(monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import verify_environment
    finally:
        sys.path.pop(0)

    calls = []

    def forbidden(name):
        def reject(*args, **kwargs):
            calls.append(name)
            raise AssertionError("verifier attempted forbidden operation: {}".format(name))

        return reject

    monkeypatch.setattr(verify_environment.torch, "load", forbidden("torch.load"))
    monkeypatch.setattr(verify_environment.torch.hub, "load", forbidden("torch.hub.load"))
    monkeypatch.setattr(
        verify_environment.torch.hub,
        "load_state_dict_from_url",
        forbidden("torch.hub.load_state_dict_from_url"),
    )
    monkeypatch.setattr(
        verify_environment.torch.utils.model_zoo,
        "load_url",
        forbidden("torch.utils.model_zoo.load_url"),
    )
    monkeypatch.setattr(urllib.request, "urlopen", forbidden("urllib.request.urlopen"))

    report, exit_code = verify_environment.collect_report("cpu")
    assert exit_code == 0
    assert calls == []
    assert report["checkpoint_integration"] == "not_run"
    assert report["checkpoint_deserialization"] == "not_attempted"
    assert report["network_access"] == "not_attempted"
    assert report["safety_guards"] == "checkpoint/network operations prohibited"


def test_clean_process_cpu_forward_keeps_import_bound_download_aliases_guarded():
    probe = r'''
import json
import sys
import urllib.request

import torch
import torch.utils.model_zoo

blocked = []

def guard(name):
    def reject(*args, **kwargs):
        blocked.append(name)
        raise AssertionError("forbidden operation: {}".format(name))
    return reject

torch_load = guard("torch.load")
hub_load = guard("torch.hub.load")
hub_state_dict = guard("torch.hub.load_state_dict_from_url")
model_zoo_load = guard("torch.utils.model_zoo.load_url")
urlopen = guard("urllib.request.urlopen")
torch.load = torch_load
torch.hub.load = hub_load
torch.hub.load_state_dict_from_url = hub_state_dict
torch.utils.model_zoo.load_url = model_zoo_load
urllib.request.urlopen = urlopen

sys.path.insert(0, "scripts")
import verify_environment

report, exit_code = verify_environment.collect_report("cpu")
import lib.models.modules.resnet as resnet
import lib.models.modules.resnet_dct as resnet_dct

if resnet.load_state_dict_from_url is not hub_state_dict:
    raise AssertionError("resnet cached an unguarded download alias")
if resnet_dct.load_state_dict_from_url is not hub_state_dict:
    raise AssertionError("resnet_dct cached an unguarded download alias")
if blocked:
    raise AssertionError("forbidden operations were called: {}".format(blocked))
if exit_code != 0 or report["cpu_random_forward_shape"] != [320, 320]:
    raise AssertionError("CPU official forward contract failed: {}".format(report))
if report["checkpoint_deserialization"] != "not_attempted":
    raise AssertionError("checkpoint deserialization status drifted")
if report["network_access"] != "not_attempted":
    raise AssertionError("network status drifted")
print(json.dumps({"blocked": blocked, "report": report}, sort_keys=True))
'''
    result = subprocess.run(
        [PYTHON, "-c", probe], cwd=str(ROOT), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["blocked"] == []
    assert payload["report"]["cpu_random_forward_shape"] == [320, 320]
    assert payload["report"]["checkpoint_deserialization"] == "not_attempted"
    assert payload["report"]["network_access"] == "not_attempted"
