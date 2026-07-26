"""Verify the pinned MoireDet runtime without reading a checkpoint."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import version
import json
import io
import platform
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np
import torch
import torchvision
import yaml

from moiredet_repro.config import load_config
from moiredet_repro.inference import MoireDetInference, set_determinism
from moiredet_repro.upstream_adapter import build_official_model


EXPECTED = {
    "torch": "1.10.0+cu113",
    "torchvision": "0.11.1+cu113",
    "cuda_runtime": "11.3",
    "opencv-python": "4.11.0.86",
    "numpy": "1.24.3",
    "PyYAML": "6.0.2",
    "einops": "0.3.0",
    "local-attention": "1.2.1",
}


def parser():
    value = argparse.ArgumentParser(description="Verify pinned MoireDet CPU/CUDA runtime")
    value.add_argument("--device", choices=("cpu", "cuda", "all"), default="all")
    return value


def _distribution_versions():
    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "opencv-python": version("opencv-python"),
        "numpy": np.__version__,
        "PyYAML": yaml.__version__,
        "einops": version("einops"),
        "local-attention": version("local-attention"),
    }


def _run_forward(device):
    # The upstream Performer dependency prints import-time capability notices.
    # Keep stdout a single machine-readable report for automation.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        config = load_config()
        service = MoireDetInference(build_official_model(config), device, None)
        tensor = torch.randn(1, 3, 320, 320, device=device)
        return list(service.predict_tensor(tensor).shape)


def collect_report(requested_device="all"):
    """Run only random model forwards and return a JSON-safe report plus status."""
    set_determinism(2)
    installed = _distribution_versions()
    pins_match = installed == EXPECTED
    cuda_available = torch.cuda.is_available()
    cpu_shape = None
    cuda_shape = None

    if requested_device in ("cpu", "all"):
        cpu_shape = _run_forward(torch.device("cpu"))
    if requested_device in ("cuda", "all") and cuda_available:
        matrix = torch.randn(512, 512, device="cuda:0")
        _ = matrix @ matrix
        torch.cuda.synchronize()
        cuda_shape = _run_forward(torch.device("cuda:0"))

    requested_shape = cpu_shape if requested_device == "cpu" else cuda_shape
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
        "einops": installed["einops"],
        "local_attention": installed["local-attention"],
        "installed_versions": installed,
        "expected_versions": EXPECTED,
        "versions_match": pins_match,
        "opencv_module_version": cv2.__version__,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "cpu_random_forward_shape": cpu_shape,
        "cuda_random_forward_shape": cuda_shape,
        "checkpoint_integration": "not_run",
        "checkpoint_deserialization": "not_attempted",
        "network_access": "not_attempted",
    }
    return report, int(not (pins_match and requested_shape == [320, 320]))


def main(argv=None):
    args = parser().parse_args(argv)
    report, exit_code = collect_report(args.device)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
