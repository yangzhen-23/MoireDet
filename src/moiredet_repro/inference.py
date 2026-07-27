"""Validated single-image inference and repeatable timing for official MoireDet."""

from dataclasses import dataclass
from time import perf_counter
from typing import Optional
import random

import numpy as np
import torch

from .checkpoint import CheckpointInfo, load_checkpoint_bundle, load_weights_strict
from .config import InferenceConfig
from .errors import DeviceError, InferenceError
from .upstream_adapter import build_official_model


@dataclass(frozen=True)
class PerformanceStats:
    forward_ms: float
    peak_memory_allocated_bytes: Optional[int]
    peak_memory_reserved_bytes: Optional[int]


@dataclass(frozen=True)
class BenchmarkStats:
    warmup: int
    iterations: int
    median_ms: float
    p95_ms: float
    peak_memory_allocated_bytes: Optional[int]
    peak_memory_reserved_bytes: Optional[int]


def set_determinism(seed: int = 2) -> None:
    """Set the random state and deterministic cuDNN policy used by the wrapper."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(requested: str) -> torch.device:
    """Resolve a supported device request without ever downgrading explicit CUDA."""
    if requested not in {"auto", "cpu", "cuda"}:
        raise DeviceError("device must be one of: auto, cpu, cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if requested == "cuda":
        raise DeviceError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device("cpu")


def unwrap_official_output(output) -> torch.Tensor:
    """Return the density tensor only when the documented model contract is met."""
    valid = (
        isinstance(output, tuple)
        and len(output) == 2
        and isinstance(output[0], list)
        and len(output[0]) == 1
        and isinstance(output[0][0], torch.Tensor)
        and isinstance(output[1], torch.Tensor)
        and tuple(output[0][0].shape) == (1, 1, 320, 320)
    )
    if not valid:
        raise InferenceError(
            "model did not satisfy the official output contract ([1x1x320x320], fea_loss)"
        )
    return output[0][0]


class MoireDetInference:
    """Own an official model, its resolved device, and validated prediction calls."""

    def __init__(
        self, model: torch.nn.Module, device: torch.device, checkpoint_info: Optional[CheckpointInfo]
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.checkpoint_info = checkpoint_info
        self.last_performance = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        manifest_path,
        config: InferenceConfig,
        device: str = "auto",
    ):
        resolved = select_device(device)
        model = build_official_model(config)
        bundle = load_checkpoint_bundle(checkpoint_path, manifest_path)
        info = load_weights_strict(model, bundle)
        return cls(model, resolved, info)

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _validate_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise InferenceError("input must be a torch.Tensor")
        if tuple(tensor.shape) != (1, 3, 320, 320):
            raise InferenceError("input tensor must have shape 1x3x320x320")
        if not tensor.is_floating_point():
            raise InferenceError("input tensor must have a floating point dtype")
        return tensor.to(self.device)

    def _forward_checked(self, tensor: torch.Tensor) -> torch.Tensor:
        try:
            raw = unwrap_official_output(self.model(tensor))
        except RuntimeError as exc:
            if self.device.type == "cuda" and "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise InferenceError("CUDA out of memory; no CPU fallback was attempted") from exc
            raise
        if not torch.isfinite(raw).all().item():
            raise InferenceError("prediction contains NaN or Inf")
        return raw

    def _peak_memory(self):
        if self.device.type != "cuda":
            return None, None
        return (
            torch.cuda.max_memory_allocated(self.device),
            torch.cuda.max_memory_reserved(self.device),
        )

    def predict_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = self._validate_input(tensor)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._synchronize()
        started = perf_counter()
        with torch.no_grad():
            raw = self._forward_checked(tensor)
        self._synchronize()
        elapsed = (perf_counter() - started) * 1000.0
        prediction = raw[0, 0].detach().to("cpu").numpy().astype(np.float32, copy=True)
        allocated, reserved = self._peak_memory()
        self.last_performance = PerformanceStats(elapsed, allocated, reserved)
        return prediction

    def benchmark_tensor(
        self, tensor: torch.Tensor, warmup: int = 5, iterations: int = 20
    ) -> BenchmarkStats:
        if not isinstance(warmup, int) or warmup < 0:
            raise InferenceError("warmup must be a non-negative integer")
        if not isinstance(iterations, int) or iterations <= 0:
            raise InferenceError("iterations must be a positive integer")
        tensor = self._validate_input(tensor)
        with torch.no_grad():
            for _ in range(warmup):
                self._forward_checked(tensor)
            self._synchronize()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            samples = []
            for _ in range(iterations):
                self._synchronize()
                started = perf_counter()
                self._forward_checked(tensor)
                self._synchronize()
                samples.append((perf_counter() - started) * 1000.0)
        allocated, reserved = self._peak_memory()
        return BenchmarkStats(
            warmup,
            iterations,
            float(np.median(samples)),
            float(np.percentile(samples, 95)),
            allocated,
            reserved,
        )
