import numpy as np
import pytest
import torch

from moiredet_repro.errors import DeviceError, InferenceError
from moiredet_repro.inference import MoireDetInference, select_device, unwrap_official_output


class ContractModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_training = None
        self.seen_grad = None

    def forward(self, tensor):
        self.seen_training = self.training
        self.seen_grad = torch.is_grad_enabled()
        density = torch.ones((1, 1, 320, 320), device=tensor.device)
        return [density], density.mean() * 0


def test_auto_device_prefers_cuda_and_explicit_cuda_does_not_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert str(select_device("auto")) == "cuda:0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(select_device("auto")) == "cpu"
    with pytest.raises(DeviceError, match="CUDA was requested"):
        select_device("cuda")


def test_unwrap_accepts_only_official_nested_contract():
    tensor = torch.zeros((1, 1, 320, 320))
    assert unwrap_official_output(([tensor], tensor.mean() * 0)).shape == (1, 1, 320, 320)
    for malformed in (
        tensor,
        ([tensor, tensor], tensor.mean()),
        ([torch.zeros(2, 1, 320, 320)], tensor.mean()),
    ):
        with pytest.raises(InferenceError, match="official output contract"):
            unwrap_official_output(malformed)


def test_predict_sets_eval_disables_grad_and_returns_independent_float32_array():
    model = ContractModel()
    service = MoireDetInference(model=model, device=torch.device("cpu"), checkpoint_info=None)
    prediction = service.predict_tensor(torch.zeros(1, 3, 320, 320))
    prediction[0, 0] = 9.0

    assert model.seen_training is False and model.seen_grad is False
    assert prediction.shape == (320, 320) and prediction.dtype == np.float32
    assert prediction[1, 1] == 1.0


def test_predict_rejects_invalid_input_shape_dtype_and_nonfinite_result():
    service = MoireDetInference(ContractModel(), torch.device("cpu"), None)
    with pytest.raises(InferenceError, match="shape"):
        service.predict_tensor(torch.zeros(3, 320, 320))
    with pytest.raises(InferenceError, match="floating point"):
        service.predict_tensor(torch.zeros(1, 3, 320, 320, dtype=torch.uint8))

    class BadModel(ContractModel):
        def forward(self, tensor):
            density = torch.full((1, 1, 320, 320), float("nan"), device=tensor.device)
            return [density], density.mean() * 0

    with pytest.raises(InferenceError, match="NaN or Inf"):
        MoireDetInference(BadModel(), torch.device("cpu"), None).predict_tensor(
            torch.zeros(1, 3, 320, 320)
        )


def test_benchmark_rejects_nonpositive_counts_and_reports_requested_counts():
    service = MoireDetInference(ContractModel(), torch.device("cpu"), None)
    tensor = torch.zeros(1, 3, 320, 320)
    with pytest.raises(InferenceError, match="warmup"):
        service.benchmark_tensor(tensor, warmup=-1, iterations=1)
    with pytest.raises(InferenceError, match="iterations"):
        service.benchmark_tensor(tensor, warmup=0, iterations=0)

    stats = service.benchmark_tensor(tensor, warmup=0, iterations=2)
    assert (stats.warmup, stats.iterations) == (0, 2)
    assert stats.median_ms >= 0 and stats.p95_ms >= stats.median_ms
