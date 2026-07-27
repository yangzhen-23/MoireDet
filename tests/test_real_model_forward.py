import pytest
import torch

from moiredet_repro.config import load_config
from moiredet_repro.inference import MoireDetInference, set_determinism
from moiredet_repro.upstream_adapter import build_official_model


@pytest.mark.integration
@pytest.mark.slow
def test_actual_model_cpu_random_forward_contract():
    set_determinism()
    service = MoireDetInference(build_official_model(load_config()), torch.device("cpu"), None)
    result = service.predict_tensor(torch.randn(1, 3, 320, 320))
    assert result.shape == (320, 320)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.gpu
def test_actual_model_cuda_random_forward_contract():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    set_determinism()
    service = MoireDetInference(build_official_model(load_config()), torch.device("cuda:0"), None)
    result = service.predict_tensor(torch.randn(1, 3, 320, 320, device="cuda:0"))
    assert result.shape == (320, 320)
