import cv2
import numpy as np
import pytest
import torch

from moiredet_repro.config import load_config
from moiredet_repro.errors import InputImageError
from moiredet_repro.preprocessing import load_bgr_image, prepare_image


def write_encoded(path, array):
    ok, encoded = cv2.imencode(path.suffix, array)
    assert ok
    encoded.tofile(str(path))


def test_preprocess_preserves_bgr_and_original_size(tmp_path):
    path = tmp_path / "中文摩尔纹.png"
    original = np.full((5, 9, 3), (10, 20, 30), dtype=np.uint8)
    write_encoded(path, original)

    prepared = prepare_image(path, load_config(), torch.device("cpu"))

    assert prepared.original_bgr.shape == (5, 9, 3)
    assert (prepared.width, prepared.height) == (9, 5)
    assert tuple(prepared.tensor.shape) == (1, 3, 320, 320)
    expected = torch.tensor(
        [
            (10 / 255 - 0.485) / 0.229,
            (20 / 255 - 0.456) / 0.224,
            (30 / 255 - 0.406) / 0.225,
        ]
    )
    assert torch.allclose(prepared.tensor[0, :, 100, 100], expected, atol=1e-6)


@pytest.mark.parametrize("kind", ["missing", "corrupt", "gray", "rgba"])
def test_invalid_input_is_rejected_before_model_use(tmp_path, kind):
    path = tmp_path / (kind + ".png")
    if kind == "corrupt":
        path.write_bytes(b"not an image")
    elif kind == "gray":
        write_encoded(path, np.zeros((4, 4), dtype=np.uint8))
    elif kind == "rgba":
        write_encoded(path, np.zeros((4, 4, 4), dtype=np.uint8))

    pattern = kind if kind in ("missing", "corrupt") else "3-channel"
    with pytest.raises(InputImageError, match=pattern):
        load_bgr_image(path)
