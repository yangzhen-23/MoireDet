# Task 9 report — diagnostics, benchmark entry point and handoff docs

## Scope delivered

- Added `scripts/verify_environment.py`: no checkpoint read/deserialization/network operation; validates pinned distribution versions, reports the distinct `cv2` module version, runs real CPU/CUDA random forwards against the official model, and keeps stdout JSON-only.
- Added `scripts/benchmark_inference.py`: trusted-checkpoint-only CUDA entry point with the fixed 5 warm-up / 20 timed forwards contract and no overwrite of an existing benchmark JSON.
- Replaced the repository README with Chinese Windows/RTX operation, provenance, fixed output, troubleshooting, and status-boundary documentation; added a ready-to-send Chinese checkpoint request.
- Added ignored-directory placeholders and aligned the intentional all-zero example manifest with the expected official filename.

## TDD evidence

Initial RED: `D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests\test_scripts.py -v` reported three failures because both script entry points were absent. A later RED test verified that upstream import/forward noise leaked to stdout. GREEN: `tests/test_scripts.py` has 4 passed after script implementation and stdout suppression.

## Verification

- Focused scripts: 4 passed (`tests/test_scripts.py`).
- Weight-independent suite: 68 passed, 3 upstream/dependency deprecation warnings, 10.66 s.
- Full suite: 68 passed, 3 upstream/dependency deprecation warnings, 10.36 s.
- Real `verify_environment.py`: CPU 2.8 s and CUDA 5.5 s; `--device all` 6.6 s. Both forward shapes were `[320, 320]` on `NVIDIA GeForce RTX 4060 Laptop GPU`. Versions matched: Python 3.8.20, torch 1.10.0+cu113, torchvision 0.11.1+cu113, CUDA 11.3, opencv-python distribution 4.11.0.86 / `cv2` 4.11.0, NumPy 1.24.3, PyYAML 6.0.2, einops 0.3.0 and local-attention 1.2.1.

## Remaining boundary

No trusted checkpoint was supplied. The diagnostic explicitly reports `checkpoint_integration: not_run`, `checkpoint_deserialization: not_attempted`, and `network_access: not_attempted`; trusted checkpoint integration and RTX formal acceptance remain SKIPPED.
