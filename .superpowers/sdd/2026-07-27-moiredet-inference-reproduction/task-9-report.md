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

## Fix Round 1 — provenance and no-load/no-network traps

- Documented the exact author-provided original Google Drive URL for `PSENet_100_loss0.000000.pth` from pinned upstream `MoireDet/script/model_download.txt` in README, upstream provenance, and checkpoint guidance. All three explicitly state that it is unavailable in this environment and has not been downloaded or validated.
- Added a CPU real-forward trap test which replaces `torch.load`, `torch.hub.load`, `torch.hub.load_state_dict_from_url`, `torch.utils.model_zoo.load_url`, and `urllib.request.urlopen` with failing sentinels. RED failed before adding the diagnostic safety status; GREEN executed the official CPU model forward with zero sentinel calls and reported `not_run` / `not_attempted` states plus `safety_guards`.
- Focused `tests/test_scripts.py`: 5 passed. Full suite: 69 passed, 3 pre-existing upstream/dependency deprecation warnings, 12.36 s. Fresh `verify_environment.py --device all`: 7.11 s, CPU and CUDA shapes `[320, 320]`, versions matched, RTX 4060 available, and no checkpoint/network attempt reported.

## Fix Round 2 — clean-process cached-alias boundary

- Added a subprocess regression that starts a fresh Python 3.8 process, installs sentinels before importing verifier/project model code, then runs the real CPU official-model forward. It blocks `torch.load`, `torch.hub.load`, `torch.hub.load_state_dict_from_url`, `torch.utils.model_zoo.load_url`, and `urllib.request.urlopen`.
- The initial probe RED exposed that `torch.utils.model_zoo` is not attached by importing `torch` alone in a clean process. Importing `torch.utils.model_zoo` explicitly before installing the sentinel made the probe portable for this Python/Torch environment. GREEN verifies zero calls, `[320, 320]`, `not_attempted` statuses, and identity of both `lib.models.modules.resnet.load_state_dict_from_url` and `resnet_dct.load_state_dict_from_url` with the sentinel after import.
- Focused `tests/test_scripts.py`: 6 passed. Full suite: 70 passed, 3 pre-existing upstream/dependency deprecation warnings, 14.44 s. Fresh `verify_environment.py --device all`: 6.78 s; CPU/CUDA shapes `[320, 320]`, versions matched, RTX 4060 available, no checkpoint/network attempt.
