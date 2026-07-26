# Task 10 report — trusted-checkpoint acceptance gate

## Scope delivered

- Added `tests/integration/test_official_checkpoint.py`, marked `checkpoint` and `integration`.
- The gate consumes `MOIREDET_CHECKPOINT` and `MOIREDET_CHECKPOINT_MANIFEST`; `MOIREDET_USER_IMAGE` controls only the optional user-image case. Once configured, artifacts are validated and invalid paths, manifests, hashes, or state dicts fail rather than silently skip.
- With trusted inputs, the suite requires strict CPU loading, RTX 4060 CUDA, the official sample's four-artifact CLI bundle, finite nonconstant raw prediction, decoded PNG/content/metadata validation, repeatability, and a fixed 5 warm-up / 20 measurement benchmark.
- Rewrote the README in clean UTF-8 Chinese. It retains the code/no-weight-complete status, documents the later commands and visual-check checklist, and states that trusted-checkpoint/RTX formal acceptance remains incomplete.

## RED/SKIP evidence

Before any implementation-side change, all three checkpoint environment variables were explicitly removed and the new focused suite was run:

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests\integration\test_official_checkpoint.py -v -rs
```

Result: 4 collected, 4 `SKIPPED`, 0 passed and 0 xfailed. Every case reported the useful prerequisite reason: `trusted checkpoint gate requires both MOIREDET_CHECKPOINT and MOIREDET_CHECKPOINT_MANIFEST`. No output directory was published.

## Verification

- Focused gate after documentation changes: 4 skipped, 0 passed/xfailed; same explicit missing-checkpoint reason.
- Marker suite with the environment cleared: `pytest -m checkpoint -v -rs` selected exactly 4 tests; 4 skipped and 70 deselected.
- Full suite with the environment cleared: 70 passed, 4 skipped, 3 existing dependency/upstream deprecation warnings, 13.64 s.
- `git diff --check` completed without whitespace errors before commit.

## Remaining external blocker

No trusted checkpoint or provenance manifest was supplied, and no weight was downloaded, searched for, loaded, or fabricated. Therefore strict CPU loading, CUDA output generation, user-image inference, visual inspection, benchmark measurements, and RTX 4060 formal acceptance are deliberately not claimed. The later operator commands and visual checklist are in `README.md`.
