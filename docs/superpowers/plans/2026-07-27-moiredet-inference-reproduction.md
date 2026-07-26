# MoireDet 单图推理复现 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Windows + RTX 4060 Laptop GPU 上建立可审计、可复用的官方 MoireDet 单图推理工程；没有可信权重时完成全部无权重验证，有可信权重后完成正式端到端验收。

**Architecture:** 用户仓库 `main` 已固定在作者提交，以两个可追踪补丁解决 torchvision 旧 API 和隐式 ImageNet 下载问题；所有业务逻辑放在独立的 `src/moiredet_repro` 兼容层中。CLI 依次调用配置、输入预处理、可信检查点、官方模型适配、推理、确定性渲染和元数据输出，核心推理对象只加载一次模型，可直接供后续视频逐帧复用。

**Tech Stack:** Python 3.8.20、PyTorch 1.10.0+cu113、torchvision 0.11.1+cu113、CUDA 11.3、OpenCV 4.11.0.86、NumPy 1.24.3、PyYAML 6.0.2、einops 0.3.0、local-attention 1.2.1、pytest 7.4.4、setuptools 68.2.2。

## Global Constraints

- 上游仓库固定为 `https://github.com/cong-yang/MoireDet`，提交固定为 `afde899f3c3beee96160610ee450618136a38f7b`。
- 目标类只能是 `lib/models/model.py` 中的 `TripleBranchWithSpecificConv`，参数只能是 `backbone=resnet18`、`fpem_repeat=2`、`pretrained=true`、`segmentation_head=FPEM_FFM`、`is_dct=false`、`is_light=true`；不得显式传入拼写为 `ouput_channel` 的内部参数。
- 保留作者 Performer 的公开张量布局行为并强制 batch size 为 `1`；不得在无作者基准的情况下自行转置“修正”。
- 输入严格使用 OpenCV BGR、直接双线性缩放到 `320 x 320`、ImageNet mean `[0.485, 0.456, 0.406]`、std `[0.229, 0.224, 0.225]`。
- 模型输出严格接受 `([moire_density], fea_loss)`，原始预测必须是 `1 x 1 x 320 x 320`；正式保存的 `prediction.npy` 为挤掉 batch/channel 后的 `320 x 320 float32`。
- 构建和加载过程不得发起隐式网络请求，不使用 AMP、模型编译、并发或 batch size 大于 `1`。
- 正式权重只接受 `official_repository`、`author_direct`、`author_team_direct`、`advisor_direct` 或 `verified_mirror` 五种来源；可信来源证据和 SHA-256 完整性必须同时通过。
- 检查点只接受顶层含 `state_dict` 的作者封装格式，只能去除键首部的 `module.`，并以 `strict=True` 加载。
- 输出目录已有 `prediction.npy`、`moire_map.png`、`comparison.png` 或 `run.json` 中任一文件时必须失败，不得覆盖。
- 没有可信 `PSENet_100_loss0.000000.pth` 时，官方检查点测试必须显示为 `SKIPPED`，项目状态只能写“代码兼容层和无权重验证完成”。
- 上游仓库没有明确代码许可证；README 必须标注来源，不得声明整项目采用 MIT。
- 所有功能按失败测试 → 最小实现 → 通过测试 → 小提交的顺序完成；不得把权重、输出目录、个人图像或 Conda 环境提交到 Git。

---

## File Map

- `pyproject.toml`：`src/` 打包、Python 版本、pytest marker 与测试路径。
- `environment.yml`：可重建的 Python/CUDA 依赖版本；不修改原 `exp` 环境。
- `.gitignore`：忽略权重、输出、缓存、可编辑安装元数据和个人素材。
- `configs/inference.yaml`：唯一的官方模型、预处理和显示参数配置。
- `MoireDet/`：远端 `main` 已有的作者源码，包括样例图和内置 Performer；只做三行兼容修改。
- `docs/upstream/UPSTREAM.md`：URL、提交、远端基线、导入根、补丁与文件哈希。
- `patches/0001-torchvision-load-state-dict-compat.patch`：两处 `load_state_dict_from_url` 兼容导入。
- `patches/0002-disable-resnet-online-download.patch`：只关闭目标类注意力分支的在线预训练初始化。
- `src/moiredet_repro/errors.py`：面向用户的领域异常。
- `src/moiredet_repro/config.py`：严格读取和验证固定推理配置。
- `src/moiredet_repro/upstream_adapter.py`：解析两条上游导入根并构建唯一目标模型。
- `src/moiredet_repro/preprocessing.py`：Unicode 路径安全的 BGR 读取与官方兼容张量预处理。
- `src/moiredet_repro/checkpoint.py`：来源清单、哈希验证、封装解析和严格状态字典加载。
- `src/moiredet_repro/inference.py`：设备选择、确定性设置、官方输出解包、单图前向和 GPU 基准。
- `src/moiredet_repro/rendering.py`：显示归一化、原尺寸恢复、对比图和四项输出写入。
- `src/moiredet_repro/metadata.py`：固定的 `run.json` schema 与运行环境采集。
- `src/moiredet_repro/cli.py`：`python -m moiredet_repro.cli infer ...` 的编排和退出码。
- `scripts/verify_environment.py`：CUDA 张量、真实模型 CPU/CUDA 随机前向的诊断入口。
- `scripts/benchmark_inference.py`：固定 5 次预热、20 次计时的验收入口。
- `weights/checkpoint.example.json`、`weights/README.md`：权重来源清单模板和放置说明。
- `docs/checkpoint-request-message.md`：权重仍不可用时可直接发给导师的中文索取消息。
- `tests/`：无权重基线测试、GPU 测试和受环境变量门控的可信权重集成测试。

## Fixed Data Contracts

来源清单 schema 固定为：

```json
{
  "schema_version": 1,
  "filename": "PSENet_100_loss0.000000.pth",
  "source_type": "official_repository",
  "source_reference": "https://drive.google.com/file/d/1QivNnHWaomJmUuBgueGzwtVooijsc_TH/view?usp=sharing",
  "retrieved_at": "2026-07-27",
  "provenance_evidence": "MoireDet/script/model_download.txt at upstream commit afde899f3c3beee96160610ee450618136a38f7b",
  "expected_sha256": "0000000000000000000000000000000000000000000000000000000000000000"
}
```

模板中的全零哈希是故意不可通过真实权重校验的哨兵值；取得可信文件后必须替换为该字节副本的真实 SHA-256，并把 `retrieved_at` 改成实际取得日期。

`run.json` 顶层和嵌套键固定为：

```json
{
  "schema_version": 1,
  "created_at_utc": "2026-07-27T08:00:00Z",
  "input": {"path": "D:\\datasets\\moire\\sample.png", "width": 1920, "height": 1080, "channel_order": "BGR"},
  "model": {
    "name": "TripleBranchWithSpecificConv",
    "upstream_repository": "https://github.com/cong-yang/MoireDet",
    "upstream_commit": "afde899f3c3beee96160610ee450618136a38f7b",
    "checkpoint": {
      "path": "D:\\models\\PSENet_100_loss0.000000.pth",
      "filename": "PSENet_100_loss0.000000.pth",
      "size_bytes": 123456789,
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "source_type": "official_repository",
      "source_reference": "https://drive.google.com/file/d/1QivNnHWaomJmUuBgueGzwtVooijsc_TH/view?usp=sharing",
      "retrieved_at": "2026-07-27",
      "provenance_evidence": "MoireDet/script/model_download.txt at pinned upstream commit",
      "checkpoint_verified": true
    }
  },
  "runtime": {
    "device_requested": "auto",
    "device_resolved": "cuda:0",
    "python": "3.8.20",
    "torch": "1.10.0+cu113",
    "torchvision": "0.11.1+cu113",
    "cuda_runtime": "11.3",
    "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU"
  },
  "preprocessing": {
    "input_size": [320, 320],
    "channel_order": "BGR",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "prediction": {"shape": [320, 320], "dtype": "float32", "min": -0.12, "max": 1.34, "dynamic_range": 1.46},
  "performance": {"forward_ms": 42.5, "peak_memory_allocated_bytes": 1073741824, "peak_memory_reserved_bytes": 1342177280},
  "outputs": {"prediction": "prediction.npy", "moire_map": "moire_map.png", "comparison": "comparison.png", "run": "run.json"}
}
```

CPU 运行时 `cuda_runtime`、`gpu_name` 和两项峰值显存允许为 JSON `null`；其余键不得缺失。

---

### Task 1: Bootstrap the Isolated Environment and Python Package

**Files:**
- Create: `pyproject.toml`
- Create: `environment.yml`
- Create: `.gitignore`
- Create: `src/moiredet_repro/__init__.py`
- Create: `src/moiredet_repro/errors.py`
- Create: `tests/test_package.py`

**Interfaces:**
- Consumes: Existing Conda environment `exp` with Python 3.8.20, torch 1.10.0+cu113 and torchvision 0.11.1+cu113.
- Produces: Importable `moiredet_repro` package, `MoireDetReproError` hierarchy, registered pytest markers, isolated `moiredet-repro` environment.

- [ ] **Step 1: Clone `exp` without modifying it, then install only the pinned missing/build/test packages**

```powershell
D:\anaconda3\Scripts\conda.exe create --name moiredet-repro --clone exp -y
D:\anaconda3\envs\moiredet-repro\python.exe -m pip install --no-deps einops==0.3.0 local-attention==1.2.1
D:\anaconda3\envs\moiredet-repro\python.exe -m pip install setuptools==68.2.2 pytest==7.4.4
D:\anaconda3\envs\moiredet-repro\python.exe -m pip check
```

Expected: the source `exp` remains unchanged; the new interpreter is `D:\anaconda3\envs\moiredet-repro\python.exe`; `pip check` reports no broken requirements and torch/torchvision remain unchanged.

- [ ] **Step 2: Write the package smoke test**

```python
# tests/test_package.py
from moiredet_repro import __version__
from moiredet_repro.errors import CheckpointError, InputImageError, OutputError, UpstreamError


def test_package_exports_version_and_domain_errors():
    assert __version__ == "0.1.0"
    for error_type in (CheckpointError, InputImageError, OutputError, UpstreamError):
        assert issubclass(error_type, RuntimeError)
```

- [ ] **Step 3: Run the test to verify the package is not yet importable**

Run: `D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_package.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'moiredet_repro'`.

- [ ] **Step 4: Add package metadata, environment lock, ignore rules and error types**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools==68.2.2", "wheel==0.44.0"]
build-backend = "setuptools.build_meta"

[project]
name = "moiredet-repro"
version = "0.1.0"
description = "Auditable single-image inference wrapper for the official MoireDet implementation"
requires-python = ">=3.8,<3.9"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "gpu: requires a working CUDA device",
  "checkpoint: requires a trusted official MoireDet checkpoint",
  "integration: crosses package and upstream boundaries",
  "slow: takes materially longer than a unit test",
  "benchmark: performs the fixed latency and memory benchmark"
]
```

```yaml
# environment.yml
name: moiredet-repro
channels:
  - defaults
dependencies:
  - python=3.8.20
  - pip=24.2
  - pip:
      - --extra-index-url https://download.pytorch.org/whl/cu113
      - torch==1.10.0+cu113
      - torchvision==0.11.1+cu113
      - numpy==1.24.3
      - opencv-python==4.11.0.86
      - PyYAML==6.0.2
      - tqdm==4.67.1
      - einops==0.3.0
      - local-attention==1.2.1
      - pytest==7.4.4
      - setuptools==68.2.2
      - wheel==0.44.0
```

```gitignore
# .gitignore
__pycache__/
*.py[cod]
.pytest_cache/
*.egg-info/
.coverage
.idea/
.vscode/
weights/*
!weights/README.md
!weights/checkpoint.example.json
outputs/*
!outputs/.gitkeep
examples/input/*
!examples/input/.gitkeep
```

```python
# src/moiredet_repro/__init__.py
__version__ = "0.1.0"
```

```python
# src/moiredet_repro/errors.py
class MoireDetReproError(RuntimeError):
    """Base class for expected, user-actionable reproduction failures."""


class ConfigurationError(MoireDetReproError):
    pass


class UpstreamError(MoireDetReproError):
    pass


class InputImageError(MoireDetReproError):
    pass


class CheckpointError(MoireDetReproError):
    pass


class DeviceError(MoireDetReproError):
    pass


class InferenceError(MoireDetReproError):
    pass


class OutputError(MoireDetReproError):
    pass
```

- [ ] **Step 5: Install the local package without dependency resolution and rerun the smoke test**

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe -m pip install -e . --no-deps --no-build-isolation
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_package.py -v
```

Expected: PASS; `pip show torch torchvision` still reports `1.10.0+cu113` and `0.11.1+cu113`.

- [ ] **Step 6: Commit the independently usable project bootstrap**

```powershell
git add pyproject.toml environment.yml .gitignore src/moiredet_repro/__init__.py src/moiredet_repro/errors.py tests/test_package.py
git commit -m "build: bootstrap isolated MoireDet reproduction package"
```

### Task 2: Patch and Audit the Existing Official Upstream Baseline

**Files:**
- Modify: `MoireDet/lib/models/model.py`
- Modify: `MoireDet/lib/models/modules/resnet.py`
- Modify: `MoireDet/lib/models/modules/resnet_dct.py`
- Create: `docs/upstream/UPSTREAM.md`
- Create: `patches/0001-torchvision-load-state-dict-compat.patch`
- Create: `patches/0002-disable-resnet-online-download.patch`
- Create: `tests/test_upstream_snapshot.py`

**Interfaces:**
- Consumes: user repository `main`, which equals `cong-yang/MoireDet` at the pinned commit.
- Produces: patched but auditable existing tree with import roots `MoireDet` and `MoireDet/script`; official sample at `MoireDet/script/00002423.png`.

- [ ] **Step 1: Write the upstream provenance and patch contract test before modifying source**

```python
# tests/test_upstream_snapshot.py
from pathlib import Path


PINNED = "afde899f3c3beee96160610ee450618136a38f7b"


def test_upstream_baseline_and_sample_are_pinned():
    root = Path(__file__).resolve().parents[1]
    provenance = (root / "docs" / "upstream" / "UPSTREAM.md").read_text(encoding="utf-8")
    assert "https://github.com/cong-yang/MoireDet" in provenance
    assert PINNED in provenance
    assert (root / "MoireDet" / "script" / "00002423.png").is_file()
    assert (root / "MoireDet" / "script" / "performer_pytorch" / "__init__.py").is_file()


def test_minimal_runtime_patches_are_applied_and_documented():
    root = Path(__file__).resolve().parents[1]
    model = (root / "MoireDet" / "lib" / "models" / "model.py").read_text(encoding="utf-8")
    resnet = (root / "MoireDet" / "lib" / "models" / "modules" / "resnet.py").read_text(encoding="utf-8")
    resnet_dct = (root / "MoireDet" / "lib" / "models" / "modules" / "resnet_dct.py").read_text(encoding="utf-8")
    target_start = model.index("class TripleBranchWithSpecificConv(nn.Module):")
    next_class = model.index("class TripleBranchWithSpecificConvNoPer", target_start)
    assert "backbone_model(pretrained=False)" in model[target_start:next_class]
    for source in (resnet, resnet_dct):
        assert "from torch.hub import load_state_dict_from_url" in source
    provenance = (root / "docs" / "upstream" / "UPSTREAM.md").read_text(encoding="utf-8")
    assert "0001-torchvision-load-state-dict-compat.patch" in provenance
    assert "0002-disable-resnet-online-download.patch" in provenance
```

- [ ] **Step 2: Run the tests and verify provenance/patches are absent**

Run: `D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_upstream_snapshot.py -v`

Expected: FAIL with `FileNotFoundError` for `docs/upstream/UPSTREAM.md`; the existing sample/source assertions would otherwise pass.

- [ ] **Step 3: Verify the current branch is based on the exact audited upstream commit**

```powershell
git merge-base --is-ancestor afde899f3c3beee96160610ee450618136a38f7b HEAD
git rev-parse origin/main
git diff --exit-code afde899f3c3beee96160610ee450618136a38f7b -- MoireDet/script/00002423.png MoireDet/script/performer_pytorch
```

Expected: the first and third commands exit `0`; the second prints exactly `afde899f3c3beee96160610ee450618136a38f7b`. No source copy is created.

- [ ] **Step 4: Add the two exact compatibility patches and apply them once**

```diff
# patches/0001-torchvision-load-state-dict-compat.patch
diff --git a/MoireDet/lib/models/modules/resnet.py b/MoireDet/lib/models/modules/resnet.py
--- a/MoireDet/lib/models/modules/resnet.py
+++ b/MoireDet/lib/models/modules/resnet.py
@@ -5 +5,4 @@
-from torchvision.models.utils import load_state_dict_from_url
+try:
+    from torchvision.models.utils import load_state_dict_from_url
+except ModuleNotFoundError:
+    from torch.hub import load_state_dict_from_url
diff --git a/MoireDet/lib/models/modules/resnet_dct.py b/MoireDet/lib/models/modules/resnet_dct.py
--- a/MoireDet/lib/models/modules/resnet_dct.py
+++ b/MoireDet/lib/models/modules/resnet_dct.py
@@ -5 +5,4 @@
-from torchvision.models.utils import load_state_dict_from_url
+try:
+    from torchvision.models.utils import load_state_dict_from_url
+except ModuleNotFoundError:
+    from torch.hub import load_state_dict_from_url
```

```diff
# patches/0002-disable-resnet-online-download.patch
diff --git a/MoireDet/lib/models/model.py b/MoireDet/lib/models/model.py
--- a/MoireDet/lib/models/model.py
+++ b/MoireDet/lib/models/model.py
@@ -136 +136 @@
-        self.attention_backbone = backbone_model(pretrained=True)
+        self.attention_backbone = backbone_model(pretrained=False)
```

Run:

```powershell
git apply .\patches\0001-torchvision-load-state-dict-compat.patch
git apply .\patches\0002-disable-resnet-online-download.patch
```

Expected: each patch changes only the paths shown. Do not globally replace the eight other `pretrained=True` occurrences in model variants that are outside the target contract.

- [ ] **Step 5: Record exact provenance and post-patch hashes**

Create `docs/upstream/UPSTREAM.md` with the URL, pinned commit, the fact that `origin/main` exactly matched it on `2026-07-27`, the two runtime import roots, the two patch filenames, and the output of:

```powershell
Get-FileHash -Algorithm SHA256 .\MoireDet\lib\models\model.py
Get-FileHash -Algorithm SHA256 .\MoireDet\lib\models\modules\resnet.py
Get-FileHash -Algorithm SHA256 .\MoireDet\lib\models\modules\resnet_dct.py
```

Also record the unpatched `model.py` Git-object SHA-256 `4a573e2112691b153af1b8a9ef904d6adfc75754e5dd48e6bd3b440086e3307c`. State explicitly that the upstream repository showed no code license and that the patch does not change the target model's computation after strict checkpoint loading.

- [ ] **Step 6: Run the upstream tests and inspect the diff**

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_upstream_snapshot.py -v
git diff --check
```

Expected: PASS; relative to pinned `origin/main`, only the three documented source lines differ.

- [ ] **Step 7: Commit the auditable upstream boundary**

```powershell
git add MoireDet/lib/models/model.py MoireDet/lib/models/modules/resnet.py MoireDet/lib/models/modules/resnet_dct.py docs/upstream/UPSTREAM.md patches tests/test_upstream_snapshot.py
git commit -m "fix: patch pinned MoireDet runtime compatibility"
```

### Task 3: Freeze the Official Configuration and Model Adapter

**Files:**
- Create: `configs/inference.yaml`
- Create: `src/moiredet_repro/config.py`
- Create: `src/moiredet_repro/upstream_adapter.py`
- Create: `tests/test_config_and_adapter.py`

**Interfaces:**
- Consumes: patched repository-root `MoireDet/`, `ConfigurationError`, `UpstreamError`.
- Produces: `InferenceConfig`, `load_config(path: Optional[Path]) -> InferenceConfig`, `resolve_upstream(project_root: Optional[Path]) -> UpstreamPaths`, `build_official_model(config, model_class=None) -> torch.nn.Module`.

- [ ] **Step 1: Write failing tests for the exact model contract and cwd-independent import roots**

```python
# tests/test_config_and_adapter.py
from pathlib import Path
import pytest

from moiredet_repro.config import load_config
from moiredet_repro.upstream_adapter import build_official_model, resolve_upstream


def test_inference_config_matches_official_contract():
    config = load_config()
    assert config.model_name == "TripleBranchWithSpecificConv"
    assert config.model_args == {
        "backbone": "resnet18", "fpem_repeat": 2, "pretrained": True,
        "segmentation_head": "FPEM_FFM", "is_dct": False, "is_light": True,
    }
    assert config.input_size == (320, 320)
    assert config.channel_order == "BGR"
    assert config.mean == (0.485, 0.456, 0.406)
    assert config.std == (0.229, 0.224, 0.225)
    assert config.batch_size == 1


def test_upstream_paths_are_resolved_independently_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = resolve_upstream()
    assert paths.repo_root.name == "MoireDet"
    assert (paths.repo_root / "lib").is_dir()
    assert (paths.performer_root / "performer_pytorch" / "__init__.py").is_file()


def test_model_factory_passes_only_official_arguments():
    captured = {}
    class SpyModel:
        def __init__(self, args):
            captured.update(args)
    model = build_official_model(load_config(), model_class=SpyModel)
    assert isinstance(model, SpyModel)
    assert captured == load_config().model_args
    assert "ouput_channel" not in captured


@pytest.mark.integration
def test_actual_official_model_builds_without_download(monkeypatch):
    import torch.hub
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *a, **k: pytest.fail("network download attempted"))
    model = build_official_model(load_config())
    assert type(model).__name__ == "TripleBranchWithSpecificConv"
```

- [ ] **Step 2: Run the tests and verify configuration imports fail**

Run: `D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_config_and_adapter.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'moiredet_repro.config'`.

- [ ] **Step 3: Add the exact YAML contract**

```yaml
# configs/inference.yaml
model:
  name: TripleBranchWithSpecificConv
  args:
    backbone: resnet18
    fpem_repeat: 2
    pretrained: true
    segmentation_head: FPEM_FFM
    is_dct: false
    is_light: true
preprocessing:
  input_size: [320, 320]
  channel_order: BGR
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  batch_size: 1
rendering:
  constant_epsilon: 1.0e-12
integration:
  minimum_dynamic_range: 1.0e-8
```

- [ ] **Step 4: Implement strict config parsing and the two-root upstream adapter**

```python
# src/moiredet_repro/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import yaml
from .errors import ConfigurationError

OFFICIAL_ARGS = {
    "backbone": "resnet18", "fpem_repeat": 2, "pretrained": True,
    "segmentation_head": "FPEM_FFM", "is_dct": False, "is_light": True,
}

@dataclass(frozen=True)
class InferenceConfig:
    model_name: str
    model_args: Dict[str, Any]
    input_size: Tuple[int, int]
    channel_order: str
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    batch_size: int
    constant_epsilon: float
    minimum_dynamic_range: float

def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "inference.yaml"

def load_config(path: Optional[Path] = None) -> InferenceConfig:
    source = Path(path) if path is not None else default_config_path()
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        model, prep = data["model"], data["preprocessing"]
        config = InferenceConfig(
            model_name=str(model["name"]), model_args=dict(model["args"]),
            input_size=tuple(prep["input_size"]), channel_order=str(prep["channel_order"]),
            mean=tuple(float(v) for v in prep["mean"]), std=tuple(float(v) for v in prep["std"]),
            batch_size=int(prep["batch_size"]),
            constant_epsilon=float(data["rendering"]["constant_epsilon"]),
            minimum_dynamic_range=float(data["integration"]["minimum_dynamic_range"]),
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError("Invalid inference config {}: {}".format(source, exc)) from exc
    if config.model_name != "TripleBranchWithSpecificConv" or config.model_args != OFFICIAL_ARGS:
        raise ConfigurationError("Model contract differs from official sample_code.json")
    if (config.input_size, config.channel_order, config.batch_size) != ((320, 320), "BGR", 1):
        raise ConfigurationError("Preprocessing must remain BGR, 320x320, batch size 1")
    if config.mean != (0.485, 0.456, 0.406) or config.std != (0.229, 0.224, 0.225):
        raise ConfigurationError("ImageNet normalization constants differ from the official sample")
    return config
```

```python
# src/moiredet_repro/upstream_adapter.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type
import sys
from .config import InferenceConfig
from .errors import UpstreamError

@dataclass(frozen=True)
class UpstreamPaths:
    repo_root: Path
    performer_root: Path

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def resolve_upstream(root: Optional[Path] = None) -> UpstreamPaths:
    base = Path(root) if root is not None else project_root()
    repo = base / "MoireDet"
    performer = repo / "script"
    missing = [p for p in (repo / "lib", performer / "performer_pytorch") if not p.exists()]
    if missing:
        raise UpstreamError("Missing pinned upstream component(s): {}".format(", ".join(map(str, missing))))
    return UpstreamPaths(repo.resolve(), performer.resolve())

def activate_upstream_imports(paths: UpstreamPaths) -> None:
    for path in reversed((paths.repo_root, paths.performer_root)):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

def build_official_model(config: InferenceConfig, model_class: Optional[Type] = None):
    if model_class is None:
        paths = resolve_upstream()
        activate_upstream_imports(paths)
        from lib.models.model import TripleBranchWithSpecificConv
        model_class = TripleBranchWithSpecificConv
    return model_class(dict(config.model_args))
```

- [ ] **Step 5: Run configuration and real construction tests**

Run: `D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_config_and_adapter.py -v`

Expected: PASS with no network request; warnings from `torch.qr` are acceptable and must not be hidden.

- [ ] **Step 6: Commit the fixed model boundary**

```powershell
git add configs/inference.yaml src/moiredet_repro/config.py src/moiredet_repro/upstream_adapter.py tests/test_config_and_adapter.py
git commit -m "feat: add fixed official MoireDet model adapter"
```

### Task 4: Implement Unicode-Safe Official BGR Preprocessing

**Files:**
- Create: `src/moiredet_repro/preprocessing.py`
- Create: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: `InferenceConfig`, `InputImageError`, image path or in-memory BGR frame, target `torch.device`.
- Produces: `load_bgr_image(path) -> np.ndarray`, `preprocess_bgr(image, config, device) -> Tensor[1,3,320,320]`, `prepare_image(...) -> PreparedImage`.

- [ ] **Step 1: Write failing tests for BGR values, direct resize and invalid inputs**

```python
# tests/test_preprocessing.py
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
    expected = torch.tensor([(10 / 255 - .485) / .229,
                             (20 / 255 - .456) / .224,
                             (30 / 255 - .406) / .225])
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
    with pytest.raises(InputImageError, match=kind if kind in ("missing", "corrupt") else "3-channel"):
        load_bgr_image(path)
```

- [ ] **Step 2: Run the tests and verify the preprocessing module is absent**

Run: `D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_preprocessing.py -v`

Expected: FAIL during collection for missing `moiredet_repro.preprocessing`.

- [ ] **Step 3: Implement image reading and tensor preparation without RGB conversion**

```python
# src/moiredet_repro/preprocessing.py
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import cv2
import numpy as np
import torch
from .config import InferenceConfig
from .errors import InputImageError

@dataclass(frozen=True)
class PreparedImage:
    original_bgr: np.ndarray
    tensor: torch.Tensor
    width: int
    height: int

def load_bgr_image(path: Union[str, Path]) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise InputImageError("missing input image: {}".format(source))
    try:
        encoded = np.fromfile(str(source), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except (OSError, cv2.error, ValueError) as exc:
        raise InputImageError("corrupt or unreadable input image: {}".format(source)) from exc
    if image is None:
        raise InputImageError("corrupt or unreadable input image: {}".format(source))
    if image.ndim != 3 or image.shape[2] != 3:
        raise InputImageError("input must be a 3-channel BGR image; got shape {}".format(image.shape))
    return np.ascontiguousarray(image)

def preprocess_bgr(image: np.ndarray, config: InferenceConfig, device: torch.device) -> torch.Tensor:
    if image.ndim != 3 or image.shape[2] != 3:
        raise InputImageError("in-memory input must be a 3-channel BGR image")
    resized = cv2.resize(image, config.input_size, interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float().div_(255.0)
    mean = torch.tensor(config.mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(config.std, dtype=torch.float32).view(3, 1, 1)
    tensor = tensor.sub(mean).div(std).unsqueeze(0)
    if tuple(tensor.shape) != (1, 3, 320, 320) or not torch.isfinite(tensor).all():
        raise InputImageError("preprocessed tensor contract failed")
    return tensor.to(device)

def prepare_image(path: Union[str, Path], config: InferenceConfig, device: torch.device) -> PreparedImage:
    original = load_bgr_image(path)
    height, width = original.shape[:2]
    return PreparedImage(original, preprocess_bgr(original, config, device), width, height)
```

- [ ] **Step 4: Run the tests and commit**

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_preprocessing.py -v
git add src/moiredet_repro/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add official-compatible BGR preprocessing"
```

Expected: all preprocessing tests PASS, including the Unicode path case.

### Task 5: Enforce Trusted Checkpoint Provenance and Strict Loading

**Files:**
- Create: `src/moiredet_repro/checkpoint.py`
- Create: `weights/checkpoint.example.json`
- Create: `weights/README.md`
- Create: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: `.pth` path, optional sidecar path, target `torch.nn.Module`.
- Produces: `CheckpointManifest`, `CheckpointInfo`, `CheckpointBundle`, `default_manifest_path()`, `load_checkpoint_bundle()`, `load_weights_strict()`.

- [ ] **Step 1: Write failing tests for manifest, byte validation, wrapping and strict keys**

```python
# tests/test_checkpoint.py
import hashlib, json
from pathlib import Path
import pytest
import torch
from moiredet_repro.checkpoint import default_manifest_path, load_checkpoint_bundle, load_weights_strict
from moiredet_repro.errors import CheckpointError

def write_manifest(path, checkpoint, sha256):
    path.write_text(json.dumps({
        "schema_version": 1, "filename": checkpoint.name,
        "source_type": "advisor_direct", "source_reference": "advisor file transfer",
        "retrieved_at": "2026-07-27", "provenance_evidence": "advisor provided the named official checkpoint",
        "expected_sha256": sha256,
    }), encoding="utf-8")

def test_default_manifest_appends_json():
    assert default_manifest_path(Path("model.pth")) == Path("model.pth.json")

def test_matching_hash_loads_wrapped_state_dict_and_strips_prefix(tmp_path):
    checkpoint = tmp_path / "model.pth"
    torch.save({"state_dict": {"module.weight": torch.ones(1, 1), "module.bias": torch.zeros(1)}}, checkpoint)
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = Path(str(checkpoint) + ".json")
    write_manifest(manifest, checkpoint, sha)
    bundle = load_checkpoint_bundle(checkpoint, manifest)
    assert set(bundle.state_dict) == {"weight", "bias"}
    assert bundle.info.sha256 == sha and bundle.info.checkpoint_verified is True

@pytest.mark.parametrize("payload", [b"", b"<!doctype html><html>download failed</html>"])
def test_empty_or_html_is_rejected_before_torch_load(tmp_path, payload):
    checkpoint = tmp_path / "bad.pth"
    checkpoint.write_bytes(payload)
    manifest = Path(str(checkpoint) + ".json")
    write_manifest(manifest, checkpoint, hashlib.sha256(payload).hexdigest())
    with pytest.raises(CheckpointError, match="empty|HTML"):
        load_checkpoint_bundle(checkpoint, manifest)

def test_bare_state_dict_is_rejected(tmp_path):
    checkpoint = tmp_path / "bare.pth"
    torch.save({"weight": torch.ones(1)}, checkpoint)
    manifest = Path(str(checkpoint) + ".json")
    write_manifest(manifest, checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    with pytest.raises(CheckpointError, match="state_dict"):
        load_checkpoint_bundle(checkpoint, manifest)

def test_strict_load_reports_shape_mismatch(tmp_path):
    model = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "shape.pth"
    torch.save({"state_dict": {"weight": torch.ones(1, 3), "bias": torch.zeros(1)}}, checkpoint)
    manifest = Path(str(checkpoint) + ".json")
    write_manifest(manifest, checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    with pytest.raises(CheckpointError, match="shape mismatch.*weight"):
        load_weights_strict(model, load_checkpoint_bundle(checkpoint, manifest))
```

- [ ] **Step 2: Run tests and verify the checkpoint module is absent**

Run: `D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_checkpoint.py -v`

Expected: FAIL during collection for missing `moiredet_repro.checkpoint`.

- [ ] **Step 3: Implement manifest validation before deserialization and strict loading**

```python
# src/moiredet_repro/checkpoint.py
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Optional
import hashlib, json, re
import torch
from .errors import CheckpointError

ALLOWED_SOURCES = {"official_repository", "author_direct", "author_team_direct", "advisor_direct", "verified_mirror"}

@dataclass(frozen=True)
class CheckpointManifest:
    filename: str; source_type: str; source_reference: str
    retrieved_at: str; provenance_evidence: str; expected_sha256: str

@dataclass(frozen=True)
class CheckpointInfo:
    path: Path; size_bytes: int; sha256: str; manifest: CheckpointManifest
    checkpoint_verified: bool

@dataclass(frozen=True)
class CheckpointBundle:
    state_dict: Mapping[str, torch.Tensor]
    info: CheckpointInfo

def default_manifest_path(checkpoint: Path) -> Path:
    return Path(str(checkpoint) + ".json")

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _read_manifest(path: Path, checkpoint: Path) -> CheckpointManifest:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.pop("schema_version") != 1:
            raise ValueError("schema_version must be 1")
        manifest = CheckpointManifest(**data)
        date.fromisoformat(manifest.retrieved_at)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise CheckpointError("invalid checkpoint manifest {}: {}".format(path, exc)) from exc
    if manifest.filename != checkpoint.name:
        raise CheckpointError("manifest filename does not match checkpoint")
    if manifest.source_type not in ALLOWED_SOURCES:
        raise CheckpointError("untrusted source_type: {}".format(manifest.source_type))
    if not manifest.source_reference.strip() or not manifest.provenance_evidence.strip():
        raise CheckpointError("source_reference and provenance_evidence are required")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest.expected_sha256):
        raise CheckpointError("expected_sha256 must be 64 lowercase hexadecimal characters")
    return manifest

def load_checkpoint_bundle(checkpoint_path: Path, manifest_path: Optional[Path] = None) -> CheckpointBundle:
    checkpoint = Path(checkpoint_path)
    manifest_file = Path(manifest_path) if manifest_path is not None else default_manifest_path(checkpoint)
    if not checkpoint.is_file():
        raise CheckpointError("checkpoint file is missing: {}".format(checkpoint))
    manifest = _read_manifest(manifest_file, checkpoint)
    size = checkpoint.stat().st_size
    if size == 0:
        raise CheckpointError("checkpoint is empty")
    with checkpoint.open("rb") as handle:
        head = handle.read(512).lower()
    if b"<html" in head or b"<!doctype html" in head:
        raise CheckpointError("checkpoint is an HTML download page")
    actual = _sha256_file(checkpoint)
    if actual != manifest.expected_sha256:
        raise CheckpointError("SHA-256 mismatch: expected {}, got {}".format(manifest.expected_sha256, actual))
    try:
        payload = torch.load(str(checkpoint), map_location="cpu")
    except Exception as exc:
        raise CheckpointError("trusted checkpoint could not be deserialized: {}".format(exc)) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), Mapping):
        raise CheckpointError("official checkpoint must contain a top-level state_dict mapping")
    normalized = OrderedDict()
    for key, value in payload["state_dict"].items():
        new_key = key[7:] if key.startswith("module.") else key
        if new_key in normalized or not isinstance(value, torch.Tensor):
            raise CheckpointError("invalid or duplicate state_dict key: {}".format(new_key))
        normalized[new_key] = value
    info = CheckpointInfo(checkpoint.resolve(), size, actual, manifest, True)
    return CheckpointBundle(normalized, info)

def load_weights_strict(model: torch.nn.Module, bundle: CheckpointBundle) -> CheckpointInfo:
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(bundle.state_dict))
    unexpected = sorted(set(bundle.state_dict) - set(model_state))
    mismatched = [(k, tuple(bundle.state_dict[k].shape), tuple(model_state[k].shape))
                  for k in set(model_state) & set(bundle.state_dict)
                  if tuple(bundle.state_dict[k].shape) != tuple(model_state[k].shape)]
    if missing or unexpected or mismatched:
        raise CheckpointError("strict state_dict mismatch; missing={}; unexpected={}; shape mismatch={}".format(
            missing[:10], unexpected[:10], mismatched[:1]))
    model.load_state_dict(bundle.state_dict, strict=True)
    return bundle.info
```

- [ ] **Step 4: Add the deliberately invalid example manifest and usage rules**

Copy the fixed manifest example from this plan's **Fixed Data Contracts** section to `weights/checkpoint.example.json`. In `weights/README.md`, state that the all-zero hash cannot validate a real checkpoint, list the five allowed source types, warn that `torch.load` is pickle-based and therefore only trusted sources are permitted, and show that `model.pth` defaults to sidecar `model.pth.json`.

- [ ] **Step 5: Run tests and commit**

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_checkpoint.py -v
git add src/moiredet_repro/checkpoint.py tests/test_checkpoint.py weights/README.md weights/checkpoint.example.json
git commit -m "feat: verify trusted MoireDet checkpoints strictly"
```

Expected: all synthetic checkpoint tests PASS without accessing a network or official weight.

### Task 6: Build the Reusable Inference and Benchmark Service

**Files:**
- Create: `src/moiredet_repro/inference.py`
- Create: `tests/test_inference.py`
- Create: `tests/test_real_model_forward.py`

**Interfaces:**
- Consumes: fixed config, official model builder, trusted checkpoint bundle, preprocessed `1x3x320x320` tensor.
- Produces: `select_device()`, `set_determinism()`, `unwrap_official_output()`, `MoireDetInference.from_checkpoint()`, `predict_tensor() -> np.ndarray`, `last_performance`, `benchmark_tensor(warmup=5, iterations=20) -> BenchmarkStats`.

- [ ] **Step 1: Write failing unit tests for devices, nested output, eval/no-grad and OOM behavior**

```python
# tests/test_inference.py
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
    for malformed in (tensor, ([tensor, tensor], tensor.mean()), ([torch.zeros(2, 1, 320, 320)], tensor.mean())):
        with pytest.raises(InferenceError, match="official output contract"):
            unwrap_official_output(malformed)

def test_predict_sets_eval_disables_grad_and_returns_raw_float32():
    model = ContractModel()
    service = MoireDetInference(model=model, device=torch.device("cpu"), checkpoint_info=None)
    prediction = service.predict_tensor(torch.zeros(1, 3, 320, 320))
    assert model.seen_training is False and model.seen_grad is False
    assert prediction.shape == (320, 320) and prediction.dtype == np.float32
    assert np.all(prediction == 1.0)

def test_nonfinite_prediction_is_rejected():
    class BadModel(ContractModel):
        def forward(self, tensor):
            density = torch.full((1, 1, 320, 320), float("nan"))
            return [density], density.mean() * 0
    with pytest.raises(InferenceError, match="NaN or Inf"):
        MoireDetInference(BadModel(), torch.device("cpu"), None).predict_tensor(torch.zeros(1, 3, 320, 320))
```

- [ ] **Step 2: Run unit tests and verify the inference module is absent**

Run: `D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_inference.py -v`

Expected: FAIL during collection for missing `moiredet_repro.inference`.

- [ ] **Step 3: Implement deterministic device selection, official output extraction and service construction**

```python
# src/moiredet_repro/inference.py
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
    warmup: int; iterations: int; median_ms: float; p95_ms: float
    peak_memory_allocated_bytes: Optional[int]
    peak_memory_reserved_bytes: Optional[int]

def set_determinism(seed: int = 2) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def select_device(requested: str) -> torch.device:
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
    valid = isinstance(output, tuple) and len(output) == 2
    valid = valid and isinstance(output[0], list) and len(output[0]) == 1
    valid = valid and isinstance(output[0][0], torch.Tensor)
    if not valid or tuple(output[0][0].shape) != (1, 1, 320, 320):
        raise InferenceError("model did not satisfy the official output contract ([1x1x320x320], fea_loss)")
    return output[0][0]

class MoireDetInference:
    def __init__(self, model, device: torch.device, checkpoint_info: Optional[CheckpointInfo]):
        self.model = model.to(device).eval()
        self.device = device
        self.checkpoint_info = checkpoint_info
        self.last_performance = None

    @classmethod
    def from_checkpoint(cls, checkpoint_path, manifest_path, config: InferenceConfig, device: str = "auto"):
        resolved = select_device(device)
        model = build_official_model(config)
        info = load_weights_strict(model, load_checkpoint_bundle(checkpoint_path, manifest_path))
        return cls(model, resolved, info)

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def predict_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        if tuple(tensor.shape) != (1, 3, 320, 320):
            raise InferenceError("input tensor must have shape 1x3x320x320")
        tensor = tensor.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._synchronize(); started = perf_counter()
        try:
            with torch.no_grad():
                raw = unwrap_official_output(self.model(tensor))
            self._synchronize()
        except RuntimeError as exc:
            if self.device.type == "cuda" and "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise InferenceError("CUDA out of memory; no CPU fallback was attempted") from exc
            raise
        elapsed = (perf_counter() - started) * 1000.0
        prediction = raw[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(prediction).all():
            raise InferenceError("prediction contains NaN or Inf")
        allocated = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None
        reserved = torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else None
        self.last_performance = PerformanceStats(elapsed, allocated, reserved)
        return prediction

    def benchmark_tensor(self, tensor: torch.Tensor, warmup: int = 5, iterations: int = 20) -> BenchmarkStats:
        if (warmup, iterations) != (5, 20):
            raise InferenceError("acceptance benchmark requires warmup=5 and iterations=20")
        tensor = tensor.to(self.device)
        with torch.no_grad():
            for _ in range(warmup):
                unwrap_official_output(self.model(tensor))
            self._synchronize()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            samples = []
            for _ in range(iterations):
                self._synchronize(); started = perf_counter()
                unwrap_official_output(self.model(tensor)); self._synchronize()
                samples.append((perf_counter() - started) * 1000.0)
        return BenchmarkStats(warmup, iterations, float(np.median(samples)), float(np.percentile(samples, 95)),
            torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
            torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else None)
```

- [ ] **Step 4: Add real no-weight CPU and CUDA forward contract tests**

```python
# tests/test_real_model_forward.py
import pytest, torch
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
```

- [ ] **Step 5: Run fast, CPU-real and CUDA-real tests separately**

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_inference.py -v
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_real_model_forward.py -m "not gpu" -v
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_real_model_forward.py -m gpu -v
```

Expected: all three layers PASS on the target RTX 4060; neither real-forward test reads a checkpoint.

- [ ] **Step 6: Commit the reusable service**

```powershell
git add src/moiredet_repro/inference.py tests/test_inference.py tests/test_real_model_forward.py
git commit -m "feat: add reusable MoireDet inference service"
```

### Task 7: Render Deterministic Maps and Write the Fixed Output Bundle

**Files:**
- Create: `src/moiredet_repro/rendering.py`
- Create: `src/moiredet_repro/metadata.py`
- Create: `tests/test_rendering_and_metadata.py`

**Interfaces:**
- Consumes: original BGR image, raw `320x320 float32` prediction, config, checkpoint/performance/runtime data.
- Produces: `PredictionStats`, `preflight_output_dir()`, `render_moire_map()`, `make_comparison()`, `build_run_metadata()`, `write_output_bundle()`.

- [ ] **Step 1: Write failing tests for normalization, dimensions, raw preservation and no-overwrite**

```python
# tests/test_rendering_and_metadata.py
import json
import cv2
import numpy as np
import pytest
from moiredet_repro.errors import OutputError
from moiredet_repro.rendering import (make_comparison, normalize_for_display,
    preflight_output_dir, render_moire_map, write_output_bundle)

def test_minmax_rounding_and_constant_protection():
    pred = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
    image, stats = normalize_for_display(pred, epsilon=1e-12)
    assert image.dtype == np.uint8
    assert image.tolist() == [[0, 128], [191, 255]]
    assert stats.minimum == 0.0 and stats.maximum == 1.0 and stats.dynamic_range == 1.0
    constant, _ = render_moire_map(np.ones((320, 320), np.float32), 9, 5, 1e-12)
    assert constant.shape == (5, 9) and np.count_nonzero(constant) == 0

def test_map_and_comparison_restore_original_dimensions():
    original = np.full((5, 9, 3), 17, dtype=np.uint8)
    moire, _ = render_moire_map(np.arange(320 * 320, dtype=np.float32).reshape(320, 320), 9, 5, 1e-12)
    comparison = make_comparison(original, moire)
    assert moire.shape == (5, 9)
    assert comparison.shape == (5, 18, 3)
    assert np.array_equal(comparison[:, :9], original)
    assert np.array_equal(comparison[:, 9:, 0], moire)

@pytest.mark.parametrize("name", ["prediction.npy", "moire_map.png", "comparison.png", "run.json"])
def test_preflight_refuses_any_existing_target(tmp_path, name):
    old = tmp_path / name
    old.write_bytes(b"keep")
    with pytest.raises(OutputError, match="refusing to overwrite"):
        preflight_output_dir(tmp_path)
    assert old.read_bytes() == b"keep"

def test_bundle_preserves_raw_prediction_and_schema(tmp_path):
    prediction = np.linspace(-1, 1, 320 * 320, dtype=np.float32).reshape(320, 320)
    original = np.zeros((5, 9, 3), dtype=np.uint8)
    metadata = {"schema_version": 1, "outputs": {
        "prediction": "prediction.npy", "moire_map": "moire_map.png",
        "comparison": "comparison.png", "run": "run.json"}}
    write_output_bundle(tmp_path, original, prediction, metadata, 1e-12)
    assert np.array_equal(np.load(str(tmp_path / "prediction.npy")), prediction)
    assert cv2.imdecode(np.fromfile(str(tmp_path / "moire_map.png"), np.uint8), 0).shape == (5, 9)
    assert json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))["schema_version"] == 1
```

- [ ] **Step 2: Run the tests and verify rendering is absent**

Run: `D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_rendering_and_metadata.py -v`

Expected: FAIL during collection for missing `moiredet_repro.rendering`.

- [ ] **Step 3: Implement deterministic visualization and Unicode-safe writes**

```python
# src/moiredet_repro/rendering.py
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import json, cv2, numpy as np
from .errors import OutputError

TARGETS = ("prediction.npy", "moire_map.png", "comparison.png", "run.json")

@dataclass(frozen=True)
class PredictionStats:
    minimum: float; maximum: float; dynamic_range: float

def preflight_output_dir(path: Path) -> Path:
    output = Path(path)
    conflicts = [name for name in TARGETS if (output / name).exists()]
    if conflicts:
        raise OutputError("refusing to overwrite existing output(s): {}".format(", ".join(conflicts)))
    try:
        output.mkdir(parents=True, exist_ok=True)
        probe = output / ".write-probe"
        probe.write_bytes(b""); probe.unlink()
    except OSError as exc:
        raise OutputError("output directory is not writable: {}".format(output)) from exc
    return output

def normalize_for_display(prediction: np.ndarray, epsilon: float) -> Tuple[np.ndarray, PredictionStats]:
    if prediction.ndim != 2 or prediction.dtype != np.float32 or not np.isfinite(prediction).all():
        raise OutputError("prediction must be a finite 2D float32 array")
    minimum, maximum = float(prediction.min()), float(prediction.max())
    dynamic = maximum - minimum
    normalized = np.zeros_like(prediction) if dynamic <= epsilon else (prediction - minimum) * (255.0 / dynamic)
    rendered = np.rint(np.clip(normalized, 0.0, 255.0)).astype(np.uint8)
    return rendered, PredictionStats(minimum, maximum, dynamic)

def render_moire_map(prediction: np.ndarray, width: int, height: int, epsilon: float) -> Tuple[np.ndarray, PredictionStats]:
    if prediction.shape != (320, 320):
        raise OutputError("prediction must have shape 320x320")
    normalized, stats = normalize_for_display(prediction, epsilon)
    resized = cv2.resize(normalized.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return np.rint(np.clip(resized, 0.0, 255.0)).astype(np.uint8), stats

def make_comparison(original_bgr: np.ndarray, moire_map: np.ndarray) -> np.ndarray:
    if original_bgr.shape[:2] != moire_map.shape:
        raise OutputError("original and rendered map dimensions differ")
    return np.concatenate([original_bgr, cv2.cvtColor(moire_map, cv2.COLOR_GRAY2BGR)], axis=1)

def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OutputError("OpenCV could not encode {}".format(path.name))
    encoded.tofile(str(path))

def write_output_bundle(output_dir: Path, original_bgr: np.ndarray, prediction: np.ndarray,
                        metadata: Dict, epsilon: float) -> Dict[str, Path]:
    output = preflight_output_dir(output_dir)
    height, width = original_bgr.shape[:2]
    moire_map, stats = render_moire_map(prediction, width, height, epsilon)
    metadata["prediction"].update({"min": stats.minimum, "max": stats.maximum,
                                   "dynamic_range": stats.dynamic_range}) if "prediction" in metadata else None
    with (output / "prediction.npy").open("wb") as handle:
        np.save(handle, prediction, allow_pickle=False)
    _write_png(output / "moire_map.png", moire_map)
    _write_png(output / "comparison.png", make_comparison(original_bgr, moire_map))
    (output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {name: (output / name).resolve() for name in TARGETS}
```

- [ ] **Step 4: Implement the exact `run.json` builder**

```python
# src/moiredet_repro/metadata.py
from dataclasses import asdict
from datetime import datetime, timezone
import platform, torch, torchvision

UPSTREAM_URL = "https://github.com/cong-yang/MoireDet"
UPSTREAM_COMMIT = "afde899f3c3beee96160610ee450618136a38f7b"

def build_run_metadata(input_path, prepared, config, service, prediction, requested_device):
    info, perf = service.checkpoint_info, service.last_performance
    manifest = info.manifest
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": {"path": str(input_path.resolve()), "width": prepared.width,
                  "height": prepared.height, "channel_order": "BGR"},
        "model": {"name": config.model_name, "upstream_repository": UPSTREAM_URL,
                  "upstream_commit": UPSTREAM_COMMIT,
                  "checkpoint": {"path": str(info.path), "filename": manifest.filename,
                    "size_bytes": info.size_bytes, "sha256": info.sha256,
                    "source_type": manifest.source_type, "source_reference": manifest.source_reference,
                    "retrieved_at": manifest.retrieved_at, "provenance_evidence": manifest.provenance_evidence,
                    "checkpoint_verified": info.checkpoint_verified}},
        "runtime": {"device_requested": requested_device, "device_resolved": str(service.device),
                    "python": platform.python_version(), "torch": torch.__version__,
                    "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name(service.device) if service.device.type == "cuda" else None},
        "preprocessing": {"input_size": list(config.input_size), "channel_order": config.channel_order,
                          "mean": list(config.mean), "std": list(config.std)},
        "prediction": {"shape": list(prediction.shape), "dtype": str(prediction.dtype),
                       "min": None, "max": None, "dynamic_range": None},
        "performance": {"forward_ms": perf.forward_ms,
                        "peak_memory_allocated_bytes": perf.peak_memory_allocated_bytes,
                        "peak_memory_reserved_bytes": perf.peak_memory_reserved_bytes},
        "outputs": {"prediction": "prediction.npy", "moire_map": "moire_map.png",
                    "comparison": "comparison.png", "run": "run.json"},
    }
```

- [ ] **Step 5: Add metadata schema assertions, run tests and commit**

```python
from pathlib import Path
from types import SimpleNamespace
import torch
from moiredet_repro.checkpoint import CheckpointInfo, CheckpointManifest
from moiredet_repro.config import load_config
from moiredet_repro.inference import PerformanceStats
from moiredet_repro.metadata import build_run_metadata

def test_fixed_metadata_schema_has_every_required_key():
    manifest = CheckpointManifest(
        filename="model.pth", source_type="official_repository",
        source_reference="https://example.invalid/model.pth", retrieved_at="2026-07-27",
        provenance_evidence="synthetic unit-test fixture", expected_sha256="1" * 64)
    info = CheckpointInfo(Path("model.pth").resolve(), 123, "1" * 64, manifest, True)
    fake_prepared = SimpleNamespace(width=9, height=5)
    fake_service = SimpleNamespace(
        checkpoint_info=info, last_performance=PerformanceStats(1.25, None, None),
        device=torch.device("cpu"))
    prediction = np.zeros((320, 320), dtype=np.float32)
    metadata = build_run_metadata(Path("sample.png"), fake_prepared, load_config(),
                                  fake_service, prediction, "auto")
    assert set(metadata) == {"schema_version", "created_at_utc", "input", "model",
                             "runtime", "preprocessing", "prediction", "performance", "outputs"}
    assert set(metadata["model"]["checkpoint"]) == {"path", "filename", "size_bytes", "sha256",
        "source_type", "source_reference", "retrieved_at", "provenance_evidence", "checkpoint_verified"}
    assert metadata["prediction"]["shape"] == [320, 320]
    assert metadata["outputs"] == {"prediction": "prediction.npy", "moire_map": "moire_map.png",
        "comparison": "comparison.png", "run": "run.json"}
```

Then run:

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_rendering_and_metadata.py -v
git add src/moiredet_repro/rendering.py src/moiredet_repro/metadata.py tests/test_rendering_and_metadata.py
git commit -m "feat: write deterministic MoireDet output bundles"
```

Expected: PASS; `.npy` remains numerically identical to the input prediction, while PNGs use only display normalization.

### Task 8: Wire the One-Command CLI Without Overwriting Outputs

**Files:**
- Create: `src/moiredet_repro/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: all earlier public interfaces.
- Produces: `build_parser()`, `run_infer(args) -> Dict[str, Path]`, `main(argv=None) -> int`, module command `python -m moiredet_repro.cli infer ...`.

- [ ] **Step 1: Write failing parser, help and preflight-order tests**

```python
# tests/test_cli.py
from pathlib import Path
import subprocess, sys
import pytest
import moiredet_repro.cli as cli

def test_parser_defaults_to_auto_and_sidecar_manifest():
    args = cli.build_parser().parse_args(["infer", "--input", "a.png", "--checkpoint", "model.pth", "--output", "out"])
    assert args.device == "auto"
    assert args.checkpoint_manifest is None

def test_module_help_runs_from_repository_root():
    result = subprocess.run([sys.executable, "-m", "moiredet_repro.cli", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "infer" in result.stdout

def test_output_conflict_stops_before_model_build(tmp_path, monkeypatch):
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"unused because preflight must run first")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"unused")
    output = tmp_path / "out"; output.mkdir(); (output / "run.json").write_text("keep")
    called = []
    monkeypatch.setattr(cli.MoireDetInference, "from_checkpoint", lambda **kwargs: called.append(True))
    code = cli.main(["infer", "--input", str(input_path), "--checkpoint", str(checkpoint), "--output", str(output)])
    assert code == 2 and called == []
    assert (output / "run.json").read_text() == "keep"
```

- [ ] **Step 2: Run tests and verify the CLI module is absent**

Run: `D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_cli.py -v`

Expected: FAIL while importing `moiredet_repro.cli`.

- [ ] **Step 3: Implement the exact command and actionable exit codes**

```python
# src/moiredet_repro/cli.py
import argparse, sys
from pathlib import Path
from .config import load_config
from .errors import MoireDetReproError
from .inference import MoireDetInference, select_device, set_determinism
from .metadata import build_run_metadata
from .preprocessing import prepare_image
from .rendering import preflight_output_dir, write_output_bundle

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official MoireDet single-image reproduction")
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer = subparsers.add_parser("infer", help="run one trusted-checkpoint inference")
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--checkpoint-manifest", type=Path, default=None)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    infer.add_argument("--config", type=Path, default=None)
    return parser

def run_infer(args):
    output = preflight_output_dir(args.output)
    config = load_config(args.config)
    set_determinism(2)
    resolved_device = select_device(args.device)
    print("[1/5] reading and preprocessing {}".format(args.input.resolve()))
    prepared = prepare_image(args.input, config, resolved_device)
    print("[2/5] verifying checkpoint provenance and SHA-256")
    service = MoireDetInference.from_checkpoint(
        checkpoint_path=args.checkpoint, manifest_path=args.checkpoint_manifest,
        config=config, device=args.device)
    print("[3/5] running official TripleBranchWithSpecificConv")
    prediction = service.predict_tensor(prepared.tensor)
    print("[4/5] building run metadata")
    metadata = build_run_metadata(args.input, prepared, config, service, prediction, args.device)
    print("[5/5] writing output bundle")
    paths = write_output_bundle(output, prepared.original_bgr, prediction, metadata, config.constant_epsilon)
    for name, path in paths.items():
        print("{}: {}".format(name, path))
    return paths

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "infer":
            run_infer(args)
            return 0
    except MoireDetReproError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print("UNEXPECTED ERROR: {}".format(exc), file=sys.stderr)
        return 1
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add a fake-service end-to-end CLI test**

```python
def test_fake_service_end_to_end_writes_and_reports_all_outputs(tmp_path, monkeypatch, capsys):
    import numpy as np
    import torch
    from moiredet_repro.checkpoint import CheckpointInfo, CheckpointManifest
    from moiredet_repro.inference import PerformanceStats
    from moiredet_repro.preprocessing import PreparedImage

    input_path = tmp_path / "中文输入.png"
    input_path.write_bytes(b"decode is replaced by the prepared-image fixture")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint loading is replaced by the fake service")
    output = tmp_path / "out"
    manifest = CheckpointManifest(
        filename="model.pth", source_type="official_repository",
        source_reference="https://example.invalid/model.pth", retrieved_at="2026-07-27",
        provenance_evidence="synthetic CLI fixture", expected_sha256="2" * 64)

    class FakeService:
        device = torch.device("cpu")
        checkpoint_info = CheckpointInfo(checkpoint.resolve(), checkpoint.stat().st_size,
                                         "2" * 64, manifest, True)
        last_performance = PerformanceStats(1.25, None, None)
        def predict_tensor(self, tensor):
            return np.linspace(0, 1, 320 * 320, dtype=np.float32).reshape(320, 320)

    prepared = PreparedImage(
        np.zeros((5, 9, 3), dtype=np.uint8), torch.zeros(1, 3, 320, 320), 9, 5)
    monkeypatch.setattr(cli, "prepare_image", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(cli.MoireDetInference, "from_checkpoint", lambda **kwargs: FakeService())

    code = cli.main(["infer", "--input", str(input_path), "--checkpoint", str(checkpoint),
                     "--output", str(output), "--device", "cpu"])
    captured = capsys.readouterr()
    assert code == 0 and captured.err == ""
    for name in ("prediction.npy", "moire_map.png", "comparison.png", "run.json"):
        path = output / name
        assert path.is_file()
        assert str(path.resolve()) in captured.out
```

- [ ] **Step 5: Run CLI tests and the module help command**

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_cli.py -v
D:\anaconda3\envs\moiredet-repro\python.exe -m moiredet_repro.cli --help
```

Expected: tests PASS and help exits `0`. The conflict test proves output preflight happens before input decode and model construction.

- [ ] **Step 6: Commit the single-command workflow**

```powershell
git add src/moiredet_repro/cli.py tests/test_cli.py
git commit -m "feat: add MoireDet single-image CLI"
```

### Task 9: Add Environment Diagnostics, Benchmark Entry Point and Chinese Handoff Docs

**Files:**
- Create: `scripts/verify_environment.py`
- Create: `scripts/benchmark_inference.py`
- Create: `README.md`
- Create: `docs/checkpoint-request-message.md`
- Create: `outputs/.gitkeep`
- Create: `examples/input/.gitkeep`
- Create: `tests/test_documentation_and_scripts.py`

**Interfaces:**
- Consumes: package public APIs and the target machine.
- Produces: no-weight environment report, trusted-weight benchmark JSON, complete Chinese usage guide and mentor request text.

- [ ] **Step 1: Write failing documentation and parser-contract tests**

```python
# tests/test_documentation_and_scripts.py
from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parents[1]

def test_readme_documents_exact_command_status_and_test_layers():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "python -m moiredet_repro.cli infer" in text
    assert "PSENet_100_loss0.000000.pth" in text
    assert "代码兼容层和无权重验证完成" in text
    assert 'pytest -m "not gpu and not checkpoint"' in text
    assert 'pytest -m "gpu and not checkpoint"' in text
    assert "不能宣称任务 1 已完成" in text

def test_checkpoint_request_is_ready_to_send():
    text = (ROOT / "docs" / "checkpoint-request-message.md").read_text(encoding="utf-8")
    assert "Doing More With Moiré Pattern Detection in Digital Photos" in text
    assert "PSENet_100_loss0.000000.pth" in text
    assert "cong-yang/MoireDet" in text

def test_benchmark_help_does_not_require_a_checkpoint():
    result = subprocess.run([sys.executable, "scripts/benchmark_inference.py", "--help"],
                            cwd=str(ROOT), capture_output=True, text=True)
    assert result.returncode == 0 and "--checkpoint" in result.stdout

def test_runtime_layer_has_no_author_absolute_paths():
    for folder in (ROOT / "src", ROOT / "configs", ROOT / "scripts"):
        for path in folder.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                assert "/home/users/" not in text and "/data/zhenyu.yang/" not in text and "E:/zj/" not in text
```

- [ ] **Step 2: Run the tests and verify docs/scripts are absent**

Run: `D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/test_documentation_and_scripts.py -v`

Expected: FAIL because `README.md` and the request message do not exist.

- [ ] **Step 3: Implement the no-weight environment verification script**

```python
# scripts/verify_environment.py
import json, platform
from pathlib import Path
import cv2, numpy as np, torch, torchvision, yaml
from importlib.metadata import version
from moiredet_repro.config import load_config
from moiredet_repro.inference import MoireDetInference, set_determinism
from moiredet_repro.upstream_adapter import build_official_model

def main():
    set_determinism(2)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        matrix = torch.randn(512, 512, device="cuda")
        _ = matrix @ matrix; torch.cuda.synchronize()
    config = load_config()
    cpu_service = MoireDetInference(build_official_model(config), torch.device("cpu"), None)
    cpu_prediction = cpu_service.predict_tensor(torch.randn(1, 3, 320, 320))
    cuda_shape = None
    if cuda_available:
        cuda_service = MoireDetInference(build_official_model(config), torch.device("cuda:0"), None)
        cuda_shape = list(cuda_service.predict_tensor(torch.randn(1, 3, 320, 320, device="cuda:0")).shape)
    report = {
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "opencv": cv2.__version__, "numpy": np.__version__, "pyyaml": yaml.__version__,
        "einops": version("einops"), "local_attention": version("local-attention"),
        "cpu_random_forward_shape": list(cpu_prediction.shape),
        "cuda_random_forward_shape": cuda_shape,
        "checkpoint_integration": "not_run",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if cuda_available and cuda_shape == [320, 320] else 1

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Implement the fixed 5/20 trusted-checkpoint benchmark entry point**

```python
# scripts/benchmark_inference.py
import argparse, json
from dataclasses import asdict
from pathlib import Path
from moiredet_repro.config import load_config
from moiredet_repro.inference import MoireDetInference, set_determinism
from moiredet_repro.preprocessing import prepare_image

def parser():
    value = argparse.ArgumentParser(description="MoireDet RTX acceptance benchmark")
    value.add_argument("--input", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--checkpoint-manifest", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--device", choices=("cuda",), default="cuda")
    return value

def main(argv=None):
    args = parser().parse_args(argv); config = load_config(); set_determinism(2)
    service = MoireDetInference.from_checkpoint(args.checkpoint, args.checkpoint_manifest, config, args.device)
    prepared = prepare_image(args.input, config, service.device)
    result = asdict(service.benchmark_tensor(prepared.tensor, warmup=5, iterations=20))
    result.update({"input": str(args.input.resolve()), "checkpoint_sha256": service.checkpoint_info.sha256})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Write the Chinese README and exact mentor request**

Use this complete content for `README.md`:

````markdown
# MoireDet 单图推理复现

这是论文 *Doing More With Moiré Pattern Detection in Digital Photos* 的考核导向复现工程。项目固定作者实现、模型配置与预处理，在外围提供可审计的 Windows/RTX 推理接口。

## 当前状态

- **代码兼容层和无权重验证完成**：环境、配置、模型构建、随机前向、检查点核验、CLI 与输出逻辑可以在没有权重时验证。
- **可信权重集成与 RTX 正式验收**：只有可信 `PSENet_100_loss0.000000.pth` 通过来源、SHA-256、严格加载、样例/自选图和视觉检查后才完成。

第二项仍为 `SKIPPED` 时，不能宣称任务 1 已完成。

## 来源与兼容修改

上游为 `https://github.com/cong-yang/MoireDet`，固定提交为 `afde899f3c3beee96160610ee450618136a38f7b`。用户仓库 `main` 精确基于该提交；详情见 `docs/upstream/UPSTREAM.md`。上游仓库未提供明确代码许可证，因此这里只标注来源，不把整项工程声明为 MIT。

仅有两类、共三行运行时修改：`resnet.py` 和 `resnet_dct.py` 回退到 `torch.hub.load_state_dict_from_url`；目标 `TripleBranchWithSpecificConv` 的注意力骨干使用 `pretrained=False`，阻止隐式下载。完整检查点仍以 `strict=True` 覆盖并校验所有参数。

## 固定环境与安装

目标环境：Python 3.8.20、PyTorch 1.10.0+cu113、torchvision 0.11.1+cu113、CUDA 11.3、OpenCV 4.11.0.86、NumPy 1.24.3、PyYAML 6.0.2、einops 0.3.0、local-attention 1.2.1。

```powershell
D:\anaconda3\Scripts\conda.exe create --name moiredet-repro --clone exp -y
D:\anaconda3\envs\moiredet-repro\python.exe -m pip install --no-deps einops==0.3.0 local-attention==1.2.1
D:\anaconda3\envs\moiredet-repro\python.exe -m pip install setuptools==68.2.2 pytest==7.4.4
D:\anaconda3\envs\moiredet-repro\python.exe -m pip install -e . --no-deps
D:\anaconda3\envs\moiredet-repro\python.exe -m pip check
```

## 无权重验证

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe .\scripts\verify_environment.py
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest -m "not gpu and not checkpoint" -v
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest -m "gpu and not checkpoint" -v
```

随机前向只证明环境与网络结构可执行，不证明检测结果正确。

## 可信权重

把权重放到 `weights/PSENet_100_loss0.000000.pth`，并把来源与真实 SHA-256 写入同名旁文件 `weights/PSENet_100_loss0.000000.pth.json`。schema、允许来源和无效示例见 `weights/README.md`；只有来源证据和哈希同时通过时，`run.json` 才记录 `"checkpoint_verified": true`。权重由 `torch.load` 反序列化，只能使用可信文件。

## 单图命令

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe -m moiredet_repro.cli infer --input .\MoireDet\script\00002423.png --checkpoint .\weights\PSENet_100_loss0.000000.pth --checkpoint-manifest .\weights\PSENet_100_loss0.000000.pth.json --output .\outputs\official-sample --device cuda
```

输入保持 OpenCV BGR 顺序，直接双线性缩放到 `320 x 320`，使用 ImageNet mean/std 归一化，batch size 固定为 1。输入和输出均使用 Unicode 安全路径读写，可直接处理中文 Windows 路径。已有任一目标文件时命令失败，不覆盖旧结果。

每次成功运行生成：

- `prediction.npy`：未做显示归一化的 `320 x 320 float32` 原始预测；
- `moire_map.png`：恢复到输入尺寸的 8 位灰度检测图；
- `comparison.png`：左侧原图、右侧检测图；
- `run.json`：输入、上游提交、可信权重、运行时、预测范围、性能与输出清单。

逐图 min-max 只用于 PNG 显示，不修改 `prediction.npy`。

## 基准与正式门控

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe .\scripts\benchmark_inference.py --input .\MoireDet\script\00002423.png --checkpoint .\weights\PSENet_100_loss0.000000.pth --checkpoint-manifest .\weights\PSENet_100_loss0.000000.pth.json --output .\outputs\acceptance\benchmark.json --device cuda
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest -m checkpoint -v -rs
```

基准固定 5 次预热、20 次计时。权重环境变量未配置时，检查点测试必须显示 `SKIPPED`，不是 `PASSED`。

## 常见错误

- `checkpoint file is missing`：放置权重及旁文件，不要修改全局 Python。
- `untrusted source_type` 或 `SHA-256 mismatch`：补全可信来源证据，或重新核对文件。
- `strict state_dict mismatch`：文件不属于目标 `TripleBranchWithSpecificConv`。
- 显式 `--device cuda` 但 CUDA 不可用：检查 NVIDIA 驱动、Conda 环境与 PyTorch CUDA 构建。
- 输出冲突：改用新的运行目录；工程不会覆盖旧结果。
- NaN/Inf：该次运行无效，不能把输出当作检测结果。

仍无法取得可信权重时，直接发送 `docs/checkpoint-request-message.md`。
````

Use this exact message body in `docs/checkpoint-request-message.md`:

```text
老师您好，我正在复现论文《Doing More With Moiré Pattern Detection in Digital Photos》及作者仓库 cong-yang/MoireDet，用于完成 BeeLab 考核中的 MoireDet 代码运行和后续视频 Demo。目前仓库提供的 Google Drive 链接无法取得官方检查点 PSENet_100_loss0.000000.pth。请问您或课题组是否保存了该官方权重，方便提供一份吗？我会记录文件来源和 SHA-256，仅用于本次考核复现。谢谢老师！
```

- [ ] **Step 6: Run docs tests, no-weight verification and all weight-independent tests**

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest -m "not checkpoint" -v
D:\anaconda3\envs\moiredet-repro\python.exe .\scripts\verify_environment.py
```

Expected: all tests PASS on the RTX 4060; the report contains CPU and CUDA shapes `[320, 320]` and `checkpoint_integration: not_run`.

- [ ] **Step 7: Commit the reproducible handoff**

```powershell
git add README.md scripts docs/checkpoint-request-message.md outputs/.gitkeep examples/input/.gitkeep tests/test_documentation_and_scripts.py
git commit -m "docs: add MoireDet reproduction and weight handoff"
```

### Task 10: Add and Execute the Trusted-Checkpoint Acceptance Gate

**Files:**
- Create: `tests/integration/test_official_checkpoint.py`
- Modify: `README.md`
- Runtime-only: `weights/PSENet_100_loss0.000000.pth`, matching `.pth.json`, user-selected image and `outputs/acceptance/**` (all ignored by Git).

**Interfaces:**
- Consumes: `MOIREDET_CHECKPOINT`, `MOIREDET_CHECKPOINT_MANIFEST`, optional `MOIREDET_USER_IMAGE`, RTX 4060 CUDA runtime.
- Produces: explicit skipped gate without weight; with weight, strict load, two-image CLI outputs, repeatability result, benchmark JSON and manual visual checklist.

- [ ] **Step 1: Write the checkpoint-gated integration tests**

```python
# tests/integration/test_official_checkpoint.py
import os
from pathlib import Path
import numpy as np, pytest, torch
from moiredet_repro.checkpoint import load_checkpoint_bundle, load_weights_strict
from moiredet_repro.cli import main as cli_main
from moiredet_repro.config import load_config
from moiredet_repro.inference import MoireDetInference, set_determinism
from moiredet_repro.preprocessing import prepare_image
from moiredet_repro.upstream_adapter import build_official_model

pytestmark = [pytest.mark.checkpoint, pytest.mark.integration]
ROOT = Path(__file__).resolve().parents[2]

def trusted_paths():
    checkpoint = os.environ.get("MOIREDET_CHECKPOINT")
    manifest = os.environ.get("MOIREDET_CHECKPOINT_MANIFEST")
    if not checkpoint or not manifest:
        pytest.skip("trusted MoireDet checkpoint environment variables are not set")
    return Path(checkpoint), Path(manifest)

def test_official_checkpoint_strict_loads_on_cpu():
    checkpoint, manifest = trusted_paths()
    info = load_weights_strict(build_official_model(load_config()), load_checkpoint_bundle(checkpoint, manifest))
    assert info.checkpoint_verified and info.sha256 == info.manifest.expected_sha256

@pytest.mark.gpu
def test_official_sample_cuda_outputs_and_repeatability(tmp_path):
    if not torch.cuda.is_available(): pytest.skip("CUDA is unavailable")
    checkpoint, manifest = trusted_paths()
    sample = ROOT / "MoireDet" / "script" / "00002423.png"
    code = cli_main(["infer", "--input", str(sample), "--checkpoint", str(checkpoint),
        "--checkpoint-manifest", str(manifest), "--output", str(tmp_path / "sample"), "--device", "cuda"])
    assert code == 0
    first = np.load(str(tmp_path / "sample" / "prediction.npy"))
    assert first.shape == (320, 320) and np.isfinite(first).all() and np.ptp(first) > 1e-8
    set_determinism(2)
    service = MoireDetInference.from_checkpoint(checkpoint, manifest, load_config(), "cuda")
    prepared = prepare_image(sample, load_config(), service.device)
    a, b = service.predict_tensor(prepared.tensor), service.predict_tensor(prepared.tensor)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)

@pytest.mark.gpu
def test_user_image_cuda_output(tmp_path):
    user_image = os.environ.get("MOIREDET_USER_IMAGE")
    if not user_image: pytest.skip("MOIREDET_USER_IMAGE is not set")
    checkpoint, manifest = trusted_paths()
    code = cli_main(["infer", "--input", user_image, "--checkpoint", str(checkpoint),
        "--checkpoint-manifest", str(manifest), "--output", str(tmp_path / "user"), "--device", "cuda"])
    assert code == 0 and (tmp_path / "user" / "comparison.png").is_file()

@pytest.mark.gpu
@pytest.mark.benchmark
@pytest.mark.slow
def test_fixed_cuda_benchmark_contract():
    if not torch.cuda.is_available(): pytest.skip("CUDA is unavailable")
    checkpoint, manifest = trusted_paths()
    service = MoireDetInference.from_checkpoint(checkpoint, manifest, load_config(), "cuda")
    sample = ROOT / "MoireDet" / "script" / "00002423.png"
    prepared = prepare_image(sample, load_config(), service.device)
    stats = service.benchmark_tensor(prepared.tensor, warmup=5, iterations=20)
    assert stats.median_ms > 0 and stats.p95_ms >= stats.median_ms
    assert stats.peak_memory_allocated_bytes > 0 and stats.peak_memory_reserved_bytes > 0
```

- [ ] **Step 2: Run the gate now and verify honest skips without a weight**

Run: `D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests/integration/test_official_checkpoint.py -v`

Expected now: all four tests show `SKIPPED` with explicit missing checkpoint/image reasons; no test is marked `XFAIL` or `PASS`.

- [ ] **Step 3: Commit the acceptance gate before external weight arrival**

```powershell
git add tests/integration/test_official_checkpoint.py README.md
git commit -m "test: add trusted MoireDet checkpoint acceptance gate"
```

- [ ] **Step 4: When a trusted file arrives, establish provenance before loading it**

Place it at `weights/PSENet_100_loss0.000000.pth`; compute SHA-256 with `Get-FileHash -Algorithm SHA256`; create `weights/PSENet_100_loss0.000000.pth.json` using the fixed manifest schema, the actual acquisition date and evidence; then set:

```powershell
$env:MOIREDET_CHECKPOINT = (Resolve-Path .\weights\PSENet_100_loss0.000000.pth).Path
$env:MOIREDET_CHECKPOINT_MANIFEST = (Resolve-Path .\weights\PSENet_100_loss0.000000.pth.json).Path
$env:MOIREDET_USER_IMAGE = (Resolve-Path .\examples\input\user_moire.png).Path
```

- [ ] **Step 5: Run the complete RTX 4060 acceptance suite and benchmark**

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest -m checkpoint -v
D:\anaconda3\envs\moiredet-repro\python.exe .\scripts\benchmark_inference.py --input .\MoireDet\script\00002423.png --checkpoint $env:MOIREDET_CHECKPOINT --checkpoint-manifest $env:MOIREDET_CHECKPOINT_MANIFEST --output .\outputs\acceptance\benchmark.json --device cuda
```

Expected with trusted weight: all checkpoint tests PASS; both inputs produce four files; repeatability meets `rtol=1e-5`, `atol=1e-6`; benchmark records 5 warmups, 20 measurements, median, P95 and both peak-memory fields.

- [ ] **Step 6: Perform and record the required visual check**

Open both `comparison.png` files and record in `outputs/acceptance/visual-check.md`: files decode correctly; maps are not unexpectedly all black/white; high-response regions correspond to visible moiré regions; raw min/max/range from each `run.json`; reviewer name and date. Only after this check may README status change from “代码兼容层和无权重验证完成” to “任务 1 单图推理验收完成”.

- [ ] **Step 7: Run the final repository verification**

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest -v
git status --short
git diff --check
```

Expected: all available tests pass, checkpoint tests either pass with a trusted weight or explicitly skip without one, and no weight/output/personal image is staged.

---

## Execution Order and Checkpoints

1. Execute Tasks 1–2 sequentially because every later task depends on the isolated environment and patched upstream baseline.
2. Tasks 3–5 are separable reviewer gates but must all be complete before Task 6.
3. Execute Tasks 6–8 sequentially to establish the inference result, metadata and CLI contracts.
4. Task 9 completes the honest no-weight deliverable and is the first point at which “代码兼容层和无权重验证完成” may be reported.
5. Task 10 remains an explicit external gate until a trusted checkpoint is obtained; skip output is evidence of an unmet prerequisite, not successful reproduction.
