# MoireDet 单图推理复现

这是论文 *Doing More With Moiré Pattern Detection in Digital Photos* 的考核导向复现工程。当前阶段固定论文作者公开的 MoireDet 网络与单图预处理行为，在 Windows + RTX 4060 Laptop GPU 上提供可审计、可重复的推理入口；后续视频 Demo 可以复用同一个常驻模型服务。

## 当前状态

- **代码兼容层和无权重验证完成**：隔离环境、固定配置、官方模型构建、CPU/CUDA 随机前向、CLI 编排、四文件输出与安全保护已有自动化验证。
- **可信检查点与 RTX 4060 正式验收仍为 `SKIPPED / 未完成`**：当前没有经过代码审查授权的 `PSENet_100_loss0.000000.pth`、来源证据和真实 SHA-256，也没有执行正式检查点推理、基准或人工视觉检查。
- 因此，本仓库目前**不得宣称考核 Task 1 已完成**，也不声称已经复现论文检测效果。随机权重前向只证明环境和网络结构可运行。

## 上游来源、最小补丁与许可边界

- 官方仓库：[cong-yang/MoireDet](https://github.com/cong-yang/MoireDet)
- 固定提交：`afde899f3c3beee96160610ee450618136a38f7b`
- 仓库内官方源码树：`MoireDet/`
- 官方样图：`MoireDet/script/00002423.png`
- 详细哈希与来源记录：`docs/upstream/UPSTREAM.md`

本工程只保留两个可追踪兼容补丁，共涉及三个上游源文件位置：

1. `patches/0001-torchvision-load-state-dict-compat.patch`：在 `resnet.py` 和 `resnet_dct.py` 中为旧版 `torchvision.models.utils.load_state_dict_from_url` 增加 `torch.hub` 兼容回退；
2. `patches/0002-disable-resnet-online-download.patch`：仅把目标类 `TripleBranchWithSpecificConv` 的注意力 ResNet 初始化改为离线构建，防止隐式下载 ImageNet 权重。

补丁不改动 Performer 张量布局，也不改写 MoireDet 检测结构；完整检查点仍必须以 `strict=True` 覆盖目标模型参数。上游仓库没有提供明确的代码许可证，本仓库只做来源标注与本地兼容记录，不把上游代码擅自声明为 MIT 或其他许可证。

作者在固定提交的 `MoireDet/script/model_download.txt` 中指定检查点：

```text
PSENet_100_loss0.000000.pth
https://drive.google.com/file/d/1QivNnHWaomJmUuBgueGzwtVooijsc_TH/view?usp=sharing
```

该 Google Drive 链接在当前环境不可用；这里仅记录作者来源，**不声称已经下载、验证或找到可信镜像**。可直接发送 `docs/checkpoint-request-message.md` 向导师或作者索取原始文件。

## 已验证环境

实际验证版本如下：

| 组件 | 版本 |
| --- | --- |
| Python | `3.8.20` |
| pip | `24.2` |
| PyTorch | `1.10.0+cu113` |
| torchvision | `0.11.1+cu113` |
| CUDA runtime | `11.3` |
| opencv-python distribution | `4.11.0.86` |
| `cv2.__version__` | `4.11.0` |
| NumPy | `1.24.3` |
| PyYAML | `6.0.2` |
| tqdm | `4.67.1` |
| einops | `0.3.0` |
| local-attention | `1.2.1` |
| pytest | `7.4.4` |
| setuptools | `68.2.2` |
| wheel | `0.44.0` |

### 首选安装：克隆已验证的 `exp` 环境

这是本机实际走通的安装路线；不会修改原 `exp` 环境。请在仓库根目录运行：

```powershell
D:\anaconda3\Scripts\conda.exe create --name moiredet-repro --clone exp -y
D:\anaconda3\envs\moiredet-repro\python.exe -m pip install --no-deps einops==0.3.0 local-attention==1.2.1
D:\anaconda3\envs\moiredet-repro\python.exe -m pip install setuptools==68.2.2 pytest==7.4.4
D:\anaconda3\envs\moiredet-repro\python.exe -m pip install -e . --no-deps --no-build-isolation
D:\anaconda3\envs\moiredet-repro\python.exe -m pip check
```

最后一条的期望输出是 `No broken requirements found.`。`--no-deps --no-build-isolation` 很重要：它使用已经固定的本地构建工具，不在可编辑安装阶段访问包索引。

`environment.yml` 是版本参考与另一种全新环境重建入口：

```powershell
D:\anaconda3\Scripts\conda.exe env create -f .\environment.yml
```

但当前经过完整实机验证的是“从 `exp` 克隆”路线，不能把 `environment.yml` 当作唯一已验证证据。

### pip/OpenSSL 出现 TLS EOF 时的离线 wheel 路线

本机曾观察到克隆环境中的 Python/OpenSSL 无法连接镜像和官方 PyPI，pip 报错：

```text
SSLZeroReturnError: TLS/SSL connection has been closed (EOF)
```

系统 Windows `curl.exe` 和浏览器仍可访问官方 PyPI。不要关闭证书验证、不要换用来源不明的 wheel，也不要把忽略目录中的临时 wheel 提交到 Git。应从各包的官方 PyPI **Files** 页面下载到仓库外的 `D:\moire-wheels`，也可以把 Files 页给出的 `https://files.pythonhosted.org/...` 直链交给 Windows `curl.exe -L --fail --output` 下载：

- [einops 0.3.0](https://pypi.org/project/einops/0.3.0/#files)
- [local-attention 1.2.1](https://pypi.org/project/local-attention/1.2.1/#files)
- [setuptools 68.2.2](https://pypi.org/project/setuptools/68.2.2/#files)
- [pytest 7.4.4](https://pypi.org/project/pytest/7.4.4/#files)
- [exceptiongroup 1.2.2](https://pypi.org/project/exceptiongroup/1.2.2/#files)
- [iniconfig 2.0.0](https://pypi.org/project/iniconfig/2.0.0/#files)
- [pluggy 1.5.0](https://pypi.org/project/pluggy/1.5.0/#files)
- [tomli 2.0.1](https://pypi.org/project/tomli/2.0.1/#files)

本次实际安装前核验通过的文件与官方 SHA-256 为：

| 文件 | SHA-256 |
| --- | --- |
| `einops-0.3.0-py2.py3-none-any.whl` | `a91c6190ceff7d513d74ca9fd701dfa6a1ffcdd98ea0ced14350197c07f75c73` |
| `local_attention-1.2.1-py3-none-any.whl` | `6a5217b0fa17e09afdb31e2f5b4b95a0747a74cfd1522c5605639a08ddc9eb29` |
| `setuptools-68.2.2-py3-none-any.whl` | `b454a35605876da60632df1a60f736524eb73cc47bbc9f3f1ef1b644de74fd2a` |
| `pytest-7.4.4-py3-none-any.whl` | `b090cdf5ed60bf4c45261be03239c2c1c22df034fbffe691abe93cd80cea01d8` |
| `exceptiongroup-1.2.2-py3-none-any.whl` | `3111b9d131c238bec2f8f516e123e14ba243563fb135d3fe885990585aa7795b` |
| `iniconfig-2.0.0-py3-none-any.whl` | `b6a85871a79d2e3b22d2d1b94ac2824226a63c6b741c88f7ae975f18b6778374` |
| `pluggy-1.5.0-py3-none-any.whl` | `44e1ad92c8ca002de6377e165f3e0f1be63266ab4d554740532335b9d75ea669` |
| `tomli-2.0.1-py3-none-any.whl` | `939de3e7a6161af0c887ef91b7d41a53e7c5a1ca976325f429cb46ea9bc30ecc` |

先用 `Get-FileHash -Algorithm SHA256 D:\moire-wheels\文件名.whl` 与上表逐项比对，再进行纯本地安装：

```powershell
$moireWheelDir = 'D:\moire-wheels'
D:\anaconda3\envs\moiredet-repro\python.exe -m pip install --no-index --no-deps "$moireWheelDir\einops-0.3.0-py2.py3-none-any.whl" "$moireWheelDir\local_attention-1.2.1-py3-none-any.whl"
D:\anaconda3\envs\moiredet-repro\python.exe -m pip install --no-index --find-links "$moireWheelDir" setuptools==68.2.2 pytest==7.4.4
D:\anaconda3\envs\moiredet-repro\python.exe -m pip install -e . --no-deps --no-build-isolation
D:\anaconda3\envs\moiredet-repro\python.exe -m pip check
```

克隆的 `exp` 已包含已验证的 `wheel==0.44.0`；若本机克隆结果不同，应先核对源环境，而不是悄悄改变固定版本。可编辑安装成功后出现 pip 版本检查的 TLS 警告不等于安装失败，仍以安装退出码、包版本和 `pip check` 为准。

## 无权重验证

诊断脚本明确禁止检查点反序列化和网络访问；它不会调用 `torch.load`，也不会尝试下载模型：

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe .\scripts\verify_environment.py --device all
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest -m "not checkpoint" -v
```

在本机 RTX 4060 Laptop GPU 上，真实官方模型的 CPU/CUDA 随机前向均已得到 `[320, 320]` 输出；报告同时应包含：

```text
checkpoint_integration: not_run
checkpoint_deserialization: not_attempted
network_access: not_attempted
```

要单独确认外部检查点 gate 仍被诚实跳过：

```powershell
Remove-Item Env:MOIREDET_CHECKPOINT -ErrorAction SilentlyContinue
Remove-Item Env:MOIREDET_CHECKPOINT_MANIFEST -ErrorAction SilentlyContinue
Remove-Item Env:MOIREDET_USER_IMAGE -ErrorAction SilentlyContinue
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest .\tests\integration\test_official_checkpoint.py -v -rs
```

预期是 4 个用例全部 `SKIPPED`，不是 `PASSED`。

## 可信检查点：sidecar 与跟踪信任库缺一不可

PyTorch 1.10 的 `torch.load` 基于 pickle，反序列化恶意文件可能执行代码。为此，正式加载需要同时满足两层授权：

1. 权重旁边的本地 sidecar manifest，例如 `weights/PSENet_100_loss0.000000.pth.json`，记录文件名、允许的来源类型、来源证据、取得日期和该字节副本的真实 SHA-256；
2. Git 跟踪并经过代码审查的 `configs/trusted_checkpoints.json`，其中必须存在与 sidecar 六个来源字段完全一致的授权条目（信任库条目不重复 sidecar 的 `schema_version`）。

信任库初始内容为：

```json
{
  "schema_version": 1,
  "checkpoints": []
}
```

因此，即使有人在被忽略的 `weights/` 中放入 `.pth` 和 sidecar，也**不能单方面授权 pickle 加载**。取得可信文件后，应先计算真实哈希、补全 sidecar，再把 `filename`、`source_type`、`source_reference`、`retrieved_at`、`provenance_evidence` 和 `expected_sha256` 六个字段有意加入跟踪信任库，审查来源证据和变更后才允许加载。不要把权重本体、sidecar、个人图片或运行输出提交到 Git。

允许的 sidecar `source_type` 只有：`official_repository`、`author_direct`、`author_team_direct`、`advisor_direct` 和 `verified_mirror`。文件名相同、只有哈希、只有 sidecar 或只有信任库条目都不充分。加载顺序必须是来源/schema 校验、跟踪信任库匹配、实际字节 SHA-256 匹配，然后才允许 `torch.load`；最终仍要求顶层 `state_dict` 且 `strict=True` 加载成功。

检查权重哈希：

```powershell
Get-FileHash -Algorithm SHA256 .\weights\PSENet_100_loss0.000000.pth
```

sidecar schema 参考 `weights/checkpoint.example.json`；其中 64 个零是故意设置的无效哨兵，不能用于正式运行。

## 单图推理 CLI

下面是可信检查点和跟踪信任库均已审查通过后，处理官方样图的完整命令。`--checkpoint-manifest` 虽可省略并默认查找 `<checkpoint>.json`，正式验收仍显式传入：

```powershell
$moireRunStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$moireOutput = Join-Path (Get-Location) "outputs\single\$moireRunStamp-official"
if (Test-Path -LiteralPath $moireOutput) { throw "Output leaf already exists: $moireOutput" }

D:\anaconda3\envs\moiredet-repro\python.exe -m moiredet_repro.cli infer --input .\MoireDet\script\00002423.png --checkpoint .\weights\PSENet_100_loss0.000000.pth --checkpoint-manifest .\weights\PSENet_100_loss0.000000.pth.json --output $moireOutput --device cuda --config .\configs\inference.yaml
```

输入按官方示例使用 OpenCV BGR、直接双线性缩放到 `320 x 320`、ImageNet mean/std 归一化，batch size 固定为 1。`--device cuda` 不会在失败时静默回退 CPU。

`--output` 必须是**尚不存在的新目录叶节点**；即使目录为空也会被拒绝。程序先在同级私有 staging 目录完整写入并回读验证，最后原子重命名发布，避免留下半套结果。成功目录恰好包含：

- `prediction.npy`：未经显示拉伸的 `320 x 320 float32` 原始预测；
- `moire_map.png`：恢复到输入宽高的 8 位灰度检测图；
- `comparison.png`：左侧原始 BGR 图，右侧同尺寸灰度检测图；
- `run.json`：输入、固定上游、已验证检查点、环境、耗时、显存、原始范围和输出清单。

PNG 的逐图 min-max 仅用于显示，不会改写 `prediction.npy`。处理自选图片时只替换 `--input`，并继续使用新的时间戳输出叶节点。

## 可信检查点与 RTX 4060 验收 gate

只有检查点、sidecar 和跟踪信任库都准备完成后，才设置前两个环境变量。用户图片是可选的，只控制用户图片用例：

```powershell
$env:MOIREDET_CHECKPOINT = (Resolve-Path .\weights\PSENet_100_loss0.000000.pth).Path
$env:MOIREDET_CHECKPOINT_MANIFEST = (Resolve-Path .\weights\PSENet_100_loss0.000000.pth.json).Path
# 可选：$env:MOIREDET_USER_IMAGE = (Resolve-Path .\examples\input\user_moire.png).Path

D:\anaconda3\envs\moiredet-repro\python.exe -m pytest .\tests\integration\test_official_checkpoint.py -v -rs
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest -m checkpoint -v -rs
```

`MOIREDET_CHECKPOINT` 与 `MOIREDET_CHECKPOINT_MANIFEST` 都未设置或只设置一个时，gate 会明确 `SKIPPED`。两者都设置后，路径不存在、sidecar 不完整、信任库未授权、SHA-256 不一致、pickle 结构不符或严格 state-dict 不匹配都必须失败，不能再被当成“缺少权重”跳过。正式 CUDA 用例还要求设备名包含 RTX 4060。

固定基准为 5 次预热和 20 次计时，输出 JSON 文件也必须是尚不存在的新叶节点：

```powershell
$moireBenchmarkStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$moireBenchmarkOutput = Join-Path (Get-Location) "outputs\acceptance\$moireBenchmarkStamp-benchmark.json"
if (Test-Path -LiteralPath $moireBenchmarkOutput) { throw "Benchmark output already exists: $moireBenchmarkOutput" }

D:\anaconda3\envs\moiredet-repro\python.exe .\scripts\benchmark_inference.py --input .\MoireDet\script\00002423.png --checkpoint $env:MOIREDET_CHECKPOINT --checkpoint-manifest $env:MOIREDET_CHECKPOINT_MANIFEST --output $moireBenchmarkOutput --device cuda
```

验收要求 `median_ms > 0`、`p95_ms >= median_ms`、两项峰值显存均为正数；相同输入连续推理还必须满足 `rtol=1e-5, atol=1e-6`。

## 人工视觉检查

正式 CLI 生成官方样图和用户图（若提供）后，逐一打开 `comparison.png`，并在被忽略的 `outputs/acceptance/visual-check.md` 中记录：

- PNG 能正确解码，左侧确为原图，右侧确为同尺寸检测图；
- 检测图不是意外全黑或全白；
- 高响应区域与肉眼可见摩尔纹区域相对应；
- 对应 `run.json` 中 raw `min`、`max`、`dynamic_range`；
- 检查人和日期。

只有可信权重双重授权、严格加载、RTX 4060 自动化验收和人工视觉检查全部通过后，才能把状态改为“Task 1 单图推理验收完成”。当前这些步骤没有执行。

## 常见问题

- **环境不存在或只有半套包**：先运行 `D:\anaconda3\Scripts\conda.exe env list` 和目标解释器的 `pip check`；不要把安装命令混入 `exp`，也不要在未核对时覆盖已有 `moiredet-repro`。
- **pip 报 OpenSSL/TLS EOF**：使用上面的官方 PyPI wheel、官方 SHA-256 与 `--no-index` 本地安装；不要用 `--trusted-host` 绕过校验，也不要降低版本。
- **checkpoint 环境变量缺失或只设一个**：这是外部 gate 未配置，测试应 `SKIPPED`；同时设置二者后，所有无效输入都应失败。
- **`checkpoint manifest is not authorized by repository trusted checkpoint store`**：sidecar 不能自行授权 pickle；需要把完整来源记录和真实 SHA-256 有意加入 `configs/trusted_checkpoints.json` 并完成代码审查。
- **SHA-256 mismatch**：当前字节文件不是 sidecar 和信任库锁定的副本，不能加载。
- **strict state_dict mismatch**：权重不是目标 `TripleBranchWithSpecificConv` 的完整检查点；只允许去除键首部 `module.`，不能用 `strict=False` 掩盖问题。
- **CUDA unavailable / GPU 不是 RTX 4060**：显式 `--device cuda` 必须失败；检查 NVIDIA 驱动、当前 Conda 解释器和 `torch.cuda.is_available()`，不要静默改成 CPU 充当正式验收。
- **输出路径已存在**：生成新的时间戳叶节点；不要删除或覆盖旧证据来复用路径。
- **prediction contains NaN or Inf**：该次推理无效，不得发布四文件结果或写成成功结论。
- **作者 Drive 链接失效**：发送 `docs/checkpoint-request-message.md`；不要用无法追溯来源的第三方权重替代。

## 最终自检

帮助和参数探针不会加载权重：

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe -m moiredet_repro.cli --help
D:\anaconda3\envs\moiredet-repro\python.exe -m moiredet_repro.cli infer --help
D:\anaconda3\envs\moiredet-repro\python.exe .\scripts\verify_environment.py --help
D:\anaconda3\envs\moiredet-repro\python.exe .\scripts\benchmark_inference.py --help
```

提交或交接前运行：

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe .\scripts\verify_environment.py --device all
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest -v
D:\anaconda3\envs\moiredet-repro\python.exe -m pip check
git diff --check
git status --short
```

没有可信权重时，完整测试中的 checkpoint 用例必须保持 `SKIPPED`；这不影响无权重代码验证，但也不构成正式复现完成证据。
