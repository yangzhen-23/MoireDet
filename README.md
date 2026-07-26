# MoireDet 单图推理复现

这是论文 *Doing More With Moiré Pattern Detection in Digital Photos* 的考核导向复现工程。上游为 [cong-yang/MoireDet](https://github.com/cong-yang/MoireDet)，固定提交 `afde899f3c3beee96160610ee450618136a38f7b`；本仓库的适配和说明以该提交为准（见 `docs/upstream/UPSTREAM.md`）。上游没有给出明确代码许可证，因此这里只记录来源，不将整个项目声明为 MIT。

## 状态边界

- 代码兼容层和无权重验证可以完成：固定环境、配置、官方模型构建、随机前向、CLI 和输出契约均可验证。
- 可信检查点集成与 RTX 正式验收仍为 **SKIPPED**，直到取得可信的 `PSENet_100_loss0.000000.pth`、来源证据和真实 SHA-256，并通过严格加载、样例/自选图视觉检查及基准测试。

因此，在没有可信权重时，绝不能声称考核 Task 1 已完整复现。

## 固定环境与安装（Windows/RTX）

目标为 Python 3.8.20、PyTorch `1.10.0+cu113`、torchvision `0.11.1+cu113`、CUDA 11.3、`opencv-python` wheel `4.11.0.86`、NumPy 1.24.3、PyYAML 6.0.2、einops 0.3.0、local-attention 1.2.1。注意 `cv2.__version__` 实测为 `4.11.0`，它不是 wheel/distribution 版本；诊断脚本同时报告两者。

```powershell
D:\anaconda3\Scripts\conda.exe env create -f .\environment.yml
D:\anaconda3\envs\moiredet-repro\python.exe -m pip install -e . --no-deps
D:\anaconda3\envs\moiredet-repro\python.exe -m pip check
```

若学校网络的 TLS 或 PyPI 访问失败，使用仓库 `.superpowers\sdd\2026-07-27-moiredet-inference-reproduction` 中已缓存的 wheel，并以 `--no-index --find-links <该目录>` 安装；不要放宽证书校验或替换固定版本。

## 无权重诊断

诊断不会读取检查点、不会调用 `torch.load`、不会联网；它会校验实际安装版本、执行 CUDA 矩阵乘法，并对未加载权重的官方模型做真实随机前向。任何目录均可执行：

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe .\scripts\verify_environment.py --device cpu
D:\anaconda3\envs\moiredet-repro\python.exe .\scripts\verify_environment.py --device cuda
D:\anaconda3\envs\moiredet-repro\python.exe .\scripts\verify_environment.py --device all
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest -m "not checkpoint" -v
```

CPU/CUDA 的 `*_random_forward_shape` 应为 `[320, 320]`；`checkpoint_integration` 必为 `not_run`。随机前向只能说明环境和网络结构可以执行，不证明检测质量。

## 可信权重、单图推理和输出

将可信的原始文件放到 `weights/PSENet_100_loss0.000000.pth`，并在同目录创建 `weights/PSENet_100_loss0.000000.pth.json`。sidecar 必须使用 `weights/checkpoint.example.json` 的 schema，提供允许来源类型、可核验来源证据和精确小写 SHA-256。示例中的全零 hash 是故意不可用的哨兵值。加载器仅在来源和哈希验证都成功后才会反序列化；因为 `torch.load` 是 pickle-based，只能使用可信文件。

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe -m moiredet_repro.cli infer --input .\MoireDet\script\00002423.png --checkpoint .\weights\PSENet_100_loss0.000000.pth --checkpoint-manifest .\weights\PSENet_100_loss0.000000.pth.json --output .\outputs\official-sample --device cuda
```

`--output` 必须是新建且不存在的目录叶节点；已有目录、文件或任何目标文件都会拒绝，避免覆盖结果。一次成功运行恰好生成四个文件：

- `prediction.npy`：未显示归一化的 `320 x 320 float32` 原始预测；
- `moire_map.png`：恢复输入尺寸的 8 位灰度图；
- `comparison.png`：左侧原图、右侧检测图；
- `run.json`：输入、上游提交、权重可信状态、运行时、预测范围、性能和输出清单。

逐图 min-max 仅用于 PNG 显示，不会改写 `prediction.npy`。输入采用 OpenCV BGR、直接双线性缩放到 320×320、ImageNet mean/std、batch size 1；中文 Windows 路径可用。

## RTX 正式基准

在可信权重已验证后，基准固定为 5 次预热和 20 次计时：

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe .\scripts\benchmark_inference.py --input .\MoireDet\script\00002423.png --checkpoint .\weights\PSENet_100_loss0.000000.pth --checkpoint-manifest .\weights\PSENet_100_loss0.000000.pth.json --output .\outputs\acceptance\benchmark.json --device cuda
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest -m checkpoint -v -rs
```

没有权重或 `MOIREDET_*` 检查点环境变量时，检查点测试应为 `SKIPPED`，不是 `PASSED`。

## 排障

- `checkpoint file is missing`：补齐原始权重及同名 manifest，不要改全局 Python。
- `untrusted source_type` 或 `SHA-256 mismatch`：补齐可信来源证据，重新计算真实文件哈希。
- `strict state_dict mismatch`：文件不是目标 `TripleBranchWithSpecificConv` 权重，停止使用它。
- `--device cuda` 但 CUDA 不可用：检查 NVIDIA 驱动与 PyTorch 的 cu113 构建。
- 输出冲突：使用新的、尚不存在的 `--output` 目录/基准 JSON 文件。
- NaN/Inf：此次输出无效，不能作为检测结果。

仍无法取得可信权重时，可直接发送 `docs/checkpoint-request-message.md`。
