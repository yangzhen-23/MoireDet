# MoireDet 单图推理复现

这是论文 *Doing More With Moiré Pattern Detection in Digital Photos* 的考核导向复现工程。上游固定为 [cong-yang/MoireDet](https://github.com/cong-yang/MoireDet) 提交 `afde899f3c3beee96160610ee450618136a38f7b`；适配层以该快照为准，详见 `docs/upstream/UPSTREAM.md`。

## 当前状态

- 代码兼容层和无权重验证已完成：固定环境、配置、官方模型构建、随机前向、CLI 和四文件输出契约均可验证。
- 可信检查点与 RTX 4060 正式验收仍为 **SKIPPED / 未完成**。当前没有可信的 `PSENet_100_loss0.000000.pth`、其来源证据和真实 SHA-256；因此不能声称完成有权重复现或正式验收。

## 固定环境

目标环境为 Python 3.8.20、PyTorch `1.10.0+cu113`、torchvision `0.11.1+cu113`、CUDA 11.3、OpenCV wheel `4.11.0.86`、NumPy 1.24.3、PyYAML 6.0.2、einops 0.3.0 和 local-attention 1.2.1。

```powershell
D:\anaconda3\Scripts\conda.exe env create -f .\environment.yml
D:\anaconda3\envs\moiredet-repro\python.exe -m pip install -e . --no-deps
D:\anaconda3\envs\moiredet-repro\python.exe -m pip check
```

无权重诊断不会读取检查点、调用 `torch.load` 或联网：

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe .\scripts\verify_environment.py --device all
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest -m "not checkpoint" -v
```

随机前向只证明环境和网络结构可运行，不证明检测质量。

## 可信检查点验收门

不要下载、伪造或提交权重、manifest、个人图片或运行输出。权重与验收产物均由 `.gitignore` 排除。只有从允许的来源获得原始权重并记录可核验来源证据后，才可在 `weights/PSENet_100_loss0.000000.pth` 放置文件，并依照 `weights/checkpoint.example.json` 的 schema 创建同名 `.pth.json` manifest。manifest 的 SHA-256 必须来自该文件的真实哈希。

```powershell
$env:MOIREDET_CHECKPOINT = (Resolve-Path .\weights\PSENet_100_loss0.000000.pth).Path
$env:MOIREDET_CHECKPOINT_MANIFEST = (Resolve-Path .\weights\PSENet_100_loss0.000000.pth.json).Path
$env:MOIREDET_USER_IMAGE = (Resolve-Path .\examples\input\user_moire.png).Path  # 可选

D:\anaconda3\envs\moiredet-repro\python.exe -m pytest tests\integration\test_official_checkpoint.py -v -rs
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest -m checkpoint -v -rs
D:\anaconda3\envs\moiredet-repro\python.exe .\scripts\benchmark_inference.py --input .\MoireDet\script\00002423.png --checkpoint $env:MOIREDET_CHECKPOINT --checkpoint-manifest $env:MOIREDET_CHECKPOINT_MANIFEST --output .\outputs\acceptance\benchmark.json --device cuda
```

该 gate 只有在 `MOIREDET_CHECKPOINT` 和 `MOIREDET_CHECKPOINT_MANIFEST` **都已设置**时才开始验证：两者均未设置或只设置其中一个都会明确 `SKIPPED`；两者都已设置后，缺失路径、来源无效、哈希不匹配或严格 state-dict 不匹配都会失败，绝不静默跳过。用户图片仅控制用户图片用例。正式验收还要求实际 CUDA 设备为 RTX 4060。

在可信输入和 RTX 4060 到位后，验收会严格 CPU 加载权重，再对官方样图和可选用户图执行 CUDA 推理；每个输出目录必须恰好包含：

- `prediction.npy`：原始 `320 x 320 float32` 预测；
- `moire_map.png`：恢复到输入尺寸的灰度图；
- `comparison.png`：输入和检测图的并排图；
- `run.json`：输入、上游提交、已验证权重、预测范围、性能和输出清单。

CLI 的 `--output` 必须是一个新的、尚不存在的目录叶节点；该路径上已有任何文件或目录都会被拒绝，避免覆盖既有结果。成功发布会原子地创建这个叶目录，并且其中恰好只有上述四个文件。每次运行请使用新的带时间戳路径，例如 `outputs\acceptance\2026-07-27T153000-sample`，而不要复用旧输出目录。

基准固定为 5 次预热和 20 次计时，并要求正的延迟以及两项峰值显存字段。重复推理必须满足 `rtol=1e-5, atol=1e-6`。

## 人工视觉检查

在执行正式验收后，打开官方样图和用户图（若提供）的 `comparison.png`，并在 `outputs/acceptance/visual-check.md` 记录：

- 两个 PNG 都能正确解码；
- 图像不是意外全黑或全白；
- 高响应区域与可见的摩尔纹区域相对应；
- 各 `run.json` 的 raw min、max 和 dynamic range；
- 审核人姓名和日期。

只有完成这项人工检查、可信权重验证和 RTX 4060 测试后，才能将状态改为“Task 1 单图推理验收完成”。

最后运行：

```powershell
D:\anaconda3\envs\moiredet-repro\python.exe -m pytest -v
git diff --check
git status --short
```
