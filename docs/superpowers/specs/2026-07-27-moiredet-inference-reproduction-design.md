# MoireDet 单图推理复现设计

- 日期：2026-07-27
- 状态：已获用户确认
- 阶段：苏州大学 BeeLab 夏令营考核任务 1

## 1. 背景

考核要求依次完成：在个人电脑上跑通 MoireDet、使用本人手机拍摄的摩尔纹视频制作左右对比 Demo，以及制作一份 15 分钟汇报 PPT。本设计仅覆盖第一阶段的代码复现，并确保产物可以被第二阶段视频处理直接复用。

指定论文为 *Doing More With Moiré Pattern Detection in Digital Photos*（IEEE TIP 2023，DOI `10.1109/TIP.2022.3232232`）。指定实现为作者仓库 `cong-yang/MoireDet`。本项目固定使用已审计的上游提交 `afde899f3c3beee96160610ee450618136a38f7b`，避免 `main` 分支后续变化影响结果。

## 2. 已确认事实与约束

### 2.1 论文任务

MoireDet 是图像到图像的回归模型，而不是去摩尔纹模型。输入是包含摩尔纹的 RGB 图像，论文训练尺寸为 `320 x 320 x 3`；输出是单通道灰度摩尔纹边缘图，用于表达摩尔纹的位置、形状和强度。

模型由高层、低层和空间三个编码部分组成，核心组件包括 ResNet-18、两层 BiFPN、Performer 和位置相关的 `5 x 5` 自适应卷积核。第一阶段只运行作者模型，不重写网络，也不尝试修正论文与代码之间的理论差异。

### 2.2 上游工程状态

作者 README 指定 CUDA 10.0、PyTorch 1.4 和 torchvision 0.5。样例脚本还包含作者机器上的绝对路径、固定 GPU 编号、冗余训练配置和不完整依赖，因此不能直接在当前 Windows 电脑上运行。

作者提供的单图样例执行以下预处理：OpenCV 读取、直接缩放到 `320 x 320`、`ToTensor`、ImageNet 均值方差归一化。样例没有执行 BGR 到 RGB 转换，因此兼容复现默认保留 BGR 通道顺序，并在文档中明确记录。

### 2.3 本机环境

目标机器为 Windows，GPU 是 NVIDIA GeForce RTX 4060 Laptop GPU，显存 8 GB。现有 `exp` Conda 环境使用 Python 3.8、PyTorch 1.10.0+cu113 和 torchvision 0.11.1+cu113，已通过实际 CUDA 张量运算验证。

实施时不直接修改 `exp`，而是从它复制一个独立的 `moiredet-repro` 环境。所有新增依赖在通过模型构建和前向测试后冻结到环境清单中。

### 2.4 权重约束

当前工作区没有 `PSENet_100_loss0.000000.pth`。作者仓库给出的 Google Drive 权重链接在公开 Issues 中已有失效反馈，当前环境也无法连通该下载地址。

权重是完整推理验收的必要前置条件。项目只接受以下来源：作者仓库当前链接、论文作者或作者团队、考核导师直接提供，或能够证明与官方文件完全一致的可信镜像。不得用来源不明的模型假装完成复现。

若公开渠道无法取得权重，项目仍完成环境、模型构建、随机输入前向、命令行接口和无权重测试，但明确标记“集成验收等待可信权重”，并生成一段可直接发给考核导师的权重索取消息。从头训练不自动进入本阶段范围。

## 3. 目标与非目标

### 3.1 目标

1. 在独立 Conda 环境中稳定导入并构建官方 MoireDet。
2. 提供一条可重复执行的单图推理命令。
3. 支持 CUDA 和 CPU 设备选择，默认优先使用 CUDA。
4. 同时保存原始浮点预测、灰度检测图和左右对比图。
5. 记录上游提交、环境版本、权重来源与 SHA-256、运行参数、耗时和显存信息。
6. 提供自动化测试和中文运行文档。
7. 保持推理核心可被后续逐帧视频流水线复用。

### 3.2 非目标

本阶段不执行以下工作：

- 下载完整 MoireScape 并从头训练；
- 复现论文表 III、表 IV 或其他定量指标；
- 修改 MoireDet 网络结构或损失函数；
- 使用 MoireDet+ 或其他模型替代指定模型；
- 制作最终视频 Demo；
- 制作 PPT。

这些工作只有在任务 1 验收完成后，或用户明确扩大范围时才进入后续设计。

## 4. 方案选择

评估过三条路线：

1. **官方源码兼容迁移与可信权重恢复**：保留论文指定模型，只在外围增加可移植推理层。该路线与考核最一致，选为主方案。
2. **下载数据后从头训练**：理论上可绕过权重缺失，但数据超过 8 GB、训练约需数十小时，且论文缺失部分生成参数，不适合作为第一阶段默认路线。
3. **使用第三方模型或权重**：交付速度较快，但来源与指标无法充分核验，不能证明复现了指定论文，因此拒绝作为正式方案。

## 5. 项目边界与目录

最终项目位于独立目录 `moire_reproduction/`，自身使用 Git 管理，不把论文 PDF、考核原文或临时调查目录纳入版本库。

```text
moire_reproduction/
  docs/
    superpowers/specs/
  upstream/
    MoireDet/
    UPSTREAM.md
  patches/
    0001-disable-resnet-online-download.patch
  src/
    moiredet_repro/
      __init__.py
      cli.py
      checkpoint.py
      inference.py
      preprocessing.py
      rendering.py
      upstream_adapter.py
  configs/
    inference.yaml
  examples/
    input/
  tests/
  scripts/
  weights/
    README.md
    checkpoint.example.json
  outputs/
  pyproject.toml
  environment.yml
  README.md
  .gitignore
```

`upstream/MoireDet/` 保存固定提交的纯源码快照；外围兼容代码不反向混入上游目录。`weights/` 和 `outputs/` 默认不提交，避免误提交大文件和个人素材。

## 6. 组件设计

### 6.1 上游快照与来源记录

`UPSTREAM.md` 记录仓库 URL、提交哈希、获取日期和已做的完整性检查。上游快照通过 Git archive 导出，不携带嵌套 `.git` 目录。若必须对上游代码做最小兼容修改，每一处修改都以补丁文件或清晰的提交记录呈现，不能无记录地改写核心网络。

已确认目标类在构造注意力分支时硬编码 `backbone_model(pretrained=True)`，会隐式下载 ImageNet 权重。项目通过 `patches/0001-disable-resnet-online-download.patch` 仅把 `TripleBranchWithSpecificConv` 中这一处改为 `pretrained=False`。严格加载完整 MoireDet 状态字典后，初始化值会被检查点覆盖；若任何参数未覆盖，严格加载直接失败，因此该补丁不改变已验收检查点的推理参数。补丁内容、应用命令和应用后的文件哈希写入 `UPSTREAM.md`。

`upstream_adapter.py` 从已安装包位置解析仓库根目录，同时校验 `upstream/MoireDet/lib` 和 `upstream/MoireDet/script/performer_pytorch` 存在；随后只把仓库内的 `upstream/MoireDet` 与 `upstream/MoireDet/script` 两个相对位置加入当前进程搜索路径，再导入作者的 `lib.models` 和随仓库提供的 `performer_pytorch`。它不依赖当前工作目录、环境变量或作者绝对路径；首版只支持从完整源码检出目录执行，缺少任一上游组件时明确失败。

### 6.2 环境层

环境层负责创建独立 `moiredet-repro` Conda 环境，并冻结以下类别的实际验证版本：Python、PyTorch、torchvision、OpenCV、NumPy、PyYAML、tqdm、einops 与 local-attention。项目使用 `pyproject.toml` 声明 `src/` 包布局；环境建立后执行 `python -m pip install -e . --no-deps`，确保从仓库根目录可运行模块命令，第三方依赖仍只由 `environment.yml` 管理。

环境验证包含两步：先执行最小 CUDA 张量运算，再执行 MoireDet 随机输入前向。仅“能够识别 GPU”不算通过。

### 6.3 检查点管理

`checkpoint.py` 接收检查点文件和一份来源清单。来源清单默认使用 `<checkpoint-path>.json`（例如 `model.pth.json`），也允许由 `--checkpoint-manifest` 显式指定；至少包含文件名、来源类型、来源说明或 URL、获取日期、来源证据说明和预期 SHA-256。正式推理不接受缺失必填字段或哈希不一致的来源清单。仓库提供 `weights/checkpoint.example.json` 模板，但不提交权重本体。

可信性与完整性分开判断：来源必须先属于第 2.4 节允许的可信渠道，来源清单保存可复核的作者 URL、作者回复或导师提供记录；SHA-256 随后在首次取得文件时计算，用于锁定该字节副本，不能单独证明来源可信。只有“来源证据合格”和“实际哈希匹配”同时成立，运行记录才标记 `checkpoint_verified=true`。

`checkpoint.py` 负责：

- 检查文件是否存在且可读；
- 先映射到 CPU，再加载状态字典；
- 只接受与作者样例一致、含顶层 `state_dict` 映射的检查点封装；
- 兼容去除 DataParallel 生成的 `module.` 前缀；
- 以严格模式校验参数名称和形状；
- 拒绝空文件、HTML 下载页和结构不匹配的文件；
- 计算 SHA-256，与来源清单中的预期值精确比对，并将来源、文件大小、哈希和获取日期写入运行记录。

模型构建与检查点加载全过程禁止网络访问，因为完整 MoireDet 检查点应提供所需参数，且离线复现不能依赖隐式网络请求。

### 6.4 预处理

默认预处理严格贴近官方样例：

1. OpenCV 以 BGR 读取输入；
2. 检查文件存在、图像非空且具有 3 个通道；
3. 直接双线性缩放到 `320 x 320`；
4. 转成 `[0, 1]` 浮点张量；
5. 按 ImageNet mean `[0.485, 0.456, 0.406]` 和 std `[0.229, 0.224, 0.225]` 归一化；
6. 添加 batch 维度并移动到目标设备。

原始图像及其宽高在预处理前保留，供输出恢复和对比图使用。

### 6.5 推理服务

`inference.py` 封装模型生命周期：构建模型、加载检查点、切换 `eval()`、选择设备并在 `torch.no_grad()` 下执行前向。模型契约固定为作者 `sample_code.json` 中的 `TripleBranchWithSpecificConv`，参数固定为 `backbone=resnet18`、`fpem_repeat=2`、`segmentation_head=FPEM_FFM`、`is_dct=false`、`is_light=true`；配置中的 `pretrained=true` 只表示作者原始初始化意图，离线兼容补丁按第 6.1 节处理实际构建。

目标模型的返回契约是二元组 `([moire_density], fea_loss)`。输出选择严格沿用作者 `sample_code.py` 的语义，即从 `model(img)[0][0][0][0]` 取得首个样本的二维预测；兼容层对二元组、单元素预测列表、批次/通道维和最终 `320 x 320` 形状逐层显式校验，不以启发式规则猜测其他输出。服务返回统一的二维 `float32` 摩尔纹边缘图，不把显示归一化混入模型结果。首版固定 batch size 为 1，不引入 AMP、模型编译或并发等非必要优化。

### 6.6 输出与可视化

每次推理生成一个独立输出目录，至少包含：

- `prediction.npy`：未经显示归一化的 `320 x 320` 二维 `float32` 模型输出；
- `moire_map.png`：将预测图双线性恢复到输入原始宽高后得到的 8 位灰度检测图；
- `comparison.png`：左侧原图、右侧同宽高检测图，画布高度等于原图高度、宽度等于原图宽度的两倍；
- `run.json`：输入路径、设备、环境、上游提交、权重哈希、图像尺寸、耗时和输出范围。

灰度图采用确定性的逐图 min-max 显示流程：先在原始 `320 x 320` 浮点预测上计算最小值、最大值和动态范围；动态范围小于或等于 `1e-12` 时生成全零图，否则线性映射到 `[0, 255]`；再以双线性插值恢复原始宽高，裁剪到 `[0, 255]`、四舍五入并转为 `uint8`。`run.json` 同时记录原始最小值、最大值和动态范围，避免仅凭拉伸后的 PNG 判断信号强弱。该操作只影响 PNG，不改变 `prediction.npy`。视频阶段将另行设计跨帧固定归一化，避免逐帧 min-max 造成闪烁。

### 6.7 命令行接口

用户入口为：

```text
python -m moiredet_repro.cli infer --input <image> --checkpoint <pth> --checkpoint-manifest <json> --output <dir> --device cuda
```

`--checkpoint-manifest` 默认查找与权重同名的 JSON 旁文件。`--device auto` 作为默认值：CUDA 可用时使用 CUDA，否则使用 CPU。`--output` 表示本次运行目录；若其中已经存在任一目标产物，首版直接报错，不覆盖旧结果。README 先要求激活 `moiredet-repro` 并执行一次可编辑安装，之后该命令可直接从仓库根目录运行。命令必须输出清晰的阶段日志和最终产物路径，并以进程退出码区分成功与失败。

## 7. 数据流

```text
输入图片
  -> 格式与通道校验
  -> 官方兼容预处理（BGR、320 x 320、归一化）
  -> MoireDet 前向
  -> 原始二维浮点边缘图
  -> prediction.npy
  -> 仅用于显示的归一化与原尺寸恢复
  -> moire_map.png + comparison.png + run.json
```

模型推理与显示后处理严格分离，保证后续视频模块可以直接复用原始输出，而不受单图可视化策略限制。

## 8. 错误处理

所有可预期失败都返回可操作的信息：

- 缺少权重：说明预期文件、可信来源规则和导师索取方式；
- 权重结构不符：列出缺失键、额外键或首个形状冲突；
- CUDA 不可用：`auto` 模式回退 CPU，显式 `cuda` 模式直接报错；
- GPU 显存不足：释放缓存后报错，不静默降低模型或改变结果；
- 输入损坏或格式不支持：在模型执行前失败；
- 输出目录不可写：不启动推理；
- 依赖缺失：指出缺失包和目标 Conda 环境，不给出修改全局 Python 的指令；
- 预测含 NaN 或 Inf：拒绝写成有效检测结果，并保留诊断日志。

## 9. 测试与验证

### 9.1 不依赖权重的测试

- 配置解析与默认值；
- 正常图像可接受，以及灰度图、损坏文件和缺失文件应拒绝的输入校验；
- 预处理张量形状、类型和数值范围；
- 常量预测与普通预测的显示归一化；
- 输出目录和 `run.json` 结构；
- 模型在 `pretrained=False` 下构建；
- CPU 随机输入前向；
- CUDA 随机输入前向及输出 `1 x 1 x 320 x 320`、全有限值检查。

### 9.2 依赖可信权重的集成测试

- 检查点严格加载；
- 官方样例图在 RTX 4060 上完成推理；
- 原始输出形状严格为 `320 x 320`、数值全有限且动态范围大于 `1e-8`，两个 PNG 的尺寸符合第 6.6 节；
- 将 Python、NumPy 与 PyTorch 随机种子固定为 `2`，设置 `torch.backends.cudnn.benchmark=False` 和 `torch.backends.cudnn.deterministic=True`；相同输入和权重连续运行两次，以 `rtol=1e-5`、`atol=1e-6` 比较原始预测；
- `prediction.npy`、`moire_map.png`、`comparison.png` 和 `run.json` 四个输出文件完整；
- 预热 5 次后，用 `torch.cuda.synchronize()` 包围 20 次单图前向并记录中位数与 P95；在计时前重置峰值统计，并记录 `max_memory_allocated` 与 `max_memory_reserved`；
- 人工检查 `moire_map.png` 和 `comparison.png` 文件可正常打开、没有非预期全黑或全白，并记录高响应区域是否与样例图中肉眼可见的摩尔纹区域对应。

论文没有公开官方样例输出的逐像素基准，因此本阶段不虚构数值相等标准。若后来取得作者输出，则追加基于同一权重、输入和预处理的数值对比。

## 10. 验收标准

仅当以下条件全部满足时，任务 1 才标记完成：

1. 独立环境可以从文档重建；
2. 上游提交固定且来源记录完整；
3. 使用可信检查点，并记录 SHA-256；
4. 一条命令在 RTX 4060 上成功处理上游 `MoireDet/script/00002423.png` 样例和至少一张自选图片；
5. 每次运行生成 `prediction.npy`、`moire_map.png`、`comparison.png` 和 `run.json`；
6. 自动化测试全部通过；
7. README 包含安装、权重放置、运行、输出解释和常见错误；
8. 项目不存在作者机器绝对路径，也不依赖隐式下载；
9. 实际结果经过视觉检查。

如果可信权重仍未取得，则只报告“代码兼容层和无权重验证完成”，不能宣称任务 1 已完成。

## 11. 权重缺失处置

实施阶段按以下顺序处理：

1. 复核作者仓库当前链接及公开 Issues；
2. 检查作者仓库分支、历史和公开 forks 是否存在可核验副本；
3. 检查作者主页或作者团队后续项目是否给出同一检查点；
4. 若仍无可信副本，生成包含论文、仓库、文件名和用途的简短索取消息，由用户发给考核导师；
5. 等待期间继续完成所有无权重工作；
6. 只有用户明确批准扩大范围后，才评估从头训练。

## 12. 后续视频接口

下一阶段视频 Demo 只需新增视频解码、逐帧调用、跨帧固定显示尺度和视频编码。它直接调用本阶段的预处理、推理与原始输出接口，不复制模型加载代码。左右画面同步、分辨率、帧率和编码格式将在视频阶段单独设计。

## 13. 风险与原则

- **权重失效**是首要外部风险，必须显式报告。
- 上游代码仓库未展示明确代码许可证，因此成果应标注来源，不擅自声明代码采用 MIT；数据指针中的许可表述不能代替代码许可。
- 论文与代码的指标、通道顺序或 Performer 张量布局若存在不一致，第一阶段优先复现作者公开样例行为，并把差异记录为汇报素材，不在没有基准的情况下自行“修正”。
- 所有兼容修改必须小、可追踪、可解释，方便在 PPT 中展示遇到的问题和解决过程。
