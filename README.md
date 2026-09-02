# GaiaZero

GaiaZero 是一个面向《盖亚计划》类确定性多人策略游戏的 AlphaZero + PIMCTS
研究项目。当前版本提供一条可以直接运行的完整链路：

```text
不可变规则状态
    -> 多玩家 PUCT/PIMCTS
    -> 策略/价值网络
    -> 自博弈经验回放
    -> 策略损失 + 多玩家价值损失
    -> 竞技场评测与检查点
```

默认的 `standard-v22` 规则引擎已覆盖六轮主流程、完整建筑链、六条科研轨、能量碗、
盖亚计划、科技选择、联邦、助推板块和计分，并加入可复现的随机地图、种族、计分与科技设置。
它是面向固定动作空间的 AI 规则核心，不是对实体桌游所有图案与种族能力的逐项复刻。
项目只保留正式 `GaiaState`（`standard-v22`）规则环境。

## 快速开始

需要 Python 3.11 或更高版本。训练默认使用 PyTorch；有 CUDA 时会自动选择 GPU。

```powershell
python -m pip install -e .
gaiazero demo --simulations 32 --show-actions
```

不安装 editable package 时，也可以从仓库直接运行：

```powershell
$env:PYTHONPATH = "src"
python -m gaiazero demo --simulations 32
```


同步 `train` 和 `train-all` 已删除。所有模型训练统一使用五进程异步 NPZ 流水线，
可从 Dashboard 一键启动，也可以在终端运行：

```powershell
gaiazero pipeline --players 4 --device auto
```

评测检查点，每局轮换神经网络所在座位：

```powershell
gaiazero evaluate runs/gaia-standard.pt --players 2 --games 20 --simulations 128
```

不同人数分别使用独立的流水线目录和模型。评测时必须使用匹配人数和架构的检查点：

- 2/3/4 人局统一使用 KataGo 风格网络：全局门控残差塔、独立策略头和多人价值头。

架构选择由 `--players` 自动完成并写入检查点；2/3/4 人局都使用同一
KataGo 网络族，但每个人数仍使用匹配的价值头维度。由于当前观察是结构化扁平向量，网络使用适配该状态表示的
全局门控残差块，而不是围棋棋盘卷积。

```powershell
gaiazero evaluate runs/models/gaia-standard-4p-katago.pt --players 4 --games 20 --simulations 128
```

环境固定为 `standard-v22`；2/3/4 人模型的观察维度和动作维度不同，检查点不能混用。
标准规则的 2、3、4 人局也使用独立检查点：地图尺寸、观察维度和价值头输出人数不同，
不能将一个人数的检查点直接用于另一个人数。

## 异步训练监控台

启动仪表盘：

```powershell
gaiazero dashboard --port 8765
```

浏览器访问 `http://127.0.0.1:8765`。概览页可以配置人数、设备、MCTS、NPZ 分片、PyTorch
训练和守门参数，并一键启动或停止以下五个独立 Python 进程：

- 自对弈：持续写入完整对局 `raw/*.npz`
- 洗牌：合并并打包为 `shuffled/*.npz`
- 训练：原生 PyTorch 读取 NPZ，生成候选 `.pt`
- 导出：把已批准权重转换为 GaiaZero `.bin`
- 守门测试：候选权重对阵当前权重，达标后批准

导航栏为五个进程分别提供监控页面，展示进程状态、原生状态文件、最近产物、结构化训练或
守门记录以及日志尾部。监控数据位于 `runs/multiplayer-pipeline`，不再读取
`runs/metrics.jsonl`，旧的自博弈棋盘监控页和事件诊断页已删除。

人工对战会按局原子保存到 Dashboard 数据目录同级的 `history` 目录，默认即
`runs/history/play-*.json`。可以显式指定其他本地目录：

```powershell
gaiazero dashboard --history-dir runs/play-history --port 8765
```

人工对战内按“初始设置 → 角色与对局”两步工作区配置随机拼接地图、种族座位、
计分/科技/助推板块，并可在同一局中切换人工与 AI；系统会按人数寻找匹配的 KataGo 检查点，
找不到时使用启发式 PIMCTS。训练模型不会读取这套人工对局配置。

在 `http://127.0.0.1:8765/import/bga` 输入 BGA 账号、密码和一条已结束对局的
`gamereview?table=...` 或 `archive/replay/...` 地址，可以手动下载单局复盘。勾选“加密保存”后，
账号、密码和 Cookie 会通过 Windows DPAPI 绑定到当前系统用户，并保存到历史目录中的
`.bga-session.bin`；它们不会写入复盘 JSON，也不会由 API 返回给前端。页面可随时清除保存的会话。
转换结果保存为
`runs/history/bga-<table>.json`（或 `--history-dir` 指定的目录），重复导入同一桌号会原子更新
同一文件。完成后可在“历史回放”中查看连续行动、玩家资源、科研轨、建筑和星图；BGA 原始
通知也会保存在文件中。每个步骤额外记录 VP 前值、计分通知、增减值和后值，并核对 BGA 结果页
终局分数，便于对照板块编号、开销、收入和计分修复规则。

初始设置地图使用本地化的实体星区扫描图；图片来源与版权说明见
[`src/gaiazero/web/assets/sectors/ATTRIBUTION.md`](src/gaiazero/web/assets/sectors/ATTRIBUTION.md)。

历史回放只读取 `history` 目录中已经物化为 JSON 的本地人工对战、BGA 导入和手动转换后的 NPZ 回放。
训练过程产生的原始/洗牌 NPZ 不会自动进入历史回放。人工对战在新建、每步行动、角色切换和撤销后
自动更新本地 JSON，关闭并重启仪表盘后仍可从“历史回放”选择加载。每局可使用滑杆或播放
控制逐步检查棋盘、资源、科研轨与动作账本；页面同时检查轨迹连续性、轮次顺序、资源上限、
星球所有权、终局状态和规则引擎动作转移。手动执行 `npz-to-history` 后生成的 `training_npz` JSON
副本会出现在历史回放中，并可直接删除；历史工具栏的“导入 NPZ”按钮也可以选择本地 NPZ 完成同样转换。
删除只影响 JSON 副本，不会删除原始 NPZ。Dashboard 直接读取异步流水线的状态 JSON、产物目录
和日志；监控本身不加载训练模型，也不占用 GPU 显存。

## 核心设计

`GameState` 是搜索与规则引擎之间唯一的必需边界。状态不可变，并提供固定长度观察、
合法动作掩码、状态转移和终局多人价值。未来替换为完整规则引擎时，搜索和训练代码不需要
理解具体规则。

PUCT 搜索始终回传绝对玩家顺序的价值向量：

```text
V(s) = [v0, v1, ..., vn]
```

节点选择使用当前行动玩家的 `Q[player]`。实现中没有二人零和算法常见的价值正负翻转，
所以同一搜索器可以处理二至四人对局。

策略网络只在合法动作上归一化。训练目标来自根节点访问次数，价值目标来自终局两两排名：
第一名趋近 `+1`，最后一名趋近 `-1`，中间名次保留连续多人反馈。

## 目录

```text
src/gaiazero/core.py            通用游戏和评估器协议
src/gaiazero/game/gaia_setup.py 随机初始设置生成器
src/gaiazero/game/gaia_state.py standard-v22 AI 规则核心
src/gaiazero/mcts.py            多玩家 PUCT/PIMCTS
src/gaiazero/model.py           PyTorch 策略/价值网络与检查点
src/gaiazero/selfplay.py        自博弈数据生成
src/gaiazero/replay.py          有界经验回放
src/gaiazero/training.py        AlphaZero 联合损失训练器
src/gaiazero/arena.py           座位轮换评测
src/gaiazero/pipeline_monitor.py 五进程控制与状态采集
src/gaiazero/dashboard.py       监控及本地历史 HTTP 服务
src/gaiazero/web/               响应式监控页面
src/gaiazero/cli.py             命令行入口
tests/                          规则、搜索和训练回归测试
```

## 当前规则范围

| 模块 | 状态 |
|---|---|
| 2-4 人、六轮与行动顺序 | 已实现 |
| BGA 风格随机星球地图与人数地图规模 | 已实现；2 人固定 7 星区/40 颗星球，3 人可选 10 星区/61 颗或小地图 8 星区/49 颗，4 人固定 10 星区/61 颗 |
| 7 张双面种族板、14 个种族随机分配与起始放置 | 已实现 |
| 资源、收入与信用点/矿石/知识上限 | 已实现 |
| 七种母星地形、盖亚/跨维星球、航行与改造成本 | 已实现 |
| 8 矿场、4 贸易站、3 研究所、1 行星研究院、2 学院 | 已实现 |
| 改造、航行、人工智能、盖亚、经济、科学六条科研轨及逐级收益 | 已实现 |
| 三个能量碗、充能/消耗/弃置和公共能量行动 | 已实现 |
| 盖亚塑形者、盖亚区能量、盖亚阶段和占领 | 已实现 |
| 9 个基础科技与 6 个高级科技的随机摆放 | 已实现选择、覆盖、行动与持续收益 |
| 6 个回合计分、2 个终局计分与随机联邦板块 | 已实现 |
| 联邦强度、最少卫星连接、联邦令牌与科研钥匙 | 已实现的规范化版本 |
| 玩家数 + 3 个助推板块、末位到首位开局选取、过轮归还/重选、两项助推特殊行动 | 已实现 |
| Terrans、Lantids、Xenos、Gleens、Taklons、Ambas、Hadsch Hallas、Ivits、Geodens、Bal T'aks、Firaks、Bescods、Nevlas、Itars | 已实现部分核心差异 |

为控制 AlphaZero 的分支数，当前规则存在明确的建模约束：星区来源仍使用固定紧凑坐标模板，
但星球数量和地形配额按 BGA/实体组件（42 颗母星、12 颗超维星、7 颗盖亚星）建模；
被动充能按行动者后的顺时针顺序逐个询问玩家，可选择接受或拒绝；充能量按范围 2 内被动方最高建筑等级计算，扣分为实际充能量减 1 VP。资源支付和
联邦连接只生成一个规范方案。14 个种族均可参与随机设置，但多数种族特殊行动、高级科技效果、
完整 QIC 自由转换和部分种族特殊行动仍未实现。训练结果应视为 `standard-v22` 环境内的策略，不应直接
作为官方比赛裁决器。`standard-v22` 统一了 BGA 板块编号、科研逐级收益、初始科研收益、科技板块的可选科研推进、公共能量与 Q.I.C. 行动、航行 5 级失落星球、学院类型及助推特殊行动，并加入 Taklons 脑石、Terrans 行星研究院的盖亚阶段资源兑换、Lantids 共存矿场和行星研究院奖励、Gleens 的 Q.I.C. 转换/盖亚殖民/专属联邦板块、Ivits 的空间站/单一联邦扩建/Q.I.C. 卫星规则、Bal T'aks 的航行锁定和盖亚塑形者兑换、Firaks 的研究所降级与免费科研、Bescods 的最低科研轨行动和研究院强度规则、Nevlas 的盖亚区知识转换、可任意搭配的双倍资源兑换与公共能量工位，以及 Itars 燃烧能量入盖亚区和研究院科技兑换。`standard-v22` 增加被动充能决策状态和接受/拒绝动作，因此 v21 检查点的输入及策略输出维度均不兼容。

人工对战入口为 `http://127.0.0.1:8765/play`；旧的
`http://127.0.0.1:8765/setup/random` 和 `http://127.0.0.1:8765/setup/manual` 路径仍会兼容打开
人工对战的初始设置步骤，也可以在页面内切换 BGA 随机与手动星区路径。手动路径
支持三人局在标准 10 星区与 BGA 推荐的 8 星区小地图之间切换；两人局固定为 7 星区小地图，
四人局固定为 10 星区标准地图。手动编辑时可选中任意单颗星球并将其放置到整张拼接星图的合法空六角格，
也可以按地形新增或删除星球；
提交预览时会检查数量上限、素材来源、重叠、边界、同色母星相邻和起始母星数量限制。

完整版规则的模块边界与实施顺序见 [docs/full-rules-roadmap.md](docs/full-rules-roadmap.md)。

## 测试

测试只依赖 Python 标准库的 `unittest`：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## C++/CMake 基础工程

仓库同时提供一个可独立编译的 C++20 基础工程，用于后续 C++ selfplay、规则引擎和 TensorRT 推理适配。当前默认只构建契约与推理接口 smoke 测试，不要求 NVIDIA GPU、CUDA 或 TensorRT SDK。

在 VS2026 Build Tools 的 Developer PowerShell 中运行：

```powershell
cmake --preset windows-msvc-ninja
cmake --build --preset windows-msvc-ninja
ctest --preset windows-msvc-ninja
```

普通 PowerShell 可先加载 `VC\Auxiliary\Build\vcvars64.bat`，完整说明见 [`docs/cpp-cmake.md`](docs/cpp-cmake.md)。CUDA、ONNX Runtime 和 TensorRT 通过 `GAIA_ENABLE_CUDA`、`GAIA_ENABLE_ORT_CPU`、`GAIA_ENABLE_TENSORRT` 可选开关接入，SDK 根目录分别由 CMake/CUDA 环境、`ONNXRUNTIME_ROOT`、`TENSORRT_ROOT` 指定。启用 ONNX Runtime 后，`gaiazero::OnnxRuntimeCpuBackend` 会校验模型签名并执行 CPU 参考推理。
