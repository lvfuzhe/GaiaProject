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

默认的 `standard-v14` 规则引擎已覆盖六轮主流程、完整建筑链、六条科研轨、能量碗、
盖亚计划、科技选择、联邦、助推板块和计分，并加入可复现的随机地图、种族、计分与科技设置。
它是面向固定动作空间的 AI 规则核心，不是对实体桌游所有图案与种族能力的逐项复刻。
早期 `MiniGaia` 规则仍保留用于快速回归。

## 快速开始

需要 Python 3.11 或更高版本。训练默认使用 PyTorch；有 CUDA 时会自动选择 GPU。

```powershell
python -m pip install -e .
gaiazero demo --simulations 32 --show-actions
```

不安装 editable package 时，也可以从仓库直接运行：

```powershell
$env:PYTHONPATH = "src"
python -m gaiazero demo --ruleset standard --simulations 32
```

运行一个小型自博弈训练：

```powershell
gaiazero train `
  --ruleset standard `
  --players 2 `
  --iterations 5 `
  --games-per-iteration 8 `
  --simulations 64 `
  --output runs/gaia-standard.pt
```

评测检查点，每局轮换神经网络所在座位：

```powershell
gaiazero evaluate runs/gaia-standard.pt --ruleset standard --players 2 --games 20 --simulations 128
```

按人数分别训练三个独立模型（顺序执行 2、3、4 人训练）：

```powershell
gaiazero train-all `
  --iterations 5 `
  --games-per-iteration 8 `
  --simulations 64 `
  --output-dir runs/models `
  --metrics-dir runs/metrics-by-players
```

输出文件为 `runs/models/gaia-standard-2p-nnue.pt`、`gaia-standard-3p-katago.pt`、
`gaia-standard-4p-katago.pt`，对应指标分别写入
`runs/metrics-by-players/metrics-standard-2p-nnue.jsonl`、
`metrics-standard-3p-katago.jsonl` 和 `metrics-standard-4p-katago.jsonl`。
评测时必须使用匹配人数和架构的检查点：

- 2 人局固定使用 NNUE：双玩家特征累加器、平方裁剪激活和反对称价值头。
- 3/4 人局固定使用 KataGo 风格网络：全局门控残差塔、独立策略头和多人价值头。

架构选择由 `--players` 自动完成，并写入训练指标和检查点；不能把 NNUE 与 KataGo
检查点跨人数加载。由于当前观察是结构化扁平向量，KataGo 网络使用适配该状态表示的
全局门控残差块，而不是围棋棋盘卷积。

```powershell
gaiazero evaluate runs/models/gaia-standard-4p-katago.pt --players 4 --games 20 --simulations 128
```

`standard` 是 `demo`、`train` 和 `evaluate` 的默认规则集。需要运行旧版快速模型时使用
`--ruleset mini`；两个规则集的观察维度和动作维度不同，检查点不能混用。
标准规则的 2、3、4 人局也使用独立检查点：地图尺寸、观察维度和价值头输出人数不同，
不能将一个人数的检查点直接用于另一个人数。

## 训练监控台

训练命令默认把结构化事件写入 `runs/metrics.jsonl`。在另一个终端启动只读仪表盘：

```powershell
gaiazero dashboard --metrics runs/metrics.jsonl --port 8765
```

浏览器访问 `http://127.0.0.1:8765`。监控台包含概览、人工对战、自博弈、历史回放和诊断
五个视图。人工对战内按“初始设置 → 角色与对局”两步工作区配置随机拼接地图、种族座位、
计分/科技/助推板块，并可在同一局中切换人工与 AI；训练模型不会读取这套人工对局配置。

初始设置地图使用本地化的实体星区扫描图；图片来源与版权说明见
[`src/gaiazero/web/assets/sectors/ATTRIBUTION.md`](src/gaiazero/web/assets/sectors/ATTRIBUTION.md)。

历史回放按训练运行、迭代和对局建立索引。每局可使用滑杆或播放控制逐步检查棋盘、资源、
科研轨与动作账本；页面同时检查轨迹连续性、轮次顺序、资源上限、星球所有权、终局状态和
规则引擎动作转移。新训练会记录每一步规则状态，旧日志缺少的步骤会标记为“需补录”。

训练过程会记录每一步棋盘状态，并默认每隔四步附带一次完整搜索候选；可按运行规模调整
搜索详情的采样间隔：

```powershell
gaiazero train --metrics runs/experiment-a.jsonl --metrics-move-interval 8
gaiazero dashboard --metrics runs/experiment-a.jsonl --port 8765
```

Dashboard 与训练进程相互独立，只读取 JSONL 文件，不加载模型或占用 GPU 显存。文件保留后，
训练结束或异常退出时仍可查看最后状态。

CPU 冒烟训练可以显著缩小参数：

```powershell
gaiazero train `
  --iterations 1 `
  --games-per-iteration 1 `
  --updates-per-iteration 1 `
  --eval-games 0 `
  --simulations 2 `
  --hidden-size 32 `
  --residual-blocks 1 `
  --batch-size 8 `
  --output runs/smoke.pt
```

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
src/gaiazero/game/gaia_state.py standard-v14 AI 规则核心
src/gaiazero/game/mini_gaia.py  旧版轻量规则切片
src/gaiazero/mcts.py            多玩家 PUCT/PIMCTS
src/gaiazero/model.py           PyTorch 策略/价值网络与检查点
src/gaiazero/selfplay.py        自博弈数据生成
src/gaiazero/replay.py          有界经验回放
src/gaiazero/training.py        AlphaZero 联合损失训练器
src/gaiazero/arena.py           座位轮换评测
src/gaiazero/telemetry.py       结构化 JSONL 训练事件
src/gaiazero/dashboard.py       只读监控 HTTP 服务
src/gaiazero/web/               响应式监控页面
src/gaiazero/cli.py             命令行入口
tests/                          规则、搜索和训练回归测试
```

## 当前规则范围

| 模块 | 状态 |
|---|---|
| 2-4 人、六轮与行动顺序 | 已实现 |
| 2 人 7 星区、3/4 人 10 星区的 BGA 风格随机星球地图 | 已实现；双人 40 颗、三/四人 61 颗 |
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
| Terrans、Lantids、Xenos、Gleens、Taklons、Geodens | 已实现部分核心差异 |

为控制 AlphaZero 的分支数，当前规则存在明确的建模约束：星区来源仍使用固定紧凑坐标模板，
但星球数量和地形配额按 BGA/实体组件（42 颗母星、12 颗超维星、7 颗盖亚星）建模；
被动充能采用“可承受时自动接受”；资源支付和
联邦连接只生成一个规范方案。14 个种族均可参与随机设置，但多数种族特殊行动、高级科技效果、
完整 QIC 自由转换、失落星球和部分种族特殊行动仍未实现。训练结果应视为 `standard-v14` 环境内的策略，不应直接
作为官方比赛裁决器。`standard-v14` 统一了 BGA 板块编号、科研逐级收益、初始科研收益、公共能量与 Q.I.C. 行动、学院类型及助推特殊行动，并加入 Taklons 脑石、Terrans 行星研究院的盖亚阶段资源兑换、Lantids 共存矿场和行星研究院奖励、Gleens 的 Q.I.C. 转换/盖亚殖民/专属联邦板块、Ivits 的空间站/单一联邦扩建/Q.I.C. 卫星规则，以及 Bal T'aks 的航行锁定和盖亚塑形者兑换；旧版检查点的状态与动作维度不同，不能直接沿用。

人工对战入口为 `http://127.0.0.1:8765/play`；旧的
`http://127.0.0.1:8765/setup/random` 和 `http://127.0.0.1:8765/setup/manual` 路径仍会兼容打开
人工对战的初始设置步骤，也可以在页面内切换 BGA 随机与手动星区路径。手动路径
支持选中任意单颗星球并将其放置到整张拼接星图的合法空六角格，也可以按地形新增或删除星球；
提交预览时会检查数量上限、素材来源、重叠、边界、同色母星相邻和起始母星数量限制。

完整版规则的模块边界与实施顺序见 [docs/full-rules-roadmap.md](docs/full-rules-roadmap.md)。

## 测试

测试只依赖 Python 标准库的 `unittest`：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```
