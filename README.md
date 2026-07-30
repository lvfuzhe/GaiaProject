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

默认的 `standard-v2` 规则引擎已覆盖六轮主流程、完整建筑链、六条科研轨、能量碗、
盖亚计划、科技选择、联邦、助推板块和计分。它是面向固定动作空间的 AI 规则核心，
不是对实体桌游所有版块与 14 个种族的逐项复刻。早期 `MiniGaia` 规则仍保留用于快速回归。

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

`standard` 是 `demo`、`train` 和 `evaluate` 的默认规则集。需要运行旧版快速模型时使用
`--ruleset mini`；两个规则集的观察维度和动作维度不同，检查点不能混用。

## 训练监控台

训练命令默认把结构化事件写入 `runs/metrics.jsonl`。在另一个终端启动只读仪表盘：

```powershell
gaiazero dashboard --metrics runs/metrics.jsonl --port 8765
```

浏览器访问 `http://127.0.0.1:8765`。监控台包含概览、自博弈和诊断三个视图，展示损失曲线、
流水线进度、回放池、吞吐量、六角棋盘、玩家资源、搜索候选、运行参数和原始事件。

训练过程中每隔四步记录一次棋盘与搜索快照；可按运行规模调整：

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
src/gaiazero/game/gaia_state.py standard-v2 AI 规则核心
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
| 资源、收入与信用点/矿石/知识上限 | 已实现 |
| 七种母星地形、盖亚/跨维星球、航行与改造成本 | 已实现 |
| 8 矿场、4 贸易站、3 研究所、1 行星研究院、2 学院 | 已实现 |
| 改造、航行、人工智能、盖亚、经济、科学六条科研轨 | 已实现 |
| 三个能量碗、充能/消耗/弃置和公共能量行动 | 已实现 |
| 盖亚塑形者、盖亚区能量、盖亚阶段和占领 | 已实现 |
| 基础科技选择及科研前进 | 已实现的固定动作版本 |
| 联邦强度、最少卫星连接、联邦令牌与科研钥匙 | 已实现的规范化版本 |
| 助推板块、过轮顺序、轮次计分和两项终局排名 | 已实现 |
| Terrans、Xenos、Taklons、Geodens | 已实现核心差异 |

为控制 AlphaZero 的分支数，当前规则存在明确的建模约束：地图固定为 19 个星球；
被动充能采用“可承受时自动接受”；资源支付和联邦连接只生成一个规范方案；基础科技按科研轨抽象。
尚未实现官方模块地图、其余 10 个种族、高级科技、完整 QIC/特殊行动、失落星球，以及完整的
助推/轮次/终局板块池。训练结果应视为 `standard-v2` 环境内的策略，不应直接作为官方比赛裁决器。

完整版规则的模块边界与实施顺序见 [docs/full-rules-roadmap.md](docs/full-rules-roadmap.md)。

## 测试

测试只依赖 Python 标准库的 `unittest`：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```
