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

项目首先使用 `MiniGaia` 规则切片验证算法与工程结构。它不是完整版桌游规则实现；
当前已实现六轮、收入、建矿、建筑升级、四条研究轨、过轮、轮次得分和终局排名。

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

运行一个小型自博弈训练：

```powershell
gaiazero train `
  --players 2 `
  --iterations 5 `
  --games-per-iteration 8 `
  --simulations 64 `
  --output runs/mini-gaia.pt
```

评测检查点，每局轮换神经网络所在座位：

```powershell
gaiazero evaluate runs/mini-gaia.pt --players 2 --games 20 --simulations 128
```

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
src/gaiazero/game/mini_gaia.py  可运行的 Gaia 规则切片
src/gaiazero/mcts.py            多玩家 PUCT/PIMCTS
src/gaiazero/model.py           PyTorch 策略/价值网络与检查点
src/gaiazero/selfplay.py        自博弈数据生成
src/gaiazero/replay.py          有界经验回放
src/gaiazero/training.py        AlphaZero 联合损失训练器
src/gaiazero/arena.py           座位轮换评测
src/gaiazero/cli.py             命令行入口
tests/                          规则、搜索和训练回归测试
```

## 当前规则范围

| 模块 | 状态 |
|---|---|
| 2-4 人、六轮与行动顺序 | 已实现 |
| 资源收入和上限 | 已实现的简化版本 |
| 星球距离、航行和地形改造成本 | 已实现的简化版本 |
| 矿场、贸易站、研究所 | 已实现 |
| 四条研究轨 | 已实现的算法验证版本 |
| 轮次计分和终局排名 | 已实现的简化版本 |
| 能量碗、燃烧能量、被动充能 | 待实现 |
| 盖亚化与盖亚塑形者 | 待实现 |
| 科技板块、联盟和卫星路径 | 待实现 |
| 14 个种族及其能力 | 待实现 |
| 完整随机地图和计分设置 | 待实现 |

完整版规则的模块边界与实施顺序见 [docs/full-rules-roadmap.md](docs/full-rules-roadmap.md)。

## 测试

测试只依赖 Python 标准库的 `unittest`：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

