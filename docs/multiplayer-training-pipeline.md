# 多人局异步训练流水线

该流水线借鉴 KataGo 的异步训练组织方式，但不接入 KataGo。游戏规则、MCTS、网络、
样本和导出权重均为 GaiaZero 自有格式。全流程只依赖 Python、NumPy 和 PyTorch，不使用
TensorFlow 或 TFRecord。

## 五个进程

该管线支持 2、3、4 人局。三种人数都使用同一 KataGo 风格残差网络和 MCTS 自对弈路径，只有价值头输出维度和游戏状态维度随人数变化。

1. `selfplay`：轮询 `approved/current.pt`，后续对局自动加载新权重；每局完成后将训练样本
   原子写入 `raw/game-*.npz`。
2. `shuffle`：扫描尚未处理的原始对局，合并并洗牌，按固定数量输出
   `shuffled/shuffle-*.npz`。
3. `train`：直接把洗牌后的 NPZ 载入 ReplayBuffer，用 PyTorch/CUDA 训练，定期写入
   `candidates/candidate-*.pt` 和可恢复的 `training/latest.pt`。
4. `export`：将已通过守门的 PyTorch 权重转换为 GaiaZero 原生
   `exported/model-*.bin`，同时更新 `exported/current.bin`。
5. `gatekeeper`：让候选权重与 `approved/current.pt` 对弈；通过阈值后原子替换当前权重，
   self-play 会在下一局开始前发现并加载。

`*.bin` 由 `gaiazero.distributed.load_exported_model` 加载。它不是围棋 KataGo 引擎的权重，
也不包含任何围棋规则或协议。

## 启动完整闭环

四人局：

```powershell
gaiazero pipeline `
  --players 4 `
  --root runs/pipeline-4p `
  --device cuda `
  --simulations 128 `
  --batch-size 512 `
  --updates-per-cycle 64 `
  --gate-games 40 `
  --gate-threshold 0.55
```

三人局必须使用独立目录：

```powershell
gaiazero pipeline --players 3 --root runs/pipeline-3p --device cuda
```

两人局使用相同流程和模型架构：

```powershell
gaiazero pipeline --players 2 --root runs/pipeline-2p --device cuda
```

命令会启动五个 Python 子进程，日志分别写入 `root/logs/`。按 `Ctrl+C` 会创建 `STOP`
文件并结束进程；下次启动完整闭环会自动移除旧 `STOP` 文件。

## 单独启动进程

首次由 `gaiazero pipeline` 创建的 `pipeline.json` 可供各进程共用：

```powershell
python scripts/selfplay.py --config runs/pipeline-4p/pipeline.json
python scripts/shuffle.py --config runs/pipeline-4p/pipeline.json
python scripts/train.py --config runs/pipeline-4p/pipeline.json
python scripts/export_model.py --config runs/pipeline-4p/pipeline.json
python scripts/gatekeeper.py --config runs/pipeline-4p/pipeline.json
```

也可直接给单个脚本传完整参数。使用 `--once` 时，进程只扫描或执行一个周期，适合诊断。

## 目录协议

```text
root/
  pipeline.json
  raw/                 每局原始 NPZ
  shuffled/            合并洗牌后的训练 NPZ
  training/latest.pt   训练进程恢复点
  candidates/          待守门模型
  approved/current.pt  self-play 当前模型
  exported/current.bin GaiaZero 推理权重
  logs/                 五进程日志和守门结果
```

所有跨进程产物先写临时文件，再通过同目录原子重命名发布。各阶段使用 JSON 状态文件记录
已处理输入，因此进程重启后不会重复打包或重复评估同一候选模型。
