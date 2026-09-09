# 生产级 C++ selfplay

`gaiazero_selfplay` 是规则层之外的独立 C++ 自博弈 worker。它负责：

- 使用 `GaiaState` 和参数化 `ActionTuple` 生成合法动作；
- 运行多人 PUCT/MCTS，支持温度采样、根 Dirichlet 噪声和模型推理；
- 在每局结束后原子写入一个 `npz-trajectory-v1` 原始 NPZ；
- 保存完整状态、状态 hash、合法动作 tuple、访问次数、策略目标、pairwise WDL、终局 utility 和 403 桶 VP belief 标签；
- 轮询 ONNX 模型文件，并且只在下一局开始前切换模型；
- 支持 TensorRT serialized engine 的批量叶节点推理：每个 MCTS 波次收集多个未展开叶子，
  一次 enqueue 后再扩展/回传，减少 GPU kernel launch 和 PCIe 往返；
- 监听 `STOP` 文件，支持异步训练管线优雅停止。
- 原子写入 `status.json`，供五进程监控读取运行阶段、已完成对局和步数。

## 构建

CPU 参考构建（不需要 CUDA、TensorRT 或 ONNX Runtime）：

```powershell
cmake --build build/cpp-msvc --config Release --target gaiazero_selfplay
```

## CPU smoke

```powershell
build/cpp-msvc/gaiazero_selfplay.exe `
  --players 2 `
  --games 1 `
  --simulations 1 `
  --max-moves 512 `
  --output runs/cpp-selfplay/raw `
  --status-file runs/cpp-selfplay/status.json `
  --once
```

## TensorRT 批量叶节点

TensorRT 是可选后端，engine 必须由本项目导出的同一组 ONNX 输入/输出构建，并包含
动态 batch optimization profile（建议 `1/16/64` 或 `1/64/256`）。构建时打开
`GAIA_ENABLE_TENSORRT=ON`，同时提供 `TENSORRT_ROOT`；无 CUDA/TensorRT 的开发机仍可
使用上述 CPU smoke 和 ONNX Runtime 路径。

```powershell
build/cpp-msvc/gaiazero_selfplay.exe `
  --players 3 `
  --backend tensorrt `
  --tensorrt-engine runs/pipeline-3p/exported/current.engine `
  --leaf-batch-size 64 `
  --simulations 256 `
  --output runs/pipeline-3p/raw
```

`--leaf-batch-size` 是 MCTS 叶节点波次大小，不是游戏并发数。TensorRT engine 的最大
batch 小于该值时，worker 会在推理阶段报告明确的 shape/profile 错误；应降低该参数或
重新构建 profile。`--backend auto` 在提供 `--tensorrt-engine` 时优先 TensorRT，
否则使用 `--model` 的 ONNX Runtime CPU 后端。

生成的文件可以直接由 Python 读取：

```powershell
python -c "from pathlib import Path; from gaiazero.distributed import read_npz_shard, read_npz_trajectory; p=next(Path('runs/cpp-selfplay/raw').glob('*.npz')); e,_=read_npz_shard(p); t=read_npz_trajectory(p); print(len(e), len(t['position_index']), t['terminal_valid'])"
```

## 持续运行

```powershell
build/cpp-msvc/gaiazero_selfplay.exe `
  --players 3 `
  --games 4 `
  --simulations 128 `
  --output runs/pipeline-3p/raw `
  --model runs/pipeline-3p/approved/current.onnx `
  --stop-file runs/pipeline-3p/STOP `
  --status-file runs/pipeline-3p/status.json
```

持续运行时创建 `runs/pipeline-3p/STOP` 即可在当前对局完成后退出；状态文件的
`phase` 会变为 `stopped`。模型发布应使用临时文件写完后原子重命名，worker 会在
下一局边界验证并切换，加载失败时继续使用上一份有效模型。

没有可用模型时默认使用确定性规则启发式 value 和均匀 policy，便于 CPU 验证；生产运行应提供当前 approved ONNX，并使用带 `GAIA_ENABLE_ORT_CPU=ON` 的参考构建，或后续 TensorRT backend 构建。模型文件通过临时文件写完后再改名，selfplay 不会读取半成品。

当前 worker 是可运行的 C++ PUCT/NPZ 基线：ONNX 模型接入后策略头使用参数化
动作 tuple 组合合法动作；CPU fallback 的 value 仍为规则启发式。TensorRT 批量叶评估、
完整 VP belief utility、线程池多局并发和五进程统一 telemetry 仍属于后续性能/训练闭环阶段。

当前 `.npz` 写入使用无压缩 ZIP member，兼容 NumPy；训练 worker 只消费四个训练数组，历史回放转换器才读取轨迹字段。
