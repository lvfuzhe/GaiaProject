# 待执行：C++ Selfplay/Gatekeeper 与 TensorRT 推理

本文档记录下一阶段的目标架构和分步执行顺序。当前仓库仍运行五个 Python 进程；本文档中的步骤全部是待执行项，完成前不得把目标架构标记为已部署。

## 目标架构

```text
Python train
  -> training/latest.pt / candidates/candidate-*.pt
     (普通权重 + 优化器状态 + SWA 权重)
  -> Python export
     提取 SWA 权重并导出 exported/candidate-*.onnx
  -> C++ gatekeeper
     TensorRT 读取 ONNX，候选模型对战 approved/current.onnx
  -> approved/current.onnx
  -> C++ selfplay
     TensorRT 读取 approved/current.onnx，输出 raw/*.npz
  -> Python shuffle -> Python train
```

ONNX 只作为 PyTorch 与 TensorRT 之间的交换格式，不引入 ONNX Runtime。TensorRT engine 可以在 C++ 进程启动时从 ONNX 构建，并按模型哈希缓存；缓存不是训练和模型发布的权威文件，权威模型仍是通过守门的 ONNX 文件。

网络规模统一使用 `b{blocks}c{channels}` 命名，例如 `b10c128` 表示 10 个残差块、`hidden_size=128`。这里的 `c` 是当前向量网络的隐藏宽度，不是二维卷积通道数。

## 执行顺序

### 0. 冻结跨进程契约与基线

- [ ] 固定 `standard-v22` 的观察向量、合法动作掩码、策略输出和值输出的 shape、dtype、动作编号和玩家顺序。
- [ ] 固定 NPZ 训练样本格式：`observations`、`legal_masks`、`policy_targets`、`value_targets`，以及 self-play 完整复盘 metadata 的兼容要求。
- [ ] 固定模型清单格式：规则版本、玩家数、观察/动作维度、网络架构、SWA 是否可用、导出 opset、TensorRT 精度模式和权重 SHA-256。
- [ ] 为 Python 参考实现增加一组固定种子状态和网络输出 golden fixtures，作为 C++ 对齐基准。
- [ ] 记录当前 Python self-play、单次 MCTS、网络 batch=1/batch=N 的吞吐和显存基线。

验收：契约文档、golden fixtures 和基线数据已提交；后续 C++ 或导出改动均能复现这些输入输出。

### 1. 在 Python train 中加入 SWA

- [ ] 在 `AlphaZeroTrainer` 中增加可配置的 SWA/平均权重模型和更新周期。
- [ ] 明确 SWA 起始步数、更新频率、平均算法、设备和 checkpoint 恢复行为。
- [ ] checkpoint 同时保存普通 `model_state`、SWA `swa_state`、优化器状态和必要的计数器。
- [ ] 继续训练时恢复普通权重、SWA 权重和优化器；仅推理时不得加载优化器状态。
- [ ] 增加普通权重与 SWA 权重不同、恢复后继续平均、无 SWA 时明确报错或回退的测试。

验收：同一个 checkpoint 可以分别加载普通模型和 SWA 模型；SWA 计数器跨进程重启保持一致。

### 2. 网络递进训练：b10c128 -> b15c192 -> b20c256

#### 2.1 固定代际配置与产物

- [ ] 在 `pipeline.json` 增加显式的 `network_id`、`residual_blocks`、`hidden_size`、`generation` 和训练阶段，禁止只根据文件名猜测网络结构。
- [ ] 模型 checkpoint、ONNX manifest、守门记录、训练状态和 Dashboard 都记录同一个 `network_id`，例如 `b10c128-g0`。
- [ ] 2、3、4 人局采用相同的递进表，但分别训练和晋级；不同玩家数的价值头维度不同，不共享 checkpoint 或 SWA 状态。
- [ ] 每个代际使用独立的 candidate、approved、rejected 和 TensorRT engine 缓存记录，同时保留上一代冠军用于跨代守门和回退。
- [ ] 固定跨代兼容条件：规则版本、观察维度、动作编号和玩家数必须相同；任一条件变化都启动新训练线，不得当作单纯扩容。

验收：任一 `.pt` 或 `.onnx` 都能仅通过 manifest 确定玩家数、代际、块数、隐藏宽度、规则版本、父模型和模型哈希。

#### 2.2 递进阶段

| 阶段 | 网络 | 目标 | 进入下一阶段的条件 |
| --- | --- | --- | --- |
| G0 | `b10c128` | 验证完整闭环，以较高吞吐积累首批有效自博弈数据 | 达到最低新鲜样本数和更新数；训练/验证指标不再快速改善；连续候选可稳定通过同代守门 |
| G1 | `b15c192` | 作为宽度和深度扩展的过渡代，验证跨架构蒸馏、双 TensorRT engine 守门和回退 | 对 G0 冠军达到跨代守门阈值；吞吐、显存和延迟在预算内；新模型自产数据占比达到配置要求 |
| G2 | `b20c256` | 目标主力网络，进入长期自博弈和持续优化 | 不再自动扩容；后续扩容必须重新增加代际、基准和资源预算 |

- [ ] 将默认阶段表写入可版本化的 `network_schedule` 配置，而不是硬编码；实测证明过渡代无收益时可以删除 G1，但仍必须执行跨代预热、守门和回退检查。
- [ ] 把最低样本数、最低更新数、指标平台窗口、跨代守门局数、通过阈值和资源上限写成配置，初始值通过 G0 基准测试确定，不能硬编码在训练脚本中。
- [ ] 阶段切换必须人工确认或由明确的自动晋级开关触发；仅达到训练步数不得自动替换生产模型。
- [ ] Dashboard 显示当前代际、下一代际、阶段条件完成度、当前数据代际分布和预计扩容资源需求。

#### 2.3 跨代初始化

- [ ] 不直接对 `b10c128` 和 `b20c256` 使用严格 `state_dict` 续载；宽度、残差块数以及优化器张量形状均不兼容。
- [ ] 默认使用教师-学生过渡：冻结上一代 approved SWA 模型作为 teacher，初始化新一代 student，用现有 NPZ 的策略/价值目标加 teacher 的策略 logits 和价值输出进行预热。
- [ ] 蒸馏损失权重、温度、预热更新数和退出条件均配置化；预热结束后逐步衰减蒸馏权重，回到以自博弈目标为主的 AlphaZero 损失。
- [ ] 可将“复制公共张量切片并把新增残差路径初始化为近似恒等映射”作为实验选项，但必须先通过同输入数值回归；它不能成为默认迁移方式。
- [ ] 每次改变网络规模都重新创建优化器、学习率调度器和 SWA 累计器；不得加载上一代不兼容的优化器或 SWA 状态。
- [ ] 新一代 SWA 只在预热结束并完成规定更新数后开始累计，避免随机初始化和早期蒸馏状态污染平均权重。

验收：相同 NPZ batch 上，新一代 student 完成预热后，其合法动作策略和价值输出与 teacher 的误差达到配置门槛，并能继续使用真实训练目标优化。

#### 2.4 数据换代与跨代守门

- [ ] 新一代训练初期可以读取旧代 NPZ，因为观察、动作和训练目标契约不变；每个 shard 必须记录生成它的 `network_id` 和模型哈希。
- [ ] 为 replay 设置按代际采样和逐步淘汰策略：扩容初期保留旧数据稳定训练，通过跨代守门后逐步提高新一代自产数据权重，避免旧数据永久主导。
- [ ] 新一代未通过守门前，C++ selfplay 继续使用上一代 `approved/current.onnx`，不得因 train 已切换网络尺寸而提前切换生产模型。
- [ ] C++ gatekeeper 同时加载上一代冠军和新一代候选各自的 TensorRT engine；二者只要求输入/输出契约相同，不要求内部块数和宽度相同。
- [ ] 跨代比赛固定初始设置、座位轮换、随机种子和思考预算；MCTS 应按相同模拟次数比较，同时额外记录实际耗时和每秒模拟数。
- [ ] 新一代通过跨代守门后，原子更新 `approved/current.onnx` 与 manifest；旧冠军归档为 rollback 模型，selfplay 从下一局开始切换。
- [ ] 若新一代连续失败、训练发散、出现 NaN/Inf 或吞吐低于资源门槛，则保持旧冠军在线，并允许重新预热或调整网络阶段，不能跳过守门。

验收：可以完整演练一次 G0 -> G1 -> G2 晋级和一次失败回退；任何时刻 selfplay 都只读取已经通过守门且产物完整的模型。

#### 2.5 训练资源与超参数适配

- [ ] 分别测量三个网络在训练 batch、TensorRT 叶节点 batch、selfplay 并发局数下的显存、吞吐和延迟，不沿用同一套 batch 参数。
- [ ] 随网络扩大逐代调整训练 batch size、梯度累积、学习率、权重衰减和 checkpoint 周期，并把有效 batch size 写入训练记录。
- [ ] 为每一代设置独立资源预算和最低吞吐门槛；若 `b20c256` 达不到门槛，优先降低并发或 batch，不静默退回不同网络结构。
- [ ] 保留同一固定验证集和固定种子对局集，用于比较各代策略损失、价值误差、守门胜率和单位 GPU 时间的棋力收益。

验收：三个代际都有可复现的训练与推理基准，扩容决策同时依据棋力、数据新鲜度、吞吐和显存，而不是只看训练 loss。

### 3. export 进程改为 SWA -> ONNX

- [ ] 将导出输入从“已批准 `.pt`”改为训练产生的候选 checkpoint `.pt`，导出顺序先于 gatekeeper。
- [ ] 加载 checkpoint 的 SWA 权重，切换 `eval()`，冻结参数，不导出优化器、训练指标或 Python 对象。
- [ ] 使用固定的输入签名导出 ONNX：观察张量、合法动作掩码（如网络图需要）以及策略/价值输出；明确动态 batch 维度。
- [ ] 固定并记录 ONNX opset、float32/float16 策略、输出名称、网络版本和状态维度。
- [ ] 导出后用 ONNX 图检查器和 PyTorch 对同一 golden fixture 做数值比对；记录最大绝对误差和相对误差。
- [ ] 通过临时文件、SHA-256 和 manifest 原子发布 `exported/candidate-*.onnx`，失败导出不得进入守门队列。
- [ ] 保留训练 `.pt` 作为可恢复训练和审计文件；ONNX 是推理发布副本，不覆盖训练 checkpoint。

验收：每个候选都有可验证的 ONNX manifest；PyTorch SWA 与 ONNX 输出在约定误差内一致。

### 4. C++ TensorRT 推理适配层

- [ ] 新建独立 C++ 推理库，负责 ONNX 解析、TensorRT network/engine 构建、上下文创建和资源释放。
- [ ] 支持动态 batch 或预设 batch profile，并实现多请求合批接口，避免 self-play 每个叶节点单独推理。
- [ ] 实现 CUDA stream、pinned host memory、异步拷贝和 batch 输出回收；默认提供 FP32 校验模式，再启用 FP16/TF32 优化。
- [ ] 按 ONNX SHA-256、TensorRT/CUDA 版本和精度配置缓存序列化 engine；缓存失效时自动重建。
- [ ] 增加 TensorRT 输出与 Python PyTorch SWA 输出的逐元素校验工具。
- [ ] 对非法 shape、动作维度、NaN/Inf、engine 版本不兼容和显存不足提供明确错误。

验收：C++ 推理库可以独立加载一个 ONNX，在 CPU 参考输出和 GPU TensorRT 输出之间完成 golden fixture 校验。

### 5. C++ Gaia 规则状态与编码器

- [ ] 将 `GaiaState` 的状态字段、玩家顺序、地图坐标、资源、科技、联邦、助推和待决策状态映射为 C++ 数据结构。
- [ ] 实现与 Python 一致的初始设置、随机种子、合法动作生成、动作应用、终局返回值和观察编码。
- [ ] 明确 C++ 状态复制/撤销策略，优先使用紧凑数组、结构共享或可回滚状态，避免每个节点深拷贝大对象。
- [ ] 为每一类动作建立 Python/C++ 双向序列化和逐状态对比测试。
- [ ] 使用固定种子执行短局、完整局和边界规则测试，比较合法动作集合、资源、VP、终局和 NPZ trace。

验收：C++ 与 Python 在 golden fixtures 和随机短局上产生相同的状态摘要、合法动作和最终结果。

### 6. C++ PIMCTS Selfplay

- [ ] 实现与当前 PUCT/PIMCTS 相同的多玩家价值回传、根噪声、温度、动作采样和最大步数语义。
- [ ] 将 MCTS 树节点改为紧凑结构，使用线程池运行多局 self-play；每棵树的随机种子必须可追踪。
- [ ] 将叶节点请求提交给 TensorRT 批量推理队列，支持虚拟损失或等价并发机制。
- [ ] 按现有 NPZ schema 写入完整训练样本和复盘 metadata；写入采用临时文件后原子重命名。
- [ ] C++ selfplay 轮询 `approved/current.onnx`，只在完整模型发布后切换，不读取未完成文件。
- [ ] 进程状态、对局数、位置数、吞吐、模型哈希、错误和最近 shard 写入现有五进程监控协议。

验收：C++ selfplay 生成的 NPZ 可被现有 Python shuffle/train 读取；同等模拟次数下规则结果与 Python 参考实现一致。

### 7. C++ TensorRT Gatekeeper

- [ ] 将 gatekeeper 输入改为 `exported/candidate-*.onnx`，当前模型改为 `approved/current.onnx`。
- [ ] 使用与 selfplay 相同的 C++ Gaia 规则、MCTS 和 TensorRT 推理适配层，避免守门与训练数据生成规则分叉。
- [ ] 固定候选/冠军座位轮换、种子、局数、阈值、平局处理和多玩家排名统计。
- [ ] 通过后原子发布 `approved/current.onnx` 及 manifest；拒绝模型写入 rejected 目录和结构化日志。
- [ ] 首个模型没有冠军时定义 bootstrap 规则，并测试重复候选、半成品文件和进程重启恢复。

验收：候选只有在 C++ TensorRT 对战通过后才会成为 selfplay 可见模型；守门结果可复现并能追溯模型哈希。

### 8. 五进程编排与 Dashboard

- [ ] 将进程角色改为：C++ `selfplay`、Python `shuffle`、Python `train`、Python `export`、C++ `gatekeeper`。
- [ ] 更新 `pipeline.json`、状态文件、日志目录和产物目录协议，区分 `.pt` 训练 checkpoint、`.onnx` 推理模型和 TensorRT engine 缓存。
- [ ] 让 supervisor 以可配置路径启动 C++ 可执行文件，并检查启动前的 CUDA/TensorRT、模型和规则版本。
- [ ] 五个监控页面显示进程语言、推理后端、模型哈希、SWA 导出状态、engine 构建耗时、吞吐和错误。
- [ ] 保留停止、重启、断点恢复和单进程 `--once` 诊断语义；C++ 进程异常退出时不得静默继续写入数据。
- [ ] 更新 README、流水线文档和运行示例，明确当前 Python 方案与新 C++ 方案的切换条件。

验收：Dashboard 可以一键启动和停止混合语言五进程；任一进程重启后不会重复消费或误发布模型。

### 9. 性能、正确性与发布门槛

- [ ] 对比 Python selfplay、C++ batch=1、C++ 批量 TensorRT 和多局并发的 simulations/s、positions/s、对局时长、GPU 利用率和显存。
- [ ] 分别测试 FP32、FP16 和可选 TF32；任何精度模式切换都要通过规则和网络 golden fixtures。
- [ ] 运行随机合法对局、状态哈希、NPZ 读取、模型切换、断点恢复和守门回归测试。
- [ ] 至少完成一轮小规模 A/B：同一初始设置和模型下，比较 Python 与 C++ 的动作、终局分数和训练样本摘要。
- [ ] 设置上线门槛：无规则差异、无非法动作、无 NaN/Inf、模型哈希可追踪、吞吐达到基线目标后，才替换默认 selfplay/gatekeeper。
- [ ] 保留 Python selfplay/gatekeeper 作为离线参考和故障回退，直到 C++ 版本完成稳定性观察期。

## 依赖与不可并行项

```text
0 契约冻结
  ├─> 1 SWA checkpoint
  │     └─> 2 网络递进契约与迁移机制
  │             └─> 3 SWA -> ONNX export
  │                     └─> 4 TensorRT adapter
  └─> 5 C++ Gaia rules

4 + 5
  ├─> 6 C++ selfplay
  └─> 7 C++ gatekeeper

2 + 3 + 4 + 5 + 6 + 7
  └─> 8 五进程编排与 Dashboard
          └─> 9 性能、正确性与发布
                  └─> 实际执行 G0 -> G1 -> G2
```

步骤 1 和步骤 5 可以在步骤 0 完成后并行。步骤 6 和步骤 7 必须共用同一套 C++ 规则与 TensorRT 适配层，不能分别实现两套逻辑。步骤 2 先实现代际配置、迁移和数据换代机制；实际的 G0 -> G1 -> G2 训练晋级，必须等步骤 3 至步骤 8 的相应能力通过步骤 9 的基准与回归检查后再执行。步骤 8 不应在模型格式和 C++ 状态契约未稳定前提前替换现有流水线。

## 非目标

- 不把 ONNX Runtime 引入 selfplay 或 gatekeeper。
- 不把训练迁移到 C++；训练、优化器和 SWA 仍由 Python/PyTorch 负责。
- 不让 `.bin`、TensorRT engine 或临时导出文件成为训练恢复的唯一来源。
- 不在 C++ 版本通过完整回归前删除 Python 参考实现。
