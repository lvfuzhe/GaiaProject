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
- [ ] 固定 NPZ 训练样本格式：输入、掩码、2.6 节定义的全部监督标签、逐头 loss mask/weight，以及 self-play 完整复盘 metadata 的兼容要求。
- [ ] 固定模型清单格式：规则版本、标签 schema/head 版本、玩家数、观察/动作维度、网络架构、SWA 是否可用、导出 opset、TensorRT 精度模式和权重 SHA-256。
- [ ] 为 Python 参考实现增加一组固定种子状态和网络输出 golden fixtures，作为 C++ 对齐基准。
- [ ] 记录当前 Python self-play、单次 MCTS、网络 batch=1/batch=N 的吞吐和显存基线。

验收：契约文档、golden fixtures 和基线数据已提交；后续 C++ 或导出改动均能复现这些输入输出。

#### 0.1 虚拟竞拍 VP offset 的 observation 契约

本项目不让 AI 参与竞拍，也不把竞拍加入动作空间。改为由离线评估器根据批准模型的对局结果生成每位玩家的开局 VP 补偿，记原始终局分为 `S_i`、开局补偿为 `K_i`，用于排名和训练的调整后终局分为 `S'_i = S_i + K_i`。它是多人向量形式的 komi，而不是一个全局标量。

KataGo 把 komi 当作全局条件，是因为同一棋盘在不同 komi 下具有不同效用。盖亚的 `K_i` 在第一步前直接写入每位玩家当前 `vp` 后，当前 observation 已经包含它并满足马尔可夫性；无需保存出价过程。为支持补偿表迭代和跨版本分析，仍建议显式保留 `starting_vp_offset`。

- [ ] 为 `GaiaState.initial()` 增加按座位排列的 `starting_vp_offsets [P]`；初始化 VP 为 `10 + K_i`，且不产生竞拍 phase 或竞拍 action。
- [ ] observation 保留当前 `vp`，并增加每位玩家的归一化 `starting_vp_offset`。相同局面但 offset 不同必须产生不同 observation hash。
- [ ] 分别为 2、3、4 人局维护补偿表；禁止把不同人数的 offset 混用。
- [ ] 每局所有 offset 采用统一规范消除平移自由度，例如先读取所选种族/座位的表值，再减去本局均值使 `sum(K_i)=0`；若最终使用整数 VP，应固定取整、余数分配和上下限规则。
- [ ] `final_vp_targets`、rank/WDL、MCTS terminal utility 和守门结果都使用调整后分数；同时保存未补偿的 `raw_final_vp_targets`，用于重新估算而不污染原始强度数据。
- [ ] NPZ 和复盘记录 `starting_vp_offsets`、`compensation_version`、原始终局分和调整后终局分；不得只保存调整后结果。
- [ ] 模型与 ONNX manifest 记录 `compensation_mode=offline-vp-offset`、补偿版本、归一化和取整规则；只有 observation、规则与补偿契约兼容的模型才能直接守门对战。

验收：同一初始设置仅改变 `K_i` 时，初始 `PlayerState.vp`、observation、终局 rank/WDL 和 MCTS 价值随之改变；合法动作和其他规则状态保持不变。

#### 0.2 离线补偿估算与迭代闭环

补偿评估是五进程之外的离线任务，在模型代际边界运行，不增加第六个常驻训练进程。补偿表只有完成独立验收并原子发布后，才会被后续新对局读取；运行中的对局不得切换版本。

##### 0.2.1 C0：零补偿基准采样

- [ ] 先完成一个可稳定对局的 G0 bootstrap 模型，所有玩家使用 `K_i=0`；随机网络产生的结果不得用于估算种族补偿。
- [ ] 冻结一个 approved 模型作为评估器，评估期间不更新权重、搜索参数或规则版本。
- [ ] 为同一 setup seed 生成成组对局，轮换种族和座位；在合法种族组合范围内覆盖全部 14 个种族、先后手和对手组合。
- [ ] 2、3、4 人局分别采样；小地图/标准地图、地图种子、回合计分、终局计分、科技和助推布局全部记录为上下文，不把不同配置直接混成一个均值。
- [ ] 所有比较使用相同的 MCTS 模拟数、温度、根噪声方案和最大步数；评估局应关闭会妨碍配对比较的非必要随机项，并保留可复现种子。
- [ ] 单独写入 `compensation/evaluations/*.npz` 或结构化列式数据，至少包含模型哈希、规则版本、人数、setup hash、座位、种族、原始最终 VP、排名、pairwise WDL 和完整性标记；不进入普通训练 shard。
- [ ] 为每个种族/座位/人数单元设置最低有效对局数，并报告均值、标准差、置信区间和缺失组合；样本不足时不得发布 offset。

验收：相同 seed 的座位/种族轮换对局能够配对复现，且可以区分种族效应、座位效应和设置噪声。

##### 0.2.2 C1：拟合多人 VP offset

- [ ] 先拟合带正则化的分层模型，至少分解 `faction + seat/order + player_count + setup context + opponent mix`；种族样本少时向总体均值收缩，避免极端补偿。
- [ ] 2 人局先由配对原始分差估计初值；若 `E[S_f-S_g]=d`，只需满足 `K_f-K_g=-d`，再通过约束固定唯一解。
- [ ] 3/4 人局不能只对齐平均 VP；优化目标同时包含第一名率、平均排名和平均 pairwise utility 的偏差，并对 offset 大小施加正则。
- [ ] 将地图模式、顺位以及影响显著且样本足够的设置因素做成有限 context bucket；不为每一个完整随机 setup 单独拟合，避免表规模爆炸和过拟合。
- [ ] 在训练集拟合、独立 holdout seed 上验证；报告补偿前后的原始分差、第一名率、平均排名、pairwise WDL、校准误差和 bootstrap 置信区间。
- [ ] 把连续估计值按固定规则转换为游戏使用的整数 VP，并在取整后重新计算公平性指标；禁止仅报告取整前结果。
- [ ] 发布版本化 `compensation/offsets-vNN.json`，包含适用人数、context bucket、每种族/座位 offset、训练模型哈希、数据范围、拟合参数、置信区间、父版本和 SHA-256。

验收：holdout 数据上所有配置化公平性门槛均通过，且任何 offset 都能追溯到模型、对局集合和拟合版本。

##### 0.2.3 C2：带补偿重新 selfplay 和训练

- [ ] C++ selfplay 每局开始时根据人数、种族、座位和 context 读取固定版本补偿表，将 `K_i` 写入初始 VP；模型不执行竞价。
- [ ] 搜索、最终计分、rank/WDL、value target 和 gatekeeper 全部基于调整后分数；原始分数仅用于公平性分析，不能参与本局策略回传。
- [ ] 新补偿版本只对新开对局生效；进程轮询到更新后，必须等当前对局结束再切换，并在 shard 中写入实际版本和 offset。
- [ ] 旧 NPZ 保留其原 observation、offset 和 policy target，不把旧策略样本事后改成新 offset。训练窗口逐步提高新补偿版本自产数据权重，直到旧版本退出。
- [ ] 若保留显式 `starting_vp_offset`，不同补偿版本的完整旧样本可以共同训练；若 observation schema 改变，则必须启动新训练线或进行明确迁移。
- [ ] compensation 版本变化时重新建立或继续 SWA 的策略写入配置；大幅变化默认重置 SWA，小幅变化可在验证通过后延续，但必须记录选择。

验收：训练样本中的 observation VP、offset、原始分数、调整后分数、value/rank 目标相互一致；从任一 shard 可以重放相同终局效用。

##### 0.2.4 C3：公平性守门与收敛

- [ ] 模型 gatekeeper 比较新旧模型时，双方使用完全相同的补偿表、配对 setup、座位/种族轮换和搜索预算；模型棋力晋级与补偿表公平性验收分别出报告。
- [ ] 每个新 approved 模型积累足够新数据后重新运行 C0/C1，但以当前 offset 为基线拟合残余补偿 `delta_K_fit`，而不是每次从零估计。
- [ ] 使用阻尼更新 `K_next=K_current+alpha*delta_K_fit`，`alpha` 配置化；若拟合器输出绝对推荐表 `K_fit`，等价写为 `K_next=(1-alpha)*K_current+alpha*K_fit`。限制单次最大变化，避免模型策略与补偿相互追逐而振荡。
- [ ] 新表只有在 holdout 公平性改善、置信区间合格且没有明显伤害其他人数/context 后才发布；失败则保留当前表并归档候选结果。
- [ ] 配置停止条件：连续多轮最大 offset 变化、第一名率偏差、平均排名偏差和 pairwise utility 偏差均低于阈值；阈值由基准数据确定，不硬编码。
- [ ] 网络从 G0 扩展到 G1/G2 时继承上一代 offset 作为初值，但新架构通过守门并积累自产数据后必须重新校准，不能假设种族相对强度恒定。
- [ ] Dashboard 增加非进程型的“补偿评估”报告页，显示当前/候选版本、样本覆盖、各人数种族 offset、补偿前后公平性、置信区间和收敛历史；不加入五进程一键启动。

验收：至少演练一次 `零补偿 -> offsets-v1 -> 带补偿训练 -> 残差评估 -> offsets-v2`，以及一次候选补偿被拒绝并回退的流程。只有满足配置门槛的表才成为新对局默认值。

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

#### 2.6 KataGo 风格神经网络头与 Gaia 训练标签

参考依据是 KataGo 官方的 [ONNX 输出定义](https://github.com/lightvector/KataGo/blob/master/docs/ONNX_Model_Files.md)、[短期价值与分数目标](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#short-term-value-and-score-targets)以及 [PyTorch NPZ 读取结构](https://github.com/lightvector/KataGo/blob/master/python/katago/train/data_processing_pytorch.py)。KataGo 的核心是 policy、W/L/no-result、score belief/lead/uncertainty、ownership 和逐动作 Q。Gaia 是 2-4 人、按 VP 排名、允许同星球共存且没有隐藏信息，不能直接照搬围棋通道。

当前 `TrainingExample` 只有 observation、legal mask、MCTS policy 和终局 utility 四组数据。下列字段均属于待实现的新 label schema；在 schema 版本升级前，旧 NPZ 不得被误认为具有缺失 head 的零值标签。

符号约定：`P` 为本模型的玩家数，`A` 为固定动作空间，`N` 为星球槽位，`S` 为可放卫星/空间站的星图位置，`T=6` 为科技轨数量，`H` 为短/中/长三个时间尺度。所有玩家维度必须使用与 observation 相同的座位顺序；2、3、4 人模型分别保存，不通过 padding 混训。

优先级约定：P0 是 G0 训练前必须完成的基础标签；P1 是完整 C++ 搜索统计可用后加入的低风险增强；P2 是需要消融和守门结果证明收益的实验标签。P2 不得阻塞首个可运行闭环。

##### 2.6.1 策略头标签

| NPZ 标签 | Shape | 来源与用途 | 优先级 |
| --- | --- | --- | --- |
| `policy_visit_targets` | `[A]` | 根节点访问次数归一化后的主策略目标；非法动作必须为 0 | P0 |
| `root_visit_counts` | `[A]` | 未归一化访问次数，用于复核温度、重分析和低访问样本降权 | P0 |
| `root_policy_priors` | `[A]` | 网络输出并经过合法动作过滤的原始先验；用于校准与复现搜索 | P0 |
| `root_noised_policy_priors` | `[A]` | 加入根噪声后实际用于搜索的先验；用于 policy surprise/KL 采样权重 | P0 |
| `played_actions` | scalar | selfplay 实际采样动作，用于复盘、行为统计和策略校准 | P0 |
| `action_value_targets` | `[A,P]` | MCTS 已访问子节点的多人胜负效用 Q；未访问动作由独立 mask 排除 | P1 |
| `action_vp_targets` | `[A,P]` | 每个已访问动作的预计最终 VP/VP 差 Q，帮助区分同胜率但得分不同的动作 | P1 |
| `optimistic_policy_targets` | `[A]` | 类似 KataGo optimistic policy；由风险/得分偏好搜索产生，不从基础 visit target 复制 | P2 |

- [ ] ONNX 的基础 policy 输出为原始 logits `[B,A]`，合法动作过滤在 C++ 搜索端完成，不能把 `legal_masks` 烘焙进网络权重。
- [ ] 逐动作 Q 标签同时保存 `action_value_masks [A]`，仅训练实际搜索过且访问数达到阈值的动作。
- [ ] P2 乐观策略只在基础策略、VP head 和跨代守门稳定后启用；多人局先明确“乐观”是追求当前玩家 VP、排名还是风险调整效用。

##### 2.6.2 多人结果与价值头标签

| NPZ 标签 | Shape | 来源与用途 | 优先级 |
| --- | --- | --- | --- |
| `final_utility_targets` | `[P]` | 当前 `returns()` 的平均两两胜负效用，直接供 PIMCTS 回传 | P0 |
| `final_rank_targets` | `[P,P]` | 每位玩家取得第 1 到第 P 名的分布；并列时在占用名次上均分概率 | P0 |
| `pairwise_wdl_targets` | `[P,P,3]` | 任意两名玩家的胜/平/负 one-hot，保留多人排名中被单一标量丢失的信息 | P0 |
| `root_value_targets` | `[P]` | 该位置 selfplay MCTS 根价值，用于重分析、蒸馏和短期价值监督 | P1 |
| `td_value_targets` | `[H,P]` | 对后续根 MCTS 价值做三种指数衰减平均的短/中/长期目标 | P1 |
| `shortterm_value_error_targets` | `[P]` | 当前预测与短期 MCTS 价值目标的平方误差/方差目标，供搜索置信度加权 | P1 |
| `value_settle_time_targets` | scalar | 从当前位置到排名/效用基本稳定的剩余语义决策数，映射 KataGo variance-time | P2 |

- [ ] Gaia 的时间尺度按“语义决策”而不是原始 action 计数；免费兑换、被动充能确认、科技选择等微步骤不能把时间轴无限拉长。先从约 8、24、80 个语义决策的均值范围做基准，再根据完整对局长度校准 lambda。
- [ ] 正常完成的 Gaia 对局没有 KataGo 的 `no-result` 类别；非法、损坏、超出最大步数或缺少终局的对局标为 `terminal_valid=0`，不得伪装成平局训练。真实同分通过 pairwise draw 和并列 rank 表示。
- [ ] MCTS 仍消费 `[P]` 多人效用；rank/WDL 是训练辅助头，不能在 C++ 与 Python 中使用不同的效用换算。

##### 2.6.3 VP、分数分布和不确定性标签

| NPZ 标签 | Shape | 来源与用途 | 优先级 |
| --- | --- | --- | --- |
| `raw_final_vp_targets` | `[P]` | 不包含虚拟竞拍 offset 的每名玩家原始最终 VP，用于补偿重估和审计 | P0 |
| `final_vp_targets` | `[P]` | 包含 `starting_vp_offsets` 的每名玩家调整后最终 VP，供 value/rank/MCTS 使用 | P0 |
| `final_vp_belief_targets` | `[P,V]` | 固定 VP 桶的终局分布，包含下溢/上溢桶，类似 KataGo score belief | P0 |
| `final_vp_lead_targets` | `[P]` | 每名玩家相对其余玩家均值的最终 VP 差；另由分数计算对领先者差值 | P0 |
| `final_vp_component_targets` | `[P,6]` | `原始局内VP、开局offset、科技轨终局分、剩余资源分、终局板块1、终局板块2` | P0 |
| `td_vp_targets` | `[H,P]` | 后续根节点预计终局 VP 的短/中/长期指数平均 | P1 |
| `shortterm_vp_error_targets` | `[P]` | 当前 VP 预测与短期 MCTS VP 目标之间的误差方差 | P1 |
| `vp_source_targets` | `[P,K]` | 更细 VP 台账：轮次计分、联邦、科技、助推、QIC 行动、种族能力等 | P2 |

- [ ] `V` 的范围和桶宽由完整随机对局统计确定并写入 label manifest；不允许训练脚本和 TensorRT 后处理各自假设不同区间。
- [ ] VP head 至少输出 mean、stdev 和 belief logits；`lead` 由独立输出校准，不能只用当前 VP 差代替最终预测。
- [ ] P0 的六项分解可以直接从 `final_scores()` 和本局 offset 计算；必须断言 `raw_final_vp_targets + starting_vp_offsets == final_vp_targets`。P2 细分台账要求所有 VP 变化携带稳定 `score_source_id`，不能依赖动作说明文字解析。

##### 2.6.4 星图控制与建筑头标签

这组标签对应 KataGo ownership，但 Gaia 不能使用单一所有权值：Lantids 可与主建筑共存，Ivits 空间站和卫星也不位于普通星球槽位。

| NPZ 标签 | Shape | 来源与用途 | 优先级 |
| --- | --- | --- | --- |
| `final_planet_owner_targets` | `[N,P+1]` | 终局主建筑所有者分类，额外一类表示无人殖民 | P0 |
| `final_coexisting_owner_targets` | `[N,P+1]` | Lantids 共存矿场所有者分类，额外一类表示不存在 | P0 |
| `final_structure_targets` | `[N,6]` | 无建筑、矿场、贸易站、研究所、行星研究院、学院 | P0 |
| `final_planet_terrain_targets` | `[N,terrain_count]` | 终局地形，包括 Transdim 转 Gaia 和 Lost Planet | P1 |
| `final_federated_targets` | `[N,2]` | 主建筑与共存矿场是否已计入联邦 | P0 |
| `final_satellite_owner_targets` | `[S,P+1]` | 每个星图位置的卫星所有者或空 | P0 |
| `final_space_station_owner_targets` | `[S,P+1]` | Ivits 空间站所有者或空 | P0 |
| `final_space_station_federated_targets` | `[S]` | 空间站是否已计入联邦 | P1 |
| `round_end_board_targets` | 同上并带 round 维度 | 最近轮末的较短期星图目标，降低只看终局的高方差 | P2 |

- [ ] 每组位置标签必须带 `planet_masks [N]` 或 `board_space_masks [S]`，屏蔽小地图、未启用 Lost Planet 槽位和不存在的位置。
- [ ] primary owner、coexisting owner 和 structure 必须分头预测，不能把 Lantids 共存编码成互斥的单 owner 类别。
- [ ] Gaiaformer 若在数据审计中终局几乎总为空，不建立无信息的终局 head；它只进入轮末/短期发展标签。

##### 2.6.5 玩家发展与终局计分辅助标签

| NPZ 标签 | Shape | 内容 | 优先级 |
| --- | --- | --- | --- |
| `final_structure_count_targets` | `[P,5]` | 五类建筑终局数量 | P0 |
| `final_research_level_targets` | `[P,T,6]` | 六条科技轨最终 0-5 层 categorical | P0 |
| `final_colonized_type_targets` | `[P,terrain_count]` | 各星球类型最终是否已殖民 | P0 |
| `final_scoring_metric_targets` | `[P,2]` | 本局两个终局计分板块的原始指标值 | P0 |
| `final_scoring_award_targets` | `[P,2]` | 处理排名、并列和 2 人中立玩家后的实际奖励 VP | P0 |
| `final_resource_targets` | `[P,10]` | 信用点、矿石、知识、QIC、三能量区、盖亚区能量、可用/盖亚区 Gaiaformer | P1 |
| `round_end_resource_targets` | `[P,10]` | 最近轮末的同组资源，作为短期经济监督 | P1 |
| `final_brainstone_targets` | `[P,5]` | Taklons 脑石不存在或位于 I/II/III/盖亚区的 categorical | P1 |
| `final_academy_type_targets` | `[P,2]` | 知识学院和 QIC 学院的终局数量 | P1 |
| `final_tech_targets` | `[P,standard+covered+advanced]` | 标准科技、被覆盖状态和高级科技所有权 | P1 |
| `final_federation_targets` | `[P,F+4]` | 各类联邦片、未使用/已使用数、版图联邦数和 Gleens 专属联邦数 | P1 |
| `final_map_metric_targets` | `[P,6]` | 联邦内建筑、总建筑、星球类型、Gaia 星球、星区、卫星/空间站 | P1 |
| `moves_to_round_end_targets` | scalar | 距离本轮结束的语义决策数 | P1 |
| `moves_to_game_end_targets` | scalar | 距离终局的语义决策数 | P1 |

- [ ] 这些 head 是共享 trunk 的辅助监督，不全部进入 MCTS。TensorRT 生产 ONNX 默认保留 policy、多人 value/rank、VP/uncertainty；星图和发展 head 可由 export 配置决定是否保留用于诊断。
- [ ] 聚合指标虽可由星图标签推导，仍单独监督，因为终局计分直接依赖这些全局数量；两者必须在标签生成器中做一致性断言。
- [ ] 不预测当前 observation 已明确给出的资源、轮次、待决策阶段或合法动作；这里只预测轮末/终局未来量，防止网络通过复制输入获得无意义的低 loss。

##### 2.6.6 网络头与 ONNX 输出契约

| 网络头 | 推理输出 | 训练标签 | 生产 ONNX |
| --- | --- | --- | --- |
| `policy_head` | `policy_logits [B,A]` | `policy_visit_targets` | 必须保留 |
| `action_q_head` | `action_value [B,A,P]`、`action_vp [B,A,P]` | 两组 action Q 标签和 mask | P1 启用后保留 |
| `outcome_head` | `utility [B,P]`、`rank_logits [B,P,P]`、`pairwise_wdl_logits [B,P,P,3]` | utility、rank、pairwise WDL | utility 必须；其余可配置保留 |
| `score_head` | `vp_belief_logits [B,P,V]`、`vp_mean/stdev/lead [B,P]` | VP belief、最终 VP、lead、分解 | belief/mean/lead 必须保留 |
| `uncertainty_head` | `value_error [B,P]`、`vp_error [B,P]`、`settle_time [B,1]` | 短期误差和稳定时间 | P1 启用后保留并供 MCTS 使用 |
| `map_head` | planet/satellite/space-station 分类 logits | 星图控制与建筑标签 | 默认训练保留，生产导出可裁剪 |
| `development_head` | research/resource/tech/federation 输出 | 玩家发展辅助标签 | 默认训练保留，生产导出可裁剪 |

- [ ] ONNX 输出使用原始 logits 或未缩放回归量；softmax、softplus、tanh、合法动作 mask 和数值缩放统一由 C++ 后处理，并把版本及缩放常数写入 manifest。
- [ ] 训练 checkpoint 保存全部 head，生产 ONNX 可以裁剪不被 MCTS/诊断消费的辅助输出；裁剪前后 policy、utility、VP 和 uncertainty 的 golden fixture 输出必须一致。
- [ ] 每个 head 使用独立 loss、权重、有效样本计数和梯度统计；总 loss 不能掩盖某个 head 没有有效标签或量级失衡。

##### 2.6.7 样本权重、掩码和审计字段

以下字段不对应网络输出，但缺少时无法正确训练或定位标签错误：

- [ ] `label_schema_version`、`rules_version`、`player_count`、`network_id`、`model_hash`、`game_id`、`setup_seed`、`position_index`、`semantic_turn_index`、`round`、`player_to_move`、`starting_vp_offsets [P]` 和 `compensation_version`。
- [ ] `legal_masks [A]`、`player_masks [P]`、`pairwise_masks [P,P]`、`planet_masks [N]`、`board_space_masks [S]`、`action_value_masks [A]` 和每个可选 head 的 `*_loss_masks`。
- [ ] `sample_weights`、`policy_weights`、`value_weights`、`ownership_weights`；其中 sample weight 可结合完整搜索、policy surprise、value surprise、终局距离和是否重分析。
- [ ] `root_total_visits`、`search_simulations`、`search_temperature`、`root_noise_applied`、`source_generation`、`reanalyzed` 和 `terminal_valid`。
- [ ] 每局只保存一次终局真值和状态轨迹，位置行通过 `game_id + position_index` 关联；shuffle 后不得丢失标签数组或把逐位置数值塞进 JSON metadata。
- [ ] 标签生成采用“两阶段写入”：C++ selfplay 先缓存每步 observation、搜索统计与状态摘要，终局后反向生成 outcome、VP、ownership、短期目标和 masks，再原子写入 NPZ。

##### 2.6.8 明确不作为标签的内容

- `observations`、公开随机设置、种族、科技/计分/助推板块和当前资源是网络输入，不是预测标签。
- `legal_masks` 由规则引擎精确生成，是约束和审计数据，不训练一个“猜合法动作”的 head。
- 完整 JSON 历史用于人工复盘；训练所需目标必须是版本化 NPZ 数组，训练流程不得解析前端复盘 JSON。
- Gaia 没有隐藏手牌或战争迷雾，不增加 opponent-belief/hidden-information head。
- 动作文字、BGA 通知文本和素材 ID 不进入损失；规则事件必须先映射为稳定的枚举 ID。

实施顺序：P0 标签与多头 loss 在 G0 开始前完成；P1 在 C++ selfplay 能稳定输出完整搜索统计后启用；P2 必须分别做消融实验，只有提升守门结果或样本效率时才保留。每个 head 都需要独立 loss 曲线、有效样本数、梯度量级和开关，不能只记录总 loss。

### 3. export 进程改为 SWA -> ONNX

- [ ] 将导出输入从“已批准 `.pt`”改为训练产生的候选 checkpoint `.pt`，导出顺序先于 gatekeeper。
- [ ] 加载 checkpoint 的 SWA 权重，切换 `eval()`，冻结参数，不导出优化器、训练指标或 Python 对象。
- [ ] 使用固定的输入签名导出 ONNX：网络只接收观察张量，导出 2.6.6 定义的启用 head；合法动作掩码由 C++ 规则引擎在后处理阶段应用；明确动态 batch 维度。
- [ ] 固定并记录 ONNX opset、float32/float16 策略、输入/输出名称、head 版本、缩放常数、网络版本和状态维度。
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
- [ ] 按 2.6 节的新版本化 NPZ schema 写入完整训练标签、逐头 masks/weights 和独立复盘 metadata；写入采用临时文件后原子重命名。
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
                  └─> G0 bootstrap（K=0）
                          └─> C0 基准采样 -> C1 拟合 offsets-v1
                                  └─> C2 带补偿 G0 训练 -> C3 收敛
                                          └─> G1 -> 重新校准 -> G2 -> 重新校准
```

步骤 1 和步骤 5 可以在步骤 0 完成后并行。步骤 6 和步骤 7 必须共用同一套 C++ 规则与 TensorRT 适配层，不能分别实现两套逻辑。步骤 0.1/0.2 先冻结补偿 schema、拟合器和发布契约；实际 C0-C3 必须等混合语言流水线通过步骤 9，并产生一个非随机的零补偿 G0 bootstrap 模型后执行。步骤 2 先实现代际配置、迁移和数据换代机制；G1/G2 每次晋级后都需要重新校准 offset。步骤 8 不应在模型格式和 C++ 状态契约未稳定前提前替换现有流水线。

## 非目标

- 不把 ONNX Runtime 引入 selfplay 或 gatekeeper。
- 不把训练迁移到 C++；训练、优化器和 SWA 仍由 Python/PyTorch 负责。
- 不让 AI 执行竞拍，不增加竞拍 phase/action/policy head；离线补偿只产生开局 VP offset。
- 不把补偿评估变成第六个常驻进程，也不让运行中对局切换补偿版本。
- 不让 `.bin`、TensorRT engine 或临时导出文件成为训练恢复的唯一来源。
- 不在 C++ 版本通过完整回归前删除 Python 参考实现。
