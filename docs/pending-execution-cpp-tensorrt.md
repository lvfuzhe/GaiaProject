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

若最终选择当前向量残差网络，网络规模使用 `b{blocks}c{channels}` 命名，例如 `b10c128` 表示 10 个残差块、`hidden_size=128`；这里的 `c` 是隐藏宽度，不是二维卷积通道数。该命名和 `b10c128 -> b15c192 -> b20c256` 递进表是候选方案，必须先通过 2.0 节的表示架构基准，不得把 KataGo 的二维卷积块命名直接等同于当前平铺向量网络。

## 方案决策摘要

| 决策 | G0 最小闭环 | G0 后增强 | 明确不做 |
| --- | --- | --- | --- |
| 训练闭环 | 五进程异步闭环、SWA、ONNX、C++ TensorRT selfplay/gatekeeper | 多机共享目录和生产者背压 | TensorFlow、TFRecord、ONNX Runtime、C++ 训练 |
| 人数模型 | 2/3/4 人共用网络架构、标签语义和 MCTS 实现，但分别训练/发布 checkpoint | 未来另行消融可变人数 padding 模型 | 把 3 人权重直接加载到 2 人局 |
| 核心推理头 | policy、pairwise WDL、VP belief | 独立 TD utility/VP、uncertainty、动作 Q | rank head、独立 utility logits |
| MCTS 效用 | 只使用 pairwise utility，`beta_vp=0` | 小权重、有界、动态居中的 VP utility，消融通过后启用 | 直接把原始 VP 无界加入效用 |
| 数据 | 完整状态轨迹、终局真值、full/cheap search 标记、可审计窗口 | policy surprise、对称增强、重分析 | 解析前端 JSON 作为训练输入 |
| 辅助监督 | G0 只实现核心标签与 loss | 玩家发展 P1；星图/建筑 P2 | 让大量辅助头阻塞 G0 |
| 模型发布 | 多人配对守门、置信区间、原子发布 | 历史锚点模型池和非传递性检查 | 只按单个平均分点估计晋级 |

优先级统一解释如下，全文不再把“标签可保存”和“网络头必须启用”混为一件事：

- `P0`：首个 G0 闭环前必须完成，范围限于正确性、可恢复性和核心 policy/WDL/VP 训练。
- `P1`：G0 稳定后加入的低风险效率或样本质量增强；必须有独立开关和基线对照。
- `P2`：实验功能；只有消融显示守门棋力、样本效率或校准显著改善才保留。

## 与 KataGo 的适配对照

参考 KataGo 官方的 [Selfplay Training](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md)、[KataGo Methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)和[训练 selfplay 配置](https://github.com/lightvector/KataGo/blob/master/cpp/configs/training/selfplay1.cfg)。本项目借鉴训练工程和搜索方法，不复用围棋规则、特征或模型文件。

| KataGo 机制 | Gaia 处理 | 优先级与原因 |
| --- | --- | --- |
| C++ selfplay/gatekeeper + Python shuffle/train/export | C++ TensorRT 推理，Python PyTorch 训练，ONNX 交换 | P0，直接采用五进程职责边界 |
| SWA 推理权重 | checkpoint 同存普通权重、优化器和 SWA；只导出 SWA | P0，避免把恢复训练状态带入推理格式 |
| 二维卷积/全局池化棋盘网络 | 先比较平铺向量、实体 Transformer、星图 GNN/实体混合网络 | P0，Gaia 不是规则二维网格，不能按名称照搬 |
| W/L/no-result + score belief | 多人 pairwise WDL + 每玩家 VP belief；无 no-result | P0，适配 2-4 人和真实同分 |
| 二人零和、行动方视角价值回传 | 保留绝对玩家顺序的 `[P]` 效用；每个节点由实际决策玩家选择自身分量 | P0，不能使用二人符号翻转，充能等响应决策也必须切换决策者 |
| komi 条件与随机化 | 版本化多人 VP offset；训练时加入有界零和扰动 | P0，防止种族与固定 offset 共适应 |
| 棋盘尺寸、规则和 komi 随机化 | 版本化 Gaia setup 分布，覆盖地图、板块、种族和座位组合 | P0，随机设置是网络输入且必须监控覆盖率 |
| Playout Cap Randomization / cheap search | 按语义决策区分 full/cheap search，强制步骤不推理 | P0，避免兑换和确认步骤吞噬算力与样本 |
| 训练 lookback window 与每份新数据训练上限 | tapered window、fresh-data 下限、train/data ratio、no-repeat shard | P0，避免异步训练反复消费旧分片而过拟合 |
| 胜负效用 + 动态分差效用 | G0 只用 pairwise；P1 尝试小权重、有界 VP utility | P1，先保证目标稳定，再让险胜/大胜影响同胜率动作 |
| 多个 TD value/score 输出 | 独立 `td_utility_head` 和 `td_vp_head`，不污染终局 WDL/VP head | P1，目标语义不同必须分头 |
| Policy Surprise Weighting / root policy temperature | 保留原始与加噪 prior，按 KL 重采样；温度按语义阶段衰减 | P1，提升盲点动作学习和探索稳定性 |
| 棋盘对称增强与多棋盘尺寸 mask | 只使用当前 setup 的合法六角图自同构；2/3/4 人仍分模型 | P1，不应用破坏座位、种族或地图配置语义的伪对称 |
| NN cache、tree reuse、graph search | 完整状态哈希验证后启用缓存、树复用和转置图 | P1，先确保复合待决策状态不会误合并 |
| uncertainty weighting、subtree value bias、optimistic policy | 分别作为 P2 消融项 | P2，依赖稳定 TD 头和充足搜索统计 |
| ownership 辅助头 | Gaia 拆为主建筑、Lantids 共存、卫星、空间站等可选头 | P2，保存可派生标签，但不阻塞首轮训练 |
| 可选 gatekeeper | 本项目早期保留多人守门 | P0，当前规则和 C++ 迁移风险较高，先作为回归保护 |
| selfplay 中途切换新模型 | Gaia 只在新对局边界切换 | P0，保证一局的生成模型、offset 和复盘审计单一可追踪 |
| soft resignation / 胜势降 visits | G0 不截断；P1 只降 visits 并完成真实终局 | P1，Gaia 需要完整 VP、计分来源和规则复盘 |
| opening fork、side positions、非对称 playout | 先不进入 G0；按动作类别覆盖不足时再做分支续局实验 | P2，多玩家分支会改变阵容交互分布，必须单独消融 |

## 执行顺序

### 0. 冻结跨进程契约与基线

- [ ] 固定 `standard-v22` 的观察向量、合法动作掩码、策略输出、pairwise WDL 输出和 VP 输出的 shape、dtype、动作编号和玩家顺序。
- [ ] 固定 NPZ 训练样本格式：`raw/*.npz` 一文件只表示一局从初始设置到真实终局的完整对局，包含输入、掩码、G0 启用的稠密监督标签、逐头 loss mask/weight，以及可派生 P1/P2 标签的完整状态轨迹和复盘 metadata；未启用标签不得用全零数组伪装存在。`shuffle` 输出的训练 pack 可以跨局混排 position，但必须保留 `game_id + position_index`、终局标签关联和来源 raw hash。
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
- [ ] `final_vp_targets`、pairwise WDL、MCTS terminal utility 和守门结果都使用调整后分数；实际排名只由调整后终局 VP 离线派生用于统计，不建立 rank 训练标签或网络头。同时保存未补偿的 `raw_final_vp_targets`，用于重新估算而不污染原始强度数据。
- [ ] NPZ 和复盘分别记录 `published_vp_offsets [P]`、`vp_offset_perturbations [P]`、实际 `starting_vp_offsets = published + perturbation`、`compensation_version`、原始终局分和调整后终局分；不得只保存调整后结果或无法还原来源的合并 offset。
- [ ] selfplay 训练使用 `K_used = K_published + delta_K`：`delta_K [P]` 从有界分布采样并强制 `sum(delta_K)=0`，扰动范围、分布和随机种子写入 manifest/NPZ。公平性评估和 gatekeeper 固定 `delta_K=0`，只使用发布表，保证比较可复现。
- [ ] offset 扰动必须覆盖同一玩家/种族在多个相邻 VP 条件下的样本，防止网络把固定 offset 当作种族或座位捷径；扰动后的实际 `K_used` 必须进入 observation，并用于本局所有终局目标。
- [ ] 模型与 ONNX manifest 记录 `compensation_mode=offline-vp-offset`、补偿版本、归一化和取整规则；只有 observation、规则与补偿契约兼容的模型才能直接守门对战。

验收：同一初始设置仅改变 `K_i` 时，初始 `PlayerState.vp`、observation、终局 pairwise WDL、由其聚合的 MCTS 价值和离线统计排名随之改变；合法动作和其他规则状态保持不变。

#### 0.2 离线补偿估算与迭代闭环

补偿评估是五进程之外的离线任务，在模型代际边界运行，不增加第六个常驻训练进程。补偿表只有完成独立验收并原子发布后，才会被后续新对局读取；运行中的对局不得切换版本。

##### 0.2.1 C0：零补偿基准采样

- [ ] 先完成一个可稳定对局的 G0 bootstrap 模型，所有玩家使用 `K_i=0`；随机网络产生的结果不得用于估算种族补偿。
- [ ] 冻结一个 approved 模型作为评估器，评估期间不更新权重、搜索参数或规则版本。
- [ ] 为同一 setup seed 生成成组对局，轮换种族和座位；在合法种族组合范围内覆盖全部 14 个种族、先后手和对手组合。
- [ ] 2、3、4 人局分别采样；小地图/标准地图、地图种子、回合计分、终局计分、科技和助推布局全部记录为上下文，不把不同配置直接混成一个均值。
- [ ] 所有比较使用相同的 MCTS 模拟数、温度、根噪声方案和最大步数；评估局应关闭会妨碍配对比较的非必要随机项，并保留可复现种子。
- [ ] 单独写入 `compensation/evaluations/*.npz` 或结构化列式数据，至少包含模型哈希、规则版本、人数、setup hash、座位、种族、原始最终 VP、由终局 VP 派生的实际排名、pairwise WDL 和完整性标记；不进入普通训练 shard，实际排名也不作为网络标签。
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

- [ ] C++ selfplay 每局开始时根据人数、种族、座位和 context 读取固定版本补偿表，再按 0.1 节采样训练专用零和扰动，将实际 `K_used` 写入初始 VP；模型不执行竞价。
- [ ] 搜索、最终计分、pairwise WDL、由 WDL 聚合的 value target 和 gatekeeper 全部基于调整后分数；原始分数仅用于公平性分析，不能参与本局策略回传。
- [ ] 新补偿版本只对新开对局生效；进程轮询到更新后，必须等当前对局结束再切换，并在 shard 中写入实际版本和 offset。
- [ ] 旧 NPZ 保留其原 observation、offset 和 policy target，不把旧策略样本事后改成新 offset。训练窗口逐步提高新补偿版本自产数据权重，直到旧版本退出。
- [ ] 若保留显式 `starting_vp_offset`，不同补偿版本的完整旧样本可以共同训练；若 observation schema 改变，则必须启动新训练线或进行明确迁移。
- [ ] compensation 版本变化时重新建立或继续 SWA 的策略写入配置；大幅变化默认重置 SWA，小幅变化可在验证通过后延续，但必须记录选择。

验收：训练样本中的 observation VP、offset、原始分数、调整后分数、pairwise WDL 和聚合 utility 目标相互一致；从任一 shard 可以重放相同终局效用。统计排名必须能从调整后终局 VP 重算，不进入 label schema。

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
- [ ] SWA 起始点和更新周期按累计有效训练样本数定义，而不是仅按 optimizer step；梯度累积或 batch size 改变后不得静默改变平均节奏。
- [ ] 明确 SWA 起始样本数、更新频率、平均算法、设备和 checkpoint 恢复行为。
- [ ] checkpoint 同时保存普通 `model_state`、SWA `swa_state`、优化器状态和必要的计数器。
- [ ] checkpoint 同时保存学习率调度器、AMP GradScaler、累计有效样本数、window manifest/hash、数据游标以及 Python/NumPy/Torch CPU/CUDA RNG 状态；恢复后下一批样本、学习率和 SWA 更新时间必须可复现。
- [ ] 继续训练时恢复普通权重、SWA 权重和优化器；仅推理时不得加载优化器状态。
- [ ] 增加普通权重与 SWA 权重不同、恢复后继续平均、无 SWA 时明确报错或回退的测试。

验收：同一个 checkpoint 可以分别加载普通模型和 SWA 模型；SWA 累计样本计数器跨进程重启保持一致，改变 batch/梯度累积后按样本数触发的更新时间不漂移。

### 2. 网络表示选择与递进训练

#### 2.0 扩容前冻结表示架构

当前实现是把完整 observation 平铺后输入全连接残差网络；它只能作为基线，不能因类名包含 `KataGo` 就视为已获得 KataGo 二维卷积网络的空间归纳偏置。任何 `b10c128 -> b15c192 -> b20c256` 扩容都必须等待本节完成。

- [ ] 用同一规则版本、固定训练集、近似参数量、训练样本数和推理预算比较三类候选：平铺向量残差基线、实体 Transformer、星图 GNN + 玩家/科技/公共板块实体混合网络。
- [ ] 每个候选显式编码稳定实体 ID、实体类型、所属玩家、六角坐标/邻接、公共与个人状态及 mask；禁止依赖输入数组偶然排列学习空间关系。
- [ ] 同时比较固定 `[A]` 全连接策略头与参数化动作头。参数化方案至少拆分动作类型、目标星球/轨道/板块和必要参数，并能无损映射回固定 action ID 供规则引擎 mask。
- [ ] 基准同时报告 policy cross-entropy、pairwise WDL log loss/Brier、VP belief NLL/MAE/校准、固定搜索预算守门结果、TensorRT 延迟/吞吐和显存；不能只按监督 loss 选择。
- [ ] 选定方案后写入 `architecture_family`、实体/动作 schema 版本、参数量和 ONNX 输出契约；表示或参数化动作 schema 变化视为新训练线，不能加载不兼容 checkpoint。

验收：至少一个非平铺候选与当前向量基线完成同预算 A/B；最终选择有数据依据。若仍选择向量网络，文档记录其在棋力/吞吐上的优势，而不是默认沿用。

#### 2.1 固定代际配置与产物

- [ ] 在 `pipeline.json` 增加显式的 `architecture_family`、`network_id`、架构参数、`generation` 和训练阶段；向量残差候选使用 `residual_blocks/hidden_size`，其他架构使用各自版本化字段，禁止只根据文件名猜测。
- [ ] 模型 checkpoint、ONNX manifest、守门记录、训练状态和 Dashboard 都记录同一个 `network_id`，例如 `b10c128-g0`。
- [ ] 2、3、4 人局采用相同的递进表，但分别训练和晋级；不同玩家数的价值头维度不同，不共享 checkpoint 或 SWA 状态。
- [ ] 每个代际使用独立的 candidate、approved、rejected 和 TensorRT engine 缓存记录，同时保留上一代冠军用于跨代守门和回退。
- [ ] 固定跨代兼容条件：规则版本、观察维度、动作编号和玩家数必须相同；任一条件变化都启动新训练线，不得当作单纯扩容。

验收：任一 `.pt` 或 `.onnx` 都能仅通过 manifest 确定玩家数、代际、块数、隐藏宽度、规则版本、父模型和模型哈希。

#### 2.2 递进阶段

下表只适用于 2.0 节最终选择向量残差网络的情况；若选择实体 Transformer 或 GNN，必须先用等算力基准重新定义 G0/G1/G2 容量表，不能机械套用块数和宽度。

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
- [ ] 默认使用教师-学生过渡：冻结上一代 approved SWA 模型作为 teacher，初始化新一代 student，用现有 NPZ 的策略、pairwise WDL 与 VP 目标，加 teacher 的策略 logits、pairwise WDL logits 和 VP 输出进行预热。
- [ ] 蒸馏损失权重、温度、预热更新数和退出条件均配置化；预热结束后逐步衰减蒸馏权重，回到以自博弈目标为主的 AlphaZero 损失。
- [ ] 可将“复制公共张量切片并把新增残差路径初始化为近似恒等映射”作为实验选项，但必须先通过同输入数值回归；它不能成为默认迁移方式。
- [ ] 每次改变网络规模都重新创建优化器、学习率调度器和 SWA 累计器；不得加载上一代不兼容的优化器或 SWA 状态。
- [ ] 新一代 SWA 只在预热结束并完成规定更新数后开始累计，避免随机初始化和早期蒸馏状态污染平均权重。

验收：相同 NPZ batch 上，新一代 student 完成预热后，其合法动作策略、pairwise WDL、聚合 utility 和 VP 输出与 teacher 的误差达到配置门槛，并能继续使用真实训练目标优化。

#### 2.4 训练窗口、数据换代与跨代守门

- [ ] 新一代训练初期可以读取旧代 NPZ，因为观察、动作和训练目标契约不变；每个 shard 必须记录生成它的 `network_id` 和模型哈希。
- [ ] shuffler 每轮发布不可变 `window-manifest.json`，记录纳入的 shard、最早/最新模型代、位置数、按年龄衰减权重、各代占比和 SHA-256；trainer 只读取一个完整 manifest，不边训练边观察变化目录。
- [ ] replay 使用 tapered/lookback window：配置最小/最大位置数、年龄衰减和最新 approved 模型自产数据的最低占比。扩容初期可保留旧数据稳定训练，通过跨代守门后逐步提高新一代自产数据权重，避免旧数据永久主导。
- [ ] 配置 `max_train_samples_per_new_position` 和单周期训练上限；训练债务耗尽后必须等待新 selfplay 数据，不能因 GPU 空闲无限重复旧窗口。
- [ ] 同一个 window manifest 内启用 no-repeat shard：完整遍历一次后等待 shuffler 重新随机化并发布新 manifest；批内、分片内和跨分片随机种子必须可追踪。
- [ ] shuffler/trainer 使用 reader lease 或引用清单；垃圾回收只能删除不在当前/回滚窗口且没有活跃 reader 的 raw/shuffled shard。设置磁盘高低水位和生产者背压，禁止删除正在训练的数据来解除磁盘压力。
- [ ] 固定独立 validation seed/shard，不参与 selfplay replay 采样或参数更新；窗口变化时验证集保持稳定，同时另报最新代数据指标。
- [ ] 新一代未通过守门前，C++ selfplay 继续使用上一代 `approved/current.onnx`，不得因 train 已切换网络尺寸而提前切换生产模型。
- [ ] C++ gatekeeper 同时加载上一代冠军和新一代候选各自的 TensorRT engine；二者只要求输入/输出契约相同，不要求内部块数和宽度相同。
- [ ] 跨代比赛固定初始设置、座位轮换、随机种子和思考预算；MCTS 应按相同模拟次数比较，同时额外记录实际耗时和每秒模拟数。
- [ ] 新一代通过跨代守门后，原子更新 `approved/current.onnx` 与 manifest；旧冠军归档为 rollback 模型，selfplay 从下一局开始切换。
- [ ] 若新一代连续失败、训练发散、出现 NaN/Inf 或吞吐低于资源门槛，则保持旧冠军在线，并允许重新预热或调整网络阶段，不能跳过守门。

验收：可以完整演练一次 G0 -> G1 -> G2 晋级和一次失败回退；任何时刻 selfplay 都只读取已经通过守门且产物完整的模型。断开 selfplay 后，trainer 在训练债务耗尽时停止更新；重启 shuffler/train 不会重复消费、误删或泄漏 validation 数据。

#### 2.5 训练资源与超参数适配

- [ ] 分别测量三个网络在训练 batch、TensorRT 叶节点 batch、selfplay 并发局数下的显存、吞吐和延迟，不沿用同一套 batch 参数。
- [ ] 随网络扩大逐代调整训练 batch size、梯度累积、学习率、权重衰减和 checkpoint 周期，并把有效 batch size 写入训练记录。
- [ ] G0 冻结优化器、按有效训练样本数推进的 warmup/decay 调度、梯度裁剪、AMP 动态缩放和非有限梯度处理契约；配置变化必须进入训练 manifest，不能仅存在进程命令行。
- [ ] 每个 head 的 loss 先除以实际有效元素/玩家对数量，再乘显式权重；记录未加权 loss、加权 loss、有效标签数和 trunk/head 梯度范数，避免玩家数、合法动作数或某组稠密辅助标签隐式改变总 loss 尺度。
- [ ] 训练出现 NaN/Inf、GradScaler 连续回退、梯度范数异常或固定验证集明显退化时不生成可守门候选；保留最后有效 checkpoint 和失败批次的 shard/hash 供复现。
- [ ] 配置并监控 `selfplay_positions_generated / train_samples_consumed`、窗口新鲜度、训练债务、推理队列等待和 gatekeeper 队列长度；通过实测调节 selfplay/train 算力比例，不硬编码照搬 KataGo 的硬件比例。
- [ ] export 周期同时受累计新 selfplay 样本数、最小训练样本数和 gatekeeper 队列上限约束；后期模型变化变慢时降低候选导出频率，避免守门排队和 selfplay 频繁切换模型。
- [ ] 为每一代设置独立资源预算和最低吞吐门槛；若 `b20c256` 达不到门槛，优先降低并发或 batch，不静默退回不同网络结构。
- [ ] 保留同一固定验证集和固定种子对局集，用于比较各代策略损失、pairwise WDL log loss/Brier score、聚合 utility 误差、VP MAE/校准、守门结果和单位 GPU 时间的棋力收益。

验收：三个代际都有可复现的训练与推理基准，扩容决策同时依据棋力、数据新鲜度、吞吐和显存，而不是只看训练 loss。

#### 2.6 KataGo 风格神经网络头与 Gaia 训练标签

参考依据是 KataGo 官方的 [ONNX 输出定义](https://github.com/lightvector/KataGo/blob/master/docs/ONNX_Model_Files.md)、[短期价值与分数目标](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#short-term-value-and-score-targets)以及 [PyTorch NPZ 读取结构](https://github.com/lightvector/KataGo/blob/master/python/katago/train/data_processing_pytorch.py)。KataGo 的核心是 policy、W/L/no-result、score belief/lead/uncertainty、ownership 和逐动作 Q。Gaia 是 2-4 人、按 VP 排名、允许同星球共存且没有隐藏信息，不能直接照搬围棋通道。

当前 `TrainingExample` 只有 observation、legal mask、MCTS policy 和终局 utility 四组数据。下列字段均属于待实现的新 label schema；在 schema 版本升级前，旧 NPZ 不得被误认为具有缺失 head 的零值标签。

符号约定：`P` 为本模型的玩家数，`Q=P(P-1)/2` 为无序玩家对数量，`A` 为固定动作空间，`N` 为星球槽位，`S` 为可放卫星/空间站的星图位置，`T=6` 为科技轨数量，`H` 为短/中/长三个时间尺度。所有玩家维度必须使用与 observation 相同的座位顺序；2、3、4 人模型分别保存，不通过 padding 混训。

优先级沿用文档开头定义。G0 的网络输出严格限制为 policy、pairwise WDL 和 VP belief；完整状态轨迹仍需保存，以便 P1/P2 标签以后离线派生，但未启用的辅助标签不要求在 G0 NPZ 中展开为稠密数组。

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
- [ ] P2 乐观策略只在基础策略、VP head 和跨代守门稳定后启用；多人局默认追求风险调整后的 pairwise utility 或 VP 分差。若以后改为直接优化第一名率，应新增独立的 `first_place_head [P]` 并做消融验证，而不是恢复完整 rank head。

##### 2.6.2 多人结果与价值头标签

| NPZ 标签 | Shape | 来源与用途 | 优先级 |
| --- | --- | --- | --- |
| `pairwise_wdl_targets` | `[P,P,3]` | 行玩家相对列玩家的胜/平/负 one-hot；这是 P0 多人价值真值，必须保留完整矩阵 | P0 |
| `final_utility_targets` | `[P]` | 从 `pairwise_wdl_targets` 聚合出的平均两两胜负效用，用于训练审计及 multiplayer MCTS 终局回传，不对应独立 utility head | P0 |
| `root_value_targets` | `[P]` | selfplay MCTS 根节点的聚合多人效用原始记录，用于构造 TD 目标、重分析和审计；不直接对应网络头 | P1 |
| `td_utility_targets` | `[H,P]` | 对后续根 MCTS 聚合效用做三种指数衰减平均的短/中/长期目标 | P1 |
| `shortterm_utility_error_targets` | `[H,P]` | 独立 TD utility 预测与对应未来根价值目标的误差，供搜索置信度加权 | P1 |
| `value_settle_time_targets` | scalar | 从当前位置到 pairwise utility 基本稳定的剩余语义决策数，映射 KataGo variance-time | P2 |

- [ ] Gaia 的时间尺度按“语义决策”而不是原始 action 计数；免费兑换、被动充能确认、科技选择等微步骤不能把时间轴无限拉长。先从约 8、24、80 个语义决策的均值范围做基准，再根据完整对局长度校准 lambda。
- [ ] 正常完成的 Gaia 对局没有 KataGo 的 `no-result` 类别；非法、损坏、超出最大步数或缺少终局的对局标为 `terminal_valid=0`，不得伪装成平局训练。真实同分通过 pairwise draw 表示，实际并列排名仅由终局 VP 派生用于统计。
- [ ] 不实现 `final_rank_targets` 或 `rank_head`。2 人局中它与 WDL 完全冗余；3/4 人局所需的期望排名可由 pairwise WDL 计算，完整联合名次分布和第一名概率不属于当前 MCTS 目标。
- [ ] 网络只输出 `pairwise_wdl_logits [B,Q,3]`，每个 `i < j` 无序玩家对只计算一次 loss；C++/Python 后处理镜像展开为 `[B,P,P,3]`，保证 `win(i,j)=loss(j,i)` 且 `draw(i,j)=draw(j,i)`。完整目标矩阵的对角线必须 mask，不能双计数非对角 loss。
- [ ] C++ 与 Python 都按同一公式把 WDL 概率聚合成 MCTS 消费的 `[P]` 多人效用：`u_i = sum_{j != i}(p_win(i,j) - p_loss(i,j)) / (P - 1)`；NPZ 中的 `final_utility_targets` 必须逐样本通过这一公式重算校验。
- [ ] 统计所需的期望排名可按 `E[rank_i] = 1 + sum_{j != i}(p_win(j,i) + 0.5 * p_draw(i,j))` 计算，但两两边缘概率不能恢复完整联合名次分布；守门的第一名率和实际排名直接从完整对局终局 VP 统计，不能伪装成网络预测。
- [ ] 终局 pairwise WDL 只监督 `pairwise_value_head`；`root_value_targets` 是构造未来指数平均的原始序列，只有 `td_utility_targets` 监督独立 `td_utility_head`。禁止把终局 one-hot、单步根搜索值和不同时间尺度 TD 目标同时回归到同一输出。
- [ ] `td_utility_head` 每个时间尺度的 `[P]` 输出必须中心化为和为 0，保持与 pairwise 聚合效用一致；不得让独立 TD 输出引入所有玩家同时获益的漂移分量。
- [ ] `shortterm_utility_error_targets` 使用对应 TD 预测的 `stop_gradient` 误差构造，误差头采用 Huber 等抗异常值损失；不得反向推动 TD 输出去迎合误差预测。
- [ ] G0 固定 `beta_vp=0`，搜索只使用 pairwise utility。P1 消融可启用 `u_search_i = u_pairwise_i + beta_vp * tanh((vp_lead_i - c_i) / vp_scale)`：`c_i` 是一次根搜索内固定的根 VP lead 中心，`vp_scale`、`beta_vp` 配置化且 score 项必须有界；终局节点使用真实 VP 代入同一公式。
- [ ] 启用 VP utility 后必须重新校准 cPUCT/FPU 的效用范围，并分别守门 `beta_vp=0` 与候选配置。目标是区分 pairwise 概率近似相同的动作，不允许 VP 项压倒胜负效用。

##### 2.6.3 VP、分数分布和不确定性标签

| NPZ 标签 | Shape | 来源与用途 | 优先级 |
| --- | --- | --- | --- |
| `raw_final_vp_targets` | `[P]` | 不包含虚拟竞拍 offset 的每名玩家原始最终 VP，用于补偿重估和审计 | P0 |
| `final_vp_targets` | `[P]` | 包含 `starting_vp_offsets` 的每名玩家调整后最终 VP，供 VP head、pairwise WDL 真值和 MCTS 终局效用使用 | P0 |
| `final_vp_belief_targets` | `[P,V]` | 固定 VP 桶的终局分布，包含下溢/上溢桶，类似 KataGo score belief | P0 |
| `final_vp_component_targets` | `[P,6]` | `原始局内VP、开局offset、科技轨终局分、剩余资源分、终局板块1、终局板块2` | P1 |
| `root_vp_targets` | `[P]` | selfplay MCTS 根节点聚合的预计终局 VP 原始记录，用于构造 TD VP、重分析和审计；不直接对应网络头 | P1 |
| `td_vp_targets` | `[H,P]` | 后续根节点预计终局 VP 的短/中/长期指数平均 | P1 |
| `root_vp_lead_targets` | `[P]` | 指定比例高预算根搜索得到的相对其他玩家平均 VP lead，用于可选独立 lead 校准 | P1 |
| `shortterm_vp_error_targets` | `[H,P]` | 独立 TD VP 预测与对应未来根 VP 目标之间的误差 | P1 |
| `vp_source_targets` | `[P,K]` | 更细 VP 台账：轮次计分、联邦、科技、助推、QIC 行动、种族能力等 | P2 |

- [ ] `V` 的范围和桶宽由完整随机对局统计确定并写入 label manifest；不允许训练脚本和 TensorRT 后处理各自假设不同区间。
- [ ] manifest 固定下溢/上溢桶的代表值和 moments 算法，并设置训练/验证 overflow-rate 门槛；超过门槛必须升级 VP schema 或扩大范围，不能让边界桶长期压扁大分差。
- [ ] G0 的 VP head 只直接输出 `vp_belief_logits [B,P,V]`。softmax 后用固定 VP 桶中心计算 `vp_mean` 和 `vp_stdev`，再计算 `vp_lead_i = vp_mean_i - mean(vp_mean_{j != i})`；三者是确定性后处理，不再建立互相可能矛盾的独立 G0 回归输出。
- [ ] 可在 belief loss 之外对派生 `vp_mean` 增加小权重 Huber 校准 loss，但这不增加 ONNX 输出。只有 `root_vp_lead_targets` 的高预算数据足够且消融有效时，P1 才增加独立 `lead_head`；它不能用当前局面 VP 差冒充预计终局 lead。
- [ ] 任意玩家对的期望终局分差可由 `vp_mean_i - vp_mean_j` 计算；独立的每玩家 VP belief 不包含玩家间相关性，不能据此替代 pairwise WDL head，也不能假设其能精确恢复胜率或第一名率。
- [ ] 必须断言 `raw_final_vp_targets + starting_vp_offsets == final_vp_targets`。P1 六项分解可直接从 `final_scores()` 和本局 offset 计算；P2 细分台账要求所有 VP 变化携带稳定 `score_source_id`，不能依赖动作说明文字解析。

##### 2.6.4 星图控制与建筑头标签

这组标签对应 KataGo ownership，但 Gaia 不能使用单一所有权值：Lantids 可与主建筑共存，Ivits 空间站和卫星也不位于普通星球槽位。

| NPZ 标签 | Shape | 来源与用途 | 优先级 |
| --- | --- | --- | --- |
| `final_planet_owner_targets` | `[N,P+1]` | 终局主建筑所有者分类，额外一类表示无人殖民 | P2 |
| `final_coexisting_owner_targets` | `[N,P+1]` | Lantids 共存矿场所有者分类，额外一类表示不存在 | P2 |
| `final_structure_targets` | `[N,6]` | 无建筑、矿场、贸易站、研究所、行星研究院、学院 | P2 |
| `final_planet_terrain_targets` | `[N,terrain_count]` | 终局地形，包括 Transdim 转 Gaia 和 Lost Planet | P2 |
| `final_federated_targets` | `[N,2]` | 主建筑与共存矿场是否已计入联邦 | P2 |
| `final_satellite_owner_targets` | `[S,P+1]` | 每个星图位置的卫星所有者或空 | P2 |
| `final_space_station_owner_targets` | `[S,P+1]` | Ivits 空间站所有者或空 | P2 |
| `final_space_station_federated_targets` | `[S]` | 空间站是否已计入联邦 | P2 |
| `round_end_board_targets` | 同上并带 round 维度 | 最近轮末的较短期星图目标，降低只看终局的高方差 | P2 |

- [ ] 每组位置标签必须带 `planet_masks [N]` 或 `board_space_masks [S]`，屏蔽小地图、未启用 Lost Planet 槽位和不存在的位置。
- [ ] primary owner、coexisting owner 和 structure 必须分头预测，不能把 Lantids 共存编码成互斥的单 owner 类别。
- [ ] Gaiaformer 若在数据审计中终局几乎总为空，不建立无信息的终局 head；它只进入轮末/短期发展标签。

##### 2.6.5 玩家发展与终局计分辅助标签

| NPZ 标签 | Shape | 内容 | 优先级 |
| --- | --- | --- | --- |
| `final_structure_count_targets` | `[P,5]` | 五类建筑终局数量 | P1 |
| `final_research_level_targets` | `[P,T,6]` | 六条科技轨最终 0-5 层 categorical | P1 |
| `final_colonized_type_targets` | `[P,terrain_count]` | 各星球类型最终是否已殖民 | P1 |
| `final_scoring_metric_targets` | `[P,2]` | 本局两个终局计分板块的原始指标值 | P1 |
| `final_scoring_award_targets` | `[P,2]` | 处理排名、并列和 2 人中立玩家后的实际奖励 VP | P1 |
| `final_resource_targets` | `[P,10]` | 信用点、矿石、知识、QIC、三能量区、盖亚区能量、可用/盖亚区 Gaiaformer | P2 |
| `round_end_resource_targets` | `[P,10]` | 最近轮末的同组资源，作为短期经济监督 | P2 |
| `final_brainstone_targets` | `[P,5]` | Taklons 脑石不存在或位于 I/II/III/盖亚区的 categorical | P2 |
| `final_academy_type_targets` | `[P,2]` | 知识学院和 QIC 学院的终局数量 | P2 |
| `final_tech_targets` | `[P,standard+covered+advanced]` | 标准科技、被覆盖状态和高级科技所有权 | P2 |
| `final_federation_targets` | `[P,F+4]` | 各类联邦片、未使用/已使用数、版图联邦数和 Gleens 专属联邦数 | P2 |
| `final_map_metric_targets` | `[P,6]` | 联邦内建筑、总建筑、星球类型、Gaia 星球、星区、卫星/空间站 | P1 |
| `moves_to_round_end_targets` | scalar | 距离本轮结束的语义决策数 | P1 |
| `moves_to_game_end_targets` | scalar | 距离终局的语义决策数 | P1 |

- [ ] 这些 head 是共享 trunk 的可选辅助监督，不进入 G0 MCTS。完整状态轨迹先保证以后可离线派生标签；玩家发展 head 从 P1 开始逐组消融，星图/建筑 head 保持 P2。生产 ONNX 默认裁剪未被搜索消费的辅助输出，不导出 rank 输出。
- [ ] 聚合指标虽可由星图标签推导，仍单独监督，因为终局计分直接依赖这些全局数量；两者必须在标签生成器中做一致性断言。
- [ ] 不预测当前 observation 已明确给出的资源、轮次、待决策阶段或合法动作；这里只预测轮末/终局未来量，防止网络通过复制输入获得无意义的低 loss。

##### 2.6.6 网络头与 ONNX 输出契约

| 网络头 | 推理输出 | 训练标签 | 生产 ONNX |
| --- | --- | --- | --- |
| `policy_head` | `policy_logits [B,A]` | `policy_visit_targets` | 必须保留 |
| `action_q_head` | `action_value [B,A,P]`、`action_vp [B,A,P]` | 两组 action Q 标签和 mask | P1 启用后保留 |
| `pairwise_value_head` | `pairwise_wdl_logits [B,Q,3]` | 仅终局 pairwise WDL | 必须保留；后处理展开，不导出独立 utility/rank logits |
| `score_head` | `vp_belief_logits [B,P,V]` | VP belief；mean/stdev/lead 从 belief 派生 | 必须保留 |
| `td_head` | `td_utility [B,H,P]`、`td_vp [B,H,P]` | 独立 TD utility/VP 目标 | P1 启用后保留 |
| `lead_head` | `vp_lead [B,P]` | 高预算 `root_vp_lead_targets` | P1 数据足够且消融通过后保留 |
| `uncertainty_head` | `utility_error [B,H,P]`、`vp_error [B,H,P]`、`settle_time [B,1]` | 短期误差和稳定时间 | P1 启用后保留并供 MCTS 使用 |
| `map_head` | planet/satellite/space-station 分类 logits | 星图控制与建筑标签 | P2 消融，生产默认裁剪 |
| `development_head` | research/resource/tech/federation 输出 | 玩家发展辅助标签 | P1 分组消融，生产默认裁剪 |

- [ ] ONNX 输出使用原始 logits 或未缩放回归量；softmax、pairwise 镜像展开、utility 聚合、VP belief moments、softplus、合法动作 mask 和数值缩放统一由 C++ 后处理，并把版本及缩放常数写入 manifest。
- [ ] 训练 checkpoint 保存全部已启用 head，生产 ONNX 可以裁剪不被 MCTS/诊断消费的辅助输出；裁剪前后 policy、pairwise WDL、聚合 utility、VP 和 uncertainty 的 golden fixture 输出必须一致。
- [ ] 每个 head 使用独立 loss、权重、有效样本计数和梯度统计；总 loss 不能掩盖某个 head 没有有效标签或量级失衡。

##### 2.6.7 样本权重、掩码和审计字段

以下字段不对应网络输出，但缺少时无法正确训练或定位标签错误：

- [ ] `label_schema_version`、`rules_version`、`player_count`、`network_id`、`model_hash`、`game_id`、`setup_seed`、`position_index`、`semantic_turn_index`、`round`、`player_to_move`、`starting_vp_offsets [P]`、`compensation_version`、`search_tier` 和 `symmetry_id`。
- [ ] `legal_masks [A]`、`player_masks [P]`、`pairwise_masks [P,P]`、`planet_masks [N]`、`board_space_masks [S]`、`action_value_masks [A]` 和每个可选 head 的 `*_loss_masks`。
- [ ] `sample_weights`、`policy_train_masks`、`policy_weights`、`pairwise_value_weights`、`vp_weights`、`ownership_weights`；其中 sample weight 可结合完整/cheap 搜索、policy surprise、pairwise utility surprise、VP surprise、终局距离和是否重分析。
- [ ] `root_total_visits`、`search_simulations`、`search_temperature`、`root_noise_applied`、`source_generation`、`reanalyzed` 和 `terminal_valid`。
- [ ] 每个 raw NPZ 只保存一局，并只保存一次终局真值和完整状态轨迹；位置行通过 `game_id + position_index` 关联。shuffle pack 可以包含多局的位置，但不得丢失标签数组、局边界、来源 raw hash，或把逐位置数值塞进 JSON metadata。
- [ ] 标签生成采用“两阶段写入”：C++ selfplay 先缓存每步 observation、搜索统计与状态摘要，终局后反向生成 outcome、VP、ownership、短期目标和 masks，再原子写入 NPZ。

##### 2.6.8 明确不作为标签的内容

- `observations`、公开随机设置、种族、科技/计分/助推板块和当前资源是网络输入，不是预测标签。
- `legal_masks` 由规则引擎精确生成，是约束和审计数据，不训练一个“猜合法动作”的 head。
- 完整 JSON 历史用于人工复盘；训练所需目标必须是版本化 NPZ 数组，训练流程不得解析前端复盘 JSON。
- Gaia 没有隐藏手牌或战争迷雾，不增加 opponent-belief/hidden-information head。
- 动作文字、BGA 通知文本和素材 ID 不进入损失；规则事件必须先映射为稳定的枚举 ID。

实施顺序：G0 前只完成 policy、pairwise WDL、VP belief 及其审计字段和 loss；完整状态轨迹负责保留未来派生 P1/P2 标签的能力。P1 在 C++ selfplay 能稳定输出完整搜索统计后逐项启用；P2 必须分别做消融实验，只有提升守门结果或样本效率时才保留。每个 head 都需要独立 loss 曲线、有效样本数、梯度量级和开关，不能只记录总 loss。

#### 2.7 语义决策、搜索预算与数据增强

Gaia 的一次规则行动可能展开为多次兑换、选板块、充能确认或自动收入步骤。原始 action 数不能直接等同于值得训练 policy 的决策数。

- [ ] 规则引擎标记 `semantic_decision_id` 和动作类别。只有一个合法动作且不含玩家真实选择时自动执行，不调用网络、不启动 MCTS、不生成 policy loss；完整状态轨迹仍记录该步骤用于复盘和终局标签。
- [ ] 发布版本化 `setup_distribution`：分别定义 2/3/4 人局、小/标准地图、星区排列与旋转、轮次/终局计分、科技/高级科技/助推板块、合法种族组合及座位分配的采样权重。每局记录分布版本和实际 setup hash，训练时不得依赖未记录的进程默认值。
- [ ] 按人数、地图模式、种族、座位、板块和主要交互组合持续报告覆盖率；设置最低样本数和最大单 bucket 占比，避免高频配置淹没稀有种族/动作。固定验证集、gatekeeper 和公平性评估使用各自冻结的 setup 清单，不随训练采样器漂移。
- [ ] 定义 `search_tier = forced | cheap | full | lead_estimation`。关键分支使用 full search；普通分支按配置概率使用 cheap/full；指定小比例位置使用高预算 lead estimation。各轮次、行动类别和玩家人数都设置最低 full-search 覆盖率，避免某类动作长期只有弱标签。
- [ ] full search 默认 `policy_train_mask=1`。cheap search 默认不训练 policy 或显著降权，但仍可提供终局 WDL/VP 标签；P1 可在 cheap search 发现高 KL surprise 时纳入 policy。保存实际访问数和每个 head 的独立权重，不能仅凭配置名推断样本质量。
- [ ] G0 不允许提前认输或截断终局，保证完整 VP、计分来源和复盘轨迹。P1 可在连续高置信状态降低 visits 并降低对应 policy 权重，但仍必须把对局按规则运行到真实终局。
- [ ] 根噪声使用固定“总浓度”而不是每合法动作固定 alpha；总浓度先按版本化动作类别权重分配，再在类别内部展开，避免合法兑换动作数量改变探索强度。P1 再消融基于 prior 的 shaped noise 和 policy target pruning。
- [ ] 根 policy softmax 温度、落子采样温度和噪声分别配置，并按轮次/语义决策数衰减；gatekeeper 全部关闭根噪声，最终动作温度固定为 0。
- [ ] 保存 raw prior、实际加噪 prior 和 visit target。P1 policy surprise 使用 visit target 相对实际搜索 prior 的 KL，并同时保留均匀基础采样权重，禁止让极少数高 surprise 样本占满窗口。
- [ ] P1 只对当前 setup 的真实六角图自同构做旋转/镜像增强；observation、坐标、合法动作、policy、星图标签和历史轨迹必须使用同一双射。用 property test 验证变换前后合法动作集合和终局结果一致，不对不具备对称性的随机板块布局强行增强。
- [ ] P1 评估把当前行动玩家规范到玩家维第 0 位并循环置换其他玩家；只有 observation、动作语义、pairwise WDL、VP 与回传索引都可逆映射时才启用，不能丢失真实座位/顺位输入。
- [ ] P1 在完整状态哈希覆盖待决策类型、当前玩家、offset、全部资源/板块/星图状态后，再启用 NN eval cache、根树复用和转置图；随机短局必须证明命中缓存不改变合法动作和搜索结果。

验收：按玩家人数、轮次和动作类别报告 forced/cheap/full 数量、平均 visits、policy 有效权重和 GPU 时间；自动步骤不再占用推理预算，且任何增强样本都能逆变换回原状态和动作。

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
- [ ] 实现与 Python 共用定义的 `semantic_decision_id`、动作类别和 forced-step 判定；自动执行只消除无选择步骤，不能越过充能接受/拒绝、资源组合、板块选择等真实决策。
- [ ] 明确 C++ 状态复制/撤销策略，优先使用紧凑数组、结构共享或可回滚状态，避免每个节点深拷贝大对象。
- [ ] 为每一类动作建立 Python/C++ 双向序列化和逐状态对比测试。
- [ ] 定义完整状态哈希，覆盖当前玩家、待决策类型、offset、资源、科技、板块、星图和影响后续合法动作/收益的所有状态；使用固定种子执行短局、完整局和边界规则测试，比较合法动作集合、资源、VP、终局、状态哈希和 NPZ trace。

验收：C++ 与 Python 在 golden fixtures 和随机短局上产生相同的状态摘要、合法动作和最终结果。

### 6. C++ 多人 PUCT/MCTS Selfplay

本文所需的是完全信息多人 MCTS：状态中没有待确定化的隐藏信息，价值始终采用绝对玩家顺序。旧讨论中的 `PIMCTS` 在此仅指 perfect-information MCTS，不引入针对手牌或战争迷雾的 determinization；实现和配置统一使用 `multiplayer_mcts` 命名。

- [ ] 实现绝对玩家顺序的多玩家价值回传；叶节点先把 pairwise WDL 按 2.6.2 的唯一公式聚合为 `[P]` utility。G0 只回传 pairwise utility；P1 启用 VP utility 时按同节公式使用一次根搜索内固定的 VP 中心，并对终局/非终局节点保持一致。
- [ ] 选择阶段由节点状态的实际 `current_player` 使用自己的效用分量；不得使用 KataGo 二人零和的父子符号翻转。被动充能、资源选择等响应节点由响应玩家优化自身动作，完成后再返回原行动流程。
- [ ] 实现 2.7 节的 forced/cheap/full/lead-estimation 调度、根 policy 温度、动作采样温度和分类总浓度噪声；所有搜索参数按人数、网络代际和语义动作类别版本化，禁止沿用当前单一常数而不重新标定。
- [ ] 将 MCTS 树节点改为紧凑结构，使用线程池运行多局 self-play；每棵树的随机种子必须可追踪。
- [ ] 将叶节点请求提交给 TensorRT 批量推理队列，支持虚拟损失或等价并发机制。
- [ ] G0 先实现可审计的 PUCT/FPU/根噪声基线；P1 再分别加入 NN eval cache、树复用、转置图、root LCB、policy surprise；动态 cPUCT、subtree value bias、uncertainty-weighted playout 和 optimistic policy 保持 P2，不能打包一次启用。
- [ ] 按 2.6 节的新版本化 NPZ schema 为每个真实完成的对局写入一个 raw NPZ，包含完整训练标签、逐头 masks/weights、初始 setup、逐步状态轨迹、终局结果和独立复盘 metadata；写入采用临时文件后原子重命名，损坏或未到真实终局的文件不得进入 shuffle 窗口。
- [ ] C++ selfplay 轮询 `approved/current.onnx`，只在完整模型发布后切换，不读取未完成文件。
- [ ] 进程状态、对局数、语义决策数、forced/cheap/full 比例、推理 batch、positions/s、模型哈希、错误和最近 shard 写入现有五进程监控协议。

验收：C++ selfplay 生成的 NPZ 可被现有 Python shuffle/train 读取；同等模拟次数下规则结果与 Python 参考实现一致。

### 7. C++ TensorRT Gatekeeper

- [ ] 将 gatekeeper 输入改为 `exported/candidate-*.onnx`，当前模型改为 `approved/current.onnx`。
- [ ] 使用与 selfplay 相同的 C++ Gaia 规则、MCTS 和 TensorRT 推理适配层，避免守门与训练数据生成规则分叉。
- [ ] 以 `setup seed + 地图/计分/科技/助推配置 + 种族组合` 为配对统计块；每个块让候选依次占据全部座位，其余座位使用冠军，且双方搜索模拟数完全相同。时间和 simulations/s 另作性能指标，不混入棋力主检验。
- [ ] gatekeeper 固定发布版 VP offset、`delta_K=0`、关闭根噪声、动作温度 0、禁用训练专用 fork/增强；每局记录完整阵容、座位、种族、最终 VP、pairwise WDL、实际排名、搜索参数和模型哈希。
- [ ] primary 晋级指标是按配对块聚合的候选平均 pairwise utility；使用 block bootstrap 置信区间或预注册顺序检验，不按单个点估计过门。第一名率、平均/分位 VP、非法动作、超步数和崩溃率作为保护指标。
- [ ] 配置最小/最大配对块数、通过/拒绝边界和多重候选排队策略；任何提前停止都必须保存当时置信区间和完整已测块，不能因偶然连胜立即批准。
- [ ] P1 增加混合阵容测试，让候选分别占 `1..P-1` 个座位，并维护少量历史 approved 锚点模型；若候选只克制当前冠军却对锚点显著退化，标记非传递风险并进入扩展测试，不直接晋级。
- [ ] 通过后原子发布 `approved/current.onnx` 及 manifest；拒绝模型写入 rejected 目录和结构化日志。
- [ ] 首个模型没有冠军时定义 bootstrap 规则，并测试重复候选、半成品文件和进程重启恢复。

验收：候选只有在 C++ TensorRT 的多人配对检验通过后才会成为 selfplay 可见模型；从结构化逐局记录可重算 primary/保护指标、置信区间和最终决定，并能追溯模型及配置哈希。

### 8. 五进程编排与 Dashboard

- [ ] 将进程角色改为：C++ `selfplay`、Python `shuffle`、Python `train`、Python `export`、C++ `gatekeeper`。
- [ ] 更新 `pipeline.json`、状态文件、日志目录和产物目录协议，区分 `.pt` 训练 checkpoint、`.onnx` 推理模型、TensorRT engine 缓存、不可变训练窗口 manifest 和 reader lease。
- [ ] 所有 raw shard、shuffle pack、candidate 和状态文件使用 `run_id + producer_id + 原子序号/hash` 唯一命名；P1 允许多个 selfplay producer 共享目录，但目录创建、模型轮询和垃圾回收必须保持幂等，不能依赖单进程本地计数器避免冲突。
- [ ] 让 supervisor 以可配置路径启动 C++ 可执行文件，并检查启动前的 CUDA/TensorRT、模型和规则版本。
- [ ] supervisor 根据磁盘水位、训练债务、shuffler/gatekeeper 队列长度实施背压；背压只暂停生产或导出，不删除活跃窗口、不跳过守门、不发布半成品。
- [ ] 五个监控页面显示进程语言、推理后端、模型哈希、SWA 导出状态、engine 构建耗时、训练窗口新鲜度、train/data ratio、forced/cheap/full 比例、gatekeeper 置信区间、吞吐和错误。
- [ ] 保留停止、重启、断点恢复和单进程 `--once` 诊断语义；C++ 进程异常退出时不得静默继续写入数据。
- [ ] 更新 README、流水线文档和运行示例，明确当前 Python 方案与新 C++ 方案的切换条件。

验收：Dashboard 可以一键启动和停止混合语言五进程；任一进程重启后不会重复消费或误发布模型。

### 9. 性能、正确性与发布门槛

- [ ] 对比 Python selfplay、C++ batch=1、C++ 批量 TensorRT 和多局并发的 simulations/s、positions/s、对局时长、GPU 利用率和显存。
- [ ] 分别测试 FP32、FP16 和可选 TF32；任何精度模式切换都要通过规则和网络 golden fixtures。
- [ ] 运行随机合法对局、状态哈希、pairwise 镜像/聚合、VP belief moments、TD 头隔离、NPZ 读取、窗口换代、模型切换、断点恢复和多人守门回归测试。
- [ ] 注入进程终止、磁盘高水位、半写 shard/manifest、过期 TensorRT cache 和候选积压故障；验证原子写入、reader lease、背压、重启幂等和回滚路径。
- [ ] 对每个合法星图自同构做 observation/action/target 往返 property test；对 forced-step 折叠前后做逐状态对比，确保没有跳过玩家决策或改变终局结果。
- [ ] 至少完成一轮小规模 A/B：同一初始设置和模型下，比较 Python 与 C++ 的动作、终局分数和训练样本摘要。
- [ ] 设置上线门槛：无规则差异、无非法动作、无 NaN/Inf、核心 head 契约一致、训练窗口无重复/泄漏、模型哈希可追踪、守门统计可重算且吞吐达到基线目标后，才替换默认 selfplay/gatekeeper。
- [ ] 保留 Python selfplay/gatekeeper 作为离线参考和故障回退，直到 C++ 版本完成稳定性观察期。

## 依赖与不可并行项

```text
0 契约冻结
  ├─> 1 SWA checkpoint
  ├─> 2.0 表示架构基准与选择
  │       └─> 2.1-2.6 G0 核心网络、标签、窗口与迁移契约
  │               └─> 3 SWA -> ONNX export
  │                       └─> 4 TensorRT adapter
  └─> 5 C++ Gaia rules

2.7 P0 搜索采样契约 + 4 + 5
  ├─> 6 C++ selfplay
  └─> 7 C++ gatekeeper

2 + 3 + 4 + 5 + 6 + 7
  └─> 8 五进程编排与 Dashboard
          └─> 9 性能、正确性与发布
                  └─> G0 bootstrap（K=0）
                          └─> C0 基准采样 -> C1 拟合 offsets-v1
                                  └─> C2 带补偿 G0 训练 -> C3 收敛
                                          └─> P1 逐项消融 -> G1 -> 重新校准 -> G2 -> 重新校准
```

步骤 1、2.0 和步骤 5 可以在步骤 0 完成后并行。2.0 没有完成前不得承诺向量网络扩容表；2.6 先交付 G0 三个核心 head，P1/P2 不能阻塞闭环。步骤 6 和步骤 7 必须共用同一套 C++ 规则、搜索效用与 TensorRT 适配层，不能分别实现两套逻辑。步骤 0.1/0.2 先冻结补偿 schema、拟合器和发布契约；实际 C0-C3 必须等混合语言流水线通过步骤 9，并产生一个非随机的零补偿 G0 bootstrap 模型后执行。G1/G2 每次晋级后都需要重新校准 offset。步骤 8 不应在模型格式、训练窗口和 C++ 状态契约未稳定前提前替换现有流水线。

## 非目标

- 不把 ONNX Runtime 引入 selfplay 或 gatekeeper。
- 不把训练迁移到 C++；训练、优化器和 SWA 仍由 Python/PyTorch 负责。
- 不让 AI 执行竞拍，不增加竞拍 phase/action/policy head；离线补偿只产生开局 VP offset。
- 不把补偿评估变成第六个常驻进程，也不让运行中对局切换补偿版本。
- 不让 `.bin`、TensorRT engine 或临时导出文件成为训练恢复的唯一来源。
- 不在 C++ 版本通过完整回归前删除 Python 参考实现。
