# 强化学习研究 Agent 审计：任务需求、历史经验与架构缺口

> 审计日期：2026-08-14
>
> 本文是对 `rl_agent_architecture_proposal.md` 的补充审计，不覆盖或回写用户已有修改的
> `docs/rl_agent_architecture.md`。它回答一个具体问题：**Taili 强化学习任务在长期会话中已经暴露出的
> 需求和经验，哪些还没有被 Agent 设计表达为可验证、可执行的能力。**

## 1. 审计范围与方法

### 1.1 数据范围

本轮回看了 Codex 本地会话目录下的 90 个 JSONL 会话，总计约 1.57 GB。按内容筛选出的强化学习相关覆盖为：

| 主题 | 命中会话数 | 用途 |
|---|---:|---|
| Taili / 盲态 locomotion | 57 | 任务目标、训练路线、奖励与课程 |
| 诊断 / 物理回放 | 52 | 评价口径、视觉与物理证据 |
| MuJoCo | 29 | sim2sim、控制器和接触差异 |
| sim2sim | 35 | 部署契约和迁移故障 |
| resume | 75 | 谱系、优化器状态和非单调训练 |
| fresh | 65 | 可复现性、冷启动和奖励可达性 |

会话筛选同时排除了相机云台、激光云台和普通 Agent 设计会话。历史内容中出现过凭据，审计只记录
故障类别和结论，不复制任何凭据、主机口令或私密连接信息。

### 1.2 证据优先级

会话叙述不是唯一事实源。审计采用以下优先级：

1. 真实运行的 `EffectiveRunSnapshot`、原始诊断轨迹和部署输出；
2. 训练 payload、实际 import/override 链和远端环境版本；
3. 机器验收 scorer、配置与测试；
4. 已封存的交接记录和专题调查；
5. 会话中的人类观察、假设和当时的临时判断。

因此，下面的“已验证”只表示有足够证据支持其在指定范围内成立；不把一次成功的 resume
或一个 checkpoint 自动升级为通用规则。

### 1.3 交叉来源

- 任务契约：[taili_spec.md](../../taili_spec.md)
- 奖励和课程理由：[taili_strategy_decisions.md](../../taili_strategy_decisions.md)
- 历史经验：[taili_session_lessons.md](../taili/taili_session_lessons.md)
- 长期调参记录：[taili_tuning_followups.md](../taili/taili_tuning_followups.md)
- 当前边界：[taili_env_boundary.md](../../taili_env_boundary.md)
- 遥测契约：[taili_telemetry_contract.md](../../taili_telemetry_contract.md)
- 迁移调查：[taili_mujoco_investigation_20260803.md](../taili/taili_mujoco_investigation_20260803.md)
- 当前运行交接：[taili_live_handoff.md](../../taili_live_handoff.md)
- 运营故障：[taili_ops_runbook.md](../../taili_ops_runbook.md)
- 被审计设计：[rl_agent_architecture_proposal.md](./rl_agent_architecture_proposal.md)

CodeGraph 当前索引 255 个文件、4521 个节点和 6269 条边；结构核对还暴露出历史策略备份与当前
`autotuner` 并存，这本身就是“实际生效源码必须证明”的证据，而不是可以忽略的目录噪声。

## 2. 先固定任务本身

### 2.1 Taili 的基本盘不是一个总分

最终目标是一个多目标物理控制策略，至少包含四个不可互相掩盖的维度：

| 维度 | 必须达到的行为 |
|---|---|
| 名义平地 | 零命令由运动自然、快速且不过冲地进入严格安静站立；前进、后退、横移、yaw 都跟踪机体坐标系命令；核心、机身高度、步态、支撑、髋/小腿几何、足端轨迹、duty、对称、滑移、触地和动作平滑同时合格。 |
| 名义楼梯 | 以前进为主要方向分别完成上楼和下楼的真实整段换层；允许合理俯仰和适度降速，但不能用原地踏步、绕行、自由落体或课程记账冒充通过；接触以足端为主，碰撞、滑移、冲击、偏航、髋部大幅动作和高频/高幅振荡受控。 |
| 域随机化 | 在逐渐扩大的质量、摩擦、质心、执行器、传感器、延迟和推扰组合下，保护上述能力；DR 是鲁棒性条件，不是替代名义运动质量的奖励。平地和楼梯的 DR 强度可以不同，楼梯优先保护质量。 |
| 迁移与安全 | IsaacLab、MuJoCo 和实际控制器使用同一观测、动作、历史、频率、PD、力矩和信息可得性语义；在迁移前能区分部署 bug、物理差异和策略缺陷。 |

### 2.2 需求中的两个重要口径冲突

审计发现需求记录与当前契约还有两处必须由人类确认的差异：

1. 近期讨论把负重/质量 DR 的进度描述为 `10 kg -> 20 kg -> 35 kg -> 50 kg`，而
   `taili_spec.md` 的 E5 写的是“added mass `-5 ... +20 kg`”。这两者可能分别表示总负重、附加质量或质量扰动范围，
   不能当作同一个数值。Agent 必须在 ContractBundle 中明确 `base_mass`、`added_payload_mass`、`total_mass`、
   CoM 变化和施加位置，否则所谓“50 kg 级别”不可验收。
2. `taili_spec.md` 曾保留 slope、rough、boxes 的历史电池，而当前阶段明确楼梯和平地优先、多方向地形靠后。
   这不是删除历史电池，而是要在契约中标注 `required_now`、`required_later` 和 `report_only`，避免阶段性目标再次漂移。

### 2.3 楼梯的最小语义

历史已经证明人工楼梯状态机、固定腿序、固定落脚点和势能式复杂链条会把问题变成不可达或错误的局部解。
最小但完整的楼梯语义应表达：

1. 沿机体坐标系命令方向产生真实沿程和净换层；
2. 只有为通过障碍所必需的抬脚/前送/受控探低才获得信用，不为“抬腿本身”给无条件高分；
3. 足端接触允许但代价可测，非足端严重碰撞、打滑、自由落体和失去支撑不能成为成功捷径；
4. 通过期间维持节律、支撑、可接受的姿态和高度，允许楼梯需要的 pitch，但禁止无关的大髋偏角、侧漂和剧烈振荡；
5. 上楼、下楼分别计分，完整轨迹和真实终点才是通过证据，课程等级只是训练分布状态。

这四/五层不是新的状态机；它们是任务信用、质量约束和验收条件的不同职责。Agent 必须检查每层是否
有正向可达路径、是否重复、是否相互抹掉梯度，而不能仅检查名字存在。

## 3. 历史经验总账

### 3.1 已经明确且应保留的经验

| 经验 | 反复出现的证据 | 对 Agent 的约束 |
|---|---|---|
| 奖励是主驱动，惩罚是质量/防作弊补充 | 楼梯只加门控长期不涨；过重站立惩罚造成急停或运动退化 | 每个能力必须有可追溯的正向信用路径，不能只报告 penalty 数值 |
| 奖励语义过宽会被钻，过窄会让 fresh 不可探索 | 横向/碰撞/原地抬腿曾可拿分；窄核使策略起不来 | 必须做可达性、反例和梯度审计，而非只做静态公式检查 |
| 日志、诊断、视频、配置各回答不同问题 | 课程 frontier 假涨、日志合格但视频失败、探针误判滑移 | 证据类型、时间窗、样本资格和结论等级必须分开 |
| resume 链产生能力但不等于可 fresh 复现 | 楼梯 7+ 来自多次 resume；新机制在成熟点上难以吸收 | 保存每条 ResumeEdge 的真实状态、暴露分布和能力变化，支持因果回放 |
| 学习非单调，最新点不是最佳点 | 约 18k 或某些中间点优于后续点，继续跑会退化 | 需要能力/Pareto checkpoint registry、晋升规则、早停和回滚 |
| 课程平台可能是门控/统计错误 | diagonal 被 yaw 污染，air-time 不符合步态，terrain mean 被困难子集压住 | gate 必须有已知可行基线校准、有效样本定义和可达性报告 |
| 命令覆盖决定能力是否有梯度 | mixed sampler 缺少纯 yaw；后退/横移样本逐渐消失 | 记录目标、实际、有效、终止样本，而不是重采样事件数 |
| 探索噪声影响严格站立是否数学可达 | global log_std 使 rollout 中持续运动；mean-action 与训练噪声不等价 | 训练随机性、部署确定性和静止验收必须分开评估 |
| PPO 的全局优化会让局部分支互相伤害 | 静止奖励改变通过 advantage 标准化影响 yaw/移动 | 需要分支 advantage/KL/参数组或至少记录其相互影响，不能假定权重缩放有效 |
| Adam 会抵消简单的梯度缩放 | 只改 reward scale 未必改变更新方向/步幅 | 实验必须记录 advantage、梯度、KL、clip fraction 和实际参数更新 |
| resume 状态远不止 actor 权重 | scheduler、normalizer、log_std、AMP、课程和 RNG 丢失会改变路线 | ResumeEdge 必须逐项证明恢复/重置状态及兼容性 |
| 物理量的定义会改变结论 | 球足滚动曾被算作滑移；touchdown 与 settled stance 混淆；阈值低于物理可达下限 | 每个训练项和验收项必须绑定同一物理量、事件窗口和标定参考 |
| 乘法质量核可能抹掉所有梯度 | 支撑、滑移、核心、触地相乘后任一短板接近零，其他能力无法恢复 | 质量门必须报告梯度保留/饱和情况，失败区不能整体归零 |
| collapse gate 也可能抹掉恢复梯度 | 过度 gate 会让策略不能学回高度和姿态 | 严重状态需有独立、连续、可恢复的纠偏信用；不能把惩罚全关掉 |
| terminal 与 timeout 语义不同 | 惩罚 timeout 会教策略主动结束 episode | 两者必须在训练、日志和验收中独立统计 |
| 配置“改了”不等于运行“用了” | 未 import 文件、子类覆盖父类、YAML/dataclass/payload 多来源、重复 acceptance 模块 | 启动前要生成字段到最终调用函数的执行证明 |
| MuJoCo 差异先查契约再归因策略 | gait clock、PD、timestep、碰撞半径和控制路径曾改变结论 | sim2sim 必须有固定 triage 顺序和单变量 parity 测试 |
| AMP 资产和实现可能过期/误读 | 子类实际 command-conditioned；磁盘 npz 仍是旧站姿；AMP 不能主导楼梯 | 记录实际 discriminator 输入、参考资产 hash/统计和权重影响 |
| 训练和诊断有真实资源约束 | GPU kernel hang、OOM、SSH 断线、诊断白屏、回放帧过多 | 故障签名、资源预算、artifact 流式化和互斥调度必须可执行 |

### 3.2 还不能被当作通用知识的内容

以下结论仍是 Taili/特定 payload 的证据，不能直接写成所有机器的规则：

- 某个 Kp/Kd 值、周期、duty、碰撞半径或楼梯高度；
- 历史 7+ resume 链中某一次干预的单独因果贡献；
- 某种宽核、严格惩罚、AMP 权重或地形课程对 fresh 的普遍效果；
- MuJoCo 中某个失败是否完全来自控制器、接触求解器还是策略；
- “训练到某个轮数自然会变好”的等待经验。

Agent 应把这些保存为带范围的 `tentative` claim，并要求新的运行满足预测后才能升级。

## 4. 覆盖矩阵：现有提案还缺什么

状态含义：

- **完整**：已有明确对象、流程和机器/实验验收方式；
- **部分**：提案提到了主题，但不能证明真实运行或不能指导下一步实验；
- **缺失**：没有一等对象或强制流程。

### 4.1 必须进入核心架构的 RL 动力学缺口

| ID | 缺口 | 证据来源 | 提案现状 | 状态 / 优先级 | 必须补成什么 |
|---|---|---|---|---|---|
| R1 | 训练信用与验收物理量不是同一量 | `taili_tuning_followups.md` 的滑移、B1/B3 重标定；`taili_spec.md §1` | 有 RewardMechanismGraph 和 EvidenceRecord | 部分 / P0 | `MetricAlignmentContract`：单位、参考系、事件、窗口、资格、统计量、阈值和训练项一一绑定 |
| R2 | 统计口径改变结论 | percentile、mean、p95、有效接触、完整楼梯轨迹的历史争议 | EvidenceRecord 只有 `extracted_facts` | 部分 / P0 | 保存原始样本资格 mask、窗口边界、分位数算法、空样本处理和聚合层级 |
| R3 | reward 的严重失败区可能没有梯度 | `taili_mujoco_investigation_20260803.md` 的 clearance/质量乘积调查 | RewardMechanismGraph 记录 exploit，但不记录梯度 | 缺失 / P0 | 每项记录 activation、饱和率、导数/有限差分、优势相关性和失败区梯度 |
| R4 | 乘法质量核、hard gate 与恢复路径冲突 | 质量核同时压低支撑、滑移、核心、冲击；collapse gate 关闭恢复成本 | P3 只规定奖励/惩罚职责 | 缺失 / P0 | `GateGradientAudit` 和“恢复状态仍保留的正向路径”检查；禁止未证明的全局归零 |
| R5 | reward reduction 尺度漂移 | per-robot mean/fraction 与 raw sum 的历史修复 | proposal 没有 reduction schema | 缺失 / P0 | 对每个 term 固定 reduction、foot/joint/env 归一化和 batch 聚合；改变维度时拒绝启动 |
| R6 | terminal 与 timeout 混淆 | `taili_strategy_decisions.md #4` | 没有专门对象 | 缺失 / P0 | 训练事件、回报、课程和验收分别记录 terminal reason；timeout 不可伪装成失败/成功 |
| R7 | 探索噪声使目标不可达 | global `log_std`、mean-action 部署和 quiet stand 的历史 | PolicyDeploymentContract 只写部署口径 | 部分 / P0 | `PolicyStochasticityProfile`：log_std、噪声注入位置、entropy、mean/noisy 双评估和 regime 切换 |
| R8 | PPO advantage/optimizer 的跨目标干扰 | 静止样本改变全批 advantage，yaw/移动退化；Adam 抵消缩放 | ExperimentPlan 有 seed，但无优化器动力学 | 缺失 / P0 | 记录分支 advantage、KL、clip fraction、梯度范数、参数组、LR 和分支贡献；必要时支持分支中心化/参数组 |
| R9 | resume 只恢复权重而改变优化路径 | scheduler、normalizer、RNG、AMP、gate、课程状态丢失 | ResumeEdge 有布尔字段 | 部分 / P0 | 把 policy/value/optimizer/scheduler/normalizer/log_std/AMP/RNG/rollout/gate 全部版本化并做恢复后 parity |
| R10 | 时间信用与延迟未建模 | touchdown 后窗口、刹车长尾、gait clock、控制延迟 | RewardMechanismGraph 没有时序信用对象 | 缺失 / P0 | `TemporalCreditSpec`：事件触发、延迟、信用窗口、相位/历史状态和 bootstrap 规则 |
| R11 | fresh 阶段奖励可能不可探索 | 过窄奖励/严格惩罚导致策略起不来，宽核又可能停在中间 | P3 只是原则 | 部分 / P0 | 训练前做 cold-start reachability smoke：随机/已知可行策略的 reward、梯度和恢复路径；失败阻断长训 |
| R12 | 物理上限与阈值未先校准 | 完美 IK/参考动作也曾因错误阈值失败；电机弱 thigh 约束 | PolicyDeploymentContract 有 limits | 部分 / P0 | `FeasibilityCalibration`：解析参考、无滑移参考、执行器可达域和阈值来源；不可达阈值禁止进入实验 |

### 4.2 课程、命令和 checkpoint 缺口

| ID | 缺口 | 证据来源 | 提案现状 | 状态 / 优先级 | 必须补成什么 |
|---|---|---|---|---|---|
| C1 | 课程 gate 可能不可达或被错误指标卡住 | diagonal 被 yaw 污染、air-time 定义错误、terrain mean 被困难子集压住 | TrainingDistributionManifest 记录实际分布，但没有 gate 校准 | 缺失 / P0 | `GateCalibrationRecord`：已知可行策略、有效样本、可达上下界、观测 ceiling、推进/回退不变量 |
| C2 | phase 只有编号，没有训练目的 | phase0 被 flatcore/stand 卡住；phase1 意义反复质疑 | proposal 的 lifecycle 不是 curriculum contract | 部分 / P0 | 每个 phase 的目的、必须形成的能力、允许退化、进入/退出条件和不可破坏不变量 |
| C3 | 目标采样与有效样本覆盖不一致 | 纯 yaw 缺失；后退/横移逐渐消失；重采样数不等于有效 rollout | Manifest 只有 bucket counts | 部分 / P0 | `CommandCoverageLedger` 按方向、强度、过渡、稳定/移动、terminal/timeout 和有效资格计数 |
| C4 | 课程难度和 acceptance 混用 | frontier/mean/course level 被误作通过能力 | proposal 文字上区分，但无类型约束 | 部分 / P0 | `CurriculumState`、`CapabilityEvidence`、`AcceptanceResult` 三种不可互换类型；frontier 永不直接验收 |
| C5 | 学习非单调，newest 不是 best | 中间 checkpoint 峰值、继续训练后退化、resume 短暂上涨 | BaselineSet 支持多个锚点 | 部分 / P0 | `CheckpointCapabilityRegistry`：last/best/Pareto/counterexample、晋升与早停规则、能力族回归检测 |
| C6 | resume 链的干预与暴露因果不清 | 7+ 由多次 resume、replay/碰撞信用/高难度训练累积 | ResumeEdge 只描述恢复状态 | 部分 / P0 | `ResumeEdge` 增加 config diff、实际暴露窗口、首个生效 step、能力 delta、保护能力 delta 和反事实说明 |
| C7 | 多目标冲突没有明确决策结构 | 平地、楼梯、站立、DR 互相交换；总分掩盖短板 | BaselineSet 但无 Pareto/词典序策略 | 部分 / P0 | Contract 规定 hard invariants、受保护能力、允许退化预算和 Pareto 晋升，不允许单一总分裁决 |
| C8 | 训练/验收分布污染与过拟合未审计 | 固定诊断楼梯、历史 replay 和课程暴露可能重复使用 | ExperimentPlan 无 holdout 字段 | 缺失 / P1 | 训练/诊断/保留测试场景分层；记录 seed、几何、命令时间线和是否见过同一 case |
| C9 | seed 不确定性未量化 | 单个 run 常被当成成功/失败，resume 路线高度依赖历史状态 | ExperimentPlan 有 seed_strategy | 部分 / P1 | 最小重复种子、置信区间/效应量、失败率和“不可区分”结论；区分 checkpoint variance 与 policy variance |

### 4.3 Taili 任务语义仍需实例化为契约

| ID | 实例缺口 | 历史/需求 | 提案现状 | 状态 / 优先级 | 应明确的契约 |
|---|---|---|---|---|---|
| T1 | 平地“全面质量”仍可能退化成几个分数 | 用户反复强调髋、核心、高度、足轨迹、脚重、duty、刹车和跟踪共同满足 | §8.1 只列三条主线 | 部分 / P0 | 平地场景电池：四方向、混合命令、恒定命令、命令切换、长时窗口；每项绑定物理量和保护关系 |
| T2 | 从运动切零的站立过渡缺少独立场景契约 | 站立有时成功、有时持续摇晃；刹车过急；部署 gait clock 曾冻结 | §8.2 只写站立目标 | 部分 / P0 | 明确命令时间线、初始 gait phase、停止响应窗口、最大冲击/高度变化、settled tail 和四脚支撑 |
| T3 | 核心稳定与髋/足端视觉质量没有同一时间窗 | 平地高度起伏、髋外翻/内收、足轨迹不自然 | RewardGraph 泛化描述 | 缺失 / P0 | `FlatQualityProfile`：roll/pitch/yaw、高度、角速度/角加速度、髋偏角、小腿平面、足端轨迹、duty、触地分别统计并联合验收 |
| T4 | 楼梯“通过”没有足够严格的完整轨迹定义 | MuJoCo 20 cm 上楼只走一小段；下楼滑移/振荡；课程等级误导 | §8.2 只有真实通过原则 | 部分 / P0 | 上/下楼专用场景契约：整段沿程、净换层、方向、终态、非终止、支撑、速度/姿态/冲击和碰撞事件 |
| T5 | 允许楼梯 pitch 与平地核心约束的边界未机器化 | 楼梯允许一定俯仰，不能因此放弃 roll、偏航、过度髋动和振荡 | §8.2 仅文字表达 | 部分 / P1 | 明确 terrain-specific relaxation：允许的 pitch 来源/范围，哪些 core invariants 永不放松 |
| T6 | 楼梯主驱动和质量约束职责没有实例图 | 历史证明方向、净空桥接、支撑信用有用；状态机/势能/逐腿路径失败 | RewardMechanismGraph 模板存在 | 部分 / P0 | 为上楼、下楼各建一张实际图，注明正向驱动、桥接、惩罚、资格、作弊假设和梯度健康；不以机制数量代替语义 |
| T7 | DR 的质量级别语义未统一 | 10→20→35→50 kg 与 spec 的 added mass 范围不一致；stress 要专门训练；综合 DR 不能只变负重 | §7.2 只有通用顺序 | 部分 / P0 | Taili `DRLevelContract`：总质量/附加质量/CoM/摩擦/PD/延迟/IMU/推扰/几何的范围、组合方式、平地/楼梯强度和升级验收 |
| T8 | AMP 资产与任务目标的冲突没有一等审计 | AMP 参考旧站姿；子类实际 command-conditioned；风格不应主导楼梯/静止 | §8.2 只写“有限先验” | 缺失 / P1 | `AMPAssetContract`：输入字段、命令条件、clip hash、姿态/高度/duty 统计、权重和任务奖励的冲突测试 |
| T9 | 盲态/特权监督的信息路径没有可证明审计 | `foot_h4` 等训练监督可能帮助 critic/latent，但 actor 不得见未来地形 | §3.4 和 §8.2 有 privileged 字段 | 部分 / P0 | `InformationPathAudit`：每个字段的 producer→tensor→actor/critic/aux→deploy 路径，防 latent/scaler 泄漏 |
| T10 | 控制周期、PD、碰撞几何和电机能力的物理边界未成为验收前置 | IsaacLab/MuJoCo 现象不同；timestep 0.0025、PD 硬编码、足端半径、thigh 110 N·m | PolicyDeploymentContract 有字段 | 部分 / P0 | 迁移前 parity vectors、每关节 torque-speed/limit 校准、碰撞几何 hash 和 timestep/decimation 证明 |
| T11 | gait clock、history、duty 的隐状态恢复未独立建模 | 零命令时 clock 冻结曾造成部署侧假失败；duty 跳变和历史顺序影响行为 | PolicyDeploymentContract 有 history 字段 | 部分 / P0 | `TemporalStateContract`：clock、history、phase、reset、命令保持、duty/period 定义及训练/部署更新顺序 |

### 4.4 运行时、诊断和运营缺口

| ID | 缺口 | 历史证据 | 提案现状 | 状态 / 优先级 | 应补内容 |
|---|---|---|---|---|---|
| I1 | 字段到实际执行函数的证明不足 | 未 import、父子类覆盖、YAML/dataclass/payload 多源；CodeGraph 还发现 backups 与当前源码并存 | EffectiveRunSnapshot 有 hash/override_trace | 部分 / P0 | `RuntimeExecutionProof`：字段→解析→import→override→最终函数/对象，带远端源码 hash 和启动时断言 |
| I2 | 重复模块/旧 payload 冲突没有启动阻断 | 两份 acceptance/reward/策略备份可能被不同 `sys.path` 选中 | Fact Store 可记录 payload | 部分 / P0 | 扫描重复模块、shadowed import、sys.path 顺序和生成物过期；冲突时拒绝训练 |
| I3 | sim2sim 排查顺序没有强制化 | 先后混淆 reward 与 gait clock、PD、timestep、接触求解器 | §7.1 有顺序概念 | 部分 / P0 | `DeploymentTriageRecord` 固定顺序：I/O→命令活动→clock/history/normalizer→频率/PD/力矩→接触/solver→策略归因 |
| I4 | 诊断 reset、场景和完整轨迹不具备机器契约 | wrapper 只 reset 一次；stairs_up 误测平台；只看前 1/3 误判通过 | EvidenceRecord/HMI 泛化 | 缺失 / P0 | `DiagnosticScenarioContract`：场景几何、方向、reset count、case timeline、完整轨迹、终止/换层判定和版本 |
| I5 | 诊断 artifact 与系统页面没有统一注册 | 外部生成目录不显示；newest 不一定 best；900 帧限制和白屏/跳动 | §4.9 只列 HMI 展示 | 缺失 / P1 | `ArtifactRegistry`：job/run/checkpoint/scenario 关联、索引、分段/流式回放、缩略图与原始数据分离 |
| I6 | 训练/诊断/验证资源隔离还不够细 | 完整 Isaac 诊断与训练并行 OOM；小 CPU smoke 可并行；GPU hang 与 OOM 不同 | Operations 有资源租约 | 部分 / P0 | 资源等级、显存/RSS 预算、run-id 锁、PyTorch 互斥规则、smoke 白名单和故障签名自动分类 |
| I7 | 数据盘清理与 checkpoint 保留没有能力感知 | 用户多次要求保留有价值历史、不要误删 payload；最新点不一定最好 | Operations 有删除审计 | 部分 / P1 | retention policy 按 lineage 引用、能力锚点、反例、可重建性和当前活动租约保留；删除前 dry-run |
| I8 | 监控策略没有“等待/干预”的统计决策模型 | 用户要求有间隔监控、不要被 resume 短暂上涨误导，也不要无意义诊断 | EvidencePolicy 有表格 | 部分 / P1 | `MonitoringDecisionPolicy`：窗口、最小样本、趋势置信度、resume warm-up、诊断成本和干预阈值 |
| I9 | 大信息源查询和增量记忆成本未定义 | 反复全文读取触发上下文压缩、重复调查 | Store/Knowledge 有概念 | 缺失 / P1 | 按事实/关系/时间/适用域建索引；支持摘要带原始引用、增量更新、查询预算和 freshness 标记 |
| I10 | 故障自愈边界不清 | GPU kernel hang、OOM、SSH 断线、后端退出的处置不同 | JobLifecycle/Operations 有基础状态 | 部分 / P1 | `FailureSignature` 与恢复策略：可自动重连/重试/降规模/封存，哪些必须人工确认；禁止把 hang 当普通失败重启 |

## 5. 当前提案已经覆盖的部分

不能因为存在上述缺口就否定提案。以下内容已经有相对完整的架构表达，应保留：

1. **目标、事实、判断、方案分离**：`ContractBundle`、`EvidenceRecord`、`HypothesisSet` 和
   `DecisionRecord` 的分层是正确的。
2. **历史谱系和多能力基线**：`ResumeEdge`、`BaselineSet` 已经比“只看 newest checkpoint”可靠，方向正确。
3. **证据成本分级**：`EvidencePolicy` 已吸收“日志坏不必先做昂贵诊断、日志好要用物理证据确认、迁移差异先查契约”的经验。
4. **实验保护和回滚**：`ExperimentPlan` 有 protected capabilities、允许退化、窗口和 rollback condition。
5. **训练、研究问题、后台 job 分离**：避免把“训练是否运行”和“是否需要调查”混成一个大状态机。
6. **部署盲态和权限/安全**：Actor 特权边界、授权租约、凭据不进日志、删除默认可逆等原则是必要的。
7. **知识可废止**：`validated/rejected/superseded` 防止一次历史成功变成永久圣经。

这些是治理骨架；缺口集中在“骨架如何知道 RL 数学路径真的可达、样本真的存在、代码真的生效、行为真的通过”。

## 6. 建议新增的一等对象

提案不应再增加一串没有所有权的机制。优先补充以下数据对象，令现有模块能执行审计。

### 6.1 `MetricAlignmentContract`

```yaml
metric_alignment:
  id: ...
  capability: flat_core | stairs_up | stand_transition
  physical_quantity: base_height_local | touchdown_vz | settled_foot_speed
  train_signal_ref: reward_component_or_input
  eval_signal_ref: scorer_field
  frame: body | world | terrain_local
  units: ...
  eligibility: contact_state_and_command_mask
  event_window: {start: ..., end: ...}
  statistic: {level: per_step | per_event | per_episode, aggregation: median|p90|mean}
  calibration_ref: analytic_or_physical_reference
  mismatch_policy: block | invalidate | investigate
```

### 6.2 `RewardAuditRecord` 与 `GateGradientAudit`

```yaml
reward_audit:
  term_id: ...
  owner_capability: ...
  sign: positive_drive | quality_penalty | safety_guard
  eligibility_rate: ...
  value_range: {min: ..., p50: ..., p95: ..., max: ...}
  saturation_rate: ...
  finite_difference_gradient: {early: ..., failure_band: ..., recovery_band: ...}
  advantage_correlation: ...
  conflicts: [...]
  exploit_tests: [...]
  cold_start_reachable: true|false|unknown

gate_gradient_audit:
  gate_id: ...
  closed_band: ...
  recovery_signal_preserved: true|false
  product_zero_rate: ...
  calibration_policy_ref: ...
```

它不要求 agent 仅凭数字断定“奖励正确”，但要求 agent 在调权重前知道：这个项是否激活、是否饱和、
是否在失败区留下梯度、是否被另一个项抵消。

### 6.3 `GateCalibrationRecord` 和 `CommandCoverageLedger`

```yaml
gate_calibration:
  gate_id: ...
  purpose: curriculum_progress | final_acceptance
  reference_policy_ref: known_feasible_policy
  distribution_ref: realized_command_terrain_distribution
  observed_range: {p05: ..., p50: ..., p95: ...}
  reachable_interval: ...
  ceiling_reason: ...
  enter_invariants: [...]
  exit_invariants: [...]

command_coverage:
  window: ...
  bucket: {direction, magnitude, yaw, transition, terrain, dr}
  target_count: ...
  applied_count: ...
  eligible_count: ...
  settled_count: ...
  terminal_count: ...
  timeout_count: ...
  effective_weight: ...
```

### 6.4 `OptimizationStateSnapshot`

```yaml
optimization_state:
  policy_hash: ...
  value_hash: ...
  optimizer: {type: Adam, state_hash: ..., param_groups: [...], lr: [...], betas: [...]}
  scheduler: {state_hash: ..., last_step: ...}
  normalizer: {state_hash: ..., update_mode: ...}
  stochasticity: {log_std_hash: ..., entropy: ..., noise_injection: ...}
  amp: {discriminator_hash: ..., optimizer_hash: ..., clips_hash: ...}
  curriculum: {phase: ..., gates: ..., terrain: ..., dr: ...}
  rollout: {rng_state_hash: ..., env_reset_state_hash: ..., sampler_state_hash: ...}
```

`ResumeEdge` 应引用它，而不是用一组容易误读的布尔值概括“恢复了 optimizer”。

### 6.5 `RuntimeExecutionProof` 和 `DiagnosticScenarioContract`

```yaml
runtime_execution_proof:
  run_id: ...
  field_sources: [{field, editable_source, generated_source, parser, final_object}]
  import_resolution: [{module, resolved_path, shadowed_candidates: [...]}]
  override_trace: [...]
  remote_source_hashes: [...]
  call_path_assertions: [...]
  status: proven | conflict | unknown

diagnostic_scenario:
  id: stairs_up_20cm_nominal
  geometry_hash: ...
  command_timeline: ...
  reset_protocol: {fresh_reset_each_case: true, verified_count: ...}
  full_trajectory_required: true
  success: {net_height, along_path, support, terminal, final_pose}
  artifacts: {raw_trace, frames, summary, registry_id}
```

## 7. 对现有架构的具体改动建议

### 7.1 核心架构层（P0）

在提案的 `Observation and Evidence Service` 与 `Evaluator` 之间增加一个明确的
**RL Dynamics Audit**，但它不是新的训练状态机，也不是另一个奖励系统。它只做四件事：

1. 对齐训练项、验收项和原始物理量；
2. 检查样本资格、课程可达性和梯度健康；
3. 检查 PPO 随机性、优化器/resume 状态和实际参数更新；
4. 在实验前阻断不可达阈值、无样本方向、全局零梯度和未证明的配置变更。

同时把 `ResumeEdge`、`TrainingDistributionManifest`、`RewardMechanismGraph` 扩展为上述对象的引用关系。

### 7.2 Taili 实例层（P0）

为 Taili 补充四个实例契约：

- `FlatQualityProfile`：把平地全局质量和命令切换单独列为场景，不允许只用 A1/A2 代表平地；
- `StairTraversalProfile`：上下楼完整轨迹、机体坐标方向、允许 pitch、非足端碰撞/滑移/冲击和质量尾部；
- `DRLevelContract`：先解决总质量/附加质量口径，再定义 10/20/35/50 kg 对应的全通道组合；
- `AMPAssetContract` 与 `InformationPathAudit`：证明风格资产和盲态信息路径没有与任务目标冲突或泄漏。

### 7.3 运行设施层（P0/P1）

训练启动必须经过以下 preflight，顺序固定：

1. 生成 `EffectiveRunSnapshot` 和 `RuntimeExecutionProof`；
2. 检查重复模块、shadowed import、远端源码和生成 artifact；
3. 检查 `PolicyDeploymentContract` parity；
4. 运行 reward/metric/gate 的廉价语义与梯度 smoke；
5. 校准课程 gate 和命令覆盖；
6. 确认资源租约、磁盘保留和诊断互斥；
7. 再启动或恢复训练。

诊断启动则必须先注册 `DiagnosticScenarioContract`，而不是把外部生成目录直接当作系统可见结果。

## 8. Agent 不应自动化的事情

这次审计也发现一些“看起来可以自动化、实际不应自动化”的边界：

- 不能根据单次视频自动定义“自然”“美观”或修改硬阈值；应记录人类观察并要求目标批准。
- 不能根据 frontier、总 reward 或 newest checkpoint 自动宣布楼梯通过。
- 不能把一次成功 resume 的配置直接推广到 fresh 或另一个机器人。
- 不能在 reward 梯度/样本覆盖/实际 import 未证实前自动烧长训练。
- 不能用增加机制数量代替对正向驱动和局部最优的解释。
- 不能为了让 scorer 通过而修改 scorer、删除失败场景或静默降低覆盖要求。

这些限制不是降低 Agent 能力，而是把“能执行”与“有权改变研究结论”分开。

## 9. 仍然开放的事实问题

本轮审计不能凭会话材料替代新的实证，以下问题必须在实现时保持 `open`：

1. 历史 7+ 楼梯 resume 链中每个改动的独立因果贡献尚未由完整对照证明；只能作为带范围的历史证据。
2. 当前工作区和远端训练机的最终运行快照是否完全一致，需要启动前重新生成，不能引用旧交接文档代替。
3. `10/20/35/50 kg` 的质量语义需要人类确认后才能编译成 E4/E5 契约。
4. AMP 资产、gait clock、PD 和碰撞几何在每个部署版本上的实际一致性需要 parity 运行，而不是文档声明。
5. 训练能否达到某个理想平地/楼梯质量，仍是实证问题；架构只能减少盲调和错误归因，不能保证策略必然收敛。

## 10. 最终判断

现有 Agent 提案已经足以成为一个“不会轻易失忆、不会随意删资产、能记录实验谱系”的研究监督骨架，
但还不足以成为可靠的 RL 调参者。缺口的核心不是再增加一个 LLM、一个探针或一个阶段，而是把下面这条
数学与事实链变成可检查对象：

```text
目标物理量
  -> 有效样本
  -> 正向信用与质量约束
  -> 保留梯度的 PPO 更新
  -> 完整优化器/课程/随机性状态
  -> 可复现的 checkpoint 能力
  -> 同口径物理验收
```

其中任一箭头没有证据，Agent 都只能说“可能在训练”，不能说“策略正在改善”。

**建议优先级：**先实现 R1–R12、C1–C7、T2/T4/T6/T7/T9/T10/T11、I1–I4；再实现统计重复、
artifact/HMI、故障自愈和大信息源索引。训练代码、奖励权重和当前运行在这些事实基础设施完成前不应被自动大规模重构。
