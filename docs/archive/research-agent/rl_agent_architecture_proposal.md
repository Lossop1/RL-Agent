# 机器人强化学习研究智能体：证据闭环架构

> **文档性质**：独立架构提案，用于与 `docs/rl_agent_architecture.md` 对照评审。
> 本文不覆盖原设计；实现状态必须按日期和测试证据标注，不能把远期能力写成既成事实。

> **2026-08-17 落地状态**：本文提出的最小“活系统”闭环已经进入生产代码，包括持久研究状态、
> 受限机制 AST、奖励/指标/门控合成与验证、Taili 运行时接入、隔离实验、独立验收、晋升/回滚、
> 结果学习、历史盲回放、Agent/API 接入、资源租约和受控 SSH backend。实际操作与权限边界见
> [RL 研究 Agent 活系统运维说明](../../rl_agent_live_system_operations.md)。文中更大范围的真机自治、
> 跨机器人迁移和长期元学习仍是后续能力，不应与已实现闭环混为一谈。
> 强化学习任务与历史经验的覆盖缺口见同目录的 [`rl_agent_rl_gap_audit_20260814.md`](./rl_agent_rl_gap_audit_20260814.md)；
> 在该审计中的 P0 缺口补齐前，本文的治理对象不能被视为完整的 RL 调参能力。
>
> **核心判断**：系统的价值不是替 LLM 增加更多流程，而是让任务目标、运行事实、
> 强化学习机制、实验结果和部署行为形成可验证、可回滚、可积累的闭环。

---

## 1. 系统要解决什么

### 1.1 目标

构建一个机器人强化学习研究智能体。它接收人类定义的任务目标，在明确授权范围内完成：

1. 把自然语言目标转成版本化验收契约。
2. 建立可复现的训练和部署环境。
3. 运行训练并持续监控真实进展。
4. 从日志、物理诊断、视觉观察、配置和谱系中定位问题。
5. 解释策略为什么获得当前行为，而不只是描述指标。
6. 提出最小且有因果依据的优化实验。
7. 保护已有能力，验证结果，失败时回滚。
8. 把得到验证的知识积累下来，并允许旧结论被废止。
9. 完成 sim2sim、sim2real 口径核对和最终交付。

最终交付物不是单独的 checkpoint，而是：

- 策略和部署包；
- 版本化验收结果；
- 完整训练谱系；
- 可复现运行快照；
- 已验证能力与已知缺陷；
- sim2sim/sim2real 契约与核对报告。

### 1.2 非目标

- 不代替人类决定最终目标和可接受的行为取舍。
- 不宣称仅靠静态 lint 就能证明奖励没有局部最优或作弊路径。
- 不假设 LLM 的第一次归因正确。
- 不把课程等级、聚合 reward 或单次视频当作真实能力。
- 不用复杂流程替代强化学习机制分析。
- 不要求所有任务使用同一种奖励、课程、状态机禁令或训练路线。
- 不直接执行未经授权的真机高风险操作。

### 1.3 能力边界

系统可以通过外部记忆、工具、持续实验和证据积累超过单次 LLM 会话的可靠性，
但不会自动超过底层模型的研究判断上限。系统必须通过实验结果校正自身，才能形成超出单次会话的专用能力。

---

## 2. 核心原则

### P1：人工批准的契约是目标权威

目标不能由 agent 自己修改。人类批准的 `ContractBundle` 是唯一目标源。
说明文档、scorer、测试和 HMI 展示均由该契约生成或绑定到同一版本。
运行期间发现 scorer 错误时，结果标记为无效，不能通过修改 scorer 让策略通过。

### P2：事实、判断和目标严格分离

- **目标**：人类批准的任务要求。
- **事实**：带来源、时间、运行和配置哈希的测量结果。
- **判断**：可被反证的解释和假设。
- **方案**：尚未执行的实验提案。

任何报告必须保留这四类边界。

### P3：奖励负责驱动，惩罚负责质量与防作弊

每个任务能力必须能追溯到明确的正向信用路径。惩罚可以排除危险、低质量或作弊行为，
但不能成为策略唯一的运动原因。例外是严格禁止条件，例如碰撞、越界或零命令下的持续运动；
即便如此，也必须确认任务的正向路径仍然可达。

### P4：证据成本与决策风险匹配

不是每次都进行完整诊断或严格 A/B。明显失败可以直接由日志或实现事实证明；
高风险、跨模块或原因不确定的修改必须使用更强证据。跳过某项证据时必须记录理由。

### P5：优化先保护已验证能力

优化的默认含义是保留有效部分，调整不合理部分，补充缺失部分。
每项实验必须声明保护能力、允许的暂时退化、观察期限和回滚条件。

### P6：训练从第一天就服从部署契约

观测布局、历史顺序、归一化、动作缩放、关节顺序、控制频率、PD、仿真 timestep 和部署可用信息
必须在训练开始前冻结到版本化契约。sim2sim 契约测试不是最终阶段才做的工作。

### P7：历史是证据，不是权威

历史 checkpoint、payload 和结论都必须携带适用范围。历史能力用于提供基线、机制线索和反例，
不能因为曾经成功就直接照搬，也不能因为当前路线不同就丢弃。

### P8：不把研究判断伪装成机器保证

字段可达性、哈希一致性和数值边界可以机器检查；奖励是否形成正确数学路径、动作是否自然、
原因是否查明属于实证或判断问题。系统必须明确检查类型和置信度。

### P9：奥卡姆剃刀约束机制数量

新增机制前必须说明已有机制为什么不足、增量语义是什么、可能怎样被钻空子。
能够通过修正现有语义解决的问题，不新增状态机、门控链或局部补丁。

### P10：权限与能力分离

agent 能生成代码或执行远程命令，不代表它有权修改目标、删除资产或操作真机。
权限按对象、操作、环境、有效期和风险等级授予。

---

## 3. 核心数据对象

所有对象都有稳定 ID、schema 版本、创建时间、内容哈希和来源。运行记录采用 append-only 事件日志，
派生状态可以重建。

### 3.1 ResearchProgram

研究实例不只由“机器人、任务、世界”组成，而由完整控制系统定义：

```yaml
research_program:
  id: taili_blind_locomotion
  robot_model_ref: robot-model/taili@hash
  control_stack_ref: control/taili-position-pd@version
  task_contract_ref: contract/taili@version
  training_world_ref: world/isaaclab@version
  deployment_worlds:
    - world/mujoco@version
    - world/real-taili@version
  budget:
    gpu_hours: ...
    wall_clock_hours: ...
    storage_gb: ...
```

机器人几何相同但控制栈、观测契约或部署世界不同，应被视为显式变更，而不是同一实例的隐藏差异。

### 3.2 ContractBundle

```yaml
contract_bundle:
  id: contract/taili@2026-08-...
  status: draft | approved | retired
  approved_by: human
  goals: [...]
  qualitative_intent: [...]
  quantitative_gates: [...]
  required_scenarios: [...]
  protected_capabilities: [...]
  deployment_constraints: [...]
  generated_artifacts:
    compiler_hash: ...
    scorer_hash: ...
    spec_hash: ...
    contract_test_hash: ...
```

规则：

- agent 可以提议修订，不能批准修订。
- approved 版本不可原地修改，只能产生新版本。
- scorer、说明文档和契约测试必须绑定同一 bundle。
- 契约编译器本身也必须版本化，并用固定 golden vectors 验证生成结果。
- 某次 run 永远引用启动时的契约版本，不能被后续编辑静默改变。

### 3.3 EffectiveRunSnapshot

这是“现在到底训练了什么”的唯一事实快照：

```yaml
effective_run_snapshot:
  run_id: ...
  source:
    git_commit: ...
    dirty_patch_hash: ...
  config:
    editable_source_hash: ...
    resolved_config_hash: ...
    override_trace: [...]
  environment:
    os_image: ...
    python_lock_hash: ...
    cuda: ...
    simulator: ...
  assets:
    urdf_hash: ...
    mjcf_hash: ...
    motion_data_hash: ...
  policy_contract_ref: ...
  distribution_manifest_ref: ...
  launch_command_redacted: ...
  seeds: {...}
```

配置系统必须输出每个关键字段的来源和最终生效值；存在多来源冲突时拒绝启动。

### 3.4 PolicyDeploymentContract

```yaml
policy_deployment_contract:
  observation:
    fields: [...]
    dimension: ...
    ordering: ...
    history_length: ...
    history_stride: ...
    history_order: oldest_first | newest_first
    scaling: {...}
    clipping: {...}
  action:
    joint_order: [...]
    dimension: ...
    scale: ...
    offset: ...
  controller:
    policy_hz: ...
    physics_hz: ...
    decimation: ...
    kp: [...]
    kd: [...]
    torque_limits: [...]
  privileged_inputs:
    actor: []
    critic: [...]
    training_supervision: [...]
```

训练、导出、IsaacLab、MuJoCo 和真机适配器必须通过同一个契约测试。

### 3.5 TrainingDistributionManifest

配置目标分布和实际采样分布必须分开记录：

```yaml
training_distribution:
  command_buckets: {...}
  terrain_mix: {...}
  curriculum_state: {...}
  dr_channels: {...}
  reset_sampling: {...}
  realized_window:
    steps: ...
    bucket_counts: {...}
    terrain_counts: {...}
    dr_quantiles: {...}
```

没有实际采样统计，就不能声称某方向、某楼梯高度或某 DR 强度已经训练过。

### 3.6 CapabilityProfile 与 BaselineSet

单个“最佳 checkpoint”不足以表示多目标能力。系统保存一组能力锚点：

```yaml
baseline_set:
  anchors:
    - checkpoint_ref: ...
      capabilities: {flat: strong, stairs: weak, stand: strong, dr: none}
      evidence_refs: [...]
    - checkpoint_ref: ...
      capabilities: {flat: strong, stairs: strong, stand: failed, dr: partial}
      evidence_refs: [...]
```

新实验选择父检查点时，必须说明使用哪个能力锚点以及准备补足什么，而不是只看总分。

### 3.7 ResumeEdge

```yaml
resume_edge:
  parent_checkpoint_ref: ...
  child_run_ref: ...
  restored:
    policy: true
    value: true
    optimizer: true
    normalizer: true
    curriculum: false
    dr_state: false
    amp_state: true
    rng_state: false
  reset_reason: {...}
  compatibility_report_ref: ...
```

谱系必须能区分完整 resume、仅权重初始化、部分状态重置和结构迁移。

### 3.8 RewardMechanismGraph

这是 RL 研究判断的核心对象：

```yaml
mechanism_graph:
  capability: stairs_up_progress
  eligibility: stair_scene and forward_command
  positive_drives:
    - name: supported_command_progress
      formula_ref: ...
      intended_credit: whole_body_progress
  bridges:
    - name: contact_response_to_clearance
      intended_credit: obstacle_response_then_required_clearance
  guards:
    - name: non_foot_collision
      intended_effect: reject_unsafe_contact
  couplings:
    - target: flat_core_quality
      expected_effect: neutral
  exploit_hypotheses:
    - stationary_leg_lifting_can_score
  runtime_observables: [...]
```

每项 reward、penalty、gate 和 curriculum 条件必须有唯一语义所有者。图中还应记录：

- 激活资格和样本覆盖；
- 数值范围、饱和程度和实际贡献；
- 与其他项的协同或冲突；
- 已知作弊路径；
- 对验收能力的预期影响。

### 3.9 EvidenceRecord

```yaml
evidence:
  id: ...
  kind: telemetry | physical_diag | video_observation | source_audit | sim2sim | real_test
  run_ref: ...
  checkpoint_ref: ...
  scenario_ref: ...
  contract_ref: ...
  measured_at: ...
  raw_artifact_ref: ...
  extracted_facts: [...]
  observer: machine | human
  freshness: current | stale | historical
```

人类视觉判断必须落成结构化证据，至少包含场景、检查点、命令时间线和观察到的现象，
不能只保留“88 分”或一段脱离上下文的文字。

### 3.10 HypothesisSet

归因不是单选枚举，而是一组竞争假设：

```yaml
hypothesis_set:
  symptom_ref: ...
  candidates:
    - claim: terrain_drive_has_no_clearance_bridge
      confidence: 0.7
      supporting_evidence: [...]
      contradicting_evidence: [...]
      prediction: ...
      disambiguation: ...
  unknowns: [...]
```

常见原因维度至少包括：目标语义、奖励机制、权重尺度、资格/采样、课程、优化动力学、
观测动作契约、实现或配置未生效、仿真保真、测量错误、checkpoint 状态和物理上限。

### 3.11 ExperimentPlan

```yaml
experiment:
  problem_statement: ...
  baseline_ref: ...
  hypotheses_addressed: [...]
  intervention_diff: ...
  unchanged_fields: [...]
  protected_capabilities: [...]
  allowed_temporary_regression: {...}
  training_window: {...}
  evaluation_plan: {...}
  seed_strategy: ...
  success_condition: ...
  rollback_condition: ...
  resource_budget: ...
  authorization_ref: ...
```

窄小、低风险且原因明确的调整可以采用单路线验证；高风险或归因不确定的调整需要对照、重复种子或分支实验。

### 3.12 DecisionRecord 与 ExperimentOutcome

```yaml
decision:
  id: ...
  trigger: ...
  problem_statement: ...
  evidence_refs: [...]
  hypotheses_considered: [...]
  selected_experiment_ref: ...
  rejected_options:
    - {option: ..., reason: ...}
  check_results: [...]
  authorization_ref: ...
  status: proposed | approved | executing | evaluated | abandoned

experiment_outcome:
  experiment_ref: ...
  actual_diff_ref: ...
  exposure_reached: {...}
  capability_delta: {...}
  protected_capability_results: {...}
  prediction_results: [...]
  disposition: promote | continue | rollback | inconclusive
  knowledge_updates: [...]
```

结果必须允许 `inconclusive`。训练成本已经发生不代表必须强行得出成功或失败结论。

### 3.13 KnowledgeClaim 与 HandoffSnapshot

```yaml
knowledge_claim:
  statement: ...
  status: tentative | validated | rejected | superseded
  scope: universal | robot_family | program | config_range
  evidence_refs: [...]
  supersedes: [...]
  valid_from: ...
  last_reviewed: ...

handoff:
  sealed_at: ...
  active_runs: [...]
  active_jobs: [...]
  current_contract_ref: ...
  established_facts: [...]
  active_hypotheses: [...]
  pending_experiments: [...]
  intervention_conditions: [...]
```

handoff 是带原始引用的状态索引，不是新的事实来源。恢复后必须核对远程状态和 artifact 哈希。

---

## 4. 系统模块

### 4.1 Contract Authority

- 管理 ResearchProgram 和 ContractBundle。
- 接受 agent 的修订提案和人类批准。
- 从 approved bundle 生成 spec、scorer schema、测试向量和 HMI 字段。
- 保证运行绑定的契约版本不可漂移。

### 4.2 Fact and Provenance Store

- 保存事件日志、EffectiveRunSnapshot、payload、checkpoint 和 ResumeEdge。
- 生成字段来源图和完整 lineage DAG。
- 管理 BaselineSet，而非强制唯一基线。
- 检测 dirty source、过期生成物、重复配置源和不可兼容 resume。

### 4.3 Observation and Evidence Service

- 读取按命令、地形、DR 等级分桶的遥测。
- 统计样本数、趋势、平台、方差和 resume 暂态。
- 运行单环境物理诊断、视频回放和 sim2sim。
- 接收人类视觉观察并与原始诊断绑定。
- 根据 EvidencePolicy 决定需要什么证据，不机械执行完整 battery。

### 4.4 Research Reasoner

由 LLM 驱动，但只能基于已引用事实工作：

1. 将现象整理为问题陈述。
2. 读取完整有效配置和 RewardMechanismGraph。
3. 生成多个竞争假设。
4. 提出最便宜的区分证据或实验。
5. 生成带保护能力和回滚条件的 ExperimentPlan。
6. 解释结果并更新假设，而不是直接把结果写成通用规则。

Reasoner 可以迭代调用代码检索、配置检查和数据分析工具，不应被实现成一次无工具的 LLM 调用。

### 4.5 Experiment Supervisor

- 校验实验 diff、兼容性、资源和授权。
- 创建训练分支，不要求为了准备修改先停止当前有效训练。
- 维护当前 run、候选 run 和诊断任务的资源租约。
- 执行启动、暂停、恢复、终止和回滚。
- 在观察窗口结束后调用 evaluator，禁止用 resume 初期上涨提前宣布成功。
- 防止同一设备上意外并行多个高显存 PyTorch 任务。

### 4.6 Evaluator

- 在绑定的 ContractBundle 下运行分层 battery。
- 输出按能力族和场景分桶的结果，不只输出聚合分数。
- 检查保护能力、成功条件和回滚条件。
- 区分“训练趋势改善”“阶段通过”“最终验收”。
- 验收规则自身不由正在调参的 Reasoner 修改。

### 4.7 Knowledge and Recovery Service

- 保存 DecisionRecord、KnowledgeClaim、案例和 handoff。
- 支持冲突、废止和适用范围查询。
- 只有满足证据标准的结论才能从 tentative 升为 validated。
- 压缩、进程重启或模型更换后恢复当前研究状态。

### 4.8 Operations and Security

- 负责 SSH、payload、远程状态、磁盘和归档。
- 使用 SecretStore；密码、token 和私钥不得写入会话、日志、payload manifest 或知识库。
- 记录主机指纹、权限范围和操作审计。
- 删除前检查活动租约、谱系引用、重建能力和保留策略；默认归档。
- 所有命令必须幂等或带操作 ID，避免恢复后重复执行。

### 4.9 Human Interface

只展示决策所需信息：

- 当前目标和受保护能力；
- 当前运行、真实 payload 和 lineage；
- 分桶趋势及样本数；
- 最新诊断和视觉标注；
- 竞争假设、证据缺口和实验提案；
- 授权、回滚和目标修订入口。

HMI 不以曲线数量代替信息质量，也不让聚合分数遮蔽场景失败。

---

## 5. 三类独立状态

不能用一个大状态机同时表示训练进程、研究判断和后台任务。

### 5.1 RunLifecycle

```text
CREATED -> VALIDATED -> RUNNING -> STOPPING -> STOPPED
                              \-> FAILED
```

训练在 Reasoner 调查或准备新实验时可以继续运行。

### 5.2 ResearchCaseLifecycle

```text
OBSERVED -> ATTRIBUTING -> EVIDENCE_NEEDED -> PROPOSED
         -> EXPERIMENTING -> EVALUATING -> RESOLVED | OPEN
```

一个研究问题可以跨多个 run，多个问题也可以引用同一个 run。

### 5.3 JobLifecycle

```text
QUEUED -> LEASED -> RUNNING -> SUCCEEDED | FAILED | CANCELLED
```

训练、诊断、导出、sim2sim、清理和同步都是带资源需求的 job。调度器负责互斥、优先级和恢复。

---

## 6. 证据与干预策略

### 6.1 EvidencePolicy

| 情况 | 最小证据 | 默认动作 |
|---|---|---|
| 日志明显低劣且样本覆盖充分 | 趋势、分桶样本、有效配置 | 可跳过昂贵视频，先查机制/配置 |
| 日志良好但行为质量未知 | 单环境物理诊断 + 视频 | 不得仅凭日志验收 |
| 视频和指标矛盾 | 原始轨迹、指标定义、时间对齐 | 阻断调参，先修测量语义 |
| IsaacLab 与 MuJoCo 不同 | PolicyDeploymentContract parity | 先排除部署差异 |
| resume 后短暂上涨 | 完整观察窗口 | 不干预、不宣布成功 |
| 已知配置字段未生效 | SourceMap/运行快照 | 直接修实现，无需重复行为诊断 |
| 高风险奖励或架构修改 | 对照或分支实验 + 非回归 battery | 达标才晋升候选 |

### 6.2 干预前必须回答

1. 当前不满足哪项契约，观察到的事实是什么？
2. 原因是已知、推断还是未知？有哪些竞争假设？
3. 当前正向驱动是什么，策略怎样获得分数？
4. 现有机制哪里不足，新增语义是否重复？
5. 为什么选择这个父检查点和 fresh/resume 方式？
6. 哪些能力必须保护，允许多大、多久的暂时退化？
7. 如何判断修改成功、失败或没有生效？
8. 失败后如何回滚，当前训练是否需要继续保留？

### 6.3 检查类型

| 类型 | 示例 | 执行者 |
|---|---|---|
| 确定性检查 | schema、hash、字段可达、维度、权限、资源互斥 | 代码 |
| 静态语义检查 | reward 重复来源、部署特权依赖、明显无界公式 | 代码 + 规则 |
| 实证检查 | 局部最优、奖励作弊、课程有效性、sim2sim 一致性 | 实验 |
| 研究判断 | 原因排序、奥卡姆、动作自然性、实验价值 | LLM + 人类 |
| 目标裁定 | 阈值、优先级、可接受取舍 | 人类 |

系统不得把后三类显示成“lint 已证明”。

### 6.4 实验完成规则

实验只有在以下条件满足后才能关闭：

- 实际运行与提案 diff 一致；
- 训练分布达到计划覆盖；
- 观察窗口结束；
- 成功指标和保护能力已评测；
- 结果更新了假设置信度；
- checkpoint 被晋升、保留为反例或按策略归档；
- DecisionRecord 和 handoff 已更新。

---

## 7. sim2sim 与 DR

### 7.1 sim2sim 是持续契约检查

实施顺序：

1. SETUP 时对固定输入进行观测和动作 parity 测试。
2. 第一个可站立策略进行短时无地形迁移。
3. 每个主要能力里程碑进行对应场景迁移。
4. 最终验收再进行长时和完整场景迁移。

发现差异时先分为：部署契约差异、物理模型差异、策略真实脆弱性。三者不能混为同一调参原因。

### 7.2 DR 必须记录实际扰动

DR 路线由实例契约定义，但系统级顺序默认是：

1. 名义环境能力成立。
2. 单通道扰动验证敏感性。
3. 多通道组合训练。
4. 综合压力和边界测试。

每个阶段记录目标范围和实际采样范围。不能只改变“DR 等级”标签而不验证质量、摩擦、延迟等通道是否真正生效。

---

## 8. Taili 实例映射

本节是实例配置，不是所有机器人任务的通用禁令。

### 8.1 目标

Taili 的三个基本能力同属最终硬目标：

1. 非零命令切零后的自然制动和严格安静站立。
2. 前进、后退、横移、yaw 的完整平地质量。
3. 以前进为主、安全稳定且高质量的上下楼梯。

三者形成基本盘后推进全面 DR；多方向地形是较低优先级扩展。

### 8.2 实例机制约束

- Actor 只使用部署可获得的命令、本体观测和历史。
- AMP 只提供有限风格先验，不主导任务。
- 楼梯方向使用机体坐标系。
- 楼梯主驱动必须表达整机真实通过和连续推进。
- 抬脚信用应服务必要净空，不鼓励无条件高抬腿。
- 惩罚负责非足端碰撞、滑移、冲击、过大动作和不稳定。
- 不使用人工楼梯状态机、固定足序、固定落脚点或逐腿 trace。
- 课程等级只表示训练探索状态，不是最终通过证据。

### 8.3 Taili 的 BaselineSet

至少保留：

- 历史楼梯 `7+` 能力锚点：楼梯和平地较强，严格静止失败，DR 不完整。
- 当前平地/静止锚点：名义平地较强，楼梯真实迁移能力待核对。
- 失败但信息量高的 fresh/resume 分支：用于验证哪些机制依赖历史状态。

不能用单一总分覆盖这些能力差异。

### 8.4 当前启动条件

在让新 agent 自动调参前，先完成：

1. 人工重新批准 Taili ContractBundle，清理旧 spec 中的阶段性目标漂移。
2. 统一训练、导出、IsaacLab 和 MuJoCo 的 PolicyDeploymentContract。
3. 重建历史关键 checkpoint 的 ResumeEdge 和 CapabilityProfile。
4. 为站立、平地和楼梯建立首版 RewardMechanismGraph。
5. 用历史干预节点构造 agent 回放评测集。

---

## 9. 权限模型

### 9.1 默认权限

| 操作 | 默认权限 |
|---|---|
| 读取代码、日志、诊断、谱系 | 自动 |
| 生成假设和实验提案 | 自动 |
| 运行只读检查和低成本解析 | 自动 |
| 修改候选配置或代码分支 | 需已授予研究干预权 |
| 部署到训练机并启动普通训练 | 可由有范围和时限的授权覆盖 |
| 停止当前有效训练、覆盖 checkpoint | 单次确认或预批准规则 |
| 删除资产、修改目标/scorer、真机高风险操作 | 人工确认 |

### 9.2 授权租约

授权包含对象范围、环境、操作类别、资源上限、有效期和撤销条件。
“进入十分钟监控，必要时干预”可以授予限定范围的持续租约，避免每次普通部署都打断人类；
超出范围的架构改变、删除或真机操作仍需单次确认。

---

## 10. 实施路线

不先冻结全部接口，也不同时建设所有模块。所有 schema 版本化，通过一条纵向闭环逐步稳定。

### 阶段 0：修复事实基础

- 统一 Taili 契约、scorer 和说明文档。
- 建立 PolicyDeploymentContract parity 测试。
- 输出 EffectiveRunSnapshot 和配置 override trace。
- 清理凭据处理方式。

验收：任意当前 run 都能回答“实际运行了什么”，IsaacLab/MuJoCo 能证明 I/O 与控制契约一致。

### 阶段 1：只读研究台账

- 事件日志、lineage、ResumeEdge、BaselineSet。
- 分桶遥测和 EvidenceRecord。
- handoff seal/restore。
- 先由人类继续做最终调参判断。

验收：上下文清空后不重复历史调查，且不会把过期结果当当前事实。

### 阶段 2：影子决策

- 实现 RewardMechanismGraph、HypothesisSet、ExperimentPlan。
- Reasoner 只提出方案，不自动执行。
- 在历史会话关键节点和当前训练上比较其建议与实际后果。

验收：能识别配置未生效、无关修改、奖励驱动缺失、resume 暂态和部署契约差异。

### 阶段 3：受保护自动干预

- 允许小范围配置/奖励修改。
- 强制保护能力、成功条件、回滚条件和资源租约。
- 当前有效训练默认保留，候选失败自动回滚。

验收：连续若干干预没有未记录的能力退化，没有重复重启或错误覆盖资产。

### 阶段 4：知识闭环

- Experiment outcome 自动更新假设。
- 案例经过证据门槛进入 KnowledgeClaim。
- 支持 rejected/superseded 和适用范围。

验收：后续相似问题能够利用先前证据，同时不会照搬已废止结论。

### 阶段 5：迁移和扩展

- 第二机器人或第二任务实例。
- 全面 DR 与 sim2real 清单。
- 根据真实瓶颈再决定是否拆分多个 agent 进程或增加评审模型。

---

## 11. Agent 自身的验收

### 11.1 历史回放评测

从真实历史中选择带后续结果的决策节点，隐藏未来信息，测试 agent 是否能够：

- 恢复当时真实运行状态；
- 区分事实、推断和未知；
- 找到关键配置与机制；
- 避免已知无效干预；
- 给出可证伪实验和正确保护能力；
- 在看到结果后正确更新结论。

### 11.2 在线影子评测

在不控制训练的情况下运行完整监控和建议流程，对比人类最终决策。评价建议的证据完整性、
无关修改率、重复调查率、错误归因率和预计资源成本，而不是评价报告是否流畅。

### 11.3 受控执行评测

硬指标至少包括：

- 上下文或进程重启后状态恢复正确；
- payload 与实际运行配置一致；
- 未经授权操作为零；
- 活动训练和有价值资产误删为零；
- 部署契约漂移能够在训练前检出；
- 每项干预都有结果和回滚状态；
- 保护能力的意外退化能够在规定窗口内检出；
- 凭据泄漏为零；
- 相比人工历史路线，达到同等能力所需的无效训练时间下降。

“从不提出错误方案”不是合理指标。系统应允许假设失败，但必须让失败成本受控并产生知识。

---

## 12. 主要风险

1. **程序化麻木**：检查清单变成形式合规。缓解方式是保留原始证据、竞争假设和人类视觉裁定。
2. **错误知识积累**：一次成功被泛化。缓解方式是证据等级、适用范围和 superseded 机制。
3. **训练成本失控**：每个问题都做严格对照。缓解方式是风险分级 EvidencePolicy。
4. **过度自动化**：agent 为了完成目标牺牲已有能力。缓解方式是 protected capabilities 和回滚租约。
5. **指标替代目标**：scorer 通过但行为不自然。缓解方式是定量契约、物理诊断和版本化视觉证据共同验收。
6. **框架先于能力**：建设大量模块但不能改善一次真实调参。缓解方式是阶段 0 到阶段 3 的纵向实施路线。

---

## 13. 最终判断

这个系统首先应成为一个可靠的机器人 RL 研究记录者和实验监督者，其次才是自动调参者。
它真正超过单次 LLM 会话的路径不是增加代理数量或状态机复杂度，而是：

```text
准确目标
  + 完整运行事实
  + 明确奖励因果模型
  + 有保护和回滚的实验
  + 能被反证和废止的长期知识
  = 可持续提升的专用强化学习研究能力
```

当这条闭环在 Taili 的真实历史回放和在线训练中被证明有效后，再扩大自治范围和跨实例能力。
