# 端到端机器人强化学习智能体 —— 顶层设计

> **文档性质**：本文件是系统的顶层设计，作为实现、讨论与评审的共同基准，不是单方面权威；
> 实现中发现的设计问题以讨论结论为准，本文件随讨论持续修订。
>
> **系统代号**：`rl-agent`（端到端机器人强化学习智能体）。建议新建顶层包 `agent/`，与 `autotuner/` 并列；
> 通过适配器复用 `autotuner` 既有能力（payload/诊断/console）。是否并入 `autotuner` 属实现细节，可提案。
>
> **阅读顺序**：
> ① 本文件 → ② `docs/taili_spec.md`（验收契约范式）→ ③ `docs/taili_strategy_decisions.md`（机制契约范式）
> → ④ `docs/taili_live_handoff.md`（活状态范式）→ ⑤ `docs/archive/research-agent/rl_agent_design.md`（设计论证过程）
> → ⑥ `docs/archive/taili/taili_session_lessons.md`（历史素材，仅作背景，不作为实现依据）。
>
> **术语**：实例（Instance）= 三元组 (机器人, 任务, 仿真世界)；家族（Family）= 验收契约中的一组硬门槛；
> 口径（Orthodoxy）= 部署侧约束（盲、mean-action、无特权）；谱系（Lineage）= 检查点/resume 的 DAG；
> 层（Layer）= 知识条目的通用/实例标记；干预权 = 持续授予的调参授权，操作确认 = 一次性授权。

---

## 1. 目标与边界

### 1.1 系统目标
一个专用于机器人强化学习的端到端智能体：接收任务需求，走完
**契约化 → 系统建立 → 训练 → 监控 → 诊断 → 调参 → 验收 → sim2sim/sim2real → 交付**，
交付物 = 策略 + 完整谱系 + 验收报告。人类在环的角色固定为：目标裁定、授权、视觉判断、
理想状态描述、纠偏（§7）。

### 1.2 非目标（边界）
- 不是通用 agent 框架：流水线、设施、知识库均为机器人 RL 专用。
- 不做通用机器人控制算法研究：奖励/门控/课程的形式与规则由知识库约束，数值由实例导出。
- 不替代真机部署工程：sim2real 只到"部署口径验证与核对清单"，真机操作人执行。
- 通用性只通过迁移协议（§5.4）获得，不通过抽象层堆砌。

### 1.3 设计原则
- **P1 单一事实源**：机器契约赢过文档（冲突时 scorer 是真相）；YAML 是唯一可编辑源；生成物不可手改。
- **P2 确定性骨架 + LLM 决策站**：流程、协议、护栏是确定性代码；LLM 只在规定决策站被调用，输入输出均为结构化 schema。
- **P3 决策-执行分离**：改变训练状态或破坏资产的提案，必须通过全局检查清单（§5.3）并完成必要授权后才可执行。
- **P4 压缩安全**：所有判断先落盘（handoff / 决策日志）再继续；恢复后先读状态、核对远程，不凭记忆。
- **P5 实例-通用分层**：一切知识与数值带层标记（通用 / 实例）；跨实例引用必须经过迁移分类。
- **P6 双证据通道**：日志（趋势）与物理诊断（真实）矛盾时，先修语义，再谈调参。
- **P7 认知开放**：agent 必须能知道自己不知道——惊讶触发中断、未知显式声明、提问是合法输出。
- **P8 研究-工程分层**：研究把未知变成可知，工程把可知冻结为契约；研究判断不能被规则替代，但必须被规则支撑；知识库必须能生长（允许长出设计者未写过的规则）。
- **P9 能力-权限分离**：能做 ≠ 允许做；可写面矩阵与自评禁令约束权限（§6）。
- **P10 不可逆默认可逆**：删除/覆盖默认归档，删除需要证据（§5.7）。

---

## 2. 总体架构

### 2.1 架构总览
```
                        ┌─────────────────────────────────────────────┐
                        │               人机接口 HMI（§4.9）           │
                        │ 智能体页 / 诊断回放 / 授权 / 理想状态 / 纠偏   │
                        └──────────────────┬──────────────────────────┘
                                           │
   ┌───────────────────────────────────────▼───────────────────────────────────────┐
   │                        核心运行时 Core Runtime（§4.1）                         │
   │     阶段状态机 · 监控调度 · 干预协议 · 压缩/恢复 · 目标跟踪（防漂移）           │
   └───┬──────────────┬──────────────┬──────────────┬──────────────┬───────────────┘
       │              │              │              │              │
 ┌─────▼─────┐ ┌──────▼──────┐ ┌─────▼──────┐ ┌────▼──────┐ ┌─────▼──────┐
 │任务契约内核│ │ 资产与溯源  │ │  感知层    │ │  决策层   │ │   执行层   │
 │ §4.2      │ │ §4.3       │ │  §4.4     │ │  §4.5    │ │   §4.6    │
 │ 实例注册  │ │ 单源配置    │ │ 遥测/趋势  │ │ 归因      │ │ 训练生命周期│
 │ 验收契约  │ │ 谱系库      │ │ 诊断执行   │ │ 提案      │ │ 远程运维   │
 │ 机制契约  │ │ payload    │ │ 交叉验证   │ │ 案例检索  │ │ 资源管理   │
 │ 可行性门  │ │ diff/回滚  │ │           │ │ 检查清单  │ │ 授权矩阵   │
 └─────┬─────┘ └──────┬──────┘ └─────┬──────┘ └────┬──────┘ └─────┬──────┘
       └──────────────┴──────────────┴──────────────┴──────────────┘
                                    │
                        ┌───────────▼───────────┐     ┌────────────────────┐
                        │      记忆层（§4.7）   │     │   认知层（§4.8）   │
                        │  KB(契约/规则/案例)   │◄────│ 惊讶/未知/提问     │
                        │  活状态 handoff       │     │ 自我遥测/元回顾    │
                        │  决策日志→案例闭环    │     │ （横跨运行时与决策）│
                        └───────────────────────┘     └────────────────────┘
```

核心循环：**测量 Measure → 归因 Attribute → 决策 Decide → 改变 Change**，以验收契约为固定参照系。
认知层不参与这个循环的业务，而是监视"执行这个循环的 agent 自身"。

### 2.2 关键架构决策（ADR）
- **ADR-1 确定性骨架**：编排器是 Python 状态机，不是 LLM；LLM 只封装在 4 个决策站：
  D1 契约化翻译、D2 归因、D3 提案生成、D4 状态解释/报告。案例检索是确定性检索，不占决策站。
  理由：历史失败模式全是过程违规（漂移/局部/未查原因/狂热重启/压缩失忆），过程只能被代码强制。
- **ADR-2 契约即真相**：验收 scorer 机器执行（家族缺失=不通过）；机制契约以 lint/单测执行；
  提案影响面由 `spec_coverage` 类台账机器计算；scorer 与 spec 对 agent 只读（自评禁令，§6.2）。
- **ADR-3 外部记忆为生存条件**：handoff 在压缩前自动封存；恢复流程 = 读 handoff → 核对远程 → 再行动。
- **ADR-4 实例-通用分层**：知识库与契约全部带层标记（§3 通则）；迁移 = 新实例 + diff + 重导出（§5.4）。
- **ADR-5 两类授权分离**：干预权（持续授予的调参授权）与操作确认（一次性、不可逆/高危）分别管理（§6.3）。
- **ADR-6 双证据通道交叉验证**：四象限协议是归因的强制入口。
- **ADR-7 认知层**：惊讶即中断、已知-未知台账、提问通道、自我遥测——agent 监视自己，麻木是可度量的。
- **ADR-8 工具可锻造、权限有边界**：设施缺口可由提案锻造新工具并注册执行层（§5.8）；
  可写面矩阵（§6.1）划定权限边界。

### 2.3 模块划分
9 个模块（§4.1–4.9），依赖方向单一：HMI → Runtime → {契约内核, 感知, 决策, 执行} → 记忆层；
认知层横跨 Runtime 与决策层（监视全系统）。契约内核不依赖感知/决策；决策只读感知与契约、写决策日志；
执行只接受 Runtime 转发的已批准指令。

---

## 3. 数据模型

**通则**：所有条目携带 `layer`（通用 / 实例）标记，迁移协议据此分类；阈值一律为机器可执行表达式；
未定细节在字段处标注"待定"，不得以口头约定代替字段。

### 3.1 InstanceContext（实例上下文）
```yaml
instance:
  id: "taili"
  layer: instance
  robot: { urdf, joint_count, mass_kg, actuator_limits, leg_geometry, weakest_joint }
  perception: { mode: blind | vision, exteroception: none | camera(...), deploy_available }
  task: { terrain_families, targets: {stairs_up_cm, ...}, envelope: {fwd,bwd,lat,yaw} }
  world: { simulators: [isaaclab, mujoco], fidelity: {pd_kp_kd, timestep, foot_collision_r, com} }
  orthodoxy: { blind: true, mean_action: true, privileged_at_deploy: false }
  baseline: { seed_policy_ref, parent_checkpoint_ref, provenance }
  experiment_budget: { gpu_hours, wall_clock }      # 支持"训练有成本"与"多跑跑"的权衡
```
`InstanceRegistry` 提供 CRUD 与 `diff(instance_a, instance_b)` → 差异清单（驱动迁移分类）。

### 3.2 AcceptanceContract（验收契约）
```yaml
contract:
  instance_id, layer: instance            # 形式通用、数值实例
  goals:                                  # 当前优化目标与优先级（GoalTracker 读取的"§0"）
    - { objective: stand_still, priority: same_as_flat_and_stairs }
    - { objective: flat_quality, priority: ... }
    - { objective: stairs_up_down, priority: ... }
    - { objective: robustness_dr, priority: after_above }
  families:                               # 每族由若干 gate 组成，全部通过才通过
    - { family: A1, metric: vel_tracking, stat: [median, p90],
        threshold: "|v-cmd| <= max(0.10, 0.15*|cmd|)", bucket: fwd05 }
  coverage_required: [A1..F2]             # 未评测家族 = 不通过，不静默跳过
  orthodoxy: { blind, mean_action }
  scorer_ref: "acceptance_score.py"
```
要求：阈值机器可执行；与 scorer 不一致时 scorer 赢。

### 3.3 MechanismContract（机制契约）
```yaml
mechanism:
  layer: universal                       # 条目各自可再标注
  invariants:                            # lint/单测执行
    - one_reward_source
    - per_robot_scalar_reduction
    - bounded_shaping
    - no_degenerate_optimum
    - drive_first_penalty_guard          # 奖励=驱动，惩罚=防钻空子
    - gates_gate_penalties_shape
  prohibitions:                          # lint 执行，违反即拒绝提案
    - manual_stair_state_machine
    - potential_energy_style_semantics
    - unrelated_config_changes
    - privileged_dependency_at_deploy
```

### 3.4 RunRecord / CheckpointNode（谱系）
```yaml
run:   { id, config_hash, code_version, seed, instance_id, parent_checkpoint,
         launch_ts, stop_ts, interventions: [decision_ref...], payload_manifest_hash }
node:  { id, run_id, step, parent_id, lineage_path, evidence: [diag_ref...], capability_tags }
```
`LineageDB` 支持查询："该能力的形成链"、"某配置 fresh 复现记录"。

### 3.5 DecisionRecord / CaseRecord（决策与案例）
```yaml
decision:
  ts, phase, instance_id
  symptom: [可观测信号列表]
  claims: [ { text, claim_type: measured|inferred|assumed|opinion, evidence_refs, confidence } ]
  attribution: { cause_class ∈ {机制错误|权重不当|仿真保真|物理受限|语义漂移}, confidence, evidence }
  falsifiable_prediction: { observable, within_window }     # 若归因正确，窗口内应观察到什么
  questions: [向用户或探针提出的问题]                        # 合法输出，非失败
  proposal: { diff, impact_report_ref, checklist_results[] }
  auth: { intervention_granted, op_confirmed }
  outcome_ref
case:
  layer: { prototype | instance }
  prototype: { pattern, root_cause, remedy }                # 跨实例可检索层
  instance:  { values, context_ref }                        # 实例绑定层，仅参考
```
决策日志 append-only；`CaseClosureJob` 把 决策+结果 沉淀为案例，包括"新的认知方式"而不仅是结论。

### 3.6 HandoffState（活状态）
```yaml
handoff: { sealed_at, run_ref, payload_hash, current_phase,
           established_judgments: [], pending_actions: [], intervention_conditions: [],
           active_falsifiable_predictions: [] }
```
压缩前由 Runtime 强制 `seal()`；恢复时 `restore()` 读取，禁止以压缩摘要为事实。

### 3.7 TrendReport / StallReport / SurpriseRecord（趋势、停滞、惊讶）
```yaml
trend:   { metric, window, slope, platform_flag, resume_bump_flag, alerts: [] }
stall:   { triggered, criteria_met: [连续N窗口无恢复斜率|平台|验收族恶化], since_ts }
surprise: { registered_expectation, violation, action: interrupt_and_reattribute }
```

### 3.8 KnownUnknownLedger（已知-未知台账）
```yaml
unknown: { id, instance_id, description, importance, probe_plan, status: open|probed|resolved }
```
未知必须显式声明；消解未知的默认手段是廉价探针，而非假设。

### 3.9 CapabilityPermissionMatrix（可写面矩阵）
```yaml
permission: { object_class, ops, permission: auto|confirm|manual|forbidden, layer }
```

---

## 4. 模块规格

### 4.1 核心运行时 Core Runtime（骨架）
- **职责**：驱动端到端流程与监控循环，是所有协议的强制执行者。
- **阶段状态机**（§5.1）：
  `BOOT → CONTRACTING → SETUP → TRAINING → MONITORING ⇄ DIAGNOSING ⇄ TUNING ⇄ ACCEPTING → TRANSFERRING ⇄ TUNING → DONE`；
  另有 `PAUSED`（人工打断，可回到原状态）、`FAILED`（封存 handoff，需人工）。
- **关键接口**：
  - `run_phase(state, event) → next_state`（事件表驱动）
  - `MonitorScheduler(cadence, events)`：定时 + 事件（checkpoint/stall/surprise/human）混合唤醒
  - `InterventionProtocol.execute()`：强制顺序 诊断→交叉验证→归因→提案→检查清单→授权→执行→注册可证伪预测→恢复监控
  - `HandoffManager.seal() / restore()`（§5.5）
  - `GoalTracker`：读取 `contract.goals`，任何提案须声明与其关系（防漂移）
- **历史依据**："20/10 分钟间隔监控"、"干预后必须重启训练恢复监控"、"修改时不必停训练"。

### 4.2 任务契约内核 Contract Kernel
- **职责**：实例与契约的唯一管理方；新任务的第一站。
- 接口：
  - `InstanceRegistry.create/update/diff`
  - `AcceptanceContractBuilder(intent) → contract草案`（D1 辅助，人工确认）
  - `FeasibilityGate.check(instance, contract)`：运动学可达/力矩余量/感知可用性——不可行则拒绝或降级，
    降级走目标演化协议（§5.6）
  - `Scorer.run(battery) → verdict`（泛化既有 `acceptance_score.py`）
  - `CoverageAudit`：家族缺失=不通过
  - `MechanismLint.check(config_diff)`：invariants + prohibitions（3.3）
- **历史依据**："从运动学来说是绝对可达的"、220Nm 连杆放大核算、"盲狗不可能提前知道自己在哪"。

### 4.3 资产与溯源 Assets & Provenance
- **职责**：配置单源化、字段可达性与冲突检测、payload、谱系。
- 接口：
  - `ConfigStore`：单源 YAML；schema 校验；字段→运行时对象可达性 lint；**多来源覆盖/冲突检测**
    （历史真问题是"多套生效/过期配置"）；diff/回滚
  - `SourceMap`：机器化 `taili_runtime_source_map.md`（谁是权威、字段真实生效位置）
  - `PayloadBuilder`：manifest + 内容哈希（部署包=可追溯资产）
  - `LineageDB`：3.4 的 run/node 存取与查询
- **基线规则**：未确定的策略不得设为基线（历史明确教训）。
- **历史依据**："多套生效/散乱/乱码"导致改不动、"关键是配置是不是真的改了"、7+ 能力随 resume 链而来。

### 4.4 感知层 Perception
- 接口：
  - `TelemetryClient.get(key_fields) / stats(window)`：只取关键字段子集；滑动均值/斜率/平台检测；
    resume 假性上升识别（重启后短窗口不作为趋势）
  - `DiagnosticsRunner.launch(scene) → {video, physical_stats, family_scores}`：
    双用途——**干预前强制诊断** 与 **里程碑进展诊断**（对检查点做质量评分，如"平地 88 分"）；
    回放接 HMI；探针降级为辅助；日志已明显差时可跳过诊断（协议强制）
  - `CrossValidator.quadrant(log, diag)`：四象限对齐；不一致→产出"语义待修"标记，阻断调参；
    语义修复本身走决策层提案流程（修实现、不修 scorer）
- **历史依据**：日志 progress 高分但视频垮掉；"stance_slip 只给指标不给值"；探针被质疑。

### 4.5 决策层 Decision（LLM 重区）
- 决策站（结构化 I/O，输出不符合 schema 即打回）：
  - `Attributor(D2)`：输入{symptom, trend, diag, config_diff, lineage} →
    输出{cause_class, confidence, evidence, known_unknowns[], questions[]}；
    confidence 低于阈值时输出探针计划或提问，而不是硬凑答案（阈值待定）
  - `Proposer(D3)`：输入{attribution, contract, case_matches} →
    输出{diff, impact_report（哪些家族受影响，spec_coverage 驱动）, occam_selfcheck,
    attack_selfcheck（怎么被钻空子）, falsifiable_prediction}
  - `Explainer(D4)`：状态/判断/提案的人类可读报告
  - `CaseRetriever`：确定性检索，只在原型层匹配
- `ChecklistExecutor`：§5.3 的 7 门机器检查，未通过即打回，不进入执行。
- **历史依据**：纠偏全集——"先查原因"、"动无关项"、"奥卡姆"、"全局"、"奖励会被钻空子"。

### 4.6 执行层 Execution
- 当前实现入口为 `autotuner/execution/`。它是系统源码层，不属于任何 Taili
  或其他机器人产品；IsaacLab/Isaac Sim/PhysX 是由 runtime identity 引用的
  外部不可变运行时，产品 payload 只携带任务适配代码和资产。
- 接口：
  - `RuntimeIdentity` / `PayloadManifest`：为 runtime、产品合同和 payload 生成内容哈希，
    并在运行 manifest 中记录声明证据与实际观测证据。
  - `ChangeSet`：配置路径、源码精确补丁和 runtime 替换三类变更；应用前校验 base hash
    与旧值，失败不写入，提交后保留可验证回滚日志。
  - `ResumeCompatibility`：比较产品/任务、观测动作结构、网络、归一化、物理步长、
    runtime 和配置身份；缺少证据时阻断 resume，而不是猜测兼容。
  - `VersionedRemoteDeployer`：按 digest staging `runtime/payload`，按 run id staging
    独立运行目录，校验后原子激活；回滚只切换 active 指针。
  - 当前不把 `TrainingController`、资源清理策略或授权矩阵伪装成执行层实现。
    进程生命周期、SSH 重连和人工确认仍由 `autotuner/infrastructure/` 及兼容适配器
    承担；它们通过 `DeploymentSpec` 消费执行层产物，不能反向改变 digest 身份。
    这些能力完成后再注册为独立接口，不在本次执行层边界内偷偷扩张。
- **历史依据**：后端重启中断训练、15G 之谜、清理误删风险、SSH 冷却。

### 4.7 记忆层 Memory
- 接口：
  - `KBStore`：契约 / 规则 / 案例（原型+实例双层）/ 地图；全部带层标记；git 管理；
    **可生长**——允许新增规则与案例，包括设计者未写过的
  - `HandoffStore`：3.6 的封存与恢复
  - `DecisionLog`：append-only；`CaseClosureJob` 定期把 决策+结果 沉淀为案例（含新的认知方式）
- **历史依据**：压缩失忆、重复调查、"边调查边记录"、"写在记录文档里"。

### 4.8 认知层 Cognitive（监视 agent 自己）
- `SurpriseRegistry`：注册所有可证伪预测与监控不变量（resume 假升、日志好实际坏、
  "指标合格但 count 不增"等）；违背即中断并强制重新归因。**零惊讶窗口越长越危险**。
- `KnownUnknownLedger`：3.8 台账的读写；决策站可新增未知，探针可消解未知。
- `QuestionChannel`：`questions[]` 是决策站的合法输出；知识缺口大于决策所需时，提问是成功输出。
- `SelfTelemetry`：提案新颖度、提问率、惊讶率、无学习周期数——麻木是可度量的；告警阈值待定。
- `MetaReview`：每 N 次干预强制元回顾（N 待定）：这几轮学到了什么？什么从未被质疑？正在假设什么？
- **历史依据**："你每次都只说状态机有潜能"（无证据重复）、"你难道看不出这诸多问题吗"（麻木）、
  "从 7+ 到现在我们实现了什么"（元回顾）、"不要过于公式化"（被点名麻木）。

### 4.9 人机接口 HMI
- 智能体页规范（历史 UI 纠偏的固化）：只展示最关键信息——门控差距**带具体值**、
  方向 progress 完整不截断、关键趋势、当前判断与待办动作、状态透明（执行中 vs 卡住）；
  不堆砌全部曲线、不要纯装饰卡片、不重复字段。
- 诊断回放；授权面板（操作确认、干预权授予、目标裁定）；理想状态描述输入（供 D1/D3 使用）；
  纠偏入口（§5.6）；对话通道（用户提问/追问/指令 ↔ D4 报告）。

---

## 5. 核心工作流

### 5.1 端到端主线状态机
| 状态 | 进入条件 | 主要动作 | 出口 |
|---|---|---|---|
| BOOT | 启动 | restore handoff → 核对远程 | CONTRACTING / 恢复原状态 |
| CONTRACTING | 新实例 | 建 InstanceContext → 验收契约草案 → 可行性门 | SETUP（人工确认后） |
| SETUP | 契约就绪 | codegraph、单源配置、契约测试、基线 | TRAINING |
| TRAINING | 已启动 | 进入监控循环 | MONITORING |
| MONITORING | 训练中 | §5.2 循环 | DIAGNOSING / ACCEPTING / PAUSED |
| DIAGNOSING | 干预或惊讶触发 | 强制诊断 + 交叉验证 | TUNING（归因后）/ MONITORING（无问题） |
| TUNING | 归因完成 | 提案 → 检查清单 → 授权 → 执行 → 部署 → 恢复训练 | MONITORING |
| ACCEPTING | 阶段里程碑 | battery + 覆盖审计 | TRANSFERRING / **TUNING（发现问题回去修）** / MONITORING |
| TRANSFERRING | 验收过 | sim2sim 保真核对 → sim2real 口径检查 | DONE / **TUNING（迁移暴露问题回去修）** |
| DONE | 交付 | 策略 + 谱系 + 报告 | — |
| PAUSED / FAILED | 人工/异常 | 封存 handoff | 人工 |

### 5.2 监控循环与干预协议（伪码）
```
every cadence or on event:
    snap = telemetry(key_fields)
    trend = stats(snap)
    if SurpriseRegistry.violated(trend, diag): interrupt → 强制重新归因   # 惊讶即中断
    if stall_or_degradation(trend) and intervention_authorized:
        diag = DiagnosticsRunner.launch()              # 干预前强制
        cross = CrossValidator.quadrant(log, diag)
        if cross.mismatch: 语义待修标记 → 阻断调参，修复走提案流程（scorer 禁改）
        attr = Attributor(...)
        if attr.confidence < threshold: probe 或 QuestionChannel 提问     # 不硬凑答案
        prop = Proposer(...)
        if ChecklistExecutor.pass(prop):
            if AuthorizationMatrix.need_confirm(prop): await human()
            TrainingController.apply(prop); RemoteOps.deploy()
            SurpriseRegistry.register(prop.falsifiable_prediction)
            resume_training()
    seal_handoff_if_needed()
```
干预授权来源于用户"进入 X 分钟间隔监控，必要时干预"；干预后必须回到监控。

### 5.3 全局检查清单（7 门，机器强制）
1. 原因已查明？依据当前 payload/完整趋势/真实诊断，而非记忆？
2. 影响面已报告？改动驱动/影响哪些验收家族？需要哪些配套协调？
3. 未动与当前问题无关的配置？
4. 奥卡姆：新增机制是否与已有语义重复、能否删？（禁止状态机/势能语义）
5. 全局优先级一致？（本门内容为实例层：站立/平地/楼梯同优先级、DR 不交换质量）
6. 契约测试通过？字段真实生效（可达性 lint + 冲突检测）？
7. 干预理由满足判定规则（停滞/退化/矛盾证据）？
清单的形式与第 1/2/3/4/6/7 门为通用层；第 5 门内容随实例契约的 goals 展开。

### 5.4 迁移协议
```
new = InstanceContext(...); d = InstanceRegistry.diff(old, new)
assets_classified = classify(each asset: 复用 | 重导结构 | 重标数值 | 重新基线)
执行顺序（固定）：契约 → 机制 → 参数 → 基线 → 训练 → 回归 battery（全家族）
```
案例检索仅在原型层；验收契约的形式复用、数值重标；知识库条目全部带层标记。

### 5.5 压缩/恢复协议
- 压缩前：Runtime 强制 `HandoffManager.seal()`（run/payload 哈希/已确定判断/待办动作/干预条件/活跃预测）。
- 恢复后：读 handoff → 核对远程真实状态 → 恢复监控；压缩摘要与历史记忆不得作为当前事实。

### 5.6 目标演化与纠偏回路
- **目标演化（同实例内需求变化：新增动作/更高地形/新零部件）**：
  意图 → 契约修订草案（新家族/新阈值/新 goals）→ 可行性门 → **谱系影响评估**
  （既有能力是否会被破坏、是否需要 fresh、历史 resume 链是否可复用）→ 人工裁定 → 契约生效 →
  检查清单第 5 门同步。
- **纠偏（用户对 agent 判断/行为的纠正）**：分类为四类落点——规则修正（KB）、案例新增（KB）、
  契约修订（走目标演化）、立即中断（惊讶）；纠偏必须留下记录，后续决策中可被引用与检索。

### 5.7 破坏性操作协议（删除/覆盖/重启等不可逆操作）
- 不可逆操作是**决策事件**而非执行动作：产生 DecisionRecord（症状/归因/提案/证据）。
- **归档优先**：默认归档而非删除。
- **删除需三类证据**：依赖图无引用、谱系可重建（有重建路径）、计划不需要（预算内无用途）。
- 白名单只是候选生成的默认值，不是决定本身；**删/留/归档的价值判断进决策中心**，
  执行层只提供候选与证据；保留锚点检查点集合为实例层数据。

### 5.8 工具锻造协议
- 设施缺口发现（既有设施无此能力且任务需要）→ 提案（新工具、用途、风险评估、注册类别）→
  人工确认 → 实现 → 契约测试 → 注册进执行层 → 此后调用走标准授权路径。
- 例：远程开关机工具——探测 IPMI / Wake-on-LAN / 智能 PDU 后实现控制脚本并注册。

---

## 6. 能力与权限面

### 6.1 可写面矩阵
| 对象 | 权限 |
|---|---|
| 单源配置 YAML | 可写（自动） |
| 奖励/门控/课程实现（`taili_reward.py` 等） | 可写（过 lint + 契约测试） |
| 诊断/工具脚本、payload 组装规则 | 可写（人工确认） |
| 验收 scorer、spec、机制契约、训练框架核心 | **只读，禁止** |
| 契约数值、禁止项、goals | 仅人工 |

### 6.2 自评禁令
agent 不得修改验收 scorer、spec 与机制契约——**禁止自己出题、自己判卷**。
奖励实现的修改必须经由独立的契约测试验证，测试本身不在 agent 可写面内。

### 6.3 两类授权
- **干预权**：持续性授权（"进入 X 分钟间隔监控，必要时干预"），可随时收回；授予后才允许走 §5.2 干预路径。
- **操作确认**：一次性授权，针对不可逆/高危操作（部署、重启、清理、大改）；每次执行前确认。
- 两者生命周期不同，不得混用同一字段。

### 6.4 远程能力探测与降级
- SSH 可达：自动执行。
- SSH 不可达：探测硬件手段（IPMI / Wake-on-LAN / 智能 PDU，具体待定）。
- 均不可达：降级为"人工执行 + agent 验证"模式，并在 handoff 中记录。

---

## 7. 人类在环模型

| 交互点 | 内容 | 形态 |
|---|---|---|
| 目标裁定 | 契约确认、目标演化裁定、可行性降级协商 | 授权面板 |
| 授权 | 操作确认（不可逆/高危）+ 干预权授予/收回 | 授权面板 |
| 视觉判断 | 用户看渲染视频，agent 读物理数据，交叉验证 | 诊断回放 |
| 理想状态描述 | "优雅、轻脚、像漂浮"等意图输入 | 智能体页，供 D1/D3 翻译 |
| 纠偏 | 对 agent 判断/行为的纠正，四类落点（§5.6） | 纠偏入口 |

---

## 8. 与现有资产的对接

| 现有资产 | 处置 |
|---|---|
| `acceptance_score.py` / `taili_spec.md` | 泛化为 AcceptanceContract + Scorer（保留 Taili 为默认实例） |
| `spec_coverage.py` | 接入 Proposer 的影响面报告 |
| `taili_strategy_decisions.md` | 转写为 MechanismContract 的 invariants/prohibitions lint |
| `taili_runtime_source_map.md` | 机器化为 SourceMap |
| payload / 诊断 / console（autotuner） | 以适配器复用，不重写 |
| `taili_live_handoff.md` | 机器化为 HandoffState |
| `docs/archive/research-agent/rl_agent_design.md`、`docs/archive/taili/taili_session_lessons.md` | 论证与素材，保留为背景文档 |

---

## 9. 工作分解与分工

设计划分为若干并列的板块（工作包）。每个板块是一个需要被完整承担的独立工作单元；
本设计只界定各板块的目标、交付物、验收与依赖，不涉及由谁承担。

### 9.1 板块清单

| 板块 | 目标 | 交付物 | 验收 | 依赖 |
|---|---|---|---|---|
| **0 数据模型与接口冻结** | 冻结 §3 全部 schema、§4 接口签名、各模块间钩子（含认知层与运行时） | 代码骨架、类型定义、测试桩 | 契约/单测可跑；各板块按此对齐 | 无（先行） |
| **A 任务契约内核** | 实例与契约的唯一管理方 | InstanceRegistry、Scorer 泛化、MechanismLint、FeasibilityGate、目标演化入口 | Taili 契约迁移通过；禁止项可拦截 | 0 |
| **B 资产与溯源** | 配置单源化与谱系 | ConfigStore（可达性+冲突检测）、LineageDB、PayloadBuilder、基线规则 | 历史 run 可重建谱系；多套生效可检出 | 0 |
| **C 感知层** | 遥测/诊断/交叉验证 | Telemetry 统计与停滞检测、诊断双用途执行、CrossValidator | 四象限样例通过；resume 假升可识别 | 0 |
| **D 核心运行时** | 流程骨架 | 状态机（含回边）、监控循环、干预协议、Handoff seal/restore | 模拟场景跑通 §5.2；惊讶可触发中断 | 0、A、C |
| **E 决策层** | 归因/提案/检索 | 4 个决策站 + CaseRetriever + ChecklistExecutor | 历史决策样本回放，7 门可拦截；低置信触发提问 | 0、A、B、C |
| **F 执行层** | 训练与远程执行 | 训练控制、远程运维+能力探测降级、资源管理分层、授权矩阵 | 高危操作需授权；清理走决策事件 | 0、B |
| **G 记忆层** | KB 与状态持久化 | KBStore（带层标记、可生长）、HandoffStore、DecisionLog+案例闭环 | 压缩/恢复协议可演示 | 0 |
| **H 认知层** | 监视 agent 自身 | SurpriseRegistry、KnownUnknownLedger、QuestionChannel、SelfTelemetry、MetaReview | 注入的惊讶可触发中断；未知被声明 | 0（与 D 的钩子在 0 冻结） |
| **I 人机接口** | 智能体页与授权 | 智能体页（按 §4.9 规范）、诊断回放、授权/纠偏面板、对话通道 | 页面无冗余字段；授权与纠偏流可用 | D、E、F |
| **J 迁移与集成验证** | 端到端闭环 | 迁移协议、破坏性操作协议、第二实例试迁移 | §10 的 A3 通过 | 全部 |

### 9.2 依赖与实施顺序

```
0（先行，含各模块接口与钩子冻结）
 └─► A、B、C、F、G（互相解耦，可并行推进）＋ H（认知层，接口随 0 冻结）
       ├─► D ──► I
       └─► E ──► I
               └─► J（集成与第二实例验证）
```

顺序原则：先契约与数据（0），再平铺基础板块（A/B/C/F/G/H），随后拼装运行时（D）与决策（E），
最后接人机界面（I）并做端到端集成验证（J）。

---

## 10. Agent 自身的验收标准

- **A1 闭环**：Taili 实例上，人工只做授权、视觉判断、目标裁定与纠偏，agent 独立完成 监控→诊断→调参→验收 闭环。
- **A2 恢复**：上下文清空/重启后，仅凭 handoff+KB 恢复正确状态并继续，不重复调查。
- **A3 迁移**：一个新实例（换地形要求或换机器人），agent 不依赖重新教学，走完契约化→训练→验收主线。
- **A4 漂移收敛**：决策日志中"动无关项/未查原因/加冗余机制"类提案为零（被 7 门拦截或不再出现）。
- **A5 溯源**：任意交付策略可回答"能力从哪条 resume 链、哪些干预长出"。
- **A6 目标演化**：中期插入新要求后，agent 能完成契约修订、谱系影响评估并给出 fresh/resume 判断。
- **A7 认知与纠偏**：注入的惊讶能触发中断；未知被显式声明而非假设；用户纠偏进入 KB 并被后续决策引用；
  设施缺口能被锻造为新工具并注册。

---

## 11. 风险与待定项

1. 可行性门的运动学/力矩估计：先用解析近似；是否引入最小仿真验证待定。
2. 决策站 LLM 质量波动：schema 打回 + Checklist 兜底；必要时在 D3 增加评审 LLM（预案，不预先实现）。
3. 单智能体上下文压力：先按本设计运行，若 A2/A4 不达标再拆 Monitor/Reviewer 独立进程（预案）。
4. console 改造范围：优先"适配器+新智能体页"，避免重写 autotuner 引入回归。
5. 远程环境脆弱性：执行层必须可降级为"人工执行+agent 验证"模式。
6. 待定参数集中记录：监控间隔默认值（历史 10/20 分钟）、stall 判定窗口参数、归因置信度阈值、
   惊讶检测与自我遥测告警阈值、MetaReview 周期 N、自评禁令粒度（文件级/对象级）、
   远程能力探测的支持清单（IPMI/WOL/PDU）——一律以字段表达，不得口头约定。
