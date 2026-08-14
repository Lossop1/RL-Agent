# 端到端机器人强化学习智能体 —— 顶层设计

> **文档性质**：本文件是系统的顶层设计，作为实现、讨论与评审的共同基准，不是单方面权威；
> 实现中发现的设计问题以讨论结论为准，本文件随讨论持续修订。
>
> **系统代号**：`rl-agent`（端到端机器人强化学习智能体）。建议新建顶层包 `agent/`，与 `autotuner/` 并列；
> 通过适配器复用 `autotuner` 既有能力（payload/诊断/console）。是否并入 `autotuner` 属实现细节，可提案。
>
> **阅读顺序**：
> ① 本文件 → ② `docs/taili_spec.md`（验收契约范式）→ ③ `docs/taili_strategy_decisions.md`（机制契约范式）
> → ④ `docs/taili_live_handoff.md`（活状态范式）→ ⑤ `docs/rl_agent_design.md`（设计论证过程）
> → ⑥ `docs/taili_session_lessons.md`（历史素材，仅作背景，不作为实现依据）。
>
> **术语**：实例（Instance）= 三元组 (机器人, 任务, 仿真世界)；家族（Family）= 验收契约中的一组硬门槛（A–F）；
> 口径（Orthodoxy）= 部署侧约束（盲、mean-action、无特权）；谱系（Lineage）= 检查点/resume 的 DAG；
> 原型案例 / 实例案例 = 知识库跨实例可复用层 / 实例绑定层。

---

## 1. 目标与边界

### 1.1 系统目标
一个**专用于机器人强化学习**的端到端智能体：接收任务需求，走完
**契约化 → 系统建立 → 训练 → 监控 → 诊断 → 调参 → 验收 → sim2sim/sim2real → 交付**，
交付物 = 策略 + 完整谱系 + 验收报告。人类全程只做四件事：授权、视觉判断、理想状态描述、目标裁定。

### 1.2 非目标（边界）
- 不是通用 agent 框架：流水线、设施、知识库均为机器人 RL 专用，为任务形态量身定做。
- 不做通用机器人控制算法研究：奖励/门控/课程的**形式**与规则由知识库约束，数值由实例导出。
- 不替代真机部署工程：sim2real 只到"部署口径验证与核对清单"，真机操作人执行。
- 通用性只通过**迁移协议**（§5.4）获得，不通过抽象层堆砌。

### 1.3 设计原则（对实现的硬约束）
- **P1 单一事实源**：机器契约赢过文档（scorer 与文档冲突时 scorer 是真相）；YAML 是唯一可编辑源；生成物不可手改。
- **P2 确定性骨架 + LLM 决策站**：流程、协议、护栏是确定性代码；LLM 只在规定的决策站被调用，且输入输出均为结构化 schema。
- **P3 决策-执行分离**：任何改变训练状态的提案必须通过全局检查清单（§5.3）后才可执行。
- **P4 压缩安全**：所有判断先落盘（handoff / 决策日志）再继续；恢复后先读状态、核对远程，不凭记忆。
- **P5 实例-通用分层**：一切知识与数值要么标记为"通用"，要么绑定实例；跨实例引用必须经过迁移分类。
- **P6 双证据通道**：日志(趋势)与物理诊断(真实)互相矛盾时，先修语义，再谈调参。
- **P7 编码规范**：注释中文、UTF-8 无 BOM（沿用 `CURRENT_FINAL_STRATEGY.md` 的整理原则）。

---

## 2. 总体架构

### 2.1 架构总览
```
                        ┌─────────────────────────────────────────────┐
                        │               人机接口 HMI（§4.8）           │
                        │  智能体页 / 诊断回放 / 授权 / 理想状态输入    │
                        └──────────────────┬──────────────────────────┘
                                           │
   ┌───────────────────────────────────────▼───────────────────────────────────────┐
   │                        核心运行时 Core Runtime（§4.1）                         │
   │   阶段状态机 · 监控调度 · 干预协议 · 压缩/恢复 · 目标跟踪（防漂移）             │
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
                        ┌───────────▼───────────┐
                        │      记忆层（§4.7）   │
                        │  KB(契约/规则/案例)   │
                        │  活状态 handoff       │
                        │  决策日志→案例闭环    │
                        └───────────────────────┘
```

核心循环（一切工作的归约）：**测量 Measure → 归因 Attribute → 决策 Decide → 改变 Change**，
以验收契约作为固定参照系。五层模块都是这个循环的载体。

### 2.2 关键架构决策（ADR）
- **ADR-1 确定性骨架**：编排器是 Python 状态机，不是 LLM。LLM 只封装在 5 个决策站：
  D1 契约化翻译、D2 归因、D3 提案生成、D4 案例检索、D5 状态解释/报告。
  理由：历史失败模式全是过程违规（漂移/局部/未查原因/狂热重启/压缩失忆），过程只能被代码强制。
- **ADR-2 契约即真相**：验收 scorer 机器执行（家族缺失=不通过）；机制契约以 lint/单测执行；
  提案影响面由 `spec_coverage` 类台账机器计算。
- **ADR-3 外部记忆为生存条件**：handoff 在压缩前自动封存；恢复流程 = 读 handoff → 核对远程 → 再行动。
- **ADR-4 实例-通用分层**：知识库与契约全部带层标记（见 §3）；迁移 = 新实例 + diff + 重导出（§5.4）。
- **ADR-5 决策-执行分离 + 人类授权矩阵**：高危操作（远程部署/重启/清理/大改）需人工确认。
- **ADR-6 双证据通道交叉验证**：日志好≠实际好；四象限协议是决策层归因的强制入口。

### 2.3 模块划分
8 个模块（§4.1–4.8），依赖方向单一：HMI → Runtime → {契约内核, 感知, 决策, 执行} → 记忆层；
契约内核不依赖感知/决策；决策只读感知与契约，写决策日志；执行只接受 Runtime 转发的已批准指令。

---

## 3. 数据模型（契约层，实现必须遵守）

### 3.1 InstanceContext（实例上下文）
```yaml
instance:
  id: "taili"
  robot: { urdf, joint_count, mass_kg, actuator_limits, leg_geometry, weakest_joint }
  perception: { mode: blind | vision, exteroception: none | camera(...), deploy_available }
  task: { terrain_families, targets: {stairs_up_cm, ...}, envelope: {fwd,bwd,lat,yaw} }
  world: { simulators: [isaaclab, mujoco], fidelity: {pd_kp_kd, timestep, foot_collision_r, com} }
  orthodoxy: { blind: true, mean_action: true, privileged_at_deploy: false }
  baseline: { seed_policy_ref, parent_checkpoint_ref, provenance }
```
`InstanceRegistry` 提供 CRUD 与 `diff(instance_a, instance_b)` → 差异清单（驱动迁移分类）。

### 3.2 AcceptanceContract（验收契约）
```yaml
contract:
  version, instance_id
  families:                       # 每族由若干 gate 组成，全部通过才通过
    - { family: A1, metric: vel_tracking, stat: [median, p90],
        threshold: "|v-cmd| <= max(0.10, 0.15*|cmd|)", bucket: fwd05 }
  coverage_required: [A1..F2]     # 未评测家族 = 不通过，不静默跳过
  orthodoxy: { blind, mean_action }
  scorer_ref: "acceptance_score.py"
```
要求：阈值必须是机器可执行的表达式；与 `acceptance_score.py` 不一致时 scorer 赢。

### 3.3 MechanismContract（机制契约）
```yaml
mechanism:
  invariants:   # lint/单测执行
    - one_reward_source
    - per_robot_scalar_reduction
    - bounded_shaping
    - no_degenerate_optimum
    - drive_first_penalty_guard      # 奖励=驱动，惩罚=防钻空子
    - gates_gate_penalties_shape
  prohibitions: # lint 执行，违反即拒绝提案
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
`LineageDB` 支持查询："该能力的形成链"、"某配置 fresh 复现成功率标记"。

### 3.5 DecisionRecord / CaseRecord（决策与案例）
```yaml
decision: { ts, phase, symptom(可观测信号列表), attribution{cause_class, confidence, evidence},
            proposal{diff, impact_report_ref, checklist_results[]}, auth, outcome_ref }
case:
  prototype: { pattern, root_cause, remedy }        # 跨实例可检索层
  instance:  { values, context_ref }                 # 实例绑定层，仅参考
```
决策日志 append-only；离线任务把 `decision+outcome` 沉淀为新案例。

### 3.6 HandoffState（活状态）
```yaml
handoff: { sealed_at, run_ref, payload_hash, current_phase,
           established_judgments: [], pending_actions: [], intervention_conditions: [] }
```
压缩前由 Runtime 强制 `seal()`；恢复时 `restore()` 读取，禁止以压缩摘要为事实。

### 3.7 TrendReport / StallReport（趋势与停滞判定）
```yaml
trend:  { metric, window, slope, platform_flag, resume_bump_flag, alerts: [] }
stall:  { triggered, criteria_met: [连续N窗口无恢复斜率|平台|验收族恶化], since_ts }
```

---

## 4. 模块规格

### 4.1 核心运行时 Core Runtime（骨架）
- **职责**：驱动端到端流程与监控循环，是所有协议的强制执行者。
- **阶段状态机**（§5.1）：
  `BOOT → CONTRACTING → SETUP → TRAINING → MONITORING ⇄ DIAGNOSING ⇄ TUNING → ACCEPTING → TRANSFERRING → DONE`；
  另有 `PAUSED`（人工打断，可回到原状态）、`FAILED`（不可恢复，需人工）。
- **关键接口**：
  - `run_phase(state, event) → next_state`（事件表驱动）
  - `MonitorScheduler(cadence, events)`：定时 + 事件（checkpoint/stall/human）混合唤醒
  - `InterventionProtocol.execute()`：强制顺序 诊断→交叉验证→归因→提案→检查清单→授权→执行→恢复监控
  - `HandoffManager.seal() / restore()`（§5.5）
  - `GoalTracker`：读取契约 §0 当前目标，任何提案须声明与其关系（防漂移）
- **历史依据**："20/10 分钟间隔监控"、"干预后必须重启训练恢复监控"、"修改时不必停训练"。

### 4.2 任务契约内核 Contract Kernel
- **职责**：实例与契约的唯一管理方；新任务的第一站。
- 接口：
  - `InstanceRegistry.create/update/diff`
  - `AcceptanceContractBuilder(intent) → contract草案`（D1 辅助，人工确认）
  - `FeasibilityGate.check(instance, contract)`：运动学可达/力矩余量/感知可用性——不可行则拒绝或降级目标
  - `Scorer.run(battery) → verdict`（泛化既有 `acceptance_score.py`）
  - `CoverageAudit`：家族缺失=不通过
  - `MechanismLint.check(config_diff)`：invariants + prohibitions（3.3）
- **历史依据**："从运动学来说是绝对可达的"、220Nm 连杆放大核算、"盲狗不可能提前知道自己在哪"。

### 4.3 资产与溯源 Assets & Provenance
- **职责**：配置单源化、字段可达性、payload、谱系。
- 接口：
  - `ConfigStore`：单源 YAML；schema 校验；**字段→运行时对象可达性 lint**；diff/回滚
  - `SourceMap`：机器化 `taili_runtime_source_map.md`（谁是权威、字段真实生效位置）
  - `PayloadBuilder`：manifest + 内容哈希（部署包=可追溯资产）
  - `LineageDB`：3.4 的 run/node 存取与查询
- **历史依据**："多套生效/散乱/乱码"导致改不动、"关键是配置是不是真的改了"、7+ 能力随 resume 链而来。

### 4.4 感知层 Perception
- 接口：
  - `TelemetryClient.get(key_fields) / stats(window)`：只取关键字段子集；滑动均值/斜率/平台检测；
    **resume 假性上升识别**（重启后短窗口不作为趋势）
  - `DiagnosticsRunner.launch(scene) → {video, physical_stats, family_scores}`；回放接 HMI；
    探针降级为辅助；**干预前必诊断、日志已明显差时可跳过诊断**（协议强制）
  - `CrossValidator.quadrant(log, diag)`：四象限对齐；不一致→产出"语义待修"标记，阻断调参
- **历史依据**：日志 progress 高分但视频垮掉；"stance_slip 只给指标不给值"；探针被质疑。

### 4.5 决策层 Decision（唯一的 LLM 重区）
- 决策站（结构化 I/O，任何一处输出不符合 schema 即打回）：
  - `Attributor(D2)`：输入{symptom, trend, diag, config_diff, lineage} →
    输出{cause_class ∈ {机制错误|权重不当|仿真保真|物理受限|语义漂移}, confidence, evidence}
  - `Proposer(D3)`：输入{attribution, contract, case_matches} →
    输出{diff, impact_report（哪些家族受影响，spec_coverage 驱动）, occam_selfcheck, attack_selfcheck（怎么被钻空子）}
  - `CaseRetriever(D4)`：只在**原型层**检索匹配
  - `Explainer(D5)`：状态/判断/提案的人类可读报告
- `ChecklistExecutor`：§5.3 的 7 门机器检查，未通过即打回，不进入执行。
- **历史依据**：纠偏全集——"先查原因"、"动无关项"、"奥卡姆"、"全局"、"奖励会被钻空子"。

### 4.6 执行层 Execution
- 接口：
  - `TrainingController`：fresh/resume/stop/status、检查点选择（依 L2 决策规则）
  - `RemoteOps`：SSH 部署 payload、冷却重连、远程状态核对（"训练还在吗"一键真相）
  - `ResourceManager`：磁盘清理白名单、GPU 争用提示（诊断 vs 训练）、**本地禁止并行 pytorch**
  - `AuthorizationMatrix`：操作分三类 {auto, confirm, forbidden}；confirm 类走 HMI 授权
- **历史依据**：后端重启中断训练、15G 之谜、清理误删风险、SSH 冷却。

### 4.7 记忆层 Memory
- 接口：
  - `KBStore`：L1 契约 / L2 规则 / L4 案例（原型+实例双层）/ L5 地图，git 版本化
  - `HandoffStore`：3.6 的封存与恢复
  - `DecisionLog`：append-only；`CaseClosureJob` 定期把 决策+结果 沉淀为案例
- **历史依据**：压缩失忆、重复调查、"边调查边记录"、"写在记录文档里"。

### 4.8 人机接口 HMI
- 智能体页规范（历史 UI 纠偏的直接固化）：
  只展示**最关键信息**：门控差距**带具体值**（`stance_slip 0.08 / 需 ≤0.05`）、方向 progress 完整不截断、
  关键趋势、当前判断与待办动作、状态透明（执行中 vs 卡住）；**不堆砌全部曲线、不要纯装饰卡片、不重复字段**。
- 诊断回放；授权面板（confirm 类操作）；理想状态描述输入（供 D1/D3 使用）。

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
| DIAGNOSING | 干预条件触发 | 强制诊断 + 交叉验证 | TUNING（归因后）/ MONITORING（无问题） |
| TUNING | 归因完成 | 提案 → 检查清单 → 授权 → 执行 → 部署 → 恢复训练 | MONITORING |
| ACCEPTING | 阶段里程碑 | battery + 覆盖审计 | TRANSFERRING / MONITORING（未过继续） |
| TRANSFERRING | 验收过 | sim2sim 保真核对 → sim2real 口径检查 | DONE |
| DONE | 交付 | 策略 + 谱系 + 报告 | — |
| PAUSED / FAILED | 人工/异常 | 封存 handoff | 人工 |

### 5.2 监控循环与干预协议（伪码）
```
every cadence or on event:
    snap = telemetry(key_fields)
    trend = stats(snap)
    if stall_or_degradation(trend) and intervention_authorized:
        diag = DiagnosticsRunner.launch()          # 强制
        cross = CrossValidator.quadrant(log, diag) # 不一致→阻断并标记语义待修
        attr = Attributor(...)
        prop = Proposer(...)
        if ChecklistExecutor.pass(prop):
            if AuthorizationMatrix.need_confirm(prop): await human()
            TrainingController.apply(prop); RemoteOps.deploy(); resume_training()
    seal_handoff_if_needed()
```
干预授权来源于用户"进入 X 分钟间隔监控，必要时干预"；干预后必须回到监控。

### 5.3 全局检查清单（7 门，机器强制）
1. 原因已查明？依据当前 payload/完整趋势/真实诊断，而非记忆？
2. 影响面已报告？改动驱动/影响哪些验收家族？需要哪些配套协调？
3. 未动与当前问题无关的配置？
4. 奥卡姆：新增机制是否与已有语义重复、能否删？（禁止状态机/势能语义）
5. 全局优先级一致：站立/平地/楼梯同优先级；DR 不交换运动质量？
6. 契约测试通过？字段真实生效（可达性 lint）？
7. 干预理由满足判定规则（停滞/退化/矛盾证据）？

### 5.4 迁移协议
```
new = InstanceContext(...); d = InstanceRegistry.diff(old, new)
assets_classified = classify(each asset: 复用 | 重导结构 | 重标数值 | 重新基线)
执行顺序（固定）：契约 → 机制 → 参数 → 基线 → 训练 → 回归 battery（全家族）
```
案例检索仅在原型层；验收契约的**形式**复用、**数值**重标；知识库条目全部带层标记。

### 5.5 压缩/恢复协议
- 压缩前：Runtime 强制 `HandoffManager.seal()`（封存 run/payload 哈希/已确定判断/待办动作/干预条件）。
- 恢复后：读 handoff → 核对远程真实状态 → 恢复监控；压缩摘要与历史记忆**不得**作为当前事实。

---

## 6. 人类在环模型

| 交互点 | 内容 | 形态 |
|---|---|---|
| 授权 | 远程部署/重启/清理/大改配置 | 授权面板（confirm 类操作） |
| 视觉判断 | 用户看渲染视频，agent 读物理数据，交叉验证 | 诊断回放 |
| 理想状态描述 | "优雅、轻脚、像漂浮"等意图输入 | 智能体页输入，供 D1/D3 翻译 |
| 监控授权 | "进入 10/20 分钟间隔监控，必要时干预" | 智能体页指令 |

---

## 7. 与现有资产的对接

| 现有资产 | 处置 |
|---|---|
| `acceptance_score.py` / `taili_spec.md` | 泛化为 AcceptanceContract + Scorer（保留 Taili 为默认实例） |
| `spec_coverage.py` | 接入 Proposer 的影响面报告 |
| `taili_strategy_decisions.md` | 转写为 MechanismContract 的 invariants/prohibitions lint |
| `taili_runtime_source_map.md` | 机器化为 SourceMap |
| payload / 诊断 / console（autotuner） | 以适配器复用，不重写 |
| `taili_live_handoff.md` | 机器化为 HandoffState |
| `docs/rl_agent_design.md`、`taili_session_lessons.md` | 论证与素材，保留为背景文档 |

---

## 8. 工作分解与分工

设计划分为若干并列的板块（工作包）。每个板块是一个需要被完整承担的独立工作单元；
本设计只界定各板块的目标、交付物、验收与依赖，不涉及由谁承担。

### 8.1 板块清单

| 板块 | 目标 | 交付物 | 验收 | 依赖 |
|---|---|---|---|---|
| **0 数据模型与接口冻结** | 冻结 §3 全部 schema 与 §4 接口签名 | 代码骨架、类型定义、测试桩 | 契约/单测可跑；各板块按此对齐 | 无（先行） |
| **A 任务契约内核** | 实例与契约的唯一管理方 | InstanceRegistry、Scorer 泛化、MechanismLint、FeasibilityGate | Taili 契约迁移通过；禁止项可拦截 | 0 |
| **B 资产与溯源** | 配置单源化与谱系 | ConfigStore+可达性 lint、LineageDB、PayloadBuilder | 历史 run 可重建谱系 | 0 |
| **C 感知层** | 遥测/诊断/交叉验证 | Telemetry 统计与停滞检测、诊断执行、CrossValidator | 四象限样例通过 | 0 |
| **D 核心运行时** | 流程骨架 | 状态机、监控循环、干预协议、Handoff seal/restore | 模拟场景跑通 §5.2 | 0、A、C |
| **E 决策层** | 归因/提案/检索 | 5 个决策站 + ChecklistExecutor | 历史决策样本回放，7 门可拦截 | 0、A、B、C |
| **F 执行层** | 训练与远程执行 | 训练控制、远程运维、资源护栏、授权矩阵 | 高危操作需授权；清理白名单生效 | 0、B |
| **G 记忆层** | KB 与状态持久化 | KBStore、HandoffStore、DecisionLog + 案例闭环 | 压缩/恢复协议可演示 | 0 |
| **H 人机接口** | 智能体页与授权 | 智能体页（按 §4.8 规范）、诊断回放、授权面板 | 页面无冗余字段；授权流可用 | D、E、F |
| **I 迁移与集成验证** | 端到端闭环 | 迁移协议、案例闭环、第二实例试迁移 | §9 的 A3 通过 | 全部 |

### 8.2 依赖与实施顺序

```
0（先行）
 └─► A、B、C、F、G（互相解耦，可并行推进）
       ├─► D ──► H
       └─► E ──► H
               └─► I（集成与第二实例验证）
```

顺序原则：先契约与数据（0），再平铺五个基础板块（A/B/C/F/G），随后拼装运行时（D）与决策（E），
最后接人机界面（H）并做端到端集成验证（I）。

## 9. Agent 自身的验收标准

- **A1 闭环**：Taili 实例上，人工只做授权与视觉判断，agent 独立完成 监控→诊断→调参→验收 闭环。
- **A2 恢复**：上下文清空/重启后，仅凭 handoff+KB 恢复正确状态并继续，不重复调查。
- **A3 迁移**：一个新实例（换地形要求或换机器人），agent 不依赖重新教学，走完契约化→训练→验收主线。
- **A4 漂移收敛**：决策日志中"动无关项/未查原因/加冗余机制"类提案为零（被 7 门拦截或不再出现）。
- **A5 溯源**：任意交付策略可回答"能力从哪条 resume 链、哪些干预长出"。

---

## 10. 风险与开放问题
1. 可行性门的运动学/力矩估计：先用解析近似，不足再上最小仿真验证（避免回到"状态机式过度设计"）。
2. 决策站 LLM 质量波动：用 schema 打回 + Checklist 兜底；必要时在 D3 增加评审 LLM（历史"评审角色"）。
3. 单智能体上下文压力：先按本设计跑，若 A2/A4 不达标再拆 Monitor/Reviewer 独立进程（预案，不预先实现）。
4. console 改造范围：优先"适配器+新智能体页"，避免重写 autotuner 引入回归。
5. 远程环境脆弱性（重启/磁盘/SSH）：执行层必须可降级为"人工执行+agent 验证"模式。
