# RL Agent 落地交接（2026-08-15）

## 2026-08-17 续接状态（覆盖下文“只读阶段”描述）

只读审计阶段已经扩展为可执行的最小活系统闭环。当前新增并接通：

- 持久 `ResearchState`、CAS、跨进程锁、哈希事件链和崩溃恢复；
- `MechanismSpec` 受限 AST，以及奖励/指标/门控的合成、patch、编译和运行时；
- Taili 生产奖励接入，候选执行时自动注入 `TAILI_MECHANISM_BUNDLE`；
- 静态、单位、数值梯度、特权依赖、反作弊和 evaluator 保护验证；
- `ResearchCycleManager`、隔离 workspace、独立 evaluator、晋升/回滚和 outcome 学习；
- no-repeat、候选 baseline fingerprint 绑定、历史决策盲回放；
- Agent 研究工具与“提出周期/执行已批准周期”动作；
- API 的 state/validate/propose/register/execute/replay 路由；
- 参数级提案确认、持久 at-most-once 执行 receipt、资源租约列表；
- 本地日志型 backend 和受控 SSH backend。

统一状态根为 `output/research`。完整运维步骤、环境变量、权限边界与验证命令见
[`rl_agent_live_system_operations.md`](../../rl_agent_live_system_operations.md)。

### 2026-08-17 真实 GPU 验收

不再只依赖 mock、单测或 CPU SSH smoke。隔离研究周期
`cycle:rgpu:060209` 已在真实 RTX 4090 IsaacLab 环境中完成 70 step：系统合成并
选择正向驱动候选，动态 reward、metric 和 gate 均进入 Taili 生产奖励与遥测，
64 样本 gate 最终 ready/passed；独立 evaluator 通过后 disposition 为 promote，
case 为 resolved，重复执行被 receipt 阻断。原始证据和失败反例见运维说明第 11 节。

该过程实际发现并修复了不可达 gate 窗口、非确定性 fingerprint、复合 AST 单位传播
和嵌套动态指标丢失四项缺陷。不得把这次“机制闭环能力”验收扩大解释为 locomotion
checkpoint 的平地、楼梯、DR 或实机能力验收。

下文保留 2026-08-15 的只读审计阶段记录，作为实现演进证据，不再代表当前完成度。

## 当前目标

把同目录的 `rl_agent_architecture_proposal.md` 落地为可验证的研究基础设施。当前先完成只读审计闭环，
不自动改奖励、不自动启动训练、不连接远端。

## 本轮实现范围

- `autotuner/locomotion_console/research_audit.py`
  - `RuntimeExecutionProof`
  - `RewardAuditRecord`
  - `MetricAlignmentContract`
  - `GateCalibrationRecord`
  - `CommandCoverageLedger`
  - `OptimizationStateSnapshot`
  - `CheckpointCapabilityRegistry`
  - `ResearchAuditReport`
- `python -m autotuner.locomotion_console.research_audit`
  - 默认只读本地仓库
  - `--manifest` 接收运行时证据
  - `--checkpoint-root` 登记本地 checkpoint，但没有诊断证据时保持 `unknown`
  - `--strict` 在 P0 证据不完整时返回 2
- Agent 只读工具：`get_research_audit`
- 示例 manifest：`config/research_audit.manifest.example.json`
- 单测：`tests/autotuner/locomotion_console/test_research_audit.py`

## 当前完成度（2026-08-15）

已完成并验证：

- 本地静态源文件 hash、关键 symbol AST 存在性和历史同名模块风险扫描；
- 奖励项结构从 `taili_reward.py` AST 推导，不手抄 term/weight/gate；
- A1/A2/A3/B1/B2/B3/B4/D 的训练信号与 scorer 信号声明；
- 课程 gate、命令覆盖、优化器/resume 状态、checkpoint 能力证据的 manifest schema；
- Agent 只读工具 `get_research_audit` 与意图路由；
- 严格模式的阻断语义；缺证据不报告 ready。

验证结果：

- `python -m pytest tests/autotuner/locomotion_console/test_research_audit.py -q`：5 passed；
- `python -m pytest tests/autotuner/locomotion_console/test_knowledge_model.py tests/autotuner/locomotion_console/test_agent_synthesis.py -q`：50 passed；
- `python -m autotuner.locomotion_console.research_audit --root . --strict --output output/research_audit_20260815.json`：返回 2，报告状态为 `incomplete`，这是预期的 P0 证据阻断，不是实现错误。

当前严格报告明确发现：本地源码结构可读，但没有 runtime manifest，因此 metric alignment、gate calibration、command coverage 和 PPO/resume optimization state 尚未证明；历史 `strategy_backups` 中存在同名 reward/env/config 文件，远端 import 仍需执行证明。

## 不可误读的状态

当前代码只证明本地源结构和静态奖励关系，不能证明：

- 远端 payload 使用了相同源码；
- 课程 gate 可达；
- PPO/resume 状态完整恢复；
- 奖励在失败区保留梯度；
- 某 checkpoint 具备平地、楼梯、站立或 DR 能力。

因此没有 manifest 时，报告应为 `incomplete`，这不是失败训练的结论，而是禁止把缺失证据当成完成。

## 下一步

1. 将真实训练启动器输出的 effective config、payload hash、optimizer/scheduler/normalizer/log_std/AMP/
   curriculum/RNG 状态整理为 manifest 生成器。
2. 将诊断 raw trace 和 scorer 统计写入 metric alignment manifest，完成 A1/A2/A3/B1/B2/B3/B4/D 的同量对齐。
3. 将课程实际有效样本和已知可行策略校准结果写入 gate calibration。
4. 将 `RuntimeExecutionProof` 接到远端启动 preflight；未证明时禁止长训和自动干预。

## 验证记录

任何上下文压缩恢复都必须先读取本文件、审计报告、`git status` 和 `output/research_audit_20260815.json`，再继续。

## 下一步的唯一主线

不要继续添加抽象。应先实现真实训练启动器的 manifest 生成：在 payload 启动时记录 effective config、payload/source hash、
resolved import path、policy/value/optimizer/scheduler/normalizer/log_std/AMP/curriculum/RNG 状态；在诊断完成时追加
raw trace、metric alignment、gate calibration、command coverage 和 checkpoint capability evidence。只有该 manifest
能使严格审计达到 `ready`，才继续把 preflight 接到远程启动和自动干预。

## Continuation record (2026-08-15)

Implemented after the first audit slice:

- `autotuner/locomotion_console/research_ledger.py`: versioned research objects, cross-field validation, runtime-manifest import, append-only JSONL events, and a tamper-evident hash chain.
- `autotuner/locomotion_console/policy_contract.py`: resolved deployment contract construction and IsaacLab/MuJoCo/adapter parity comparison.
- `autotuner/locomotion_console/research_supervisor.py`: lifecycle transition guards, evidence-cost policy, and protected-capability experiment readiness checks.
- `get_research_ledger` and `GET /research/ledger`: local read-only ledger access. No SSH, training, reward edit, checkpoint deletion, or remote mutation.
- `tools/export_taili_deployment.py` now writes the shared `policy_contract` into deployment metadata.
- Runtime manifests now carry contract and distribution references when supplied through environment variables.

Verification completed in this continuation:

- Relevant blind-locomotion and locomotion-console suites: 248 passed.
- New ledger, policy-contract, and supervisor tests: 12 passed.
- `launch_taili_train --dry-run`: passed; no IsaacLab or GPU training started.
- Strict audit: exit 2, status `incomplete`. Remaining blockers are real missing evidence: runtime command coverage, gate calibration, and complete PPO/resume state. This is intentional; no `ready` claim is allowed without a real run manifest and aligned evaluation artifacts.

Next evidence-dependent boundary:

- Run the payload-owned training entry once in the intended runtime so it emits a real manifest, telemetry, optimization snapshot, and checkpoint inventory.
- Assemble scorer/diagnostic evidence and gate calibration from that run; then re-run strict audit.
- Keep `incomplete` until the report has real runtime evidence and all required protected-capability checks pass.

## Final verification slice (2026-08-15)

- The runtime proof order in `autotuner/blind_locomotion/train_taili.py` is now explicit: create the IsaacLab environment and SKRL `Runner` first, then capture the modules and source paths actually loaded by that process. The module alias for `python -m taili_blind_runtime.train_taili` prevents the entry module from being imported a second time during proof collection.
- `python -m pytest tests/autotuner/blind_locomotion tests/autotuner/locomotion_console/test_research_audit.py tests/autotuner/locomotion_console/test_research_evidence.py tests/autotuner/locomotion_console/test_research_ledger.py tests/autotuner/locomotion_console/test_policy_contract.py tests/autotuner/locomotion_console/test_research_supervisor.py --basetemp .pytest-tmp-run2 -rA`: `118 passed`.
- `python -m py_compile` passed for the changed runtime, audit, ledger, contract, evidence, and supervisor modules.
- `git diff --check` passed.
- `launch_taili_train --dry-run` passed and emitted the planned runtime manifest path; it did not start IsaacLab, GPU training, SSH, or remote mutation.
- The latest strict report is `output/research_audit_20260815_continue_final.json`; it intentionally returns exit `2` with status `incomplete`. The remaining blockers are evidence-dependent: real runtime command coverage, scorer/diagnostic metric alignment, calibrated curriculum gates, and complete PPO/resume state.
- The current local Python environment has PyTorch but does not have `isaaclab`, `isaaclab_tasks`, `isaaclab_rl`, or `skrl`; a real payload-owned training run must therefore be performed in the intended IsaacLab runtime, not simulated from this workspace.

## Remote runtime evidence (2026-08-15)

- The training host `183.147.142.40:31376` was reachable and had an idle RTX 4090. No pre-existing training process was reused or terminated.
- Payload archive `taili_blind_runtime_20260815_1445.tar.gz` was built from the validated local manifest and uploaded to a unique data-disk directory. Remote and local SHA256 both equal `630af9562bb94d881726a74f30ade371d29ecda572c4a63179232b14eb694a35`.
- Run `taili_agent_manifest_smoke_20260815_1450` used the payload-owned `taili_blind_runtime.train_taili`, IsaacLab, SKRL, and 64 environments for 4 steps. It produced `runtime_manifest.json`, `runtime_preflight.json`, `optimization_state.json`, `train.telemetry.jsonl`, and effective configuration artifacts without loading or writing a historical checkpoint.
- The real manifest records `runtime_execution=proven`, fresh optimization state `proven`, resume edge `proven` for fresh initialization, and runtime preflight `pass`. The telemetry was assembled into `output/runtime_evidence/taili_agent_manifest_smoke_20260815_1450/assembled_manifest.json` and imported into the local research ledger; the ledger chain verified with two events.
- The remote smoke proves the runtime evidence path only. Its 4 samples cover `forward` but not all required command buckets, and it has no physical scorer, gate calibration, or checkpoint capability evidence. The corresponding strict report is `output/research_audit_20260815_remote_smoke.json`, status `incomplete`, with `0/8` metric alignment, `0/22` gate calibration, `1` command bucket, and optimization state `proven`.
- Do not mark the research contract `ready` or claim flat, stair, standing, DR, or resume capability from this smoke. The next evidence boundary is a sufficiently long, deliberately covered runtime/evaluation window plus scorer artifacts and calibrated gates; the fresh 4-step smoke is retained only as runtime provenance.

## 修复后的第二轮远程证据（2026-08-15）

- 第一轮 `taili_agent_manifest_smoke_20260815_1450` 暴露出真实缺陷：后续 manifest 更新对嵌套 `run` 段执行了浅覆盖，导致 `run_id` 和 `task` 丢失。该运行只保留为缺陷反例，不作为有效运行证据。
- `runtime_manifest.update_manifest()` 已改为对受控嵌套 section 合并，并新增回归测试 `test_manifest_updates_preserve_nested_launch_facts`。
- 修复后的 payload 为 `taili_blind_runtime_20260815_1500.tar.gz`，SHA256 为 `af8dd424e7c0d2cbbc934f042d8670c6df509f26e535975fb2566ca6c1dd3161`。
- 第二轮运行 `taili_agent_manifest_smoke_20260815_1505` 使用 32 个环境、2 个训练步；`run_id/task` 保留正确，`runtime_execution=proven`、`optimization_state=proven`、`runtime_preflight=pass`，没有加载或生成 checkpoint。
- 远程原始证据已下载到 `output/runtime_evidence/taili_agent_manifest_smoke_20260815_1505/`，并组装为同目录的 `assembled_manifest.json`。
- `research_ledger.py` 的 runtime manifest 导入现在会记录运行快照、遥测证据、源码审计证据和训练分布；同一 manifest 重复导入不会重复追加事件。第二轮首次导入新增 4 个事件，重复导入新增 0 个事件，当前 ledger 共 6 个事件且 hash chain 验证通过。
- 对应严格审计报告为 `output/research_audit_20260815_remote_smoke_1505.json`，状态仍为 `incomplete`：`runtime_execution=proven`、`optimization_state=proven`，但 metric alignment 为 `0/8`、gate calibration 为 `0/22`、命令覆盖仅 1 个 bucket，checkpoint capability 仍为空。
- 定向 runtime/ledger 测试共 13 项通过；随后全仓 `pytest`、`python -m compileall -q autotuner tools` 和 `git diff --check` 均通过。

第二轮只证明修复后的真实运行证据链和 ledger 导入语义。它仍不能证明平地、楼梯、严格站立、DR、sim2sim 或 resume 后能力，因此不能把审计状态提升为 `ready`。

## 命令覆盖证据修复（2026-08-15）

- 发现旧证据组装器使用整批环境的均值命令划分 bucket。该口径会把实际同时存在的前进、后退、横移、yaw 和站立样本压成一个 `forward` 样本，因此不能证明训练分布。
- `telemetry_payloads.py` 现在输出互斥的批次级 `forward/backward/lateral/yaw/stand/mixed` target、applied 和 eligible 环境计数；`research_evidence.py` 优先使用这些真实计数，不再从均值命令推断批次分布。
- `research_audit.py` 现在要求每个必需 bucket 的 `target_count`、`applied_count` 和 `eligible_count` 都大于 0；仅存在一个零样本 bucket 名不能通过覆盖审计。
- payload `taili_blind_runtime_20260815_152113.tar.gz` 的本地与远端 SHA256 均为 `3d2b403cc9b7f5883e04d4706bf90160b60ceea8e21f24e3db8c5eda26712ab5`。
- phase 0 对照运行 `taili_agent_command_coverage_smoke_20260815_1522` 每步实际分布为 `forward=32, backward=10, lateral=9, yaw=9, stand=4, mixed=0`。审计正确保持 `commands.coverage_missing: mixed`。
- phase 1 覆盖运行 `taili_agent_command_coverage_phase1_smoke_20260815_1530` 只把 `init_phase` 改为 1；配置 SHA256 为 `412611c22a3b1a8014433117b3c8eed04b603f44ea4471a0de2ad311b92c930a`。每步实际分布为 `forward=25, backward=2, lateral=6, yaw=5, stand=7, mixed=19`，target/applied/eligible 完全一致，命令覆盖阻断被真实解除。
- 两轮均为 64 环境、2 步、fresh、无 checkpoint 的 bounded smoke。phase 1 严格审计仍为 `incomplete`，当前唯一 error 是课程 gate 尚未校准；metric alignment 仍为 `0/8`，因此不得将命令覆盖证明解释为任何运动能力证明。
- 两轮证据均已导入 ledger；各自首次导入新增 4 个事件，重复导入为 0。ledger 当前 14 个事件，hash chain 验证通过。
- 命令覆盖、证据组装、审计和 ledger 定向测试共 44 项通过。

This verification slice proves the local implementation and its conservative blocking behavior only. It does not prove that a policy has flat-ground, stair, standing, or domain-randomization capability. A real payload-owned run and aligned evaluation artifacts are still required before `ready` is valid.
