# RL 研究 Agent 活系统运维说明

## 1. 系统现在“活”在哪里

本系统不是固定权重白名单，也不是只读审计器。它已经具备以下闭环：

`问题 -> 证据与竞争假设 -> 合成新奖励/指标/门控 -> 静态与数值验证 -> 有界实验 -> 独立验收 -> 晋升/回滚 -> 结果学习与防重复`

可演化对象：

- 奖励和补充惩罚；
- 训练指标；
- 课程门控；
- 机制参数、课程、训练分布与诊断场景；
- 对已有机制的新增、替换和删除候选。

固定内核：

- 人工批准的目标与验收契约；
- 原始事实、证据引用和历史资产；
- 权限边界与批准记录；
- 独立 evaluator；
- protected capabilities；
- checkpoint、payload 和诊断证据的保护策略。

生成机制使用受限表达式 AST。系统不会执行 LLM 生成的任意 Python，也不能通过修改 evaluator 或验收目标让候选通过。

## 2. 持久目录

默认根目录是 `output/research`。相对路径必须仍位于本仓库内；部署或测试隔离可通过 `LOCOMOTION_RESEARCH_ROOT` 指定绝对路径，但不能指定文件系统根目录。

| 路径 | 内容 |
|---|---|
| `output/research/state` | 当前研究状态、revision、哈希事件链与崩溃恢复记录 |
| `output/research/ledger` | 契约、证据、假设、候选、计划、结果和知识声明 |
| `output/research/cycles` | 每个周期的 proposal、编译 artifact 和执行声明 |
| `output/research/experiments` | 隔离实验 workspace、日志、独立评估结果和当前晋升指针 |

同一个 `cycle_id + plan_id` 只能执行一次。执行开始前会原子创建 receipt；即使进程中断，也不会在恢复后静默重跑。需要重试时必须登记新的计划 ID。

## 3. Agent 能力

只读工具：

- `get_research_state`：读取当前问题、假设、保护能力、待处理候选和 CAS revision；
- `get_research_ledger`：读取并校验追加式研究谱系；
- `validate_research_mechanism`：执行 schema、单位、信号可用性、特权依赖、梯度、反作弊和 evaluator 保护验证；
- `run_decision_replay`：在不泄露后验结果的前提下运行历史决策盲回放；
- `get_research_audit`：核对源码、运行 manifest、训练/验收语义和部署证据。

动作：

- `propose_research_cycle {synthesis, expected_state_revision}`：合成、验证、编译并记录候选，不批准实验；
- `execute_research_cycle {cycle_id, plan_id}`：只执行已登记且已批准的计划。

执行动作严格只接受 `cycle_id` 和 `plan_id`。命令、环境变量、evaluator 或机制正文不能在确认时内联替换。控制台的提案确认同时绑定动作名和完整参数，不能用一个已确认提案执行另一组参数。

## 4. API 顺序

1. `POST /research/state/initialize`：首次建立目标、保护能力、当前基线和问题状态。
2. `GET /research/state`：每次恢复上下文后先读取，并核对 revision。
3. `POST /research/state/update`：用 `expected_revision` 做 CAS 更新；陈旧上下文返回冲突。
4. `POST /research/mechanisms/validate`：可在提案前单独验证 bundle。
5. `POST /research/cycles/propose`：提交 `SynthesisRequest`，生成候选和不可变 artifact。
6. 人工审查候选、训练窗口、独立 evaluator、保护能力、成功与回滚条件。
7. `POST /research/plans/register`：只接受 `status=approved` 且有 `authorization_ref` 的 `ExperimentPlan`。
8. `POST /research/cycles/execute`：请求体只能含 `cycle_id`、`plan_id`。
9. `GET /research/ledger`：检查结果、知识更新和晋升/回滚记录。
10. `GET /research/replay?split=holdout`：定期检查决策策略是否退化。

所有状态修改路由仍受控制台 mutation token 保护。计划登记接口不暴露为 LLM 动作，因此 LLM 不能自我批准实验。

## 5. 实验计划约束

训练和 evaluator 命令必须是参数数组，不是 shell 字符串：

```json
{
  "training_window": {
    "command": ["python", "-m", "taili_blind_runtime.train_taili", "--config", "approved.json"],
    "max_seconds": 7200
  },
  "evaluation_plan": {
    "command": ["python", "evaluate_candidate.py"]
  }
}
```

独立 evaluator 的 stdout 必须是 JSON 对象，至少包含：

```json
{
  "success": true,
  "evidence_complete": true,
  "protected": {
    "flat": {"passed": true},
    "stairs": {"passed": true}
  }
}
```

任何保护能力缺失或失败都会回滚。只有 `success=true`、证据完整且所有保护能力通过时才晋升。

候选复制到隔离 workspace 后，supervisor 自动设置：

- `RL_RESEARCH_EXPERIMENT_REF`；
- `RL_RESEARCH_CANDIDATE_ROOT`；
- `TAILI_MECHANISM_BUNDLE=<candidate>/mechanisms.json`。

因此生成的新奖励、指标和门控会真正进入 Taili 的 `compute_reward_components()`，不依赖计划作者手工接线。

## 6. 本地与远程执行

配置项：

```text
LOCOMOTION_RESEARCH_BACKEND=local|ssh
LOCOMOTION_RESEARCH_REMOTE_ROOT=/root/gpufree-data/research_experiments
```

当控制台 source 为 `real` 时默认使用 SSH backend；source 为 `fake` 时默认使用本地 backend。

本地 backend：

- 禁用 shell 解释，直接执行批准的 argv；
- stdout/stderr 持续写入 workspace 的 `training.log`，不会因无人读取管道而卡死；
- evaluator 与候选训练进程分离执行。

SSH backend：

- 使用现有 RemoteSSH 主机密钥策略与凭据配置；
- 上传候选 artifact 和批准计划；
- 由系统生成 runner，并逐项 shell quote argv；
- 用独立 PID、进程组、退出码文件和日志管理启动、轮询、超时与终止；
- 只转发 `CUDA_VISIBLE_DEVICES`、`TAILI_*` 和系统定义的研究环境；
- 将训练和 evaluator 日志同步回本地 workspace。

远程密码不会写入研究目录、计划、日志或 API 响应。

## 7. 资源与并行规则

实验计划通过 `resource_budget.resources` 声明资源，例如 `gpu:0`。租约存储支持多个互不重叠的活动租约，但任何重叠资源都会被拒绝；过期租约会在下一次操作时清理。

本项目验证规则仍是严格串行：

- 不并行启动多个 `pytest`；
- 不并行启动多个 PyTorch、IsaacLab 或本地诊断进程；
- 训练与独立诊断是否可并行，必须由明确资源租约决定，不能靠约定猜测。

## 8. 故障与恢复

- 状态写入使用 revision/CAS、跨进程锁、原子替换和哈希事件链；快照中断时从事件链恢复。
- 候选绑定 baseline fingerprint；基线变化后旧候选不能执行。
- 编译 artifact 带逐文件哈希，运行前可重新验证。
- 训练失败、超时、evaluator 失败或保护能力回归均不删除 checkpoint、payload 或候选 artifact。
- 晋升仅更新受管 `ACTIVE_MECHANISM.json` 指针；回滚不改写历史资产。
- 历史已拒绝 fingerprint 写入 no-repeat，避免上下文压缩后重复同一失败机制。
- SSH 不可达时保留 execution receipt 和本地证据，不伪造完成状态；重试使用新批准计划。

## 9. 验证命令

Windows 上使用仓库内短路径作为 pytest 临时根，并始终串行运行：

```powershell
New-Item -ItemType Directory -Path .pt -Force | Out-Null
python -m pytest tests/autotuner/locomotion_console/test_research_agent_api.py -q --basetemp .pt/agent
python -m pytest tests/autotuner/locomotion_console/test_research_cycle.py -q --basetemp .pt/cycle
python -m pytest tests/autotuner/locomotion_console/test_research_execution.py -q --basetemp .pt/exec
python -m pytest tests/autotuner/locomotion_console/test_research_remote.py -q --basetemp .pt/ssh
```

不要同时运行这些命令。完整回归也只启动一个 pytest 进程。

## 10. 真实远程机验证

2026-08-17 在实际训练机上完成了 CPU-only SSH 闭环验证。该验证不启动
PyTorch、IsaacLab 或正式训练，也不读取、覆盖或删除 checkpoint 和 payload。

第一次执行真实暴露了 Windows 到 Linux 的行尾问题：本地生成的 Bash runner
被写成 CRLF，训练命令虽然偶然执行，但远端无法可靠写入 `training.exit`，因此
supervisor 将实验回滚。`SSHExperimentBackend._write_local()` 随后改为直接写入
UTF-8 字节，确保远端脚本始终使用 LF，并增加了对应回归断言。

修复后使用新的批准计划重新执行，实际验证结果为：

- 候选 `mechanisms.json` 成功上传，训练进程通过远端
  `TAILI_MECHANISM_BUNDLE` 读取并解析；
- 本地与远端候选 SHA-256 一致；
- 独立训练 PID 正常退出，验证后不再存活；
- `training.exit=0`，`evaluation.exit=0`；
- 训练日志回传成功，独立 evaluator 返回完整 JSON；
- protected capability `smoke` 通过；
- supervisor 最终判定为 `promote`，并更新受管活动指针；
- 远端 runner 经字节检查确认不含 CRLF。

本地可审计证据保存在
`output/research_real_machine_smoke/result-20260817T045445Z.json`，隔离 workspace
和首次失败现场均保留在 `output/research_real_machine_smoke/execution/runs/`。远端
证据位于 `/root/gpufree-data/research_experiments/smoke/`。这些目录不包含 SSH
密码。

## 11. 真实 GPU 能力验收

CPU-only SSH smoke 只能证明执行器连通，不能证明系统生成的机制进入实际强化学习
运行时。2026-08-17 又执行了一次隔离的真实 GPU 研究周期：

- 当前源码重新构建为 105 文件的不可变 payload；本地与远端 SHA-256 均为
  `22735cb87971df2ab4bb5a6c84549935d6aa23be7a853be4d1d2b619dad6afbd`；
- `ResearchCycleManager` 从两个竞争候选中选择更简洁的
  `runtime-observability:drive`，而不是带额外 guard 的候选；
- 计划经过登记后，由 SSH backend 在 RTX 4090 上启动真实 IsaacLab、SKRL 和
  Taili 生产奖励路径；运行使用 16 个环境、fresh 初始化和 70 个有界 step；
- 新增 `dynamic/reward:runtime-observability:drive` 每步进入总奖励；
- 新增 `metric:runtime-observability` 保留真实数值并进入结构化遥测；
- 新增 `gate:runtime-observability` 使用 64 样本窗口，在末步达到
  `ready=true`、`passed=true`、`streak=7`；
- runtime manifest 为 complete，runtime execution、preflight 和 optimization
  state 均为 proven/pass；
- 独立 evaluator 的所有检查通过后，supervisor 才返回 `promote`；研究 case
  更新为 `resolved`；
- 同一 `cycle_id + plan_id` 的第二次执行被持久 receipt 阻断；
- runtime manifest 首次导入账本新增 3 个事件，重复导入新增 0 个事件，最终
  10 个事件的哈希链验证通过；
- 运行未加载历史 checkpoint，也没有生成 checkpoint；结束后无 Taili 进程，
  GPU 占用回落到 66 MiB。

真实运行同时暴露并修复了以下本地测试未充分覆盖的问题：

1. 合成 gate 要求 64 个样本，但 metric 历史窗口默认只有 1，导致 gate 永远
   不可达。现在合成器使用一致的 64 步窗口，schema 也拒绝所有同类配置。
2. bundle fingerprint 曾受 `set` 迭代顺序影响。现在所有无序集合先规范排序，
   再计算哈希。
3. 复合 AST 的单位传播误把单位集合当成语义对象。现在高层合成表达式可以完成
   静态和数值验证。
4. 遥测 emitter 曾把嵌套 `dynamic_metrics` 强制转换为 `0.0`。现在 JSONL 保留
   结构化指标，TensorBoard 使用分层 key 记录。

通过证据位于
`output/research_real_gpu_capability/sessions/20260817T060209Z/REAL_GPU_CAPABILITY_RESULT.json`，
完整远端运行产物位于同目录的 `remote_run_evidence/`。此前 evaluator 因遥测契约
失败而回滚的运行保留在 `sessions/20260817T055617Z/`，不得改写为成功。

这次验收证明的是系统能够真实生成、验证、执行、观察、晋升和学习一个新机制。
它不等价于证明某个 locomotion checkpoint 已满足平地、楼梯、DR 或实机运动学
要求；这类能力仍必须由对应的真实训练和独立物理诊断验收。
