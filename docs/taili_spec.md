# Taili Blind Locomotion — Acceptance Specification (`taili_spec.md`)

## §0 — 当前优化目标（2026-07-21）

当前最高优先级由三项不可互相交换的并行目标组成：**运动到零命令后的自然、严格、安静站立**、
**极致优化平地四方向的完整运动质量**、**突破并完善以前进为主的上下楼梯能力**。
三项共同恢复并达到较高质量后，重点推进**在不交换前三项能力的前提下形成强鲁棒性**；
多方向地形是其后的低优先级扩展。
本节描述的是持续优化方向，不应被机械地改造成训练早期的一次性硬门。后文仍保留较宽的历史验收电池，
但斜坡、粗糙地面和 boxes 仍不参与本阶段策略优劣判断，也不能代替楼梯能力。
DR 不是替代前三项的得分项；质量、执行器、摩擦、质心、传感器、外力和延迟扰动应在前三项形成后逐级进入训练，
最终覆盖 §3.5 的 E5 范围。任何 DR 得分或抗扰结果都不能掩盖四方向跟踪、步态、核心稳定、轻脚、
对称或上下楼梯能力退化。

### 理想平地状态

- 平地不存在“前进优先”：前进、后退、横移和 yaw 的采样、驱动与质量要求应保持均衡，任何方向都不能靠牺牲另一方向达标。
- 平地是完整物理质量目标，不等同于方向跟踪分数；跟踪、核心、机身高度、步态、支撑、duty、对称、髋和小腿几何、足端轨迹、净空、触地、滑移以及动作/力矩平滑必须共同评价。
- 站立以及前进、后退、横移、yaw 都能准确、持续地跟踪机体坐标系命令，长时间运行不漂移、不中断、不失稳。
- 前进、后退和横移时同时稳定 roll、pitch、yaw 与机身高度；yaw 时准确跟踪 yaw rate，同时严格稳定 roll、pitch 与高度。
- 核心既没有高频颤抖，也没有低频摇摆、俯仰和上下起伏；低中速不依赖明显下蹲换取稳定，视觉上接近“漂浮”。
- 步态左右、前后和对角关系协调，支撑时序完整；直线运动时髋关节没有不必要的外翻或内翻，小腿轨迹尽量位于合理运动平面。
- 足端位置、速度和加速度轨迹连续自然，抬脚高度充分但不过度；落地前垂直与水平速度较小，接触冲击低，落地后又能迅速建立充足支撑。
- 支撑脚低滑移，摆动腿不颤抖；横移不会因前后腿失配而斜移或转圈，yaw 的步幅、步频、承重和动作形态规整。

### 理想楼梯状态

- 楼梯训练以前进通过为最高要求和主要样本/梯度来源；后退、横移和 yaw 只保留一定覆盖，用于方向修正、鲁棒性和防止共享 Actor 退化，不能稀释前进换层主驱动。
- 上楼和下楼分别训练、分别评价，首先沿机体坐标系命令方向产生持续推进和真实换层；允许降低速度大小，但不能丢失方向。
- 策略有足够的抬脚、前送或受控探低能力，也有足够的推进力和支撑力；不能依靠原地抬腿、绕行、自由落体或课程记账获得能力。
- 接近和通过楼梯时尽量不磕碰、不滑移、不重落脚，不出现触阶即垮、四脚失去支撑、侧翻或 reset。
- 允许完成换层所必需的机身高度和姿态变化，但角速度、角加速度、冲击及多余摇晃仍应很小；通过过程稳定、干净、连贯且通过率高。
- 最终目标是稳定通过约 `25–30 cm` 的上、下楼梯；低等级课程只证明探索已开始，不能作为最终能力证明。

### 理想鲁棒状态

- 在质量、Kp/Kd、摩擦、质心、IMU 偏置、控制延迟和外力推扰变化下，仍保持平地四方向与上下楼梯的任务能力和动作质量。
- 扰动期间允许短时误差，但不能摔倒、持续打滑、力矩饱和或依靠长期下蹲维持；扰动结束后应快速恢复原命令跟踪和稳定姿态。
- DR 逐级增加难度，每一级都必须实际触发对应扰动，并在完整统计窗口内重新证明跟踪、步态、支撑、核心、轻脚和楼梯能力后才能升级。
- 最终强度以 §3.5 的 E5 目标为准；训练聚合值不能替代无扰动基线诊断和各扰动通道的独立验收。

### 优化与验证原则

- 安静站立、完整平地质量和前进上下楼梯为同一最高优先级；不得用先修其中一项、长期牺牲另外两项的串行路线冒充全局优化。
- 从当前已经形成的平地和低级楼梯能力继续演进，不因一个局部问题推倒整个奖励路径。
- 正向任务奖励负责提供“不得不朝目标运动”的主驱动；惩罚只补充安全、质量和防钻空子，不能把站立变成比运动更优的局部解。
- 不恢复人工楼梯状态机。Actor 只使用真机可获得的本体与接触历史；训练期奖励和 Critic 可以使用特权物理量，但部署不依赖它们。
- 日志用于判断学习趋势和采样覆盖，诊断物理数据与视频用于判断真实动作，实际配置用于追溯数学原因；三者不一致时先修语义，不凭单一指标调参。
- 楼梯进步必须同时检查平地是否退化；平地分数改善也不能掩盖楼梯长期停滞。每次干预只在证据表明停滞、退化或结构矛盾后进行。
- 进入 DR 后仍须并行观察站立、平地和楼梯；短暂适应下降可以等待完整窗口，但连续窗口无恢复斜率时应先修正 DR 课程或扰动通道，不能用修改运动奖励掩盖课程跳变。

> **Status.** This document is the single source of truth for what "Taili passes" means.
> It is the human-readable contract; the machine-enforced contract is
> `products/taili/blind_locomotion/acceptance_score.py` (pure, unit-testable, no sim). Every
> threshold below is the exact value that scorer enforces — when the two disagree, the scorer
> wins and this document is the bug. The scorer is what `physeval_blind.py` /
> `physeval_blind_e.py` feed measured statistics into, and what `acceptance_aggregate.py`
> merges into the final §2 verdict.
>
> Scope: a **blind** quadruped (`Taili`, ~39 kg, 12 actuated joints, weakest joint = thigh at
> 110 N·m) trained with AMP + PPO on IsaacLab/skrl and deployed with **no exteroception**.

---

## §1 — Measurement口径 (evaluation protocol)

All acceptance numbers are measured under the **deployment口径**, not the training口径:

1. **Blind.** The deployed policy sees only proprioception + command + the learned terrain
   latent; no height-scan / camera is available at deploy time. Acceptance is measured on the
   same observation the robot will actually have (`taili_spec §1`).
2. **Mean-action.** Evaluation rolls out the policy **mean action** (no exploration noise) —
   this is what runs on hardware. `physeval_blind.py` sets the policy to mean-action mode.
3. **Off-sim / off-policy scoring.** The simulator measures raw physical quantities
   (velocities, foot states, torques); the pass/fail thresholds live in `acceptance_score.py`
   so the "judge" is verifiable independently of the simulator.
4. **Statistic definitions.** Unless stated otherwise: `median` and `p90` are per-command-bucket
   order statistics of the per-step error; `p95` is the per-touchdown-event order statistic;
   slip / clearance are measured only on **settled** contact frames (touchdown and liftoff
   transition frames are excluded from slip; swing frames only for clearance).
5. **Non-strict bounds.** Every comparison uses the spec's `<=` / `>=` exactly;
   threshold-equal ⇒ **PASS**.

---

## §2 — 最终通过条件 (final acceptance)

A policy **PASSES** the benchmark iff **every hard-gate family is present AND ok** across the
**merged** results of all physeval runs (flat + each terrain + push + DR). A required family
that was **never evaluated** counts as **NOT passed** — coverage is never silently skipped
(`final_verdict()` in `acceptance_score.py`).

- **Hard families (all must pass):**
  `A1, A2, A3, A4, B1, B2, B3, B4, C, D1, D2, D3, D4, E1, E2, E3, E4, E5, F2`
- **Soft families (report-only, never gating):** `A5, B5, F1, F3`

The `[label]` suffix (e.g. `A1[fwd05]`) denotes a per-bucket sub-check; a family passes only if
**all** of its buckets pass.

---

## §3 — Acceptance battery

### §3.1 A — Command tracking & standing

| Gate | Name | Threshold (median **and** p90 unless noted) |
|------|------|---------------------------------------------|
| **A1** | Linear velocity tracking | `|v − cmd| ≤ max(0.10, 0.15·|cmd|)` m/s, for **both** median and p90, in every forward/back/lateral bucket. |
| **A2** | Yaw-rate tracking | `|ω_z − cmd_z| ≤ 0.15` rad/s, for both median and p90, in every yaw bucket. |
| **A3** | Clean stop / settle (立正) | `|v| ≤ 0.05` m/s **and** `|ω_z| ≤ 0.05` rad/s **and** settle time `≤ 1.0` s **and** four-foot duty `≥ 0.95` **and** upright fraction `≥ 0.99`. |
| **A4** | Mixed commands | Each component (vx, vy, ω_z) simultaneously in its A1/A2 band — **no** sacrificing vy/ω_z to serve vx. |
| **A5** *(soft)* | Velocity envelope | Robot tracks up to the envelope: forward **1.5** m/s, backward **0.8** m/s, lateral **0.7** m/s, yaw **1.5** rad/s. |

### §3.2 B — Gait quality

| Gate | Name | Threshold |
|------|------|-----------|
| **B1** | No hard landing / impact | touchdown vertical speed `p95 ≤ 0.10` m/s on **flat**, `≤ 0.20` m/s on **terrain**. |
| **B2** | No stance slip | settled-stance foot slip speed `p90 ≤ 0.05` m/s on **flat**, `≤ 0.10` m/s on **terrain**. |
| **B3** | Foot clearance | **flat:** swing-peak clearance ≈ 0.08 m, graded pass band `0.05–0.15` m. **terrain:** peak `≥ obstacle_h + 0.04` m margin (obstacle-relative; report-only where local obstacle height is unavailable). |
| **B4** | Symmetry | In every symmetric scene, L/R duty difference `Δduty ≤ 0.05` **and** clearance difference `Δclr ≤ 0.01` m. Family passes only if all symmetric scenes pass. |
| **B5** *(soft)* | Smooth action / torque | Action-rate and torque time-series are regular (no chatter); reported, not gating. |

### §3.3 C — Standing posture & recovery (立正)

**C** passes iff `A3` passes **and** standing L/R symmetry `Δduty ≤ 0.05` **and** at-rest upright
fraction `≥ 0.99` **and** at-rest four-foot duty `≥ 0.95`. Push-recovery to a clean stand is
covered by the E battery.

### §3.4 D — Terrain traversal

Terrain is scored as **controlled progress**, not flat-band tracking. Families:
`slope (≤20°)`, `rough (≤0.12 m noise)`, `boxes (≤0.25 m)`, `stairs_up (≤0.25 m step)`,
`stairs_down (≤0.25 m step)`. `stairs_down` is treated as **more dangerous** than `stairs_up`.

| Gate | Threshold |
|------|-----------|
| **D1–D4** | forward speed `≥ 0.30` m/s **and** fall rate `= 0` (controlled progress at each difficulty before it is raised). |
| **D4 (rough) additionally** | base-height drop `< 0.05` m. |

### §3.5 E — Robustness / disturbance

| Gate | Condition | Threshold |
|------|-----------|-----------|
| **E1** | Walking push (≥1.0 m/s equiv.) | no fall **and** recover `≤ 1.0` s **and** tracking resumes **and** no torque saturation. |
| **E2** | Standing push (≥1.0 m/s) | no fall **and** recover `≤ 1.0` s **and** returns to a clean stand. |
| **E3** | Low friction (μ ≥ 0.4) | stand + walk stable, no sustained slipping, no fall. |
| **E4** | Mass randomization | all `DR_HARD_GATES` still pass under the mass-randomization condition. |
| **E5** | Full domain randomization | all `DR_HARD_GATES` still pass under full DR. |

`DR_HARD_GATES = A1, A2, A3, B1, B2, B3, B4, C, D, E1, E2, E3, F2`. E4/E5 re-evaluate these
gates *under the DR condition*; a gate not evaluated on that battery is reported missing, **not**
silently passed.

**DR ranges (E5 target):** friction `0.4–1.4`, added mass `−5 … +20` kg, CoM offset `±0.05` m,
Kp/Kd `±40%`, plus IMU gyro/gravity bias and latency (deploy-faithful, always-on from φ0).

### §3.6 — Terrain-mode relaxations

On terrain the A1/A2 tracking band is **relaxed** in favor of controlled traversal (this section
is the anchor cited by `acceptance_score.score_D` and `physeval_blind`): B1 relaxes `0.10 → 0.20`,
B2 relaxes `0.05 → 0.10`, B3 becomes obstacle-relative, and D replaces flat tracking with the
forward-speed-floor + no-fall progress test above.

### §3.7 F — Actuator margin & smoothness

| Gate | Name | Threshold |
|------|------|-----------|
| **F2** | Actuator margin | 99.5-percentile `|τ|/limit ≤ 0.85` (worst joint) **and** peak `|τ|/limit ≤ 1.0` **and** clamp rate `= 0`. |
| **F1, F3** *(soft)* | Efficiency / smoothness | reported, not gating. |

---

## §4 — Mapping to the training runtime

The training mechanisms that target each gate, and where each is measured, are enumerated in
`autotuner/locomotion_console/spec_coverage.py` (the coverage ledger the console/LLM read).
The reward/gate/curriculum design that implements these targets is documented in
[`taili_strategy_decisions.md`](./taili_strategy_decisions.md). The single editable strategy
contract is `products/taili/blind_locomotion/taili_blind_config.yaml`; all runtime artifacts
(`agent.skrl.yaml`, `effective_config.yaml`) are generated from it per run.
