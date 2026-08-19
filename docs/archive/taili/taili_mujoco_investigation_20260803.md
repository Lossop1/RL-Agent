# Taili MuJoCo Investigation - 2026-08-03

## Fixed Context

- Canonical MuJoCo project root: `/root/rl_sar`.
- Taili is integrated below the existing `rl_sar` tree.
- Deployed policy: `/root/rl_sar/policy/taili/taili_current/policy.pt`.
- Policy SHA256: `d0319ca270bd93939d95742b158ecd483acde9d00eabf0499a692a2d03175ddb`.
- Source checkpoint: training run `taili_train_20260802_flat35_dr3_post_move_capture_resume20k_long`, complete `agent_50000.pt`.
- The training host was manually powered off. Do not restart training implicitly.
- No policy, reward, training configuration, PD, or observation change has been made during this investigation.

## Confirmed Stop Failure

The user observed that a move-to-zero transition can either become quiet or
oscillate indefinitely depending on the gait state at the instant of stopping.

The policy contract uses a 0.74 second gait period at `vx=0.4 m/s` (37 policy
steps at 20 ms). A phase sweep was run from identical home states.

With the training-correct zero-command observation (all 8 gait-clock channels
masked to zero), all eight tested phases became quiet in 1.35-1.51 seconds.
Their final two-second body angular velocity and joint velocity were nearly
zero. Phase alone therefore does not make the trained policy fail when the
deployment observation is correct.

The C++ `rl_sar` implementation in
`src/rl_sar/library/core/rl_sdk/rl_sdk.cpp` stops advancing
`taili_gait_phase` at zero command but still sends the frozen nonzero sine and
cosine clock to the actor. Reproducing this behavior with the same Python
controller gives phase-dependent failure:

| Frozen phase | Quiet | Tail joint velocity RMS | Tail body wxy RMS |
|---|---:|---:|---:|
| 0.000 | yes | 0.0043 rad/s | 0.0022 rad/s |
| 0.243 | no | 0.1514 rad/s | 0.0750 rad/s |
| 0.486 | yes | 0.0031 rad/s | 0.0014 rad/s |
| 0.757 | no | 0.2082 rad/s | 0.0928 rad/s |

This exactly explains the visual phase dependence. The first required fix is
in the C++ observation contract: zero-command gait features must match
training. Do not compensate for this deployment bug by changing rewards.

Raw results are on the MuJoCo host under:

- `/root/rl_sar/diagnostics/taili_mujoco_20260803/stop_phase`
- `/root/rl_sar/diagnostics/taili_mujoco_20260803/stop_phase_cxx_clock`

### 部署侧修复状态

远端 `/root/rl_sar/src/rl_sar/library/core/rl_sdk/rl_sdk.cpp` 已改为与训练共用
同一个活动命令判定：

```text
norm(command_xy) > 0.1 || abs(command_yaw) > 0.05
```

命令不活动时，8 维 gait clock 全部输出零，而不是冻结非零相位。原文件备份为
`rl_sdk.cpp.pre_zero_clock_20260803`，本地补丁副本为
`tools/patches/rl_sdk.cpp.remote`。`rl_sdk` 编译和 `rl_sim_mujoco` 链接均已通过；
全量构建只在无关的硬件目标处因缺少 Unitree/Lite3 头文件失败。尚待使用修复后的
C++ 二进制完成多停止相位端到端复核。

### 严格单变量 gait-clock 复核

为排除早先两批 Python 控制器版本不同造成的混淆，现已在同一个
`taili_mujoco_long_horizon.py` 控制回路中加入仅供诊断的
`--standing-clock={zero,freeze,continue}`。固定同一 actor、contract、平地模型、
seed、初态、`vx=0.4 m/s`、6 秒停止窗口；使用 37 个策略步的运动周期，在
`0/5/9/14/18/23/28/32` 步相位停止。三种模式串行运行，共 24 个 trial。

逐相位比较时，停止前的 `mean_vx`、线速度 RMSE、roll RMS 和 pitch RMS 在三种
模式间的最大差值全部严格为 `0`，因此停止后的差异没有混入前段轨迹差异：

| 零命令时钟 | 安静相位 | 尾段 vxy RMS 中位数 | 尾段核心 wxy RMS 中位数 | 尾段关节速度 RMS 中位数 |
|---|---:|---:|---:|---:|
| `zero` | 8/8 | 0.000030 m/s | 0.000045 rad/s | 0.000100 rad/s |
| `freeze` | 4/8 | 0.018277 m/s | 0.028459 rad/s | 0.044651 rad/s |
| `continue` | 0/8 | 0.029727 m/s | 0.115453 rad/s | 0.584338 rad/s |

`zero` 的八个相位均在归零后 `1.37-1.53 s` 内达到安静判据；`freeze` 在
`p135/p243/p622/p757` 失败；`continue` 全部失败。这使因果结论成立：策略已经
学会从任意所测步态相位收敛到静止，视觉上的相位依赖持续晃动来自部署观测把
非零 gait clock 留给 actor，而不是训练奖励不足。原始结果位于远端
`/root/rl_sar/diagnostics/taili_mujoco_20260803/stop_clock_strict_matrix`，本地副本为
`output/stop_clock_strict_matrix`。

实际 `rl_sim_mujoco` 二进制随后通过原生 FSM 和 stdin 键盘路径串行执行四次
`0 -> 1 -> w*4 -> space`。实测停止相位约为
`0.159/0.403/0.646/0.889`；每次最后连续 8 个零命令状态采样中：

- gait phase 范围均严格为 `0`，说明停止后冻结内部相位；actor 收到的 8 维
  gait clock 则由上述 C++ 修复全部置零；
- 高度范围为 `0.061-0.088 mm`，upright 最小值为 `0.999799`；
- `vxy` 均值最大 `0.000397 m/s`，核心 `wxy` 均值最大 `0.000628 rad/s`；
- 12 关节速度 RMS 均值最大 `0.000518 rad/s`。

因此真实 C++ 部署闭环也已确认修复。状态日志只在已有每 200 物理步日志中补充
`vxy/wxy/joint_dq_rms/command/gait_phase`，未改变控制频率或控制数据。结果位于
远端 `/root/rl_sar/diagnostics/taili_mujoco_20260803/cxx_zero_clock_e2e`，本地副本为
`output/cxx_zero_clock_e2e`。

当前远端构建身份：

- `rl_sim_mujoco`: `a5e9ae1c3f60ed1578354b8f2171db5a9fed726cdc51c6a93c83e484f8e806d9`
- `rl_sdk.cpp`: `dacb1d35c14e25ca05f0aeb842829bcf389251344e9728d6c8e19ce4b4aeb7e9`
- `rl_sim_mujoco.cpp`: `374e9d6d0cb843cacb9fbf845b2aec0a1cff57119f48418b13d7222446d90ebc`

## 20 cm 楼梯实测结论

移动命令下的 gait clock 在 Python 与 C++ 中一致，因此静止观测错误不能解释
楼梯失败。接触探针 `tools/taili_mujoco_stair_probe.py` 复用原策略、观测历史、
动作延迟和 PD，只增加测量，不改变控制。

测试场景每级高 `0.20 m`、踏面深 `0.40 m`、共 5 级，未启用 DR。

### 上楼

- 只完成有效楼梯长度的 `15.7%`，最大 `x=1.538 m`；
- `5.34 s` 后翻倒，横向偏移 `0.623 m`，最大 yaw `33.2 deg`；
- 核心角速度 RMS/峰值为 `1.191 / 4.664 rad/s`；
- 足端滑移 RMS/p95/max 为 `0.248 / 0.377 / 2.313 m/s`；
- 非足端碰撞只占 `1.73%` 的仿真步，但 p95 接触力达 `1472 N`；
- 力矩饱和率仅 `1.69%`，不是电机能力耗尽。

平地段到 `x ~= 0.9 m` 基本正常。接触第一阶后，pitch 快速达到约
`19-26 deg`，前向速度由约 `0.24 m/s` 突增至 `0.69 m/s`，yaw 累积到
`20-33 deg`，随后支撑被破坏并侧翻。失败不是单纯净空不足，而是第一阶接触后
方向、核心和支撑同时失控。

### 下楼

- `9.86 s` 完成并落地，但横向偏移 `0.872 m`，最大 yaw `18.84 deg`；
- 核心角速度 RMS/峰值为 `0.527 / 2.030 rad/s`；
- 足端滑移 RMS/max 为 `0.123 / 2.799 m/s`；
- 存在 FR calf 碰撞，最大约 `1005 N`；
- 平均接触足数 `2.36`；
- 力矩饱和率仅 `0.136%`。

偏航和横向漂移从进入楼梯后持续累积，不是落地瞬间的偶发误差。下楼属于
“完成但质量差”，不能据完成率判定为优秀能力。

原始数据位于：

- `/root/rl_sar/diagnostics/taili_mujoco_20260803/stairs20`

### 优秀楼梯能力的训练表达

当前目标不是规定逐腿动作，而是让最优回报路径与完整、安全通过一致：

1. **主驱动**：沿机体系命令方向的有符号速度和实际进展持续给正奖励；达到命令
   速度后收益封顶，反向为负。楼梯允许欠速，但不能靠超速冲撞提高收益。
2. **连续质量**：稳定支撑、低滑移、干净触地、低动作高频和核心稳定决定能否领取
   满额地形信用；失稳时仍保留 25% 探索信用，避免策略在尚未学会越障时完全失去
   正梯度。
3. **直接安全成本**：足端立面碰撞以及 hip/thigh/calf/base 接触只产生即时成本，
   严重接触关闭满额任务信用；碰撞绝不再解锁抬脚、进展倍增或约束退让。
4. **课程验收**：必须同时满足真实净换高、整段沿程、横漂、航向、非终止、末态
   高度/直立/支撑/wxy 和受控速度。课程等级只是采样难度，不能代替 MuJoCo 整段
   通过验收。
5. **合理姿态**：允许楼梯所需 pitch，只约束 roll 和超过合理范围的 pitch；不使用
   状态机、换层势能、逐腿落点、探高或教师策略。

候选 `history7_flat35_dr3_stair_course_semantics_20260803` 已落实上述路径，同时保持
actor、AMP、PD、平地奖励、命令和 DR 不变。首轮训练先看低楼梯是否恢复完整
`course_ok`，再看 20 cm MuJoCo 整段通过及碰撞/滑移/核心指标。当前 yaw 漂移很可能
是首阶碰撞后失稳的下游结果；在移除碰撞捷径前不预先增加新的 yaw 机制。只有当
`height_ok` 与 `forward_ok` 已恢复而 `course_ok` 仍明确由 heading 单项卡住时，才调整
现有一个步态周期的 `heading_window`，不引入世界方向目标。

## 历史策略的同场景对照

部署前备份对应 2026-07-27 历史 7+ 系列 `agent_26000.pt`。在完全相同的
MuJoCo 20 cm、5 级楼梯和相同控制契约下：

- 上楼仍失败，最大 `x=1.114 m`、最大 yaw `62.7 deg`；
- 下楼能够完成，最终横向偏移仅 `0.164 m`，显著优于当前策略的 `0.872 m`。

结果文件为 `history7_july27_up20.json` 和 `history7_july27_down20.json`。这说明：

1. 当前上楼失败不能只归因于后续训练回退，历史 7+ actor 在该 MuJoCo 接触场景中
   也不能通过；
2. 当前 actor 的下楼方向保持发生了真实退化，但不能因此整体回滚，因为其他指标
   并非全部更差；
3. IsaacLab 中同体系较早的 `agent_44000.pt` 曾在显式 20 cm、DR2 诊断中完成
   `10/10`，前进约 `2.2-2.5 m`、换高约 `0.69-0.91 m`。因此还必须区分训练
   成功语义、共享 actor 演化和 PhysX 到 MuJoCo 的接触契约差异。

又对远端保留的两份更早部署备份做了完全相同的 20 cm、5 级楼梯测试：

- `policy.pt.before_20260727_170251_agent42000`：上楼失败，最大前进
  `1.415 m`，最大横漂 `1.021 m`，最大 yaw `38.65 deg`，非足端接触
  p95 `813 N`；下楼完成，但最终横漂 `0.722 m`，存在 FL/RR calf 碰撞。
- `policy.pt.before_20260723_1256_agent28000`：上楼失败，最大前进
  `1.269 m`，最大横漂 `1.154 m`，最大 yaw `30.82 deg`；下楼完成，最终
  横漂 `0.451 m`，存在 RL/RR calf 碰撞。

原始结果分别为 `history_agent42000_{up,down}20.json` 和
`history_agent28000_{up,down}20.json`。三组历史/当前 actor 在同一 MuJoCo
上都无法完成上楼，因此没有证据支持直接回滚某个历史 actor；历史策略只能用来
保留较好的平地和下楼结构。真正需要修复的是训练把“接近或跨过第一阶”计成课程
成功，以及碰阶后才产生的反应式抬脚信用。IsaacLab 的 `agent_44000.pt` 同 actor
跨引擎复核仍有价值，但不应阻塞训练语义修复。

## 已确认的课程与奖励语义漏洞

以下结论来自生成当前检查点的精确 payload
`history7_flat35_dr3_post_move_capture_20260802/taili_blind_runtime`，不是后来
变化的工作区：

- `rew_climb`、`rew_terrain_up/down`、`rew_terrain_support_transfer`、
  `rew_terrain_contact_quality` 和 `rew_terrain_event_collapse` 全部为 `0`；当前楼梯
  能力不是状态机式事件奖励产生的。
- 当前主路径仍是连续 tracking、机体系 direction 和 progress，这一基础应保留。
- `stair_course_ok` 已计算横向漂移和航向误差，但没有进入 `controlled_height` 或
  `move_up`，所以方向失控仍可升级。
- 楼梯升级只要求净换高达到单级高度的 `75%`，且前进仅 `0.30 m`。对于 20 cm
  楼梯，只获得约 15 cm 换高、接近第一阶就可能被计为成功；它没有表达完成整段
  楼梯和受控落地。
- `terrain_up_completion_scale` 实际由课程行高度上界计算，不是真实空间完成度；
  名称与数学语义不一致。
- `terrain_direction_quality_floor=0.50` 允许质量很差时仍领取至少一半方向信用，
  削弱了方向主驱动与安全质量的联立关系。
- 奖励输入中的 `body_collision` 当前固定为零，calf/thigh/base 危险接触没有进入
  统一奖励；`stumble` 只看足端水平接触力，不能替代腿段碰撞。

训练楼梯几何为 `8 m` 地形、`3 m` 中央平台，上/下楼踏面分别约
`0.32/0.30 m`，台阶覆盖 `4-30 cm`、共 10 个课程行。level 5/6/7 的高度
上界约为 `19.6/22.2/24.8 cm`；当前 `75%` replay 集中 level 5-6。课程均值或
`7+` 因而不是“完整通过 20 cm 楼梯”的充分证据。

## 当前待查的训练语义

必须以生成当前检查点的精确 payload
`history7_flat35_dr3_post_move_capture_20260802/taili_blind_runtime` 为准，
不能用后来变化的工作区代替。下一步只检查：

1. 训练中的楼梯高度、踏面和课程等级真实映射；
2. direction/progress 主驱动是否被质量 gate 削弱或截断；
3. heading、核心、滑移、冲击、支撑和腿段碰撞项的作用域及启用时机；
4. 课程成功是否允许局部高度变化或短距离位移冒充完整跨越。

优秀楼梯的数学优先级是：持续沿机体坐标系中的命令方向完成整段楼梯并安全
落地；速度大小可以适度放宽，但正向进展不能丢失；在此前提下约束 yaw/横漂、
支撑、滑移、冲击、腿段危险碰撞和核心运动。楼梯所需 pitch 必须保留，不能把
合法俯仰当成平地核心误差。禁止回到状态机、逐腿落脚规定或教师策略。

## 训练根因与隔离候选

进一步沿精确 payload 追到了一条与 MuJoCo 时间序列直接对应的错误信用路径：

- 碰撞/支撑响应使 `terrain_drive = 1 + 2.7 * response * clearance_success`，
  首阶后进展驱动可从 `1.0` 跳到 `3.7`；
- `terrain_task_gate` 在地形上把稳定门直接旁路为 `1.0`，失稳动作仍领取完整任务
  信用；
- 同一碰撞迹线还解锁逐腿抬脚正奖励，并减轻足端轨迹和关节参考约束；
- calf/thigh/hip/base 接触未进入奖励，课程又允许单阶结果升级。

这不是单个权重问题，而是“撞阶后驱动增大、约束变松、危险接触不付费、单阶即
成功”的联合局部最优。它解释了 MuJoCo 中首阶后 `vx 0.24 -> 0.69 m/s`、核心
高频、偏航累积和侧翻，也解释了该行为对接触求解器变化高度敏感。

已从该精确 payload 建立隔离候选：

`history7_flat35_dr3_stair_course_semantics_20260803`

候选保留持续方向/tracking/进展主驱动，删除有效行为中的碰撞记忆、探高正信用、
碰撞后进展倍增和约束退让；地形失稳保留 `25%` 探索底线，完整信用随稳定性单调
恢复。课程按 IsaacLab pyramid-stairs 真实几何要求约 `75%` 整段净换高和沿程，
并把横漂/yaw 纳入成功。新增即时足端立面成本与 hip/thigh/calf/base 接触力成本。
楼梯姿态成本只评价 roll 和超过 `0.20 rad` 的 pitch 余量，合法 pitch 不按平地
倾角惩罚。平地、AMP、actor、PD、命令和 DR 配置未改。

新课程验证、原核心/站立、下楼、AMP、gait clock、DR 分层 verifier 及 compileall
均已串行通过。训练机关闭，尚未部署训练。

部署归档为 `history7_flat35_dr3_stair_course_semantics_20260803.tar.gz`，SHA256：
`D171DFBB3165C1F8F03F6FDCC3B6BD7D5253DC10BCD3ECE350D4416D8CDC2D22`。

## Global Objective

Do not redesign from scratch. Preserve the policy's useful flat-ground,
standing, stair, and DR capabilities; adjust or add only what evidence shows is
missing. Excellent stair locomotion means successful commanded traversal first,
then stable and safe support, low slip and segment collision, controlled core
motion, reasonable foot clearance and impact, direction retention, and a clean
landing. Terrain pitch itself is not an error; unnecessary oscillation and
failure to make progress are.

## 课程语义部署与 nominal 楼梯主训

首次部署 `stair_course_semantics` 时，真实 IsaacLab reset 暴露
`terrain_curriculum.staircase_course_geometry` 未导入模块名的运行期
`NameError`。该问题发生在加载检查点后的首个 reset，未产生任何训练步。已改为在
三种包布局中显式导入 `staircase_course_geometry` 并直接调用，同时在 verifier 中
加入 AST 导入契约。修正版归档 SHA256 为
`427948C62BAFEA80202D5F697E0E581F33C4E728E779229EC3366B6FFC94FD33`。

修正版从成熟 `agent_50000.pt` 恢复，运行
`taili_train_20260803_stair_course_semantics_resume50k_long_v2`。到新训练 step 6000
时已写出 `agent_6000.pt`；但实际有效分布仍是 80% 平地、10% 下楼、10% 上楼，
且 DR3 楼梯人口为 60% nominal、20% real、20% fault。新整段语义下完整升阶约
16%、失败降级约 39%，说明该运行可验证语义，却不能作为 nominal 楼梯主攻分布。

据此建立增量候选：

`history7_flat35_dr3_nominal_stair_focus_20260803`

该候选不改 actor、平地奖励、AMP、PD、方向/沿程主驱动、课程成功公式或碰撞安全
语义，只改训练暴露：30% 平地、25% 下楼、45% 上楼；所有楼梯人口固定 nominal，
平地保留原 DR3 混合；75% 楼梯 replay 覆盖等级 3-6，其余维护高难度前沿。归档
SHA256 为 `5463A8371EA196C42712CDC9FCB720C8A6C50AD39DDCE56B371593D1CB42BBC4`。

当前正式运行：

`taili_train_20260803_nominal_stair_focus_resume6k_long`

恢复点为上一运行的 `agent_6000.pt`。启动日志确认 1024 环境中下楼 256、上楼
460、其余 308 为平地；上下楼均为 100% nominal；下楼 replay 192、上楼 replay
345、前沿 179。首批完整 episode 到 step 1160 时，全历史完整成功约 10.3%，最近
20 个可评估窗口约 17.5%；最近 `height_ok/forward_ok/stable_end` 分别约
60%/77.5%/52.5%，速度受控 100%，终止约 27.5%。当前继续训练，优先判断完整
成功是否持续上升、终止和碰撞是否下降；平地四方向、站立、高度与核心作为同优先级
保真约束监控，不以课程均值替代真实通过。

## 20 cm DR0 物理诊断与质量候选

对 `taili_train_20260803_nominal_stair_focus_resume6k_long/agent_6000.pt` 执行了
单环境、精确 `0.20 m`、平台 `1.5 m`、DR0 的上下楼诊断。原始结果位于训练机：

`/root/gpufree-data/diag_runs/nominal_stairs20_agent6000_20260803_1500`

上楼完成 `4.749 m` 沿程和 `1.688 m` 地形净换高，下楼完成 `5.075 m` 沿程和
`-1.618 m` 地形净换高。训练内已经存在完整通过能力，但质量不能验收：

- 上楼横漂 `0.698 m`，最大绝对 yaw `10.18 deg`，触地竖直速度
  mean/p95/max=`0.404/2.032/2.660 m/s`，滑移 p95=`0.476 m/s`，动作变化率
  p95=`112.0/s`；
- 下楼横漂 `0.329 m`，最大绝对 yaw `10.04 deg`，pitch p95=`27.15 deg`，
  局部机身高度最低 `0.228 m`，滑移 p95=`0.505 m/s`；
- 两向力矩均未长期饱和，故不是电机能力耗尽。结合 MuJoCo 上楼失败，可以判定当前
  actor 学到的是接触求解器敏感的激烈通过路径。

逐帧奖励同时显示 `terrain_clearance=0`，而上楼合法抬腿的 `clearance_over` 可单帧
达到约 `-2.9`；方向主奖励只按纵向投影给分，横向速度不降低其得分；质量核又将
支撑、滑移、核心与触地直接相乘，任一短板接近零后会削弱其余短板的优化梯度。

据此建立增量候选：

`history7_flat35_dr3_nominal_stair_quality_20260803`

候选保留方向/进展驱动和所有直接安全成本；用机体系横向速度修正满额方向信用，将
楼梯质量改为算术/调和联合核，并把 body/foot 碰撞纳入满额信用；只在楼梯上封顶
通用过高净空成本，平地语义不变。该设计不包含状态机、逐腿落点、探高或教师策略。
