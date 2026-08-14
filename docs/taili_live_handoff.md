# Taili Live Handoff

更新时间：2026-07-29

## 2026-08-04 固定轴驱动后的历史净空桥接

- 已停止但必须保留的对照 run：`/root/gpufree-data/taili_runs/taili_train_20260803_fixed_course_drive_restorefix_resume50k_long`。它从历史 actor 的 `agent_50000.pt` 恢复，只把楼梯主驱动改为 episode 初始 yaw 固定轴上的有符号净位移速度，并修复了 1024 -> 4096 环境恢复时丢失 phase/DR 的问题。
- 固定轴 run 的四个完整窗口表明语义纠偏有效但能力路径不完整：方向信用稳定约 `terrain_progress=0.77`、`terrain_direction=2.05`，fall/terminal 和平地四方向未坍缩；但第四窗 `23.71k-26.53k` 的上楼均值仅 `0.567 -> 0.570`，`course_ok=0.453 -> 0.396`，`move_up=0.108 -> 0.131`，不能把课程均值缓升当作通过能力。
- 历史 7+ 对照是 `/root/gpufree-data/taili_runs/taili_train_20260723_214211_descent_quality_resume/checkpoints/agent_30000.pt`。其 `25k-30k` 窗口实际正信用均值为 `linear_progress=0.780`、`terrain_progress=0.988`、`terrain_direction=1.844`、`terrain_clearance=0.510`，上下楼约 `7.48/7.59`。当前固定轴版本保留了正确方向信号，却显式删除了历史的碰障后净空正信用。
- 历史净空机制不是状态机：真实足端立面碰撞后，仅对同一只脚相对碰撞支撑面的竖直抬离给连续、有界正信用；不规定腿序、落脚点、步态相位，不向 actor 输入地形真值。碰撞本身仍付费，非足端严重接触仍硬否决。
- 新候选为 `history7_flat35_dr3_fixed_course_drive_clearance_bridge_20260804`。它保留固定轴有符号净推进、楼梯上的旧机体系 `linear_progress=0`、现有平地/站立/AMP/PD/DR 和全部直接安全成本；只恢复碰障高度短记忆与 `w_terrain_clearance=4.0`。`terrain_clearance_drive` 仍为 `0`，因此碰撞不会放大方向主奖励。
- 候选归档 SHA256：`c16662bbf60d56da8db03a94ac4f3f511167038145e5728b29dae108ff1abed2`。本地和远端均串行通过楼梯课程、下楼、核心/站立、command-conditioned AMP、stand gait clock、DR 分层 verifier 及 `compileall`。
- 当前正式 run：`/root/gpufree-data/taili_runs/taili_train_20260804_clearance_bridge_resume30k_long`，从固定轴 run 的 `agent_30000.pt` 恢复并一次性只重置 stairs/stairs_up。启动确认唯一 trainer、`phi3/DR3/flat=3` 保持、replay=`0`；以首个 PPO 后的 `step 390` 为十分钟监控基线。早期 `terrain_clearance` 约 `+0.30` 只证明链路生效，不能提前宣称能力改善。
- 后续只用完整窗口判断：上下楼课程均值必须与 `course_ok/height_ok/move_up/move_down` 联合提高，同时 F/B/L/Y、gait/duty、slip、support、tilt、碰撞/冲击和 fall 不得被交换。仍不使用 frontier 作为能力证据，也不引入状态机、固定足序、固定落脚点或教师策略。

## 2026-08-03 Resume Restore Fix

- The fixed-course payload initially resumed `agent_50000.pt` into 4096 environments from a 1024-environment sidecar. The sidecar correctly held phase 3 and DR3, but the restore code validated the per-environment terrain array first. Its fallback restored terrain only, silently restarting phase/DR at 0.
- This is a launch-state bug, not evidence about the policy. The short invalid run was stopped without creating a checkpoint. Do not use its `curriculum_state.json` as a resume source.
- The fix restores global state (phase, DR, penalty gate, clearance gate) before validating or remapping terrain arrays. A layout mismatch can now remap terrain while preserving phase 3 / DR3. The one-time reset still affects only `stairs` and `stairs_up` levels.
- Required post-launch facts: checkpoint `agent_50000.pt` loaded; phase=3; dr_level=3; flat remains at 3; stairs/stairs_up start at 0; replay=0; and the trainer reaches PPO before monitoring real course progress.
- Verified run: `/root/gpufree-data/taili_runs/taili_train_20260803_fixed_course_drive_restorefix_resume50k_long`, payload `history7_flat35_dr3_fixed_course_drive_restorefix_20260803`. Its first PPO telemetry reports phase=3, dr_level=3, flat_mean=3.0, stairs/stairs_up near 0 after the one-time reset, and replay=0.
- At about step 3.2k, the valid run remains phase3/DR3 with flat_mean=3.0 and no fall trend. Stairs/stairs_up means have only begun rising from zero (about 0.32/0.28); `terrain_course_ok_rate` and up `move_up_rate` are still 0. This is an early-training fact, not a capability claim. Continue using fixed windows; do not use terrain_direction or frontier alone as evidence of traversal.

## 当前状态

- 已停止而非继续运行：`/root/gpufree-data/taili_runs/taili_train_20260729_120239_phase0_flat_swing_path_quality_fresh`，最后检查点 `agent_58000.pt`。
- 当前 payload：`/root/gpufree-data/training_payloads/taili_blind_runtime_phase0_flat_swing_path_quality_20260729_1145_verified`。
- 不能 resume 这个 actor，也不能在不改训练语义的前提下继续等待。约 30k 后 phase0 的命令、地形/DR、penalty gate 和质量门均已基本固定；40k 到 50k 的真实诊断是互相交换，不是全局共同收敛。

## 全局验收

平地必须同时满足：四方向跟踪、自然且对称的步态和足端轨迹、无持续髋外翻/内翻、低滑移和轻触地、稳定支撑、移动时核心与高度稳定、零命令安静站立、动转静平缓制动、yaw 不泄漏平移。楼梯和 DR 不能用这些质量交换；楼梯不使用状态机、固定落脚点、固定腿序或势能/层级技巧。

日志只用于趋势和健康，接受与否以同协议物理诊断为准。诊断需包含初始站立、F/B/L/yaw、各段后的零命令保持，并看原始运动学/接触/力矩数据。

## 已验证事实

### 58k flat 诊断

严格 900 帧 DR0 平地诊断：`/root/gpufree-data/diag_runs/locomotion_console_20260729_064757_directions`。

- 50k 相对 40k 不是整体变好。前进跟踪、roll/pitch、F->0 残余速度变差；后退和横移的触地变重；yaw 跟踪变差且平移泄漏几乎未变。
- 零命令稳定尾段仍非安静站立：平面残速 p95 `0.202 m/s`，高度误差 p95 `0.0446 m`，roll/pitch 合成倾角 p95 `1.50 deg`，角速度 p95 `0.206 rad/s`。四脚接触接近满分不代表机身已静止。
- 制动窗口更差：平面残速 p95 `0.372 m/s`，倾角 p95 `2.62 deg`，站立接触内 slip p95 `0.389 m/s`，足端三维速度 p95 `0.716 m/s`。这是实际急刹和不稳定支撑，不是单一日志伪象。
- 前进是平地最差方向：tracking p95 `0.239 m/s`，tilt p95 `2.36 deg`，height error p95 `0.0273 m`，vertical speed p95 `0.233 m/s`，wxy p95 `0.353 rad/s`，hip target error p95 `0.429 rad`，joint target error p95 `0.801 rad`，action delta p95 `0.418`。
- yaw 仍有问题：tracking p95 `0.144 rad/s`，平移泄漏 p95 `0.174 m/s`，tilt p95 `2.91 deg`。
- 触地并非所有方向同样差：前进/横移事件触地较轻，yaw 和制动尾部仍有重触地尖峰。训练端 touchdown 指标与诊断事件定义不同，不能只凭训练端 p95 直接改权重。
- 实际关节目标误差很大但未饱和。当前 Kp/Kd 和历史 7+ 配置相同（hip/thigh/calf Kp `120/100/120`，Kd `5/5/5`），因此暂不以改 PD 或加大 hip 惩罚作为结论。

### 门控和奖励拓扑

- `penalty_gate` 仅由方向进展和惩罚预算推进。它在约 15k 为 `0.5`，20k 已为 `1.0`，当时 flat stand/capture 仅约 `0.105/0.135`，flat core 仍未通过。
- `stand_readiness_gate = penalty_gate^4`。因此它错误地把“方向能走”当成“站立已形成”，20k 后提前切到严格站立核。
- 真实零命令中，平面残速、joint velocity、action delta、高度误差共同使严格站立质量很低。主 stand/attractor credit 在 59k 仅约 `0.0005/0.008`，而 terrain direction credit 约 `1.25`；stand capture persistent 在严格 floor 下也只有约 `0.001`。继续训练没有有效的安静站立正向路径。
- `quality_refinement_scale` 也直接跟随同一个 penalty gate；hip、轨迹、动作和触地等 refinement 项在核心/站立尚未形成时已全部收紧。这违反“主驱动先可达、约束后收紧”的要求。

### phase0 的结构冲突

- phase0 的地形比例为 flat/stairs_down/stairs_up=`45/20/35`，地形奖励从 step 0 开启。约 55% 环境为最低级楼梯，并且这些楼梯以 `stairs_forward_command_prob=1.0` 强制前进。
- 平地方向分层大致均分，因此绝大多数前进命令来自楼梯。Actor 没有部署时地形真值，却可以把“前进命令”当作楼梯代理标签。前进诊断恰好是高度、上下起伏、动作变化、髋目标误差和跟踪最差的方向。
- 这不是“地形从一开始训练”本身的错误，而是地形暴露、命令条件和收紧门之间的组合错误：它把尚未形成的通用平地前进行走与楼梯准备姿态混成一条路径。

## 历史如何使用

- 历史 7+ 检查点 `taili_train_20260723_214211_descent_quality_resume/agent_30000.pt` 证明同一模型/执行器可以获得较强楼梯能力；它不是可直接复制的 fresh 配置。
- 历史配置的 phase0 是平地单轴/站立形成，`terrain_reward_start_phase=1`；后来通过 resume 链条引入楼梯。它说明“先有可用通用平地路径，再增强楼梯”是实际成功路径的一部分，但当前目标仍要求低级楼梯从开始暴露，不能机械退回旧策略。
- 当前与历史的动作 scale、延迟和 PD 相同，不能把当前平地问题归咎于这组物理参数而盲改。

## 下一步，只能针对已证实问题

1. 站立收紧改为由实际 flat stand/capture EMA 驱动，而不是全局 penalty gate。宽核持续提供多因素正向捕获信用；严格核只在真实静止已经接近形成后逐步接管。制动路径应有独立的持续正向质量，而不是只把未来站立奖励乘成很小的 gate。
2. 将通用 refinement 收紧与实际 flat moving-core readiness 协调，保留低权重安全成本和直接核心正向质量，避免在核心未形成时把髋/足端/动作约束全部推满。
3. 保留低级楼梯从 step 0 的训练，但消除 phase0 中“前进命令几乎等于楼梯标签”的结构性相关性。优先改命令条件/阶段性楼梯命令策略，而不是把地形能力关掉或注入特权地形信息。
4. 暂不动 PD、action scale、hip 权重、AMP 主权重或 swing-path 项；当前证据不足以证明它们是根因。新 run 的早期物理诊断必须先验证核心、站立、制动和前进是否同时改善，再对轨迹/yaw/触地做第二轮协调优化。
5. 任何上述奖励拓扑改动都必须 fresh，不 resume 当前 `58k` actor。DR 和高楼梯验收在平地物理指标真正通过后再推进；它们仍是同等最终目标，不是放弃项。

## 2026-07-29: phase0 全局 readiness 解耦实现

- 本次不是只修站立。验收面保持联合：四方向跟踪、自然对称步态与足端轨迹、无持续髋外翻/内翻、稳定承重、低滑移和轻触地、移动中的核心与高度稳定、零命令安静站立、动转静平缓刹车、yaw 无平移泄漏，以及安全的前进上下楼梯；DR 不能用这些质量交换。
- 从已停止的 `phase0_flat_swing_path_quality_fresh/agent_58000.pt` 精确复制 payload 后确认根因：`penalty_gate` 已在 flat stand/capture 和 moving core 尚未形成时走满，既提前收紧站立严格核，也提前收紧髋、轨迹、触地和动作 refinement。与此同时 phase0 的 55% 楼梯样本几乎都强制前进，盲 actor 可以把前进命令当成地形代理，破坏通用平地前进行为。
- 新 payload：`output/target_22000_rework/staging_phase0_global_readiness_decouple_20260729`。静止严格核只由 flat stand-attractor 与 motion-to-zero capture 的几何均值 EMA 逐渐接管；移动精修只在四个纯平地方向中最差的 flat-core EMA 达标后逐渐接管。`penalty_gate` 仍只作总压力上限，不再充当这些物理能力的形成时钟。
- phase0 保留低级楼梯从 step 0 暴露，但改为 flat/down/up = `0.70/0.12/0.18`，楼梯中仅 40% 前进、其余保持；phase1/2 再恢复 100% 前进楼梯主任务。没有注入特权地形信息，也没有恢复状态机、固定落脚点或固定腿序。
- 未改 PD、action scale、AMP 主权重、足端碰撞几何和既有的直接楼梯方向/进展驱动。先验证被证明的结构冲突，避免把髋、足端、核心、楼梯等不同问题错误归为同一个物理参数。
- 已通过 `verify_fresh_learning_path.py`、静止质量、平地足端、楼梯安全进展、直接驱动与运行时编译验证。验证同时断言 readiness 门控不进入 reward total，不能凭门控值虚增收益。
- fresh 训练的早期物理诊断必须联合检查 F/B/L/yaw、移动核心/高度、支撑与触地、髋/足端轨迹、动转静刹车和零命令静止。日志只判断趋势；没有这些物理面同时改善，不以单项课程或单个奖励上升宣布成功，也不提前把高难楼梯或 DR 当作通过。

### 本次 fresh 实例与早期监控

- 已部署并启动：`/root/gpufree-data/taili_runs/taili_train_20260729_163743_phase0_global_readiness_decouple_fresh`，payload 为 `taili_blind_runtime_phase0_global_readiness_decouple_20260729_verified`，确认无 resume checkpoint。保存间隔为 `10k`，仅为避免 6 GB 空闲盘被相邻检查点耗尽，不改变训练语义。
- `10.1k` 不能判为成型，但比 `4-6k` 更健康：forward/yaw 约 `0.59/0.62`，backward 从约 `0.02` 回升到约 `0.15`，lateral 从约 `0.14` 升到约 `0.35`；gait、duty、支撑不稳、平均 tilt、跌倒率和线速度误差均在最近窗口改善。moving-core readiness 已从零开始打开，说明物理而非轮数的 refinement 路径确实开始工作。
- 仍不合格且必须持续看：strict stand readiness 仍为零；平地 tilt/wxy/高度误差仍明显大于最终目标；touchdown 速度尾部、最差足端轨迹、hip deviation 与 landing impact 没有随其他质量同步改善。此时不做早期渲染诊断或重构：先观察 refinement 打开后这些面是否共同改善；若方向再次分裂或核心/高度/接触/轨迹持续背离，再回到具体 reward 与采样语义做原因调查。

## 2026-07-29: 从全局表现回到奖励主路径

### 当前现象，不以单个 progress 代替

- 当前仍在运行的 `phase0_global_readiness_decouple_fresh` 到约 `42.5k` 时，raw F/B/L/Y 约为 `0.93/0.82/0.82/0.92`，质量调整后的 progress 约为 `0.80`。这说明 Actor 已能按命令移动，但不能证明动作满足全局质量；也不能据此简单增加 tracking 权重。
- 同协议物理诊断 `output/diagnostics/20260729_101123_phase0_global_flat_20k/record.csv` 中，固定命令的平均真实跟踪误差多为 `0.05-0.08 m/s`。同时前进 tilt p95 `4.01 deg`、wxy p95 `0.377 rad/s`、高度跨度 `4.62 cm`、触地竖直速度峰值 `0.868 m/s`；横移偏轴 p95 `0.142 m/s`；yaw 平移泄漏 p95 `0.119 m/s`。
- settled stand 仍有平移 p95 `0.106 m/s`、wxy p95 `0.232 rad/s`、高度跨度 `2.92 cm`。F/B/L 动转静均不能连续安静 `0.3 s`，并有约 `0.125/0.096/0.103 m/s` 的反向过冲。直行摆动足还存在约 `3.4-9.4 cm` 的横向跨度。
- 因而当前属于“raw 跟踪较好，但实际综合质量差”。日志中的 progress 混合了跟踪与质量资格；它既不是纯速度跟踪，也不能在旧资格定义下代表真实核心、站立或轨迹已经合格。

### 已闭环确认的原因

1. moving-core readiness 使用了可被好分量补偿的质量分数，并以 `0.35 -> 0.70` 收紧。当前严格核心 EMA 约 `0.68`，readiness 却已约 `0.991`，与物理诊断明显矛盾。
2. fresh 初期直接使用严格核心尺度，缺少可达的远场正信用；零命令核心正信用又在 readiness 前大部分关闭。这会同时造成“约束已收紧”和“正确动作没有足够驱动”。
3. 平地高度成本在约 `2 cm` 后趋于饱和，严重高度误差反而缺少继续修正的梯度。
4. 原 `swing_path_quality` 只处理过高与反向速度，没有处理相对解析运动方向的横向速度；髋惩罚存在也不能补上这个足端语义盲区。
5. 原制动路径带宽过宽。诊断制动窗 planar/angular acceleration p95 约为 `1.475/3.044`，旧 instant path quality 中位数仍约 `0.909`；同时没有按停止前命令识别跨零反向过冲。
6. `stand_capture_delta` 已经是有符号的静止质量改善，不应再叠加同义“制动进展”机制。缺失的是可达的持续站立吸引、合理的捕获窗强度和反向过冲语义。

### 数学调整

- 每个核心分量使用 `q_i=exp(-|e_i|/s_i)`。严格核覆盖 roll、pitch、roll/pitch 角速度、线性运动时的 yaw 残差、高度与竖直速度，尺度分别为 `0.07/0.07 rad`、`0.24/0.24 rad/s`、`0.28 rad/s`、`0.025 m`、`0.10 m/s`；聚合为 `0.2*mean(q)+0.8*harmonic(q)`，避免好分量掩盖最差分量。
- fresh 宽核使用 `0.12 rad`、`0.48 rad/s`、`0.07 m`、`0.22 m/s`，并随真实 refinement readiness 连续收紧：`Q_train=Q_wide+r*(Q_strict-Q_wide)`。readiness 与 tracking 资格只读取 `Q_strict`，以 `0.45 -> 0.78` 映射；训练宽核不能虚增验收。
- 平地主任务信用保留 `0.35` 恢复底线，并按 `G_core=0.35+0.65*Q_strict` 取得完整信用。这样错误动作仍能探索，但最大化主奖励必须同步提高严格核心质量。
- 平地高度使用 Huber tail。令 `x=max((|h-h*|-0.01)/0.02,0)`，成本为 `0.5*x^2`（`x<=1`）或 `x-0.5`（`x>1`），严重误差区保持线性梯度。
- 中摆段横向偏轨使用 `v_cross=|v_x*d_y-v_y*d_x|`，其中 `d` 是解析足端参考方向；仅评价速度方向，不规定落脚点、腿序或状态机。超过 `0.05 m/s` 后按 Huber tail 计入已有足端质量。
- 动转静反向过冲使用 `o_lin=max(-v_xy dot d_stop,0)`，yaw 使用 `o_yaw=max(-omega_z*sign(omega_cmd_stop),0)`；只在既有捕获窗内生效。制动路径 free/scale 收紧为 planar `0.45/1.20`、vertical `0.30/0.90`、angular `0.50/2.00`。
- 零命令从 step 0 保留 `0.60` 宽核心信用；stand/capture 的严格核仍由真实 joint/foot/body/height/path 联合质量接管。readiness 只负责收紧，不进入 reward total。

### 保留项与下一轮判定

- 本轮没有修改 PD、action scale、AMP 主权重、碰撞几何、四方向直接驱动、duty/diag、滑移/触地已有成本或楼梯方向/进展/安全语义。诊断证明这些面仍需验收，但尚不足以把根因归为对应权重或底层参数。
- hip 权重已经存在且不低；本轮补的是足端横向语义盲区，并把 hip/轨迹 refinement 的收紧时机改由严格 moving-core readiness 协调。新 run 若核心改善而髋外翻、偏轨或触地尾部不改善，再分别检查测量域、激活门与权重，不把它们混成核心问题。
- 楼梯直接驱动和安全约束保持逐项不变，所有 terrain verifier 的总奖励与旧 payload 一致。phase0 先验证完整平地契约；进入 phase1 后必须继续守住平地，同时验收安全稳定上下楼梯，之后进入 DR。平地、楼梯和 DR 是同一个最终全局目标，不互相交换。
- 新候选 staging 为 `output/target_22000_rework/staging_phase0_global_reward_path_20260729`。全部 `verify_*.py`、YAML 字段契约和 runtime `compileall` 已通过；运行时差分只有 `taili_reward.py`、`blind_tp_env.py` 与 `taili_blind_config.yaml`。

### 部署与当前 fresh run

- 当前有效远端 payload：`/root/gpufree-data/training_payloads/taili_blind_runtime_phase0_global_reward_path_20260729_verified_v2`，archive SHA256 为 `1820a77212560c2b8904f1b4b6e816107c6e7878a8a2cd3188fc14735d6bb442`。首个无 `v2` 的远端目录只是验证封装缺少基线而未通过，不能用于训练。
- 旧 `phase0_global_readiness_decouple_fresh` 在约 `54.4k` 受控停止，保留到 `agent_50000.pt`；没有 resume。
- 当前 run：`/root/gpufree-data/taili_runs/taili_train_20260729_193103_phase0_global_reward_path_fresh`。`run.json` 已确认 payload 绑定正确且 `resume_checkpoint` 为空；启动时为 `phi0/DR0`、1024 环境、seed 42、checkpoint 间隔 `10k`。
- 第 1 步 `flat_core_quality=0.297`，moving-core/stand readiness 均为 `0`。这只验证严格质量语义没有被虚开，不能用于判断最终效果。按约 10 分钟窗口监控 raw 四方向、严格核心、stand/capture、touchdown/trajectory/tilt/wxy/height 的联合趋势；达到可诊断水平后再做同协议物理诊断。

### fresh 启动后的前五个监控窗口

- `3.7k`：raw F/B/L/Y 约 `0.39/0.04/0.19/0.51`，strict core 约 `0.255`，stand/capture 约 `0.065/0.076`。这是方向尚未共同形成的早期状态，不做诊断或干预。
- `7.2k`：raw 前进/yaw 约 `0.92/0.76`，横移约 `0.31`，后退仍约 `0.013`；strict core 回升到约 `0.291`。该分裂与已知的早期串行形成相符，不能据此重启。
- `10.7k`：后退恢复到 raw `0.262`，横移 `0.652`，前进/yaw 约 `0.887/0.767`；strict core `0.334`，stand/capture `0.083/0.084`。证明共享 actor 未进入站立局部最优，宽核和四方向主驱动均可达。
- `14.3k`：raw 四方向汇合到约 `0.77-0.84`，strict core `0.411`，stand/capture `0.095/0.106`，高度 p95 约 `2.66 cm`；penalty gate 约 `0.44`，但 physical flat refinement 因 strict core 尚未达到 `0.45` 而保持 `0`。此时 touchdown p95 约 `2.68 m/s`、轨迹尾部约 `1.27`，是明确风险但还没有对应 refinement 响应窗口。
- `17.8k`：raw F/B/L/Y 约 `0.835/0.843/0.814/0.888`，质量 progress 约 `0.671/0.697/0.665/0.720`；strict core `0.476`，moving-core readiness 首次只打开到 `0.018`，stand/capture `0.108/0.117`。tilt p95 `0.075 rad`、wxy mean `0.445 rad/s`、vz p95 `0.180 m/s`；touchdown p95 从高点约 `2.68` 降到 `2.07 m/s`，轨迹尾部从约 `1.27` 降到 `1.09`，但两者仍远不合格。
- 当前结论：新拓扑没有卡死 fresh，也没有虚开 readiness；四方向、宽核核心和 stand/capture 正在共同形成，physical refinement 刚开始介入。此刻不改权重、不重启、不做物理诊断，继续按约 10 分钟窗口观察。重点看 readiness 上升后 touchdown、轨迹、wxy/vz/高度是否持续改善且 raw 四方向不坍缩；若不响应，再分别调查事件口径、作用域和权重。

### `22k-30k` 实时窗口补录

- `22k`：raw F/B/L/Y 约 `0.861/0.846/0.821/0.885`；strict moving core `0.534`，refinement readiness `0.162`；stand/capture `0.120/0.170`。touchdown p95 `1.73 m/s`，轨迹最差 p95 `1.05`，说明精修刚介入，仍远未验收。
- `26k`：raw F/B/L/Y 约 `0.867/0.886/0.868/0.901`；strict moving core `0.571`，readiness `0.306`；stand/capture `0.130/0.158`。wxy mean 降至 `0.350 rad/s`，touchdown p95 降至 `1.12 m/s`，轨迹最差 p95 降至 `0.740`；四方向没有因质量收紧而坍缩。
- `30k`：raw F/B/L/Y 约 `0.832/0.921/0.864/0.891`，质量 progress 约 `0.714/0.809/0.743/0.771`；strict moving core `0.576`，readiness `0.327`。stand/capture 约 `0.136/0.177`，stand readiness 仍为 `0`。tilt p95 `0.0534 rad`、wxy mean `0.338 rad/s`、vz p95 `0.162 m/s`、高度误差 p95 `1.74 cm`、touchdown p95 `1.24 m/s`、轨迹最差 p95 `0.678`。
- 从 `14k` 到 `30k`，strict core 约 `0.410 -> 0.576`，wxy `0.574 -> 0.338 rad/s`，高度误差约 `2.38 -> 1.74 cm`，touchdown 从高点约 `2.68 -> 1.24 m/s`，轨迹最差尾部约 `1.27 -> 0.68`；这些是同向改善，当前不能判为平台。
- 风险集中在两处：stand/capture 上升慢且波动，可能成为下一真实阻塞；touchdown 与轨迹仍严重不合格且近期有反弹。继续到约 `33k` 的下一个窗口；若 stand/capture 连续多个窗口不再抬升，先分解已有 strict stand 的 body residual、joint velocity、foot speed、action delta、高度和制动路径，不笼统增加 `w_stand`。若 touchdown 成本上升但原始事件 p95 不降，检查事件口径、作用域及其与主信用的耦合，不直接叠加惩罚。

## 2026-07-29: 站立平台的训练分布根因与修复

### 已确认的不可达冲突

- `phase0_global_reward_path_fresh` 到 `40.5k` 时，strict moving core 继续升至约 `0.636`，但 stand/capture 从 `30k` 的约 `0.136/0.177` 长期停在约 `0.139/0.177`；这不是所有能力共同平台。旧 run 在 `41.6k` 受控停止，保留 `agent_40000.pt`。
- Actor 的 `log_std` 是全局 12 维参数，训练端又对每个关节施加不随命令变化的探索下限。`agent_30000.pt` 的实际参数约为 `-2.30` 到 `-1.59`，相邻独立高斯动作采样的理论 action-delta RMS 约 `0.228`；即使降到配置下限，理论 RMS 仍约 `0.125`。
- strict stand 的 `sigma_stand_action_delta=0.04`，并且同一采样噪声还会进入 joint velocity 与 foot velocity。仅 action-delta 一项按 `exp(-(rms/0.04)^2)` 就已接近零，因此训练 rollout 中的 `phase_gate_flat_stand_quality_0=0.65` 在原分布下基本不可达。不能通过继续等待、增加 `w_stand` 或降低验收门槛解决。

### 修复语义

- 新 staging：`output/target_22000_rework/staging_phase0_command_conditioned_exploration_20260729`。移动命令的 actor mean、原 `log_std` 和逐关节探索下限保持不变；机器人坐标系命令从零到运动阈值连续控制训练方差，不改变命令、不规定动作序列，也不是状态机。
- 零命令使用 `log_std` 区间 `[-6.0,-4.5]`；运动强度为 `u=max(||v_cmd||/0.10, |w_cmd|/0.05)` 截断到 `[0,1]` 后经 smoothstep，`log_std=(1-s(u))*log_std_quiet+s(u)*log_std_move`。端点显式选择原张量，完整移动命令的分布逐位保持。
- `log_std=-4.5` 时理论 action-delta RMS 为 `0.01571`，strict action-delta 质量约 `0.857`；当前移动分布对应 RMS 约 `0.228`，地形和四方向探索没有被削弱。
- SKRL 在 policy 前标准化 observation，因此新增 `CommandPreservingRunningStandardScaler`：body53 的 command 索引 `[6:9]` 保持原始机器人坐标系值，其他 1597 维照常使用同一 RunningStandardScaler。类同时注册到 Runner 组件表和 SKRL 1.4.3 的配置解析模块命名空间；checkpoint 的 policy/scaler state key 与旧格式严格兼容。
- `tools/export_taili_deployment.py` 已同步 command passthrough，并把该契约写入部署 metadata；诊断通过同一 Runner/scaler 自动一致，后续 MuJoCo 使用导出的 TorchScript 时不会输入错位。

### 验证与当前 fresh

- 全部既有 verifier、命令条件探索数学验证、runtime compile、远端真实 SKRL/GaussianMixin、Runner `_process_cfg` 和 `agent_40000.pt` 严格加载均通过。远端有效 payload：`/root/gpufree-data/training_payloads/taili_blind_runtime_phase0_command_conditioned_exploration_20260729_verified_v2`，archive SHA256 `c14938c8093387d8cc973cb98cac290786fae979ef1780cb01565c7c7156080f`。无 `v2` 目录缺少 Runner eval 注册，启动失败，不能使用。
- 当前 run：`/root/gpufree-data/taili_runs/taili_train_20260729_213941_phase0_command_conditioned_exploration_fresh`，`resume_checkpoint` 为空，1024 envs、seed 42。首个 `275` step 已进入完整 PPO 循环，无 OOM/NaN/shape 错误；stand/capture 约 `0.270/0.438`，但移动、核心和高度尚未形成，不能据此验收。
- 接下来按约 10 分钟窗口联合监控四方向形成、strict core、stand/capture、touchdown、轨迹、wxy/vz/高度。必须验证站立改善不会交换掉四方向和足端质量；日志达到可观水平后才做同协议全方向物理诊断。

## 2026-07-29: 保留低噪声修复并恢复 actor 输入契约

### command passthrough 反例

- `phase0_command_conditioned_exploration_fresh` 证明零命令低方差方向正确，但 `CommandPreservingRunningStandardScaler` 同时改变了 actor mean 的命令输入尺度。到 `7.5k` 时 raw F/B/L/Y 仅约 `0.65/0.005/0.19/0.21`；到 `11.3k` 仍约 `0.69/0.007/0.17/0.23`，后退没有形成，横移和 yaw 显著慢于原标准化输入契约。stand/capture 虽早期达到约 `0.53/0.47`，却随前进形成回落到约 `0.54/0.23`，形成新的移动/静止交换。
- 高度误差曾在 `3k-6k` 长期约 `11-12 cm`，到 `6.5k` 后恢复到约 `3 cm`；因此该 run 不是完全不能学习，而是输入尺度改变造成方向形成严重失衡，不能用继续等待或改奖励解释。该 run 已在约 `11.4k` 受控停止，保留 `agent_10000.pt` 作为反例。

### 不改 actor mean 的方差门

- 当前 `agent_4000.pt` 的 scaler 实测：零命令被屏蔽后的八维 gait clock 标准化 RMS 为 `0.000083`；运动小跑时最小/平均/最大 RMS 为 `1.16023/1.16029/1.16041`。初始化方差为 1 时理论上也是 `0` 对约 `0.707`，该信号在训练全程有稳定分离。
- 新策略恢复完整 `RunningStandardScaler`，actor mean 重新读取历史验证过的全标准化 body53+history 输入。log-std 单独读取标准化 body gait clock 的 RMS，以 `0.10 -> 0.50` smoothstep 区分静止与运动；静止仍使用 `[-6.0,-4.5]`，运动仍逐位保持原 log-std 与 12 关节探索下限。没有改命令、奖励、AMP、PD、action scale、课程、足端、楼梯或 DR 语义，也没有增加部署输入、状态机、腿序或落脚点。
- 新 staging：`output/target_22000_rework/staging_phase0_gait_clock_conditioned_exploration_20260729`。本地全部 verifier 与 compileall 均通过；远端真实 SKRL 严格加载旧 checkpoint 成功，actor command 标准化误差为 `0`，stand/move 方差门实测为 `[0,0,1,1]`。
- 有效远端 payload：`/root/gpufree-data/training_payloads/taili_blind_runtime_phase0_gait_clock_conditioned_exploration_20260729_verified`；archive SHA256 为 `4690458d03e2487f82fda93df6df06452aa36eb9fbea92f012557bc956894cd4`。

### 当前 fresh run

- 当前 run：`/root/gpufree-data/taili_runs/taili_train_20260729_221543_phase0_gait_clock_conditioned_exploration_fresh`。`run.json` 已确认 payload 绑定正确、`resume_checkpoint` 为空、1024 envs、seed 42；首个完整 PPO tick 已到达，无 OOM/NaN/Traceback。
- 先按约 10 分钟窗口与前两次 fresh 的同期数据比较四方向形成、strict core、stand/capture 和高度。当前修复只有在站立可达性保留、四方向恢复原形成速度且后续核心/足端质量不被交换时才成立；日志尚差时不做物理诊断。

## 2026-07-29: 条件方差下的 PPO 更新尺度与当前 fresh

### gait-clock 反例与根因

- `phase0_gait_clock_conditioned_exploration_fresh` 在 `4k` 的 raw F/B/L/Y 约为 `0.120/0.134/0.126/0.205`，stand/capture 约为 `0.453/0.448`，但高度误差 p95 约 `10.7 cm`；`agent_4000.pt` 的优化器 LR 已降到下限 `3e-5`。该 run 在约 `7.3k` 受控停止并保留 `agent_4000.pt`。
- gait-clock 门本身已被真实 scaler 状态验证：零命令标准化 gait-clock RMS 约 `3.4e-5`，运动样本最小约 `1.15865`，门输出严格为 stand `0`、move `1`。失败不再来自 actor 输入契约或运动/静止误分类。
- 高斯 mean score 随 `1/sigma` 放大，固定参数步长诱发的 mean KL 近似随 `1/sigma^4` 放大。新实现仅对 mean 反向梯度乘 `(sigma_conditioned/sigma_moving)^2`，使静止样本自身的更新产生与移动样本同阶的 KL；actor mean、采样 action、log-probability、PPO ratio/clipping、条件方差及全部任务奖励均不变。
- 本地和远端真实 SKRL 验证得到：静止 action-delta RMS `0.015710`、strict action-delta quality `0.857047`、移动 RMS `0.228447`，静止 mean 梯度尺度约 `0.001072`、移动为 `1.0`；全部 verifier、compileall、旧 checkpoint 严格加载与前向数学测试通过。

### 部署与实时判定

- 当前有效 payload：`/root/gpufree-data/training_payloads/taili_blind_runtime_phase0_variance_kl_equalized_20260729_verified`，archive SHA256 `82347c1db1bbfaeb574b58235a348cc260d3bd2918b8c76a637094ddb61457b5`。当前 fresh run：`/root/gpufree-data/taili_runs/taili_train_20260729_223908_phase0_variance_kl_equalized_fresh`，1024 envs、seed 42、`resume_checkpoint` 为空。
- 同 seed 精确对照：`4k` 当前 raw F/B/L/Y=`0.555/0.029/0.177/0.538`，旧全局方差 run 为 `0.572/0.034/0.249/0.513`；当前 strict core `0.287` 对旧 `0.258`，stand/capture `0.350/0.339` 对旧 `0.066/0.072`，高度误差 p95 `1.8 cm` 对旧 `3.4 cm`。因此当前已基本恢复原移动形成路径，同时显著提高静止可达性，不能因单个 LR 信号提前重启。
- `6k` 当前 raw F/B/L/Y=`0.703/0.007/0.200/0.815`，旧全局方差 run 为 `0.765/0.031/0.206/0.802`；后退是唯一明显滞后项。stand/capture 回落到约 `0.276/0.288`，仍显著高于旧 run，但必须继续确认移动能力形成时不会交换掉静止能力。
- `agent_2000/4000/6000.pt` 的 LR 均为 `3e-5`。SKRL scheduler 使用 rollout 旧 log-prob 与更新后 log-prob 的全样本近似 KL；当前 hook 只能缩小静止样本自身贡献的 mean 梯度，移动样本更新共享 actor 后仍会改变静止状态的 mean，并被极小静止方差放大。这是尚未关闭的跨状态 KL 风险，但当前真实学习速度表明它还不能被等同为行为失败。
- 继续观察到 `10k` 左右：要求后退和横移沿已知串行形成路径恢复，stand/capture 不持续坍缩，核心、高度、touchdown 和轨迹不以移动进展为代价。日志未成型前不做物理诊断；若方向停滞或静止优势持续丢失，再处理共享 actor 的跨状态干扰或 scheduler 口径，不修改奖励来掩盖优化器问题。

## 2026-07-30: 静止分支优化器闭环与当前 fresh

### 已确认的优化器根因

- 对真实 PPO 更新按命令状态重算 approximate KL 后，静止样本约占 `19%`。旧条件方差实现中 moving KL 多数约为 `0.01-0.02`，quiet KL 常为 `0.05-0.9`，部分 epoch 达 `2.5-12`；第一个 optimizer step 前两者仅约 `2.4e-7/2.0e-8`，因此 scaler 漂移和 rollout 旧策略误差均不是根因。
- 静止 mean 已与移动 mean 参数隔离，但原实现只把静止 mean 梯度乘 `(sigma_quiet/sigma_move)^2`。Adam 对近似恒定缩放满足 `m/sqrt(v) ~= sign(g)`，会基本抵消该梯度尺度；所以参数隔离和梯度缩放仍不能保证静止参数步长足够小。
- 修复保留低静止方差、独立静止 mean、原奖励和课程；把静止 encoder/actor 放入独立 Adam 参数组。初始 LR 比例取 `sigma_quiet/sigma_move ~= exp(-4.5+1.3) ~= 0.04`。moving KL 只调主参数组，quiet KL 只调静止参数组；mean 梯度缩放仍保留，作用仅是防止静止梯度占据全局 grad clipping。
- 单组旧 checkpoint 加载后可无损拆分；新双组 checkpoint 在加载前预拆分并严格恢复 Adam state、组标签和各组 LR。诊断 `resume_finetune` 已改为 `{}`，避免诊断恢复时暗改源 checkpoint 的 optimizer/PPO 设置。

### 验证与部署

- 远端真实 SKRL 已通过 branch-aware optimizer、分桶 KL、零命令 mean 隔离、命令条件探索和 `compileall`。在旧 `agent_14000.pt` 上连续 8 次更新后，quiet KL 约 `0.004-0.039`、moving KL 约 `0.011-0.017`，quiet LR 自动稳定在约 `4.05e-6-9.11e-6`；双参数组 `agent_2000.pt` 也已严格 resume 成功。
- 当前有效 payload：`/root/gpufree-data/training_payloads/taili_blind_runtime_phase0_branch_aware_quiet_optimizer_20260730_verified_v2`，archive SHA256 为 `461a0eabcf605bfbf289579e507946d799992117db7a3db071c1055d590544da`。无 `v2` 的旧 payload 保存间隔仍为 `2k`，不能用于正式训练。
- 当前正式 run：`/root/gpufree-data/taili_runs/taili_train_20260730_011500_phase0_branch_aware_quiet_optimizer_fresh`。`run.json` 已确认 `resume_checkpoint` 为空、1024 envs、seed 42、checkpoint 间隔 `10k`。
- fresh 的第一个 quiet 更新存在初始化瞬态：quiet KL `28.55 -> 3.20 -> 0.173 -> 0.052 -> 0.025 -> 0.019 -> 0.012`，quiet LR 同时由 `4e-6` 降到 `1.2e-6`，随后恢复至约 `9.1e-6`；同期 moving KL 约 `0.010-0.033`。到第 12 次更新，quiet/moving KL 均约 `0.02`，说明分支调度已稳定，不能因首批瞬态停训。

### 继续监控规则

- 修改、打包和静态验证与在训 run 并行完成，不因准备候选而停止训练。只有趋势或物理诊断已证明当前语义需要干预，并且替代方案已验证完毕，才在切换瞬间受控停止和重启。
- 约每 10 分钟联合查看 raw F/B/L/Y、stand/capture、strict moving core、height/tilt/wxy、touchdown/trajectory/duty/diag/slip，以及 quiet/moving KL 和各自 LR。日志尚未联合成型前不做物理诊断；成型后使用同协议全方向诊断验收，再恢复楼梯、推进 DR，最后完成 sim2sim。

### 首个约 10 分钟窗口

- `3k` 当前 raw F/B/L/Y=`0.113/0.108/0.124/0.236`，源 `phase0_isolated_zero_mean_fresh` 同期为 `0.135/0.099/0.113/0.258`，方向形成基本同轨，不能判为新优化器阻塞。
- 当前相对源 run 的 strict core 为 `0.282` 对 `0.271`，高度误差 p95 `9.03 cm` 对 `10.69 cm`，wxy mean `0.857` 对 `0.927 rad/s`，touchdown p95 `0.838` 对 `0.754 m/s`，轨迹最差 p95 `0.644` 对 `0.751`。物理质量仍远不合格，但多面正在改善而非整体坍缩。
- 当前 stand/capture=`0.312/0.323`，源同期为 `0.396/0.346`；capture 接近而 stand 偏低，是需要跨后续窗口持续跟踪的风险，尚不足以停训。当前结论为继续训练、不改配置、不做未成型物理诊断。

## 2026-07-30 02:30：非自消失核心路径 fresh

- `phase0_branch_aware_quiet_optimizer_fresh` 已在 `21310` step 受控停止，保留 `agent_20000.pt` 作为失败语义证据，不用于 resume。其约 `16k -> 21.3k` 期间主要是 `wxy` 改善推动 readiness；机身均值高度仍约 `0.514 m`、tilt 约 `6.3 deg`、strict flat core 约 `0.411`、stand gate 约 `0.262`，没有形成全局联合改善。
- 10k 全方向诊断确认移动高度仅约 `0.454-0.467 m`、各方向 pitch 约 `-13 deg` 至 `-16 deg`、tilt 均值约 `14-17 deg`，零命令 planar p95 `0.376 m/s`。根因是平地 `orient/base_vz/base_wxy/base_ang_accel/flat_move_height` 纠偏都乘 `stable_motion_gate`：高度或姿态越差，纠偏成本反而越接近零；tracking 同时由固定 `flat_core_credit_floor=0.35` 保留信用，导致低身前倾成为可盈利路径。
- 新 payload 为 `/root/gpufree-data/training_payloads/taili_blind_runtime_phase0_non_erasing_core_path_20260730_verified`，archive SHA256 `38bc13e977d7c8fed7c316e86f1b1cea48cf28b9371b02e534cf05913f458b6a`。平地核心纠偏保留最低 `0.35` 的恢复门，仅真实 terminal 关闭；核心正奖励可随现有质量信号提前收紧最多 `0.35`；tracking 核心信用底线随能力形成从 `0.35` 连续降至 `0.10`。楼梯、DR、AMP、PD、命令采样、足端参考和静止分支优化器均未改。
- 当前正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260730_023000_phase0_non_erasing_core_path_fresh`。`run.json` 已核对 payload 正确、`resume_checkpoint` 为空、1024 envs、seed 42、`TAILI_INIT_PHASE=0`、checkpoint 间隔 `10k`；首个 PPO tick 正常，无 OOM/NaN/Traceback。
- 继续以约 10 分钟窗口监控 raw F/B/L/Y、strict core/readiness、stand/capture/readiness、height/tilt/wxy、touchdown/trajectory/duty/slip 和 quiet/moving KL/LR。只把多项联合趋势视为改善；方向能力形成前不做低信息物理诊断。允许正式训练与单个远端诊断并行，但禁止本地并行运行多个 PyTorch 测试。

## 2026-07-30 04:15：制动探索死锁与方向信用擦除

- `phase0_non_erasing_core_path_fresh` 从 `18k -> 28.8k` 的 moving-core quality/readiness 约由 `0.491/0.043` 升至 `0.579/0.337`，tilt、wxy、高度均明显改善；但 stand 始终约 `0.25-0.27`，capture 始终约 `0.06-0.08`。同一窗口 yaw progress 从约 `0.65` 单调降至约 `0.20`，不是全局共同改善。
- 20k 系统全方向诊断为 `/root/gpufree-data/diag_runs/locomotion_console_20260729_194105_directions`。前进约 `0.487/0.5 m/s`，但后退、横移、yaw 分别仅约 `0.337/0.4 m/s`、`0.091/0.3 m/s`、`0.263/0.6 rad/s`；横移/yaw slip p95 约 `0.948/1.448 m/s`。F/B/L 切零后速度 p95 仍约 `0.523/0.532/0.470 m/s`，动作 mean 却在 1-2 帧内由约 `0.62-0.65` 降至约 `0.04`，证明当前不是主动制动，而是瞬间撤掉运动动作。
- 静止分支 mean 立即切换本身保留；根因是训练方差也立即切到 `log_std=-4.5`，即 `sigma≈0.011`。刚切零、仍需主动制动的 quiet 分支既无法采到运动幅值的制动动作，其 Gaussian mean 梯度又被 KL equalizer 同步压小，因而增加 stand 权重也不能解决探索死锁。
- 第二个根因是 `flat_core_credit_floor` 的收紧误用了随训练时间完成的 `quality_gate/_penalty_gate`，而非已经计算出的真实 moving-core readiness。约 20k penalty ramp 完成后，tracking 信用底限直接由 `0.35` 收至 `0.10`；尚未同时形成严格核心的 yaw 因而被擦除。移动核心正向路径本身有效，必须保留。
- 新候选为 `output/target_28750_rework/staging_phase0_capture_exploration_readiness_20260730`。mean 和 optimizer 分桶仍用当前命令 gate；训练方差改用 `max(command_gate, recent_q_des_history_gate)`。已有 actor 本体历史中 `q_des_rel[24:36]` 的 25 帧峰值 RMS 以 `0.50 -> 0.90` smoothstep，使切零后的约 `0.5 s` 保留移动尺度探索，历史清空后恢复原低噪声端点；不增加部署输入，不修改命令，也不是动作状态机。
- 候选同时把 tracking 核心信用底限改为由 `quality_refinement_gate` 收紧，并把课程/遥测 progress 改为保留 signed ratio、求均值后再截断，避免零均值随机运动被逐样本下截断抬成正能力。任务 tracking reward 未改，楼梯、DR、AMP、PD、action scale、步态和足端逻辑均未改。
- 本地 compile、全部可用 verifier 和远端 CPU `verify_zero_command_mean_isolation.py` 均通过。验证明确保证 moving mean/log-std 端点不变、settled quiet 端点不变、capture history 只延长方差、静止/移动参数梯度隔离不变；候选验证完成前旧训练未停止。
- 已提升的 payload 为 `/root/gpufree-data/training_payloads/taili_blind_runtime_phase0_capture_exploration_readiness_20260730_verified`，文件清单 SHA256 为 `cd2500178f068a18771b8b1ae6b1dcc4674f3476d0e7a3e674a8f63e7e6cfe0d`。旧 run 在 `32k` 受控停止并保留 `agent_30000.pt`。
- 新正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260730_042404_phase0_capture_exploration_readiness_fresh`；`resume_checkpoint` 为空、1024 envs、seed 42、单一 trainer，首个完整遥测已到 `150` step，无 OOM/NaN/Traceback。继续采用 10 分钟间隔监控；未形成可观察运动前不做低信息诊断，允许正式训练与一个远端系统诊断并行，本地 PyTorch 验证始终串行。

### 当前 fresh 的 10k/20k 交叉验证

- `10k` 系统全方向诊断为 `/root/gpufree-data/diag_runs/locomotion_console_20260729_210330_directions`。确定性 mean 尚未形成四方向运动，F/B/L/yaw 的中位跟踪误差约为 `0.50/0.45/0.32/0.60`；移动段 tilt p95 约 `18-19 deg`、高度误差 p95 约 `8.7-10.4 cm`。该点只能作为早期反例，不能据此验收制动探索修改。
- 约 `12k -> 21.6k` 出现联合改善：signed F/B/L/yaw 由约 `0.61/0.41/0.14/0.18` 发展到约 `0.77/0.45/0.45/0.81`；moving-core quality/readiness 到约 `0.50/0.063`；stand/capture 到约 `0.46/0.21`；flat tilt p95、wxy、height error p95 约为 `4.1 deg/0.44 rad/s/3.1 cm`。`penalty_gate` 已到约 `0.69` 而 yaw 未像上一轮一样被擦除，支持信用底限改由真实 readiness 收紧的修复有效。后退有回落，stand/capture 尚未通过，不能宣称收敛。
- `20k` 同协议系统诊断为 `/root/gpufree-data/diag_runs/locomotion_console_20260729_214037_directions`。确定性 F/B/L/yaw 稳态实际值约为 `0.38/-0.26/0.15/0.42`，对应目标 `0.5/-0.4/0.3/0.6`；移动段 tilt p95 约 `1.6-2.5 deg`、wxy p95 约 `0.28-0.35 rad/s`、高度误差 p95 约 `1.2-1.6 cm`。真实 touchdown |vz| p95 约 `0.04-0.16 m/s`，明显低于训练 rollout 的约 `1.0-1.5 m/s`，说明训练尾值受探索动作显著影响；但诊断 stance-slip 尾值仍高，尤其后退约 `1.43 m/s`，必须继续核对实际支撑滑移。
- 动转静已经从“瞬间撤动作且不制动”变为真实减速，但仍不安静自然。F/B/L 切零后沿原命令速度分别约从 `0.385/0.441/0.213 m/s` 衰减，在约 `0.3-0.5 s` 后反向过冲，反向峰值约 `0.212/0.132/0.164 m/s`；最后约 `0.3 s` 的 planar speed p95 仍约 `0.219/0.174/0.177 m/s`，后退在 `0.8 s` 窗口内未稳定。quiet mean RMS 在制动窗口约 `0.13-0.22`，不再立刻降到旧诊断的约 `0.04`，说明新增探索路径有实际作用，但主动制动的阻尼、过冲和保持静止仍未解决。
- 当前结论是继续训练并按约 10 分钟窗口观察，不修改、不重启。下一次干预必须基于趋势和确定性物理诊断：确认 stand/capture 是否继续提高、后退是否恢复、核心各分量是否共同改善，以及制动过冲是否随学习减弱。训练可与单个远端系统诊断并行；本地 PyTorch 测试始终禁止并行。

## 2026-07-30 06:30：站立严格核的中间质量死区

- `30k` 同协议全方向诊断为 `/root/gpufree-data/diag_runs/locomotion_console_20260729_221152_directions`。相对 `20k`，确定性 F/B/L/yaw 实际值由约 `0.382/-0.258/0.153/0.415` 改善到 `0.466/-0.377/0.294/0.488`，但动转静没有共同改善：F/B/L 反向过冲峰值约 `0.219/0.220/0.154 m/s`，尾段 speed p95 约 `0.218/0.221/0.154 m/s`，后退和横移在 `0.8 s` 内均未稳定。quiet mean 切零后快速接近固定站立动作，后续变化很小，表现为依赖 PD 静态姿态的欠阻尼反向过冲，而不是充分的状态相关制动。
- 训练约 `30k -> 34.9k` 时，raw F/B/L/yaw 大体保持在 `0.78-0.96`，moving-core quality/readiness 由约 `0.600/0.431` 升至 `0.627/0.556`，gait/diag/slip 也继续改善；但 stand/capture 分别只在约 `0.52-0.56/0.26-0.32` 震荡，其几何均值只在 `0.379-0.418`，严格 stand readiness 始终严格为 `0`。日志中的 stand/capture 小幅上升没有转化为诊断中的安静制动。
- 根因是当前 `flat_stand_readiness_start/full=0.45/0.65`，而 readiness 输入为 `sqrt(flat_stand_quality * flat_capture_quality)`。宽核把联合质量推到约 `0.38-0.42` 后，严格吸引子与 persistent 信用仍保持关闭；策略停在宽核可接受的中间水平，并且无法靠同一条关闭的严格路径跨过 `0.45`。capture 探索、四方向驱动和移动核心路径均已证明有效，不能随此问题一起删除。
- 新候选为 `output/target_28750_rework/staging_phase0_stand_readiness_continuation_20260730`。只把 `flat_stand_readiness_start/full` 改为 `0.28/0.60`：低于 `0.28` 时完全保留原宽核启动；当前约 `0.38` 的真实质量开始温和接入严格核；达到 `0.60` 后才完整收紧。它仍由真实 flat stand/capture EMA 驱动，不使用轮数门，不新增惩罚、状态机、命令过渡或动作规定。
- 与当前 payload 的目录差分仅有 YAML、同义默认值及 verifier 共 4 个文件；AMP、PD、action scale、四方向、足端、楼梯、DR、capture 探索和分支优化器均未改。本地可运行的 13 项 verifier 与 `compileall` 通过；本机缺少 `gymnasium` 的最后一项已在远端 CPU 环境补跑，远端全部 14 项串行通过。已验证 payload 为 `/root/gpufree-data/training_payloads/taili_blind_runtime_phase0_stand_readiness_continuation_20260730_verified`，archive SHA256 为 `497d403d5ce3e1c59ea0aefa504fd1a7d4ebe6724311a99e1445c60311b141ec`。

## 2026-07-30 13:30：严格站立核与移动方向的 PPO 优势耦合

- `phase0_stand_readiness_continuation_fresh` 因远端实例中断停在约 `28.2k`，保留 `agent_20000.pt`。从该点恢复的 `taili_train_20260730_122710_phase0_stand_readiness_host_recovery_resume20k` 明确重演了新回归：严格站立 readiness 接入后，yaw progress 在新计数约 `2k/3k/4k/5k` 由 `0.631/0.514/0.288/0.090` 持续坍缩；F/B/L、移动核心和 gait 同期没有同幅坍缩。因此问题不是训练整体失稳，也不能靠等待解释。
- 远端真实 SKRL `AMP._update` 会对 quiet 与 moving 样本统一执行全批 advantage 标准化。即使 quiet/moving actor mean、Adam 参数组和 KL scheduler 已隔离，严格站立收益改变仍会移动全批 advantage 均值，使较弱的 moving 任务首先得到系统性负偏置。修复只改变优化口径：先按 quiet/transition/moving 分支分别减去各自均值，再用一个全局标准差统一恢复 PPO 步长尺度，即 `A'_i = (A_i - mean(A | branch_i)) / std(A - mean(A | branch))`。GAE、returns、critic、奖励、课程、动作和分支内信用排序均不变。
- 新候选为 `output/target_28750_rework/staging_phase0_branch_centered_advantages_20260730`；本地归档 `phase0_branch_centered_advantages_20260730.tar.gz` 的 SHA256 为 `7ca5535aeea49f41d5c87ecc841030d3aa9bdded4a14f4a59853aa1816dc2bb0`。远端 15 项 verifier 全部串行通过，包括真实环境中的 zero-command mean 隔离；64 环境、96 步的真实 SKRL smoke 完成 3 次 PPO update，每次 quiet/moving 样本齐全、两支 `mean_after` 约为 `1e-8`、输出全局标准差为 `1.0`，无 OOM、NaN、shape mismatch 或异常退出。
- 同一 `agent_20000.pt`、seed 42、1024 环境的短因果复核为 `taili_causal_20260730_branch_centered_adv_resume20k`。新旧约 `250` 步的 F/B/L/Y 与质量信号几乎同轨；到 `5k`，旧/new 的 yaw 为 `0.090/0.457`。新 run 同点 F/B/L=`0.782/0.775/0.795`、stand/capture=`0.528/0.252`、stand readiness 输入/门=`0.365/0.172`、moving core=`0.542`、gait/duty/slip=`0.896/0.579/0.073`，证明严格站立核已经生效，同时没有以其他方向或核心质量交换 yaw。因果 run 在 `5.325k` 后优雅停止；resume 只作为根因验证，不作为最终训练路径。
- 下一步把该候选提升为正式 payload，并从空权重、1024 环境、seed 42 启动 fresh。继续按约 10 分钟联合监控四方向、stand/capture/readiness、移动核心、高度/倾角/wxy、步态/足端、quiet/moving KL/LR 和优势分支统计；日志联合成型后才做系统全方向物理诊断。平地通过后恢复楼梯，再推进 DR，最后完成 sim2sim。

## 2026-07-30 14:50：停止 yaw 优化器重建，回到有效历史长训

- `command-conditioned advantage` 因果复核最终失败：同一 `agent_20000.pt`、seed 42、1024 环境到 `5.5k` 时，F/B/L 仍为 `0.770/0.865/0.808`，yaw 却降到 `0.019`。按方向减去 advantage 均值没有关闭回归，不能继续细分 PPO 桶，也不能提升该候选。
- 主线回到已验证的 `phase0_capture_exploration_readiness_fresh/agent_30000.pt`。该 run 最后 `35.2k` 的 raw F/B/L/yaw 为 `0.953/0.865/0.910/0.826`，moving-core quality/readiness 为 `0.636/0.593`；phase0 物理量为 tilt p95 `0.055 rad`、wxy `0.290 rad/s`、高度误差 p95 `0.0224 m`。这是最近同时保住四方向与移动核心的有效 fresh 历史，不再从失败分支重建。
- 它被 phase0 卡住的具体项是训练探索下 touchdown p95 `0.766 m/s` 对硬门 `0.60`，以及 stand/capture `0.561/0.296` 对硬门 `0.65/0.60`。确定性诊断已证明移动核心明显好于这些 rollout 尾值所表达的程度；静止和制动仍需优化，但不应无限阻塞楼梯阶段。
- 新长训 payload 只将 phase0 准入阈值改为 touchdown `0.90`、stand `0.50`、capture `0.25`。站立、制动、核心、足端与轻触地奖励全部保留；phase1 仍保留 `70%` 平地、完整四方向和 `20%` 动转静样本。其余奖励、AMP、PD、action scale、PPO、地形驱动和 DR 均与该有效历史完全一致。
- 后续直接从该 `agent_30000.pt` 长训，不再做 smoke/短因果复核。监控重点是历史能力是否保留、phase1 楼梯是否恢复增长，以及 stand/capture、核心、触地和制动是否继续改善；不能因为放宽课程准入就把未达标质量宣布为已解决。

## 2026-07-30 16:56：直接恢复历史 7+ 楼梯及完整课程状态

- 上一条长训路线选择了 `phase0_capture_exploration_readiness_fresh/agent_30000.pt`。它虽然保留了较好的四方向与移动核心，但其源 run 的课程状态为楼梯 `0`、DR `0`；恢复机制按设计精确恢复了这个零状态。因此该路线仍然是在从零学习楼梯，并未利用已验证的历史楼梯能力，现已停止。
- 可直接利用的历史终点为 `/root/gpufree-data/taili_runs/taili_train_20260723_214211_descent_quality_resume/checkpoints/agent_30000.pt`，原 payload 为 `/root/gpufree-data/training_payloads/global_stair_dr_descent_quality_20260723_213358`。对应 `curriculum_state.json` 为 schema v4、step `30000`、phase `3`、DR `3`，包含与当前训练一致的 1024 个环境逐环境地形等级；历史遥测为下楼/上楼 `7.46/7.59`，frontier `8.44/8.69`。
- 当前正式长训为 `/root/gpufree-data/taili_runs/taili_train_20260730_history7_exact_curriculum_longrun`。启动时同时使用原 payload、历史 `agent_30000.pt`、`TAILI_RESUME_CHECKPOINT` 和 `TAILI_INIT_PHASE=3`；没有修改奖励、AMP、PD、PPO、yaw 或楼梯机制。启动日志已确认 `[CURRICULUM] restored exact state`，首条遥测严格恢复为 `phi3`、DR `3`、下楼/上楼 `7.463/7.589`、frontier `8.437/8.687`。
- 后续以这条路线为主线长训，不再从零重建楼梯，也不做短 smoke。先观察历史楼梯、四方向和平地核心是否在训练更新中保持，再依据趋势与同协议物理诊断优化仍未解决的安静站立、制动、平地全局质量、楼梯安全质量和 DR；不能把恢复课程等级本身当作这些质量已经通过。

## 当前全局目标与执行边界（上下文压缩后必须先读）

- 历史 `7+` 检查点只是已验证能力底座，不是最终答案。直接使用原 payload 只能恢复四方向、较好的移动步态、楼梯和已有 DR；它没有数学理由自行补上原本缺失的安静站立、自然制动与更强鲁棒性。当前 run 在约 `2.7k` 时仍保留下/上楼 `7.40/7.57` 和 raw F/B/L/yaw `0.926/0.889/0.865/0.934`，但 stand/capture 仅 `0.021/0.038`、平均高度约 `0.514 m`、倾角约 `6.0 deg`，因此只能作为优化起点。
- 平地验收是完整联合要求：四方向准确跟踪；自然、轻柔、对称步态；髋无持续外翻/内翻；小腿和足端轨迹自然且不拖地；低滑移、轻触地、无颤抖；支撑、duty 和对角配合合理；前后横移稳定 roll/pitch/非命令 yaw，yaw 稳定 roll/pitch，四方向都稳定高度；零命令严格安静站立；动转静自然制动、无急停和反向过冲；静转动无高度突变；无偏航漂移和横向泄漏。
- 楼梯与平地、安静站立同级。主驱动是在机体坐标系内真实向前通过，速度大小在楼梯可适度放松；随后要求安全、稳定、合理步态和高通过率，不打滑、不高频大幅摇晃、不失稳、不磕碰肘部或腿段。目标覆盖约 `20-30 cm` 台阶；课程等级只是趋势信号。禁止用状态机、势能换层、固定落脚点、固定腿序或逐腿 trace 代替任务语义。
- DR 首先覆盖正常 sim2real 参数差异、延迟和误差，再覆盖意外异常、较强延迟与推搡，最后强化高强度推搡、恶劣环境和负重；不能用名义平地或楼梯质量交换。actor 部署时不依赖特权地形信息。MuJoCo sim2sim 最后完成。
- 优化始终保留已验证有效部分，只调整或补充真实缺口。奖励提供主驱动，惩罚仅补安全边界；宽核负责可探索正向路径，严格核负责最终收敛，最大化总收益时必须走向目标行为。禁止再次重建 PPO、yaw 或楼梯机制来掩盖局部问题；日志看趋势，物理诊断看真实效果，并与实际配置和底层计算交叉验证。候选准备完成前不停止当前训练。

## 2026-07-30 17:20：历史 7+ 基础上的首轮全局优化

- 历史 payload 的明确静止冲突是：actor 的零命令 gait clock 已被屏蔽，task 侧也已有 stand/capture 正奖励，但 AMP 判别器不读取命令、参考库没有命令条件静止片段；零命令样本仍领取完整 `style_reward_weight=2.0`、`discriminator_reward_scale=3.0` 的运动风格信用。因此策略在零命令下持续运动并非只因 stand 权重不足，而是 task 静止目标与 AMP 运动目标直接竞争。
- 新 payload 以历史原 payload 为主体：`/root/gpufree-data/training_payloads/history7_global_optimize_20260730_verified`，archive SHA256 `b6eca848a86b50c09e870ee94884c59323efcfd11fcf52c97cf54dcf9015737d`。只新增命令条件 AMP scale：线速度命令幅值在 `0.03 -> 0.15 m/s`、yaw 命令幅值在 `0.02 -> 0.18 rad/s` 间使用 smoothstep 从 0 过渡到 1，取两轴最大值后乘原 terrain AMP scale。正常运动完整保留历史 AMP，命令趋零时连续让位，零命令时 AMP 风格信用严格为 0；不修改命令、actor 输入或动作，不是状态机。
- phase3 平地 `stand_prob` 从 `0.10` 提到 `0.20`，为成熟 actor 提供足够静止样本；楼梯环境仍保留原强制前进主任务。stand/capture/core、楼梯、DR、四方向、AMP 总权重、PD、action scale、PPO 和 actor 结构均保持历史原值。没有移植后期上万行演化，也没有移植失败的 branch/command-conditioned advantage。
- 当前正式 run：`/root/gpufree-data/taili_runs/taili_train_20260730_history7_global_optimize`。它从历史 `descent_quality_resume/agent_30000.pt` 与同一 `curriculum_state.json` 恢复；启动已确认 `phi3`、DR `3`、下/上楼 `7.463/7.589`、frontier `8.437/8.687`。首批全局 stand gate 约 `0.112`，高于原精确恢复 run 的约 `0.067`；楼梯环境不采样站立，因此全局值不会等于 YAML 的 `0.20`。
- 接下来按约 10 分钟窗口与原精确恢复 run 同步比较：楼梯等级/frontier、raw F/B/L/yaw、stand/capture、核心/高度/倾角、fall，以及 AMP 有效 scale。只有静止与制动改善且历史移动和楼梯能力保持，才继续；若楼梯或方向持续回落，先判定共享更新干扰，不通过重造 PPO 或楼梯奖励补救。日志形成趋势后再做同协议全方向物理诊断，不能仅凭 stand 日志上涨验收。

### 首个约 10 分钟同轮数窗口

- 当前 run 到 `3.6k`、约 `9m52s` 时仍只有一个 trainer，未见 OOM、NaN 或 traceback；数据盘剩余约 `5.7 GB`。同轮数比较必须把 stand/capture 除以实际 `stand_gate`，否则 `stand_prob` 增加会被误读为能力提高。
- 最近 `2.7k-3.6k` 窗口相对原 payload 精确恢复 run 的同期均值：下/上楼为 `7.468/7.554` 对 `7.384/7.575`，frontier 为 `8.446/8.616` 对 `8.279/8.659`；raw F/B/L/yaw 为 `0.914/0.886/0.860/0.947` 对 `0.914/0.893/0.857/0.935`。这些差异都属于保持或正常波动，没有楼梯或方向交换。
- 当前静止样本占比约 `0.111`，原同期约 `0.068`。按实际静止样本归一后，stand/capture 质量为 `0.362/0.594`，原同期为 `0.336/0.575`，分别提高约 `0.026/0.019`；当前最新 stand 归一质量已到 `0.388`。这只是训练分布中的早期正趋势，不能替代固定零命令和动转静物理诊断。
- 移动核心均值当前为 `0.240`，原同期 `0.253`；倾角为 `5.776 deg` 对 `5.706 deg`，高度 `0.520 m` 对 `0.517 m`。差异很小但方向略差，必须继续确认静止收益没有交换移动核心。gait/duty/slip/touchdown 与原同期大致持平，fall 未增加。
- 当前判定为继续训练、不改配置。再观察至少一个约 10 分钟窗口；只有 stand/capture 归一质量持续提高、楼梯和四方向保持，并且移动核心不持续回落，才启动同协议全方向物理诊断。

### `agent_8000.pt` 严格同协议平地诊断

- 默认系统 `directions` 诊断 `20260730_085047` 只有初始约 `1.8s` 静止、各移动方向后 `0.8s` 切零，且 yaw 后没有静止段，不能与历史长静止诊断直接验收。随后通过系统启动 `20260730_085613`，使用历史 `20260725_051221_directions` 完全相同的 9 段协议：初始静止 `2s`，F/B/L/yaw 各 `4s`，每个方向后静止 `4-5s`；只按既定要求把环境数从历史 8 改为 1。当前记录为 `/root/gpufree-data/diag_runs/locomotion_console_20260730_085613_directions/record.csv`，历史记录为 `/root/gpufree-data/diag_runs/locomotion_console_20260725_051221_directions/record.csv`。
- 静止方向有真实改善，但尚未完成。初始静止的 planar speed p95 从历史 env0 的 `0.093` 降至 `0.072 m/s`，tilt p95 从 `2.41` 降至 `1.34 deg`，wxy p95 从 `0.122` 降至 `0.097 rad/s`，足端速度 p95 从 `0.222` 降至 `0.035 m/s`。各方向后 settled planar speed p95 当前约为 F/B/L/yaw=`0.076/0.055/0.071/0.075 m/s`，历史 env0 为 `0.084/0.084/0.084/0.083 m/s`；多数改善，但仍不属于严格安静站立。
- 以 planar `<0.03 m/s`、wxy `<0.10 rad/s`、`|wz|<0.05 rad/s`、`|vz|<0.03 m/s` 连续 `0.3s` 为统一严格观测口径，当前 F/B/L 后在 `4s` 内均未稳定，只有 yaw 后约 `4.04s` 达到；历史 env0 四个方向均未达到。当前反向过冲峰值 F/B/L/yaw=`0.020/0/0.016/0.025`，历史 env0=`0.028/0.026/0.047/0.081`，证明制动方向改善，但轴向加速度 p95 仍约 `1.83/1.16/1.71 m/s^2` 和 yaw `3.67 rad/s^2`，自然制动尚未验收。
- 移动能力总体保持或改善。稳态 tracking error p95 当前 F/B/L=`0.049/0.062/0.051 m/s`，历史 env0=`0.072/0.115/0.081 m/s`；yaw 当前 `0.079 rad/s`，历史 `0.069 rad/s`，略差。移动 tilt p95 四方向均优于历史；但核心并非所有分量共同改善：前进高度误差 p95 `2.60 cm` 对历史 `2.20 cm`，横移 wxy/vz p95 `0.269/0.096` 对历史 `0.243/0.068`，yaw 高度误差 `1.72 cm` 对历史 `1.30 cm`。这些是继续监控项，不能被 tilt 单项掩盖。
- 训练端 `stance_slip≈0.05` 是高滑移样本比例，不是物理速度。诊断中排除接触切换并要求前后各连续 2 帧支撑后，当前/历史 env0 的支撑足滑移 p95 为：前进 `0.232/0.140`、后退 `0.087/0.177`、横移 `0.084/0.074`、yaw `0.033/0.028 m/s`；前进超过 `0.2 m/s` 的比例为 `5.5%`，历史 `3.4%`。因此日志与诊断语义可以对应，但前进支撑滑移是真实回归，横移有轻微风险，不能把原始接触边缘的 `0.6-0.8 m/s` 最大尾值直接当作持续滑移。
- 诊断结束时训练约到 `14.5k`：最近窗口下/上楼约 `7.51/7.59`，frontier `8.54/8.69`，raw F/B/L/yaw=`0.907/0.881/0.852/0.948`；历史能力未丢。stand/capture 按 stand 样本归一约 `0.373/0.600`，相对早期已改善但近期震荡，尚未证明继续上升。当前 checkpoint 实际每 `2k` 保存，已产生 `agent_2000...`，并非此前记录的 `20k`；数据盘约剩 `5.4 GB`，后续需要稀疏保留而不能任其写满。
- 当前决策是训练继续到至少 `20k`，不回滚命令条件 AMP，也不因站立未完成立刻加惩罚。到 `20k` 联合判断 stand/capture 趋势、楼梯/四方向、移动核心和前进滑移；随后对 `agent_20000.pt` 复用同一 9 段物理协议。若静止质量已平台，再从现有 stand/capture 正向路径的可达性、作用域和共享更新中追根因，禁止重新设计 PPO、yaw 或楼梯机制。

## 2026-07-30 18:00：历史 7+ 主线修复 stand_contact 奖励漏洞

- 全局目标不变：平地完整质量、严格安静站立和安全稳定楼梯同级；随后推进 DR，MuJoCo sim2sim 最后。当前明确未通过的是动转静后的严格静止、自然制动、前进/横移支撑滑移，以及与其相关的核心/高度过渡和脚下洁净度。历史楼梯与四方向能力仍是必须保留的有效底盘，课程数值不能替代楼梯安全质量验收。
- 同协议 20k 物理诊断与训练奖励分解共同确认根因：`stand_contact` 原实现只计算 `w_stand_contact * four_foot * stand_gate * ...`。最近 20 个训练窗口按实际静止样本归一后，strict stand/capture/stand_far/stand_contact 分别约为 `0.374/1.5`、`0.601/1.2`、`0.994/1.5`、`1.756/2.0`；机器人尚未安静时四足项已取得约 88%。这会从切零第一帧推动四脚同时压下，却不要求机身、关节和动作停止，解释了初始静止改善但动转静变硬、滑移与竖直/角运动增加、严格静止平台。
- 新候选从已部署的 `output/history7_global_optimize_20260730` 独立复制为 `output/history7_stand_contact_capture_coupling_20260730`。训练语义只增加 `stand_contact_capture_floor: 0.35`，并将接触信用改为 `w_contact * four_foot * (0.35 + 0.65 * stand_capture_score) * stand_gate * nonterminal * recovery`。未捕获时保留 35% 四足支撑正向底线，避免依赖惩罚逼出急停；完整接触信用必须同时满足宽核的机身、关节和动作安静。楼梯、四方向、AMP 运动段、PPO、PD、action scale 和 DR 均未修改。
- 本地及远端均串行通过 `verify_core_stand_capture.py`、`verify_command_conditioned_amp_style.py`、`verify_stand_gait_clock_mask.py`、`verify_terrain_descent_reward.py` 和全包 `compileall`。归档 `history7_stand_contact_capture_coupling_20260730.tar.gz` 的 SHA256 为 `b40f689ce046d7983b5d85b485a0d5a05921d4a8fe805bd6726ce108e31ea586`，远端 payload 为 `/root/gpufree-data/training_payloads/history7_stand_contact_capture_coupling_20260730_verified`。
- 旧正式 run `taili_train_20260730_history7_global_optimize` 在切换前训练到 `30k`。已把 `agent_30000.pt` 与同一步的 schema v4 课程状态冻结到 `/root/gpufree-data/taili_runs/taili_snapshot_20260730_history7_contact_switch`，状态为 phase3、DR3、1024 个逐环境地形等级；旧 trainer 随后安全停止。
- 新正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260730_history7_stand_contact_capture_resume30k`。启动日志确认 `[CURRICULUM] restored exact state step=30000`、`real_mean=7.551`、DR3。首批 step 100 的下/上楼为 `7.493/7.584`，step 400 为 `7.449/7.595`，四方向 raw progress 为 `0.899/0.901/0.822/0.953`，说明切换没有重置历史能力。step 400 的 stand/capture/contact 为 `0.0354/0.0637/0.1289`，新的 contact 信用已在 capture 较低时按设计收缩；400 步不足以判定能力改善。
- 后续按约 10 分钟窗口监控，不做并行本地 PyTorch。先确认唯一 trainer、无 NaN/OOM/持续 fall，楼梯和四方向不持续回落；再看按实际 stand_gate 归一的 stand/capture/contact 是否朝“capture 提升、contact 随之恢复”演化，以及核心、高度、duty、slip 和 touchdown 是否没有被交换。形成趋势后复用同一九段系统诊断，重点验收前进/横移后的严格静止时间、制动加速度、wxy/vz、joint/foot velocity 和稳定支撑滑移。

### 约 27.7k 的趋势判定

- 新公式在运行时已经生效，并非 payload 未加载：按约 5k 分窗，stand/capture/contact 的静止样本归一值依次为 `0.365/0.594/1.214`、`0.377/0.607/1.246`、`0.374/0.602/1.225`、`0.370/0.599/1.219`、`0.367/0.598/1.217`、`0.363/0.597/1.208`。contact 已随 capture 收缩，而不是像旧公式一样仅凭四足接触接近 `2.0`。
- 行为改善没有持续：5-10k 的短暂上升随后回落并平台；moving core 约 `0.24`，平均 tilt 从首窗约 `5.93 deg` 到末窗约 `6.17 deg`，没有全局核心改善。同期下/上楼约 `7.49/7.57`、四方向保持，说明这是站立路径不足而不是历史能力坍缩。
- 若同协议物理诊断确认严格静止、制动、wxy/vz、joint/foot velocity 和稳定支撑滑移没有改善，根因应从 `stand_capture` 本身继续追：当前宽核尺度 speed/yaw/wxy/vz/tilt/joint/action/height 为 `0.35/0.45/0.65/0.25/0.20/0.80/0.20/0.08`，对诊断中的明显残余运动仍给中等分；contact 又线性继承该宽核，所以可能形成约 0.5-0.6 capture 的新局部最优。实际静止样本约 11%，共享更新也会稀释站立梯度。
- 调整顺序不能直接加惩罚或改楼梯：先用最新可用 checkpoint 复用九段诊断，定位宽核中真实最差分量。随后保留宽核远场正向梯度，但将完整 contact bonus 改由 capture completion（例如对 capture 使用连续 `smoothstep`）接管，并按诊断只收紧真实失效尺度；必要时把 phase3 flat `stand_prob` 从 `0.20` 小幅提高到约 `0.30`，提高实际静止样本而不改变楼梯命令。若终态改善但急刹仍在，再补小权重、带安全质量的有符号捕获势差；不能用无符号减速奖励、状态机或命令过渡制造急停/振荡漏洞。

### agent 60000 同协议物理诊断

- 系统诊断 job 为 `20260730_122722`，远端记录 `/root/gpufree-data/diag_runs/locomotion_console_20260730_122722_directions/record.csv`，本地副本为 `output/diagnostics/20260730_122722_history7_contact_60000/record.csv`。使用与历史源、旧 8k、旧 20k 完全相同的九段、单环境、DR0、平地协议；前端回放已接入。
- stand-contact 耦合并非无效。当前 60k 的前进/后退/横移/yaw 转静后，分别约在 `3.04/3.96/3.56/4.06 s` 达到连续 `0.3 s` 的统一严格机身静止；旧 20k 只有后退和 yaw 达到。当前各转静尾段 planar speed p95 约 `0.028/0.031/0.050/0.021 m/s`，joint velocity RMS p95 约 `0.120/0.129/0.127/0.108`，足端三维速度 p95 约 `0.042/0.040/0.038/0.040 m/s`。因此问题从“无法静止”推进为“达到太慢、初始站立仍未严格通过、姿态与制动质量未完成”。
- 新的明确漏洞是用持续俯仰偏置换取低速度。当前初始站立尾段 pitch 均值约 `+4.02 deg`，F/B/L/yaw 后站立约 `+3.28/+2.91/+2.73/+3.12 deg`；旧 20k 对应值约 `+0.49/-0.67/-0.61/-0.37/-0.43 deg`。当前后退移动 pitch 均值 `+3.55 deg`、最大 `6.36 deg`，倾角 p95 `6.08 deg`；旧 20k 后退倾角 p95 `2.56 deg`。宽核尾段 `q_tilt` 仅约 `0.70-0.78`，是机身分量的真实短板。
- 当前 contact 只线性依赖宽核 capture，而 `stand_capture_tilt_scale=0.20 rad` 对约 `3-4 deg` 静态俯仰仍给 `0.70-0.78`；平地 tracking 的 `tracking_tilt_target/width=0.04/0.18 rad`，再叠加 `tracking_motion_floor=0.55` 与 `validated_tracking_floor=0.55`，使后退约 `6 deg` 倾斜仍能取得较高主任务信用。这解释了日志 capture 平台、物理静止改善但核心姿态回归。
- 移动质量不能宣告通过：当前 tracking error p95 F/B/L/yaw 约 `0.077/0.179/0.159 m/s(or rad/s)`，前进和 yaw 改善，但后退明显回归、横移仍不足；稳定支撑滑移 p95 F/B/L/yaw 约 `0.200/0.080/0.087/0.028 m/s`，前进滑移仍高。制动加速度较旧 20k 多数改善，但达到严格静止仍需约 3-4 秒，不能称作自然快速制动。
- 下一候选应协调而不是增加静止比例：保持 phase3 `stand_prob=0.20`，将 contact 完整信用改为 `floor + (1-floor) * Q_capture * Q_strict_stand`，使四足接触必须同时满足远场捕获和现有严格姿态/关节/动作核；把站立 capture 倾角尺度从 `0.20` 适度收紧到 `0.15 rad`；把仅在平地生效的 tracking tilt width 从 `0.18` 收紧到 `0.10 rad`。不改楼梯倾斜放松、tracking floor、其他宽核分量、惩罚、站立采样、楼梯/方向/DR/AMP/PPO/PD/action scale。先验证同一 actor 上的奖励排序，再受控 resume；后续诊断同时要求静止时间、pitch、后退跟踪和前进滑移不发生交换。

### 20:53：严格站立完成度接管完整接触信用

- 新候选为 `output/history7_stand_completion_core_alignment_20260730`。训练语义只做三处已由 `agent_60000.pt` 同协议诊断定位的调整：`stand_contact_scale = 0.35 + 0.65 * Q_capture * Q_strict_stand`、`stand_capture_tilt_scale: 0.20 -> 0.15`、仅平地生效的 `tracking_tilt_width: 0.18 -> 0.10`。phase3 `stand_prob=0.20` 保持不变；楼梯倾斜放松、tracking floor、其他 capture 分量、AMP、PPO、PD、action scale、DR、四方向和步态项均未改。
- verifier 增加了静态 `0.07 rad` 俯仰样本，明确要求其四足接触信用低于水平安静样本；接触公式的期望值同时使用 `stand_capture / w_stand_capture` 与 `stand / w_stand`。本地和远端 CPU 均串行通过四个 verifier 与全包 `compileall`。归档 `output/history7_stand_completion_core_alignment_20260730.tar.gz` 的 SHA256 为 `03e5f7909c5637b50716c9d11c3f42b8bce6061dfb603a74b0b56e95c287da7d`；远端 payload 为 `/root/gpufree-data/training_payloads/history7_stand_completion_core_alignment_20260730_verified`。
- 切换前从仍在运行的 contact-capture run 冻结了精确配对快照：`/root/gpufree-data/taili_runs/taili_snapshot_20260730_history7_core_alignment_switch`。其中 `agent_70000.pt` 与 `curriculum_state.json.step=70000` 一致，phase3、DR3、1024 个逐环境地形状态；同时保存来源 run、effective config、源 YAML 和 agent YAML。旧 trainer 随后受控停止。
- 新正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260730_history7_stand_completion_core_alignment_resume70k`。启动日志确认从上述快照 `[CURRICULUM] restored exact state step=70000`，`real_mean=7.593`、DR3，并加载正确的新 payload。首条 step 300 遥测为 stand/capture/contact=`0.0394/0.0643/0.0818`、stand gate `0.1113`，说明完整 contact 信用已按严格核明显收缩；同时平均 tilt `8.47 deg`，仍不能宣布核心改善。后续按约 10 分钟窗口联合验收 pitch/tilt、严格静止和自然制动、后退/横移跟踪、前进滑移、四方向、楼梯等级与安全质量，禁止只用单点日志判定。

## 2026-07-30 21:55：成熟 7+ actor 改为站立/核心专门训练

- `stand_completion_core_alignment` 从成熟检查点继续约 `14k` 后，站立样本仍只有约 `11%`；stand/capture/tilt 约为 `0.243/0.493/6.23 deg`，没有持续改善。优化器 LR 仍为 `1.5e-5`，不是冻结问题。结论是严格语义已经生效，但在 phase3 四方向、楼梯和 DR 的共享更新中只是附带训练，不足以改写成熟 actor 已固化的动转静行为。
- 专训 payload 为 `/root/gpufree-data/training_payloads/history7_stand_core_specialization_mix_20260730_verified`，本地归档 SHA256 为 `bea4baf45768b96f0881121f9a90df5495713cac80be2c0fe716797be3bbed02`。只把 phase3 `stand_prob` 从 `0.20` 提到 `0.60`，并把平地/下楼/上楼配额改为 `0.70/0.11/0.19`；实际 20 列布局为 14/3/3。奖励、AMP、PPO、PD、action scale、DR、四方向和楼梯机制均保持不变。目标是约四分之一真实静止样本、约四成平地移动和三成楼梯保持，不把站立继续当作附带任务。
- 首次直接恢复旧课程状态的 run `taili_train_20260730_history7_stand_core_specialization_resume14k` 暴露了课程布局兼容问题：`terrain_types` 保存的是列号而不是语义类型，改变列比例后旧等级被按新列解释，下/上楼降到约 `6.5`。该 run 在约 `5.4k` 受控停止，其权重不再使用；这不是 actor 能力坍缩，也不能作为专训效果证据。
- 新增可审计迁移工具 `tools/remap_taili_curriculum_layout.py`。它按 IsaacLab 同一列分配公式恢复旧、新列语义，并分别对每种楼梯的固定 replay 与 frontier 做分位数迁移；不改 actor 或优化器。旧状态下楼/上楼均值为 `7.532/7.595`、frontier 为 `8.573/8.698`；迁移后为 `7.526/7.601`、frontier 为 `8.571/8.688`。迁移状态 SHA256 为 `3e70d5285e44566d99fb696d35b6a2b08c7df198572b12d7037e7815b4d796bd`，报告 SHA256 为 `5a17090446454d5457b2c060af5255d0e46fd10c500b4f95d3d895427b4675ae`。
- 当前正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260730_history7_stand_core_specialization_remapped_resume14k`。它从 `/root/gpufree-data/taili_runs/taili_snapshot_20260730_history7_stand_core_specialization_remapped/checkpoints/agent_14000.pt` 恢复；该 checkpoint 与专训前成熟快照 SHA256 相同，没有使用错误布局 run 的权重。启动确认 phase3、DR3、`real_mean=7.564`。step 100 的 stand gate 为 `0.234`，raw F/B/L/yaw=`0.937/0.919/0.893/0.962`，下/上楼=`7.526/7.601`，frontier=`8.571/8.688`，说明专训覆盖和历史能力均已正确建立。
- 后续先看约 10 分钟窗口，不用单点宣布站立改善。必须联合观察按 stand gate 归一的 stand/capture/contact、tilt/core/height、四方向、下/上楼与 frontier、fall 和 DR。只有日志形成真实趋势后，才复用同协议九段物理诊断验收零命令静止、动转静时间、刹车过冲、pitch/wxy/vz、关节/足端速度和支撑滑移。
- 专训前成熟 run 最后 `13.3k-14.7k` 的同口径归一 stand/capture/contact 为 `0.371/0.592/0.782`，不能使用未归一总奖励比较。当前专训最近 `2.4k-4.3k` 为 `0.378/0.600/0.803`，只有很小正变化；同期 raw F/B/L/yaw=`0.926/0.913/0.900/0.959`，下/上楼=`7.552/7.608`，frontier=`8.623/8.701`，fall=`0`。当前证据支持继续到约 `10k`，尚不支持宣布站立解决，也不支持立刻重热 LR。若 `10k` 后归一质量仍与源窗口重合，下一步才受控处理成熟 checkpoint 的 `1.5e-5` 低 LR/优化器状态；不能用继续增加站立奖励、修改楼梯或 fresh 替代该判断。

## 2026-07-30 23:25：站立专训平台与单变量 LR 重热

- 正确迁移后的专训已运行约 `34k`。按连续约 `5k` 分窗，归一 stand/capture 依次约为 `0.367/0.586`、`0.383/0.604`、`0.371/0.598`、`0.366/0.596`、`0.385/0.605`、`0.385/0.605`；专训前成熟窗口为 `0.371/0.592`。提高站立覆盖后只有周期震荡和很小净变化，样本不足已排除，四方向、楼梯、gait/duty/slip 则保持或小幅改善。
- 直接在成熟 checkpoint 上 resume 早期失败曾包含两个真实奖励漏洞：零命令仍领取运动 AMP 信用，以及 `stand_contact` 可仅靠四脚压地领取大部分信用；两者均已修复。剩余平台不是旧 Adam 动量造成的，因为此前每次 resume 实际都执行 `reset_optimizer_state=true`。更准确的限制是成熟 actor 在当前专训分布下仍使用 LR `1e-5`、边界 `5e-6..1.5e-5`、epochs `2`、clip `0.10`、entropy `0.005`，共享能力保持良好但站立适配步长过于保守。
- 新候选为 `/root/gpufree-data/training_payloads/history7_stand_core_specialization_rewarm_20260730_verified`，归档 SHA256 为 `c0dce6b9d15ead7322a01593e6707a8589fd8581958c92ce7be909c23bc2a5c6`。它只改为 `reset_optimizer_state=false`、LR `2e-5`、边界 `1e-5..3e-5`；epochs、clip、entropy、奖励、AMP、采样、楼梯、DR、PD、actor 均不变。目的是保留已经在同一专训分布下形成的优化器状态，只放大适配步长，避免再次混入多个因果变量。
- 切换快照为 `/root/gpufree-data/taili_runs/taili_snapshot_20260730_history7_stand_core_specialization_rewarm_switch34k`。`agent_34000.pt` SHA256 为 `b615352a8086e7e20009c0f98e5f61cd40378ec4a7dda3343b5b5f54546b43e4`，课程状态 SHA256 为 `a7ba3c2a0091c14d97e5a1c5c461eca3df633bcf1efa9faafde468e14f139baa`；状态严格为 version 4、step `34000`、phase3、DR3。此前尝试复制 `32k` 时课程恰好更新到 `34k`，该不配对临时快照已删除，绝不能使用。
- 首次启动 rewarm 时未显式传 `--config`，启动器默认解析到了旧专训配置，日志显示 `reset_optimizer=1`、LR `1e-5`；该错误 run 仅运行 `222` 步即停止，不参与任何判断。随后 `--dry-run` 证明显式指定候选 YAML 后生成配置正确。
- 当前正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260730_history7_stand_core_specialization_rewarm_explicit_resume34k`。启动日志已确认课程从 step `34000` 精确恢复，`real_mean=7.603`、DR3，并确认 `reset_optimizer=0`、LR `2e-5`、epochs `2`、clip `0.10`、entropy `0.005`、LR 边界 `1e-5..3e-5`。后续先看 `5k-10k` 的归一 stand/capture 是否脱离旧平台，同时要求四方向、下/上楼及 frontier、fall、DR、gait/duty/slip 均保持；只有日志形成真实改善趋势后才做九段物理诊断。

## 2026-07-31：删除宽核，单变量验证严格零命令惩罚

- LR 重热线运行到约 `49.7k` 后判定为能力交换，已停止且不再使用其 actor。下一轮必须回到精确配对快照 `/root/gpufree-data/taili_runs/taili_snapshot_20260730_history7_stand_core_specialization_rewarm_switch34k/checkpoints/agent_34000.pt`，不能从 rewarm 检查点继续。
- 同协议源 `34k` 诊断的 950 个零命令帧显示，旧宽核允许明显残余运动停在中间分数：速度、关节速度、竖直速度、roll/pitch 角速度和足端速度越界率分别约 `75.6%/59.9%/38.9%/32.6%/27.4%`。因此宽核不是终态解法；但严格惩罚也不能被宣称为对共享 actor、critic 和 PPO batch 的其他能力“完全无影响”。
- 新候选为 `output/history7_strict_stand_penalty_only_20260731`，远端 payload 为 `/root/gpufree-data/training_payloads/history7_strict_stand_penalty_only_20260731_verified`，归档 SHA256 为 `7490159850F2EC7109EEEFF75345074353E5742F627CFA81C388DFB969C91D0D`。相对专训基线只删除 `stand_capture/stand_far` 宽核训练收益与 contact 的 `35%` 失败底线，保留原 `w_stand=1.5`、AMP、PPO、采样、楼梯、DR、PD 和 actor；新增仅在平地精确零命令下生效的严格 Huber 惩罚。
- 严格成本覆盖 planar speed、wxy、yaw rate、vz、tilt、height error、joint velocity RMS 和 max foot velocity；越界成本按 `0.65*max + 0.35*mean` 聚合并乘 `-0.15`。自由区分别为 `0.03 m/s`、`0.10 rad/s`、`0.05 rad/s`、`0.03 m/s`、`0.035 rad`、`0.02 m`、`0.10 rad/s`、`0.10 m/s`。Huber 尾部不饱和；near-zero、移动命令、楼梯和 terminal 不直接触发。
- 本地和远端均串行通过严格边界、八维逐项越界、尾部梯度、门控范围、contact/critic/总奖励守恒、命令条件 AMP、gait clock、下楼奖励及 `compileall`。正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260731_history7_strict_stand_penalty_resume34k`，启动确认 `1024` 环境、精确恢复 step `34000`、`real_mean=7.603`、DR3；首批遥测已出现 `stand_strict` 负值，且下/上楼约 `7.55/7.65`、fall=`0`。
- 验收必须同时看两条轴：零命令的 residual speed、pitch/tilt、wxy/vz、joint/foot velocity、制动过冲和达到严格静止时间是否改善；四方向、自然步态、核心/高度、滑移/触地、下/上楼课程与安全质量、DR 是否保持。严格惩罚若只改善静止而持续损害任一能力，仍判定失败，不能用直接门控范围为其辩护。

### `w=0.15` 的 10k 判定与 `w=0.30` 受控对照

- `w=0.15` run 训练到约 `11.5k` 后受控停止，保留 `agent_10000.pt` 作为失败证据但不继承其 actor。按 2k 分窗，`stand_strict` 为 `-0.0412/-0.0372/-0.0393/-0.0398/-0.0402`，没有持续下降；stand/contact、tilt 和核心也未形成改善。同期下楼从约 `7.55` 降到 `8-10k` 的 `7.50`、停止前约 `7.46`，fall 窗口略升，不能继续用训练轮数不足解释。
- 新候选为 `output/history7_strict_stand_penalty_weight030_20260731`，相对 `w=0.15` 版本的代码差分只有 `w_stand_strict_penalty: 0.15 -> 0.30` 及 verifier 断言；其余训练语义和优化器均不变。源诊断估算平均/p95/观测最大惩罚约 `-0.35/-1.48/-7.65`，观测最大值仍低于单次 `terminal_penalty=-10`，但共享策略和提前终止风险仍必须由训练验证。
- 本地及远端再次串行通过全部 verifier 和 `compileall`；归档 SHA256 为 `085B3DB9A475316EA4BBD9121DAE9808958FE25CCEE8136E18D3A477A5B60617`，远端 payload 为 `/root/gpufree-data/training_payloads/history7_strict_stand_penalty_weight030_20260731_verified`。正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260731_history7_strict_stand_penalty030_resume34k`，仍从同一精确 `agent_34000.pt` 和课程状态恢复，不继承 `w=0.15` actor。

### 严格惩罚可达性修复

- `w=0.30` run 在约 `10.3k` 受控停止，不继承其 actor。`8-10k` 对照中，`w=0.15/0.30` 的 stand 约 `0.0890/0.0900`、tilt 约 `4.306/4.346 deg`、下/上楼约 `7.504/7.621` 对 `7.551/7.646`；惩罚除以权重后的物理成本约为 `0.268/0.265`。权重翻倍只放大日志，没有改变严格静止行为，不能继续靠机械加权。
- 源 `34k` 检查点的实际 `log_std` 基本贴在历史逐关节下限，独立相邻采样的理论 action-delta RMS 为 `0.123987`。源同协议确定性诊断在命令切零 `1 s` 后的 `705` 帧中，action-delta Huber 越界成本均值约 `0.4041`，高于 speed/joint/foot 的约 `0.2416/0.2773/0.2156`；原八维严格成本没有直接包含该持续关节目标变化。当前失败同时包含直接驱动遗漏和训练探索使严格边界不可达，不能解释为单纯权重或轮数不足。
- 新候选为 `output/history7_strict_stand_penalty_reachable_20260731`，远端 payload 为 `/root/gpufree-data/training_payloads/history7_strict_stand_penalty_reachable_20260731_verified`，归档 SHA256 为 `ba0119e666b9c004bc9d1e9b871b804a948fbd76400beb8c794aa3386f8db780`。它保留无宽核、`w=0.30` 的严格负惩罚，并把 action-delta 作为第九项，free/scale 均为 `0.04`。
- 为使该负惩罚在训练分布中可达，移动命令逐位保留历史探索下限；切零后的近期动作历史继续保留移动方差用于主动制动；历史清空后零命令 `log_std` 端点为 `-3.6`，理论相邻 action-delta RMS 为 `0.038642 < 0.04`。没有独立 quiet actor、额外优化器、优势分桶、命令过渡、动作序列或状态机。
- 本地和远端已串行通过九维严格边界、总奖励守恒、静止/制动/移动方差端点、AMP、gait clock、下楼奖励、`compileall`，并用候选 actor 对源 `agent_34000.pt` 完成 strict state 加载；移动 `log_std` 逐位相等。该端点相等只保证移动采样的直接语义不变，不能保证共享 actor、critic 和 PPO 更新对四方向、楼梯、DR 完全无损，训练必须联合监控。
- 两条失败 run 仅各保留 `agent_10000.pt` 作为证据，其余重复检查点已清理；源配对快照未动。下一正式训练必须仍从 `/root/gpufree-data/taili_runs/taili_snapshot_20260730_history7_stand_core_specialization_rewarm_switch34k/checkpoints/agent_34000.pt` 开始，不能从 `w=0.15/0.30` 的失败 actor 继续。

### 条件降噪试验失败与严格净收益排序

- `history7_strict_stand_reachable_resume34k` 在约 `10.17k` 停止并保留 `agent_10000.pt` 作为失败证据，不继承其 actor。按 2k 分窗，`stand_strict` 为约 `-0.0613/-0.0640/-0.0628/-0.0592/-0.0617`，没有持续改善；后段归一 stand/contact 上升，但 tilt/core 未共同改善，F/L/Y 已出现小幅回落。结论是条件降噪改变了采样和易得分项，却没有建立严格静止物理路径。
- 根因不是“严格项尚未加权到足够大”这一句可以概括。精确零命令违规样本仍同时领取 `stand` 和 `stand_contact` 正信用，原 `0.30*C` 只是可被补偿的附加成本；因此奖励排序没有做到“违反任一静止物理边界必然比合格静止更差”。共享 actor 的零命令梯度仍可能改变移动、楼梯和 DR，任何门控都不能保证完全无损。
- 新候选为 `output/history7_strict_stand_penalty_dominant_20260731`，远端 payload 为 `/root/gpufree-data/training_payloads/history7_strict_stand_penalty_dominant_20260731_verified`，归档 SHA256 为 `8dd54ea5db973c6d64cc745b3644680a2fb20e067571a35ba41c5b37c157cf1b`。它恢复历史原始 actor/exploration，保留九维连续 Huber 成本；八个物理静止量中任一越界时，取消该平地精确零命令样本已领取的 stand/contact 正信用，再叠加连续严格成本。action-delta 继续直接受罚，但不作为硬合格开关，避免把 PPO 随机采样方差当成部署策略的物理违规。
- 远端串行通过严格边界、八维逐项违规信用取消、action-delta 连续惩罚、梯度和作用域、AMP、gait clock、下楼奖励及 `compileall`。正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260731_history7_strict_stand_dominant_resume34k`，PID `55775`；已确认从同一个 `agent_34000.pt` 精确恢复 step `34000`、phase3、DR3、下/上楼约 `7.55/7.65`。首批 `stand_strict` 约 `-0.317` 只证明违规正信用被实际取消，不代表行为已经改善。
- 后续约 10 分钟分窗联合看 strict、零命令 stand/core/tilt/height、四方向、步态/滑移/触地、上下楼和 DR。strict 必须持续向零收敛并由同协议物理诊断确认；若只改善静止日志，或以任何移动/楼梯/DR 能力持续下降交换，立即判定失败并回到同一 `34k` 源。

### 方向修正：站立先由任务量级的直接严格惩罚建立

- 上述取消 `stand/contact` 正信用的 run 在首批遥测后立即受控停止，未继承其 actor。它只证明信用取消代码生效，不符合当前确定的优化方向：站立首先依靠足够强的直接负惩罚得到明确效果，移动、楼梯和 DR 的一定阶段性损失属于预期交换，不能再以“完全无损”为前提削弱站立压力。
- 当前候选为 `output/history7_strict_stand_direct_penalty_20260731`，远端 payload 为 `/root/gpufree-data/training_payloads/history7_strict_stand_direct_penalty_20260731_verified`，归档 SHA256 为 `3b13fac089be5bd2bca07b7fc903ff4f6d205e38d3b28671040adb2db06691be`。它从未修改探索分布、未取消正信用的 `w=0.30` 候选出发，只补入遗漏的 action-delta 第九项，并把 `w_stand_strict_penalty` 提升到 `2.0`。
- 数学语义为 `C = 0.65 * max(H_i) + 0.35 * mean(H_i)`，`r_strict = -2.0 * I[平地且精确零命令且非 terminal] * C`。九项依次为 planar speed、wxy、yaw rate、vz、tilt、height error、joint velocity RMS、max foot velocity 和 action-delta RMS。没有宽核、条件 `log_std`、信用取消、独立 actor、状态机或额外门控。
- 远端已串行通过核心严格站立、命令条件 AMP、零命令 gait clock、下楼奖励及 `compileall`。当前正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260731_history7_strict_stand_direct_resume34k`，仍从固定配对源 `agent_34000.pt` 启动，检查点 SHA256 已核对；启动恢复 phase3、DR3、下/上楼约 `7.55/7.65`、fall=`0`，保存间隔为 `10000`。
- 首批 `stand_strict` 约为 `-0.87~-0.97`，说明直接惩罚已进入任务主量级，但不能据此宣布行为改善。先按连续时间窗验证 strict、stand、tilt/core/height 是否共同改善，并记录四方向、楼梯、fall 和 DR 的交换；日志形成可信变化后再做同协议物理诊断。

### 严格标准站立与非平地安全静止长训

- 上一轮只在平地生效且带自由区的严格项只能略微改善。当前语义改为零点唯一最优的连续 Huber：所有 `stand_strict_*_free=0`，scale 只归一化梯度。所有地形的精确零命令共同约束 planar speed、wxy、yaw rate、vz、joint velocity RMS、max foot velocity、support fault 和 contact fault；平地额外约束 tilt、标准高度、默认关节姿态和最差髋偏差。非平地不要求绝对水平、固定高度或默认关节姿态，允许适应台阶与坡面。
- PPO 采样 action-delta 含探索噪声，不能作为部署策略是否真实运动的判据，已从严格物理成本删除。移动奖励、楼梯通过奖励、AMP、PD、DR、PPO、actor 和既有移动髋约束均未改。楼梯样本明确划出 `15%` 零命令安全静止，其余 `85%` 仍全部执行原正向通过主任务，不让剩余楼梯样本退化为横移、后退或 yaw。
- 候选为 `output/history7_strict_standard_stand_terrain_hold_20260731`，远端 payload 为 `/root/gpufree-data/training_payloads/history7_strict_standard_stand_terrain_hold_20260731_verified`。本地和远端均通过 strict 逐项作用域、action-delta 隔离、平地/非平地姿态分离、AMP、gait clock、下楼奖励和 `compileall`；上传归档 SHA256 为 `723d9ac2c2df4f567e9c9370c02d3e7363e2e36e8a61f55f6aa1d3be10e93aa1`。
- 正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260731_history7_strict_standard_stand_terrain_hold_resume34k`，仍从固定配对源 `agent_34000.pt` 启动，源 SHA256 为 `b615352a8086e7e20009c0f98e5f61cd40378ec4a7dda3343b5b5f54546b43e4`。启动确认 `1024` 环境、课程 step `34000`、phase3、DR3、历史 `real_mean=7.603`；首窗下/上楼约 `7.54/7.65`、四方向约 `0.75-0.87`、fall 基本为零，说明恢复和布局正确。`stand_strict` 首窗约 `-1.4~-2.1` 只表示历史 actor 对新零点要求违约较大，尚不能判为改善或失败。
- 数据盘从约 `925 MB` 清理到约 `5.1 GB` 可用：删除了 7 月 24-25 日重复诊断，但保留 `agent108000_complete` 与历史同协议 `20260725_051221_directions`；历史 contact-capture run 只保留 `agent_60000.pt`、`agent_70000.pt` 和 `best_agent.pt`，固定 `34k` 源、当前 run、近期诊断和所有 payload 未动。后续严格按约 10 分钟窗口联合看 strict 向零趋势、stand/core/height/tilt、四方向、步态/髋/足端、上下楼、fall 和 DR；没有日志联合改善前不做低信息物理诊断。

## 2026-07-31 18:52：理想平地作用域与驱动/惩罚可达性修复

- 全局原则进一步明确：成熟 checkpoint 只证明已有能力，不保证能适应任意新目标；主奖励驱动不能弱于辅助惩罚，不同场景的配置不能对同一状态提出不可达要求。理想平地必须与 DR 平地、崎岖/楼梯严格分层，不能仅靠不同权重近似区分。
- nominal 理想平地的硬作用域现为 `flat terrain AND dr_env_level == 0`。所有场景的精确零命令仍要求真实机身、关节和足端静止以及安全支撑；只有 DR0 理想平地额外要求标准姿态、水平机身、精确站立高度和 `flat_move_height` 窄带。DR 平地允许为负载、参数误差和扰动作必要姿态补偿；崎岖/楼梯允许适应地形。平地跟踪、楼梯驱动、DR 分布、AMP、PD 和 PPO 均未改。
- 前一 run 的实测零命令样本中，按 stand gate 折算的 `stand+stand_contact` 正信用约 `+1.75`，无界 Huber 严格成本约 `-5.64`；这证明旧 `w=2.0` 路径由负惩罚主导。单纯降权也不能从数学上解决，因为有界正奖励与无界成本总会在大残差处反转排序。
- 严格成本改为每项 `x/(1+x)`：自由区仍严格为 `0`，任意非零真实运动都有成本，任意有限误差处都有单调梯度，但单项成本小于 `1`。`w_stand_strict_penalty=0.4`；DR/崎岖的聚合严格成本最多 `0.4`，DR0 加标准姿态成本后最多 `0.8`，低于 stand 正信用上限 `3.5`。验证还固定了成熟残差样本中正信用大于严格成本，避免再次形成负回报悬崖。
- 当前 payload：`/root/gpufree-data/training_payloads/history7_flat_direct_quality_balanced_20260731`；本地归档 SHA256 为 `153e235d7de6cb7bfbd731068972555503bc524fc9b33cdb8cbc3d8526b0abd8`。核心 strict、DR0/DR/地形作用域、平地质量驱动、AMP、gait clock、下楼奖励和 `compileall` 已串行通过。
- 当前正式 run：`/root/gpufree-data/taili_runs/taili_train_20260731_history7_flat_direct_quality_balanced_resume2k`。它从 scope-fix run 的对齐 `agent_2000.pt` 恢复，后者继承历史 `7+` actor 并已训练 2000 步 nominal 平地直接质量成本；启动精确恢复 phase3、DR3、楼梯课程状态。当前已保存 `agent_4000.pt`，训练继续运行。
- 约 10 分钟最近/前一半窗口：F/B/L/Y 均保持或上升，最近约 `0.814/0.823/0.772/0.846`；楼梯下/上约 `7.50/7.62`，replay `6.48` 不变，frontier 约 `8.52`；fall 约 `0.0004`。stand 正信用上升，strict 约 `-0.06` 保持辅助量级；髋、冲击、支撑和倾角略改善，滑移、足端轨迹、duty 与 nominal 综合质量成本约 `-0.028` 尚未形成明确净改善。
- 当前结论是继续长训、不干预、不并行做低信息诊断。后续按约 10 分钟联合看四方向、stand 的真实物理残差、nominal 平地质量分解、楼梯 replay/frontier/fall 和 DR。不能因 strict 数值量纲变小宣称静止已解决；日志形成可信趋势后，再用同协议 DR0 全方向诊断验证髋、足端轨迹、滑移、冲击、核心、高度和动转静过程。

## 2026-07-31 20:05：保留 22k，恢复平地/楼梯/DR 全局推进

- 保留检查点为 `/root/gpufree-data/taili_runs/taili_train_20260731_history7_flat_direct_quality_balanced_resume2k/checkpoints/agent_22000.pt`，SHA256 为 `41fb53f5da42ef2b076e442ca5e5d42f2a7904511aaf5379f2f24896ddbf9bc5`。对应同协议 DR0 平地诊断为 `/root/gpufree-data/diag_runs/locomotion_console_20260731_114111_directions`，本地副本为 `output/diagnostics/20260731_114111_history7_balanced_22k`。后续清理不得删除该检查点和诊断。
- 该检查点的定位不是全局最优，而是“移动能力与历史 7+ 课程状态保持、动转静明显缩短、尾段接近安静，但标准站姿仍有来向依赖”的参考点。排除初始化后，前进/横移稳态跟踪约 `0.500/0.301 m/s`；后退约 `-0.369 m/s`。动转静进入总平面速度 `<0.05 m/s` 约需前进 `0.28 s`、横移 `0.78 s`、后退 `0.50 s`，反向过冲较小。
- 尚未通过的平地项包括：停止后高度约 `0.522-0.530 m`，低于 `0.547 m` 标准且随停止前方向变化；尾段平移仍约 `0.011-0.017 m/s`；初始稳态髋姿态不对称；横移后腿足端仍有明显前后斜向分量；后退欠速、横向漂移和支撑滑移较突出。前进/横移后的制动滑移与高度瞬态也不够干净。本诊断未执行 yaw，因此不能用它验收 yaw。
- 安静站立优先级现已下调：它仍属于平地全局质量，已有改善必须保留，但不再作为单项门控或单独主导重启。下一阶段按平地完整质量、前进上下楼梯能力与安全性、DR 鲁棒性联合净收益推进；不得用静止继续改善交换四方向、自然步态、核心/高度、髋/足端、低滑移轻触地、楼梯和 DR。
- 当前 run 继续运行，约 `33.8k` 的 2k 分窗表明四方向从开局约 `0.801/0.788/0.753/0.843` 缓慢升至约 `0.829/0.829/0.791/0.867`；下/上楼长期约 `7.48/7.63`，replay `6.48`，frontier 约 `8.47`，DR3，均以平台为主。gait/髋/落脚/滑移/核心没有形成一致净改善。下一次干预前应以最新检查点交叉验证完整 yaw、代表性上下楼梯和代表性 DR，而不是继续围绕 stand 单项加机制或加权。

## 2026-07-31 20:30：优先完成理想平地，再逐级全面推进 DR

- 安静站立单项的优先级下降不等于放宽平地。当前第一主线是把 `DR0` 理想平地完整打磨到目标：四方向准确跟踪、自然对称步态和足端轨迹、髋关节无持续内外翻、核心与高度稳定、低滑移和轻触地、稳定支撑、yaw 无平移泄漏，以及已有动转静效果不回退。前向楼梯已经在 `27 cm` 诊断中证明存在通过能力边界，因此暂时停止继续抬高楼梯难度；训练中只保留能力，不能用楼梯或静止交换平地质量。
- 正确的最新全方向交叉诊断已通过系统启动：job `20260731_122612`，checkpoint 为当前 run 的 `agent_40000.pt`。协议显式覆盖前进、后退、左右横移、左右 yaw 及每段后的零命令，修复 `22k` 诊断中 backward/yaw 被错误配置为 stand 的协议缺口。诊断完成前不根据旧缺口机械改权重。
- 平地验收后，第二主线是逐渐但全面地推进域随机化。等级约定为：`DR0=0 kg` 标称；`DR1<=5 kg` 真实 sim2real 差异；`DR2<=10 kg` 单项异常；`DR3<=10 kg` 且加入复合异常；新增 `DR4<=50 kg` 极限专项。两个 `10 kg` 等级以扰动组合复杂度区分，不是重复训练同一分布。
- 负重必须实现为机身 base link 的正附加载荷 `0..上限`，并配合安装位置/质心偏移；不得通过缩放整机或腿部质量伪造负重。当前实现确实只给 base link 加质量增量，但旧范围含负值且只有 `DR0..DR3`、最高 `+20 kg`，后续需要在平地通过后修正为上述五级语义。
- DR 覆盖不能退化为只随机负重。顺序为：真实摩擦、质量/质心、执行器增益和阻尼、传感偏差与现实延迟；再加入较强单项异常；随后训练接触、执行、感知、延迟和推扰的复合异常；最后进入 `50 kg` 极限专项。每一级都保留 nominal 与较低等级样本，避免灾难性遗忘。DR1 尽量保持理想平地质量，DR2/DR3 要保持可用跟踪和安全稳定，DR4 首先验收方向正确、可控、稳定、安全和不倒，再优化动作质量。
- `DR4` 的其他随机化强度必须与 `50 kg` 目标匹配，不能只提高负重。执行器衰减/逐关节不对称、摩擦与接触、质心偏移、动作和观测延迟/抖动、传感偏差、推扰等都要有独立的极限端点；训练按“50 kg 负重渐增单项 -> 极限单项 -> 中高负重加复合异常 -> 50 kg 全量端点”逐步提高复合比例，而不是从第一批样本就把全部极值相乘。当前 push 语义是速度增量，同一 `delta-v` 在 50 kg 附加载荷下已经对应更大物理冲量，因此匹配强度必须按实际动力学衡量，不能把数值再按质量机械同比放大。

## 2026-07-31 20:55：切换为 DR0 理想平地主训

- `agent_40000.pt` 全方向诊断与原 run 的 `18k..48k` 遥测共同表明：四方向跟踪已经可用，但核心/高度、方向不对称、髋周期偏移、横移后腿轨迹、滑移和触地质量长期主要震荡。原 phase3 只有约 `17.6%` nominal DR，乘以 `70%` 平地和移动配额后，真正用于 DR0 理想平地移动的样本不足总量一成；这是当前首先应修正的训练分布问题，不是继续增加奖励机制的证据。
- 新 payload 为 `/root/gpufree-data/training_payloads/history7_ideal_flat_focus_20260731`，归档 SHA256 为 `76fe026d6d87c63296e12bb2917412c35454f592ef96025fa6391f62b2ff510b`。它只改 phase3 分布：`stand_prob=0.20`、`flat_single_axis_fraction=0.70`、平地/下楼/上楼比例 `0.90/0.05/0.05`，并把 `DR0..DR3` 已解锁状态下的 tier mixture 暂时全部锁为 nominal。奖励、AMP、PPO、PD、动作尺度、严格站立、楼梯驱动和 replay 均未改。
- 理论配额约为 `78.5%` DR0 平地移动、`11.5%` DR0 平地动静训练、`8.5%` 楼梯移动保持；上下楼各保留一列、各约 51 个环境。当前重点是完整理想平地，不是只优化静止；楼梯样本只防止已证明的前向能力遗忘，暂不继续抬高高度，DR 在平地验收后再逐级恢复。
- 正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260731_history7_ideal_flat_focus_resume48k_long`，从 `/root/gpufree-data/taili_runs/taili_train_20260731_history7_flat_direct_quality_balanced_resume2k/checkpoints/agent_48000.pt` 恢复，源 SHA256 为 `bfeb408a14e4742be67c9b53bd29e1215fa77f0f0f9afc7748786d33fc831473`；该源检查点在本轮结束前不得删除。运行时已确认 phase3、1024 环境、下/上楼各 51 个环境、DR population=`1.000/0/0/0`。step 210 时移动约 `81.3%`、纯单轴约 `67.8%`，四方向单轴桶约 `15.9%-17.8%`，fall=`0`。
- 第一次同名短启动因 SSH 生命周期结束，只运行到 step 1，目录为 `taili_train_20260731_history7_ideal_flat_focus_resume48k`，没有产生检查点且不是策略失败；后续只使用带 `_long` 的正式 run。下一判断先看连续窗口，再用 DR0 全方向诊断验收真实物理表现，不能把 nominal 样本增多造成的日志量纲变化直接当成能力改善。
- 数据盘清理后约剩 `2.9 GB`。旧 balanced run 只保留 `agent_22000.pt`、`agent_40000.pt`、`agent_48000.pt` 和 `best_agent.pt`；其余无独立诊断或恢复用途的 2k 间隔副本已删除。当前正式 run、两个诊断参考点、恢复源、payload 和诊断记录均未动。

## 2026-07-31 22:04：DR0 完整平地验收与高度判断纠偏

- 最新完整系统诊断为 `/root/gpufree-data/diag_runs/locomotion_console_20260731_135120_directions`，job `20260731_135120`，使用当前正式 run 的 `agent_20000.pt`。协议在单个 DR0 平地环境中实际覆盖 stand、前进、后退、左右横移、左右 yaw，并在各移动段之间记录零命令；共 `2100` 帧，诊断和报告均正常完成。
- 纠正此前机械高度锚点：`0.547 m` 只是若干移动方向的观测均值，不是静止与所有方向必须收敛到的统一目标。训练遥测前后各 50 个窗口的 batch `base_h` 均值约为 `0.53915 -> 0.53937 m`，不存在朝 `0.547 m` 单调收敛的趋势，也没有必要要求这种趋势。诊断中移动高度均值按方向约 `0.540-0.547 m`、标准差约 `4-8 mm`；各停步尾段约 `0.523-0.532 m`、标准差约 `1-3 mm`。应按姿态几何、波动和切换自然性判断，而不是把动静高度差直接认定为失败。
- 六个移动方向的姿态稳定占比均为 `1.0`，无 reset、done 或主要异常事件；跟踪误差前/后/左横/右横/左 yaw/右 yaw 分别约为 `0.036/0.036/0.052/0.042/0.050/0.044`。移动段 roll/pitch p95 大多在 `2 deg` 内，后退启动段 pitch p95 约 `3.26 deg`；前进/后退髋角绝对值 p95 约 `0.095/0.052 rad`，左右横移髋动作约 `0.21-0.26 rad`，符合横移需要。系统报告的 `max_roll=13.85 deg` 来自第 0 帧初始化姿态，不能用于评价成型策略。
- 动转静在较严格的持续低速阈值下约需 `0.8-1.6 s`，尾段已接近安静；仍可见少量单帧 high-slip bout 和方向相关制动尾差，但没有形成主要事件、失稳或失败。综合真实回放与分段数据，当前 `DR0` 名义平地基本盘评为约 `89-92/100`，用户给出的约 `90` 分判断合理；此前根据单一高度均值和少量 p95 事件作出的保守否定存在偏差。
- 该评分只适用于当前名义平地基本盘，不代表最终鲁棒策略已达 90 分。当前实时遥测约在 `26k`：F/B/L/Y 约 `0.87/0.84/0.84/0.87`，下/上楼课程约 `7.51/7.31`，fall 近零；虽然字段显示课程 `DR3`，但 `dr_tier_nominal_frac=1.0`、其余 tier 全为 `0`，实际没有启用 DR。下一主线应从保留 nominal 验收样本的小比例 DR1 开始，首批约 `80% nominal + 20% DR1`，验证真实 sim2real 差异下的净退化后再逐级扩大，不能继续等待脱离物理意义的“完美高度”。

## 2026-07-31 23:36：DR1 现实误差层与摩擦语义纠偏

- DR 主线从约 `90` 分的 DR0 平地、`7+` 楼梯成熟 actor 开始，不 fresh、不改奖励、AMP、PD、命令、楼梯或网络结构。第一轮 payload 为 `/root/gpufree-data/training_payloads/history7_dr1_real20_20260731`，训练人口约为 `80% nominal + 20% DR1`；DR1 同时覆盖 `0..5 kg` base 正附加载荷、轻量 CoM 偏移、执行器增益/阻尼与逐关节不对称、IMU 偏置、动作延迟和低强度推扰。
- 第一轮系统 DR0-vs-DR1 全方向诊断为 `/root/gpufree-data/diag_runs/locomotion_console_20260731_150402_dr`，checkpoint 为第一轮 `agent_4000.pt`，共 `1300` 帧。DR1 实例为 `+2.76 kg`、friction `1.115`、CoM 约 `(+5.2,-5.9) mm`、stiffness `0.980`、damping `0.955`；四方向无跌倒、跟踪 MAE 约 `0.014-0.034`。但资产名义摩擦实测为 `0.5`，旧 DR1 绝对摩擦区间 `[0.7,1.3]` 只会提高摩擦；其滑移改善是环境变容易，不能算鲁棒性证据。旧 DR1 延迟又固定为与 nominal 相同的 `1 step`，没有覆盖延迟误差。
- 修正版 payload 为 `/root/gpufree-data/training_payloads/history7_dr1_real20_centered_20260731`，归档 SHA256 为 `391d3af6dd5eb2e4b5c658cc20420cc5c78b85e5cb2ada50ccd9d6308f0ee172`。唯一语义修改是 friction `[0.7,1.3] -> [0.35,0.80]`、latency `[1,1] -> [0,2]`；本地和远端 `compileall`、YAML 到 env cfg 映射均通过。
- 当前正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260731_history7_dr1_real20_centered_resume10k_long`，从第一轮成熟 DR1 `/root/gpufree-data/taili_runs/taili_train_20260731_history7_dr1_real20_resume40k_long_v2/checkpoints/agent_10000.pt` 恢复，源 SHA256 为 `41ea43200b7eb5514afdb8d6403b5da806ec007bdb8161b12e7a3e60c0df7356`。运行时人口为 `77.34% nominal / 22.66% DR1 / 0% fault / 0% stress`；动作延迟均值约 `1.0` 只与 `0/1/2` 对称分布一致，不能单凭均值证明三个端点均被采样。
- 修正版 `0-2k` 与 `2-4k` 连续窗口的 F/B/L/Y 为约 `0.848/0.853/0.833/0.876 -> 0.858/0.851/0.839/0.883`，raw 四方向约 `0.93-0.95`；下/上楼约 `7.36/7.45 -> 7.42/7.48`，fall 约 `0.9e-4 -> 1.0e-4`。低摩擦加入后 slip 约 `0.043-0.044`，相对旧 DR1 后窗约 `0.040` 有符合物理预期的小幅增加，但未形成恶化趋势；tilt 约 `2.35 deg`，核心、高度、髋、足端和冲击未出现持续能力交换。当前证据支持继续，不支持因启动单点回退。
- 修正版系统 DR0-vs-DR1 全方向诊断已启动，job `20260731_153604`，checkpoint 为当前 `agent_4000.pt`。验收重点是各方向真实净退化、低摩擦滑移/支撑/触地、核心与高度、以及 record 中 friction 是否落入 `[0.35,0.80]`；延迟必须检查逐帧/逐 case 分布，不能只看均值。诊断通过前可以构造 DR2，但不能部署混入训练。
- 数据盘通过裁剪七个已结束旧 run 的中间重复 checkpoint 释放约 `3.94 GB`；每个旧 run 均保留 `best_agent.pt`、最后 `agent_*.pt`、日志与配置。当前 run、当前恢复源、历史 `7+`、DR0 约 `90` 分 checkpoint、关键诊断和全部 payload 均未动，当前可用空间约 `5.4 GB`。
- 后续等级仍采用四列实现：tier0=DR0，tier1=DR1，tier2=DR2 强单项异常，tier3 通过 single/compound/full profile 承载 DR3 复合异常和 DR4 极限专项。DR2 每个环境只启用 mechanics/contact/actuation/sensing/push 五类中的一个，类别均衡，base 正附加载荷上限 `10 kg`；不能沿用旧 tier2 的负质量或全因素同时随机。DR3 再逐步组合少量因素；`50 kg` 与匹配的执行、接触、感知、延迟和推扰端点只属于独立 full/extreme profile，不能与普通 compound 共用参数。

## 2026-08-01 00:35：DR1 验收、DR2 单项异常与精确人口修复

- 修正版 DR1 系统诊断为 `/root/gpufree-data/diag_runs/locomotion_console_20260731_153604_dr`，本地副本为 `output/diagnostics/20260731_153604_history7_dr1_centered_4k`。DR1 实例为 `+2.758 kg`、friction `0.6612`、CoM `(+5.2,-5.9) mm`、stiffness `0.9799`、damping `0.9547`、latency `2 steps`。总报告中的 touchdown `1.54/1.00 m/s` 峰值均来自 reset 后 `0.08 s` 内；排除每段前 `0.2 s` 后，DR1 移动段 touchdown p95 约 `0.08-0.13 m/s`，高于 DR0 的 `0.03-0.05 m/s`，但没有形成失稳或持续冲击。移动姿态没有整体退化，主要尾差是 yaw tilt 和延迟/载荷下较重触地。
- DR1 正式 run `/root/gpufree-data/taili_runs/taili_train_20260731_history7_dr1_real20_centered_resume10k_long` 的 `0-14k` 连续窗口中，F/B/L/Y 始终约 `0.83-0.88`，上下楼约 `7.4-7.5`，fall 约 `1e-4`；slip、核心、高频动作和冲击均无持续恶化。因此 DR1 已建立，不再用更多同分布训练替代下一等级覆盖。
- 第一版 DR2 payload 为 `/root/gpufree-data/training_payloads/history7_dr2_single10_20260731`。语义为约 `70% DR0 + 20% DR1 + 10% DR2`；DR2 每个环境只启用 mechanics/contact/actuation/sensing/push 中的一类，并在 flat/down/up 内均衡。范围为 base 正附加载荷 `0..10 kg`、friction `[0.25,1.00]`、stiffness `[0.75,1.25]`、damping `[0.65,1.35]`、latency `[0,3]`、joint gain spread `0.08`、每 `12 s` 一次 `0.5 m/s` push。
- 运行时发现原层级人口使用一次独立随机抽样，目标 `70/20/10` 实际固定成 `65.625/21.191/13.184`，使 DR2 压力比设计高约三成。该问题不是 actor 退化，而是训练分布实现误差。现已改为在每类地形内使用最大余数法精确配额，再随机打散环境；纯 helper、层级分配、五类单项、全部既有契约、`compileall` 均在本地和远端通过。
- 精确版 payload 为 `/root/gpufree-data/training_payloads/history7_dr2_single10_exactmix_20260731`，归档 `history7_dr2_single10_exactmix_20260731.tar.gz` 的 SHA256 为 `da49c5322e8cef44021f38ba90b93cce6beb63f5bf5c0a97cdb4ef253dcf8c63`。正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260801_history7_dr2_single10_exactmix_resume4k_long`，从第一版 DR2 的 `agent_4000.pt` 继续；没有 fresh，也没有修改奖励、AMP、PD、命令、楼梯或网络结构。
- 精确版实测人口为 `70.0195/20.0195/9.9609/0%`，五类 DR2 因素各约 `19.6-20.6%`，latency min/max 为 `0/3`。`0-14k` 窗口中 F/B/L/Y 大致保持 `0.86-0.88/0.83-0.85/0.83-0.84/0.88-0.89`，上下楼约 `7.3-7.5`，fall 约 `1e-4`；slip 约 `0.045-0.049`、tilt 约 `2.34-2.48 deg`、动作高频约 `0.065`，没有能力坍缩或以质量交换进度。后退和横移有窄幅震荡，但没有持续下降。
- 当前采用 10 分钟间隔监控。下一步先用 DR2 各单项的真实物理诊断定位最弱类别；只有连续训练与诊断都确认 DR2 净退化可控，才从当前成熟 checkpoint 引入 tier3 compound。普通 DR3 仍限制在 `<=10 kg` 并组合少量因素；`50 kg` 极端负载及与之匹配的执行、接触、感知、延迟和推扰必须使用独立 full/extreme 参数，不能直接沿用当前 tier3 的旧 `-5..20 kg` 范围。

## 2026-08-01 05:00：DR2 单项暴露提高到 20% 及 30k 物理验收

- `10% DR2` 精确人口 run 持续到 `agent_70000.pt` 后，连续窗口仍保持 F/B/L/Y 约 `0.84-0.88`、下/上楼约 `7.3/7.5`、近零 fall、slip 约 `0.038-0.044`，没有平地、楼梯或核心的持续能力交换。因此按既定顺序只提高单项异常暴露，不改奖励、AMP、PD、命令、楼梯、网络或五类 DR2 范围。
- 新 payload 为 `/root/gpufree-data/training_payloads/history7_dr2_single20_20260801`，归档 `history7_dr2_single20_20260801.tar.gz` 的 SHA256 为 `5414605e0785c40d378188f2e591c10c865a1f56600af2fcd33dee7a9acfc9ae`。唯一语义变化是 level 2/3 人口从 `70/20/10/0` 改为 `60/20/20/0`；本地与远端 `compileall`、DR2 精确分配和全部 payload 契约均通过。
- 正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260801_history7_dr2_single20_resume70k_long`，从上一阶段 `agent_70000.pt` 恢复，课程状态也精确恢复到 step `70000`。运行时人口为 `60.0586% DR0 + 20.0195% DR1 + 19.9219% DR2`，五类 DR2 单项各约 `20%`。到本 run `30k`，完整窗口大致保持 F/B/L/Y `0.84-0.89`、下/上楼 `7.25-7.56`、fall `0.0001-0.0003`、tilt `2.4-2.8 deg`、slip `0.035-0.040`，启动适应期的楼梯/tilt 波动均能恢复。
- `agent_30000.pt` 的系统单项诊断 job 为 `20260731_203757`，远端 `/root/gpufree-data/diag_runs/locomotion_console_20260731_203757_dr`，本地副本 `output/diagnostics/20260731_203757_history7_dr2_single20_factors_30k`。五类共 25 个命令段均无 done/fall，整体 F/B/L/Y 跟踪误差约 `0.033/0.048/0.040/0.023`，稳定占比 `1.0`。
- 新旧诊断使用了相同的随机参数实例，可以直接差分。排除每段前 `0.3 s` 后，五类 tilt p95 均改善约 `0.21-0.36 deg`，stance-slip p95 均改善；actuation slip 从约 `0.163` 降到 `0.057 m/s`。方向跟踪大体持平或改善，但 mechanics/sensing/push 的 touchdown 尾部变重，sensing 横移存在少量高冲击离群点；push case 仍未在 record 中记录内部 push，不能据此宣称推扰通过。
- 当前不进入 compound。保留 `20% DR2` 专项继续适应，在后续成熟 checkpoint 复测 touchdown、sensing 横移以及训练端真实 push；只有单项异常在真实动作、轻触地和稳定性上同时通过，才启用普通 `<=10 kg` 的少因素复合异常。数据盘已再次裁剪两个当前链条的冗余 `2k` 检查点，保留每 `10k`、关键诊断点、最新点、恢复源和 best，可用空间约 `3.2 GB`。

## 2026-08-01 07:00：DR2 sensing 专项与首轮连续窗口

- `agent_60000.pt` 同协议复测 job 为 `20260731_215117`，本地副本为 `output/diagnostics/20260731_215117_history7_dr2_single20_factors_60k`。排除各段前 `0.3 s` 后，mechanics/contact/actuation/sensing/push 的 tilt p95 约为 `1.79/1.82/2.58/2.25/1.82 deg`，slip p95 约为 `0.050/0.042/0.065/0.099/0.079 m/s`，touchdown p95 约为 `0.070/0.075/0.120/0.198/0.130 m/s`。mechanics/contact/actuation 明显改善；sensing 的 slip 从 30k 约 `0.048` 升至 `0.099`，touchdown 从约 `0.186` 升至 `0.198`，是明确未闭合项。
- 显式横向 `0.5 m/s delta-v` push 诊断 job 为 `20260731_221114`，本地副本 `output/diagnostics/20260731_221114_history7_dr2_single20_push_60k`。DR0 平地静止/前进均无 done/reset，tilt 峰值 `<3 deg`、最低高度 `>0.543 m`；完整跟踪约 `0.66/0.58 s` 恢复，但高滑移持续约 `0.30/0.14 s`。因此推扰生存与恢复已建立，接触质量仍需后续优化。
- 均衡 DR2 run 的最后两个 10 分钟窗口显示四方向和 `7+` 楼梯未坍缩，但 touchdown 均值升至约 `0.351 m/s`，末段约 `0.359`；support instability 末段约 `0.0328`，core 略降，上楼均值出现回落。这与单项诊断中的 sensing 短板同向，满足只调整 DR2 内部配额的干预条件。
- 新 payload 为 `/root/gpufree-data/training_payloads/history7_dr2_sensing35_20260801`，归档 SHA256 为 `c82843921de4e40de32390593560f1cca0b56ebbf58e7286be1f5af4513d3a11`。唯一语义变化是 DR2 内 mechanics/contact/actuation/sensing/push 配额从均衡改为 `0.15/0.15/0.20/0.35/0.15`；总人口仍为 `60/20/20/0`，奖励、AMP、PD、命令、楼梯、网络和随机化范围均未改。本地及远端全部 verifier、`compileall` 与真实 runtime 人口检查通过。
- 正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260801_history7_dr2_sensing35_resume80k_long`，从均衡 run 的完整 `agent_80000.pt` 恢复。实测四层人口 `60.0586/20.0195/19.9219/0%`，DR2 五因素 `15.196/14.216/20.098/34.804/15.686%`，课程恢复到下/上楼约 `7.31/7.53`。
- 新 run `0.5k-5.0k` 窗口中 touchdown 从早段约 `0.342` 降至末段 `0.330 m/s`；`5.2k-9.6k` 进一步降至均值 `0.331`、末段 `0.319`，support 末段降至约 `0.0298`，tilt、core、高频动作和高度风险恢复，说明 sensing 专项方向有效。尚未闭合的交换是 slip 均值约 `0.0399` 且末段约 `0.0409`、duty 降至约 `0.614`，B/L 均值偏弱且 fall 尾部略升。当前继续 10 分钟窗口监控，不回退、不进入 compound；先确认上述交换是否自行恢复，再做同协议 sensing 物理复测。

## 2026-08-01 09:40：sensing 专项闭环、DR2 五因素均衡巩固

- sensing 专项 `agent_48000.pt` 的同协议五因素诊断 job 为 `20260801_003728`，远端目录 `/root/gpufree-data/diag_runs/locomotion_console_20260801_003728_dr`。排除命令切换后前 `0.3 s`，mechanics/contact/actuation/sensing/push 的 tilt p95 为 `2.280/2.205/3.038/2.442/2.281 deg`，slip p95 为 `0.0710/0.0526/0.0808/0.0657/0.0751 m/s`，touchdown p95 为 `0.0561/0.0960/0.0880/0.0827/0.0676 m/s`。相比专项前 sensing 的 `2.564 deg / 0.1003 / 0.1802`，sensing 滑移和触地尾部已显著闭环；继续给 sensing `35%` 的边际收益低，且 mechanics/contact/push tilt 有小幅回退、actuation tilt 仍偏高。
- 新 payload 为 `/root/gpufree-data/training_payloads/history7_dr2_rebalanced20_20260801`，归档 SHA256 为 `46126ce1b128f9b51a48cd1dcb94cefb3c2420bf1ee64516df7fc0cd5abc8f45`。唯一语义变化是 DR2 内 mechanics/contact/actuation/sensing/push 从 `0.15/0.15/0.20/0.35/0.15` 恢复为 `0.20/0.20/0.20/0.20/0.20`；总人口、随机化强度、奖励、AMP、PD、命令、楼梯、网络、actor 和 optimizer 均未改。本地与远端全部 verifier、`compileall` 通过，且源码差分仅有 env cfg、YAML 和对应 verifier 三处配额契约。
- 正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260801_history7_dr2_rebalanced20_resume58k_long`，tmux 为 `rl_train_dr2_rebalanced20`。它从 sensing run 的完整 `agent_58000.pt` 恢复，源 SHA256 为 `97e25ede48b75c23ac5b4fb60d4640fb99b1fbd84393562f7bcaea5e3a5f3823`；课程精确恢复到 step `58000`、楼梯均值约 `7.44`，不是 fresh。运行时总人口为 `60.0586/20.0195/19.9219/0%`，五因素为 `20.098/20.098/19.608/20.098/20.098%`，latency 覆盖 `0..3 step`。
- 新 run 的三个连续 10 分钟窗分别覆盖 `1.21k-5.69k`、`6.01k-10.49k`、`10.75k-15.32k`。F/B/L/Y 均值依次为 `0.878/0.865/0.848/0.887`、`0.888/0.853/0.852/0.886`、`0.873/0.873/0.847/0.887`；下/上楼为 `7.307/7.546`、`7.325/7.476`、`7.300/7.509`，fall 为 `0.00032/0.00030/0.00029`。第二窗上楼低位在第三窗后半段回升到约 `7.53`，属于周期震荡，不是持续坍缩。
- 三窗的 slip 均值为 `0.03981/0.03876/0.03882`，高滑移为 `0.04490/0.04224/0.04197`，duty 为 `0.61264/0.62481/0.62667`，touchdown 为 `0.33154/0.34031/0.33264`，core 为 `0.27316/0.27188/0.27374`。第三窗后半段 tilt 和 touchdown 下降、上楼和 duty 回升；当前没有平地、楼梯或接触质量的持续能力交换，不干预、不回到 sensing `35%`，也不提前进入 compound。
- 后续继续使用完整 10 分钟窗监控四方向、上下楼、fall、tilt、slip/high-slip、分方向 duty、touchdown、core 和 support。先让五类单因素均衡巩固；成熟后再做同协议五因素物理验收，只有各单因素真实动作和接触质量共同通过才进入普通少因素 compound。数据盘只裁剪了无独立诊断用途的 `2k` 重复 checkpoint，保留 sensing `20k/48k/58k/best/每10k`、当前 `10k/best`、恢复源、payload、日志和诊断，训练未受影响。

## 2026-08-01 10:45：均衡 DR2 的 20k-35k 连续窗口

- 继续监控的三个完整 10 分钟窗覆盖 `20.54k-25.18k`、`25.50k-30.04k`、`30.45k-34.97k`。F/B/L/Y 均值依次为 `0.870/0.860/0.843/0.884`、`0.867/0.864/0.847/0.892`、`0.870/0.861/0.847/0.888`，没有方向能力持续坍缩。下/上楼依次为 `7.300/7.423`、`7.299/7.438`、`7.246/7.422`；相对本 run 前三个窗口约 `7.30/7.51`，上楼形成约 `0.08-0.09` 的小幅跨窗下移，第三窗下楼也处于低位，但仍稳定在 `7+`，尚不足以修改课程或楼梯驱动。
- duty 三窗为 `0.619/0.608/0.612`，duty spread 为 `0.102/0.106/0.107`。第一窗后半段 duty 由 `0.629` 降至 `0.609`，但第二窗后半段由 `0.604` 回升至 `0.613`；第三窗横移 duty 后半段回升约 `0.018`。因此 duty 存在明显周期震荡和较低平台，但当前证据不支持持续坍缩，也不支持直接增加惩罚。若后续需要调查，应先按方向和样本组成核对实际作用域与奖励梯度。
- slip 三窗约 `0.0408/0.0407/0.0398`，high-slip 约 `0.0464/0.0458/0.0436`，touchdown 约 `0.313/0.302/0.307`，core 约 `0.274/0.273/0.274`，support 约 `0.0350/0.0335/0.0347`。轻触地相对前期约 `0.33-0.34` 明显改善，核心稳定；没有证据表明策略通过全面保守化交换接触质量，但楼梯小幅下移必须在下一次同协议五因素物理诊断中重点核对真实通过和接触安全。
- 当前不干预、不回退 sensing 专项、不进入 compound。继续均衡 DR2 长训和 10 分钟窗口；成熟 checkpoint 的下一交叉验证应复用既有五因素协议，重点比较 mechanics/contact/actuation/sensing/push 的 tilt、slip、touchdown、方向跟踪和真实楼梯表现。只有日志趋势与物理诊断共同通过，才推进少因素 compound。

## 2026-08-01 11:10：DR2 44k 验收进行中

- 验收对象固定为均衡 run 的 `agent_44000.pt`；训练继续运行。平地五类隔离诊断 job `20260801_025131` 已完成，共 `3250` 帧且无 done/reset。排除命令切换后 `0.3 s`，mechanics/contact/actuation/sensing/push 的跟踪误差均值为 `0.026/0.028/0.037/0.034/0.028`，tilt p95 为 `2.19/2.36/2.72/2.17/2.31 deg`，slip p95 为 `0.057/0.051/0.064/0.068/0.070 m/s`。
- 平地尚不能完整验收：五类 touchdown p95 为 `0.084/0.125/0.104/0.207/0.061 m/s`。sensing 后退出现一次 `0.533 m/s` 落脚，超过诊断 hard-impact 阈值 `0.40 m/s`；这是切换后约 `2.74 s` 的真实事件，不是 reset 瞬态。均衡训练改善了 actuation tilt 和多数跟踪/滑移，但 sensing 触地尾部相对 sensing35 `48k` 的 `0.085 m/s` 明显回退。
- 显式横向 `0.5 m/s delta-v` 诊断 job `20260801_030711` 已完成。静止与 `0.5 m/s` 前进受击均无跌倒，最低高度分别约 `0.557/0.547 m`，最大 tilt 约 `3.83/3.20 deg`；按误差连续 `0.2 s <=0.10 m/s` 的一致算法，恢复约为 `0.60/0.92 s`。生存和姿态恢复成立，但前进恢复比上一 actor 的同算法约 `0.58 s` 慢，且滑移尾部仍存在，因此 push 只能判为“能力成立、质量未完全闭环”。
- 五因素协议每类只抽取一个实例，不能代表完整 DR2 包络：本轮 mechanics 仅 `+3.53 kg`、contact 摩擦仅 `0.769`，且普通 push factor 没有实际 push 帧。它适合做同实例差分，不能单独作为最终包络验收。
- 原计划的 `27 cm` DR2 楼梯诊断 job `20260801_031113` 已在写入数据前取消。用户明确指出当前 nominal 的 `27 cm` 尚非轻松从容通过，因此 DR2 基础验收改为显式 `20 cm` 上/下楼、五类单因素；`27 cm` 保留为后续上限测试。当前结论是 DR2 **尚未验收**，必须完成 `20 cm` 楼梯测试并处理/复核 sensing 落脚与前进受击恢复尾部后再决定是否进入 `20 kg` compound。

## 2026-08-01 12:00：DR2 20cm 楼梯完成、阶段结论与楼梯优化边界

- 完整系统诊断 job 为 `20260801_033326`，远端目录 `/root/gpufree-data/diag_runs/locomotion_console_20260801_033326_terrain`；仍使用均衡 run 的 `agent_44000.pt`。协议显式固定上下楼 `20 cm`、mechanics/contact/actuation/sensing/push 五类 DR2 单因素，每类一个环境，`0.8 s` 站立后以 `0.5 m/s` 前进 `6 s`。共完成 `10/10` case、`3400` 帧，全部无 done/reset。
- 五个上楼 case 前进约 `2.48-2.55 m`，支撑地形高度上升约 `0.78-0.84 m`；五个下楼 case 前进约 `2.24-2.53 m`，支撑地形高度下降约 `0.69-0.91 m`。因此当前 DR2 单因素实例下的 `20 cm` 上下楼能力保持成立。结合平地五因素跟踪、显式推扰生存与恢复，DR2 可判为“阶段能力基本通过”，不再因楼梯共同质量问题持续困在 DR2；但这不是完整随机包络的最终证明，sensing 平地单次 `0.533 m/s` 落脚、前进受击恢复尾部及端点覆盖仍作为保留项进入后续监控。
- 楼梯问题不能归因于某一个 DR 因素。五类实例表现出同向的高倾角、滑移和冲击；尤其 push 类在本次 `6.8 s` case 内没有真实 push 帧、其他物理量为 nominal，却仍出现相同质量问题。实际抽样也只到 mechanics `+3.53 kg`、contact 摩擦 `0.769`、actuation stiffness/damping `1.151/0.979`、sensing latency `0.04 s`，并没有足以单独解释共同现象的极端扰动。
- 楼梯能力成立但质量尚未达到理想状态。上楼的共同问题是落脚冲击、支撑滑移、机身角度波动和部分扭矩饱和；下楼的共同问题是滑移、短时高度下探以及方向保持，mechanics/actuation/sensing 三个下楼实例在 `6 s` 内出现约 `0.45-0.60 m` 横向偏移和 `21-26 deg` 航向漂移。系统报告合计记录 `178` 段高滑移、`62` 段硬冲击，最高倾角约 `19.8 deg`；所以“能通过 20cm”不能替代“安全、稳定、从容”。
- 当前配置并非缺少楼梯驱动或安全项：`w_terrain_direction=4.0`、`w_terrain_progress=1.5`，同时已有 heading window、off-axis、stance slip、landing impact、touchdown slip、核心与高度语义。更可能的结构问题是主方向信用与质量的耦合仍偏宽：`terrain_direction_quality_floor=0.65`、`terrain_stability_floor=0.75` 允许明显不稳定或重落脚时仍保留大部分主驱动，直接安全代价不足以消除该局部最优。后续楼梯优化应保留已经验证的直接方向/进展驱动，只调整连续质量耦合与相对信用，重点收紧上楼冲击/滑移/扭矩尾部和下楼短窗航向/横向漂移/支撑；允许楼梯速度幅值适当降低，但方向必须正确。不增加状态机、固定落脚点、逐腿序列或层级势能机制。
- DR 后续主验收仍以平地全面质量和异常恢复为主；地形侧要求是在 DR 增强时维持既有 `7+` 课程水平与 `20 cm` 可靠通过，不把 nominal 楼梯质量债务误算成 DR 失败。`27 cm` 继续只作为能力上限观察，不作为当前 DR 基础门槛。
- 数据盘清理未影响训练或诊断。清理 manifest 为 `/root/gpufree-data/cleanup_manifests/phase0_dense_checkpoint_prune_20260801_115323.json`：仅在 2026-07-28 至 07-30 已被后续路线取代的 `phase0` run 中删除 `89` 个密集中间 checkpoint，保留每个 run 的最高迭代与 `best`，并保留全部配置、日志、遥测、payload 和诊断；共释放约 `3.67 GiB`，数据盘可用空间恢复到约 `3.9 GB`。

## 2026-08-01 12:35：按地形解耦楼梯质量与下一阶段 DR

- 当前主线已经确定：同一个共享 actor 上，楼梯的 DR 推进慢于平地。楼梯维持已经阶段通过的简单 `DR0/DR1/DR2` 单因素分布，先优化安全、稳定、从容的通过质量；平地进入重新定义的 `20 kg` 下一阶段 DR。不是拆成两个策略，也不是重建楼梯机制。
- 执行源固定为正在运行的 `/root/gpufree-data/taili_runs/taili_train_20260801_history7_dr2_rebalanced20_resume58k_long`，tmux `rl_train_dr2_rebalanced20`；在新 payload 完成源码/运行时校验且新 run 成功接管前，不停止旧训练。当前约 `72.4k`，下/上楼约 `7.35/7.51`，作为本轮能力锚点。
- 目标条件分布：楼梯暂定 `60% DR0 / 20% DR1 / 20% DR2 / 0% 新DR3`；平地暂定 `40% DR0 / 20% DR1 / 20% DR2 / 20% 新DR3`。新 DR3 必须按既定 `10 -> 20 -> 35 -> 50 kg` 路线定义为新的 `20 kg` 单因素专项，不得直接启用旧的 `-5..+20 kg`、compound/full 混合语义。稳定后才逐步加入少因素 compound。
- 楼梯保留现有直接方向和进展主驱动，不加状态机、固定落脚点、逐腿序列或势能机制。只在真实楼梯移动作用域内收紧可归因的质量尾部：持续高滑移、最差腿重落脚、持续扭矩饱和、roll/wxy/vz/高度下探以及相对命令积分的短窗航向和横向漂移；允许合理 pitch 和一定降速。质量耦合必须仍给探索留下连续梯度，不能把已经成立的通过驱动切断。
- 因 actor 共享，两类训练压力仍会互相影响。验收必须联合进行：平地检查 `20 kg` 下跟踪、核心/高度、步态/髋/足端、滑移/触地、动静切换和异常恢复；楼梯检查课程保持 `7+`、`20 cm` 可靠通过且滑移/冲击/漂移下降。任何一侧以另一侧持续退化换来的单项改善都不算优化。
- 候选 payload 已完成并部署为 `/root/gpufree-data/training_payloads/history7_flat20_stair_quality_20260801`，本地归档 SHA256 为 `054d46ab0c53ebe9896786f6f74886aad5815d7c6232f1ae160450a5e6fa836c`。源码差分只涉及 DR 配置/条件采样、楼梯质量奖励和两个 verifier；actor、optimizer、AMP、PD、动作尺度、命令、课程与平地奖励未改。远端 `compileall`、地形分层 DR、核心/站立、AMP、gait clock 和下楼奖励 verifier 全部串行通过。
- 楼梯质量实现没有降低 `terrain_stability_floor`；该字段实际用于保留地形上的稳定成本比例，不能误解成主信用 floor。真正收紧的是：`terrain_direction_quality_floor 0.65 -> 0.50`，并将支撑、滑移、跟踪和轻触地共同纳入剩余方向信用；楼梯额外增加相对命令积分的短窗 heading、roll（不惩罚 pitch）、高滑移持续比例、最差落脚冲击、低角速度核心正信用和扭矩饱和成本。所有增量在平地 scope 严格为零。
- 新正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260801_flat20_stair_quality_resume78k_long`，tmux `rl_train_flat20_stairq`。固定恢复源为旧 run 的 `agent_78000.pt`，SHA256 `709ca41fe99426ccc0e8eca1ad7dfcdb7de53e9ba32b3504332af0c4e02bf307`；checkpoint 与 optimizer 已确认载入，课程精确恢复 step `78000`、`real_mean=7.441`，旧进程已退出。
- 运行时人口验证：全局约 `42.1/19.9/19.9/18.1%`；平地 `40.0/20.0/20.0/20.1%`；下楼和上楼均约 `60.8/19.6/19.6/0%`。DR3 profile 实测 `single/compound/full=1.0/0/0`。因此“楼梯慢走 DR、平地进入 20 kg”不是只写在 YAML，而是已经由实际环境分配确认生效。
- 首个严格 10 分钟窗口覆盖新 run `6.79k-11.29k`。四方向均值约 `0.863/0.859/0.840/0.873`，下/上楼约 `7.340/7.494`，fall `0.000206`；相对旧 run 最后窗口 `0.869/0.863/0.843/0.879` 与 `7.360/7.561` 只有小幅下移，没有能力坍缩。接触质量尚未形成净改善：duty `0.620 -> 0.606`、slip `0.03810 -> 0.03866`、high-slip `0.04004 -> 0.04121`、touchdown `0.3151 -> 0.3255`；support instability `0.0324 -> 0.0296` 略好，tilt `2.326 -> 2.406 deg` 略差。
- 新窗口前后半段中，F/B/L 有所恢复，但下楼均值 `7.362 -> 7.318`、touchdown `0.321 -> 0.330`、duty `0.610 -> 0.603`，不能宣布质量优化有效。由于 telemetry 是平地与楼梯聚合值，且新增平地 DR3 占总人口约 `18%`，当前也不能把接触变化直接归因于楼梯奖励。暂不干预、不诊断；继续一个严格 10 分钟窗口，只有课程/四方向持续下移或接触交换延续才调整。
- 第二个严格窗口覆盖 `11.89k-16.38k`。四方向约 `0.875/0.856/0.843/0.878`，下/上楼约 `7.365/7.476`；下楼已回到旧锚点，上楼全窗仍低约 `0.085`，但前后半窗从 `7.426` 回升到 `7.527`，不是持续下移。touchdown 全窗 `0.332` 仍高于旧锚点，但后半窗从 `0.341` 降至 `0.323`；slip/high-slip 约 `0.03864/0.04098`，基本回到旧水平。tilt `2.314 deg`、support instability `0.0291` 比旧锚点略好，duty `0.607` 仍偏低，fall `0.000273` 略高。
- 该窗口同时存在恢复与保留项，不能判为净改善，也不满足干预条件。继续第三个严格 10 分钟窗口，重点确认上楼后半窗恢复、touchdown 回落能否持续，以及 B/duty/fall 是否形成代价。
## 2026-08-01 16:05：平地 35 kg 边界搜索与楼梯质量第二轮

- 用户明确改变 DR 推进规则：平地不再等待逐级完整包络验收，而是按 `20 -> 35 -> 50 kg` 加速寻找联合崩溃边界；若四方向、核心/高度、接触质量和 fall 连续共同恶化，则回到最后稳定强度。后续系统诊断必须使用训练中对应地形的最高等级：平地当前强制 DR3，楼梯当前强制 DR2。诊断串行通过系统启动，不在本地并行运行多个 PyTorch。
- 新 payload 为 `/root/gpufree-data/training_payloads/history7_flat35_stair_quality_20260801`，本地归档 SHA256 为 `565ba61c9db50d7ae0f13c40df166ed51e19b6a3a92522059d467e4f7400a6e1`。它完整继承 `history7_flat20_stair_quality_20260801` 的 actor、optimizer、AMP、PD、动作尺度、命令、课程和平地奖励；没有重建机制。
- 平地 DR3 改为 35 kg 单因素边界搜索：`mass=0..35 kg`、stiffness `0.5..1.5`、damping `0.45..1.55`、latency `0..5 step`、gain spread `0.16`、friction `0.15..1.30`、COM offset `0.065 m`、gyro bias `0.065`、gravity bias `0.04`、push `1.0 m/s / 8 s`。平地人口由 `40/20/20/20` 改为 `20/20/20/40`；楼梯严格保持 `60/20/20/0`，DR3 仍为 single-only，不提前混入 compound。
- 楼梯保留现有方向/进展主驱动，只补三个由同协议诊断确定的连续质量漏洞：约 `11.5 deg` 以内 pitch 免罚、超过后使用 Huber 尾部；最差腿滑移超过 `0.20 m/s` 后增加未饱和 Huber 尾部；terrain 重触地改用原始 touchdown `|vz|` 的 Huber 尾部，解决旧 `impact_over` 在约 `1.1 m/s` 后把 `1.4` 与 `3.4 m/s` 判成同分的问题。所有增量在平地 scope 严格为零，不增加状态机、落脚点或逐腿序列。
- 本地及远端 `compileall`、核心/严格站立、DR 人口、command-conditioned AMP、stand gait clock、下楼奖励 verifier 均串行通过。远端归档哈希与本地一致。数据盘切换前只裁剪当前旧 run 的密集重复 checkpoint，保留 `20k/40k/64k(诊断)/78k/88k/latest/best`；清理清单为 `/root/gpufree-data/cleanup_manifests/flat20_stairq_dense_prune_20260801.json`，释放约 `1.70 GB`，payload、诊断和其他历史点未动。
- 新正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260801_flat35_stair_quality_resume88k_long`，tmux `rl_train_flat35_stairq`。固定恢复源为旧 run 的 `agent_88000.pt`，SHA256 `5ec836c30ea5a30670d95b5a5666af4a9f0e24faeeb7982cda6352745f28385e`；课程已精确恢复 step `88000`、`real_mean=7.520`、DR level 3。运行时总人口实测约 `24.1/19.9/19.9/36.0%`，五类单因素均约 20%，single/compound/full=`1/0/0`；effective config 已确认平地 `20/20/20/40`、楼梯 `60/20/20/0`。
- 启动约 1k 的瞬态数据为 F/B/L/Y=`0.865/0.827/0.825/0.872`、下/上楼=`7.37/7.57`、fall=`0`，不能据此判定 35 kg 边界。下一步必须用完整 10 分钟窗口联合判断：四方向、base height/core、tilt/fall、duty、slip/high-slip、touchdown，以及楼梯课程与新增 `terrain_pitch_excess`/重触地/最差腿滑移成本。没有联合崩溃就继续 35 kg 适应并准备 50 kg；出现连续联合崩溃才回到 20 kg 最后稳定强度。楼梯优化效果用训练趋势与后续强制 DR2 的 20 cm 系统诊断交叉确认。
## 2026-08-01 16:20：35 kg 阶段提前进入两因素 compound

- 用户进一步明确：compound 应当尽早进入，不能等单因素推到极致后才考虑。路线已修正为“单因素只用于识别短板；在当前强度站住后立即加入少因素组合；single 与 compound 共同向后续强度推进”。
- 35 kg single-only 的首个完整 10 分钟窗口覆盖 `1.35k-5.71k`。相对切换前 20 kg 最后窗口，F/B/L/Y 仅下降约 `0.009/0.012/0.012/0.014`，上下楼各约 `-0.030`，base height 不变；slip/high-slip 增加约 `0.0027/0.0029`、support instability 增加 `0.0032`、fall 从约 `0.00015` 到 `0.00047`。后半窗 tilt、touchdown、impact/pitch/slip-tail 成本均改善，只有 duty 与下楼偏弱，因此不是明确联合崩溃，不回退 20 kg。
- 新 payload 为 `/root/gpufree-data/training_payloads/history7_flat35_compound40_stair_quality_20260801`，本地归档 SHA256 `64d256e2117c4516026ccae870856579352bb309d30656af76727b725dcf4075`。相对已验证的 35 kg 包，唯一训练语义变化是 DR3 profile 从 `1/0/0` 改为 `0.60 single / 0.40 two-factor compound / 0 full`；35 kg 强度、`20/20/20/40` 平地人口、奖励、actor/optimizer、AMP、PD、命令、课程及楼梯 `60/20/20/0` 均未变。本地/远端 compile、核心奖励和 DR profile 契约串行通过。
- 新正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260801_flat35_compound40_stair_quality_resume8k_long`，tmux `rl_train_flat35_c40`。恢复源是前一 35 kg single run 的 `agent_8000.pt`，SHA256 `c49f47c0d2bbe0793cf2ae2c10ae7a4e1a7e317da6bc04f13a39ff06eda64eac`；课程精确恢复 step `8000`、`real_mean=7.373`、DR3。
- 运行时 DR3 profile 实测约 `56.4% single / 43.6% compound / 0 full`；五因素 active 比例之和约 `1.436`，与 `0.564*1 + 0.436*2` 一致，证明每个 compound row 确实同时启用两个因素。首个约 2 分钟值只作为压力跃迁瞬态，不作结论；必须继续完整 10 分钟窗口，判断四方向、核心/高度、duty/slip/touchdown/fall 和楼梯 `7+` 是否联合恢复。若 compound 可适应，下一步在保留 compound 的情况下推进 50 kg；若连续联合崩溃，则回到 35 kg single 或降低 compound 比例，而不是笼统回退全部 DR。
## 2026-08-01 16:35：35 kg compound 首个完整窗口

- 首个避开启动段的完整 10 分钟窗口覆盖 `0.98k-5.28k`。F/B/L/Y=`0.854/0.844/0.812/0.858`，上下楼=`7.396/7.537`，duty=`0.594`，slip/high-slip=`0.0400/0.0417`，touchdown=`0.358`，tilt=`2.41 deg`，support instability=`0.0345`，fall=`0.00059`。
- 相对 35 kg single 最后 10 分钟，F/B/L/Y 变化约 `-0.009/-0.002/-0.011/-0.005`，上下楼反而约 `+0.057/+0.060`；duty `-0.016`、slip/high-slip `+0.0018/+0.0032`、touchdown `+0.0165`、support `+0.0021`、fall `+0.00015`。这说明 compound 已暴露接触与横移短板，但没有四方向、核心、楼梯和生存共同坍缩。
- 窗口后半段 F、duty、touchdown、support、impact 和上楼改善；slip/high-slip 与 fall 仍恶化，横移停在约 `0.812`。当前不回退 compound、不升 50 kg、不做物理诊断，继续第二个完整 10 分钟窗口。若接触尾部继续恶化，则先定位两因素组合并调整 compound 配额/覆盖；若恢复，则保留 compound 推进 50 kg。所有平地诊断强制当前最高 DR3，楼梯诊断强制当前最高 DR2。

## 2026-08-01 16:45：保留早期 compound 推进 50 kg

- 第二个完整窗口覆盖 `5.29k-9.37k`。F/B/L/Y=`0.862/0.841/0.817/0.856`，上下楼=`7.393/7.530`，duty=`0.602`，slip/high-slip=`0.03982/0.04151`，touchdown=`0.348`，tilt=`2.42 deg`，support instability=`0.0357`，fall=`0.00056`。相对首窗，F/L、duty、touchdown、slip/high-slip 和 fall 均恢复，B/Y 仅小幅波动，楼梯守住 `7+`；不存在四方向、核心/高度、接触与生存的联合崩溃。
- 路线口径固定为：compound 从较早强度开始与 single 共存，二者随 DR 强度共同推进；不再等待单因素推到极致后才考虑组合。single 负责保留覆盖与可归因性，compound 负责提前暴露耦合短板。若后续失败，优先定位具体组合并调整组合覆盖/比例，不笼统退回 single-only。
- 新 payload 为 `/root/gpufree-data/training_payloads/history7_flat50_compound40_stair_quality_20260801`，归档 SHA256 `29f0f9b80ac4b56a7bf7d0c6394400b61ecc2b0a0c715763ccfd3e42a1f526e8`。相对 35 kg compound 包，唯一训练语义变化是平地 DR3 `mass_range_3: 0..35 -> 0..50 kg`；`60/40/0` single/two-factor compound/full、其余 DR 范围、平地人口、楼梯 DR2、奖励、actor/optimizer、AMP、PD、命令和课程均保持不变。本地及远端 compile、DR 分层和核心/站立契约已串行通过。
- 新正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260801_flat50_compound40_stair_quality_resume10k_long`，tmux `rl_train_flat50_c40`。恢复源固定为上一 run 的 `agent_10000.pt`，SHA256 `af4b10535e6ca7ef26726859da36e8a4770275fec3af6ca1937f54b531e109b8`。旧进程确认退出后才启动新进程；课程精确恢复 step `10000`、`real_mean=7.490`、DR3。effective config 已确认质量 `0..50 kg`，运行时 profile 仍约 `56.4% single / 43.6% compound / 0 full`。从启动瞬态之后开始严格 10 分钟窗口，按四方向、核心/高度、duty/slip/touchdown/fall 与楼梯 `7+` 联合判断是否触及崩溃边界。

## 2026-08-01 17:20：纠正为平地综合 DR 与楼梯 nominal 质量双主线

- 用户再次明确：`10/20/35/50 kg` 是整套 DR 强度的参考刻度，不是只提高附加质量；平地 DR 应直接使用综合随机化，质量/质心、摩擦接触、执行器、延迟与感知、推搡的强度共同推进，不再等待 single 或 compound 推到极致。楼梯只需要一定程度的 DR，不承受强综合/高负重；楼梯当前主目标是 nominal 条件下的通过质量，不要求课程等级继续提高。两个任务在同一 actor 上并行推进。
- 因此当前 `flat50_compound40` run 只能解释为“50 kg 质量轴 + 上一档其余 DR 端点”的对照，不能称为完整 50 kg 档。其三个完整窗口依次约为：F/B/L/Y `0.852/0.834/0.814/0.857`、`0.855/0.840/0.811/0.854`、`0.848/0.825/0.810/0.852`；上下楼 `7.437/7.541`、`7.384/7.494`、`7.443/7.504`。slip/high-slip 后续改善，但 touchdown 从约 `0.349 -> 0.365 -> 0.375`，第三窗 B、support 与高度风险也变差。没有联合崩溃，但不是净优化；楼梯只是在 `7+` 震荡，没有明确质量改善。
- 已构造本地候选 `history7_flat50_full_stair_quality_20260801`。平地人口仍为 DR0/1/2/3=`20/20/20/40`，但进入 DR3 的平地环境改为 `100% full`，五类因素同时启用；低层 `60%` 继续提供 nominal、现实误差和单项异常锚点。楼梯仍为 `60/20/20/0`，不会进入强 DR3 full，现有方向/进展主驱动和楼梯质量项不变。
- 候选 DR3 全包端点同步增强为：附加质量 `0..50 kg`，stiffness `0.4..1.6`，damping `0.35..1.65`，latency `0..6 step`，逐关节 gain spread `0.20`，friction `0.10..1.40`，CoM `+/-0.08 m`，gyro bias `+/-0.08 rad/s`，gravity bias `+/-0.05`，平地 push 每 `7 s` 一次、速度增量随机 `0.575..1.15 m/s`，yaw 上限约 `0.92 rad/s`。推扰是 delta-v，重载本身已放大冲量，因此没有按质量机械同比放大到 `1.4+ m/s`。
- 本地 `compileall`、DR 地形分层/full 掩码与端点、核心/严格站立、command-conditioned AMP、stand gait clock 和下楼奖励 verifier 已全部串行通过。下一步是归档部署，以当前 mass-only run 的最新完整 checkpoint 恢复，运行时必须确认 flat DR3 full 比例为 `1.0`、五类 stress active 比例均为 `1.0`、stairs DR3 为 `0`、延迟最大值为 `6`，再进入严格 10 分钟监控。后续楼梯验收看 nominal 物理质量，不以课程继续上涨代替。

## 2026-08-01 19:05：full DR 与楼梯质量密度调整正式启动

- `flat50_compound40` 后续相邻两个约 10 分钟窗仅有小幅回摆：touchdown `0.371 -> 0.367`、duty `0.618 -> 0.624`、tilt `2.53 -> 2.50 deg`，下/上楼 `7.40/7.52 -> 7.40/7.53`；仍未超过更早质量，不能判为楼梯质量改善。原始 20 cm 诊断复核表明硬触地呈明显双峰，上楼约 `19-27%` 触地事件超过 `0.40 m/s` 且 p90 常在 `1.0-1.7 m/s`，现有 Huber 尾项能够覆盖真正重砸；不能仅凭聚合 touchdown 均值机械降低阈值。
- 更明确的短板是楼梯质量梯度密度：旧人口为平地/下楼/上楼=`90/5/5%`，且楼梯一半仍在高难度前沿。新人口改为 `80/10/10%`，replay 改为 `75%` 且固定 level `5-6`（约 `18-21 cm`）。因此每个楼梯方向的 frontier 绝对人口仍是 `2.5%`，与旧配置完全相同；质量带总人口则从 `5%` 增到 `15%`。方向/进展主驱动、已有连续质量项、actor、AMP、PD、PPO 和楼梯 DR `60/20/20/0` 均未改变。
- 人口变化暴露并修复了课程恢复漏洞：v4 状态中的 `terrain_types` 是地形列索引，无法识别同一列索引下的语义比例变化，旧代码会把平地列等级错配给新增楼梯列。恢复现在比较源 run 与当前配置的语义地形比例；布局变化时从源遥测按类型重映射，楼梯优先使用源 frontier 均值，再由当前 fixed replay 覆盖质量带。phase、门控和 DR 等非地形状态仍从原状态精确恢复。
- 最终 payload 为 `/root/gpufree-data/training_payloads/history7_flat50_full_stair_quality_20260801`，归档 SHA256 `24bb4c7722e77550963f8c6485e888e1fbdf35bce0b77bb7530702732f5019a`。本地和远端已串行通过核心/严格站立、地形分层 full DR、AMP、gait clock、下楼奖励及 `compileall`。两个启动短 run 只用于发现课程错配，均未继承 actor。
- 当前正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260801_flat50_full_stair_quality_resume32k_remap2_long`，tmux `rl_train_flat50_full_stairq`；固定恢复源为 `flat50_compound40` 的 `agent_32000.pt`。启动确认 checkpoint 已加载、phase3/DR3 保持，replay 下/上均为 `5.5`，frontier 下/上为 `8/9`。平地 DR3 `full=1.0`，mechanics/contact/actuation/sensing/push active 均为 `1.0`，latency max=`6`，楼梯 push 与 DR3 均为零。
- 新总楼梯均值约 `6.14/6.39` 是 `75%` level `5-6` 与 `25%` frontier `8/9` 的采样加权值，不能与旧 `7.4/7.5` 总均值直接比较。后续严格 10 分钟监控分别看 replay/frontier 保持、平地 full DR 下四方向/核心/高度/步态/接触/fall，以及训练形成可信趋势后用 nominal 20 cm 诊断验收楼梯真实滑移、冲击、姿态、方向漂移和通过率。

## 2026-08-01 19:25：50 kg full 命中联合崩溃边界，退到 35 kg full

- 50 kg full 的首个完整约 10 分钟窗覆盖约 `3.57k-8.07k`。F/B/L/Y=`0.776/0.702/0.683/0.767`，support instability=`0.0975`，tilt=`4.31 deg`，base height=`0.537 m`，height risk=`0.0826`，fall=`0.00281`，touchdown=`0.470`。相对切换前成熟基线呈四方向、核心/高度、接触与生存联合恶化；后半窗 B/L/Y 与稳定性也没有恢复。B/L/Y 来自平地，不能由楼梯人口增加解释，因此按“加速到明显崩溃后退一档”的路线判定 50 kg full 当前不可直接适应。
- 不继承该失败 run 的 actor；保留五类同时开启的 full 结构、`80/10/10` 地形人口、`75%` level `5-6` 楼梯质量带、frontier 和全部奖励，只把 DR3 端点退回与 35 kg 匹配：mass `0..35 kg`、stiffness `0.5..1.5`、damping `0.45..1.55`、latency `0..5`、gain spread `0.16`、friction `0.15..1.30`、CoM/gyro/gravity `0.065/0.065/0.04`、push `1.0 m/s / 8 s`。
- 新 payload 为 `/root/gpufree-data/training_payloads/history7_flat35_full_stair_quality_20260801`，SHA256 `b54f8c8cd2710b6827daecc1cf17c151c2239c69cad39b894996624f475572e4`；本地与远端核心/站立、full DR 分层及 compile 契约均通过。正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260801_flat35_full_stair_quality_resume32k_long`，tmux `rl_train_flat35_full_stairq`，仍从未受失败 run 污染的原 `agent_32000.pt` 启动。运行时确认 frontier 下/上=`8/9`、full 和五类 active 均=`1.0`、latency max=`5`，现进入严格 10 分钟窗口。

## 2026-08-02：35 kg full 平台结论与 50/50 中间分布长训

- `35 kg full` 不是轮数不足。中断前后等长窗口均表现为联合平台和能力交换：恢复后 `0-5k` 的 F/B/L/Y=`0.783/0.730/0.727/0.795`，`5-10k` 为 `0.789/0.733/0.722/0.789`；duty `0.587 -> 0.582`，slip `0.0484 -> 0.0493`，tilt `4.05 -> 4.26 deg`，support instability `0.0891 -> 0.0915`，髋偏差和移动高度也继续恶化，楼梯仅维持。不能再把继续等待当作解释。
- 数据盘曾达到 `49G/49G`，导致该路线的 `agent_20000.pt` 写坏。已删除 62 个明确失效或密集冗余检查点，释放约 `2.48 GiB`；payload、诊断、日志、真实 resume 链继承点和少数有价值检查点均保留。后续统一使用 `TAILI_CHECKPOINT_INTERVAL=10000`，并持续看磁盘余量。
- 新候选只调整 DR3 stress 内部人口：`stress_profile_weights: [0.00, 0.50, 0.50]`，即 0 single、50% 双因素 compound、50% 五因素 full。35 kg 全套端点、平地 DR3 总人口、楼梯、奖励、AMP、PD、PPO、命令和质量约束均不变。目的不是退回单因素，而是在同一强度保留 full 验收压力，同时补上可学习的联合中间分布。
- payload 为 `/root/gpufree-data/training_payloads/history7_flat35_full50_compound50_stair_quality_20260801`，归档 SHA256 `e7c1915a30cd16faa4c1073a63e58a8d7f62fac4f9ad0a621fe0ee6f617f0f8b`。当前 run 为 `/root/gpufree-data/taili_runs/taili_train_20260801_flat35_full50_compound50_stair_quality_resume32k_long`，恢复源固定为未受 full 污染的 `/root/gpufree-data/taili_runs/taili_train_20260801_flat50_compound40_stair_quality_resume10k_long/checkpoints/agent_32000.pt`，而不是失败 full actor。
- 启动契约已实测生效：phase3、DR3、语义课程重映射；楼梯 replay level `5-6`、均值 `5.5`，初始 frontier 下/上=`8/9`；stress single/compound/full=`0.000/0.491/0.509`，compound 每环境 2 因素、full 每环境 5 因素；楼梯无 DR3 stress/push；检查点间隔 `10k`。
- 未污染源成熟末窗约为 F/B/L/Y=`0.859/0.834/0.805/0.848`、duty=`0.620`、slip=`0.037`、height=`0.547 m`、tilt=`2.45 deg`、support instability=`0.039`、fall=`0.0005`。当前 50/50 候选明显优于 100% full，但尚未恢复该联合质量。
- 当前候选 `0-5k -> 45-50k` 的主要变化：F `0.811 -> 0.812`、B `0.776 -> 0.775`、L `0.749 -> 0.753`、Y `0.812 -> 0.814`；duty `0.594 -> 0.589`，slip `0.0447 -> 0.0426`，tilt `3.56 -> 3.62 deg`，support `0.0759 -> 0.0699`，height `0.540 -> 0.546 m`，fall 约 `0.0011`。中间阶段 tilt 曾升至约 `3.75 deg` 后回落，髋成本、touchdown 与 duty 也开始从低点回收；楼梯 frontier 仍约下 `8.5`、上 `8.6`，未坍缩。
- 当前判断是“中间分布有效，但仍处在缓慢适应与能力再分配中”，不是通过，也不是需要立即重启的明确失败。用户要求避免频繁重启，因此保持当前训练连续长跑，继续按严格 10 分钟等长窗口联合观察四方向、核心/高度、duty/slip/touchdown/fall、髋偏差和楼梯 frontier；不并行做物理诊断，不因单点波动干预。

## 2026-08-02：50/50 长训 120k 裁决与 DR 分层诊断

- 当前 run 连续训练到 `121.9k` 后正常停止，冻结点为完整写入的 `agent_120000.pt`。`100k-120k` 的多个严格 10 分钟窗口持续表现为能力交换，而不是联合恢复：四方向、duty/slip、tilt/support、高度、touchdown、髋成本和楼梯 frontier 在相邻窗口中交替改善与恶化。最后一个完整窗口约为 F/B/L/Y=`0.814/0.777/0.768/0.824`、duty=`0.572`、slip=`0.0463`、tilt=`3.27 deg`、support=`0.0744`、height=`0.5450 m`、fall=`0.00056`、touchdown=`0.416`，楼梯 frontier 下/上=`8.24/8.76`。因此已排除“只需继续等轮数”的解释。
- 系统诊断 `20260801_191948` 使用冻结的 `agent_120000.pt`，单环境、单 IsaacLab 作业顺序执行 flat DR0 nominal、DR3 compound、DR3 full；每个 case 均覆盖站立、前进、后退、横移、yaw 和方向间 2 秒零命令段，共 `3900` 行。输出位于 `/root/gpufree-data/diag_runs/locomotion_console_20260801_191948_directions`，系统报告和回放均已生成。
- nominal 不是主驱动失效：稳态前/后/横移线速度误差均值约 `0.039/0.037/0.035 m/s`，yaw 误差约 `0.025 rad/s`，移动倾角 p95 约 `0.96/1.21/1.77/1.71 deg`。但它仍非理想终点：横移需要明显髋运动，前进 touchdown speed p90 约 `0.677 m/s`，方向切换后的严格静止通常需要约 `0.7-1.34 s`，初始站立和部分零命令段仍有约 `3 deg` 倾角尾部。
- compound 本次确定抽中 `mechanics + push`，附加质量约 `12.35 kg`，并带 CoM 偏移；它能够恢复稳态跟踪，但前进后的零命令段峰值平移速度约 `0.92 m/s`，2 秒内未达到持续严格静止，倾角 p95 达 `6.34 deg`，touchdown speed p90 达 `0.69 m/s`。这表明组合下主要短板包含受扰后的制动/恢复以及步态接触质量，而不是完全没有方向驱动。
- full 本次五类因素全部启用，实际样本约为附加质量 `19.3 kg`、friction `0.945`、stiffness/damping `1.017/0.929`、latency `0`。前进仍能跟踪，但后退和横移稳态误差分别约 `0.113/0.095 m/s`，yaw 平移泄漏均值约 `0.123 m/s`；多个零命令段在 2 秒内不能持续满足 `speed<0.05 m/s` 且 `wxy<0.15 rad/s`，移动核心/高度波动、duty、高频摆腿和 touchdown 均明显劣于 nominal。
- 当前可确定的数学问题是：方向主奖励仍有效，但在强随机化下，策略可以通过改变高度、接触节律和动作频谱保持部分跟踪；零命令严格闭合和受扰恢复又不足以在短时间内把状态拉回理想流形。不能据此直接降低全部 DR 或重构奖励，因为 compound 只覆盖 mechanics+push，full 又把五类耦合。下一步只运行一个顺序单因素系统诊断，固定 DR3 并分别强制 mechanics/contact/actuation/sensing/push，沿用相同全方向计划；得到归因后再决定 DR 人口、端点或训练覆盖如何调整。

## 2026-08-02：DR3 强单因素可辨识样本恢复

- 顺序单因素系统诊断 `20260801_193243` 已完成，共 `6500` 行，分别强制 mechanics/contact/actuation/sensing/push。actuation 与 sensing 对 touchdown、slip、duty、站立倾角和核心波动影响最大，mechanics 次之，push 主要损害受扰恢复与 yaw 平移纯度；各单因素均能生存，问题是强联合分布下的信用混杂和质量交换，不是基本控制能力缺失。contact 本次抽到 friction `0.945`，不能代表低摩擦端点。
- 新候选只把 DR3 stress profile 从 `single/compound/full=0/0.50/0.50` 改为 `0.20/0.50/0.30`，为 35 kg 强度端点补回可辨识单因素梯度，同时保留双因素耦合和五因素联合压力。35 kg 全套端点、平地 DR3 总人口、楼梯 DR0/1/2 人口、奖励、AMP、PD、PPO、动作尺度、命令和课程均不变；相对上一 payload 的目录哈希差分只有 YAML、配置默认值及两个 verifier。
- payload 为 `/root/gpufree-data/training_payloads/history7_flat35_dr3_single20_compound50_full30_stair_quality_20260802`，归档 SHA256 `8b892ea59b481ed5e387152f419cadf9862fa626d7f07933ede3a182fcb485c5`。本地五项 verifier 与 compileall、远端 DR profile、核心/严格站立及 compileall 均串行通过。
- 正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260802_flat35_dr3_single20_compound50_full30_stair_quality_resume120k_long`，tmux `rl_train_dr3_s20c50f30`；恢复源固定为上一 run 的完整 `agent_120000.pt`。checkpoint、optimizer 与课程已载入，课程精确恢复 step `120000`、DR3；楼梯 replay/frontier 布局保持。运行时实测 stress single/compound/full=`0.207/0.518/0.274`，延迟最大 `5`，楼梯 stress/push 为零。
- 首个严格窗口覆盖 `step 760-5020`。F/B/L/Y=`0.826/0.798/0.788/0.836`，duty=`0.582`，slip=`0.0441`，tilt=`3.12 deg`，support=`0.0684`，height=`0.5469 m`，fall=`0.00047`，touchdown=`0.401`，楼梯 frontier 下/上=`8.36/8.58`。相对上一 run 冻结前最后窗口，跟踪、核心/高度、接触和生存广泛恢复，但上楼暂时下降，不能单窗验收。
- 第二个严格窗口覆盖 `step 5310-9560`。F/B/L/Y=`0.820/0.812/0.781/0.831`，duty=`0.584`，slip=`0.0448`，tilt=`3.15 deg`，support=`0.0680`，height=`0.5462 m`，fall=`0.00038`，touchdown=`0.407`，楼梯 frontier 下/上=`8.28/8.60`；后半窗上楼回升到 `8.69`，排除首窗所见的持续下跌。当前表现是整体优于旧冻结窗、局部仍有交换，不构成重启依据；继续长跑，以后续等长窗口确认收益是否稳定。
- 第三个严格窗口覆盖 `step 9950-14210`。F/B/L/Y=`0.823/0.813/0.781/0.832`，duty=`0.586`，slip=`0.0439`，tilt=`3.18 deg`，support=`0.0690`，height=`0.5440 m`，fall=`0.00040`，touchdown=`0.410`，楼梯 frontier 下/上=`8.37/8.70`。上楼已基本恢复旧冻结水平，平地多数联合指标仍优于旧窗；高度和 touchdown 有局部回摆，但没有跟踪、核心、接触、生存与楼梯共同退化。`agent_10000.pt` 已完整写入，大小 `43607603` 字节；数据盘剩约 `1.2 GB`，继续原配置长跑，不频繁重启，并按检查点增长及时裁剪密集冗余点。
- 第四窗 `14650-18880` 的 F/B/L/Y=`0.824/0.798/0.781/0.833`，duty=`0.583`，slip=`0.0443`，tilt=`3.22 deg`，support=`0.0707`，height=`0.5417 m`，touchdown=`0.402`，楼梯下/上=`8.35/8.67`。聚合高度连续下降一度构成疑点，但实现复核确认 `flat_move_height` 的目标为 `0.547196 m`、免罚带 `0.012 m`，且严格项只作用于 `ideal_flat_scope = flat AND DR0`；telemetry 的 base height 则聚合全部 DR tier 和楼梯。高度下降时该惩罚反而减轻，表示 DR0 子集可能更接近目标，而强 DR 子集采用较低安全姿态，不能用聚合均值直接宣判理想平地高度失败。
- 第五窗 `20140-24360` 排除了聚合高度持续坍缩：F/B/L/Y=`0.836/0.804/0.789/0.836`，duty=`0.587`，slip=`0.0443`，tilt=`3.18 deg`，support=`0.0694`，height=`0.5433 m`，height risk=`0.0616`，fall=`0.00047`，touchdown=`0.411`，楼梯下/上=`8.37/8.69`；后半窗 height=`0.5442 m`，height risk、tilt、support、slip 和 touchdown 同时改善，上楼回到 `8.79`。`agent_20000.pt` 完整，大小 `43607603` 字节，磁盘约剩 `1.1 GB`。当前候选有持续正向证据，保持原配置长跑；成熟后用系统 DR0/DR3 分离诊断验收名义平地与强 DR 姿态，不能提前把两者混为一个均值。
- 第六窗 `24860-29090` 继续稳定：F/B/L/Y=`0.820/0.816/0.781/0.838`，duty=`0.586`，slip=`0.0444`，tilt=`3.18 deg`，support=`0.0684`，height=`0.5450 m`，height risk=`0.0603`，fall=`0.00055`，touchdown=`0.409`，楼梯下/上=`8.31/8.69`。F 有回摆但 B/Y、核心高度和风险保持，未回到旧候选的联合平台。`agent_30000.pt` 已完整写入，大小 `43607603` 字节；正式进程 PID `455965` 继续运行。当前新 run 只保留 `10k/20k/30k` 三个完整锚点，磁盘约剩 `1.1 GB`，到后续检查点再按实际价值裁剪，不干扰训练。
- 长训已推进到约 `85k`，正式进程 PID `455965` 持续运行。最近三个不重叠十分钟窗口依次为 `72.23k-76.36k`、`76.37k-80.51k`、`80.52k-84.67k`：F=`0.833/0.833/0.828`，B=`0.799/0.806/0.813`，L=`0.777/0.782/0.778`，Y=`0.830/0.834/0.835`；duty=`0.590/0.586/0.587`，slip=`0.0482/0.0469/0.0466`，touchdown=`0.404/0.403/0.402`，tilt=`3.02/3.02/3.08 deg`，support=`0.0689/0.0666/0.0692`，fall=`0.00030/0.00041/0.00027`，楼梯下=`8.20/8.17/8.23`、上=`8.75/8.62/8.71`。结论是长期保持、局部震荡：相对旧冻结窗，跟踪、duty、核心、触地、生存与楼梯总体更好；slip 已回到接近旧水平，F/L 与聚合高度仍有周期性回摆，因此当前候选有效但尚未完成最终验收，不调整、不重启。
- 数据盘曾降到 `756 MB`。已仅从当前 run 删除密集且可替代的 `20k/40k/60k/70k`，保留 `10k` 初始响应、`30k` 已验证锚点、`50k` 中期点与完整 `80k` 成熟点；payload、诊断、历史恢复源和训练进程均未触碰。清理后约剩 `921 MB`，后续继续按价值裁剪，避免 checkpoint 写坏。
- 用户随后要求主要清理更早历史且禁止触碰 payload。第一批从 10 条 `20260724-20260731` 密集旧 run 中删除 `68` 个未被任何 `run.json` 引用的中间 checkpoint，释放 `2,965,311,824` 字节；每条 run 保留 `best_agent.pt`、实际 resume 引用、末点和少量 `10k` 阶段锚点。清单为 `/root/gpufree-data/cleanup_manifests/older_dense_checkpoint_prune_20260802.json`。
- 第二批只考虑 `20260730` 及更早、排除所有 snapshot，并统一保留 best、末点、所有 `10k` 里程碑、resume 引用和手工保护的历史/诊断点；删除额外 `24` 个 `2k/4k/6k/8k` 等密集启动段 checkpoint，释放 `1,046,579,748` 字节。清单为 `/root/gpufree-data/cleanup_manifests/older_global_milestone_prune_20260802.json`。两批合计删除 `92` 个文件、释放 `4,011,891,572` 字节（约 `3.74 GiB`）。
- 清理后数据盘可用约 `4.6 GB`、占用 `91%`；`/root/gpufree-data/training_payloads` 仍为 `6.3 GB`，没有删除或修改任何 payload。历史 `7+ agent_30000`、旧全局 `30k`、站立链 `26k`、理想平地 `40k`、保留区中的 `20260717 cg stair agent_58000`、当前恢复源 `agent_120000.pt` 均完整。正式训练 PID `455965` 未中断，已到约 `92.6k`，完整 checkpoint 保留为 `10k/30k/50k/80k/90k`。

## 2026-08-02：DR 平地 completion 质量耦合

- `20/50/30` 长训在约 `155k` 前的分层系统诊断确认了一个具体奖励漏洞：`flat_tracking_completion` 在所有平地 DR tier 都可拿满，而直接 `flat_motion_quality_penalty`、精确移动高度和标准姿态只属于 `ideal_flat_scope = flat AND DR0`。普通 tracking 虽有质量调制，但仍保留非零信用底线。因此 DR 样本可以通过改变高度、接触节律和动作频谱保持完整 completion 信用，并经共享 actor 损害 nominal。该结论不等于零命令站立已经解决；`agent_100000.pt` 的动转静段仍存在不安静站立，这是独立的全局缺口。
- 新候选 `history7_flat35_dr3_completion_quality_20260802` 只修复上述漏洞：DR 平地 completion 乘以 `0.35 + 0.65 * flat_motion_quality`；DR0 completion 保持原值，DR0 的直接质量成本、精确高度与标准姿态范围也不变。DR `single/compound/full=0.20/0.50/0.30`、35 kg 端点、楼梯、AMP、PD、PPO、命令、静止分支和安全项均未改。源码逐文件哈希差分严格只有 `taili_reward.py`、YAML 和对应 verifier 三文件。
- 本地与远端均串行通过核心/严格站立、command-conditioned AMP、stand gait clock、下楼奖励、DR 分层 verifier 及全包 `compileall`。归档 SHA256 为 `7d287990ff5cee8e1f501d7386d1978e9803fdd09dc22efcb079ca83b01ed18a`；远端 payload 为 `/root/gpufree-data/training_payloads/history7_flat35_dr3_completion_quality_20260802`。
- 新 run 为 `/root/gpufree-data/taili_runs/taili_train_20260802_flat35_dr3_completion_quality_resume100k_long`，tmux `rl_train_dr3_qcompletion`，PID `863992`。恢复源固定为上一 run 的完整 `agent_100000.pt`，SHA256 `d1f63e0c27f7ddf69fc3f0bdebec6c9fbef1163f3fb48e129807102219150852`；课程 sidecar 已在切换前冻结，启动实测精确恢复 step `150000`、real mean `6.255`、DR3。原有 `resume_finetune` 契约保持：清除旧 Adam 动量，以 `1e-5`、2 epochs、ratio clip `0.10` 小步继续。
- 首个完整十分钟窗口覆盖 `step 420-4470`。全窗 F/B/L/Y=`0.825/0.802/0.766/0.829`，duty=`0.596`，slip/high-slip=`0.0520/0.0612`，tilt=`3.18 deg`，support=`0.0641`，height=`0.5447 m`，fall=`0.00031`，touchdown=`0.406`，楼梯 frontier 下/上=`8.21/8.63`。前半到后半 tilt `3.22 -> 3.14 deg`、touchdown `0.411 -> 0.401`、B/L/Y 略升，但 F `0.827 -> 0.822`、slip `0.0510 -> 0.0529`、楼梯 frontier 略降；没有整体坍缩，也尚未形成联合恢复，继续等长窗口，不重启。

## 2026-08-02：精确零命令站立语义修复候选

- `agent_100000.pt` 的顺序全方向诊断位于 `/root/gpufree-data/diag_runs/locomotion_console_20260802_000556_directions`，依次覆盖 DR0、DR3 compound、DR3 full。DR0 动转静尾段仍有线速度约 `0.030-0.036 m/s`、roll/pitch 角速度约 `0.047-0.056 rad/s`、关节速度约 `0.104-0.117 rad/s`；compound/full 更差。因此 DR 会放大站立残差，但不是唯一根因。
- 站立和移动高度数值目标相同，均为 `0.5471961541 m`。DR0 移动尾段高度约 `0.549-0.557 m`，动转静尾段约 `0.536-0.539 m`；差异来自有效奖励而不是两个目标值：移动有直接高度/核心约束，站立高度核较宽，且旧 `stand_gate` 把线速度命令 `<=0.1`、yaw `<=0.05` 的 near-zero 一并当成站立，严格成本却只作用于精确零命令。DR 平地旧逻辑还完全关闭了标准高度/核心姿态吸引，只保留安全支撑。
- 候选 `history7_flat35_dr3_completion_stand_exact_20260802` 保留当前 completion、移动、楼梯、DR 强度、AMP、PD、PPO、命令比例与课程。奖励层从命令重建互斥 scope：精确零命令才领取 `stand/contact`；非零小命令领取独立连续速度跟踪，不启用步态时钟；普通运动 gate 不变。
- 站立仍以原宽核 `stand/contact` 提供恢复 shaping；另增加严格物理完成正奖励，只有机身/关节/足端静止、支撑合格，并在平地满足高度与核心时才能拿满。原 `w_stand_strict_penalty=0.4` 有界成本保留为辅助，不机械加权。DR 平地保持与 nominal 相同的高度/水平核心目标，但标准关节姿态和髋位置只在 DR0 要求，从而允许负载与参数误差下的必要补偿；非平地仍只要求物理静止与安全支撑。
- 首次 verifier 曾尝试用严格完成度乘掉原 stand 主奖励，实测在可恢复尾态形成正信用 `0.111` 小于严格成本 `0.264` 的负回报悬崖；`0.40` shaping 下限仍不足。该方案已放弃，改为“保留宽核恢复信用 + 独立严格完成正奖励”，对应 verifier 已证明可恢复尾态正信用继续大于辅助成本，同时零误差唯一拿满严格完成奖励。

## 2026-08-02：精确零命令站立修复正式部署

- 最终 payload 已部署到 `/root/gpufree-data/training_payloads/history7_flat35_dr3_completion_stand_exact_20260802`，归档 SHA256 为 `56972bf4b3baee885c956eaa4c200bc754963360d2f6795738ea049c66b33dd9`。相对上一 completion payload 的运行时代码差分仅涉及 `taili_reward.py`、`blind_tp_env.py`、YAML 与对应 verifier；移动、楼梯、DR 强度与人口、AMP、PD、PPO、课程和 completion 质量耦合均保持不变。
- 本地与远端均已串行通过 `compileall`、核心/严格站立、command-conditioned AMP、stand gait clock、下楼奖励以及 DR 分层/profile verifier。旧 completion run 正常停止，恢复源固定为其完整 `agent_30000.pt`，SHA256 为 `7f45b2c943d27799c147f918cdc2a7888c1630f01295654c45776e93a02a3c4e`。
- 新正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260802_flat35_dr3_completion_stand_exact_resume30k_long`，tmux `rl_train_dr3_stand_exact`，训练 PID `955957`。启动已确认精确恢复 curriculum step `30000`、real mean `6.270`、phase3/DR3；继续沿用清 Adam、学习率 `1e-5`、2 epochs、ratio clip `0.10` 的小步适应契约，检查点间隔 `10000`。
- 启动早期约 step `440` 的 F/B/L/Y=`0.845/0.805/0.785/0.815`，`stand_gate=0.169`、`near_zero_gate=0.0156`、`stand_completion=0.0311`、`stand_strict=-0.0580`。该窗口只证明启动未坍缩，尚不能验收动转静；后续先按不重叠窗口观察四方向、duty/slip、核心/高度、支撑/触地、生存与楼梯 frontier，同时单独要求 `stand_completion` 上升、`stand_strict` 绝对值下降。形成可信适应后再用系统顺序诊断复查动转静，不重复旧检查点诊断，也不并行启动额外 PyTorch 作业。
- 截至约 `35.9k`，该候选没有坍缩，但尚未形成全局净优化。最新 `30k+` 窗口 F/B/L/Y=`0.815/0.812/0.773/0.826`、duty=`0.586`、slip/high-slip=`0.0606/0.0846`、touchdown=`0.434`、tilt=`3.18 deg`、support=`0.0752`、height=`0.5409 m`、fall=`0.00038`，楼梯 frontier 下/上=`8.21/8.67`。相对恢复源末端，B、tilt 和 duty 略好，但 F/L/Y、slip、touchdown、support 与 fall 有交换或退化；楼梯只证明能力基本保持，不能证明质量改善；DR3 `single/compound/full=0.207/0.518/0.274` 持续生效且未引发整体坍缩，但尚未通过联合质量验收。
- 站立项有缓慢响应：`0-5k -> 30k+` 的 `stand_completion=0.0334 -> 0.0353`，`stand_strict=-0.0580 -> -0.0554`，`stand/contact=0.103/0.137 -> 0.112/0.148`，但幅度不足以宣告动转静修复。实现复核确认 exact-zero、near-zero 与 walking 奖励作用域互斥，且当前命令过渡机制关闭，新增站立项不会直接落入移动样本；移动接触质量回摆目前应视为共享 actor 适应时的能力交换。保持原 run 连续训练，不在缺少新物理诊断证据时继续机械加权或重启。
- `agent_40000.pt` 的顺序全方向诊断为 `/root/gpufree-data/diag_runs/locomotion_console_20260802_052501_directions`，同一检查点依次覆盖 DR0、DR3 compound、DR3 full。只统计前进/后退/横移切零后 `>=1.5 s` 的尾段：DR0 的线速度/roll-pitch 角速度/yaw 角速度/关节速度 RMS 分别为 `0.0136 m/s`、`0.0425 rad/s`、`0.0208 rad/s`、`0.0793 rad/s`；compound 为 `0.0394/0.0633/0.0281/0.1082`；full 为 `0.0532/0.0942/0.0341/0.1374`。三组均保持四足接触，平均倾角为 `2.02/2.26/2.96 deg`，平均高度为 `0.5452/0.5463/0.5399 m`。因此不能把静止问题单独归因于 DR：DR0 仍有残差，说明 nominal 静止尚未完成；DR 又把动态残差显著放大，说明鲁棒静止适应同样不足。
- 与上一 `agent_100000.pt` 诊断中 DR0 尾段 `0.030-0.036 m/s` 的线速度相比，exact-zero 候选已把 nominal 平移残差降低约一半以上，证明当前语义有效而非完全失败。训练全程 `0-10k -> 60-70k` 的条件化指标仍缓慢改善：`stand_completion / stand_gate = 0.1998 -> 0.2237`，`stand_strict / stand_gate = -0.3369 -> -0.3232`。当前正式 run 保持连续，不因这次诊断立即重启。下一次决策应使用更成熟检查点再次做 DR0/DR3 分离验收；若 nominal 物理残差届时平台，优先检查“严格运动完成度被高度/姿态乘法联结削弱”和“nominal exact-zero 样本在共享 DR actor 中占比不足”，不改动已达到约 88 分的平地移动基本盘，也不重做约 75 分且能力基本合格的楼梯路径。

## 2026-08-02：静止运动完成 shaping

- 随后的三个连续窗口 `69.71-73.92k`、`73.92-78.13k`、`78.13-82.57k` 中，条件化 `stand_completion / stand_gate = 0.2366/0.2367/0.2360`，`stand_strict / stand_gate = -0.3165/-0.3170/-0.3173`，总条件化站立信用 `1.5695/1.5651/1.5556`；静止已连续平台。同期移动、楼梯和 DR3 没有联合坍缩，因此干预依据是静止数学路径平台，而不是重做平地或楼梯。
- 候选 `history7_flat35_dr3_stand_motion_completion_20260802` 只调整精确零命令的完成信用：原联合分数记为 `Q_joint = Q_motion * Q_flat_physical * Q_nominal_pose`，新分数为 `0.35 * Q_motion + 0.65 * Q_joint`。满分和 `w_stand_completion=1.0` 不变，仍只有全部因子为 `1` 才能拿满；平地多维残差下，物理不动的正向梯度不再完全被高度/姿态乘法压低；非平地因 `Q_joint == Q_motion` 而数值完全不变。移动命令、楼梯、DR、AMP、PD、PPO、采样比例、严格成本和宽核 stand/contact 均未改。
- 本地和远端均串行通过核心/严格站立、command-conditioned AMP、stand gait clock、下楼奖励、DR 分层/profile verifier 和全包 `compileall`。非缓存文件哈希差分严格只有 `taili_reward.py`、YAML 和对应 verifier。归档 SHA256 为 `81637a162c546106b7e73a436ee7f089bb481666f8f2f53fa478f6f3304ca154`；远端 payload 为 `/root/gpufree-data/training_payloads/history7_flat35_dr3_stand_motion_completion_20260802_verified`。
- 旧 run 在约 `82.6k` 正常停止；恢复源为其完整 `/root/gpufree-data/taili_runs/taili_train_20260802_flat35_dr3_completion_stand_exact_resume30k_long/checkpoints/agent_80000.pt`，大小 `43607603` 字节。新正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260802_flat35_dr3_stand_motion_completion_resume80k_long`，tmux `rl_train_dr3_stand_motion`。启动确认 `stand_completion_motion_mix=0.350`，精确恢复 phase3/DR3 课程（state step `100000`、real mean `6.245`），并使用 `reset_optimizer=1`、`lr=1e-5`、`epochs=2`、`ratio_clip=0.10`。注意新旧 `stand_completion` 数值定义已改变，不能直接把启动后的机械抬升当成行为改善；主要训练判据是条件化严格成本、stand/contact、全局移动/楼梯/DR 保持，成熟后再做 DR0/DR3 物理诊断。

## 2026-08-02：严格静止运动短板聚合

- `stand_motion_completion` 候选的前七个等长窗口没有形成严格静止净改善。前三窗 `completion/stand=0.2955/0.2918/0.2891`、`strict/stand=-0.3170/-0.3198/-0.3228`；后续至 `31k` 仍围绕 `completion/stand≈0.29`、`strict/stand≈-0.320` 震荡，stand/contact 后两窗还下降。四方向、楼梯和 DR3 没有坍缩，因此不能用继续等待解释该平台。
- `agent_20000.pt` 的系统顺序诊断为 `/root/gpufree-data/diag_runs/locomotion_console_20260802_091449_directions`，协议与 `052501` 完全一致。DR0 动转静尾段的线速度/wxy/yaw/关节速度为 `0.0142/0.0378/0.0215/0.0769`，相对旧诊断 `0.0136/0.0425/0.0208/0.0793`：wxy、关节和倾角改善，但平移/yaw 没有改善。compound 为 `0.0382/0.0596/0.0263/0.0980`；full 为 `0.0483/0.0897/0.0363/0.1446`。三组仍保持四足接触，平均倾角改善为 `1.05/1.42/1.85 deg`。
- 根因是严格运动成本把 speed、wxy、yaw、vz、joint、foot、support、contact 八项按 `0.65*max+0.35*mean` 聚合。DR0 尾段约 `70%` 样本由 joint velocity 单独占据 max 分支，非最大 speed/yaw 获得的聚合梯度约小一个数量级；策略因而可以降低关节/wxy 并换取姿态改善，同时让平移/yaw 残差不降反升。这不是权重不足或训练轮数不足，而是标量聚合允许的确定性交换。
- 新候选 `history7_flat35_dr3_stand_motion_shortfall_20260802` 只把上述八项改为项目已有的“算术均值 + 调和均值”平滑短板聚合，`stand_strict_motion_harmonic_mix=0.65`。每项始终有梯度，较差项仍获得更强梯度，但不再由单一 hard max 独占。completion mix、严格成本权重、移动、楼梯、DR、AMP、PD、PPO、采样与课程均不变；非缓存差分严格只有 `taili_reward.py`、YAML 和核心 verifier。
- 本地和远端已串行通过全部相关 verifier 与 `compileall`。归档 SHA256 为 `0f8e7719dcc9c07089c15e8dec6e0099d7474e9f9d35a774de666a7be1a27222`；远端 verified payload 为 `/root/gpufree-data/training_payloads/history7_flat35_dr3_stand_motion_shortfall_20260802_verified`。
- 当前 run 为 `/root/gpufree-data/taili_runs/taili_train_20260802_flat35_dr3_stand_motion_shortfall_resume20k_long`，tmux `rl_train_dr3_stand_shortfall`。actor 恢复自已诊断的旧候选 `agent_20000.pt`，SHA256 `c7e065d0e5d4fb0c67fae8807f3934b9068a39aab6ed8ddbf8ab75bcf8364532`；课程 sidecar 为旧 run 停止时的 step `30000`，但 phase3/DR3、real mean `6.265` 和地形分布一致。启动确认清 Adam、`lr=1e-5`、2 epochs、clip `0.10` 以及新参数真实生效。接下来按不重叠十分钟窗口要求 `strict/stand` 绝对值下降，同时守住 F/B/L/Y、duty/slip、核心/高度、fall、楼梯 frontier 和 DR3；日志形成可信趋势后再做同协议物理诊断。
- 前四个完整等长窗口 `0.5-4.89k`、`4.9-9.29k`、`9.3-13.69k`、`13.7-18.09k` 的 `completion/stand` 为 `0.3804/0.3796/0.3760/0.3770`，`strict/stand` 为 `-0.2685/-0.2699/-0.2714/-0.2714`；第四窗已停止早期退化，stand/contact 从第三窗的 `0.6745/0.8915` 回升到 `0.6828/0.9044`。随后 `18.1-22.49k -> 22.5-26.89k` 的 completion `0.3752 -> 0.3790`、strict `-0.2724 -> -0.2696`、stand/contact `0.675/0.893 -> 0.684/0.906`，且四方向、fall、楼梯和 DR3 保持，说明新梯度开始响应，不能在此时机械加权或重启。
- `agent_20000.pt` 已通过系统按相同协议完成顺序诊断，job `20260802_104211`，目录 `/root/gpufree-data/diag_runs/locomotion_console_20260802_104211_directions`，单环境依次为 DR0、DR3 compound、DR3 full，共 `4500` 行。DR0 动转静尾段的 speed/wxy/yaw/joint/foot 为 `0.01217/0.03727/0.02112/0.08225/0.01512`；上一候选 `091449` 为 `0.01417/0.03783/0.02150/0.07694/0.01364`。因此平移、wxy、yaw 已共同小幅改善，但 joint/foot 退化，当前只是消除了 hard-max 独占并开始重新分配梯度，尚未完成八项共同收敛。
- 同一诊断排除了 nominal 移动被牺牲：DR0 移动线速度误差 `0.0392 -> 0.0372`、移动倾角均值 `1.43 -> 1.17 deg`；刹车窗口 speed/joint/foot/tilt p95 也改善，wxy 基本持平，yaw 略差。compound 尾段总体接近旧候选；full 的尾段 joint/yaw 改善，但 speed/wxy、足端与刹车峰值仍有交换，说明强 DR 静止适应尚未完成。当前决策是继续原 run 做等长监控；只有日志再次形成可信平台且成熟物理诊断仍显示 joint/foot 交换时，才校准现有严格成本幅度，不退回 hard max、不堆新机制。

## 2026-08-02：40k 诊断裁决与 stop-cycle 根因

- 平滑短板候选后续日志仍缓慢改善：`26.9-31.29k`、`31.3-35.69k`、`35.7-40.09k` 的条件化 `completion/strict` 依次为 `0.3818/-0.2684`、`0.3831/-0.2673`、`0.3888/-0.2643`。但 `agent_40000.pt` 的同协议 DR0 系统诊断 `/root/gpufree-data/diag_runs/locomotion_console_20260802_113956_directions` 否定了“日志改善等于动转静改善”。
- 相对当前候选 `agent_20000.pt`，DR0 动转静尾段 speed/wxy/yaw/joint/foot 从 `0.01217/0.03727/0.02112/0.08225/0.01512` 变为 `0.01270/0.03817/0.02196/0.08623/0.01593`，五项全部没有改善。移动线跟踪 `0.03718 -> 0.03555`，但 yaw 误差 `0.03362 -> 0.03735`、移动倾角 `1.17 -> 1.53 deg` 退化；除 wxy 外的刹车 p95 也普遍回退。继续无条件等待已没有物理依据。
- 初始/预备站立确实改善：joint `0.0933 -> 0.0804`、foot `0.0168 -> 0.0137`；运动后归零尾段没有同步改善。这证明总体 exact-zero 日志被容易的初始站立样本主导，不能代表真实制动闭合。
- 进一步代码核对发现 stop-cycle 的确定性实现错误：`TailiBlindTPEnv._resample_commands` 先调用父类，父类会原地覆盖 `_cmd_target`，子类随后才从 `_cmd_target` 读取 `was_moving`。所以所谓 `3 s 运动 -> 2 s 精确零命令` 并未按上一条命令严格交替，而是被父类本次随机采样决定；这同时解释了初始/连续站立样本占优和动转静信用不足。
- 历史 `w_stand_capture=1.2` 不能直接恢复。旧实现对所有 stand 样本奖励宽核质量的逐帧增量，没有真实“上一条非零命令 -> 当前精确零命令”条件；历史 7+ 仍不安静站立正是该宽泛语义失败的证据，而不是定向 capture 不可行的证据。
- 当前正式 run `/root/gpufree-data/taili_runs/taili_train_20260802_flat35_dr3_stand_motion_shortfall_resume20k_long` 在新 payload 验证前保持运行。下一候选只修复 stop-cycle 对上一条命令的读取，并增加训练专用 `post_move_zero_gate`：不修改 actor 观测、不做命令平滑、不规定动作序列；普通移动、初始站立、楼梯和外部命令不变。该 gate 使用现有严格 `stand_completion_quality` 追加正向 capture 信用，并单独遥测 gate、条件化 completion 与 strict cost。

## 2026-08-03: post-move capture exposure and MuJoCo deployment

- 新正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260802_flat35_dr3_post_move_capture_resume20k_long`，tmux `rl_train_dr3_post_move_capture`。它从已诊断的 shortfall `agent_20000.pt` 恢复，清 Adam，`lr=1e-5`、2 epochs、clip `0.10`；仅修复 stop-cycle 采样时序，并在真实 flat `nonzero -> exact-zero` 段添加 `post_move_zero_gate` 和 `w_stand_capture=1.2`。移动、楼梯、DR、AMP、PD 和 PPO 其余语义未改。
- 首三个等长窗口 `0.5-4.49k`、`4.5-8.49k`、`8.5-12.49k` 中，capture gate 稳定为约 `0.063`，条件化 completion 为 `0.3525/0.3535/0.3488`，条件化 strict cost 为 `0.7446/0.7430/0.7499`。因此新样本和遥测均已真实生效，但截至该点没有动转静学习趋势；不能再把总体 stand 均值当作证据。同期 F/B/L/Y、duty/slip、tilt/support/height/fall、楼梯 frontier 和 DR3 没有共同退化，继续训练并以 capture 条件指标单独裁决。
- 训练推进到约 `54k` 时，完整 `agent_50000.pt`（`43,607,603` bytes）已导出为部署 TorchScript。训练 actor 与 TorchScript 的 eager/trace 误差均为 `0`；策略 SHA256 `d0319ca270bd93939d95742b158ecd483acde9d00eabf0499a692a2d03175ddb`，元数据 SHA256 `0cdd7ec898b9d760df2afce6fa1da1f938d33efe71706248d5e46e3c7b013f84`。
- 已部署到 MuJoCo 主机 `/root/taili_rl_sar_cpp_0801/policy/taili/taili_current/policy.pt`。现有 SAR `config.yaml` 的 observation history、20 ms policy period、1-step action delay、joint order、PD `[120/100/120]`、KD `5`、action scale `0.35` 与新 metadata 完全一致；MuJoCo 端 `torch.jit.load` 输出 `(1, 12)` 且全有限。替换前策略备份为 `policy.pt.before_20260803_103054_agent50000` 及同名 metadata 备份。

## 2026-08-03：楼梯本质语义候选与课程重建

- MuJoCo 已确认成熟策略的 nominal 20 cm 楼梯并不可靠：上楼不能完整通过，下楼虽能前进但滑移、冲击和姿态质量差。IsaacLab 课程等级曾高并不能替代跨引擎的真实整段验收。
- 已否定上一候选的八项算术/调和质量核。它把方向主驱动再次交给复合质量分数，早期任一短板都可能削弱探索，且不能精确表达最终要求。新候选 `history7_flat35_dr3_nominal_stair_reactive_reset_20260803` 保留成熟 actor/critic、AMP、PD、PPO、平地、站立、命令和 DR，只修改楼梯任务语义。
- 数学主路径固定为：沿机体系命令方向的有符号速度与真实净进展提供持续主驱动；滑移、横漂/yaw、动态核心、机身高度、冲击、过度抬脚、力矩和动作高频分别直接扣分。足端轻碰立面始终付费，但允许随后纠正；hip/thigh/calf/base 的严重接触按 episode 锁存，立即关闭任务信用并使该 episode 不能升级。
- 足端碰撞只建立每腿 `0.80 s` 的线性衰减迹线，后续摆动净空目标从真实失败高度增加 margin；它不产生抬脚正奖励、进展倍增、约束退让、腿序或落脚位置。重复碰撞从新的失败高度继续纠正。
- 课程成功要求 `90%` 总换高、`100%` 楼梯沿程，并在最后一级之后再推进 `0.45 m`，确保后髋和后足进入平台；到达楼梯末端但整机未通过不能升级。固定 replay 设为 `0`。
- 首次正式切换使用一次性环境变量 `TAILI_RESET_STAIR_CURRICULUM=1`，只清零上下楼的 level/peak/down-streak。payload 默认不永久启用重置，因此以后故障恢复不会再次丢失楼梯课程；平地、phase3、DR3 和其它状态始终保留。
- 本地已串行通过楼梯课程语义、核心/严格站立、下楼奖励、command-conditioned AMP、stand gait clock、DR 分层 verifier 及全包 `compileall`。期间修复了 `terrain_hard_valid` 被误计入标量总奖励/多 critic 的问题，并把旧 verifier 中已删除的质量核与 `75% replay` 断言改为当前明确契约。远端部署、恢复源、运行目录和启动实测在完成后继续记录；候选就绪前不停止当前训练。

## 2026-08-03：候选启动修复与正式接管

- 首次从 `agent_62000.pt` 启动候选时，环境初始化、精确课程恢复和重置均成功，但首个 `_get_rewards()` 暴露 `blind_tp_env.py` 中残留的 `_zero_leg_event` 引用，训练在首个环境步退出，没有产生策略更新。该错误是旧质量核删除后的实现残留，不是楼梯策略效果证据；失败 run `taili_train_20260803_nominal_stair_reactive_reset_resume62k_long` 保留为启动故障记录。
- 已将轨迹腿权重初始化改为当前 `_collision_now` 的同形状张量，并新增 verifier 断言旧符号不存在；本地与远端 `verify_stair_course_semantics.py`、`compileall` 均串行通过。修复后最终归档 SHA256 为 `9aa69138b7d792e45a9e3997fa777fe952810b01ce0b7444a77a14bd7dc589f2`。
- 正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260803_nominal_stair_reactive_reset_resume62k_v2_long`，tmux `rl_train_stair_reactive`。从固定配对 snapshot 的 `agent_62000.pt` 恢复，启动实测 checkpoint/optimizer 已加载、phase3/DR3/flat 保持、`stairs/stairs_up=0`、replay=`0`，且只有一个 trainer 进入 PPO；首窗初始 `fall=0`，尚未以早期步数判断楼梯能力。
- 首个监控基线按 `1-2490` 与 `2500-4990` 两个等长半窗比较：下楼末值 `0.266 -> 0.270`，上楼 `0.213 -> 0.261`，上楼 frontier 到 `3`；raw F/B/L/Y、duty 和 slip 均保持或改善，fall 约 `5e-5`。但 episode 加权 course-ok `0.404 -> 0.326`、hard-valid `0.961 -> 0.953`，landing impact 也略差。因此当前裁决仅为“课程未阻塞且平地基本盘未坍缩”，不能宣告楼梯质量改善；继续按独立 10 分钟窗口观察，不修改权重。
- 第二个独立窗口 `5000-9910` 中，下楼由 `0.273` 升到 `0.332`、上楼由 `0.261` 升到 `0.337`，两者 frontier 均到 `3`；episode 加权 course/height/forward/stable-ok 为 `0.364/0.364/0.950/0.953`。hard-valid 均值 `0.950` 且后半略回升，碰撞和冲击没有持续恶化；raw F/B/L/Y=`0.901/0.875/0.849/0.915`、fall=`7.1e-5`，故继续训练。保护性观察项是后半 duty `0.555 -> 0.541` 和 base height `0.537 -> 0.534 m`，尚未达到干预条件。
- 第三个窗口 `10000-14920` 中，下楼 `0.332 -> 0.371`、上楼 `0.337 -> 0.376`，上楼 frontier 到 `4`；hard-valid 后半由 `0.951` 升到 `0.955`，fall 减半且 duty 恢复。未闭合项是 course-ok 仍约 `0.347`、height-ok 约 `0.283`，后半 forward/stable-ok 从约 `0.927` 降到 `0.883`；全局 base height 降到约 `0.530 m`，但该混合均值同时受楼梯等级上升影响，不能直接判为平地回归。完整 `agent_10000.pt` 已保存，单 trainer 正常、磁盘余量约 `15 GiB`；继续下一窗口观察换高/整段成功是否追上课程上涨。
- 第四窗口 `15000-20260` 接近 `20k` 时，下楼 `0.367 -> 0.422`、上楼 `0.385 -> 0.389`，两者 frontier 为 `4`；course/height-ok=`0.380/0.329`，stable/height/contact/wxy-ok 约 `0.923/0.941/0.936/0.939`。后半 hard-valid 回升到 `0.951`、fall 降到 `2.6e-5`，slip=`0.0545`，raw F/B/L/Y=`0.899/0.901/0.859/0.913`。当前裁决仍是课程有潜力但整段完成质量未达标，未触发重启或改参条件。
- 第五窗口 `20300-25560` 出现更强上涨：下楼 `0.414 -> 0.574`、上楼 `0.393 -> 0.498`，frontier 下/上为 `4/6`；course-ok 提高到 `0.415`，raw 四方向、duty/slip 与碰撞成本均保持，hard-valid 后半由 `0.940` 回升到 `0.945`。但 height-ok 随难度在前/后半由 `0.290` 降到 `0.241`，stable-ok 约 `0.89`，说明当前是有效突破而非高质量收敛；继续到 `30k` 判断 frontier 尖峰能否扩散到均值。
- `30k` sidecar 的逐环境分布解释了 frontier/mean 差距：当前 `terrain_replay_fraction=0`，所以 frontier mask 覆盖全部楼梯环境；统计中的 `frontier_max` 只是该集合的最大 level，不是 frontier 子集均值。下楼 256 个环境分布为 `0:166, 1:50, 2:12, 3:22, 4:6`，mean=`0.641`、p50=`0`、p75=`1`、p90=`3`、max=`4`；上楼 460 个环境为 `0:327, 1:85, 2:16, 3:23, 4:4, 5:4, 6:1`，mean=`0.496`、p50=`0`、p75=`1`、p90=`2`、max=`6`。因此 max 只证明少数环境已探索到高等级，不能代表整体能力；后续验收以 mean/分位数/成功安全率为主，frontier 仅作探索上界参考。

## 2026-08-03：当前楼梯平台的原因（不再使用 frontier）

- 当前正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260803_nominal_stair_reactive_reset_resume62k_v2_long`，payload 为 `history7_flat35_dr3_nominal_stair_reactive_reset_20260803`。截至约 `46.9k` step，固定 `5000` step 窗口的真实训练量已经平台：`terrain_progress` 约 `0.606 -> 0.609`，`terrain_direction` 约 `1.933 -> 1.880`，`terrain_body_height` 约 `0.214 -> 0.202`；`terrain_clearance`、`climb`、`terrain_up/down`、`terrain_support_transfer`、`terrain_contact_quality`、`terrain_event_collapse` 全程为 `0`。`course_ok` 约 `0.29-0.34`，`height_ok` 约 `0.21-0.29`，`move_up` 约 `0.07-0.10`。这才是当前能力证据；`frontier` 完全不参与判断。
- 日志中的 `terrain_stairs_mean`/`terrain_stairs_up_mean` 上升只说明课程环境分布在继续变化，不能说明策略整体通过能力上升。当前课程仍存在低等级失败样本，且完整课程验收率没有同步上升；因此不能用课程均值或 frontier 掩盖真实平台。
- 当前奖励确实有楼梯驱动，但驱动目标错位。生效的 `w_terrain_progress=1.5` 和 `w_terrain_direction=4.0` 都主要由 `v_along = dot(v_xy, cmd_xy)/|cmd_xy|` 及其速度比构成；它奖励沿命令方向移动、接近楼梯和持续前进，不奖励“机身相对目标平台完成净换高”。所以策略可以在台阶前保持前进信用，却没有足够理由完成抬腿、承重转换和整段越过。
- 课程验收在 `terrain_curriculum.py` 的 episode 末端才同时检查净高度、完整沿程、横漂/航向、末态高度/姿态/接触/角速度和速度上限。这个布尔验收不是逐步的正向梯度；而当前 `height_ok/course_ok` 很低，说明它没有转化为训练中的连续学习信号。
- 当前盲态奖励接口又主动切断了可部署的地形几何信用：`blind_tp_env.py` 虽然计算扫描得到的前方升高和支撑面，但传给 `RewardInput` 的 `local_obstacle_h` 被置为零，`terrain_response` 也固定为零。因此扫描只剩诊断/部分净空计算用途；`terrain_clearance_drive=0`、`w_terrain_clearance=0`，并且 `rew_climb`、`rew_terrain_up/down`、`rew_terrain_support_transfer`、`rew_terrain_contact_quality`、`rew_terrain_event_collapse` 全为零。
- 这不是扫描器没有看到地形：当前遥测 `terrain_probe_scan_ok=1`，前方升高均值约 `0.023-0.027 m`、窗口最大值约 `0.37-0.58 m`，最近记录的 `capability_step_height_max` 约 `0.222 m`；但对应的 `up_active/down_active/contact_event/support_transfer` 仍全为 `0`。远端 `effective_config.yaml` 也确认了上述零权重和 `terrain_curriculum_height_required_fraction=0.9`、`course_distance_fraction=1.0`。因此问题在“已获得的几何/接触事实没有进入正向学习路径”，不是没有地形数据。
- 结果是“探索成本存在、完成收益缺失”：足端净空不足/过高、碰撞、滑移、冲击、支撑和动作质量仍在扣分，但抬脚、接触后的支撑转移、真实换层或净高度变化没有对应的连续正向信用。策略停在“向前走但不上台阶”是当前数学路径的直接局部最优，不是 frontier 统计问题。
- `terrain_body_height` 也不能补上这个缺口。它比较的是机身相对当前支撑面的高度，并不能区分已经站上下一阶与仍在原台阶前；通过姿态或局部支撑估计也可能拿到部分分数。当前该项只有约 `0.20` 的奖励量，且没有和目标平台净高度绑定。
- 当前证据不支持先改 PD、AMP、状态机或平地主路径：四方向基本进展、`stable_end`/`forward_ok` 仍有信用，生效配置中也没有状态机式逐腿落点机制。首要矛盾是楼梯任务的连续正向语义与课程验收不一致；下一步应在保留方向驱动、平地能力、盲态部署边界和已有安全成本的基础上，只补一条可部署、连续、与真实支撑面净换高相关的主驱动，再用固定楼梯等级的整段诊断验证。

## 2026-08-03：楼梯主驱动的进一步纠偏（远端重启后）

- 远端重启后，`rl_train_stair_reactive` tmux 和 trainer 已不存在；当前 run 的最近完整检查点为 `agent_50000.pt`。在主驱动语义确认前不恢复训练。
- 前一条“补充净换高主驱动”的表述仍然过于局部，容易重新走向接触事件、换层或势能链条。楼梯最小任务不是逐腿跨阶或触发某个事件，而是机器人沿命令指定的楼梯行进方向，持续、安全地把整机推进到目标平台。
- 当前实现与候选文档不一致。文档称“有符号速度与真实净进展”为主驱动；但 `linear_progress`、`terrain_progress` 和 `terrain_direction` 全部只从瞬时平面 `v_along=dot(v_xy,cmd_xy)/|cmd_xy|` 得到，未读取固定楼梯路径上的位置增量、支撑面净变化或终点距离。三项还是同一平面速度的重复信用，不是“真实净进展”。
- 现有 `terrain_direction=tanh(signed_speed/0.35)` 还存在两个明确缺口：它在当前机身坐标系中逐帧重定义方向，yaw 后仍可能把局部前进当成正确方向；并且 `0.35 m/s` 已得到约 `tanh(1)=0.762` 的方向分，无法持续要求此前确认的约 `0.60 m/s` 级别有效推进。仅把该权重加大或另加高度事件都不能修复这两个语义错误。
- 正确的最小主路径就是固定、带符号的楼梯纵向净进展，而不是再增加竖直势能或接触事件：在命令生成/episode reset 时，将机器人视角的前进命令一次性映射为实际楼梯水平纵轴；每个仿真步只奖励机身世界位置在该固定纵轴上的有符号增量。物理楼梯本身保证“不完成换高就不能继续取得纵向位移”，因此上/下楼无需两套驱动，也不需要足端顺序、落脚点、接触阶段、教师动作或 actor 特权地形输入。
- 现有代码其实已经为课程验收计算了所需事实：`_accumulate_curriculum_progress()` 中的 `step_forward_start` 使用 episode 初始 yaw 投影真实 `root_xy` 增量，并累加到 `_curriculum_stair_forward_dist`。当前错误是它只进入末端课程验收，没有进入逐步主奖励；奖励却另行使用可随当前 yaw 改写的机体系瞬时速度。下一实现应直接复用这个已验证的每步纵向位移事实，而不是再造一条地形状态链。
- 防钻空子的顺序明确：使用固定轴而不是随当前 yaw 旋转的轴；使用线性有符号位置差而不是 `max(v,0)`、单帧碰撞或只奖励正向速度，使回退和往复摆动自行抵消；按约 `0.60 m/s` 的持续推进尺度做对称有界归一化，避免 `0.35 m/s` 已接近满分。横漂、航向、滑移、冲击、核心、机身相对支撑高度、过抬脚、力矩和非足端碰撞继续作为独立直接成本或硬安全否决，不再乘到主驱动上。
- 课程的完整沿程、净换高和末态安全仍只做验收/升级。净换高是“是否真正通过”的必要验收，不是另一条逐步主驱动；这样数学上只有“沿真实楼梯方向净前进”这一条主任务路径，安全、稳定和合理步态负责约束它。

## 2026-08-03：固定轴净进展实现

- 新候选为 `history7_flat35_dr3_fixed_course_drive_20260803`，完整复制当前正式 payload 后只改楼梯主驱动。它保留 actor/critic、AMP、PD、PPO、平地、站立、DR、课程验收和所有已有直接安全成本。
- `TailiAmpEnv._accumulate_curriculum_progress()` 已将课程验收使用的 `step_forward_start` 保存为每控制步的 `_curriculum_stair_step_forward`；该值是当前 root 世界坐标增量在 episode 初始 yaw 的固定纵轴上的投影。`RewardInput` 只在前进主导的上下楼命令上将其除以控制周期后传入 `terrain_course_velocity`。
- 楼梯 `terrain_progress/terrain_direction` 现在使用同一净进展事实：`clip(course_velocity / min(command_speed, 0.60), -1, 1)`。旧的 `linear_progress` 在该楼梯作用域显式为零，避免瞬时机体系 `v_along` 继续形成平行的错误信用；平地和非楼梯 terrain 路径不变。
- `0.35 m/s` 的 `tanh` 饱和已移除，默认与 YAML 均为 `0.60`。反向位移直接负分，零净位移为零，局部朝向改变不能再重定义推进方向。横漂、yaw、滑移、冲击、核心、支撑高度、动作、力矩和非足端碰撞仍是独立约束，课程完整沿程/净换高/末态仍只用于验收。
- 本地已串行通过更新后的 `verify_stair_course_semantics.py` 和 `compileall`；verifier 覆盖正向满分、反向负分、零净进展、命令速度上限、固定轴字段链路及楼梯线速度信用隔离。远端部署、一次性 stairs 课程重置和恢复 `agent_50000.pt` 后继续记录。

## 2026-08-04：净空桥接未闭合后恢复历史主驱动

- `taili_train_20260804_clearance_bridge_resume30k_long` 的完整 600 秒窗口（step `7000-9540`）确认：stairs mean `0.473 -> 0.539`、stairs-up mean `0.430 -> 0.449`，但真实 `move_up=0.117 -> 0.113`、`height_ok=0.301 -> 0.301`。`course_ok` 虽从 `0.402` 升至 `0.479`，却没有和上楼完成及净换高联合改善，不能视为楼梯能力恢复。
- 历史 7+ payload `global_stair_dr_descent_quality_20260723_213358` 的有效主桥梁已复核：已发生的足端立面碰撞或真实支撑面变化形成 `terrain_response`；后续摆动净空成功时，`terrain_drive = 1 + 2.7 * terrain_response * clearance_success` 放大连续 terrain progress。它不是状态机、足序、落脚点或势能换层。
- 当前固定轴候选为了剔除旧机体系速度伪进展，把 `terrain_response` 与 `terrain_clearance_drive` 同时置零；更关键的是固定轴 `terrain_course_scope` 分支绕开了 `terrain_drive`。结果保留了局部 `terrain_clearance` 正信用，却切断了“碰障后抬脚成功 -> 继续真实前进”的主路径。
- 新候选 `history7_flat35_dr3_fixed_course_drive_response_amplifier_20260804` 只恢复上述历史 response，并且只把有界倍率 `[1, 3.7]` 作用于当前 episode 初始 yaw 固定轴的有符号净推进。碰撞/支撑变化不进入 actor observation；平地、四方向、站立、DR、课程验收和所有直接安全成本保持不变。碰撞但未净空、净空但未固定轴前进均不能领取额外进展。
- 本地已串行通过 `verify_stair_course_semantics.py`、`verify_core_stand_capture.py`、`verify_terrain_descent_reward.py` 和 `compileall`。部署后必须用完整 10 分钟窗口检查 `move_up / height_ok / course_ok` 是否联合改善；不再看 frontier 作为能力证据。

### Response-amplifier 首个 600 秒窗口

- 新 run `taili_train_20260804_history_drive_response_resume12k_long` 从 clearance-bridge run 的 `agent_12000.pt` 恢复；启动日志确认 `phase=3`、`DR=3`、flat level=3 保留，stairs/stairs_up 从 0 重置，无并行 trainer。
- step `550-3200` 的完整 596.9 秒窗口中，`terrain_response=0.145 -> 0.148`、新增 `terrain_progress_drive=1.222 -> 1.226`，证明历史桥接已实际进入固定轴进展路径；`terrain_progress≈1.08`，高于前候选约 `0.78`，且平地 `progress_fwd≈0.81`、fall=0。
- 但仍不能把奖励改善误称为能力：up mean `0.182 -> 0.293`，而 `move_up=0.138 -> 0.106`、`height_ok=0.288 -> 0.274`、`course_ok=0.528 -> 0.474` 未联合改善；landing impact/slip 仅轻微变差。当前裁决是继续至少一个完整窗口，不改权重；若仍不闭合，则对照历史 7+ 的实际 response/drive 幅度并做固定等级地形物理诊断。

### Response-amplifier 第二个 600 秒窗口

- step `3450-6110` 的完整 598.8 秒窗口出现弱但同向的真实联合改善：stairs-up mean `0.345 -> 0.375`，`move_up=0.107 -> 0.119`，`height_ok=0.272 -> 0.296`，`course_ok=0.443 -> 0.480`。与第一个窗口末的三项回落不同，说明桥接没有只产生局部净空分。
- `terrain_response≈0.147`、`terrain_progress_drive≈1.232`、`terrain_progress≈1.085` 保持稳定，说明历史主驱动实际持续进入固定轴路径；平地 `progress_fwd≈0.81`、fall=0。impact/slip 未持续恶化。
- 当前仍远低于可验收的稳定上楼能力，因此继续长训和完整窗口监控，不调权重、不启动高占用诊断。若后续窗口再次停止联合上涨，则按历史 7+ 的同口径倍率/response 和固定等级物理表现调查，不以单一 stairs mean 或 frontier 判断。

## 2026-08-04：response-amplifier 的 20cm 名义楼梯验收与历史对照

- 当前正式 run `taili_train_20260804_history_drive_response_resume12k_long` 的 `agent_8000.pt` 已完成单环境、DR0、20cm 上/下楼、固定 `0.5 m/s` 前进诊断：`/root/gpufree-data/diag_runs/taili_response_drive_agent8000_stairs20_20260804`。修正早期结论：上楼实际完成约 `1.751/1.800 m` 的高度变化，但以大横漂、失稳和重落脚换来，不能验收为安全通过；下楼只完成约 `1.258/1.800 m`，且诊断中已有 `would_terminate` 帧。
- 该诊断排除了“没有方向驱动”的解释。上楼落脚冲击 p95 约 `2.23 m/s`、支撑滑移 p95 约 `0.68 m/s`、pitch p95 约 `31.1 deg`；下楼落脚冲击 p95 约 `1.10 m/s`、支撑滑移 p95 约 `0.50 m/s`、pitch p95 约 `32.1 deg`。问题是方向推进没有闭合为稳定承重、低滑移和安全姿态，不能由课程均值或 frontier 掩盖。
- 已确认的历史 7+ 主驱动不是状态机，也不是指定腿序或落脚点，而是 `已发生碰障/支撑面变化 -> terrain_response -> clearance_success -> terrain_drive = 1 + 2.7 * response * clearance_success -> 连续 terrain progress`。当前 payload 已仅恢复此桥梁到固定初始 yaw 的有符号净推进上；现阶段仍失败意味着须比较“为何历史能把同一桥梁转化为足够净空和承重”，而不能继续盲目增大方向奖励。
- 已启动历史 `taili_train_20260723_214211_descent_quality_resume/checkpoints/agent_30000.pt` 的同一案例表诊断，输出目标为 `/root/gpufree-data/diag_runs/taili_history7_agent30000_stairs20_20260804`。历史诊断必须完成后，才根据净换高、冲击、滑移、支撑、横漂/yaw 和完成情况定位当前缺失的奖励语义或配置差异；在此之前不新增状态机制、不重建训练。

### 历史同条件诊断完成后的判定

- 同一 20cm、DR0、`0.5 m/s` 诊断中，历史 `agent_30000.pt` 上楼完成约 `1.717/1.800 m` 换高、下楼完成约 `1.842/1.800 m`；对应 pitch p95 为约 `12.0/19.6 deg`、支撑滑移 p95 为约 `0.39/0.41 m/s`。当前上/下楼对应为 `31.1/32.1 deg` 和 `0.69/0.51 m/s`。因此历史并非无碰撞的完美策略，但它在同一几何下明显更接近安全、稳定通过，尤其是下楼。
- 当前上楼的 response 桥梁实际比历史更活跃（诊断遥测 `terrain_response` 均值约 `0.534` 对 `0.410`，`terrain_progress` 约 `2.669` 对 `2.300`），却产生更差姿态；这排除了“再加强主驱动即可解决”的假设。历史的额外约束是有下限的软质量耦合：`terrain_direction_quality = 0.65 + 0.35 * support_quality * slip_gate * tracking_motion_gate`，并只把高台阶额外 capability bonus 与该质量、触地洁净度和动作洁净度耦合。它不会把探索奖励归零。
- 当前下楼的主要阻塞已定位：`terrain_hard_valid` 在约第 `100` 诊断步被一次严重非足端碰撞置为 `0` 后锁存到 episode 结束，致使 `terrain_task_gate` 均值仅约 `0.117`、后续 direction/progress 全为零；该碰撞帧之后即时 `body_collision` 成本已回到零，但主驱动仍被永久切断。历史没有此 episode 锁存。正确修复是保留即时非足端碰撞成本及终止否决，但将 hard-valid 变为当前帧语义，避免恢复路径被切断。
- 下一候选只恢复上述历史软耦合与可恢复的碰撞语义；固定初始 yaw 的净推进、`0.60 m/s` 方向尺度、response-clearance 桥梁、现有直接安全成本、平地、DR、PD、actor 和课程均保持。该候选应从当前有效 checkpoint resume，不 fresh、不重置课程，先以同一 20cm 诊断验收“上楼质量下降、下楼完成恢复”而非仅观察课程均值。
## 2026-08-04: 恢复历史机体系楼梯主驱动

- 旧 run `taili_train_20260804_history_quality_recovery_resume20k_long` 在两个完整监控窗口未恢复楼梯：`stairs_up` 约从 `0.713` 到 `0.701`，`move_up` 从 `0.140` 到 `0.125`，`course_ok` 从 `0.750` 到 `0.508`。同协议 20 cm / DR0 诊断 `taili_quality_recovery_agent6000_stairs20_cases_20260804` 中，上楼仅约 `0.21-0.33 m` 净前移、失去约 `0.20-0.25 m` 高度，3 秒后机体系前进速度与固定轴速度均接近零或转负；下楼也未完成且不安全。
- 历史 7+ 对照 `taili_history7_agent30000_stairs20_20260804` 同协议上楼完成约 `1.717/1.800 m` 换高、下楼约 `1.842/1.800 m`；上楼全过程机体系 `v_x` 中位数约 `0.30-0.48 m/s`。这证明问题不是几何不可达，也不是需要状态机、逐腿落点、教师或特权地形输入。
- 已定位的结构差分：历史主路径始终以机器人坐标系命令投影 `v_along` 计算 `linear_progress + terrain_progress + terrain_direction`。当前 fixed-course 分支把楼梯上的三个项改为 reset 固定世界轴净位移，并将 `linear_progress` 在楼梯上置零。固定轴适合整段几何验收，但不应取代机器人视角的连续命令驱动；这与当前上楼早期失去推进梯度相符。
- 新 payload：`/root/gpufree-data/training_payloads/history7_flat35_dr3_body_command_drive_history_recovery_20260804`。只恢复历史机体系连续方向/进展主路径，保留 `0.60 m/s` 方向尺度、response-clearance bridge、软质量耦合、高台阶 capability bonus、现有直接安全成本和当前帧严重非足端碰撞否决。固定初始航向净位移、横漂、航向、净换高和末态继续只做课程验收。无 fresh、无课程重置、无状态机。
- 远端已通过 `verify_stair_course_semantics.py`、`verify_terrain_descent_reward.py`、`verify_core_stand_capture.py`、`verify_command_conditioned_amp_style.py` 和 `compileall`。
- 正式 run：`/root/gpufree-data/taili_runs/taili_train_20260804_body_command_drive_history_resume16k_long`，从旧 run 的 `agent_16000.pt` resume。启动日志确认 phase3、DR3、平地等级和 stairs/stairs_up `3-6` 保留，replay `0`，单一 trainer 正常运行。下一个判断点应看 `stairs_up/move_up/height_ok/course_ok` 的联合趋势，而非 frontier；有趋势后再重复同协议 20 cm 物理诊断。

### 首次物理交叉验证

- `agent_6000.pt` 的同协议诊断：`/root/gpufree-data/diag_runs/taili_body_command_drive_agent6000_stairs20_20260804`。上楼实际完成 `+1.806 m` 支撑面换高（目标 `+1.800 m`）、前进约 `4.584 m`，证明机体系主驱动恢复了历史可达的通过路径；此前固定轴主驱动版本在相同协议中只前移约 `0.21-0.33 m` 并失去高度。
- 不能验收质量：上楼 `pitch p95=13.31 deg`、支撑滑移 `p95=0.517 m/s`、触地 `|vz| p95=2.184 m/s`、最大 torque utilization `0.991`。下楼只完成约 `-1.323 m`（目标 `-1.800 m`），`pitch p95=31.58 deg`、支撑滑移 `p95=0.339 m/s`、触地 `|vz| p95=0.976 m/s`。报告中的 down `terrain_request_execution_mismatch` 仍是 `stairs_down -> stairs` 类型别名，不是几何没有生成下楼。
- 结论：当前修改解决的是“方向主驱动被错误替代后上楼无法起步”的根因，不是楼梯质量的最终解。下一步先继续一个完整训练窗口，检查现有滑移、冲击、支撑、核心与过度净空成本是否能在已恢复通过路径上自然收敛；若它们不与真实通过质量联合改善，再根据这份诊断和历史质量差分只调整相应直接约束，不再重改主驱动或引入状态机。

### 2026-08-04：机体系主驱动恢复后的 10k 物理复核

- 正式 run `taili_train_20260804_body_command_drive_history_resume16k_long` 保持唯一 trainer、`phase=3`、`DR=3` 和原课程；`agent_10000.pt` 已完成同协议单环境、DR0、20cm、level 7、0.5m/s 前进的上下楼诊断，输出在 `/root/gpufree-data/diag_runs/taili_body_command_drive_agent10000_stairs20_20260804`。
- 上楼净支撑面换高约 `+1.682/1.800 m`，说明历史的机体系连续 `v_along -> linear_progress + terrain_progress + terrain_direction` 主驱动仍然生效，且不需要状态机、足序、固定落脚点或特权地形输入。但质量仍不能验收：pitch p95 约 `15.0 deg`、支撑滑移 p95 约 `0.455 m/s`、触地 `|vz|` p95 约 `1.409 m/s`、最大力矩利用率约 `0.961`。
- 下楼仍是当前明确缺口：净换高仅约 `-1.326/1.800 m`，pitch p95 约 `32.35 deg`、支撑滑移 p95 约 `0.352 m/s`、触地 `|vz|` p95 约 `0.986 m/s`、最大力矩利用率约 `0.976`。与历史 `agent_30000.pt` 同协议的约 `-1.842m` 和 pitch p95 约 `19.6deg` 相比，主任务路径已经恢复，但下楼承重、姿态和完成度没有恢复。
- 训练 `6k -> 13k` 的课程均值约 `1.334 -> 1.455`、up mean 约 `0.921 -> 0.983`；`move_up/height_ok/course_ok` 是小样本课程指标，仍有波动，不能替代上述物理验收。当前直接安全成本尚未与真实完成度共同明显改善，因此先运行完整十分钟窗口；只有随后质量仍停滞，才对当前滑移、触地冲击、下楼俯仰和承重对应的直接约束做最小调整，保留已恢复的主驱动。

## 2026-08-04：历史主驱动复核后的高阶楼梯复习恢复

- `taili_train_20260804_body_command_history_saturation35_resume20k_long` 已恢复历史的连续机体系主驱动语义：有符号 `v_along` 进入 `linear_progress + terrain_progress + terrain_direction`，方向信用为 `tanh(signed_speed / 0.35)`；`0.35` 是信用饱和尺度，不是速度上限。固定协议诊断表明它恢复了部分通过路径，但没有恢复历史 7+ 的下楼完成度和姿态质量，不能继续用“加强主驱动”解释当前缺口。
- 该 run 从 step `2000` 到 `11200` 的课程分布没有形成联合上升：`stairs_mean` 约 `1.711 -> 1.758`、`stairs_up_mean` 约 `1.156 -> 1.095`。这不是短暂 frontier 波动；已保存完整 `agent_10000.pt`，因此不再无依据地等待。
- 与历史 7+ payload 的可验证课程差分是：历史把下楼和上楼环境各自 `50%` 确定性固定复习于 level `5-8`，当前为 `0%` replay，虽然当前验收正是高等级。该机制不修改 actor 输入、奖励主驱动、平地、DR、PD 或安全成本；每类楼梯余下 `50%` 环境继续参加前沿课程。
- 新 payload：`/root/gpufree-data/training_payloads/history7_flat35_dr3_body_command_history_replay50_20260804`。只改 `terrain_replay_fraction=0.50`、`terrain_replay_level_min=5`、`terrain_replay_level_max=8`，并同步更新相应验证断言。已通过 `compileall`、`verify_stair_course_semantics.py`、`verify_core_stand_capture.py` 与 `verify_dr2_single_factor.py`。
- 正式 run：`/root/gpufree-data/taili_runs/taili_train_20260804_history_replay50_resume10k_long`，从上述 `agent_10000.pt` resume，保留 `phase=3`、DR3 和现有课程状态，不 fresh、不重置楼梯课程。启动日志确认 replay 为 `down=512`、`up=922`、levels `5-8`，余下 `frontier=1433`；旧 worker 已按精确 PID 清除，GPU 上只有此一个 trainer。
- 首个十分钟窗口到 step `2300`：平地 level=3、DR3、四方向进展、步态/接触/滑移没有立即恶化，且没有错误或额外 trainer。`stairs_mean≈4.22`、`stairs_up_mean≈3.85` 主要反映固定 replay 的样本分布，不能作为能力提升证据；后续仍以 `move_up/height_ok/course_ok` 的联合趋势和同协议 20cm / DR0 物理诊断验收。

### Replay 6k 固定协议复核

- `agent_6000.pt` 的诊断输出：`/root/gpufree-data/diag_runs/taili_history_replay50_agent6000_stairs20_20260804`。协议仍为单环境、DR0、20cm、level 7、前进 `0.5m/s`、上下楼各 `12s`；诊断时显式 `TAILI_DIAGNOSTIC=1`，因此 fixed replay 不参与诊断几何。
- 上楼实际支撑面换高为 `+1.573/1.800m`，root 高度变化为 `+1.503m`，并持续前进约 `5.15m`。这说明机体系主驱动与高阶样本暴露确实产生通过路径，但质量仍不合格：pitch p95 `14.25deg`、支撑滑移 p95 `0.390m/s`、触地 `|vz|` p95 `1.809m/s`，有 `13` 个 hard-impact bout、`25` 个 high-slip bout。
- 下楼支撑面换高仅 `-1.419/1.800m`，root 高度变化 `-1.520m`，持续前进约 `5.46m`；在 forward 段 `3.22s` 已出现重大事件，记录到高度掉落峰值约 `0.309m`、最大 pitch `35.36deg`。全段 pitch p95 `33.99deg`、支撑滑移 p95 `0.283m/s`、触地 `|vz|` p95 `1.035m/s`，不能验收为安全下楼。
- 结论：replay 在 6k 时对上楼通过幅度有一定帮助，但没有形成上/下楼与安全质量的联合恢复，尤其下楼仍失败。历史连续 `v_along` 主驱动已被保留，不能再把“缺乏方向驱动”当作当前根因；先继续长训到下一完整窗口，观察该曝光机制是否有持续学习效应，再仅依据当前物理诊断定位质量耦合或课程中的实际缺口。

## 2026-08-04：历史方向主驱动的质量闭合

- 当前 `taili_train_20260804_history_replay50_resume10k_long` 在完整窗口内保持 `stairs_mean≈4.28`、`stairs_up_mean≈3.87`，没有联合上升；同协议 DR0/20cm 诊断仍可完成大部分几何换层，但上楼触地冲击、下楼 pitch 和承重质量不可验收。不能用 replay 均值或 frontier 替代物理验收。
- 已核对历史有效主驱动仍在：机器人坐标系的有符号 `v_along` 同时进入 `linear_progress + terrain_progress + terrain_direction`，权重保持 `1.5 + 1.5 + 4.0`，以及 `terrain_response -> clearance_success -> terrain_drive` 桥梁。问题不是方向主驱动缺失，不能再增加方向权重、固定落脚点或状态机。
- 实际漏洞：`terrain_task_quality_floor=1.0` 令地形质量门恒为 1；`terrain_direction` 只乘 `terrain_hard_valid`，绕开了 `terrain_task_gate`。失稳、少支撑或高滑移时仍可领取接近完整的方向主奖励，形成激烈冲阶捷径。
- 新 payload：`/root/gpufree-data/training_payloads/history7_flat35_dr3_body_command_replay50_qualitygate_20260804`。仅将 `terrain_task_quality_floor` 改为 `0.65`，并让 `terrain_direction` 改乘 `terrain_task_gate`（其中仍包含 hard-valid）。主驱动、方向尺度、课程、replay、平地、DR、AMP、PD、PPO 和直接安全成本均不变。质量为零时方向项仍保留 `0.65 * 0.65 = 0.4225` 的连续探索信用，质量达标时为完整信用。
- 本地定点编译、奖励数学检查、`verify_stair_course_semantics.py` 与 YAML 到 `RewardConfig` 转发均通过；新 payload 在远端 Isaac 环境重复通过语义验证。正式 run：`taili_train_20260804_history_qualitygate_resume22k_long`，从旧 run 的完整 `agent_22000.pt` resume，恢复精确课程状态 `step=22000`、phase3、DR3 和 replay `50%/levels 5-8`，未 fresh、未重置课程，且只有一个 trainer。
- 启动后首个遥测的 `terrain_task_gate≈0.89`、`terrain_direction≈1.70`，对比旧 run 约 `0.99/2.10` 表明新耦合实际生效；四方向基础进展和 DR 状态未立即坍缩。后续按完整十分钟窗口观察平地是否保持，以及 `move_up/height_ok/course_ok` 与滑移、触地冲击、pitch 是否联合改善；达到稳定趋势后再做同协议 DR0/20cm 上下楼诊断。

## 2026-08-04：奥卡姆名义楼梯质量候选（尚未部署）

- 对 `history_qualitygate` 的复核结论是：把支撑、滑移、跟踪和净空组成复合门，再将其乘入主方向驱动，虽然能降低激烈冲阶的得分，却重新制造了探索死锁。楼梯主任务仍应只有机体系命令方向的连续真实推进；安全、稳定和合理步态必须各自直接付费，而不是先成为获得方向信用的前置条件。
- 新候选位于本地 `.codex_staging/occam_nominal_stair_quality_20260804/history7_nominal_stair_quality_occam_20260804`。删除 `terrain_task_quality_floor`、`terrain_task_quality_gate` 和复合 `terrain_drive_quality`；恢复 `terrain_drive = 1 + terrain_drive_bonus`，同时保留历史 `terrain_tracking_drive_gate * terrain_hard_valid`、`terrain_direction_quality_floor=0.65`、高阶 capability bonus，以及现有碰撞、滑移、冲击、核心、yaw、轨迹、净空和动作直接成本。没有状态机、腿序、落脚点、教师或 actor 地形特权输入。
- 碰障响应缩短为 `terrain_collision_trace_s=0.18`。它只在当前步态周期内提示碰障腿提高净空，不跨多个周期维持抬腿，也不要求第一次触碰就通过。碰撞后继续向命令方向推进才有主收益；没有净空或没有推进都不能靠事件本身得分。
- 楼梯训练改为固定 nominal 高度覆盖：上下楼 `terrain_replay_fraction=1.0`、levels `4-8`，所有 stair DR mixture 均为 `[1,0,0,0]`。这会有意令 frontier 环境数为 `0`，因此课程 move-up/frontier 不再有能力解释力；必须看真实 direction、clearance、非足端碰撞、滑移、冲击、核心、yaw、hip、gait，并最终用 DR0/20cm 单环境诊断验收。
- 平地不随楼梯阶段承受高 DR：flat level 2/3 均为 `[0.80,0.20,0,0]`。平地全局要求继续保留；髋偏差成本不再乘 `terrain_pattern_scale`，因此楼梯碰障或抬腿时也不能放松外翻/内翻。20cm 台阶允许腿在矢状面正常屈伸，但不需要明显髋侧摆，更不规定逐关节轨迹。
- 本地 `compileall`、`verify_stair_course_semantics.py`、`verify_core_stand_capture.py`、`verify_dr2_single_factor.py` 已串行通过。尚未上传或启动。部署时必须重新确认远端当前唯一 trainer、最新完整 checkpoint 和磁盘空间，再精确停止旧 run 并 resume；不 fresh、不重置课程。启动后核对 stair replay `100%/levels 4-8/frontier 0`、stairs `100% nominal`、flat `80% nominal + 20% DR1` 和唯一 trainer，然后进入十分钟窗口监控。

### 部署与首个十分钟窗口

- 本地归档 `history7_nominal_stair_quality_occam_20260804.tar.gz` 的 SHA256 为 `fddd868792ee9d9d8820e32418d1626928bd51940999cb5a7d2bbe55ad8f3413`；远端哈希一致，payload 为 `/root/gpufree-data/training_payloads/history7_nominal_stair_quality_occam_20260804`。远端 `compileall` 和三个专项 verifier 均再次串行通过。
- 切换前确认旧正式 run `taili_train_20260804_history_qualitygate_resume22k_long` 只有一个 trainer，最新完整 `agent_24000.pt` 大小稳定且已用 CPU 成功反序列化出 policy/value/optimizer。旧进程树经精确 PID `SIGTERM` 后全部退出，GPU 无残留。
- 新正式 run 为 `/root/gpufree-data/taili_runs/taili_train_20260804_occam_nominal_stair_quality_resume24k_long`，从上述 `agent_24000.pt` resume；课程精确恢复 `step=24000`，未 fresh、未设置课程重置。启动日志确认 stair replay `100%`、levels `4-8`、frontier `0`；flat 人口 `0.8/0.2/0/0`，stairs-up/down 均为 `1/0/0/0`，且只有一个 trainer。
- 启动日志中的 `mixed push active` 不污染 nominal 楼梯：代码触发条件显式要求环境 `_dr_env_level` 为 `1-3` 且 push factor 激活。首窗遥测也确认 stair-up/down push active fraction 恒为 `0`，仅极少 flat DR1 环境触发。
- 首个完整约 `600s` 窗口为新 step `690 -> 3320`。平地 raw F/B/L/Y 约保持 `0.932/0.928/0.900/0.950`，fall 约 `0.0004`，无整体坍缩。terrain direction/progress 基本稳定；clearance 约 `0.128 -> 0.133`、impact 成本约 `-0.499 -> -0.495`，但 stance-slip 成本约 `-0.260 -> -0.282`，不能宣称质量改善。
- 因 `100% replay`，stairs mean 固定约 `5.998`，frontier 为 `0`，并且当前课程 move/height/course 完成率字段也恒为 `0`；这些均无能力解释力。当前裁决是不修改、不重启，继续第二个十分钟窗口，观察 direction/clearance/impact/slip/core/yaw/hip/gait 是否形成联合趋势。达到完整 checkpoint 后再用单环境 DR0/20cm 上下楼诊断验收真实通过与质量。

### 2026-08-04：外部诊断接入系统

- 直接运行的诊断不会自动出现在系统历史：系统页面只通过 `/diagnostics/run` 创建的 `job_id` 和本地 `diagnostics/index.json` 建立 artifact 映射；远端目录中即使已有完整 `record.csv`、`metrics/metrics.json` 和回放数据，未注册的目录也不会被页面扫描。
- 当前名义楼梯质量诊断已登记为 `20260804_occam_nominal_stairs20_agent6000`，远端输出为 `/root/gpufree-data/diag_runs/taili_occam_agent6000_stairs20_20260804`。该登记只增加系统历史索引，不改变训练进程或远端诊断文件；报告和回放接口已验证可读（回放 `625` 帧）。
- 后续记录使用有语义的名称/时间戳，不使用 `v1`、`v2` 这类无语义序号。外部诊断若要在系统中查看，必须先通过系统任务启动，或登记其 `job_id`、预设、检查点和输出目录。

## 2026-08-04：MuJoCo C++ 部署控制契约核对

- 纠正一条此前不精确的表述：当前正式 run `taili_train_20260804_occam_nominal_stair_quality_tune_resume10k_long` 的有效配置并非统一 `Kp=120, Kd=10`。其 `effective_config.yaml` 为 hip/thigh/calf `Kp=120/100/120`、全关节 `Kd=5`，控制周期 `0.005 * 4 = 0.020 s`，动作延迟为一个完整策略步。
- MuJoCo 远端 `/root/rl_sar/policy/taili/taili_current/policy.pt.json` 明确绑定上述 run 的 `agent_22000.pt`；同目录 `config.yaml` 的 `rl_kp`、`rl_kd`、动作尺度、关节顺序、限位和一帧动作延迟均与该 checkpoint 导出的控制元数据逐项一致。C++ `rl_sim_mujoco` 在 Taili 路径读取的正是这些 `rl_kp/rl_kd`，按 `tau = Kp * (q_des - q) - Kd * dq` 和相同的力矩-速度包络写入 MuJoCo 控制量。
- MuJoCo 场景物理步长为 `0.0025 s`，C++ 用 8 个物理子步维持 `20 ms` 的策略周期；IsaacLab 训练为 `0.005 s * 4`，策略周期也为 `20 ms`。两侧均为新动作写入后延迟一个完整策略步才实际施加，初始动作历史均为零。
- 辅助 `scripts/taili_sim2sim.py` 的 `260/120/220`、`8/5/5` 是旧 `taili_tp_0801` 的独立回归脚本配置，当前统一 C++ runner 不调用它。`0` 键起身阶段确实使用 base.yaml 的固定 `260/120/220`，但切到 `1` 的 RL locomotion 时 `InitRL(taili_current)` 重置策略状态并改用 `rl_kp/rl_kd`，固定起身增益不残留到策略运行。
- 因而对于用户在统一 C++ runner 中观察到的 `agent_22000.pt` 上楼弹跳式换层、下楼接触后高频腿振荡，PD 数值、控制频率和动作延迟的不一致已被排除为首要原因。仍需保留一般 sim2sim 差异（接触求解、摩擦、楼梯 MJCF 几何）作为次级验证项，但 IsaacLab `agent_20000.pt` 已有低姿态、重触地、滑移和大俯仰证据，训练策略/楼梯奖励语义仍是主调查对象。

## 2026-08-04：`agent_36000.pt` MuJoCo 迁移与真实运行路径确认

- 已从正式训练 `taili_train_20260804_occam_nominal_stair_quality_tune_resume10k_long` 的完整 `agent_36000.pt` 导出部署包。源 checkpoint SHA256 为 `4133cebc4b037ae354342d9feb6ae742c46c49372c37d6b340079f3da1a60942`；TorchScript SHA256 为 `612d921f7f41dd5cafbf222d92ad08a3f1ba46c52414bf55fe1a242e50bbe65c`。训练端 wrapper 与 SKRL actor 的误差以及 TorchScript 与 wrapper 的误差均为 `0`，输出形状为 `(1, 12)` 且全有限。
- 用户实际运行的 `/usr/local/bin/rl_sim_mujoco taili [scene]` 已确认链接到 `/root/rl_sar/cmake_build/bin/rl_sim_mujoco`。`[scene]` 只决定 `/root/rl_sar/src/rl_sar_zoo/taili_description/mjcf/[scene].xml`，不决定策略或控制配置。
- Taili FSM 按 `1` 进入 RL locomotion 时，默认以 `taili_current` 调用 `InitRL("taili/taili_current")`；仅当环境变量 `TAILI_POLICY_CONFIG` 非空时才覆盖该目录。`InitRL` 精确读取 `/root/rl_sar/policy/taili/taili_current/config.yaml` 内的 `taili/taili_current` 节，并从同目录加载 `model_name: policy.pt`。当前登录 shell 的 `TAILI_POLICY_CONFIG` 为空。
- 已在没有运行中的 `rl_sim_mujoco` 时原子替换该真实路径下的 `policy.pt` 与 `policy.pt.json`；`config.yaml` 未修改。部署前后逐项核对 1403 维观测、历史顺序、20 ms 策略周期、一步动作延迟、步态参数、动作上下界、动作尺度 `0.35`、`Kp=[120,100,120]`、全关节 `Kd=5`、限位和默认关节位，均与 `agent_36000` 的导出契约一致。
- 旧 `agent_22000` 部署已保留为 `/root/rl_sar/policy/taili/taili_current/policy.pt.before_20260804_151410_agent_22000` 及同名 metadata 备份。此时未启动 MuJoCo runner，未停止或修改训练端唯一 trainer。

### 同一 `agent_36000.pt` 的双后端现象复核与后续调查边界

- 用户已在 IsaacLab 与统一 C++ MuJoCo runner 中观察同一 `agent_36000.pt`。两端都尚未满足楼梯质量要求，但具体表现不同：IsaacLab 重点是低姿态、非必要髋部运动、重触地、滑移与较大俯仰；MuJoCo 则更明显地出现上楼弹跳和下楼承重接触后的高频腿部振荡。这个交叉结果只排除了部署目录、PD、策略周期、一步动作延迟和 MuJoCo 特有控制器误配作为唯一首要原因；接触求解、摩擦和台阶几何仍是需要独立验收的 sim2sim 差异。
- `agent_36000` 的 IsaacLab 楼梯诊断已完成于 `/root/gpufree-data/diag_runs/locomotion_console_20260804_064801_terrain`。协议为单环境、强制 DR0、20 cm 上/下楼、初始 stand `0.8 s` 后前进 `0.5 m/s` 持续 `10.2 s`；诊断仅记录观测，不把任何指标伪装成通过/失败标签。当前训练未因这次诊断或迁移停止，仍只有一个 trainer，已继续生成 `agent_38000.pt`、`agent_40000.pt` 和 `agent_42000.pt`。
- 当前 payload 是 `/root/gpufree-data/training_payloads/history7_nominal_stair_quality_occam_tune_20260804`。初步代码核对显示连续机体系方向主驱动仍存在，`w_terrain_direction=4.0`，并带 `terrain_drive = 1 + terrain_drive_bonus`；同时 `terrain_direction_quality_floor=0.65` 令物理质量很差时方向信用仍保留正的下限。这是当前需要验证的候选漏洞，不是已经确认的结论。直接质量成本同时存在，但作用是否被质量门、terrain pattern/style 缩放、支持高度 relief、pitch 容差或事件掩码削弱，必须读清当前 reward 的完整数学路径后再调整。
- 后续优化边界不变：保留历史已证明有效的连续机体系方向主驱动和碰障后净空响应；不把问题改造成状态机、固定落脚点、固定腿序、逐腿 trace、教师策略或 fresh 重造。先以 `agent_36000` 的双后端共同失败项为依据，查明低姿态、弹跳上楼、下楼接触后振荡为何仍是高收益或低代价路径；仅在证实语义漏洞后对直接质量约束做最小且可验证的协调调整。

### 2026-08-04：双后端评估边界修正

- 不能再把 IsaacLab 与 MuJoCo 的实际表现表述为“本质相同”。同一 checkpoint 可能在两端都不满足楼梯质量要求，但具体的姿态、接触瞬态、振荡、滑移和控制恢复可以不同；MuJoCo 的台阶几何、接触求解和摩擦会叠加独立的 sim2sim 差异。
- 因果顺序必须固定：先用 IsaacLab 同协议诊断判断训练奖励是否把策略推向正确的运动学路径；MuJoCo 作为独立部署验收，用于发现迁移缺口。不能把 MuJoCo 的单独现象直接归因到训练奖励，也不能用 IsaacLab 的单独改善宣称 MuJoCo 已通过。
- 后续任何楼梯候选均应按同一 checkpoint 做成对验证，但分别报告两端结果：IsaacLab 侧看净换层、机身姿态、支撑滑移、触地冲击、髋部/轨迹与核心；MuJoCo 侧看真实场景中的通过、碰撞、弹跳、承重后振荡与恢复。只有共同改善才可作为迁移方向的正证据；若仅一端改善，优先定位该端特有差异，不盲目改写主奖励。
- 当前训练不因这条记录停止或重启。
