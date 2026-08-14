# Taili 训练遥测字段契约

更新时间：2026-07-16

本文档记录 `TailiBlindTPEnv._get_rewards` 写入 `TrainingTelemetryEmitter` 的结构化字段。字段构造集中在 `autotuner/blind_locomotion/telemetry_payloads.py`，`_get_rewards` 负责计算上下文并调用 builder。它是前端、智能体和人工调参共同依赖的观测契约。

## 总体原则

遥测分为四个 payload：

- `reward`：奖励分量、地形探针、配置快照和基础训练得分。
- `command`：命令、实际速度、方向 progress、过渡态和步态质量。
- `curriculum`：phase、terrain、DR、门控、地形课程推进和阻塞原因。
- `health`：稳定性、高度、姿态、接触、力矩和终止风险。

字段稳定性分三级：

- `稳定字段`：前端和智能体应优先依赖，不能随意改名。
- `派生字段`：可用于解释和诊断，改名前需要同步前端/智能体。
- `调试字段`：用于临时排查，但仍应避免无记录删除。

## reward payload

稳定字段：

- `total`：当前标量总奖励均值。
- `lin_err`：线速度命令误差均值。
- `speed`：机身平面速度均值。
- `gait`：稳定运动门控均值。
- `base_h`：相对局部地面的 base 高度均值。
- `upright`：直立程度均值。

动态透传字段：

- 所有 `comp` 奖励分量会以原名进入 `reward`，例如 `tracking_lin`、`tracking_yaw`、`supported_progress`、`terrain_progress`、`stance_slip`、`landing_impact`、`terrain_up`、`terrain_down`、`terrain_event_collapse`。
- `supported_progress` 表示所有线性命令下的可靠支撑推进；`terrain_progress` 只有接触后地形响应激活时才非零。
- 所有 `terrain_probe` 字段会以 `terrain_probe_<name>` 进入 `reward`，例如 `terrain_probe_rise_ahead_mean`、`terrain_probe_up_active_mean`。
- 首次发射时，奖励配置会以 `cfg_<name>` 进入 `reward`，用于确认实际权重。

用途：

- 判断主导奖励和惩罚。
- 确认速度跟踪是否被接触质量、滑移、脚重或地形事件抵消。
- 检查地形奖励是否真实激活，而不是只看地形等级。

## command payload

稳定字段：

- `cmd_vx`
- `cmd_vy`
- `cmd_wz`
- `actual_vx`
- `actual_vy`
- `actual_wz`
- `v_along`
- `speed_xy`
- `lin_err`
- `yaw_err`

方向 progress 字段：

- `progress_fwd`
- `progress_back`
- `progress_lat`
- `progress_yaw`
- `raw_progress_fwd`
- `raw_progress_back`
- `raw_progress_lat`
- `raw_progress_yaw`
- `progress_min_active`
- `progress_validity`
- `progress_lagging_dir`

方向审计字段：

- `direction_<dir>_target_samples`
- `direction_<dir>_applied_samples`
- `direction_<dir>_flat_samples`
- `direction_<dir>_steady_samples`
- `direction_<dir>_eval_samples`
- `direction_<dir>_terminal_rate`
- `direction_<dir>_posture_gate`
- `direction_<dir>_support_gate`
- `direction_<dir>_motion_gate`
- `quality_capability_gate`

这些字段只用于详细排查，不建议全部放入主监控界面。它们用于定位命令样本在哪一层消失，以及 raw progress 被哪类物理有效性压低。

过渡态字段：

- `transition_active_frac`
- `transition_strength`
- `transition_zero_frac`

步态质量字段：

- `gait_match`
- `diagonal_contact`
- `diagonal_pair_instant`
- `duty_balance`
- `duty_balance_instant`
- `duty_spread_window`
- `duty_target_score`
- `duty_symmetry_score`
- `stance_slip`
- `stance_slip_instant`
- `stance_slip_high_fraction`
- `duty_fl`
- `duty_fr`
- `duty_rl`
- `duty_rr`
- `duty_cycle_valid_frac`
- `actual_contact_period`
- `contact_period_valid_frac`

Duty 只有覆盖约一个目标周期后才有效；周期由同一只脚连续 touchdown 的间隔估计。无有效窗口时不得把初始化的 `0.5` 解读为理想 duty。

平地物理尾部字段包括 `flat_touchdown_vz_p95`、`flat_touchdown_force_rate_p95`、`flat_ang_accel_p95` 和 `flat_trajectory_worst_p95`。其中接触力上升率按体重归一化，并取 touchdown 后短窗口内峰值。

用途：

- 判断方向能力是否均衡。
- 区分 raw progress 高但姿态无效的问题。
- 判断 yaw、横移、后退到底是命令不足、速度不足、过渡不自然，还是步态质量不足。

## curriculum payload

稳定字段：

- `phase`
- `command_mode`
- `dr_level`
- `phase_count`
- `penalty_gate`
- `budget_ratio`
- `clearance_gate`
- `terrain_health_ok`

progress 和阻塞字段：

- `progress_gate`
- `progress_fwd`
- `progress_back`
- `progress_lat`
- `progress_yaw`
- `raw_progress_fwd`
- `raw_progress_back`
- `raw_progress_lat`
- `raw_progress_yaw`
- `progress_validity`
- `progress_lagging_dir`
- `active_dirs`
- `blocked_by`
- `next_gate`

地形字段：

- `terrain_mean`
- `terrain_max`
- `terrain_flat_mean`
- `terrain_flat_max`
- `terrain_real_mean`
- `terrain_real_max`
- `terrain_discrete_mean`
- `terrain_discrete_max`
- `terrain_move_up_rate`
- `terrain_move_down_rate`
- `terrain_failure_down_rate`
- `terrain_stable_end_rate`
- `terrain_speed_ok_rate`
- `terrain_eligible_frac`
- `terrain_discrete_move_up_rate`
- `terrain_discrete_move_down_rate`
- `terrain_discrete_failure_down_rate`
- `terrain_discrete_stable_end_rate`

门控字段：

- `gait_gate`
- `diagonal_gate`
- `duty_balance_gate`
- `slip_gate`
- `duty_spread_gate`
- `fall_gate`
- `yaw_ceil`

用途：

- 判断训练为什么卡在当前 phase。
- 判断地形课程是真实推进、回退、还是仅 flat 等级误导。
- 判断是 progress、fall、quality、terrain gate、DR gate 中哪个条件在阻塞。

## health payload

稳定字段：

- `stable_motion_gate`
- `moving_gate`
- `stand_gate`
- `base_h`
- `base_h_min`
- `upright`
- `tilt_deg`
- `tilt_deg_max`
- `support_instability`
- `height_low_risk_window`
- `tilt_high_risk_window`
- `contact_count`
- `contacts_mean`
- `torque_util`
- `terminal_rate`
- `fall_rate`

用途：

- 判断机器人是否真的稳定。
- 区分低世界 z、低局部 base 高度、姿态倾斜和真实跌倒。
- 判断脚重、力矩、支撑不足和 reset 风险。

## 不要随意改名的字段

以下字段直接影响前端和智能体判断，应视为强契约：

- `command.progress_*`
- `command.raw_progress_*`
- `command.progress_validity`
- `command.progress_lagging_dir`
- `command.gait_match`
- `command.diagonal_contact`
- `command.duty_balance`
- `command.stance_slip`
- `curriculum.phase`
- `curriculum.progress_gate`
- `curriculum.active_dirs`
- `curriculum.terrain_real_*`
- `curriculum.terrain_discrete_*`
- `curriculum.blocked_by`
- `curriculum.next_gate`
- `health.stable_motion_gate`
- `health.base_h_min`
- `health.tilt_deg_max`
- `health.fall_rate`
- `reward.tracking_lin`
- `reward.tracking_yaw`
- `reward.terrain_up`
- `reward.terrain_down`
- `reward.terrain_event_collapse`

## 后续整理原则

- telemetry builder 已拆到 `telemetry_payloads.py`，后续整理必须继续保持四类 payload 名称不变。
- 字段契约测试在 `tests/autotuner/blind_locomotion/test_telemetry_contract.py`，改字段前先改测试和文档。
- 新增字段可以，但删除或改名需要同步文档、前端、智能体和测试。
- debug 字段如果要删除，先确认最近训练和诊断没有依赖它。
