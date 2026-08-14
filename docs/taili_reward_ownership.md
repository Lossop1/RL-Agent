# Taili 奖励职责表

本文只描述当前生效语义。历史楼梯状态机、承重层切换、势能奖励和预设足序均不属于当前策略。

## 基础驱动

线性命令的基础正奖励为：

```text
R_linear = alive * moving * (2.25 * near_tracking + 1.00 * far_tracking + 0.75 * direction_progress)
```

yaw 命令的基础正奖励为：

```text
R_yaw = alive * moving * (2.25 * near_tracking + 1.00 * far_tracking + 0.75 * yaw_progress)
```

四个方向的基础预算均为 `4.0`。姿态、支撑、步态成熟度和命令持续时间不得乘入这组奖励。真实 terminal 或非法数值可以关闭信用。

碰到地形以后，线速度大小可以放宽，方向不放宽。`supported_progress` 连续混合平地和碰障后的低速比例，只保留一个方向推进奖励；`terrain_progress` 不再叠加第二份信用。

## 正向质量

| 目标 | 当前负责人 |
|---|---|
| 宽泛自然风格 | 条件 AMP，`style_reward_weight=0.50` |
| 解析动作轻锚点 | live imitation，`w_imitate_live=0.60` |
| 接触节律吸引 | `gait_anchor` |
| 完成真实接触周期 | `contact_exchange` |
| 站立 | `stand`、`stand_far`、`stand_contact` |

AMP、解析 imitation 和接触锚点不能替代速度跟踪。固定足端轨迹惩罚已关闭，避免同一风格被多套参考重复约束。

## 尾部约束

| 全局要求 | 当前负责人 |
|---|---|
| roll/pitch 姿态 | `orient` |
| roll/pitch 角速度 | `base_wxy` |
| 垂向起伏 | `base_vz` |
| 平地身体高度 | `flat_move_height` |
| 支撑存在与方向结构 | `support_integrity`，只含接触与支撑结构 |
| 非命令轴运动 | `off_axis`、`heading_hold` |
| 占空、对称、周期 | `duty_balance`、`diagonal_contact`、`contact_period` |
| 抬脚不足或过高 | `clearance_under`、`clearance_over` |
| 支撑滑移与碎触地 | `stance_slip`、`worst_leg_slip`、`contact_chatter` |
| 轻触地 | `landing_impact`、`touchdown_slip`、`touchdown_force_rate`、`terminal_swing_velocity` |
| 摆动平滑 | `swing_action_accel` |
| 髋与前后腿几何 | `hip_deviation`、`front_rear_extension` |
| 电机边界 | `torque_margin`、`torque_saturation` |
| 真实失败 | `terminal_penalty` |

约束只能处理对应的物理尾部，不能削弱基础任务梯度。

## 已关闭重复项

`linear_underspeed`、`yaw_underspeed`、`wrong_dir`、`terrain_progress`、`lateral_coordinated_progress`、`yaw_support_moment`、`yaw_wrong_moment`、`planar_purity`、`yaw_translation`、`foot_trajectory`、`action_magnitude` 和周期核心能量奖励当前权重均为零。

楼梯类型专属的 support、collapse、overspeed、down-control 以及首次 stumble 只做诊断，不进入奖励或 critic。楼梯使用与平地相同的方向、核心、支撑和触地目标；被动碰障后只放宽速度大小及必要动作范围。

角加速度及其原始尾部只做诊断，不进入奖励、critic 或 Gate。

## Reset 语义

RSI 可以提供合法姿态，但根速度和关节速度必须清零，不能提供免费跟踪信用。真实 terminal 后下一回合重试原命令；timeout 和首次初始化正常重采样。
