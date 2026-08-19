# Taili 当前训练基线

更新时间：2026-07-16

本文档描述当前本地准备部署的完整训练策略。它不参与运行，实际权威来源仍是：

`products/taili/blind_locomotion/taili_blind_config.yaml`

训练实现由以下文件共同完成：

- `products/taili/blind_locomotion/blind_tp_env.py`：盲态观测、奖励输入、接触事件和遥测。
- `products/taili/core/taili_reward.py`：平地与地形共用奖励。
- `products/taili/core/terrain_curriculum.py`：楼梯事件、阶段势能和地形课程判定。
- `products/taili/blind_locomotion/taili_amp_env.py`：phase、地形和 DR 推进。
- `products/taili/payload/payload_manifest.py`：部署文件边界。

## 全局目标

策略同时服务以下能力，不能通过牺牲其中一项换取另一项的表面指标：

- 平地前进、后退、横移、yaw 和站立的有效跟踪。
- 机身 roll、pitch、yaw、高度和角加速度稳定。
- 对称、支撑完整、低滑移、自然足端轨迹和轻触地。
- 命令突变后的自然过渡，但过渡不作为早期主门控。
- 上楼梯、下楼梯、boxes、粗糙地面和坡面的真实通过能力。
- 能力达标后再加入域随机化，避免用扰动掩盖基础策略问题。

25 至 30 cm 上下楼梯是最终验收目标，不代表当前配置已经证明达到该能力。

## 盲态边界

Actor 不读取前方高度扫描、地形类型或未来接触信息。Actor 输入保持可部署的本体信息：`body57` 与 `tick54` 历史；历史为 25 帧、步长 2，在 20 ms 控制周期下覆盖约 1 秒。

奖励、课程和 Critic 可以使用训练期特权信息，但严格遵守两点：

1. 只有实际接触、承重、机身响应或碰撞后，才允许切换地形响应语义。
2. 未接触区域的高度扫描不能提前告诉 Actor 或奖励“前方是什么地形”。

因此当前策略仍是被动感知的盲狗，而不是依赖前视地形扫描的显式规划器。

## 训练分布

地形从 phase 0 就参与训练，不再先训练纯平地、后补地形：

| 地形 | 比例 |
|---|---:|
| 平地 | 0.50 |
| 正坡 | 0.08 |
| 反坡 | 0.07 |
| 粗糙地面 | 0.08 |
| 下楼梯 | 0.09 |
| 上楼梯 | 0.09 |
| boxes | 0.09 |

楼梯高度范围为 `0.04~0.30 m`，boxes 为 `0.04~0.20 m`。平地仍占一半，用于持续保护平地质量；难度地形从开始就提供探索样本和梯度。

完整速度包络从 phase 0 同时出现，不再等某一阶段通过后才训练最终速度：

| 方向 | 训练范围 |
|---|---:|
| 前进 | `0.15~1.00 m/s` |
| 后退 | `0.15~0.80 m/s` |
| 横移 | `0.12~0.50 m/s` |
| yaw | `0.15~1.00 rad/s` |

phase 只改变命令组合与验收强度：

- phase 0：单轴全方向，前进、后退、横移、yaw 和站立同时训练。
- phase 1：加入 mixed commands 和近零命令。
- phase 2：提高混合命令和地形验收要求，不是地形训练起点。
- phase 3：最终混合命令和鲁棒性验收，不改变速度范围。

phase 0 的目标环境存量为 `10%` 站立，其余 `90%` 在四个方向均分，即每个方向约 `22.5%`。重采样按当前存量缺口补齐，而不是只保证重采样事件等概率。这样容易终止的前进、后退和横移不会在 reset 后逐渐消失，静止但不终止的 yaw 也不能依靠更长存活时间占据 rollout。

`phase_timeout_enable: false`，阶段不能靠等待强制通过。`terrain_start_phase: 0`、`terrain_reward_start_phase: 0`、`dr_start_phase: 0`。DR level 0 只提前随机化 `Kp 0.9~1.1`、`Kd 0.85~1.15` 且作用于 50% reset；推扰、质量、摩擦、CoM 和 IMU 偏置仍由正式 DR 等级门控。

## 严格与放松

设接触后地形响应为 `r_t in [0, 1]`。它来自足端接触、支撑面变化和机身响应，不来自前视扫描。

平地质量不会在训练早期关闭。阶段验收仍取四方向最弱能力；奖励强度使用四方向平均成熟度，避免一个落后方向让所有质量项长期停在 floor。设：

```text
c_i = clip((progress_i - 0.20) / (0.65 - 0.20), 0, 1)
c_mean = mean(c_fwd, c_back, c_lat, c_yaw)
q_t = max(q_(t-1), 0.35 + 0.65 * penalty_gate * c_mean)
```

课程是否通过仍使用最弱方向，不因平均值而放过短板。质量调度和能力验收具有不同职责。

平地时 `r_t = 0`，姿态、高度、步态和跟踪保持严格。接触到难度地形后，只放松确实需要重排的部分：

```text
gait_pattern_scale = 1 - 0.65 * r_t
orientation_scale  = 1 - 0.70 * r_t
height_scale       = 1 - 0.80 * r_t
tracking_scale     = 1 - terrain_difficulty * (1 - 0.45)
```

即使地形响应完全激活，步态结构仍保留 35%，高度约束保留 20%，姿态约束保留 30%。离散障碍首次碰触后、先导脚尚未放置前，通用推进暂时关闭，精确跟踪保留 25% 的事件压力；放置后连续恢复。角速度、角加速度、支撑完整和自由落体检查不随地形响应消失。

平地机身高度使用对称带宽约束：目标 `0.52 m`、容许带 `0.025 m`，过低和过高都会受罚，避免只惩罚下蹲后形成新的高抬身体漏洞。

## 楼梯驱动

一般线性推进和地形推进已经拆开：

```text
supported_progress: 所有线性命令下的可靠支撑推进，平地也生效
terrain_progress:   只有接触后 terrain_response > 0 才生效
```

平地不能再贡献名为 `terrain_progress` 的奖励；楼梯完全失败时不会被普通行走正奖励掩盖。平地线性与 yaw 的基础任务预算均约为 `4.5`，历史 yaw 补偿已经回调。

上、下楼梯使用同一套物理阶段框架，但动作语义不同。六个阶段是：

1. 先导腿卸载，同时其他腿保持支撑。
2. 上楼抬升并沿命令方向前送；下楼受控探低。
3. 先导腿接触新层，且触地质量有效。
4. 新层足端开始承重。
5. 机身在核心稳定条件下完成换层。
6. 其余腿跟随，并重新形成完整支撑。

上楼第二阶段不是单纯抬高足端：

```text
up_motion = 0.5 * lift + 0.5 * min(lift, forward_advance)
```

因此原地高抬腿最多取得一半该阶段分数，必须把足端送到新踏面方向才有完整驱动。

六阶段顺序仍用于验收、课程与遥测，但不再直接构成主奖励势能。主势能只使用当前可恢复的物理状态：

```text
support_q = 0.30 + 0.70 * support_transfer
Phi = support_q * (0.30*probe + 0.20*motion + 0.20*placement_with_load
                   + 0.12*lead_load + 0.10*body_confirmed + 0.08*follow_confirmed)
r_shape = 4*max(Phi_t-Phi_(t-1), 0) - 4*max(Phi_(t-1)-Phi_t, 0)
```

静止维持同一状态时奖励为零；抬脚、前送、触地承重、机身和跟随腿逐步形成连续梯度；结果丢失时完整扣回。卸载本身不进入势能，碰阶后自动卸载不能冒充上楼。下楼的正增量和首次完成奖励还必须通过受控下降门控，失控自由落体不能领取完成奖励。

新事件写入 `steps_left` 后必须立即重算事件作用域。旧实现继续使用刷新前的 `event_alive=False`，会在第一次碰阶当帧把方向清零；当前实现已修正。事件窗口为 `2.30 s`，覆盖约三个低速步态周期，使先导腿、机身和其余腿有时间完成一次真实换层。

地形进展还要求真实支撑和运动质量。自由落体、零接触、失去直立或严重机身运动时，地形跟踪与进展奖励为零。课程成功必须同时满足非终止、稳定支撑、受控速度和真实换层事件。

## 平地质量

当前平地约束集中在可解释的物理量，不继续为每个视觉瑕疵增加独立指标：

- 核心：姿态误差、roll/pitch 角速度、角加速度、竖直速度和移动高度。
- 跟踪：沿命令方向的真实速度，同时约束非命令轴漂移。
- 步态：实际接触周期、duty 区间、左右/前后/交叉对称和支撑完整。
- 轨迹：摆动相位顶点处检查 clearance，均值与最差单腿按 `0.5/0.5` 混合。
- 轻触地：检查触地竖直速度、单腿尾部冲击、触地水平速度和力变化率。
- 髋关节：直线运动严格限制最差单腿外翻；横移、yaw 和地形保留必要自由度。

Duty 现在按稳定线性命令下约一个目标周期的接触时间占比计算；实际周期按同一只脚连续 touchdown 的时间间隔估计。纯对角接触只评价对角结构，不再是 duty/period 获得有效值的前置条件。样本不足时保持无效，不把初始化的 `0.5` 当作理想步态。

轻脚的承重跃迁使用 touchdown 后 `0.10 s` 窗口内最大正向接触力上升量；不惩罚卸载，也不再漏掉触地后一两个控制帧才出现的力峰值。机身角加速度仍独立约束，避免足端速度较小但整机冲击明显。

当前执行器训练参数统一为刚度 `100`、阻尼 `5`。这更接近现有真机策略使用的量级，但仍需通过 MuJoCo 和真机长时运行验证，不能由日志分数替代。

## AMP 与优化窗口

当前 AMP 解析参考主要描述平地命令步态，没有经过物理验证的楼梯专家轨迹。为避免平地风格压制换层探索：

- `style_reward_weight: 2.0`
- `rollouts: 32`

Actor 的部署输入仍是 `body57 + history25x54`，但训练专用风险标签从 2 维扩为 8 维，用于碰触后事件方向和动态先导脚监督；总训练观测从 `1626` 变为 `1638`，因此必须 fresh。没有引入伪造的楼梯 AMP 参考，后续只有在获得事件对齐、物理可行的专家轨迹后才应加入。

## 训练与验收

本轮改变了命令分布、奖励预算、质量调度、duty/period 统计、楼梯事件作用域和 DR 起始语义，应使用 fresh 训练。旧检查点只用于固定诊断对照，不应作为本轮有效性证明。

验收必须交叉验证：

1. 日志好、诊断物理数据好：语义和能力初步一致。
2. 日志好、诊断坏：指标存在漏洞，不能继续按日志推进。
3. 日志坏、诊断好：门控或聚合语义错误，应先修评价。
4. 日志坏、诊断坏：策略本身未学会，需要检查逻辑、权重和冲突。

固定检查点至少验证：平地全方向、15/18/22 cm 上下楼梯、boxes、长时恒定命令和突变命令。成功标准必须看真实位移、换层、支撑、姿态、触地和是否终止，不能只看 `terrain_max`。

需要独立评价方向能力时，每个方向必须获得真实物理 reset。skrl IsaacLab wrapper 默认只执行一次底层 reset，诊断入口会显式重新打开其 reset 缓存；连续全方向诊断仍用于观察过渡和累积失稳，两种诊断不能互相替代。

## 本轮产物

- 修改前快照：`strategy_backups/pre_global_flat_terrain_20260716_143900`
- 本地 payload：`products/taili/payload/dist/taili_blind_runtime_20260717_structural_stair_fix.tar.gz`
- payload SHA-256：`5af99c29033cf96bad461d3374bfd735024cb1d4f15cee50c0d0964bd529d888`
- 本轮没有修改远程训练状态。

验证命令：

```powershell
python -m py_compile products/taili/core/terrain_curriculum.py products/taili/core/taili_reward.py products/taili/blind_locomotion/blind_tp_env.py products/taili/blind_locomotion/taili_amp_env_cfg.py products/taili/blind_locomotion/taili_blind_config.py
python -m pytest tests/products/taili/core tests/products/taili/blind_locomotion -q --basetemp=.pytest-tmp-global-terrain-all
python -m products.taili.payload.payload_manifest
python -m products.taili.payload.build_payload --out output --stamp global_rebalance_20260716
```
