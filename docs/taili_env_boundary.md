# Taili 环境职责边界

生成时间：2026-07-12

本文档是旁路审计文档，不参与训练、不参与 payload 打包。它记录 `taili_amp_env.py` 与 `blind_tp_env.py` 的真实职责边界，避免后续整理时把父类/子类关系搞混。

## 总体结论

当前盲态训练入口不是单独运行 `TailiAmpEnv`，而是运行：

`TailiBlindTPEnv(TailiAmpEnv)`

因此：

- `taili_amp_env.py` 是基础环境层，负责 IsaacLab 场景、资产、传感器、命令、AMP、phase/terrain/DR 课程和基础日志。
- `blind_tp_env.py` 是当前盲态训练层，负责部署观测契约、TerrainPerceiver 训练张量、统一奖励路径、质量窗口、结构化遥测和分方向 progress。
- 不能只看父类 `_get_rewards` 判断当前训练奖励，也不能只看子类忽略父类的 phase gate 和 terrain/DR gate。

## TailiAmpEnv 职责

文件：

`products/taili/blind_locomotion/taili_amp_env.py`

主要职责：

- 创建 IsaacLab 场景、机器人、接触传感器、高度扫描器和地形。
- 维护 AMP motion loader、AMP 观测缓冲和参考动作采样。
- 维护命令采样和命令过渡。
- 维护 gait phase、对角小跑时钟和 yaw 感知 cadence。
- 维护 phase gate、penalty gate、terrain gate、DR gate。
- 维护 terrain curriculum 和 DR curriculum。
- 打印 `[ENV]` 文本训练日志。
- 提供基础 `_get_rewards`，但当前盲态训练会被 `TailiBlindTPEnv._get_rewards` 覆盖。

关键方法：

- `__init__`：初始化资产、AMP、命令上限、DR 状态、phase 状态、课程门控状态。
- `_setup_scene`：创建地形、高度扫描器、机器人和接触传感器。
- `_resample_commands`：基础命令采样，按 phase command spec 和速度上限生成命令。
- `_begin_command_transition` / `_update_command_transition`：命令切换过渡，避免瞬间反向。
- `_pre_physics_step`：动作延迟、命令过渡、gait phase 更新。
- `_ensure_gate_mask`：构建课程门控 mask，避免 hard terrain 直接拖低平地阶段门控。
- `_terrain_level_stats`：按 flat / real / discrete 统计地形等级。
- `_compute_terrain_ctx`：用高度扫描器生成局部地形上下文。
- `_compute_amp_obs`：生成 AMP 判别器观测。
- `_get_observations`：基础 AMP 环境观测；盲态子类会覆盖。
- `_log_training_diag`：读取 progress、质量窗口、地形统计，执行 phase/DR 课程推进并打印日志。
- `_get_rewards`：基础 AMP 奖励路径；盲态子类会覆盖。
- `_get_dones`：超时和跌倒终止。
- `_reset_idx`：基础 reset、地形 curriculum 更新、DR 应用、命令和参考 motion 重采样。
- `collect_reference_motions`：AMP 参考样本采集。

## TailiBlindTPEnv 职责

文件：

`products/taili/blind_locomotion/blind_tp_env.py`

主要职责：

- 把策略观测改为部署契约：`body57 + tick54 history`；body57 包含当前命令、上一命令和命令年龄。
- 给 critic 和辅助训练追加 privileged obs 和 aux labels。
- 管理 TerrainPerceiver 的历史输入。
- 管理每个 env 独立命令保持时间，打散 reset 和命令切换相位。
- 覆盖 `_get_rewards`，统一走 `taili_reward.compute_reward_components`，并补充地形事件、盲态质量窗口和遥测字段。
- 计算分方向 progress，并写入父类 `_log_training_diag` 会读取的 `_fwd_prog/_back_prog/_lat_prog/_yaw_prog`。
- 维护结构化遥测 `train.telemetry.jsonl`。
- reset 时补充盲态历史、质量窗口和地形 latch 状态复位。

关键方法：

- `__init__`：初始化 tick history、奖励配置、遥测器、质量窗口、地形事件 latch。
- `_resample_commands`：在父类采样基础上应用 phase curriculum、bucketed commands、固定命令覆盖和命令过渡。
- `_draw_cmd_hold`：按 env 独立采样命令保持时长。
- `_get_observations`：组装 body57、tick54 历史、privileged obs、aux labels 和 AMP buffer。
- `_compute_aux_labels`：从高度扫描器和风险指标生成训练辅助标签。
- `_reset_idx`：调用父类 reset 后，复位盲态历史、质量窗口、support_z 和地形 latch。
- `_update_quality_windows`：更新 duty、对角接触、滑移、高度和倾斜窗口。
- `_compute_amp_obs`：盲态任务的 AMP 观测；不直接把地形标签喂给 actor。
- `_terrain_height_under_base`：估计 base 下方局部地面高度，避免下坡/楼梯世界 z 误判。
- `_get_rewards`：当前盲态训练的实际奖励路径、地形探针、分方向 progress 和结构化遥测。

## TailiBlindTPEnv._get_rewards 分段地图

`_get_rewards` 是当前盲态训练最核心也最危险的函数。它不是单纯“算 reward”，还写入课程、遥测和后续日志会读取的状态。后续重构前，应先按下面边界理解。

### 1. 原始仿真量和基础状态

输入来源：

- `self.robot.data`
- `self.commands`
- `self._in_contact`
- `self._terrain_height_under_base`
- `self._contact_sensor`

主要产物：

- `base_h`
- `tilt_rel`
- `moving`
- `stand_gate`
- `settled_contact`
- `landing_mask`
- `foot_pos`
- `foot_vel`
- `foot_vel_xy`

作用：

- 给稳定门控、滑移、触地冲击、质量窗口和 RewardInput 提供基础张量。

### 2. 高度扫描、clearance 和前方地形探针

输入来源：

- `self._height_scanner.data.ray_hits_w`
- `self._terrain_ctx`
- `self._discrete_terrain_mask`
- `self.commands`

主要产物：

- `foot_clearance_terr`
- `local_obstacle_h`
- `terrain_rise_ahead`
- `terrain_drop_ahead`
- `terrain_scan_probe`
- `_support_z`
- `_ground_z`

作用：

- 为 `taili_reward` 的 clearance / terrain-aware tracking 提供输入。
- 为地形 up/down/climb 事件提供前方 rise/drop 证据。
- 为结构化遥测提供地形扫描诊断字段。

### 3. 质量窗口和稳定门控

输入来源：

- 接触状态、稳定支撑、足端平面速度、base 高度、倾斜角、moving。

主要产物：

- `quality`
- `_base_h_min`
- `_tilt_deg_max`
- `support_instab`
- `terminal_window`
- `gate`

作用：

- `gate` 会进入 `taili_reward.stable_motion_gate`，决定 tracking、stand、gait 等塑形项是否有效。
- `quality` 里的 duty、对角接触、滑移尾部等指标既影响奖励，也会进入 telemetry 和父类 phase gate。

### 4. RewardInput 组装

输入来源：

- 原始状态、质量窗口、clearance、terrain probe、动作、力矩、命令和 gait clock。

主要产物：

- `inp = types.SimpleNamespace(...)`

作用：

- 这是 `taili_reward.compute_reward_components` 的唯一输入对象。
- 如果后续拆函数，必须保证 `inp` 字段名和语义不变，否则奖励测试可能无法覆盖到远端运行差异。

### 5. 统一奖励计算和 terminal 惩罚

输入来源：

- `inp`
- `self._rcfg`

主要产物：

- `comp`
- `total`

作用：

- `taili_reward.compute_reward_components` 输出通用奖励分量。
- terminal 惩罚在子类里按 `terminal_window` 逐 env 扣除。

### 6. 盲态额外方向辅助奖励

主要分量：

- `lateral_foot_excursion`
- `backward_underspeed`
- `lateral_underspeed`
- `settle_brake`

作用：

- 补足 `taili_reward` 中不直接表达或需要环境状态的窄项。
- 这些项会进入 `comp` 和 `total`，也会进入 reward debug / telemetry。

### 7. 地形事件奖励

输入来源：

- `terrain_rise_ahead`
- `terrain_drop_ahead`
- `_support_z`
- `_ground_z`
- `landing_mask`
- `foot_vel`
- `in_contact`
- `_discrete_terrain_mask`

主要分量：

- `climb`
- `terrain_up`
- `terrain_down`
- `terrain_support_transfer`
- `terrain_contact_quality`
- `terrain_event_collapse`

作用：

- 给上/下台阶、支撑转移、地形接触质量和地形事件崩溃提供直接奖励或惩罚。
- 该段是当前突破地形的核心之一，不能和普通平地 tracking 混为一谈。

### 8. 分方向 progress 统计

输入来源：

- `rd.root_lin_vel_b`
- `rd.root_ang_vel_b`
- `self.commands`
- `base_h`
- `tilt_rel`
- `terminal_window`
- `self._gate_mask`

主要产物：

- `_fwd_prog`
- `_back_prog`
- `_lat_prog`
- `_yaw_prog`
- `_fwd_raw_prog`
- `_back_raw_prog`
- `_lat_raw_prog`
- `_yaw_raw_prog`
- `_progress_validity`

作用：

- 父类 `_log_training_diag` 会读取这些值推进 phase、DR 和速度上限。
- `raw_progress_*` 只表示速度投影；`progress_*` 乘了姿态有效性门控。

### 9. 奖励分组和文本 debug

主要产物：

- `_reward_groups`
- `_rew_dbg`
- `_budget_ratio_ema`

作用：

- `_reward_groups` 供 multi-critic 或后续分析使用。
- `_rew_dbg` 供父类 `[ENV]` 日志展示。
- `_budget_ratio_ema` 影响父类 penalty gate。

### 10. 结构化遥测

主要 payload：

- `reward_payload`
- `command_payload`
- `curriculum_payload`
- `health_payload`

作用：

- 写入 `train.telemetry.jsonl`。
- 前端、智能体和诊断解释主要读取这部分。
- 字段名已经被前端和分析工具依赖，整理时不能随便改名。
- payload 字段构造已抽到 `telemetry_payloads.py`；`_get_rewards` 仍负责计算上下文和调用 emitter。

## 跨文件耦合点

这些点是后续整理的高风险区域，不能随意移动或改名。

### Progress

`TailiBlindTPEnv._get_rewards` 计算：

- `_fwd_prog`
- `_back_prog`
- `_lat_prog`
- `_yaw_prog`
- `_fwd_raw_prog`
- `_back_raw_prog`
- `_lat_raw_prog`
- `_yaw_raw_prog`
- `_progress_validity`

`TailiAmpEnv._log_training_diag` 读取这些值，用于：

- phase gate
- velocity curriculum
- terrain gate
- DR gate
- 文本日志

因此 progress 不是单纯遥测字段，而是课程推进输入。

### Quality Gate

`TailiBlindTPEnv._update_quality_windows` 写入：

- `_diag_contact`
- `_duty_balance`
- `_slip_now`
- `_slip_high_fraction`
- `_duty_spread`
- `_base_h_min`
- `_tilt_deg_max`

`TailiAmpEnv._log_training_diag` 会读取其中一部分，决定 `quality_ok` 和 terrain health。

因此质量窗口也是课程输入，不只是展示。

### Terrain Gate

父类负责：

- 地形等级统计。
- flat / real / discrete mask。
- terrain curriculum move up/down。
- phase gate 中的 terrain gate。

子类负责：

- 高度扫描器前方 rise/drop 探针。
- `terrain_up / terrain_down / climb` 的激活和遥测。
- `local_obstacle_h`、clearance target、地形事件 latch。

因此“地形能力”同时受父类课程推进和子类奖励/探针影响。

### Command Transition

父类提供通用命令过渡：

- `_begin_command_transition`
- `_update_command_transition`

子类决定何时重采样、是否按 env 独立保持、是否 snap、是否应用 phase curriculum。

因此过渡态不能只在父类或只在子类看。

### Telemetry

父类输出人读日志：

- `[ENV]`
- `[PHASE]`
- `[DR]`

子类输出结构化遥测：

- reward payload
- curriculum payload
- health payload
- command payload
- terrain probe payload

前端和智能体更依赖子类结构化遥测；训练现场排查仍常看父类文本日志。payload 字段构造由 `telemetry_payloads.py` 管理，字段契约仍由子类 `_get_rewards` 调用 emitter 暴露。

字段稳定性和关键字段列表见 `docs/taili_telemetry_contract.md`。

## 当前整理风险

1. `_get_rewards` 名字在父类和子类都存在。当前盲态训练实际走子类版本，但父类版本仍有价值，不能删除。
2. `_get_observations` 名字在父类和子类都存在。当前盲态任务走子类版本，但 AMP buffer 机制仍沿用父类约定。
3. `_reset_idx` 名字在父类和子类都存在。子类 reset 必须调用父类 reset，再补盲态状态。
4. progress 写在子类，phase gate 用在父类。任何字段改名都需要同时改两边和遥测。
5. 地形 rise/drop 探针和地形 curriculum 等级不是同一概念。前者是局部感知/奖励信号，后者是 IsaacLab 课程等级。
6. 高度扫描器是训练侧/诊断侧重要信号，但 actor 不能直接吃特权地形标签；整理时必须保持盲态限制。

## 后续建议

不要直接大规模合并文件。建议按以下顺序整理：

1. 先把 `_get_rewards` 内部逻辑分段写清楚，必要时只提取纯函数，不改变张量语义。
2. 继续补强 progress 计算和 telemetry 字段的回归测试或快照测试。
3. 把 terrain probe 字段整理成小的只读 helper，先保持字段名不变。
4. 把 command transition 的父类/子类调用关系写成测试或文档，避免后续重构导致命令瞬变。
5. telemetry payload 已分文件；后续再考虑 terrain reward、quality windows 等更高风险拆分，每拆一步都要先跑 manifest 和核心测试。

## 本轮不做的事

本轮不改：

- 奖励数值。
- phase/terrain/DR 门槛。
- 命令采样概率。
- progress 公式。
- reset 逻辑。
- 除新增 `telemetry_payloads.py` 外的 payload 打包列表。

本轮只记录职责边界，并修正误导性文件头说明。
