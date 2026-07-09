# Taili 训练观察记录 2026-07-09

## 判定原则

- 不按单个指标局部调参；先判断是否服务于全局目标：平地锚点和高难地形能力同等重要。
- 不放松质量门槛来换取进入地形；避免滑行、拖脚、碰撞或不均衡支撑骗取 progress。
- 不轻易提高 tracking 权重；平地 tracking 不是最终目标，不能把高地形步态压回平地模板。
- 优先使用现有能力级表达：progress、healthy contact、stance slip、duty/diag、clearance、fall，不堆局部新指标。

## 00:53 干预

旧 run：`taili_train_20260708_234503_console`

观察：

- 已从 `phi0` 进入 `phi1`，说明第二版滑移约束比上一版有效。
- `progress_gate` 已稳定高于 `phi1` 要求，但 `duty_balance` 从约 `0.61` 下滑到 `0.58` 附近，低于 `phase_gate_duty_1=0.64`。
- `stance_slip` 均值接近阈值，但 `slip_hi` 仍约 `0.26-0.27`，说明接触尾部仍不干净。

判断：

- 不是 progress 瓶颈，而是进入地形前的接触/占空质量瓶颈。
- 不应放松 gate，也不应继续加前进奖励。

干预：

- 只把 `w_duty_balance` 从 `0.45` 提到 `0.60`。
- 不动 `tracking_lin` / `tracking_yaw`。
- 不动 `phase_gate_*`。
- 不新增指标。

备份：

- `strategy_backups/pre_phi1_duty_intervention_20260709_005319/local`

新 run：

- `taili_train_20260709_005456_console`
- payload: `/root/gpufree-data/training_payloads/taili_blind_runtime_20260709_005446`

## 后续观察标准

- 若 15k-25k step 前后仍无法进入 `phi1`：说明 `w_duty_balance=0.60` 影响了早期 progress，需要回调或改变渐入方式。
- 若能进入 `phi1`，重点看 `duty_balance` 是否能向 `0.64` 靠近，同时 `progress_gate` 不崩。
- 若进入 `phi2` 后 `terrain_mean` 不推进，且 `slip_hi` 仍高：下一步应考虑更结构性的接触恢复/清障驱动，而不是继续加平地步态权重。
- 若 `diag/duty` 过高但地形失败：说明平地模板过强，需要在地形阶段进一步弱化固定节律，而不是继续加步态项。
