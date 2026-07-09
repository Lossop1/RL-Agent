# 当前最终训练策略说明

生成时间：2026-07-08  
项目根目录：`D:\RL-Dog\hand\locomotion-workspace_backup\locomotion-workspace`

## 结论

当前本地项目内，Taili blind locomotion 训练只保留了一套完整可打包的最终运行策略。真正的单一策略源是：

`autotuner/blind_locomotion/taili_blind_config.yaml`

该文件当前 SHA256：

`80AA8A4178D81366487319B52D9E647ABB2DF8751214266626DFE7FEF539ECE7`

这套策略已经被本地 payload manifest 接受，并能打包为远端训练运行包。当前记录中的最终远端 payload 是：

`/root/gpufree-data/training_payloads/taili_blind_runtime_final_single_source_20260708_204133`

它的 YAML 哈希与本地上述哈希一致。只要远端 `/root/gpufree-data/training_payloads/taili_blind_runtime_*` 中这个目录仍是最新目录，控制台“启动全新训练”会使用这套 final payload；“从检查点继续”会在同一 payload 逻辑下额外传入 checkpoint。

## 运行边界

训练运行链路不再依赖历史复制目录、旧 patch 文件或远端 `robot_lab.tasks.direct...` 覆盖。payload 是自包含的 `taili_blind_runtime` 包，放到 `PYTHONPATH` 后注册自己的 Gymnasium task：

`RobotLab-Isaac-Taili-AMP-Blind-Direct-v0`

启动入口在 payload 内：

`python -m taili_blind_runtime.launch_taili_train`

该入口会在 run 目录中生成运行证据：

- `taili_blind_config.yaml`：本次 run 使用的策略副本。
- `agent.skrl.yaml`：从策略 YAML 生成的 skrl agent 配置。
- `effective_config.yaml`：带 total steps、checkpoint interval、run path 等运行覆盖后的完整配置。
- `run.json`：payload、run、checkpoint、日志路径和启动命令元数据。
- `train.log` / `train.telemetry.jsonl` / `console.log`：训练可观测日志。
- `checkpoints/`：skrl checkpoint。

## 真正参与训练的本地文件

策略配置：

- `autotuner/blind_locomotion/taili_blind_config.yaml`
- `autotuner/blind_locomotion/taili_blind_config.py`

环境与任务注册：

- `autotuner/blind_locomotion/__init__.py`
- `autotuner/blind_locomotion/taili_amp_env.py`
- `autotuner/blind_locomotion/taili_amp_env_cfg.py`
- `autotuner/blind_locomotion/blind_tp_env.py`
- `autotuner/blind_locomotion/blind_tp_env_cfg.py`
- `autotuner/blind_locomotion/taili_blind_env_cfg.py`

奖励、课程、观测、模型和导出核心：

- `autotuner/taili_core/*.py`
- 其中 `autotuner/taili_core/taili_reward.py` 是奖励计算核心。

策略模型与辅助训练：

- `autotuner/blind_locomotion/terrain_perceiver_policy.py`
- `autotuner/blind_locomotion/terrain_perceiver_aux_patch.py`
- `autotuner/blind_locomotion/symmetry.py`
- `autotuner/blind_locomotion/_symmetry_local.py`

参考动作与 AMP 数据：

- `autotuner/blind_locomotion/motions.py`
- `autotuner/blind_locomotion/multi_motion_loader.py`
- `autotuner/blind_locomotion/parametric_ref.py`
- `autotuner/blind_locomotion/motions/clips/*.npz`

训练、诊断、验收和遥测：

- `autotuner/blind_locomotion/train_taili.py`
- `autotuner/blind_locomotion/launch_taili_train.py`
- `autotuner/blind_locomotion/telemetry_emit.py`
- `autotuner/blind_locomotion/diagnose_taili.py`
- `autotuner/blind_locomotion/diagnose_taili_cases.py`
- `autotuner/blind_locomotion/physeval_blind.py`
- `autotuner/blind_locomotion/physeval_blind_e.py`
- `autotuner/blind_locomotion/physeval_suite.py`
- `autotuner/blind_locomotion/acceptance_score.py`
- `autotuner/blind_locomotion/acceptance_aggregate.py`

机器人资源：

- `locomotion-console-ui/public/robot/taili_dog_description/urdf/robot.urdf`
- `locomotion-console-ui/public/robot/taili_dog_description/meshes/*`

payload 定义：

- `autotuner/training_payloads/taili_blind_runtime/payload_manifest.py`
- `autotuner/training_payloads/taili_blind_runtime/build_payload.py`

## 保留但不算策略源的工具

以下文件有用，应保留，但它们不是当前训练策略源，也不会被当作独立策略：

- `autotuner/blind_locomotion/gen_taili_gaits.py`：生成或维护 AMP gait clips 的工具。
- `autotuner/blind_locomotion/reward_tuning_advisor.py`：奖励调参分析工具。
- `autotuner/locomotion_console/*`、`locomotion-console-ui/*`：控制台、智能体交互、可视化和远端操作界面。
- `tests/*`：单测和回归验证。
- `docs/*`、`README.md`：说明文档。

## 已清理的历史污染

当前运行边界内不再保留这些旧策略/旧运行副本：

- `autotuner/blind_locomotion/env_edit`
- `autotuner/blind_locomotion/taili_reward.py`
- `autotuner/blind_locomotion/blind_aux_patch.py`
- `autotuner/blind_locomotion/blind_terrain_policy.py`
- `.backups`
- `.payload_smoke`

当前唯一训练 YAML 是 `autotuner/blind_locomotion/taili_blind_config.yaml`。当前唯一 `taili_reward.py` 位于 `autotuner/taili_core/taili_reward.py`。

## 策略目标

这套策略的目标不是先把平地步态训到单一漂亮形态，再临时加地形；它把最终服务目标拆成四个同时约束：

- 命令可控：前进、后退、横移、转向和站立从早期就进入训练，避免策略把某个方向固化成唯一吸引子。
- 接触可用：不只奖励速度，还要约束触地、对角步态、站立四足着地、滑移和落地冲击。
- 地形可学：地形不是最后附加项，而是在平地基础稳定后由 curriculum gate 解锁。
- 鲁棒可推进：domain randomization 不抢在策略能过地形之前打开，避免早期噪声把步态破坏掉。

## 课程策略

`training_recipe.id`：

`taili_phased_acceptance_curriculum_v3`

`init_phase: 0`

四个 phase 的意图：

- phase 0：平地，single-axis，但四个方向和站立从 step 0 都参与。范围较小，重点是让命令通道一开始就有信息。
- phase 1：平地 mixed commands，加入混合速度、近零命令和 stop exposure。
- phase 2：开始地形和 domain randomization，目标从纯平地 tracking 转为可控通过。
- phase 3：最终速度包络，覆盖更完整的前进、后退、横移、转向范围。

关键门槛：

- `phase_gate_prog_0: 0.65`
- `phase_gate_prog_1: 0.62`
- `phase_gate_prog_2: 0.62`
- `phase_gate_prog_3: 0.68`
- `phase_gate_diag_0: 0.45`
- `phase_gate_duty_0: 0.55`
- `phase_gate_air_0: 0.0`
- `phase_gate_terrain_2: 3.5`
- `phase_gate_fall_2: 0.05`
- `terrain_start_phase: 2`
- `dr_start_phase: 2`

这里最重要的设计点是：地形和 DR 不能过早破坏尚未站稳的 gait，但也不能等到最后才出现。phase 2 是当前折中点。

## 地形与 DR 策略

地形能力的直接配置：

- stairs step height：`[0.08, 0.38]`
- stairs proportion：`0.25`
- boxes grid height：`[0.05, 0.30]`
- `phase_gate_terrain_2: 3.5`

DR 配置：

- `domain_randomization.enable: true`
- `domain_randomization.unlock_terrain: 3.0`
- `domain_randomization.gate_progress: 0.55`
- `domain_randomization.gate_progress_l2: 0.62`
- `domain_randomization.gate_progress_l3: 0.68`
- level 3 friction：`[0.4, 1.4]`
- level 3 mass：`[-5.0, 20.0]`
- level 3 stiffness/damping scale：`[0.6, 1.4]`

含义：DR 不是从 0 地形能力时强行开启，而是等 terrain max/mean 有足够证据后解锁。这样做是为了避免“地形、推扰、质量、摩擦、执行器扰动”同时进入，导致 16000 轮仍不会走的失败模式。

## 抬腿与触地策略

用户反馈的核心问题是抬腿高度偏低、地形能力不足。当前 final 策略保留了更明确的 clearance 目标：

- `blind_overrides.base_clearance: 0.08`
- `reward.flat_clearance_target: 0.08`
- `reward.terrain_clearance_margin: 0.04`
- `reward.clearance_band: 0.02`
- `reward.w_clearance_under: 1.0`
- `reward.w_clearance_over: 1.8`

含义：

- 平地目标是 8 cm，不鼓励无意义高抬腿。
- 地形上目标会根据局部障碍高度加 margin。
- under-clearance 给“够不着障碍高度”的梯度。
- over-clearance 抑制过高抬腿，避免用跳、砸、夸张摆腿骗过地形。

触地相关正向/约束项：

- `w_gait_anchor: 1.0`
- `w_stand_contact: 1.8`
- `w_diagonal_contact: 0.65`
- `w_duty_balance: 0.45`
- `w_feet_air_time: 0.5`
- `air_time_target: 0.30`

含义：

- 行走时保留节律锚点，防止策略只靠速度奖励拖动身体。
- 站立时显式奖励四足接触，防止身体不动但脚乱动。
- 对角触地和 duty balance 用于防止跛行、单侧偏置和错误节律。

## 速度、方向和站立策略

命令范围：

- phase 0 forward：`[0.30, 0.80]`
- phase 0 backward：`[0.15, 0.50]`
- phase 0 lateral：`[0.15, 0.50]`
- phase 0 yaw：`[0.30, 0.90]`
- phase 3 forward max：`1.50`
- phase 3 backward max：`0.80`
- phase 3 lateral max：`0.70`
- phase 3 yaw max：`1.50`

奖励带宽：

- `sigma_lin_abs: 0.10`
- `sigma_lin_rel: 0.15`
- `sigma_yaw: 0.08`
- `w_tracking_lin: 2.5`
- `w_tracking_yaw: 3.1`
- `w_track_far: 1.0`
- `w_yaw_far: 1.5`
- `w_wrong_dir: 0.8`

站立：

- `stand_prob` 在各 phase 约 `0.08` 到 `0.10`
- `sigma_stand_speed: 0.05`
- `sigma_stand_yaw: 0.05`
- `w_stand: 1.5`
- `w_stand_far: 1.5`
- `w_stand_contact: 1.8`

含义：速度奖励不是只奖励“向前”，而是按命令方向计算。wrong direction 作为显式惩罚，用来压制“命令后退但仍向前滑”“命令 yaw 但直走”等失败。

## Homeostat 与惩罚预算

当前策略仍保留 penalty budget / homeostat 思路：

- `phase_intervals: 3`
- `penalty_ramp_intervals: 16`
- `penalty_budget_ratio_max: 0.8`

作用是避免早期惩罚项过强，把策略压成站立、不动或保守拖步；随着 tracking、接触和 gait 质量变好，再逐步放大惩罚，让策略从“能动”过渡到“动得对”。

## AMP 与 gait 设计

AMP 相关配置：

- `style_reward_weight: 2.0`
- `discriminator_reward_scale: 3.0`
- `discriminator_loss_scale: 5.0`
- `amp_batch_size: 512`
- `discriminator_batch_size: 4096`

步态时钟：

- `gait.period: 0.55`
- `gait.period_min: 0.34`
- `gait.period_slope: 0.16`
- `gait.duty: 0.50`

含义：AMP 负责“动作像一个可用 gait”，task reward 负责“命令跟踪和任务达成”。当前 gait period 保留较慢基础节律，但随速度提高更快缩短周期，减少高速大步幅导致的后腿拖滑。

## 策略网络与观测契约

部署侧 actor 输入：

- body dim：`53`
- history len：`25`
- tick dim：`54`
- history flat dim：`1350`
- terrain latent z dim：`32`
- actor input dim：`85 = 53 + 32`

训练侧额外信息：

- privileged dim：`197`
- aux label dim：`22`
- policy tensor dim：`1622`

模型：

- policy class：`TerrainPerceiverPolicy`
- actor hidden：`[1024, 512]`
- terrain perceiver history：`25 x 54`
- terrain perceiver aux ramp steps：`50000`

关键点：部署策略不直接依赖 privileged terrain truth，而是通过历史 proprioception 和 terrain perceiver latent 学到地形相关信息。

## PPO / skrl 训练参数

核心训练参数：

- rollouts：`32`
- learning epochs：`4`
- mini batches：`16`
- learning rate：`1.0e-04`
- KL threshold：`0.016`
- min lr：`3.0e-05`
- entropy loss scale：`0.02`
- value loss scale：`2.5`
- checkpoint interval：`2000`
- total timesteps：`1500000`

这里的重点是防止学习率在早期 KL 波动下塌缩到接近 0。`min_lr` 和较宽 KL threshold 是为了避免 5k 到 10k 后学习冻结。

## 与 16000 轮失败 run 的关系

之前手动停掉的 16000 轮失败 run 不是这套 final payload。记录显示它使用的是旧 payload：

`/root/gpufree-data/training_payloads/taili_blind_runtime_20260708_165206`

旧配置哈希不是当前 final 哈希。因此，不能把那个 run 的“走都走不了”直接归因到当前 final 策略。它能作为反例说明：旧策略组合中地形/DR/奖励门槛和 gait 稳定性存在冲突，但不是当前单一 final 运行包的验证结果。

## 当前启动语义

控制台“启动全新训练”：

- 创建新 run。
- 不传 checkpoint。
- 使用后端解析出的最新 payload root。
- 若当前远端 latest 仍是 `taili_blind_runtime_final_single_source_20260708_204133`，则运行 final。

控制台“从检查点继续”：

- 创建新的 resume run。
- 自动解析最新可用 checkpoint。
- 同样依赖后端 latest payload root。

后端远端配置来源不是单独只看 `config/ssh.json`，而是：

`effective_remote_config(get_settings())`

也就是项目 SSH 配置加控制台本地 profile override 后的有效配置。修改 SSH 配置后，已经运行中的后端进程通常需要重启，才能确保内存中的 settings 和连接冷却状态全部刷新。

## 本地验证结果

本次文档生成前已验证：

```powershell
python -m autotuner.training_payloads.taili_blind_runtime.payload_manifest
```

结果：

```text
runtime_package=taili_blind_runtime
files=80 generated=1
errors=0 warnings=0
```

编译验证：

```powershell
python -m compileall -q autotuner\blind_locomotion autotuner\taili_core autotuner\training_payloads\taili_blind_runtime
```

结果：通过。

单测验证：

```powershell
pytest tests\autotuner\taili_core tests\autotuner\blind_locomotion\test_curriculum_config.py -q --basetemp .pytest_tmp_single_final
```

结果：`25 passed`

payload 打包验证：

```powershell
python -m autotuner.training_payloads.taili_blind_runtime.build_payload --out $env:TEMP\taili_payload_verify
```

结果：生成 `files=81` 的 tar.gz 包。

## 当前不能由本地完全证明的部分

本地 Windows 环境可以证明：

- 策略 YAML 是唯一源。
- payload manifest 无缺失、无禁止依赖。
- Python 文件能编译。
- 核心配置/奖励单测通过。
- payload 能打包。

本地 Windows 不能完全证明：

- IsaacLab / PhysX / CUDA 环境能实际跑完整训练。
- 远端最新 payload 目录没有被外部手动改动。
- 远端训练启动时没有被用户传入额外环境变量覆盖。

这些需要在远端用 run 目录里的 `run.json`、`effective_config.yaml`、`taili_blind_config.yaml` 和日志交叉确认。判断“实际跑的是不是 final”，最可靠证据是远端新 run 目录中 `taili_blind_config.yaml` 的 SHA256 是否等于：

`80AA8A4178D81366487319B52D9E647ABB2DF8751214266626DFE7FEF539ECE7`

## 远端核验命令

在远端执行：

```bash
ls -1td /root/gpufree-data/training_payloads/taili_blind_runtime_* | head -1
```

应指向：

```text
/root/gpufree-data/training_payloads/taili_blind_runtime_final_single_source_20260708_204133
```

核验 payload 内配置：

```bash
sha256sum /root/gpufree-data/training_payloads/taili_blind_runtime_final_single_source_20260708_204133/taili_blind_runtime/taili_blind_config.yaml
```

应得到：

```text
80aa8a4178d81366487319b52d9e647abb2df8751214266626dfe7fef539ece7
```

核验某个新 run 实际生效配置：

```bash
sha256sum /root/gpufree-data/taili_runs/<run_id>/taili_blind_config.yaml
sha256sum /root/gpufree-data/taili_runs/<run_id>/effective_config.yaml
cat /root/gpufree-data/taili_runs/<run_id>/run.json
```

`taili_blind_config.yaml` 的哈希必须等于 final 哈希；`effective_config.yaml` 可以不同，因为它包含 run id、路径、total steps 等运行时覆盖。

