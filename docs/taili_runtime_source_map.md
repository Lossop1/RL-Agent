# Taili 运行源码映射

生成时间：2026-07-12

本文档记录当前本地训练、诊断和 payload 的真实源码边界。它是整理用文档，不被训练运行读取。

## 总体链路

本地源码不是直接复制到远端任务目录运行，而是通过 payload 打包成自包含运行包。

当前链路：

1. 本地编辑 `autotuner/blind_locomotion/taili_blind_config.yaml` 和相关 Python 源码。
2. `autotuner/training_payloads/taili_blind_runtime/payload_manifest.py` 校验打包边界。
3. `autotuner/training_payloads/taili_blind_runtime/build_payload.py` 生成 `taili_blind_runtime_*.tar.gz`。
4. 控制台或手工流程上传 payload 到远端。
5. 远端把 payload 解压成 `taili_blind_runtime` 包，并把目录加入 `PYTHONPATH`。
6. payload 根目录的 `sitecustomize.py` 自动导入 `taili_blind_runtime`。
7. `taili_blind_runtime.__init__` 注册 Gym task。
8. 训练入口运行 `python -m taili_blind_runtime.launch_taili_train`。
9. 每个 run 目录保存 `taili_blind_config.yaml`、`effective_config.yaml`、`agent.skrl.yaml`、日志和 checkpoint。

## 配置来源

单一可编辑策略配置：

- `autotuner/blind_locomotion/taili_blind_config.yaml`

配置加载与转发：

- `autotuner/blind_locomotion/taili_blind_config.py`

关键函数：

- `load_taili_blind_config`：读取 YAML。
- `reward_config_mapping`：把 YAML 的 `reward` section 映射到 `RewardConfig`。
- `build_skrl_config`：从 YAML 生成 skrl 配置。
- `effective_config_with_overrides`：生成带运行覆盖项的 effective config。
- `apply_env_config_to_cfg`：把 YAML 中的 env、curriculum、DR、terrain、blind_overrides 等字段写入 IsaacLab env cfg。
- `phase_command_spec`：解析当前 phase 使用的命令采样规则。
- `active_command_directions`：解析当前实际采样的方向。
- `phase_progress_directions`：解析哪些方向允许阻塞 phase progress。
- `active_direction_progress`：计算用于阶段门控的最小 progress。

注意：`taili_blind_config.py` 是当前最容易出现“配置写了但没生效”的地方。后续整理应优先审计字段转发表。

## 环境类边界

当前主要环境类：

- `autotuner/blind_locomotion/taili_amp_env.py`
- `autotuner/blind_locomotion/blind_tp_env.py`
- `autotuner/blind_locomotion/taili_amp_env_cfg.py`
- `autotuner/blind_locomotion/blind_tp_env_cfg.py`
- `autotuner/blind_locomotion/taili_blind_env_cfg.py`

当前职责划分：

- `TailiAmpEnv` 负责基础 AMP 环境、课程推进、日志、DR、地形课程、主要 task reward 组装。
- `TailiBlindTPEnv` 继承 `TailiAmpEnv`，负责盲态策略观测、历史观测、terrain perceiver 相关训练张量、分方向 progress 修正和盲态特有奖励/诊断。
- env cfg 文件负责 IsaacLab 环境配置、观测维度、资产、控制器和任务注册需要的静态配置。

后续整理风险：

- 父类和子类都参与奖励、日志和地形逻辑，不能直接合并或删除。
- `blind_tp_env.py` 中的 progress 语义和 `taili_amp_env.py` 中的 phase gate 是一组，整理时必须一起验证。
- 诊断和训练都复用部分环境逻辑，不能只按训练入口判断生效范围。

更详细的方法边界和跨文件耦合点见 `docs/taili_env_boundary.md`。

## 奖励来源

唯一共享奖励实现：

- `autotuner/taili_core/taili_reward.py`

核心对象：

- `RewardConfig`
- `default_reward_cfg`
- `reward_cfg_from_env`
- `stable_motion_gate`
- `compute_reward_components`

当前奖励语义：

- 速度跟踪不应只表示机身速度数值接近命令，还要乘以接触、滑移、支撑、落脚等有效性门控。
- `tracking_lin` 和 `tracking_yaw` 是命令跟踪项。
- `terrain_progress` 是沿线速度命令方向的地形进展，纯 yaw 不拥有线性地形进展。
- `gait_anchor`、`diagonal_contact`、`duty_balance` 约束步态结构。
- `stance_slip`、`landing_impact`、`touchdown_slip` 约束脚滑和脚重。
- `support_integrity`、`base_wxy`、`orient`、`hip_deviation` 约束机身稳定和髋关节姿态。
- `climb`、`terrain_up`、`terrain_down`、`terrain_support_transfer` 等地形项在环境内组装，权重从 YAML 的 `blind_overrides` 进入 env cfg。

注意：`RewardConfig` dataclass 中有默认值，但当前训练应以 YAML 覆盖后的值为准。整理文档和调参时不要只看 dataclass 默认值。

## 地形感知边界

当前策略部署侧不直接读取特权地形真值。

相关文件：

- `autotuner/blind_locomotion/terrain_perceiver_policy.py`
- `autotuner/blind_locomotion/terrain_perceiver_aux_patch.py`
- `autotuner/taili_core/taili_obs.py`
- `autotuner/taili_core/taili_terrain_labels.py`

当前观测契约：

- body dim：`57`
- history len：`25`
- tick dim：`54`
- history flat dim：`1350`
- terrain latent dim：`32`
- actor input dim：`89`
- training-only privileged dim：`197`
- aux label dim：`34`（geom9 + mask9 + risk8 + mask8）
- policy tensor dim：`1638`

语义：

- actor 使用 `body57 + terrain_latent32`；新增字段是上一命令和归一化命令年龄。
- terrain latent 来自历史本体感觉，不是直接把地形标签喂给部署策略。
- risk 后六维只在碰撞或预期触地失败已经触发换层事件后监督方向和先导脚。
- privileged 和 aux label 只应作为训练辅助或诊断证据，不应成为部署输入。

## 训练入口

本地源码：

- `autotuner/blind_locomotion/launch_taili_train.py`
- `autotuner/blind_locomotion/train_taili.py`

payload 内入口：

```bash
python -m taili_blind_runtime.launch_taili_train
```

每次 run 应生成：

- `run.json`
- `taili_blind_config.yaml`
- `effective_config.yaml`
- `agent.skrl.yaml`
- `train.log`
- `train.telemetry.jsonl`
- `console.log`
- `checkpoints/`

判断远端实际跑的策略时，优先看 run 目录里的 `taili_blind_config.yaml` 和 `effective_config.yaml`，不要只看本地文件或旧文档。

## 诊断入口

本地源码：

- `autotuner/blind_locomotion/diagnose_taili.py`
- `autotuner/blind_locomotion/diagnose_taili_cases.py`
- `tools/isaaclab_quad_diag_observation/`

payload 会打包诊断工具和 suites：

- `remote_probe.yaml`
- `direction_forward.yaml`
- `direction_backward.yaml`
- `direction_lateral.yaml`
- `direction_yaw.yaml`
- `terrain_probe.yaml`
- `dr_probe.yaml`
- `push_probe.yaml`

诊断不应改变训练状态。它的用途是验证 checkpoint 在方向、稳定、地形、DR 和特定 case 上的真实表现。

## Payload 打包边界

打包定义：

- `autotuner/training_payloads/taili_blind_runtime/payload_manifest.py`

打包器：

- `autotuner/training_payloads/taili_blind_runtime/build_payload.py`

payload 包名：

- `taili_blind_runtime`

主要打包内容：

- 任务注册与环境：`__init__.py`、`taili_amp_env.py`、`blind_tp_env.py`、env cfg 文件。
- 策略配置：`taili_blind_config.py`、`taili_blind_config.yaml`。
- 奖励与核心工具：`autotuner/taili_core/*.py`。
- terrain perceiver：`terrain_perceiver_policy.py`、`terrain_perceiver_aux_patch.py`。
- 训练遥测：`telemetry_emit.py`、`telemetry_payloads.py`。
- 训练与诊断入口：`launch_taili_train.py`、`train_taili.py`、`diagnose_taili.py`、`diagnose_taili_cases.py`。
- AMP 和参考动作：`motions.py`、`multi_motion_loader.py`、`parametric_ref.py`、`motions/clips/*.npz`。
- 验收工具：`physeval_*`、`acceptance_score.py`、`acceptance_aggregate.py`。
- 诊断工具：`tools/isaaclab_quad_diag_observation/` 的运行所需文件。
- 机器人资源：URDF 和 meshes。

manifest 同时做静态检查：

- 禁止旧 `robot_lab.*` 依赖泄漏。
- 检查策略网络观测维度契约。
- 检查 YAML 中 `style_reward_weight` 合理范围。
- 检查 task 注册入口。
- 检查 robot URDF 和 mesh 路径。

## 不应视为当前策略源的内容

这些内容可能有参考价值，但不能当作当前训练策略依据：

- `strategy_backups/`
- 历史 payload 解压目录。
- 历史 run 目录中的配置副本。
- 过期的 `CURRENT_FINAL_STRATEGY.md`。
- 旧的远端 task-source 覆盖目录。
- 只用于前端、LLM、控制台展示的文件。

## 后续整理检查表

整理前先做：

- 记录当前 YAML 和关键 Python 文件哈希。
- 运行 payload manifest。
- 运行奖励和课程相关测试。

整理时逐项做：

- 注释中文化。
- 删除过期历史 note。
- 标明训练入口、诊断入口和工具入口。
- 清点每个 YAML 字段是否被 `apply_env_config_to_cfg` 转发。
- 清点每个奖励字段是否被 `RewardConfig` 或环境奖励逻辑读取。
- 清点 payload manifest 中每个文件是否仍必要。
- 任何合并文件动作都先建旁路版本，再做等价测试。

整理后验证：

```powershell
python -m autotuner.training_payloads.taili_blind_runtime.payload_manifest
python -m pytest tests/autotuner/blind_locomotion/test_curriculum_config.py tests/autotuner/blind_locomotion/test_telemetry_contract.py tests/autotuner/taili_core/test_taili_reward.py
```

如果 payload 边界变化，还要打包并检查 tar 内容。
