# Taili 盲态运行 Payload

这个目录定义 Taili 盲态 locomotion 的运行 payload。它是构建配方，不是第二套源码树。

部署模型：

1. 按 manifest 构建带时间戳的 tar.gz 包。
2. 上传到远端 `/root/gpufree-data/training_payloads/`。
3. 解压到不可变的带时间戳目录。
4. 把该目录加入 `PYTHONPATH`。
5. `sitecustomize.py` 自动导入 `taili_blind_runtime`，在 IsaacLab `train.py` 解析任务前完成 Gym task 注册。
6. 通过 `python -m taili_blind_runtime.launch_taili_train` 启动训练。

这条链路替代旧的“把文件复制进远端 task-source 目录”的方式。

## 运行包内容

tar.gz 包包含：

- `sitecustomize.py`
- `taili_blind_runtime/__init__.py`
- `taili_blind_runtime/blind_tp_env.py`
- `taili_blind_runtime/blind_tp_env_cfg.py`
- `taili_blind_runtime/taili_blind_env_cfg.py`
- `taili_blind_runtime/taili_amp_env.py`
- `taili_blind_runtime/taili_amp_env_cfg.py`
- `taili_blind_runtime/terrain_perceiver_policy.py`
- `taili_blind_runtime/terrain_perceiver_aux_patch.py`
- `taili_blind_runtime/telemetry_emit.py`
- `taili_blind_runtime/launch_taili_train.py`
- `taili_blind_runtime/train_taili.py`
- `taili_blind_runtime/diagnose_taili.py`
- `taili_blind_runtime/diagnose_taili_cases.py`
- `taili_blind_runtime/taili_blind_config.py`
- `taili_blind_runtime/taili_blind_config.yaml`
- `taili_blind_runtime/multi_motion_loader.py`
- `taili_blind_runtime/motions.py`
- `taili_blind_runtime/parametric_ref.py`
- `taili_blind_runtime/taili_core/*.py`
- `taili_blind_runtime/motions/clips/*.npz`
- `taili_blind_runtime/assets/taili.py`
- `taili_blind_runtime/assets/robots/taili-dog/robot.urdf`
- `taili_blind_runtime/assets/robots/taili-dog/meshes/*.STL`
- 诊断工具和 suites。

剩余外部依赖来自远端 Python、IsaacLab、skrl、PyTorch 和 CUDA 环境。运行包不能依赖远端 RobotLab task 源码目录或 RobotLab Python 模块。

## 运行输出契约

默认情况下，启动器把每次 run 写到数据盘：

`/root/gpufree-data/taili_runs/<run_id>/`

关键产物：

- `run.json`：启动元数据、路径、checkpoint 和命令。
- `taili_blind_config.yaml`：本次 run 使用的策略副本。
- `effective_config.yaml`：应用 total steps、checkpoint interval、写日志间隔等覆盖后的完整配置。
- `agent.skrl.yaml`：生成的 skrl 运行配置，`experiment.directory` 指向 `/root/gpufree-data/taili_runs`，`experiment_name` 是 `<run_id>`。
- `train.log`：语义化 `[TPPATH]`、`[TPSTAT]`、`[TPREW]`、`[TPCURR]` 文本遥测。
- `train.telemetry.jsonl`：控制台和智能体读取的结构化遥测。
- `console.log`：训练进程完整 stdout/stderr。
- `checkpoints/`：skrl checkpoint。
- `events.out.tfevents.*`：skrl 写出的 TensorBoard 事件文件。

判断远端实际生效策略时，以 run 目录里的 `taili_blind_config.yaml` 和 `effective_config.yaml` 为准。
