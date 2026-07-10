# IsaacLab 四足策略物理诊断工具

`IsaacLabQuadDiag` 用于记录和分析 Isaac Lab 四足运动策略的真实仿真行为。

它关注的是：

- 机器人是否真正按照命令运动
- 命令出现后是否发生姿态崩溃、机身下降或 reset
- 速度来自受控步态，还是来自倒地、滑动、弹飞
- 四足接触、离地、落地、打滑和冲击情况
- 动作、关节、力矩和执行器负载
- 不同地形、域随机化和外力扰动下的行为变化

默认报告只陈述观测事实，不自动给出“通过/不通过”。只有显式使用
`ilqd-check` 并提供外部目标文件时，才会生成验收判断。

## 1. 已验证状态

当前版本已经在远程 RTX 4090 + Isaac Lab 环境中验证：

- Taili SKRL 检查点能够正确加载
- 终止前姿态不会被自动 reset 后的新状态覆盖
- raw progress 与稳定姿态下的有效 progress 分开统计
- reset 后的新 episode 不会被误算成新的命令尝试
- flat、slope、rough、stairs 的实际地形类型和高度扫描可记录
- DR0、DR3 的质量、摩擦、CoM、刚度和阻尼可从运行时读取
- push 不只记录配置，实际根速度变化也能被观测
- 多地形、多 DR case 使用独立 Isaac 子进程运行并自动合并
- 本地和远程测试均为 `13 passed`

## 2. 目录结构

```text
isaaclab_quad_diag_observation/
├─ isaaclab_quad_diag/
│  ├─ record.py          仿真运行与逐帧记录
│  ├─ metrics.py         离线指标计算与报告生成
│  ├─ events.py          姿态、reset、冲击、打滑等事件提取
│  ├─ notes.py           按优先级整理观测提示
│  ├─ slices.py          稳定姿态、命令稳定后、事件前后等切片
│  ├─ check.py           可选的外部目标验收
│  └─ schema.py          字段和诊断阈值
├─ configs/
│  └─ diagnostic_thresholds.yaml
├─ specs/
│  └─ taili.yaml         Taili 关节和足端命名
├─ suites/               各类诊断方案
├─ targets/              可选的外部验收目标
├─ tests/
├─ pyproject.toml
└─ README.md
```

本地代码位置：

```text
D:\RL-Dog\hand\demo3 -副本 副本 cli进发\selected\tools\isaaclab_quad_diag_observation
```

远程代码位置：

```text
/root/gpufree-data/tools/isaaclab_quad_diag_v051
```

## 3. 安装

进入工具目录后执行：

```bash
pip install -e .
```

安装测试依赖：

```bash
pip install -e '.[test]'
```

安装 Parquet 和绘图依赖：

```bash
pip install -e '.[parquet,plots,test]'
```

远程机应使用 Isaac Lab 所在的 Python 环境，例如：

```bash
/opt/conda/envs/isaaclab/bin/python -m pip install -e \
  /root/gpufree-data/tools/isaaclab_quad_diag_v051
```

## 4. 最快使用方法

完整流程分为两步：

1. 在 Isaac Lab 中运行策略并生成原始记录。
2. 对 `record.csv` 做离线分析并生成报告。

### 4.1 运行短探针

先用短探针确认任务、检查点、关节名称和记录接口都能工作：

```bash
ilqd-record \
  --task RobotLab-Isaac-Taili-AMP-Teacher-Direct-v0 \
  --robot-spec specs/taili.yaml \
  --policy-backend skrl \
  --checkpoint /path/to/agent.pt \
  --suite suites/remote_probe.yaml \
  --num-envs 1 \
  --out runs/remote_probe \
  --headless
```

任务注册名中的 `Teacher` 是当前代码保留的历史注册标识，仅用于找到正确
Isaac Lab task 和 SKRL 配置，不表示诊断采用 teacher/student 结构。

正常完成后，`runs/remote_probe/record_meta.json` 中应看到：

```json
{
  "status": "complete"
}
```

### 4.2 生成离线报告

```bash
ilqd-metrics \
  runs/remote_probe/record.csv \
  --out runs/remote_probe/metrics
```

优先阅读：

```text
runs/remote_probe/metrics/summary.md
```

需要程序读取全部数值时，使用：

```text
runs/remote_probe/metrics/metrics.json
```

## 5. 远程机完整命令示例

```bash
cd /root/gpufree-data/robot_lab

export PYTHONPATH=/root/gpufree-data/tools/isaaclab_quad_diag_v051:$PYTHONPATH

/opt/conda/envs/isaaclab/bin/python \
  -c 'from isaaclab_quad_diag.record import main; main()' \
  --task RobotLab-Isaac-Taili-AMP-Teacher-Direct-v0 \
  --robot-spec /root/gpufree-data/tools/isaaclab_quad_diag_v051/specs/taili.yaml \
  --policy-backend skrl \
  --checkpoint /root/gpufree-data/robot_lab/logs/skrl/你的运行目录/checkpoints/agent.pt \
  --suite /root/gpufree-data/tools/isaaclab_quad_diag_v051/suites/smoke.yaml \
  --num-envs 4 \
  --out /root/gpufree-data/diag_runs/my_diag \
  --headless
```

生成报告：

```bash
cd /root/gpufree-data/tools/isaaclab_quad_diag_v051

/opt/conda/envs/isaaclab/bin/python \
  -c "from isaaclab_quad_diag.metrics import main; main()" \
  /root/gpufree-data/diag_runs/my_diag/record.csv \
  --out /root/gpufree-data/diag_runs/my_diag/metrics
```

如果已经执行 `pip install -e .`，也可以直接使用 `ilqd-record` 和
`ilqd-metrics`。

## 6. 应该选择哪个 suite

### 6.1 快速连通性检查

```text
suites/remote_probe.yaml
```

用途：

- 检查任务能否创建
- 检查 checkpoint 能否加载
- 检查 CSV 是否能写出
- 检查运行时字段是否存在

它不是正式策略评价。

### 6.2 基础平地检查

```text
suites/smoke.yaml
```

先站立，再给前进命令。适合检查策略是否具备最基本的稳定与跟踪能力。

### 6.3 四方向诊断

四个方向必须分别运行：

```text
suites/direction_forward.yaml
suites/direction_backward.yaml
suites/direction_lateral.yaml
suites/direction_yaw.yaml
```

每个 suite 都采用：

```text
独立创建环境
-> 记录 stand 基线
-> 不 reset
-> 切换到一个目标方向
-> 记录命令触发后的行为
```

这样可以区分：

- 初始姿态本身不稳定
- 命令切换触发崩溃
- 上一个方向留下的物理状态污染下一个方向

不要在同一个环境实例中依次 reset 测试全部方向。该做法会让不同方向互相
污染，并且某些 Isaac task 的连续 reset 路径可能卡住。

### 6.4 地形检查

短测试：

```text
suites/terrain_probe.yaml
```

较长测试：

```text
suites/full_observation.yaml
```

当前支持的请求名称：

```text
flat
slope_up
slope_down
rough
boxes
stairs_up
stairs_down
```

`stairs_up` 和 `stairs_down` 使用同一个真实 `stairs` terrain。上楼还是下楼
应根据机器人相对楼梯方向和实际运动判断，不能只根据请求字符串判断。

### 6.5 域随机化检查

短测试：

```text
suites/dr_probe.yaml
```

较长测试：

```text
suites/dr_observation.yaml
```

支持 DR level：

```text
0  无随机化
1  基础随机化
2  中等随机化
3  完整随机化
```

报告读取的是运行时实际值，不是简单复制 YAML 请求值。Taili 当前可记录：

```text
dr_mass
dr_friction
dr_com_x/y/z
dr_stiffness_scale
dr_damping_scale
dr_latency
```

Taili 的 `dr_latency` 当前是固定的一步动作延迟：

```text
1 policy step = 0.02 s
```

因此 DR0 和 DR3 都显示 `0.02 s` 是策略配置本身的行为，不是记录错误。

### 6.6 外力扰动检查

短测试：

```text
suites/push_probe.yaml
```

较长测试：

```text
suites/push_observation.yaml
```

push 的 `vector` 当前表示直接施加到根速度的 `delta-v`，单位为 `m/s`，不是
力或冲量：

```yaml
pushes:
  enabled: true
  events:
    - segment: 0
      time: 0.5
      vector: [0.0, 0.5, 0.0]
```

上例表示在第 0 个命令段开始后 `0.5 s`，给所有环境增加世界坐标系 Y 方向
`0.5 m/s` 的速度。

## 7. suite 配置说明

一个完整 suite 示例：

```yaml
name: example
num_envs: 4

reset_policy: per_case
reset_initialization: command_start

terrains:
  - type: flat
    level: 0
  - type: stairs_up
    level: 5

dr_cases:
  - level: 0
  - level: 3

commands:
  - mode: stand
    vx: 0.0
    vy: 0.0
    wz: 0.0
    duration: 1.0
  - mode: forward
    vx: 0.5
    vy: 0.0
    wz: 0.0
    duration: 3.0

pushes:
  enabled: false
```

### 7.1 `num_envs`

并行环境数量。命令行的 `--num-envs` 会覆盖 suite 中的值。

短探针建议：

```text
1 到 4
```

正式统计可以增加，但 CSV 体积和运行时间会同步增加。

### 7.2 `reset_policy`

推荐：

```text
per_case
```

含义是同一个 terrain/DR case 中的多个命令连续执行，不在每个命令前 reset。

`per_segment` 会在每个命令段前 reset。它只适合已经确认连续 reset 安全的
task，不能作为四方向诊断的默认方式。

### 7.3 `reset_initialization`

推荐：

```text
command_start
```

它会让 reset 时的参考姿态：

- 与当前诊断命令匹配
- 从匹配参考 clip 的起点开始

这样可以避免环境先随机采样一个无关命令和随机参考姿态，再突然切换到测试
命令。

### 7.4 `terrains`

每项包含：

```yaml
- type: rough
  level: 5
```

记录器会分别保存：

```text
terrain_type_requested  请求类型
terrain_type            运行时实际类型
terrain_level           运行时实际等级
```

### 7.5 `dr_cases`

每项至少包含 level：

```yaml
- level: 3
```

同样会分别保存请求 level 和运行时实际 level。

### 7.6 `commands`

命令单位：

```text
vx: m/s
vy: m/s
wz: rad/s
duration: s
```

建议 `mode` 与命令语义一致：

```text
stand
forward
backward
lateral
yaw
```

### 7.7 多 case 执行

terrain 数量乘以 DR case 数量就是总 case 数：

```text
4 terrains × 2 DR levels = 8 cases
```

每个 case 都在独立 Isaac 子进程中运行。根目录自动合并结果，同时保留：

```text
cases/case_000/
cases/case_001/
...
```

这样可以避免在一次 Isaac App 生命周期中关闭环境后重新创建环境。

## 8. 命令行参数

### `ilqd-record`

| 参数 | 含义 |
|---|---|
| `--task` | Isaac Lab task 注册名 |
| `--robot-spec` | 机器人关节和足端命名文件 |
| `--policy-backend` | `skrl` 或 `none` |
| `--checkpoint` | SKRL checkpoint 路径 |
| `--suite` | 诊断 suite YAML |
| `--num-envs` | 覆盖 suite 中的并行环境数 |
| `--out` | 原始结果目录 |
| `--device` | 例如 `cuda:0` |
| `--headless` | 无窗口运行 |

`policy-backend=none` 会输出零动作，只用于检查环境或记录接口，不用于评价
训练后的策略。

### `ilqd-metrics`

```bash
ilqd-metrics RECORD_CSV --out OUTPUT_DIR
```

默认会读取与 `record.csv` 同目录的 `record_meta.json`。

### `ilqd-check`

```bash
ilqd-check \
  --metrics runs/diag/metrics/metrics.json \
  --target targets/example_external_target.yaml \
  --out runs/diag/external_check.md
```

只有这个命令会根据外部目标生成验收结论。

## 9. 原始结果文件

每次记录至少生成三个文件：

```text
record.csv
record_meta.json
record_progress.json
```

### 9.1 `record.csv`

逐帧物理数据。主要字段包括：

```text
case_id / env_id / episode_id / cmd_segment_id
time / step / control_dt / physics_dt

cmd_target_vx/vy/wz
cmd_vx/vy/wz

base position / quaternion
base linear velocity / angular velocity
projected gravity

joint position / velocity / desired position / error
action mean / applied action
applied torque / torque limit / utilization

foot position / velocity / contact force
terrain height / local clearance
touchdown / liftoff / air time / stance time / slip

terrain type / level
DR actual parameters
push event / vector / equivalent delta-v

terminated / truncated / done / reset
terminal capture quality
```

### 9.2 `record_meta.json`

记录：

- task 和 checkpoint
- 原始 suite
- 实际执行的 terrain/DR cases
- case 隔离方式
- 完成状态
- 行数
- 记录期间的兼容性提示

关键字段：

```text
status = running / complete / failed
```

只有 `status=complete` 才表示该次记录正常结束。

### 9.3 `record_progress.json`

用于长任务中查看进度。单 case 通常记录：

```text
completed_segments
rows_written
last_mode
```

多 case 通常记录：

```text
completed_cases
requested_cases
rows_written
status
```

## 10. 离线报告文件

执行 `ilqd-metrics` 后生成：

```text
metrics/
├─ summary.md
├─ metrics.json
├─ event_timeline.json
├─ segment_event_summary.json
├─ notes.json
├─ terrain_breakdown.json
└─ robustness_breakdown.json
```

### `summary.md`

最适合人工阅读的总报告。包含覆盖范围、跟踪、姿态、步态、事件、地形和鲁棒性
摘要。

### `metrics.json`

完整机器可读指标。用于进一步统计、画图或接入自动化分析。

### `event_timeline.json`

按时间列出事件。持续多帧的异常会合并为一个 bout，而不是每帧算一次事件。

例如连续 120 帧大 pitch：

```json
{
  "event_type": "large_pitch_bout",
  "start_time": 0.08,
  "duration_s": 2.4,
  "frame_count": 120,
  "peak_value": 48.5
}
```

### `segment_event_summary.json`

每个 `case + env + command segment` 对应一次命令尝试。即使段内发生 reset 并
进入新 episode，也不会增加命令尝试数量。

主要字段：

```text
first_major_event_time_since_segment_start_s
observed_event_labels
raw_progress_along_command
pose_stable_progress_along_command
pose_stable_pre_first_major_event_progress_along_command
post_first_major_event_progress_along_command
```

### `notes.json`

按优先级列出：

- 数据质量问题
- 物理事件
- 值得优先检查的行为模式

没有请求 push 时，不会误报 push 字段缺失。

### `terrain_breakdown.json`

按地形类型、等级和命令统计：

- 有效 progress
- 姿态
- 接触和打滑
- 力矩和动作
- reset/事件

### `robustness_breakdown.json`

按 DR level、质量、摩擦等区间以及 push 前后窗口统计鲁棒性观测。

## 11. 如何正确解读报告

### 11.1 不要只看速度误差

机器人倒地后滑动，可能仍然产生与命令同方向的速度。因此需要同时看：

```text
raw_progress_along_command
pose_stable_pre_first_major_event_progress_along_command
major_event_attempt_fraction
done_attempt_fraction
```

如果 raw progress 很大，但稳定姿态且重大事件前的 progress 接近零，说明
位移主要来自失稳后的滑动或翻滚，不能解释为有效步态。

### 11.2 看首次重大事件时间

```text
first_major_event_time_since_segment_start_s
```

它表示命令段开始后多久首次出现：

- reset/done
- 大 roll
- 大 pitch
- 机身高度明显下降

力矩触顶仍会作为物理事件记录，但不会单独被当成段级重大姿态事件。

### 11.3 区分命令尝试与 episode 片段

覆盖信息中：

```text
segment_attempts
episode_segment_fragments
```

`segment_attempts` 是实际命令尝试数。

`episode_segment_fragments` 可能更大，因为一次命令尝试中发生 reset 后会产生
新的 episode 片段。

### 11.4 使用正确切片

跟踪指标提供多个切片：

```text
all_frames
command_settled
pose_stable
pose_stable_command_settled
before_first_major_event
pose_stable_before_first_major_event
after_first_major_event
pre_major_event_0p5s
post_major_event_0p5s
```

评价受控运动时，优先看：

```text
pose_stable_command_settled
pose_stable_before_first_major_event
```

## 12. 关键记录语义

### 12.1 终止前状态

记录器在 `env.step()` 前临时挂接 `_reset_idx`，捕获自动 reset 前的最终物理
状态。

正常终止行应为：

```text
terminal_state_available = 1
capture_stage = terminal_pre_reset
```

如果 task 绕过该接口，会明确标记：

```text
terminal_state_available = 0
capture_stage = previous_state_fallback
```

不会把 reset 后的新站立姿态伪装成终止姿态。

### 12.2 请求命令与实际命令

```text
cmd_target_vx/vy/wz  suite 请求值
cmd_vx/vy/wz         环境实际使用值
```

如果环境内部有命令平滑或缓冲，两者在切换阶段可能不同。

### 12.3 策略动作与实际动作

```text
action_mean_*       policy 输出的均值动作
action_applied_*    延迟缓冲后真正施加的动作
```

Taili 的 `action_applied_*` 来自真实 `_delayed_action` 缓冲区。

### 12.4 足端地面高度

Taili 使用 height scanner 命中点，并按足端 XY 查找最近地面高度：

```text
foot_<leg>_terrain_height
foot_<leg>_clearance_local
terrain_height_source
```

可靠来源：

```text
height_scanner_nearest
height_at
get_height
get_terrain_height
```

近似或不可用来源：

```text
env_origin_fallback
unavailable
```

如果只能使用 `env_origin_fallback`，楼梯、粗糙地形和 boxes 上的局部 clearance
不可靠，报告会给出数据质量提示。

### 12.5 touchdown 与 liftoff

```text
touchdown = 上一帧未接触且当前帧接触
liftoff   = 上一帧接触且当前帧未接触
```

初始帧没有可靠的上一帧接触状态，因此不会被错误标记为 touchdown。

## 13. 当前远程结果示例

原始与报告结果位于：

```text
/root/gpufree-data/diag_runs/
```

四方向：

```text
v051_direction_forward/
v051_direction_backward/
v051_direction_lateral/
v051_direction_yaw/
```

地形：

```text
v051_terrain_isolated/
```

域随机化：

```text
v051_dr_isolated/
```

外力扰动：

```text
v051_push_probe/
```

例如前进诊断总报告：

```text
/root/gpufree-data/diag_runs/v051_direction_forward/metrics/summary.md
```

## 14. 测试

本地：

```bash
cd tools/isaaclab_quad_diag_observation
python -m pytest -q
```

远程：

```bash
cd /root/gpufree-data/tools/isaaclab_quad_diag_v051
/opt/conda/envs/isaaclab/bin/python -m pytest -q
```

当前预期：

```text
13 passed
```

测试覆盖：

- 连续异常按 bout 统计，不按帧冒充事件数
- 首次重大事件时间
- raw progress 与稳定姿态有效 progress 分离
- reset 后 episode 不增加命令尝试数
- 未请求 push 时不产生 push 误报
- stand-only 空 gait 切片不会使报告崩溃
- disabled push 配置不会崩溃
- terrain/DR case 不会互相合并
- 初始接触不会误报 touchdown
- reset 前最终状态捕获
- 只加载推理所需 checkpoint 模块

## 15. 已知边界

- 精确地形高度依赖 task 是否暴露 height scanner 或地形查询接口。
- 精确 applied action 依赖 task 是否暴露动作延迟缓冲区。
- DR 只能报告实际能够从 PhysX、执行器或环境配置读取的参数。
- push vector 当前是根速度 `delta-v`，不是标准化冲量测试。
- 默认报告是观测报告，不代表行业验收标准。
- 不同机器人需要单独提供 `robot-spec`，不能直接复用 Taili 的关节和足端名称。

遇到不支持的接口时，工具必须在 `record_meta.json`、`missing_fields` 或
`notes.json` 中明确记录，不允许静默填入伪造数据。
