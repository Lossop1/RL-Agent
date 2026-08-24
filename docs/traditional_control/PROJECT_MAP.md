# Taili 传统控制项目地图

这份地图是传统控制代码的阅读入口。它描述当前工作区的模块边界、依赖方向、
运行入口和产物生命周期；它不代表平地或楼梯功能已经通过动态验收。

## 先读什么

按下面的顺序阅读，可以从接口约束逐层走到产品适配和运行器：

1. `docs/traditional_control/README.md`：范围、控制链和远程边界。
2. `docs/traditional_control/architecture.md`：模块依赖和一个控制周期。
3. `autotuner/control/contracts.py`：`RobotState`、`BodyCommand`、`FootPlan`、`JointCommand` 和动力学快照。
4. `autotuner/control/controller.py`：MPC、步态、接触估计和 WBC 的编排顺序。
5. `autotuner/control/README.md` 及对应实现文件：各控制模块的职责和状态。
6. `products/taili/traditional_control/profile.py`、`kinematics.py`、`config.py`：Taili 资产和参数组合。
7. `products/taili/traditional_control/backends/`：MuJoCo/IsaacLab 的状态读取与命令写入。
8. `products/taili/traditional_control/evaluation.py`：固定场景回放和验收指标。
9. `tools/traditional_control/` 与 `tests/`：可重建入口、验收入口和契约测试。
10. `docs/traditional_control/maintenance.md`：提交、归档、清理和远程执行规则。

## 目录职责

| 目录 | 所有权 | 允许依赖 | 不应承担的职责 |
| --- | --- | --- | --- |
| `autotuner/control/` | 机器人无关的控制契约、MPC、连续步态、地形/接触估计、WBC、伺服、诊断 | 标准数值库和本目录的契约 | Taili 资产、RL 环境、策略网络、控制台、SSH、远程路径 |
| `products/taili/traditional_control/` | Taili profile、运动学、配置组装、仿真器适配、打包和评测 | `autotuner/control/`、Taili 资产 | 第二套控制律、RL payload 或远程维护逻辑 |
| `config/traditional_control/` | 版本化 schema、增益、场景和验收阈值 | 无业务代码 | 运行时隐式覆盖、机器私有参数和密钥 |
| `tools/traditional_control/` | 从源码构建 bundle、运行名义 MuJoCo 回放、运行固定验收套件 | 产品适配和稳定入口 | 新的控制实现、临时调参脚本、远程源码维护 |
| `tests/autotuner/control/` | 通用契约和数值组件测试 | 控制核心 | 依赖本地历史 trace 或未声明仿真状态 |
| `tests/products/taili/traditional_control/` | Taili 适配、后端和打包测试 | 产品适配、控制核心 | 用单元测试代替动态功能验收 |
| `docs/traditional_control/` | 架构、维护规则、当前状态和阅读入口 | 工程事实 | 把失败回放或未经复现的结果写成通过结论 |

## 入口与依赖方向

```text
RobotState + BodyCommand
        |
        v
autotuner/control/controller.py
  -> mpc.py
  -> gait.py + terrain.py
  -> wbc.py
  -> servo.py
        |
        v
products/taili/traditional_control/config.py
  -> profile.py + kinematics.py
  -> backends/mujoco.py | backends/isaaclab.py
        |
        v
tools/traditional_control/run_mujoco_nominal.py
tools/traditional_control/run_acceptance.py
```

`autotuner/control/` 是稳定的上层契约和控制核心；Taili 适配只能向下提供
资产、运动学和 I/O。后端不能反向把仿真器细节泄漏到控制核心，评测只能消费
统一的状态、命令、诊断和回放记录。

## 当前工程状态

当前工作区的基线提交为 `a54b1c45e9faffa67fd8b6fb49a14cdb56e5c509`，工作树在
本次整理开始前已经是 dirty：

- 用户已有修改位于 `.gitignore`、`pyproject.toml`、若干 `autotuner/` 模块和
  对应测试；这些修改不属于本次整理，不能用整理结果覆盖或撤销。
- 传统控制源码、配置、测试、工具和现有文档大多是未跟踪文件；它们需要在
  审查后作为一个逻辑单元决定是否纳入提交，而不是按某次运行结果零散提交。
- `autotuner/locomotion_console/task_intake_service.py` 及其接口测试也是未跟踪
  文件，但属于控制台任务接入范围，不应与传统控制 bundle 混在同一提交中。
- `__pycache__`、`.pytest_cache`、`.tmp/`、`.runtime/` 和 `output/` 是本地生成物，
  由忽略规则隔离，不能成为源码或功能证据。

整理完成后，用下面的命令重新确认这张地图没有过期：

```powershell
git status --short --untracked-files=all
git diff --check
python -m compileall -q autotuner/control products/taili/traditional_control tools/traditional_control
```

## 功能状态边界

本阶段只整理工程，不修控制逻辑。现有实现可以被静态导入、契约测试和名义
MuJoCo 运行器调用，但这不等于目标功能已经完成。已知尚未闭合的功能风险包括：

- `+/-0.25 m/s` 横向真实 MuJoCo 闭环仍未形成可接受的双向结果；左向最终翻倒，右向可能出现 WBC `primal infeasible`。
- 当前 MPC 不消费接触可实现性；WBC 求得的实际机身加速度也没有作为下一周期的反馈状态。
- 单侧或对角支撑时仍可能请求 `ay=-5 m/s^2`，实际实现方向相反并形成滚转级联。
- 楼梯场景虽有参数化地形和连续步态入口，但上下楼梯必须以完整动态回放和明确阈值通过为准，不能由代码存在或历史 trace 推断完成。

因此，本地图证明的是“代码在哪里、如何维护和如何验收”，不证明“控制效果达标”。

## 产物边界

稳定输入包括控制核心、Taili 适配、配置、URDF、依赖声明和文档规则；bundle
由 `tools/traditional_control/build_bundle.py` 生成并带有内容摘要 manifest。一次
运行的结果、trace、日志和失败诊断放在：

```text
output/traditional_control/archive/<artifact_digest>/
output/traditional_control/runs/<run_digest>/
.runtime/traditional_control/
```

它们可以作为审查证据，但不属于 Python 源码，也不应直接提交。长期保留的结果
必须带 manifest，能够追溯到 Git revision、输入文件摘要和依赖版本；没有来源或
复现价值的临时产物在验证后按 `maintenance.md` 清理。
