# 控制核心模块索引

`autotuner/control/` 是与机器人产品和仿真器无关的传统控制核心。模块只通过
`contracts.py` 中的稳定数据结构交换状态、计划、命令和诊断；产品资产与后端
适配位于 `products/taili/traditional_control/`。

## 模块职责

| 文件 | 主要职责 | 关键输出 |
| --- | --- | --- |
| `contracts.py` | 校验控制周期的数据形状、坐标系和有限性 | `RobotState`、`WholeBodyDynamics`、`BodyTarget`、`FootPlan`、`JointCommand` |
| `mpc.py` | 有限时域质心模型和身体参考 | `MpcOutput` / `BodyTarget` |
| `gait.py` | 连续相位、摆腿轨迹、落脚点和接触意图 | `FootPlan` |
| `terrain.py` | 平地/楼梯高度查询与接触支撑面估计 | `TerrainModel`、`ContactTerrainEstimator` |
| `support.py` | 支撑多边形和几何裕度计算 | `SupportGeometry` |
| `wbc.py` | 动力学、接触、摩擦、关节和力矩约束下的 QP | `WholeBodySolution`、`WholeBodyDiagnostics` |
| `servo.py` | 高频阻抗插值和力矩限幅 | 后端可执行的 `JointCommand` 力矩 |
| `controller.py` | 编排一次控制周期和失稳时的显式刹停目标 | `ControlStep` |
| `backends.py` | 后端协议，不实现具体仿真器 | `SimulatorBackend` |
| `errors.py` | 结构化控制失败，不允许静默回退 | `ControlError`、`ControlSolveError` |
| `trace.py` | 逐周期状态/计划/结果/失败回放记录 | JSONL trace 行 |
| `manifest.py` | 稳定 bundle 和运行记录的内容摘要 | artifact/run manifest |

## 一次调用的边界

`MpcWbcController.step(state, command)` 只编排控制核心：接触估计、身体目标、
连续步态、接触准入和 WBC。它不推进物理时间，也不直接读写 MuJoCo、IsaacLab、
文件、SSH 或控制台。后端负责 `read_state()`、`write_command()` 和物理子步；
评测器负责循环、指标和回放。

## 维护规则

- 修改接口时先更新 `contracts.py`、对应测试和 `docs/traditional_control/architecture.md`，再检查所有产品后端调用方。
- 修改控制律时保持产品无关；Taili 的关节顺序、URDF、增益和场景只在产品适配/配置层维护。
- 不把历史 trace、日志、缓存或 bundle 解压目录放进本目录；它们必须留在运行产物目录。
- `ControlError` 是可审查的失败边界。求解失败必须保留结构化错误和同步状态，不能回退到另一套隐含控制器。
- 本索引是阅读和维护入口，不是控制器的第二份实现；实现细节仍以对应源码和测试为准。
