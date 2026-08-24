# Taili 传统控制工程

这是传统控制代码的总入口。需要快速了解目录职责、阅读顺序、Git 分类和当前
功能边界时，先看 [`PROJECT_MAP.md`](PROJECT_MAP.md)；需要理解控制核心内部模块
时，继续看 [`autotuner/control/README.md`](../../autotuner/control/README.md)。

本目录描述一个与 RL 任务隔离的名义模型控制工程。当前范围只有 Taili
在平地各方向、速度切换、静止，以及参数化楼梯上楼/下楼；不包含域随机化、
复杂地形、真机或 sim2real。

## 控制链

```text
RobotState + BodyCommand
    -> CentroidalMpc
    -> ContinuousTrotGait + TerrainModel + ContactTerrainEstimator
    -> WholeBodyTaskSolver
    -> JointCommand
    -> simulator backend
```

MPC、步态、接触估计和 WBC 分别位于 `autotuner/control/`；Taili 的 URDF、
关节映射、限位和场景参数位于 `products/taili/traditional_control/` 与
`config/traditional_control/`。控制核心不允许导入 RL 环境、策略、控制台、
SSH 或远程路径。

## “成熟方案”的边界

这里的分层遵循四足模型控制的标准结构：有限时域质心 MPC 负责机身状态和
加速度参考，任务空间 WBC 负责足端/姿态任务与关节约束，阻抗层负责执行。
求解器和动力学模型可以在不改变上层契约的前提下替换为锁定版本的外部
Pinocchio/OCS2/Crocoddyl 等成熟实现；不能把外部库复制成第二套隐式源码。

## 可重建产物

```powershell
python tools/traditional_control/build_bundle.py
```

产物写入工作区 `output/traditional_control/archive/`，按内容摘要命名，manifest
记录源码、配置、URDF 和依赖文件的 SHA-256。`output/`、`.runtime/`、缓存和
失败运行目录不进入 Git；只有源码、配置、测试、文档和 manifest 规则进入 Git。

Git 纳管范围、临时产物生命周期、清理边界和远程执行限制见
[`maintenance.md`](maintenance.md)。这些规则是交付和提交的必要条件，不是
对运行器的可选说明。

## 远程边界

本地完成导入、契约、固定回放和名义场景回归后，才把带 manifest 的 bundle
传到远程隔离目录。远程只执行，不维护源码，不覆盖 RL 训练，不直接写入现有
payload 或运行目录。

## 当前阶段状态

本目录和对应源码已经形成可导入、可测试、可构建的工程骨架，但本阶段的目标是
整理与审查，不是宣称动态功能完成。平地全方向和上下楼梯仍必须以当前版本、
固定场景、完整指标和可追溯 manifest 的回放结果验收；历史 trace、单元测试通过
或“存在 MuJoCo 运行器”都不能替代该验收。
