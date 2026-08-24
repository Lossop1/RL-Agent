# 工程边界与维护规则

## 依赖方向

```text
autotuner/control/contracts.py
        ↑
autotuner/control/{mpc,gait,terrain,wbc,controller}.py
        ↑
products/taili/traditional_control/{profile,kinematics,config,backends}
        ↑
tools/traditional_control/*
```

控制核心只能依赖标准数值库和自身契约。它不能导入 `products.taili`、
RL 环境、策略网络、FastAPI、SSH 或输出目录。Taili profile 只负责资产
和产品参数；MuJoCo/IsaacLab adapter 只负责状态读取和命令写入；评测只
消费统一 `RobotState`、`JointCommand` 和回放记录。

## 单个控制周期

1. 后端读取一个同步的 `RobotState`。
2. `CentroidalMpc` 根据身体状态、命令、名义地形和约束生成第一步身体目标。
3. 连续相位步态生成器生成四足足端目标；不按楼梯高度复制状态机。
4. 接触估计器用接触时间、法向力和足端高度更新支撑面估计。
5. WBC 用足端任务、姿态/高度任务和关节限制求解关节速度/位置。
6. 阻抗层限幅并产生 `JointCommand`；后端负责写入仿真执行器。

任何一步的输入非法、数值非有限或求解失败都必须显式报错并由运行器记录，
不能静默切换到 RL actor 或另一套隐含参数。

## 配置优先级

`config/traditional_control/taili_nominal.yaml` 是名义控制配置的唯一源文件。
运行器可以通过显式命令行选择场景，但不能通过环境变量悄悄覆盖控制参数。
URDF 是几何、质量和关节限位的权威来源；配置只提供控制器增益、MPC 权重、
步态和场景参数。两者的摘要都必须进入 manifest。

## 版本与产物

源码、配置、测试和文档进入 Git。运行目录、缓存、编译中间文件、失败回放
和临时 payload 只放在工作区 `output/traditional_control/` 或 `.runtime/`，
并由清理步骤删除。稳定的 bundle 使用内容摘要命名，manifest 记录：

- Git revision；
- 每个源码、配置和 URDF 文件的 SHA-256；
- 控制器/场景 schema 版本；
- 依赖和运行时版本；
- 生成的 bundle 摘要。

远程只接受带 manifest 的 bundle，运行在独立目录和 session；远程目录不是
源码，也不能覆盖当前 RL 训练。

Git 与中间产物的完整纳管、归档、清理和提交前检查规则见
[`maintenance.md`](maintenance.md)。架构变更必须同时更新该规则涉及的目录
边界和可重建验证；只有“代码能运行”而不能说明产物来源、生命周期和回滚方式，
不算完成工程整理。
