# Taili Traditional Control Adapter

这里是 Taili 产品适配层，不是 RL 任务的别名，也不是 payload 的第二份源码。

- `profile.py`：从权威 URDF 读取关节顺序、限位、质量和足端几何；
- `kinematics.py`：Taili 的 FK/Jacobian；
- `config.py`：组合通用 MPC/WBC 与 Taili profile；
- `backends/`：MuJoCo、IsaacLab 的 I/O 适配；
- `packaging.py`：从 Git 源码生成带摘要的运行 bundle。

控制律位于 `autotuner/control/`。修改 Taili 关节或资产时，不复制控制律；修改
控制律时，不改 RL 环境和 `products/taili/blind_locomotion/`。
