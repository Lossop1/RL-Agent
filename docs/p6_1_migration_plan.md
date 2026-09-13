# P6.1 仿真器后端抽象迁移计划

**目标**：支持IsaacLab/MuJoCo多后端切换，验收标准：相同策略在两个后端的误差 < 5%

**设计原则**：
- 零侵入：不修改现有TailiAmpEnv实现
- 薄包装：适配器仅实现协议到环境方法的映射
- 完全等价：保证行为与直接使用原始环境一致

---

## 迁移步骤

### ✅ 步骤1：定义SimulatorBackend协议（已完成）
**文件**：`autotuner/simulation/simulator_protocol.py`
**状态**：已实现并测试通过（16个单元测试）

**协议定义**：
- 属性：`num_envs`, `device`
- 方法：`reset()`, `step()`, `close()`, `get_observations()`, `render()`
- 验证函数：`validate_backend()`

**测试覆盖**：
- 完整协议实现验证
- 不完整实现检测
- 属性缺失检测
- 方法不可调用检测
- 类型提示正确性

### ✅ 步骤2：实现IsaacLabAdapter（已完成）
**文件**：`autotuner/simulation/isaaclab_adapter.py`
**状态**：已实现并测试通过（17个单元测试）

**实现要点**：
- 包装任何DirectRLEnv子类
- terminated和truncated合并为dones
- 可选导入模式：IsaacLab未安装时优雅降级
- 鸭子类型支持测试mock对象

**测试覆盖**：
- 协议符合性
- 初始化和类型检查
- reset/step/get_observations/render方法
- 与原始环境等价性验证

**提交记录**：
```
commit 7292455
feat(仿真): 实现P6.1步骤1-2仿真器后端协议和IsaacLab适配器
```

### ✅ 步骤3：重构训练入口使用后端抽象（已完成）

**3a. 后端工厂函数**：
- 文件：`autotuner/simulation/backend_factory.py`
- 状态：已实现并测试通过（5个单元测试）
- 提交：`commit 9123d72`

**3b. 训练入口迁移**：
- 已修改5个文件，统一使用`create_backend()`：
  1. `train_taili.py` - 训练入口
  2. `diagnose_taili.py` - 诊断运行器
  3. `physeval_blind.py` - 验收评估
  4. `physeval_blind_e.py` - 鲁棒性评估
  5. `calibrate_taili_gates.py` - 门控校准（前一步已迁移）

**迁移模式**：
```python
# 旧方式
env = gym.make(args.task, cfg=env_cfg, render_mode=None)
env = SkrlVecEnvWrapper(env, ml_framework="torch")

# 新方式
env = create_backend(args.task, env_cfg, backend_type="isaaclab")
env = SkrlVecEnvWrapper(env, ml_framework="torch")
```

**验证结果**：
- 38个simulation层测试全部通过
- 工厂函数支持多后端切换
- SkrlVecEnvWrapper直接接受SimulatorBackend
- 遥测/检查点机制保持不变

**提交记录**：
```
commit c781480
refactor(训练入口): 迁移5个训练入口使用create_backend工厂函数
```

### 步骤4：验证等价性（MAE < 1e-6）

**状态**：验证工具已准备，等待IsaacLab环境可用

**验证工具**：
- 文件：`tools/verify_isaaclab_adapter_equivalence.py` (271行)
- 提交：`commit 48acc4c`

**验证方法**：
- 使用相同种子、相同策略、相同地形配置
- 对比原始环境vs适配器环境：
  - 观测值差异（逐tensor比较）
  - 奖励值差异
  - 终止标志差异
  - 每步仿真状态一致性

**验收标准**：
- 观测MAE < 1e-6
- 奖励MAE < 1e-6
- 终止标志完全一致
- info字典内容完整对应

**执行命令**：
```bash
python tools/verify_isaaclab_adapter_equivalence.py \
  --task RobotLab-Isaac-Taili-AMP-Blind-Direct-v0 \
  --num-steps 100 \
  --seed 42 \
  --output equivalence_report.json
```

**阻塞因素**：需要IsaacLab运行时环境（GPU + IsaacSim）

### 步骤5：实现MuJoCoAdapter

**文件**：`autotuner/simulation/mujoco_adapter.py`
**依赖**：MuJoCo环境实现（待开发）

**实现要点**：
- 包装MuJoCo环境实现SimulatorBackend协议
- 观测字典格式与IsaacLab保持一致
- 处理MuJoCo特有的状态查询接口
- 支持可选渲染功能

**测试要求**：
- 完整单元测试套件（参考test_isaaclab_adapter.py）
- 协议符合性验证
- 与IsaacLab观测空间维度一致性检查

### 步骤6：跨后端验证（< 5%差异）

**验证方法**：
- 使用相同策略权重在两个后端运行
- 对比训练性能指标：
  - 平均回报差异 < 5%
  - 成功率差异 < 5%
  - 运动质量指标差异 < 10%

**验证工具**：
- 创建`tools/cross_backend_validation.py`
- 运行10个episode，收集统计量
- 生成跨后端对比报告

---

## 当前进度

**已完成**：
- ✅ 步骤1：SimulatorBackend协议定义（16个测试通过）
- ✅ 步骤2：IsaacLabAdapter实现（17个测试通过）

**进行中**：
- 🔄 步骤3：重构训练入口
  - 下一步：创建`create_backend()`工厂函数
  - 下一步：实现`SkrlBackendWrapper`

**待开始**：
- ⏳ 步骤4：等价性验证
- ⏳ 步骤5：MuJoCoAdapter实现
- ⏳ 步骤6：跨后端验证

---

## 架构决策记录

### ADR-1：gym.make()作为内部实现
**决策**：保留gym.make()作为IsaacLab环境创建的内部实现，不直接暴露给上层。

**理由**：
- gym.make()是IsaacLab的标准入口，修改会破坏注册机制
- 上层代码通过SimulatorBackend协议消费，不需要知道gym细节
- 方便未来切换到其他环境创建方式（如直接实例化）

### ADR-2：SkrlVecEnvWrapper保持为适配层
**决策**：创建SkrlBackendWrapper包装SimulatorBackend，与现有SkrlVecEnvWrapper分离。

**理由**：
- SkrlVecEnvWrapper是isaaclab_rl库的实现，修改需要fork
- 保持对isaaclab_rl的依赖最小化
- 便于未来支持其他训练框架（非skrl）

### ADR-3：观测字典格式不变
**决策**：所有后端必须返回{"policy": tensor, "critic": tensor}格式的观测字典。

**理由**：
- 策略网络已依赖此格式（见taili_obs_spec.py）
- 保持与现有检查点的兼容性
- MuJoCo后端需要构造相同格式的观测

---

## 风险与缓解

### 风险1：性能开销
**描述**：适配器层可能引入额外性能开销。
**缓解**：
- 适配器仅做方法转发，无额外计算
- 使用pytest-benchmark测量开销（< 1%可接受）
- 必要时使用__slots__优化内存布局

### 风险2：SkrlVecEnvWrapper兼容性
**描述**：skrl期望的环境接口可能与SimulatorBackend不完全匹配。
**缓解**：
- 详细阅读skrl源码，理解所有必需接口
- 实现SkrlBackendWrapper时保持API完全兼容
- 添加集成测试验证skrl Runner可正常工作

### 风险3：info字典不一致
**描述**：不同后端的info字典内容可能不同，影响遥测。
**缓解**：
- 定义info字典的标准键集合（必需字段）
- 后端特有字段使用命名空间前缀（如"isaaclab_*", "mujoco_*"）
- 遥测系统只消费标准键，忽略特有键

---

## 测试策略

### 单元测试（已完成）
- ✅ `test_simulator_protocol.py`：协议定义正确性
- ✅ `test_isaaclab_adapter.py`：适配器功能完整性

### 集成测试（待实现）
- `test_backend_factory.py`：工厂函数正确创建后端
- `test_skrl_backend_wrapper.py`：skrl包装器兼容性
- `test_training_entry_integration.py`：训练入口端到端测试

### 等价性测试（待实现）
- `test_isaaclab_equivalence.py`：原始环境vs适配器环境
- `test_cross_backend_equivalence.py`：IsaacLab vs MuJoCo

### 性能测试（待实现）
- `bench_adapter_overhead.py`：测量适配器性能开销
- `bench_backend_throughput.py`：对比不同后端吞吐量

---

## 更新日志

- 2026-09-13：完成步骤1-2，创建迁移计划文档
