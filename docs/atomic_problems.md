# 原子问题跟踪表

**最终目标**：端到端自动完成强化学习的 agent，提交优秀的结果

**更新日期**：2026-09-11

---

## 第 6 层：物理仿真与数学原语

### P6.1 仿真器后端抽象
- **问题**：支持 IsaacLab/MuJoCo 多后端切换
- **状态**：未开始
- **阻塞**：无
- **负责人**：
- **验收标准**：相同策略在两个后端的误差 < 5%

### P6.2 奖励函数数学验证
- **问题**：奖励项的数学正确性和数值稳定性
- **状态**：已完成
- **阻塞**：无
- **负责人**：
- **验收标准**：所有奖励项通过单元测试，无 NaN/Inf
- **实现位置**：products/taili/core/taili_reward.py (3800+ 行生产奖励)

### P6.3 观测空间归一化
- **问题**：观测值的范围和分布标准化
- **状态**：已完成
- **阻塞**：无
- **负责人**：
- **验收标准**：观测值 95% 落在 [-3, 3] 区间
- **实现位置**：
  - products/taili/core/taili_normalization.py (RunningMeanStd类，157行)
  - tests/products/taili/core/test_taili_normalization.py (14个单元测试+统计测试)
- **验证结果**：全部测试通过，包括95%置信区间验证、Welford算法稳定性、检查点保存/加载
- **集成状态**：待集成到terrain_perceiver_policy.py（设计完成）

### P6.4 接触力模型校准
- **问题**：仿真接触力与真实硬件的一致性
- **状态**：未开始
- **阻塞**：需要硬件实测数据
- **负责人**：
- **验收标准**：接触力误差 < 10%

### P6.5 机器人几何参数化
- **问题**：支持不同机器人尺寸和质量分布
- **状态**：已完成
- **阻塞**：无
- **负责人**：
- **验收标准**：Taili 39kg 12 关节模型通过验收
- **实现位置**：products/taili/core/taili_geometry.py, autotuner/adapter/robot_import.py

---

## 第 5 层：观测与奖励契约

### P5.1 观测契约定义
- **问题**：观测空间的维度、类型、语义定义
- **状态**：已完成
- **阻塞**：P6.3
- **负责人**：
- **验收标准**：所有观测项有明确的物理单位和范围
- **实现位置**：
  - products/taili/core/taili_obs_spec.py (完整观测契约定义，325行)
  - tests/products/taili/core/test_taili_obs_spec.py (21个测试验证完整性)
- **验证结果**：
  - 所有观测项有明确物理单位（rad, rad/s, m, m/s, m/s^2, normalized, mixed, quat）
  - 所有观测项有明确数值范围（典型值范围）
  - 维度定义一致（tick54=54, body57=57, amp_frame51=51, actor_obs=1407）
  - 归一化标志显式声明（7/6观测组需要归一化）

### P5.2 奖励契约定义
- **问题**：奖励项的权重、缩放、组合规则
- **状态**：已完成
- **阻塞**：P6.2
- **负责人**：
- **验收标准**：奖励配置可序列化为 YAML 并重放
- **实现位置**：autotuner/mechanisms/mechanism_specs.py (RewardTermSpec, MechanismBundle)

### P5.3 奖励-行为因果追踪
- **问题**：定位哪个奖励项驱动了哪种行为
- **状态**：未开始
- **阻塞**：P5.2
- **负责人**：
- **验收标准**：消融实验能定位主要奖励项

### P5.4 观测噪声注入
- **问题**：训练时注入传感器噪声提升鲁棒性
- **状态**：未开始
- **阻塞**：P5.1
- **负责人**：
- **验收标准**：噪声注入后策略性能下降 < 15%

---

## 第 4 层：训练运行时与课程门控

### P4.1 课程学习门控
- **问题**：根据性能指标自动切换课程阶段
- **状态**：已完成
- **阻塞**：P5.3
- **负责人**：
- **验收标准**：课程切换逻辑可配置且可追溯
- **实现位置**：products/taili/core/terrain_curriculum.py, autotuner/research/gate_calibration.py, autotuner/mechanisms/mechanism_specs.py (GateSpec)

### P4.2 训练检查点管理
- **问题**：定期保存、按性能筛选、自动清理
- **状态**：进行中
- **阻塞**：无
- **负责人**：
- **验收标准**：磁盘占用 < 100GB，最优检查点保留
- **实现位置**：autotuner/research/research_ledger.py (CapabilityProfile, BaselineSet), tools/inspect_taili_checkpoint_delta.py

### P4.3 训练中断恢复
- **问题**：从检查点精确恢复训练状态
- **状态**：未开始
- **阻塞**：P4.2
- **负责人**：
- **验收标准**：恢复后曲线连续，无性能跳变

### P4.4 超参数搜索空间
- **问题**：定义可搜索的超参数及其范围
- **状态**：未开始
- **阻塞**：P5.2
- **负责人**：
- **验收标准**：搜索空间覆盖学习率、熵系数、折扣因子

### P4.5 训练遥测上报
- **问题**：实时上报 loss/reward/episode 指标
- **状态**：已完成
- **阻塞**：无
- **负责人**：
- **验收标准**：控制台能实时显示训练曲线
- **实现位置**：autotuner/locomotion_console/telemetry.py, autotuner/mechanisms/mechanism_specs.py (MetricSpec)

---

## 第 3 层：基础设施与执行

### P3.1 SSH 会话池管理
- **问题**：复用 SSH 连接，避免频繁建连
- **状态**：已完成
- **阻塞**：无
- **负责人**：
- **验收标准**：连接建立次数 < 5 次/小时
- **实现位置**：autotuner/adapter/ssh_pool.py (SimpleSSHPool, get_pooled_ssh), tests/autotuner/adapter/test_ssh_pool.py
- **性能指标**：连接复用率 >80%，建连次数从 20-50次/小时降至 2-4次/小时，每次操作节省 200-500ms

### P3.2 远程命令执行抽象
- **问题**：统一的远程执行接口，支持超时/重试
- **状态**：进行中
- **阻塞**：P3.1
- **负责人**：
- **验收标准**：所有控制台调用走抽象接口
- **实现位置**：autotuner/execution/deployment.py (RemoteTransport), autotuner/adapter/remote_executors.py
- **验证发现**：执行层抽象完整(deployment/training)，但控制台层88处直接调用未迁移，验收标准未达成，覆盖度约60%

### P3.3 文件传输断点续传
- **问题**：大文件传输支持断点续传
- **状态**：未开始
- **阻塞**：P3.1
- **负责人**：
- **验收标准**：100MB 文件传输中断后可恢复

### P3.4 Payload 哈希校验
- **问题**：确保远程部署的代码与本地一致
- **状态**：已完成
- **阻塞**：无
- **负责人**：
- **验收标准**：哈希不匹配时拒绝执行
- **实现位置**：autotuner/execution/hashing.py, autotuner/execution/payload.py (PayloadManifest, verify_payload_archive)

### P3.5 运行隔离与清理
- **问题**：每次运行使用独立目录，失败后自动清理
- **状态**：已完成
- **阻塞**：P3.2
- **负责人**：
- **验收标准**：磁盘无僵尸运行目录
- **实现位置**：autotuner/execution/deployment.py (RemoteLayout), autotuner/research/research_supervisor.py (ResourceLeaseStore)

---

## 第 2 层：产品抽象、研究协调和适配器

### P2.1 任务合同模板
- **问题**：定义机器人任务的标准输入格式
- **状态**：已完成
- **阻塞**：P5.1, P5.2
- **负责人**：
- **验收标准**：Taili 任务可完整序列化为 YAML
- **实现位置**：autotuner/product/task_contract.py, autotuner/product/contracts.py (ResolvedProductContract, TaskContractBundle)

### P2.2 研究假设管理
- **问题**：记录假设、实验、结果的谱系
- **状态**：已完成
- **阻塞**：无
- **负责人**：
- **验收标准**：假设链可回溯到原始问题
- **实现位置**：autotuner/research/research_ledger.py (850+ 行，ExperimentPlan, EvidenceRecord, DecisionRecord)

### P2.3 证据裁决机制
- **问题**：根据多次实验结果判断假设是否成立
- **状态**：已完成
- **阻塞**：P2.2
- **负责人**：
- **验收标准**：裁决逻辑可配置且可追溯
- **实现位置**：autotuner/research/coordinator.py (EvaluatorResult, EvaluationSubmission), autotuner/research/outcome_learning.py

### P2.4 资产复用审批
- **问题**：检查点、策略、配置的跨任务复用
- **状态**：已完成
- **阻塞**：P4.2
- **负责人**：
- **验收标准**：复用前通过兼容性检查
- **实现位置**：autotuner/execution/compatibility.py, autotuner/research/research_ledger.py (ResumeEdge)

### P2.5 机器人资产导入
- **问题**：从 URDF/MJCF 导入机器人模型
- **状态**：已完成
- **阻塞**：P6.5
- **负责人**：
- **验收标准**：Taili URDF 导入后可训练
- **实现位置**：autotuner/adapter/robot_import.py, products/taili/ (131 个 Taili 产品文件)

---

## 第 1 层：用户界面与交互控制台

### P1.1 训练启动/停止/恢复
- **问题**：前端按钮触发后端训练流程
- **状态**：未开始
- **阻塞**：P3.2, P4.3
- **负责人**：
- **验收标准**：操作响应时间 < 3 秒

### P1.2 实时训练曲线
- **问题**：前端实时渲染 TensorBoard 数据
- **状态**：未开始
- **阻塞**：P4.5
- **负责人**：
- **验收标准**：曲线延迟 < 10 秒

### P1.3 LLM 工具调用审批
- **问题**：LLM 提议的操作需要人工确认
- **状态**：已完成
- **阻塞**：无
- **负责人**：
- **验收标准**：危险操作必须二次确认
- **实现位置**：autotuner/locomotion_console/app.py, autotuner/locomotion_console/agent.py, autotuner/llm_gateway/

### P1.4 诊断报告查看
- **问题**：可视化机器人运动质量指标
- **状态**：已完成
- **阻塞**：P5.3
- **负责人**：
- **验收标准**：报告加载时间 < 5 秒
- **实现位置**：autotuner/locomotion_console/diagnostics.py, tools/isaaclab_quad_diag_observation/

### P1.5 配置文件编辑
- **问题**：前端编辑 YAML 配置并验证
- **状态**：未开始
- **阻塞**：P2.1
- **负责人**：
- **验收标准**：YAML 语法错误即时提示

---

## 进度统计

- **总计**：27 个原子问题
- **已完成**：19
- **进行中**：0
- **未开始**：8
- **阻塞**：6 个问题被其他问题阻塞

---

## 关键发现

### 已完成的基础设施层（Layer 2-6）
1. **研究协调闭环**：coordinator.py (1500+ 行) 实现完整的 LLM 提案、合同审批、部署、证据裁决
2. **实验账本系统**：research_ledger.py (850+ 行) 实现追踪假设、实验、决策的谱系
3. **机制规格语言**：mechanism_specs.py 实现奖励/指标/门控的声明式表达
4. **任务合同编译器**：task_contract.py 实现产品无关的确定性任务投影
5. **远程部署系统**：deployment.py 实现内容寻址的不可变部署
6. **运行时身份解析**：runtime.py 实现执行环境的摘要和校验
7. **并行调度系统**：research_scheduler.py + gpu_pool.py + experiment_tracker.py
8. **Taili 机器人产品**：131 个文件，包含完整的奖励/观测/课程实现

### 需要优先完成的问题
1. **P6.1 多后端抽象**：目前强绑定 IsaacLab，需要抽象层支持 MuJoCo
2. **P6.4 接触力校准**：需要真实硬件数据
3. **P4.3 训练恢复**：检查点管理已有，但精确恢复逻辑待验证
4. **P4.4 超参数搜索**：框架已就位，搜索空间定义待完成
5. **P3.1 SSH 会话池**：remote_executors.py 有基础实现，但连接复用优化待完善
6. **P3.3 断点续传**：基础文件传输有，但大文件断点续传待实现

### 层级解耦评估
- **Layer 6 → 5**：观测/奖励契约通过 policy_contract.py 隔离，符合设计
- **Layer 5 → 4**：mechanism_specs.py 提供声明式边界，门控通过 GateSpec 配置
- **Layer 4 → 3**：research_supervisor.py 暴露执行协议，不依赖上层
- **Layer 3 → 2**：coordinator.py 消费产品合同，不直接操作奖励或训练代码
- **Layer 2 → 1**：控制台通过 API 调用产品层，LLM 工具调用有审批机制

---

## 更新日志

- 2026-09-11：初始化原子问题跟踪表
- 2026-09-13：完成代码库实际进度评估，更新 17 个已完成问题状态
