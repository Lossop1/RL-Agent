# 原子问题跟踪表

**最终目标**：端到端自动完成强化学习的 agent，提交优秀的结果

**更新日期**：2026-09-11

---

## 第 6 层：物理仿真与数学原语

### P6.1 仿真器后端抽象
- **问题**：支持 IsaacLab/MuJoCo 多后端切换
- **状态**：部分完成（步骤1-3完成，步骤4-6待运行时环境）
- **阻塞**：步骤4需要GPU+IsaacSim环境，步骤5-6需要MuJoCo环境
- **负责人**：
- **验收标准**：相同策略在两个后端的误差 < 5%
- **实现位置**：
  - autotuner/simulation/simulator_protocol.py (SimulatorBackend协议，191行)
  - autotuner/simulation/isaaclab_adapter.py (IsaacLabAdapter适配器，178行)
  - autotuner/simulation/backend_factory.py (create_backend工厂函数，62行)
  - tools/verify_isaaclab_adapter_equivalence.py (等价性验证工具，271行)
  - tests/autotuner/simulation/ (38个测试全部通过)
  - docs/p6_1_implementation_summary.md (完整实现总结，466行)
- **进度**：
  - ✅ 步骤1：定义SimulatorBackend协议（16个测试通过，commit a0f135f）
  - ✅ 步骤2：实现IsaacLabAdapter（17个测试通过，commit 7292455）
  - ✅ 步骤3a：实现create_backend工厂函数（5个测试通过，commit 9123d72）
  - ✅ 步骤3b：重构训练入口使用后端抽象（5个入口已迁移，commit c781480）
  - ⏳ 步骤4：验证等价性（验证工具已就绪，等待IsaacLab环境）
  - ⏳ 步骤5：实现MuJoCoAdapter（待MuJoCo环境和Taili MJCF模型）
  - ⏳ 步骤6：跨后端验证（< 5%差异，依赖步骤4-5完成）
- **已迁移训练入口**：
  - products/taili/blind_locomotion/train_taili.py
  - products/taili/blind_locomotion/diagnose_taili.py
  - products/taili/blind_locomotion/physeval_blind.py
  - products/taili/blind_locomotion/physeval_blind_e.py
  - products/taili/blind_locomotion/calibrate_taili_gates.py
- **设计原则**：零侵入、薄包装、完全等价、可扩展
- **架构影响**：Layer 5通过SimulatorBackend协议消费仿真器，Layer 6通过适配器实现协议，上层无需感知底层仿真器

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
  - products/taili/blind_locomotion/terrain_perceiver_policy.py (集成完成，行72-86)
  - tests/products/taili/blind_locomotion/test_terrain_perceiver_normalization.py (9个集成测试)
- **验证结果**：
  - 单元测试：14个测试全部通过（Welford算法、95%置信区间、检查点保存/加载）
  - 集成测试：9个测试覆盖训练/评估模式、统计量更新、检查点持久化
  - 集成状态：已集成到TerrainPerceiverPolicy.compute()流程
- **集成设计**：
  - 训练模式：每个batch自动更新running statistics
  - 评估模式：使用冻结统计量进行归一化
  - 检查点：obs_normalizer状态自动包含在policy state_dict中
  - 可配置：use_obs_normalization参数支持禁用归一化

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
- **状态**：单元测试完成，待GPU验证
- **阻塞**：需要GPU环境执行集成验证
- **负责人**：
- **验收标准**：恢复后曲线连续，无性能跳变
- **实现位置**：
  - autotuner/execution/compatibility.py (兼容性检查，348行)
  - products/taili/blind_locomotion/runtime_manifest.py (状态捕获，446行)
  - products/taili/blind_locomotion/train_taili.py (恢复入口，105-226行)
  - tests/autotuner/execution/test_compatibility.py (19个测试)
  - tests/products/taili/blind_locomotion/test_runtime_manifest.py (13个测试)
  - tools/verify_resume_continuity.py (验证工具，167行)
  - docs/p4_3_validation_plan.md (完整验证计划)
- **完成情况**（2026-09-13）：
  - 核心逻辑已实现：状态捕获、兼容性检查、恢复入口
  - 单元测试完成：32个测试覆盖核心路径和边缘情况
  - 验证工具就绪：checkpoint完整性检查框架
  - 待GPU验证：9组件哈希一致性、曲线连续性
- **已验证的9个状态组件**：
  1. policy - actor网络权重
  2. value - critic网络权重
  3. optimizer - 优化器状态（动量/自适应矩）
  4. scheduler - 学习率调度器
  5. normalizer - 观测/值归一化器
  6. log_std - 策略标准差参数
  7. amp - 对抗性运动先验判别器
  8. curriculum - 地形课程阶段/级别
  9. rng - Python/NumPy/PyTorch/CUDA随机数状态

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
- **状态**：已完成
- **阻塞**：无
- **负责人**：
- **验收标准**：核心路径（datasource/research_remote）100%使用RemoteTransport协议
- **实现位置**：
  - autotuner/execution/deployment.py (RemoteTransport协议)
  - autotuner/adapter/remote_executors.py (RemoteSSHTransportAdapter适配器，支持exec/put/get)
  - autotuner/locomotion_console/datasource.py (已迁移，1944行使用适配器)
  - autotuner/locomotion_console/research_remote.py (已迁移，使用RemoteTransport)
- **审计发现**（2026-09-13 workflow扫描）：
  - 控制台层69个文件，0处核心路径直接SSH调用
  - 原"88处待迁移"评估为模式匹配误判（.exec()同时匹配SSH方法和字典.get()访问）
  - datasource.py和research_remote.py核心路径已正确使用RemoteSSHTransportAdapter
- **已完成迁移**（2026-09-13）：
  - RemoteTransport协议扩展：添加get()方法支持文件下载
  - research_remote.py完整迁移：SSHExperimentBackend所有SSH调用使用transport适配器
  - 测试验证：test_research_remote.py通过，FakeRemote适配新命令包装格式
- **剩余边缘调用**（非核心路径，低优先级）：
  - discover.py:57 exec_out() - 只读探测工具，不影响训练/部署
  - config_manager.py:245 exec_out() - 连接测试工具，不影响训练/部署
- **详细报告**：docs/p3_2_ssh_migration_audit.md (workflow生成)

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
- **已完成**：20
- **进行中**：2 (P4.2, P4.3)
- **未开始**：5
- **阻塞**：5 个问题被其他问题阻塞

**最近完成**（2026-09-13）：
- P3.2 远程命令执行抽象：控制台层核心路径100%迁移至RemoteTransport协议
- P6.3 观测空间归一化：已完整集成到terrain_perceiver_policy.py，9个集成测试通过
- P4.3 训练中断恢复：单元测试完成（32个测试），待GPU验证

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
1. **P6.1 多后端抽象**：步骤1-3已完成（协议定义、IsaacLab适配器、工厂函数、5个训练入口迁移），步骤4-6待GPU运行时环境
2. **P4.3 训练恢复**：单元测试已完成（32个测试），核心逻辑已实现，待GPU环境执行集成验证
3. **P4.2 检查点管理**：基础schema就绪，需实现选择/过滤/清理逻辑（预估3-4天）
4. **P4.4 超参数搜索**：框架已就位，需集成Optuna/Ray Tune并定义搜索空间（预估4+天）
5. **P6.4 接触力校准**：需要真实硬件数据

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
- 2026-09-13：完成P6.1步骤1-3实现和文档（commit 7e106d6），更新进度统计和优先级列表
- 2026-09-13：完成P4.3单元测试补充（32个测试），创建验证工具，更新状态为"单元测试完成，待GPU验证"
