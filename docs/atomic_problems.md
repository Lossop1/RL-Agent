# 原子问题跟踪表

**最终目标**：端到端自动完成强化学习的 agent，提交优秀的结果

**更新日期**：2026-09-16

---

## 第 6 层：物理仿真与数学原语

### P6.1 仿真器后端抽象
- **问题**：支持 IsaacLab/MuJoCo 多后端切换
- **状态**：部分完成
- **子进度**：步骤 1-3 完成，步骤 4-6 待运行时环境（详见下方「进度」）
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
  - ✅ 步骤3b：重构训练入口使用后端抽象（**4** 个入口已迁移，commit c781480）
  - ⏳ 步骤4：验证等价性（验证工具已就绪，等待IsaacLab环境）
  - ⏳ 步骤5：实现MuJoCoAdapter（待MuJoCo环境和Taili MJCF模型）
  - ⏳ 步骤6：跨后端验证（< 5%差异，依赖步骤4-5完成）
- **已迁移训练入口**（4 个，验证方式：`grep -rln create_backend --include=*.py products/ autotuner/`）：
  - products/taili/blind_locomotion/train_taili.py（`SkrlVecEnvWrapper(backend.unwrapped, ...)`，:312-316）
  - products/taili/blind_locomotion/diagnose_taili.py（:1123-1129）
  - products/taili/blind_locomotion/physeval_blind.py（:179-183）
  - products/taili/blind_locomotion/physeval_blind_e.py（:84-88）
- **订正（2026-09-16）**：本节此前写"5个入口已迁移"并列出
  `calibrate_taili_gates.py`，是错的。`git show --stat c781480` 只动了 **4** 个文件，
  而 `calibrate_taili_gates.py` 至今仍是 `gym.make(...)` + `SkrlVecEnvWrapper(env, ...)`
  直接建环境（:368、:377），**从未**用过 `create_backend`。夸大的来源是那次提交的
  信息本身——它的标题就写着"迁移5个训练入口"，表格照抄了提交信息而没有核对 diff。
  教训：状态记的是 **diff 里的事实**，不是 commit message 里的说法。
- **步骤3b 回归点**：抽象层暴露的是 `backend.unwrapped`，不是 backend 本身；
  四个入口都是 `create_backend(...)` 之后再包 `SkrlVecEnvWrapper`。载荷布局下
  `backend_factory` 走 `taili_blind_runtime.taili_sim.*`，源码树走 `autotuner.simulation.*`，
  两条支路都要能 import 到。
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
- **状态**：已完成
- **阻塞**：无
- **负责人**：
- **验收标准**：磁盘占用 < 100GB，最优检查点保留
- **子进度**：源码树、载荷布局、真实训练三层都已跑通并有日志/文件为证（见「远端验证」）。
- **实现位置**：
  - autotuner/product/checkpoint_curator.py (CheckpointRegistry/Selector/Curator/CapabilityPromotionService, 850行, commit cc6c97d)
  - products/taili/blind_locomotion/checkpoint_integration.py (统一接口, 280行)
  - products/taili/blind_locomotion/checkpoint_hook.py (训练流程钩子, 130行)
  - products/taili/blind_locomotion/checkpoint_curator_cli.py (CLI工具, 290行)
  - tests/autotuner/product/ (43个单元测试: registry 10 + selector 12 + curator 13 + promotion 8)
  - tests/products/taili/blind_locomotion/test_checkpoint_integration.py (7个集成测试)
- **验证结果**：源码树 50 个测试通过（另新增钩子回归 9 条、快照 3 条并改写旧用例 1 条、回填 2 条）
- **远端此前为何没生效**（2026-09-16 实测）：`checkpoint_hook` / `checkpoint_integration` /
  `checkpoint_curator` / `research_ledger` 四个模块**一个都不在**载荷清单里。训练入口的
  `except Exception` 把它吞成一行日志，于是"测试全过"和"远端跑起来了"被混为一谈：
  smoke12 日志实录 `[TAILI_TRAIN] checkpoint hook install failed:
  No module named 'taili_blind_runtime.checkpoint_hook'`。
- **补上模块之后仍然全废**：把四个模块补进载荷后，"导入"这一关过了，但**检查点管理在真实
  训练里依旧一条不登记、一行日志不打**。2026-09-16 逐个挖出三个各自独立、都会静默失败的缺陷，
  它们互相掩盖，任何一个单独存在都能让 P4.2 看起来"已完成"：

  1. **判据写错**（`checkpoint_hook.py`）。原先写
     `if not hasattr(env, "_checkpoint_integration")`，而环境在 `__init__` 里就把这个属性
     置成了 `None`（`blind_tp_env.py:179`）。属性存在、值是 None ⇒ `hasattr` 恒为真 ⇒
     集成器永远不建，注册/清理/清单导出全废，且因为走的是"已装过"那一支，
     连 `installed checkpoint management` 那行日志都不打。
     旁证：`fixstep_n4`（576 步跑到 `status: complete`）日志里 `[CheckpointHook]` **0 条**。
     修复：判据改成 `getattr(env, "_checkpoint_integration", None) is None`。
  2. **拦错了方法**（`checkpoint_hook.py`）。钩子包的是 `agent.save`，而 skrl 1.4.3 的训练循环
     调的是 `agent.write_checkpoint(timestep, timesteps)`
     （`skrl/agents/torch/base.py:677` 的 `post_interaction`；`save(path)` 在 :367 还在，
     但训练循环一次都不调它）。于是检查点照写、登记表恒空。
     旁证：`hooksave_n4`（240 步、`TAILI_CHECKPOINT_INTERVAL=200`）磁盘上躺着
     46 MB 的 `agent_200.pt`，而 `total_registered: 0`、清单 76 字节。
     修复：改包 `write_checkpoint`，用"调用前后比目录"拿落盘文件名
     （不自己拼 `agent_<t>.pt`，那个名字随 `checkpoint_store_separately` 变）。
  3. **读了一组不存在的键**（`telemetry_payloads.build_checkpoint_performance_snapshot`）。
     原先读 `reward_payload["mean"]` 和 `health_payload["episode_length_mean"]`，
     这两个键**在真遥测里都不存在**——奖励总数叫 `"total"`（`build_reward_payload`），
     health 段压根没有 episode 长度。于是 `reward_mean` 恒为 0.0、
     `episode_length_mean` 恒为默认的 100.0。危害不只是不好看：
     `checkpoint_curator` 的评分一半权重压在 `reward_mean` 上，
     质量门也拿它比阈值（`>1.5` / `>0.8` / `>0.5`，`episode_length > 100`），
     也就是说**整条择优链路在盯着两个常数看**。
     同一个键名错误还在回填路径里（`CheckpointRegistry.backfill_from_telemetry`），
     外加把 `curriculum["phase"]` 这个显示串 `"phi0"` 直接当数字用。
     真遥测核对（`hookreg_n4` 的 `train.telemetry.jsonl`）：reward 段无 `mean` 有 `total`，
     health 段无 `episode_length_mean`，curriculum 段级别字段叫 `dr_level` 不叫 `level`。

  **为什么测试全绿却全坏**：第 3 条本来有测试，但那条测试**把错误当成了契约**——
  `test_the_checkpoint_snapshot_uses_the_numeric_phase` 传的是
  `reward_payload={"mean": 1.5}` 和 `health_payload={"episode_length_mean": 42.0}`，
  两个现实中不存在的键。测试喂什么它就检查什么，于是永远绿。
  新写的 `test_the_snapshot_reads_the_reward_key_the_builder_actually_emits`
  改成调用**真的** `build_reward_payload` 再取值，不硬编码键名假设。
  这和 P6.1 那条教训是一路的：证据要来自被测对象本身，不要来自关于它的说法。
- **远端验证（2026-09-16，3060 机器）**：
  - 把四个模块补进载荷清单后，在**只有载荷在 PYTHONPATH 上**的子进程里实测：
    `CheckpointIntegration` 建成、`save()` 后 registry 登记 1 条、
    `checkpoint_manifest.json` 导出成功、telemetry 的 `_checkpoint_registry` 建起并登记 1 条。
  - 对照组：把新增的四个模块从解包结果里删掉再跑同一探针，报
    `ModuleNotFoundError: No module named 'taili_blind_runtime.checkpoint_hook'`
    ——与 smoke12 一字不差，证明探针测得的就是当初坏掉的那件事。
  - 容器内复测：容器的 kit python **没有 pydantic**，而 `checkpoint_curator` 原先在模块层
    就 `import research_ledger`（后者 `import pydantic`），把"能力提升用不了"放大成
    "检查点管理全废"。已改为按需加载，见下。
- **验证方式**：
  `python -m products.taili.payload.build_payload --out <dir>` 后用
  `.scratch_verify_pair/probe_p42_payload.py <tar.gz> <workdir>`：它在中立工作目录起子进程
  （cwd 不是仓库根，`autotuner`/`products` 都不可导入，与远端一致），只把载荷根塞进
  `sys.path`，然后 import 钩子、驱动一次保存、读 `registry._registry` 的条数。
  加 `PROBE_BLOCK_PYDANTIC=1` 可模拟训练容器（该环境下若能力提升缺席，应打印
  "跳过能力提升：研究台账不可用"而不是中断）。
  **两个路径参数必须给绝对路径**：子进程的 cwd 是中立的 `neutral/`，
  相对路径会解析到那里去，报出来的却是 `No module named 'taili_blind_runtime'`，
  看着像载荷坏了。2026-09-16 复核时踩过一次。
- **端到端验证（2026-09-16，3060 机器，run_id `hookreal_n4`）**：
  载荷 `taili_blind_runtime_fixstep11_1df8dd02e35a`，4 envs、240 步、
  `TAILI_CHECKPOINT_INTERVAL=200`（默认 2000，步数不够的训练一次都不会 save，
  验证落盘路径必须显式调小）。证据分四处，可独立复核：

  1. 日志有安装与收尾两行（此前一条都没有）：
     ```
     [CheckpointHook] installed checkpoint management
     [CheckpointHook] finalized checkpoint management, promoted 1 checkpoints
     [CheckpointHook] statistics: {'total_registered': 1, 'checkpoint_count': 1, ...}
     ```
  2. 磁盘上 skrl 写出的检查点与登记表并存：
     `run/taili_runs/hookreal_n4/checkpoints/agent_200.pt`（46 MB）与
     `checkpoint_manifest.json`（404 字节，修复前是 76 字节的空清单）。
  3. 清单里的 `reward_mean` 与**同一份遥测**里 step 200 的 `reward.total` 精确相等：
     两边都是 `-6.514138221740723`。这是拿独立来源对出来的，不是自证。
     `episode_length_mean: 406.25`（来自 `episode_length_buf`，不再是假的 100.0）。
  4. `checkpoint_mtime` 与 `os.path.getmtime(agent_200.pt)` 逐位相等
     （`1789567869.6889675`）。此前记的是快照生成时刻，回填时按 mtime 对齐检查点会配错档。

  复核命令：
  ```
  sed -n 's/.*\(statistics:.*\)/\1/p' /home/chuan/robot_lab/logs/hookreal_n4.stdout
  python3 -m json.tool /home/chuan/robot_lab/run/taili_runs/hookreal_n4/checkpoints/checkpoint_manifest.json
  python3 -c "import json,os;m=json.load(open('/home/chuan/robot_lab/run/taili_runs/hookreal_n4/checkpoints/checkpoint_manifest.json'));p,r=next(iter(m['checkpoints'].items()));print(r['checkpoint_mtime'], os.path.getmtime(p))"
  ```
- **端到端验证之二（2026-09-16，3060，run_id `ladder_long_512`）**：上一个用例为了
  让 240 步的训练落盘，显式把 `TAILI_CHECKPOINT_INTERVAL` 调到了 200，走的是**非默认**路径。
  这一档用**默认**间隔（2000，来源 `blind_tp_env.py:3115` 的
  `os.environ.get("TAILI_CHECKPOINT_INTERVAL", "2000")`）跑满 2000 步，即产品实际会走的路径：
  - `status: complete`、`errors: []`；`[CheckpointHook] finalized ... promoted 2 checkpoints`；
    `statistics: {'total_registered': 2, 'checkpoint_count': 2, 'last_cleanup_step': -1,
    'disk_usage_gb': 0.08689182624220848}`。
  - 磁盘上是 `agent_2000.pt` 与 `best_agent.pt`（各 46649299 字节）。
    登记 2 条 = skrl 真写出的 2 个文件，**含 `best_` 这个标签**——这正是"调用前后比目录"
    相对"自己拼 `agent_<t>.pt`"的差别，后者会漏掉 best 那一个。
  - 清单里 `best_agent.pt` 记为 `step: 2000`、`reward_mean: -5.844684600830078`、
    `episode_length_mean: 469.845703125`、`terminal_rate: 0.0`。
- **未验证**：「磁盘占用 < 100GB 触发清理」这条在两轮里都没被触发过——240 步档只落一个
  检查点、2000 步档两个，两轮 `last_cleanup_step` 都是 `-1`，`disk_usage_gb` 0.087。
  清理阈值（90 GB）与保留策略目前只有单元测试覆盖，没有一次真实训练跑到过阈值。
- **研究台账解耦**（2026-09-16）：`research_ledger` 在模块层 `import pydantic`，训练容器里没有。
  现在 `checkpoint_curator` 改为按需加载：registry / 定期清理 / 清单导出都不依赖它，
  只有能力提升在缺台账时抛具名异常 `ResearchLedgerUnavailable`，由 `finalize_training`
  捕获、打印一句说明并继续导出清单，不再连带中断。
- **核心功能**：
  - CheckpointRegistry: 性能快照映射 (0.5×reward + 0.3×stability + 0.1×episode + 0.1×recency)
  - CheckpointSelector: 多维度评分选择
  - CheckpointCurator: 分层保留策略 (top-10 + recent-5 + milestones)
  - CapabilityPromotionService: 能力特征提取（依赖研究台账，训练容器内不可用）

### P4.3 训练中断恢复
- **问题**：从检查点精确恢复训练状态
- **状态**：已完成
- **说明**：2026-09-16 在 3060 上完成端到端实机验证，判据取**权重本身**而非奖励曲线。
  权重确实被加载（逐位相等）、优化器动量确实续用（step 计数器 256→512）、训练确实继续
  （policy 复现误差 0.0011 对 200 步位移 0.6665）。scheduler / curriculum / rng 三项仍为**未验证**。
- **阻塞**：无
- **负责人**：
- **验收标准**：恢复后曲线连续，无性能跳变
  - **2026-09-16 订正**：这条标准**在本尺度上不可测量**，见下面的「为什么不能用曲线判定」。
    实测替代判据：从同一检查点恢复后再跑 N 步，policy 的相对 L2 距离应 **≪ 该 N 步的正常位移**。
    实测分离开销：复现误差 0.0011，同长度正常位移 0.6665，无关参照 1.9693（约 600 倍分离）。
- **验证环境**：3060 机器（chuan@100.124.24.52 / 192.168.111.129），
  nvcr.io/nvidia/isaac-lab:2.1.0 容器，载荷 `taili_blind_runtime_fixstep11_1df8dd02e35a`
  （payload_digest `1df8dd02e35a…`），4 个并行环境
  - **2026-09-16 订正**：此处原写「远程GPU服务器 (RTX 4090 24GB, 183.147.142.40:31376)」，
    那是另一台机器，本轮验证**没有**在那台上跑过。
- **实现位置**：
  - autotuner/execution/compatibility.py (兼容性检查，348行)
  - products/taili/blind_locomotion/runtime_manifest.py (状态捕获，446行)
  - products/taili/blind_locomotion/train_taili.py (恢复入口，105-226行)
  - tests/autotuner/execution/test_compatibility.py (19个测试)
  - tests/products/taili/blind_locomotion/test_runtime_manifest.py (13个测试)
  - tools/verify_resume_continuity.py (旧验证工具，**空壳，不可用**，见下)
  - tools/verify_resume_weights.py (2026-09-16 新增，权重级验证，本轮判据的实际来源)
  - docs/p4_3_validation_plan.md (验证计划；其验收标准无阈值，订正见上)
- **完成情况**（2026-09-13）：
  - 核心逻辑已实现：状态捕获、兼容性检查、恢复入口
  - 单元测试完成：32个测试覆盖核心路径和边缘情况
  - 验证工具就绪：checkpoint完整性检查框架
- **端到端实机验证**（2026-09-16，3060）：
  - 判据链：`--checkpoint` 是**启动器**参数（`launch_taili_train.py:207`），它把参数追加进
    `train_args`（`:245-246`）再用 `subprocess.Popen` 起 `train_taili`（`:270`/`:362`）；
    `train_taili.py:380` 调 `runner.agent.load(args.checkpoint)`。
  - 四个配对运行（全部 4 envs，同一载荷）：

    | 运行 | 命令要点 | 产物 |
    |---|---|---|
    | resumeA | 全新，400 步，`TAILI_CHECKPOINT_INTERVAL=200` | `agent_200.pt`、`agent_400.pt` |
    | resumeD | `--checkpoint resumeA/agent_200.pt`，10 步，间隔 1 | `agent_2.pt`…`agent_10.pt` |
    | resumeE | **对照**：同一命令、不给 `--checkpoint`，10 步，间隔 1 | `agent_2.pt`…`agent_10.pt` |
    | resumeF | `--checkpoint resumeA/agent_200.pt`，200 步，间隔 100 | `agent_100.pt`、`agent_200.pt` |

  - **判据一 · 权重被加载（逐位相等）**：`resumeD/agent_2.pt` 对 `resumeA/agent_200.pt`，
    149 个张量**全部逐位相等**，最大绝对差 `0.000e+00`。对照 `resumeE/agent_2.pt` 对同一文件
    则完全不相干（policy 余弦 0.0266，最大绝对差 3.072e+03）。
    两者文件大小不同（46,648,904 vs 46,649,146 字节）→ 不是拷贝文件，是真实的 load→save 往返。
  - **判据二 · 优化器动量续用**：resumeA 与 resumeD 各含 99 个 `optimizer/state/{i}/{exp_avg,
    exp_avg_sq,step}` 张量且全部非零；对照 resumeE 的 optimizer state 条目为 **0**
    （新 agent 在第一次 `optimizer.step()` 之前 state_dict 就是空的）。这是一条不依赖阈值的
    离散判据。
  - **判据三 · 训练确实继续**：resumeF 从 A@200 再跑 200 步后，`step` 计数器为 **512**，
    与 resumeA 自己跑到 400 步时的 512 相同。逐组件相对 L2：

    | 组件 | A@200→A@400（走 200 步） | A@400 vs F@200（复现误差） | 无关参照（E@10） |
    |---|---|---|---|
    | policy | 0.6665 | **0.0011** | 1.9693 |
    | state_preprocessor | 0.6665 | **0.0005** | 1.9734 |
    | amp_state_preprocessor | 0.6666 | **0.0000** | 1.9960 |
    | value_preprocessor | 0.7818 | 0.2594 | 1.9996 |
    | value | 0.0440 | 0.0620 | 0.0455 |
    | discriminator | 0.0496 | 0.0388 | 0.1102 |

    **value 与 discriminator 两行在本判据下不可用**：它们初值的范数就压过了训练带来的位移
    （value 的无关参照 0.0455 甚至小于它 200 步的位移 0.0440）。这两项的证据在判据一里——
    那里它们是逐位相等的。
  - **未验证明细**：`.pt` 检查点里**根本不含** scheduler / curriculum / rng，
    `_resume_parity` 自己的注释也写明「Scheduler, curriculum and RNG cannot be proven from a
    legacy .pt file」。复现不是逐位（policy 相对 L2 0.0011 而非 0），最合理的解释是 rollout
    buffer 与 RNG 不在检查点内，**但这是推测，未验证**。
- **9 个状态组件的实际证据强度**（2026-09-16 订正）：
  1. policy - actor网络权重 —— **已实测**：逐位相等；再跑 200 步后复现误差 0.0011
  2. value - critic网络权重 —— **已实测**：逐位相等（`value` 单独看复现误差，判据不可用，见上）
  3. optimizer - 优化器状态（动量/自适应矩）—— **已实测**：逐位相等 + 99 个张量全非零 + step 256→512
  4. scheduler - 学习率调度器 —— **未验证**：不在 `.pt` 里；`restored.scheduler=true` 的
     依据只是「父清单记过这个字段」，不是测量
  5. normalizer - 观测/值归一化器 —— **已实测**：`state_preprocessor` 逐位相等、复现误差 0.0005
  6. log_std - 策略标准差参数 —— **已实测**：作为 `policy/actor.log_std_param` 随 policy 逐位相等
  7. amp - 对抗性运动先验判别器 —— **已实测**：`discriminator` 与 `amp_state_preprocessor` 逐位相等
  8. curriculum - 地形课程阶段/级别 —— **未验证**：不在 `.pt` 里，同 scheduler
  9. rng - Python/NumPy/PyTorch/CUDA随机数状态 —— **未验证**：不在 `.pt` 里，同 scheduler
  - **`resume_edge.restored` 不等于实测**：那 9 个 `true` 是「按键名别名在检查点里找到了对应项」
    再叠加「父清单里记过这个字段」推出来的合理性检查，不是对权重的测量。
  - **一处反证**：`optimization_state.fields` 里 scheduler / curriculum / rng 三项的 sha256
    （`014da526…` / `afc1cf8a…` / `b40e85f7…`）在 resumeA（全新 400 步）、resumeE（全新 10 步）、
    resumeD（恢复）**三个运行里一字不差**。这三个摘要不携带任何运行信息，不能拿来当恢复证据。
    其余字段随运行不同而不同（capture 在 `agent.load()` 之后、训练之前执行，见 `train_taili.py:385`）。
  - **未解释**：resumeA 与 resumeE 同为全新跑、同一配置（`config_digest` 均为 `aae96caa…`）、
    同一载荷，但两者 `optimization_state.fields.policy.sha256` 不同（`d096ec4a…` vs `2726d410…`），
    而其余 8 个字段相同。**现象记录在此，原因未查明**；它不影响上面的权重级结论
    （那是直接对张量做的比较），但它说明「全新跑的网络初值并不完全可复现」。
- **为什么不能用曲线判定**（2026-09-16 实测）：
  - 原验收标准「恢复后曲线连续，无性能跳变」全仓**没有任何数值阈值**。
  - 实测噪声底：resumeA 这一条**没被打断**的跑，相邻采样点相对差的中位数就有 **0.180**，
    85% 的相邻点差值超过 10%。也就是说 10% 量级的「跳变」在正常训练里遍地都是。
  - 对照实验：resumeB（恢复）均值 -7.663 与 resumeC（对照）均值 -7.262 相差 0.401，
    而 21 个采样点上的标准误约 0.44、单点摆幅约 ±4。**这个设计区分不了恢复与全新开始。**
  - 结论：曲线判据在本尺度上不可用，应改用上面的权重级判据。曲线只能作为辅助观感。
- **新发现（结构上成立，未实测）**：**复用同一个 run id 会让身份门禁自证**。
  `train_taili.py:294-309` 只在 `runtime_manifest.json` **不存在**时才写初始清单
  （含本周期的 `resume_checkpoint` 与 `seed`）；而 `:359` 又从**同一路径**把清单读回来当
  `current_manifest`。于是第二次用同一个 run id 时，读到的是上一轮的清单。
  本轮实验每次都换新 run id（resumeA…F），所以上面的证据不受影响。
  **未实测**：没有实际跑过一次复用 run id 的训练去确认后果。
- **复核方式**：上面所有数字都可用仓库里的 `tools/verify_resume_weights.py` 独立复跑
  （要在容器内跑，因为需要 torch）：
  ```
  /workspace/isaaclab/_isaac_sim/python.sh tools/verify_resume_weights.py \
      run/taili_runs/resumeA/checkpoints/agent_200.pt \
      run/taili_runs/resumeA/checkpoints/agent_400.pt \
      run/taili_runs/resumeF/checkpoints/agent_200.pt \
      run/taili_runs/resumeE/checkpoints/agent_10.pt \
      --label 父运行@200 父运行@400 F_恢复后跑200步 E_对照跑10步
  ```
  脚本头部写明了判读方法，以及 value / discriminator 两组为什么在本方法下不可用。
- **旧验证工具不可用**：`tools/verify_resume_continuity.py` 是空壳，不能当验证工具用。
- **注**：P4.2检查点管理系统的GPU环境部署验证作为独立验证任务记录在 docs/P4.3_gpu_deployment_summary.md 和 docs/P4.3_remote_testing_instructions.md，不与本训练恢复功能混淆

### P4.4 超参数搜索空间
- **问题**：定义可搜索的超参数及其范围
- **状态**：进行中
- **子进度**：拆分为 7 个原子任务，第 1-7 个均已落地；全链路未接真实训练（见下表）
- **阻塞**：无（P5.2 已完成）
- **负责人**：
- **验收标准**：搜索空间覆盖学习率、熵系数、折扣因子

P4.4 不是单个原子问题，按可独立验证的最小单元拆分如下。原任务 2 中的贝叶斯采样器
依赖任务 4 产出的"观测回传"接口，无法独立验证，故拆为任务 7 并标注阻塞关系——
否则任务 2 会以"网格/随机已完成"的状态记成"全部完成"。

| 原子任务 | 内容 | 状态 | 实现位置 |
| --- | --- | --- | --- |
| 1 | 搜索空间 schema 与产品搜索空间数据 | 已完成 | autotuner/research/hyperparameter_space.py, products/taili/blind_locomotion/hyperparameter_space.yaml, tests/autotuner/research/test_hyperparameter_space.py |
| 2 | 超参数采样器（网格/随机，含去重与重放） | 已完成 | autotuner/research/hyperparameter_sampler.py, tests/autotuner/research/test_hyperparameter_sampler.py |
| 3 | 配置注入到训练流程 | 已完成 | autotuner/research/config_injection.py, tests/autotuner/research/test_config_injection.py |
| 4 | 搜索试验跟踪 | 已完成 | autotuner/research/trial_ledger.py, autotuner/research/hyperparameter_search.py, tests/autotuner/research/test_trial_ledger.py, tests/autotuner/research/test_hyperparameter_search.py |
| 5 | 早停与剪枝策略 | 已实现（123 用例通过；**未接过真实训练**，见下） | `autotuner/research/search_pruning.py`；设计稿 `docs/P4.4_task5_design.md`；审查记录 §六 |
| 6 | 搜索结果分析与最优配置导出 | 已实现（73 用例通过；**未接过真实训练**，见下） | `autotuner/research/search_analysis.py`；设计稿 `docs/P4.4_task6_design.md`；审查记录 §七 |
| 7 | 贝叶斯采样器 | 已实现（94 用例通过；**未跑过真实搜索、无任何效能证据**，见下） | `autotuner/research/bayesian_sampler.py`；设计稿 `docs/P4.4_task7_design.md`；审查记录 §八 |

任务 1 的实际范围：schema 定义域（continuous/discrete/categorical）、条件参数
（含环路与前缀冲突检测）、参数间约束（multiple）、内容指纹、产品搜索空间 YAML 与
通用加载器。搜索空间覆盖学习率、熵系数、折扣因子，且测试校验每个参数都是出厂配置里
真实存在的键、每个 default 等于出厂值。

任务 2 的实际范围：SearchPlan / GridSampler / RandomSampler / build_sampler / replay，
含条件参数的依赖序抽取、约束拒绝、跨试验去重、按 assignment_key 的重放。网格采样器
对未声明 grid_points 的参数**报错而非猜测**，因此 entropy_loss_scale 与 kl_threshold
这类无网格的连续参数只能用随机采样——这是上面"已知遗留"的处置结果，不是遗留项。

任务 3 的实际范围：把采样器产出的 assignment 写进训练配置。作业形式是**嵌套**结构
（`{"skrl": {"agent": {...}}}`），与 `default_assignment()` 一致；扁平点分键输入会被
明确拒绝并指出形状。这一点曾被端到端探针查出不一致（采样器产出嵌套、注入器只认扁平，
且对扁平输入调用的校验是空操作），修复记录见 `P4.4_review_record.md` 的 I-1。
它**不**证明训练进程读取了注入值：`--dry-run` 只到 `agent.skrl.yaml` 为止。

任务 4 的实际范围：通用 JSONL 台账引擎（`trial_ledger.py`：O(1) 追加、跨进程记录锁、
撕裂尾行自愈、哈希链校验）+ 搜索语义与读写视图（`hyperparameter_search.py`：`TrialRecord` /
`SearchRunRecord`、写入门面 `SearchTracker`、读侧 `SearchLedger`、给任务 7 的观测协议
`TrialObservationSource`）。写入侧的守卫是 `append` 的 precondition、在锁内执行；`close_run`
先跑一遍读侧检查再写，第二遍在锁内。它不跑训练、不决定停止、不排序——那是任务 5、6。
两点如实说明：设计文档 §11.3 的五个待定默认值（规范目录、是否保留跨进程锁、是否镜像进
`research_ledger`、`max_seconds` 默认层、`retry_interrupted` 默认值）**仍未定**，属策略选择
而非缺失功能；未做真实 GPU 训练验证（`TrialRunner` 在测试里是替身）。

任务 1-4 未覆盖（不属于夸大范围）：剪枝、结果分析、贝叶斯采样。三项分别见下方任务 5、6、7
的说明。

任务 5 的实现位置：`autotuner/research/search_pruning.py`（1690 行）、
`tests/autotuner/research/test_search_pruning.py`（2087 行、123 个用例，全通过；
与任务 4 的 66 个用例两个文件同跑 **189 passed in 75.30s**，均为 2026-09-15 实跑）。
第二轮审查（审的是**实现**而不是设计稿）查出 7 个缺陷，全是"审计层自信地给出错误结论"而非崩溃，
已全部修复并补 25 个用例（92→123），13 条变异验证全部杀死；
修复内容、逐条复现方式与**明确未验证事项**见 `docs/P4.4_review_record.md` §六。
设计稿 `docs/P4.4_task5_design.md` 是本轮三个独立设计 → 三名读码评委 → 一次合成的结果；
第 9 节区分"本轮实测"与"本轮仅读码"（注：该节原先把一份**静态抄写的行号清单**归在"本轮实测"之下，
2026-09-15 已改为独立小节并标注为静态阅读）。该稿的 99 条行号引用另经一次独立复核
（94 条精确、5 条区间端点差一行，已改正）；该稿 §4 原先把**计划**写成了**已经落地**，
已就地标注并补"计划 → 落地"映射。
范围边界如下，写在这里是为了让只读本表的人不会误以为剪枝已经能用：

- **剪枝这层是能用的；能驱动它的那条路还没有。** 仓库里今天仍**没有任何接进训练流程的
  `TrialRunner` 实现**：`hyperparameter_search.py:221` 的 `TrialRunner` 是 Protocol，
  唯一的实现类是 `search_pruning.py` 里的 `PruningTrialRunner`，而它**只被替身和假 runner
  驱动、没有任何调用方**（写这段时原句是"没有任何 `TrialRunner` 实现"，与下一句自相矛盾，
  且已被实现本身证伪，2026-09-15 改正）。**未接过真实训练。**
- 参考遥测适配器（曲线源、sink、watcher）只在 tmp_path 文件与替身 runner 上验证过，
  **未跑过 `blind_tp_env`、未跑过 GPU、未按真实 `TAILI_TELEMETRY_INTERVAL` 节奏验证过**。
- SSH 后端不支持：它在进程退出前不同步日志（`research_remote.py:259-261`），远程试验剪不了。
- 剪枝**不回收预算**，被剪的试验仍占用槽位。
- `products/taili/ops/early_stop.py` 的 R1/R2/R3 **不被继承**：本轮实测其三个字段名在全仓库
  没有生产者（真实载荷的 `health` 段无 `episode_length_mean`、`reward` 段无 `mean` 键）。
- **写台账前必须先有 `search_run` 头。** 实测（不是推断）：任何非空搜索台账若首个事件不是
  `search_run` append，`_validate_layout`（`hyperparameter_search.py:574-622`，
  `unknown_layout` 抛在 `:587`/`:592`/`:597`/`:609`）会让整根永久
  `unknown_layout`，`open_run` 自己也读不回来。所以策略记录走 `store_policy_record`，
  它的 `precondition` **无默认值**，`TrackerPruningSink` 传 `tracker._require_open`。

任务 6 的实现位置：`autotuner/research/search_analysis.py`（1545 行，
sha256 `075be749b49ab4ac4bf97a714ebe31581d53c0a15e50567aa75df3340e827ea6`）、
`tests/autotuner/research/test_search_analysis.py`（1967 行、73 个用例，全通过）。
两个哈希都是**本机工作区**（LF）的哈希；早先记的 `2f254174…` 是模块换行符统一为 LF **之前**
的哈希，行数也从 1544 变 1545（补了 `canonical_metric` 这一个纯函数的名字）。

> **上述数字是任务 6 收尾时的快照，任务 7 已把它们改掉**（2026-09-16 实测）：任务 7 在
> `search_analysis.py` 里删掉一段失效的依据、在 `test_search_analysis.py` 里加 3 条改 1 条，
> 于是现在 `search_analysis.py` 是 **1554 行**、sha256 `0fd73d979737546da0a32a5fd699c14dbe442ab5fa7e4fd72821dcbfbf5acf27`，
> `test_search_analysis.py` 是 **2036 行、75 个用例**、sha256
> `7e62cde6f3cc6337f315291efc6637fbb2fcf6a4db03dfe0173e53a5bd2ce66b`（两者都仍为 LF）。
> 上面那两行**保留原值**当作任务 6 的落地记录，不做替换。
范围：把一条搜索运行的台账读成一份可复核的分析（排名、并列计数、每个读数的状态），
并据此导出最优配置与回执。它**不做**的事：不重排、不写台账、不碰训练。
第二轮审查（审的是**实现**）查出 8 个缺陷，性质与任务 5 那批同类——都是"审计层自信地给出
错误结论"而不是崩溃（把被 `top_n` 截断后的行当成全体来数并列、无法表示的整数让整份分析崩掉、
`metrics` 里的非有限数被读成"缺失"、导出回执可能覆盖自己的证据、`yaml.YAMLError` 不是
`ValueError` 导致导出把解析错当崩溃…），已全部修复并补 8 个用例（63→71），
7 条变异验证全部杀死。当晚又补 2 条"守卫缺失"用例（63→**73**）：`_coverage_verdict` 里
`unstated` 早返回的顺序、以及 `export_best` 的 `injected_fingerprint` 相等守卫——
这两处**原先删掉都不影响任何一条用例**（实测删除后仍 73 passed），补上后各自都有指名用例变红。
同时用受控变异**证伪**了一条早先的推断：`_validate_no_trial_vanishes` 的调用被删除时
**没有任何用例变红**（73 passed），所以"那条守卫有杀死变异的能力"是错的。
另：`autotuner/research/search_analysis.py` 的换行符此前是 CRLF（与仓库其余文件不一致、
且让文档里的 sha256 无法从克隆复现），已统一为 LF。
一条**被实测证伪、因此没有改代码**的审计指控：有评委称重复的 trial index 会让 `analyze()`
在 `TrialCensus` 上抛错。实测台账按 index 折行（同名 index 只有最后一次尝试被读回），
重复写入后 `recorded == 1`、`run()` 正常返回、`error` 为空——指控不成立，故未动代码，
已记入设计稿 §12.4，避免后人误以为这里"修过"。
明确未验证：**未接过真实训练**（台账、配置与 manifest 都是真的写在盘的临时文件，
但产出它们的训练进程是替身，没有 GPU、没有 `blind_tp_env`）；"四条承重守卫各自被删除时
是否真有用例变红"的定律测试本轮**仍未系统性跑过**——只跑了其中三条（M3 / M5 / M13，
脚本 `.scratch_wf6/mut_b.py`），前两条已补守卫、第三条实测**存活**；其余承重守卫仍无变异证据，
设计稿 §12.3 已如实标注。
修复内容、逐条复现命令与明确未验证事项见 `docs/P4.4_review_record.md` §七。

任务 7 的实现位置：`autotuner/research/bayesian_sampler.py`（1341 行）、
`tests/autotuner/research/test_bayesian_sampler.py`（2795 行、94 个用例，全通过；
设计稿 §12.1 另有这两个文件的 sha256，用途见本文件更新日志里
2026-09-15 那条过程事故记录）。
范围：编码器（连续/离散/类别 → 单位立方体）、纯标准库的高斯过程（Cholesky）、
期望改进（EI）、自适应采样器（冷启动设计 → 后验 → 池内取最优）、
以及"每个提议都有一条写下来的决策"的记录与复核（`SurrogateDecisionRecord` / `verify_step`）。
它**不做**的事：不跑训练、不决定停止、不排序、不做效能声明。
落在 plan 里的策略（`policy` 字段）会让**所有种类的计划指纹都变**——这是选择的代价，理由见设计稿 §7 R-6。
本轮 33 条变异按预期落定（32 条真变异全被杀死、控制组存活、每个被改文件按字节还原并断言
sha256）。其中**第一轮的 28 条漏掉了两处**（`_describe_draws` 的多约束格式与排序，删掉都不
影响任何用例），补用例后才杀掉；**第三轮另加的 3 条**盯的是第一轮碰不到的三处行为，一上去
就被杀死——那说明的是**探测面**原来没盖到那三处，不是那三处没有守卫，别把这 3 条读成
"发现并修好了三个洞"。这条教训与任务 6 的 §7.7 同类：全绿不等于有守卫。
明确未验证（12 条，全文见设计稿 §11.4）：**未跑过真实搜索**，模块与文档都**不做效能声明**；
本轮探针（设计稿 §11.1 第 4 条）给出的读数反而**不利于**"自适应更省试验"这个说法；
`random.seed(str)` 的跨 Python 版本稳定性未验证（确定性只在同一版本内）；
`length_scale`/`noise`/`xi` 的默认值是给的、不是调出来的；
`_encode_discrete` 的 `identifies` 分支（"在域内但不是域的点"）**没有任何用例执行到**，
按本仓的 dead-branch 教条它应当被删或被证明可达，本轮**既没删也没证明**；
与任务 5 剪枝同时启用时的行为未验证（剪枝会把试验变成 `pruned`，默认 `learn_from=("succeeded",)`
会把它排除，自适应可能一直冷启动）。
见 `docs/P4.4_review_record.md` §八。

设计稿 `docs/P4.4_task4_design.md` 原先在开头写着「本文是设计提案，**尚未实现**」。该句已随
实现更正，并订正了 7 处状态或计数（规模估计实测 763/1612 行、`mechanisms.json` 实测 13 个、
`content_hash` 实测 27 处、A12 的判据由「grep 无输出」改为 AST 遍历、验收编号撞号改标 A42 等）；
每条订正都附可复核命令，见 `docs/P4.4_review_record.md` §4.4.1。

任务 1-3 的实现依据（多智能体审查的发现与判定）见 `docs/P4.4_review_record.md`。
该记录列出已修复的 S-1..S-10、I-1、F-OPEN-1，以及**被反驳不予采纳**的 R-1..R-5，
便于逐条复核。任务 4 的审查与**变异测试实测结果**（14 条变异，13 条被杀死、1 条存活并已
补用例）见该文件 §四，含明确未验证事项。

### P4.5 训练遥测上报
- **问题**：实时上报 loss/reward/episode 指标
- **状态**：已完成
- **阻塞**：无
- **负责人**：
- **验收标准**：控制台能实时显示训练曲线
- **实现位置**：autotuner/locomotion_console/telemetry.py, autotuner/mechanisms/mechanism_specs.py (MetricSpec)

### P4.6 训练进程在 Isaac Sim 容器里的 torchvision 同源问题
- **问题**：Isaac Sim 镜像里有两份 CUDA 构建不同的 torchvision（kit 的 site-packages
  一份 `0.20.1+cu124`、`omni.isaac.ml_archive` 扩展的 `pip_prebundle` 一份 `0.20.1+cu118`），
  且 prebundle 那份不走 `sys.path`。谁先被导入谁定终身：本包导入阶段会 `import skrl`，
  skrl 又导入 torch（拿到 site-packages 的 cu124），而 torchvision 留到 app 启动时才被
  Kit 扩展 `isaaclab_tasks` 导入（拿到 prebundle 的 cu118），两边不同源即报
  `operator torchvision::nms does not exist`，半初始化的模块留在 `sys.modules` 里，
  之后每次 `import torchvision` 都失败。崩点在 `train_taili.py` 的
  `from isaaclab_tasks.utils import parse_env_cfg`，表现为训练一步都跑不起来。
- **状态**：已修复
- **阻塞**：无
- **负责人**：
- **验收标准**：容器内跑 `taili_blind_runtime.train_taili` 能越过
  `from isaaclab_tasks.utils import parse_env_cfg`，不再出现 torchvision 相关报错
- **验收结果**：**已达标**。修复后在 3060 容器里重跑 `smoke5`，app 正常启动、
  `isaaclab_tasks` 不再报 `torchvision::nms does not exist`、`train_taili.py:274` 这一行通过，
  卡点前移到下一行的 `ModuleNotFoundError: No module named 'autotuner'`（属于另一个问题，
  见 P4.7）。日志：`/home/chuan/robot_lab/logs/smoke5.stdout` 与
  `/home/chuan/robot_lab/run/taili_runs/smoke5/console.log`
- **注**：本条只声明"torchvision 报错消失"。**训练一步都还没跑起来**是另一条问题（P4.7），
  不要因为本条标了"已修复"就以为容器里已经能训
- **实现位置**：
  - products/taili/blind_locomotion/_torchvision_pair.py（`torchvision_alongside` / `pin_torchvision`）
  - products/taili/blind_locomotion/__init__.py:30-36（在 `from skrl...` 之前钉住配对）
  - products/taili/payload/payload_manifest.py:29（新模块必须进载荷清单，否则修复到不了远端）
  - tests/products/taili/blind_locomotion/test_torchvision_pair.py（12 个用例）
- **验证方式**：
  - 根因是**实验做出来的**，不是猜的：在同一容器里只改导入顺序跑了四组对照
    （不预导入 / 预导入 site-packages 一对 / 预导入 prebundle 一对 / **只预导入 torch**），
    只有"只预导入 torch"那组复现出与失败运行**逐字相同**的报错
    （`/home/chuan/robot_lab/logs/probe_order_effect2.log`）；前三组都正常，
    说明问题在配对而不在版本本身
  - 本机受控变异 6 条全部被杀死、控制组干净、文件按字节还原（承重文件原始 sha256：
    `_torchvision_pair.py=2e9a3ec3…`、`__init__.py=fe12ff0a…`）。其中两条盯的正是这个修复的
    全部内容——**顺序**：把 `products/taili/blind_locomotion/__init__.py` 里的
    `pin_torchvision()` 调用删掉、以及把它挪到 `from skrl.utils.runner.torch import Runner`
    之后，`test_pin_runs_before_skrl_import` 各自变红
  - 复核这两条不需要任何脚本：上述改动各做一次，跑
    `pytest tests/products/taili/blind_locomotion/test_torchvision_pair.py`，该用例应当失败。
  - 载荷 `taili_blind_runtime_20260916_tvfix_fcf7a527d00a`（114 个文件）里
    `__init__.py` 的 sha256 与本机改好的源文件一致（`fe12ff0a8c1862553…`）
- **未验证**：
  - 只在这一台机器的 `nvcr.io/nvidia/isaac-lab:2.1.0` 镜像上验过；其它 Isaac Sim 版本、
    其它 torch/torchvision 组合没有验过
  - 镜像里那份 prebundle 的**其余**内容（它自己的 torch 等）没有动，也没有验证过
    "始终由 site-packages 胜出"在别的入口下是否成立

---

### P4.7 载荷里 import 不到 autotuner / products（双布局迁移不完整）
- **问题**：远端训练只把**载荷目录**放上 PYTHONPATH，载荷里只有 `taili_blind_runtime/`
  一个包，没有 `autotuner/`、也没有 `products/`。而源码树里的模块写的是
  `from autotuner.xxx import ...` 这种以仓库根为起点的绝对导入，在载荷里必然
  `ModuleNotFoundError`。
- **状态**：已修复
- **阻塞**：无
- **负责人**：
- **验收标准**：只把载荷目录放上 PYTHONPATH（不带仓库根），
  `python -m taili_blind_runtime.launch_taili_train` 能起得来并跑出训练一步
- **验收结果**：**部分达标——导入这一关过了，但"跑出训练一步"没有**。
  - 修复前：smoke5 报 `ModuleNotFoundError: No module named 'autotuner'`
    （`logs/smoke5.stdout:51`、`run/taili_runs/smoke5/console.log:44`），崩点在
    `train_taili.py` 的导入阶段，训练一步都进不去
  - 修复后：训练进程能一路走到 `train_taili.py:441` 的 `runner.run()` 并进入
    `env.step()`——导入问题确实解决了
  - **但训练仍在第一步抛异常退出**，是另一个缺陷（P4.8）。
    本节此前据 `rc=0` 与 `[TPSTAT] step=1` 判为"已达标"，**这个判断是错的**：
    `rc=0` 是 `run_in_container.sh` 末尾 echo 的产物，`step=1` 那条遥测是在
    抛异常的那次 `_get_rewards` 调用里打出来的。复核方式见 P4.8。
- **实现位置**：
  - `products/taili/blind_locomotion/*.py` 等处的双布局导入：先试包内相对导入，
    再退回源码树的绝对导入
  - `products/taili/payload/payload_manifest.py` 的 `STATIC_FILES`（手工白名单）
  - `payload_manifest.py:_validate_import_closure()`（2026-09-16 新增的构建期检查）
- **验证方式**：
  - 构建期：`python -m products.taili.payload.build_payload` 时
    `_validate_import_closure` 对每个待打包文件做 AST 扫描，凡是"载荷里解析不到、
    且不存在任何一支可解析 fallback"的导入直接判错。变异测试证明它不空转：删掉
    `checkpoint_integration` 的白名单条目 → 报"组内没有可解析分支"；注入
    `from .definitely_missing_module import nothing` → 报"相对导入未打包"。
  - 运行期：`.scratch_verify_pair/probe_p42_payload.py`，在**只有载荷在 sys.path 上**的
    子进程里 import 并驱动检查点管理，见 P4.2
- **未验证 / 遗留风险**：
  - `STATIC_FILES` 是**手工**白名单。构建期检查只能发现"白名单里的文件导不进来"，
    发现不了"整个模块压根忘了加进白名单"——P4.2 的四个模块正是这么漏的。
    结构性的修法是让包发现自动收集，未做。
  - 本节最初写"修复后 smoke12 与 ladder_n4 均 rc=0，说明能跑"，这是**读错了证据**：
    那两次运行实际都失败了（`runtime_manifest.json` 里 `status: failed`），
    详见 P4.8。教训：远端运行的成败要看 `runtime_manifest.json` 的 `status`，
    不能看退回码——容器脚本末尾那句 echo 会把退回码抹成 0。

---

### P4.8 训练在第一步就崩（`int("phi0")`），且 rc=0 掩盖了崩溃
- **问题**：两个独立缺陷叠在一起。
  1. `blind_tp_env.py` 的 `_get_rewards` 在构造检查点性能快照时，
     对 `curriculum_payload["phase"]` 做 `int()`；而 `telemetry_payloads.py:333`
     把这个字段定义成**给人看的显示串** `f"phi{phase}"`（`phase is None` 时是空串）。
     于是第一次算奖励就抛
     `ValueError: invalid literal for int() with base 10: 'phi0'`。
     `.get("phase", 0)` 的默认值 0 救不了——键一直在，只是值不是数字。
  2. `run_in_container.sh` 的容器命令以 `echo "container_end ..."` 收尾，
     所以 `docker run` 的退出码永远是那个 echo 的 0。**训练崩了也报 rc=0。**
- **状态**：已修复
- **阻塞**：无
- **负责人**：
- **验收标准**：`--total-steps 576 --num_envs 4 --telemetry-interval 1` 的远端运行
  能越过第一步、连续打出多条 `[TPSTAT]`，且 `runtime_manifest.json` 的 `status`
  不再是 `failed`
- **验收结果**：**达标**（2026-09-16，3060 机器）。第 2 条验收标准的形式
  （"多条 `[TPSTAT]`"）已经过时——同一台机器上后续跑出的 `[TPSTAT]` 条数
  由 `--telemetry-interval` 决定，不是"有没有越过第一步"的量度；改用 tqdm 的
  完成度和 `[TPSTAT] step=576` 收尾行作准。逐条对照：
  - `fixstep_n4`：tqdm 收在 `576/576`，末行 `[TPSTAT] step=576 total=576 pct=100.00`，
    `runtime_manifest.json` 的 `status: complete`、`errors: []`。
    注意它报的 `rc=0` **不能**作为证据，理由见第 2 个缺陷。
  - `hookchk_n4` / `hooksave_n4` / `hookreg_n4` / `hookreal_n4`：同载荷同流程
    各跑一次，`status` 均为 `complete`，`errors` 均为空。
  - 反证：修复前同一命令下全文只有 **1** 条 `[TPSTAT]`，
    且 `ladder2_n4` 的 traceback 指到 `blind_tp_env.py:3274` 的 `int(...)`。
- **后续发现（同一条数据链，2026-09-16）**：验收达标之后，沿着这个快照继续查，
  又挖出 `build_checkpoint_performance_snapshot` 读了一组**不存在的键**
  （`reward["mean"]`、`health["episode_length_mean"]`），危害与量级见 P4.2 第 3 条。
  本节原先的"4 条回归"里有一条把这个错误当成了契约，已改写。
- **实现位置**：
  - products/taili/blind_locomotion/telemetry_payloads.py
    的 `build_checkpoint_performance_snapshot`（收**数值** phase，不解析显示串）
  - products/taili/blind_locomotion/blind_tp_env.py 的 `_get_rewards` 调用点
  - tests/products/taili/blind_locomotion/test_telemetry_contract.py（4 条回归）
- **验证方式**：
  - 根因是**实验做出来的**，不是从代码推的：
    `run/taili_runs/ladder2_n4/runtime_manifest.json` 的 `errors[0].traceback`
    直接指到 `blind_tp_env.py:3274` 的 `int(...)`；
    再用 `--telemetry-interval 1` 跑一次，全文仍只有 **1** 条 `[TPSTAT]`——
    说明 `_get_rewards` 只被调用了一次就抛了，坐实"第一步就崩"
  - 本地：`pytest tests/products/taili/blind_locomotion/test_telemetry_contract.py`
    （新增 4 条：显示串是 `"phi0"`、快照收数值、无 phase 时为 0、
    源码里不许再出现 `int(curriculum_payload`）
  - 旁证：`int(curriculum_payload` 在修复后的载荷里出现 0 次
- **受影响的历史结论**：smoke12 / ladder_n4 / ladder2_n4 / diagstep_n4
  **全部**是失败运行（`status: failed`，同一条 ValueError）。
  凡是据这几次运行得出的"训练跑起来了"的说法都不成立，包括 P4.2 的端到端部分
  与 P4.7 的验收结果。
- **未验证**：
  - 修复只覆盖了 `curriculum_phase` 这一处。同一段代码里其它
    `int()`/`float()` 取值是否也踩了显示串，没有逐个查过。
  - 训练在**持续**跑起来之后还会不会撞下一个问题，未知——此前从未跑到过第二步。

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

下面每个数字都能由本文各条目的「状态」行数出来，不引用任何外部计数。复核命令：

```
grep -cE '^### P[0-9]\.[0-9]' docs/atomic_problems.md                     # 32
grep -E '^- \*\*状态\*\*：' docs/atomic_problems.md | sed 's/^- \*\*状态\*\*：//' | sort | uniq -c
```

第二条命令 2026-09-16 的实测输出（P4.3 转入「已完成」之后重跑）：

```
     20 已完成
      7 未开始
      3 已修复
      1 部分完成
      1 进行中
```

**状态词表**（只用这 6 个值，`状态` 行里不带括号内的解释——解释放 `子进度` 等字段，
否则上面那条 grep 数不出数）：

| 状态 | 含义 |
| --- | --- |
| 未开始 | 还没动手 |
| 进行中 | 动了，但验收标准未达标 |
| 部分完成 | 拆开的步骤里一部分达标、一部分没有 |
| 已完成 | 验收标准在**源码树**层面已达标并有可复核的验证方式 |
| 已完成（待远端端到端验证） | 单元测试/本地验证达标，但目标环境（远端容器）未跑通全链路 |
| 已修复 | 针对某个具体故障，修复有对照实验支撑 |

- **总计**：32 个原子问题（P1.1-P1.5、P2.1-P2.5、P3.1-P3.5、P4.1-P4.8、P5.1-P5.4、P6.1-P6.5）
- **已完成**：20 (含 P4.2 与 P4.3，均为 2026-09-16 端到端跑通后由「待远端端到端验证」转入)
- **已完成（待远端端到端验证）**：0
- **已修复**：3 (P4.6、P4.7、P4.8)
- **部分完成**：1 (P6.1，步骤 1-3 完成，步骤 4-6 待运行时环境)
- **进行中**：1 (P4.4，7 个原子任务均已落地，但全链路未接真实训练)
- **未开始**：7 (P6.4、P5.3、P5.4、P3.3、P1.1、P1.2、P1.5)
- **阻塞**：5 个问题被其他问题阻塞

**2026-09-16 订正（第四次）**：P4.3 由「已完成（待远端端到端验证）」转入「已完成」，
依据是 3060 上的权重级端到端验证（见 P4.3「端到端实机验证」）。「已完成（待远端端到端验证）」
这个状态值**当前没有任何条目在用**，但保留在词表里。同时订正了 P4.3 的 `验证环境`——
它此前写的是另一台机器（RTX 4090 / 183.147.142.40），本轮没在那台上跑过。

**2026-09-16 订正（第三次）**：本节此前写「总计 31」，且 P4.8 的状态行写的是
`已修复（本地已验证）；远端端到端复跑见「验收结果」`——带括号和分号的自由文本，
上面那条 `uniq -c` 数不出它，等于让 P4.8 在统计里隐身。本次把 P4.8 归入 `已修复`
（它的验收标准已在远端达标，见 P4.8「验收结果」），总数 31 → 32。
同时 P4.2 由「已完成（待远端端到端验证）」转入「已完成」：
端到端证据见 P4.2「端到端验证」，其中 `reward_mean` 是与同一份遥测的独立字段对齐过的。

**2026-09-16 订正（第二次）**：本节此前写「总计 29」，但本文当时已有 **30** 个条目
（P4.6 已存在却漏在枚举外），且枚举写的是 `P4.1-P4.5`。本次新增 P4.7 后为 31。
另外新增了 2 个状态值（`已完成（待远端端到端验证）`、`已修复`）并把原先塞在状态行括号里的
说明挪到独立字段，理由见上面的状态词表。P4.3 的旧值 `单元测试完成，待GPU验证`
是同义的另一种写法，本次归并到 `已完成（待远端端到端验证）`。

**2026-09-16 订正（第一次）**：本节曾写「总计 30、已完成 22、进行中 1 (P4.3)、未开始 7」。
这 5 个数字与本文不符，且从写下那一刻起就不符——写入它们的提交 `6ca441e`（2026-09-14）
**没有增删本文任何一个问题标题**（该提交前后本文都是 29 个问题），当时逐条状态为
19 已完成 / 8 未开始 / 1 部分完成 / 1 单元测试完成；「进行中」的也一直是 P4.4 而非 P4.3。
即「30」与「22」不是数出来的。

**最近完成**（2026-09-16）：
- P4.2 训练检查点管理：端到端跑通。补上四个载荷模块之后又挖出三个互相掩盖的
  静默缺陷（`hasattr` 判据、拦了 `agent.save` 而 skrl 调 `write_checkpoint`、
  快照读了一组不存在的键），逐个修掉并在真实训练里验到
  `total_registered: 1` 且 `reward_mean` 与同份遥测的 `reward.total` 精确相等
- P4.8 训练第一步崩 `int("phi0")`：远端 `fixstep_n4` 576/576 步跑到 `status: complete`
- P4.7 载荷里 import 不到 autotuner / products：双布局导入 + 构建期
  `_validate_import_closure` 检查。**注意**：本轮之前写的"远端 smoke12 / ladder_n4
  实测 rc=0"是读错了证据，那两次运行 `status: failed`，见 P4.8
- P4.6 torchvision 同源：钉住 torchvision 与 torch 的配对
- P6.1 仿真器后端抽象：步骤3b完成，**4** 个训练入口已迁移（commit c781480；
  此前误记为 5 个，见本节 P6.1 的订正段）

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

### 远端运行到底跑成没跑成，看什么（2026-09-16 踩坑记录）
- **不要看退回码**。`scripts/run_in_container.sh` 的容器命令以
  `echo "container_end ..."` 收尾，`docker run` 返回的是那句 echo 的状态，
  所以**训练崩了也报 `rc=0`**。实测：smoke12 / ladder_n4 / ladder2_n4 / diagstep_n4
  四次运行 `rc=0`，四次 `runtime_manifest.json` 都是 `status: failed`。
- **不要看 `[TPSTAT] step=1`**。那条遥测由 `_get_rewards` 在**第一次**被调用时打出
  （`_rew_log_step == 1` 是必打条件），而那次调用本身可能正在抛异常——
  上面四次就是如此，"step=1" 与异常出在同一次调用里。
- **要看**：`run/taili_runs/<run_id>/runtime_manifest.json` 的 `status` 与 `errors`。
  一条命令：
  ```
  python3 -c "import json;o=json.load(open('run/taili_runs/<run_id>/runtime_manifest.json'));\
  print(o['status'], (o.get('errors') or [{}])[0].get('message',''))"
  ```
- 另：`[TAILI_LAUNCH] ... rc=0 traceback=False cuda_oom=False` 这句是**启动器**
  对子进程退出码的转述，同样继承上面那个问题，不能单独当成功证据。

### 3060 上的显存阶梯（2026-09-16 实测）

五档同一台机器、同一份载荷 `taili_blind_runtime_fixstep11_1df8dd02e35a`
（五档 `payload_digest` 全等于
`1df8dd02e35a3a681920077c4a51fb8333712a403354a48356fe0219c355c11b`），
只改 `--num_envs`。GPU 为 RTX 3060，`memory.total` 12288 MiB，空载 314-315 MiB。
采样间隔约 1 s。脚本 `scripts/run_vram_ladder.sh`，原始读数在
`logs/ladder_<N>_n<N>.vram`（单列 MiB）与同名 `.vram.ts`（`epoch秒 MiB` 两列）。

| num_envs | total_steps | 峰值 MiB | 首→峰 | 峰→末 | status |
| --- | --- | --- | --- | --- | --- |
| 4 | 240 | 8806 | +97 s | +13 s | complete |
| 64 | 240 | 9987 | +105 s | +17 s | complete |
| 256 | 240 | 10759 | +100 s | +18 s | complete |
| 512 | 240 | 11065 | +106 s | +23 s | complete |
| 512 | 2000 | 11085 | +138 s | +278 s | complete |

五档 `runtime_manifest.json` 均为 `status: complete`、`errors: []`，无 OOM。

读数说明：
- **峰值出现在 +97…+138 s，不在启动瞬间**。首末两次采样都回到 314/315 MiB，
  那是容器起停前后的空载值——所以峰值只能取整列最大值，取末尾会量到空载值。
  `.vram` 是**单列**、不带时间戳，时间要另读 `.vram.ts`。
- **512 档的峰值不是"还在涨"**：240 步档 11065，2000 步档 11085，步数翻 8 倍
  只多 20 MiB；且 2000 步档的峰值在 +138 s 出现后，其后 278 s（约 1600 步）
  再没被超过。**就本次运行而言是平台期**。
- 余量 = 12288 - 11085 = **1203 MiB**。这个余量只在这条命令、这份载荷、
  这套默认地形上量过；换更长训练或换地形够不够，**未验证**。
- 4 → 512 档只多 2259 MiB，而空载到 4 档就吃掉 8491 MiB：固定开销
  （Isaac Sim/PhysX + CUDA context）占大头，环境数不是主要成本。

复核命令（在 3060 上）：
```
cd /home/chuan/robot_lab/logs
python3 -c "v=[int(x) for x in open('ladder_512_n512.vram') if x.strip()];print(max(v))"   # 11065
python3 -c "v=[int(x) for x in open('ladder_long_512.vram') if x.strip()];print(max(v))"  # 11085
```

### 需要优先完成的问题
1. **P6.1 多后端抽象**：步骤1-3已完成（协议定义、IsaacLab适配器、工厂函数、**4** 个训练入口迁移，订正见 P6.1），步骤4-6待GPU运行时环境
2. ~~**P4.3 训练恢复**：待GPU环境执行集成验证~~ —— 2026-09-16 已在 3060 上完成权重级端到端验证，
   转入「已完成」。**遗留两项未验证**：scheduler / curriculum / rng 三项状态不在 `.pt` 检查点内，
   恢复与否无从证明；且「复用同一 run id 会让身份门禁自证」这一条只在源码上成立、未实测
3. **P4.4 超参数搜索**：已拆为 7 个原子任务，任务 1-4（搜索空间、采样器、配置注入、试验跟踪）、任务 5、6（早停剪枝、结果分析与最优配置导出）与任务 7（贝叶斯采样器）全部完成。可复用research_scheduler.py作为试验执行后端。**全链路仍未接过真实训练**：仓库里至今没有任何接进训练流程的 `TrialRunner` 实现，任务 5、6、7 都只在替身 runner 与临时文件上验证过；任务 7 的模块与文档都**不做效能声明**（本轮没有"自适应 vs 随机"的对比读数，探针读数反而不利于该说法）
4. **P6.4 接触力校准**：需要真实硬件数据

### 层级解耦评估
- **Layer 6 → 5**：观测/奖励契约通过 policy_contract.py 隔离，符合设计
- **Layer 5 → 4**：mechanism_specs.py 提供声明式边界，门控通过 GateSpec 配置
- **Layer 4 → 3**：research_supervisor.py 暴露执行协议，不依赖上层
- **Layer 3 → 2**：coordinator.py 消费产品合同，不直接操作奖励或训练代码
- **Layer 2 → 1**：控制台通过 API 调用产品层，LLM 工具调用有审批机制

### 已知的文档与实现不一致（未修复）

1. **注释语言**：`docs/maintenance_standard.md:14` 要求"新增注释使用中文"，而
   `autotuner/research/` 全层是英文——14 个模块里只有 2 个的模块文档串含中文（14%），
   P4.4 任务 1-3 新增的三个模块（hyperparameter_space / hyperparameter_sampler /
   config_injection，约 2600 行中的 1841 行）**一行中文都没有**。
   `tools/check_comment_quality.py` 会对此报 ERROR/WARNING，但 CI 里该项是
   `continue-on-error: true`，所以不会被拦下。
   **处置**：目前选择在该层沿用英文（与本层既有文件一致，避免 P4.4 变成双语），
   而不是为此修改权威文档。要统一就得整体转换，属于独立任务，尚未排期。
2. **P4.3 引用的验证文档**：`docs/P4.3_gpu_deployment_summary.md` 与
   `docs/p4_3_validation_plan.md` 已存在并纳入版本控制（2026-09-14 前的提交），
   但 P4.2/P4.3 的两份总结文档直到 `1428549` 才入库——期间原子问题表引用过
   工作区里存在、仓库里不存在的文件。

---

## 更新日志

- 2026-09-11：初始化原子问题跟踪表
- 2026-09-13：完成代码库实际进度评估，更新 17 个已完成问题状态
- 2026-09-13：完成P6.1步骤1-3实现和文档（commit 7e106d6），更新进度统计和优先级列表
- 2026-09-13：完成P4.3单元测试补充（32个测试），创建验证工具，更新状态为"单元测试完成，待GPU验证"
- 2026-09-14：P4.4拆分为6个原子任务，完成第1个（搜索空间schema + 产品搜索空间数据 + 103个测试），其余5个未开始
- 2026-09-14：P4.4完成第2、3个原子任务（采样器、配置注入），贝叶斯采样器拆为任务7并阻塞于任务4，合计7个原子任务；研究层测试241个通过。审查发现与判定见 docs/P4.4_review_record.md
- 2026-09-14：新增"已知的文档与实现不一致（未修复）"一节，记录注释语言偏离与 P4.3 文档引用问题；归档 18 个工作流到 docs/archive/workflows/（commit c437c16）
- 2026-09-14：P4.4 完成第 4 个原子任务（搜索试验跟踪）：trial_ledger.py（JSONL 台账引擎，跨进程记录锁、撕裂尾行自愈、哈希链）+ hyperparameter_search.py（记录模型、写入门面、读侧视图、给任务 7 的观测协议）；研究层测试 338 个通过。该轮审查含 14 条变异测试（13 条被杀死、1 条存活并据此补了一条用例，两轮变异测试中作废的一轮也如实记录），见 docs/P4.4_review_record.md §四
- 2026-09-14：P4.4 任务 7（贝叶斯采样器）的阻塞解除：任务 4 已提供观测反馈接口 `TrialObservationSource`
- 2026-09-14：P4.4 任务 4 的文档收尾。设计稿 `P4.4_task4_design.md` 从「尚未实现」改为「已按本文实现并验收」，并逐条订正 7 处状态或计数（规模估计实测 763/1612 行、`mechanisms.json` 实测 13 个且 0 个入库、`content_hash` 实测 27 处、A12 判据由「grep 无输出」改为 AST 遍历并说明 grep 为何不可能满足、验收编号 A33 撞号改标 A42、附录的「不声明任何代码已存在」）。收尾后重跑：研究层 338 passed、`check_repository_structure.py` 末行 errors=0、§10.4 的三条一次性命令均符合预期。每条订正的可复核命令见 docs/P4.4_review_record.md §4.4.1
- 2026-09-15：P4.4 完成第 5 个原子任务（早停与剪枝策略）：`search_pruning.py`（1690 行）+ 123 个用例。**本条为事后补记**：任务 5 完成时只更新了本表的状态列与本节的说明段，漏了这一行，2026-09-15 补上。第二轮审查（审实现）查出 7 个缺陷并补 25 个用例（92→123），13 条变异全部杀死；见 docs/P4.4_review_record.md §六
- 2026-09-15：P4.4 完成第 6 个原子任务（搜索结果分析与最优配置导出）：`search_analysis.py`（1545 行）+ 73 个用例。第二轮审查（审实现）查出 8 个缺陷并补 8 个用例（63→71），7 条变异全部杀死；一条"重复 index 会让 `analyze()` 抛错"的审计指控经实测证伪，未改代码，记入设计稿 §12.4。见 docs/P4.4_review_record.md §七
- 2026-09-15：任务 6 当晚追补（§七 的 7.7）：审计另报两处"守卫存在但无任何用例钉住"，受控变异逐条确认后补用例（63→**73**）。`M5`（`_coverage_verdict` 里 `unstated` 早返回的顺序）与 `M3`（`export_best` 的 `injected_fingerprint` 相等守卫）在补之前**删掉也全绿**，补后各自有指名用例变红；`M3` 另实测其真实作用是"清理动作"（删掉后 `best.yaml` 会留在盘上而没有回执）。同时用同一脚本**证伪**了一条此前写进文档的推断：删掉 `_validate_no_trial_vanishes()` 的调用时**没有任何用例变红**（73 passed），早先据 `.scratch_wf6/m13.log` 得出的"该守卫能杀死变异"是错的，那次的红是并发会话清空台账文件造成的假象。变异脚本改为**在内存里重编译、不写盘**（`.scratch_wf6/mut_b.py`），避免重演 21:24 那次事故。另把 `search_analysis.py` 的换行符从 CRLF 统一为 LF（哈希随之重算）
- 2026-09-15：补记一条过程事故（不是代码缺陷）：`search_analysis.py` 被一个**还在后台跑的变异循环**反复改写（把变异写进模块、跑用例、再写回原文，`--basetemp=t6m9…t6m13` 轮转）。21:24 查明并终止（`Get-CimInstance Win32_Process` 看到该 bash 树在跑 pytest，对文件每 4 秒采样一次哈希可见它在干净版与变异版之间来回跳；杀掉进程树后连续 20 秒稳定）。**它咬了两次**：一次是模块上留着变异（靠设计稿 §12 那张 sha256 表对不上才发现——1541 行/`3e52aa9d…` 对 1544 行/`2f254174…`），一次是**第一次提交 `0cbd369` 装进去的就是它写下的 M9 变异**，已 `git reset --mixed HEAD~3` 撤销三个提交、从 `.scratch_wf6/mutate_backup.py` 还原并重提。留在这里的原因：`.scratch_wf6/` 是未跟踪的临时目录，删掉之后"当前文件是否就是被验过的那份"只有 §12 的哈希表能回答，所以那张表不是装饰；而且**跑会改工作区的脚本时不能让循环留在后台**，收尾前必须先确认没有写者
- 2026-09-15：P4.4 完成第 7 个原子任务（贝叶斯采样器）：`bayesian_sampler.py`（纯标准库：编码器 + Cholesky 高斯过程 + 期望改进 + 自适应采样器 + 决策记录与复核）+ 用例；研究层用例全通过。变异测试最初 30 条按预期落定（控制组存活、每个被改文件按字节还原并断言 sha256）；**第一轮的 28 条漏掉两处**（`_describe_draws` 的多约束格式与排序，那两处删掉不影响任何用例），补用例后杀掉。**本条的规模与条数在 2026-09-16 被改写**——原文写的是实现 1259 行、84 个用例、研究层 620 个用例、30 条变异，那是当晚更早的一个自洽快照（534+84+2=620），但那份快照没有留下可复原的副本；当前数见本文件正文的任务 7 段与下一行。落地过程中按本仓的 dead-branch 教条删掉两条守卫与一条兜底分支、删掉一处对离散编码恒等变换的钳位调用，并订正了 `_unit` docstring 里一处**给错理由**的说明（真正的成因是域检查的容差，不是端点浮点误差：实测 40730 个被接受位置里有 40 个位置落在 `[0,1]` 外，`[0, 1e-9]` 接受 `-1e-9`、位置算出来是 `-1.0`）。负 EI 的地板是实测的（用例自身网格 `-2.5131e-17`、更宽扫描 `-4.1715e-16`，两组都写明了扫描范围），**没有加钳位**，原用例"从不为负"的断言属夸大、已改；早先引用的 `-1.85e-16`/`-6.34e-16` 经多智能体审计指出无法复核后已换掉。12 条明确未验证事项写进设计稿 §11.4，模块与文档都**不做效能声明**。见 docs/P4.4_review_record.md §八
- 2026-09-16：任务 7 收尾的三件事。**（1）变异测试 30 → 33 条**：新加的三条各自盯住一处第一轮碰不到的行为（并列取首个候选、已花掉的试验不能再被提议、步看不到之后才跑的试验），三条一上去就被杀死，说明的是探测面原来没盖到，不是那三处没有守卫。**（2）给设计稿 §11.2 的"已核对"换了一条可跑的命令**：编号检查器（仓库外脚本，源码逐字放进设计稿 §13），两条语法——条目式只查 §9／§11.2／§12.3，内联式 `file:N-M` 全文查——本轮读数 `66 entries + 42 inline references checked, 0 failures`，另有 30 条匹配不上任何语法的按行号打印出来（"没被查"写成了读数，不是默认）。检查器自己也做了变异自检：13 条、控制组 1 条，12 条按预期落定、控制组干净，其中 1 条是**记录的预期存活**（把编号的反引号一起去掉就绕过了它）。**它一上线就查出一处真错**：设计稿 §1.3 把一条 AST 白名单用例的行号写成 1817-1870，与三个真实版本（当时、再往前、现在）**都对不上**，属写下时就错，已改正并记入设计稿 §12.3 第 9 条。**（3）订正随规模漂走的数**：实现 1259→1341 行、用例 2263→2795 行、84→94 条、研究层 620→630 条；设计稿 §12.1 与 §9 的实测列、§8 的对照句、本文件正文的任务 7 段一并改。**84→94 那 10 条只指名到 3 条**——旧的两个 sha256 在 git 里查不到、旧副本不存在，其余 7 条不逐条归因，设计稿里就是这么写的。另记一条过程事故：那 33 条变异是丢后台跑的，我在它跑到一半时量到的是当时在盘上的变异（63934 字节，正是 `pool_ignores_constraints`），结果本身没被污染（整轮仍报 `every file restored byte-exact`），但设计稿 §12.1 原来那句"没有任何后台写者"已删。
- 2026-09-16：新增 P4.6（训练进程在 Isaac Sim 容器里的 torchvision 同源问题）并修复。根因是实验做出来的：在 3060 机器的 `nvcr.io/nvidia/isaac-lab:2.1.0` 容器里只改导入顺序跑了四组对照，只有「只预导入 torch」那组复现出与失败运行逐字相同的报错，前三组都正常。修复是让 `blind_locomotion/__init__.py` 在 `import skrl` 之前先把 torchvision 钉到 torch 所在的那个目录（新模块 `_torchvision_pair.py`），并把新模块加进 `payload_manifest.py` 的 `STATIC_FILES`——不加的话修复到不了远端载荷。12 个用例 + 6 条受控变异（全部杀死、控制组干净、文件按字节还原）。**未验证**：只在这一台机器这一个镜像上验过，其它 Isaac Sim 版本与其它 torch/torchvision 组合没有验过。
- 2026-09-16：P4.2 端到端跑通，以及补上模块后仍在的三个静默缺陷。先把 `checkpoint_hook` /
  `checkpoint_integration` / `checkpoint_curator` / `research_ledger` 补进载荷清单（P4.2 原文），
  再在 3060 机器上跑真实训练，发现检查点管理**依然一条不登记、一行日志不打**。逐个挖出并修掉
  三个各自独立、都能单独让 P4.2 看起来"已完成"的缺陷：**（1）判据写错**——钩子用
  `if not hasattr(env, "_checkpoint_integration")`，而环境在 `__init__` 里就把该属性置成 `None`
  （`blind_tp_env.py:179`），`hasattr` 恒为真，集成器永远不建；证据是 `fixstep_n4`（576 步、
  `status: complete`）日志里 `[CheckpointHook]` 0 条。**（2）拦错了方法**——钩子包的是
  `agent.save`，而 skrl 1.4.3 的训练循环调的是
  `agent.write_checkpoint(timestep, timesteps)`（`skrl/agents/torch/base.py:677` 的
  `post_interaction`；`save(path)` 在同文件 :367 还在，但训练循环一次都不调它）；
  证据是 `hooksave_n4`（240 步、间隔 200）磁盘上写出了 46 MB 的 `agent_200.pt` 而
  `total_registered: 0`、清单 76 字节。改为包 `write_checkpoint`，用"调用前后比目录"取落盘
  文件名（不自己拼 `agent_<t>.pt`，该名字随 `checkpoint_store_separately` 变）。
  **（3）读了一组不存在的键**——`build_checkpoint_performance_snapshot` 读
  `reward_payload["mean"]` 与 `health_payload["episode_length_mean"]`，两个键在真遥测里都不存在
  （奖励总数叫 `"total"`，health 段没有 episode 长度），于是 `reward_mean` 恒 0.0、
  `episode_length_mean` 恒默认 100.0；而 curator 的评分一半权重压在 `reward_mean` 上、
  质量门也拿它比阈值，等于整条择优链路在盯常数。同一键名错误还在
  `CheckpointRegistry.backfill_from_telemetry` 里，外加把显示串 `"phi0"` 直接当数字。
  **为什么测试全绿**：第 3 条本来有测试，但那条测试把错误当成了契约（喂的是
  `{"mean": 1.5}` 这种现实中不存在的键），已改写为调用真的 `build_reward_payload` 再取值。
  **端到端证据**（run_id `hookreal_n4`，载荷 `taili_blind_runtime_fixstep11_1df8dd02e35a`）：
  `[CheckpointHook] installed` 与 `finalized ... promoted 1` 两行都在；
  `agent_200.pt`（46 MB）与 404 字节的 `checkpoint_manifest.json` 并存；
  清单里的 `reward_mean` 与**同一份遥测** step 200 的 `reward.total` 精确相等
  （都是 `-6.514138221740723`）；`checkpoint_mtime` 与 `os.path.getmtime()` 逐位相等。
  新增用例：钩子 9 条（其中 4 条专门钉 `write_checkpoint` 路径）、快照 3 条（另改写旧用例 1 条）、回填 2 条，
  均做过反证（去掉修复则变红）。**未验证**：清理阈值（90 GB）与保留策略只有单元测试覆盖，
  240 步只落一个检查点，`last_cleanup_step: -1`，没有一次真实训练跑到过阈值。
- 2026-09-16：**P4.3 端到端实机验证完成**，状态转入「已完成」。判据从奖励曲线换成权重本身：
  ① `resumeD/agent_2.pt` 对 `resumeA/agent_200.pt` 149 个张量逐位相等（最大绝对差 `0.000e+00`），
  对照 `resumeE`（不给 `--checkpoint`）则完全不相干（policy 余弦 0.0266）；
  ② 优化器 state 条目 resumeD 99 个全非零 vs resumeE 0 个，step 计数器 256→512；
  ③ 从 A@200 再跑 200 步，policy 复现误差 0.0011，而同长度正常位移 0.6665、无关参照 1.9693。
  **同时订正原验收标准**：「恢复后曲线连续，无性能跳变」在这套遥测上不可测量——
  一条没被打断的跑，相邻采样点相对差中位数就有 0.180、85% 超 10%；且 resumeB 与 resumeC
  的均值差 0.401 落在标准误 0.44 之内，这个设计区分不了恢复与全新开始。
  **遗留未验证**：scheduler / curriculum / rng 不在 `.pt` 内；
  `resume_edge.restored` 的 9 个 `true` 是键名匹配推出来的合理性检查而非测量；
  「复用同一 run id 会让身份门禁自证」只在源码上成立、未实测；
  `tools/verify_resume_continuity.py` 是空壳。
- 2026-09-16：P4.8 验收结果填写完毕，状态由带括号的自由文本归并为 `已修复`，
  使它能被进度统计的 `uniq -c` 数到（此前它在这条统计里是隐身的）。同时订正本节此前
  写下的"总计 31"，补入 P4.8 后为 32。（当时写的「P4.3 仍是『已完成（待远端端到端验证）』」
  在同日晚些时候被推翻——见本页更早那条 P4.3 端到端验证记录。）
  另订正 P4.7「最近完成」里的"远端 smoke12 / ladder_n4 实测 rc=0"——那两次运行实际是
  `status: failed`，属读错证据，已在原处标注。
- 2026-09-16：补测 3060 的显存阶梯，并把 P4.2 的端到端证据补到**默认**
  `TAILI_CHECKPOINT_INTERVAL` 这条路径上。四档短跑（4/64/256/512 envs，各 240 步）
  加一档长跑（512 envs，2000 步）：峰值 8806 / 9987 / 10759 / 11065 / 11085 MiB，
  五档全部 `status: complete`、`errors: []`、无 OOM。要点两条：峰值出现在 +97…+138 s
  而不是启动瞬间；512 档步数翻 8 倍峰值只涨 20 MiB，且峰值出现后 278 s 没被超过，
  所以是平台期而非持续增长。余量 1203 MiB 只在这套命令/载荷/地形上量过，
  换条件够不够**未验证**。见「关键发现」新增的「3060 上的显存阶梯」一节。
  **同时订正一个错误的进度判据**：早先 `scripts/run_taili.sh` 头部写过
  "`--total-steps 576` 跑出恰好 576 条 `[TPREW]`"，并据此把条数当步数判据。
  那是错的——`[TPREW]`/`[TPSTAT]` 的**条数被 `telemetry_interval` 节流**：
  间隔 1 时 576 步得 576 条，间隔 10 时 576 步只 59 条、2000 步 201 条、240 步 25 条
  （四个读数分别来自 `fixstep_n4` / `hookchk_n4` / `ladder_long_512` / `ladder_4_n4`）。
  可靠判据是 **tqdm 末行的 `N/N`**，13 次运行的收尾行与 `--total-steps` 全部相符。
  该脚本头部已改写并重新上传；本文正文未用过这个判据，无需改。
