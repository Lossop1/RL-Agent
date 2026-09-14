# 多智能体工作流调查归档（2026-09-11 ~ 2026-09-14）

## 这是什么

2026-09-11 到 09-14，本项目用 Workflow 多智能体编排跑了 18 个只读调查 / 实现 / 对抗审查工作流。
原始记录（每个 agent 的完整转录）只落在会话目录里，不受版本控制、随时可能被清理，总体积约 32 MB：

```
C:\Users\。。。。。\.claude\projects\D--taili-RL-locomotion-workspace\<session-id>\subagents\workflows\wf_*/
C:\Users\。。。。。\.claude\projects\D--taili-RL-locomotion-workspace\<session-id>\subagents\workflows\scripts\*.js
```

本目录保存其中**可追溯的部分**：

| 文件 | 内容 | 可信度 |
| --- | --- | --- |
| `journal_digest_20260911_20260914.txt` | 每个 workflow 每个 agent 的**最终返回文本**，保留原始措辞（单条按 400~800 字符截断，findings 压成 `[severity] file:line 标题`） | **原始材料，未核实** |
| 本文件 | 从上述材料筛出的**有效结论、未决事项、不可采信清单** | 筛选结果，**核实程度逐节标在 §2 开头** |

P4.4 三个审查工作流的判定与处置另有一份专门记录：`docs/P4.4_review_record.md`。

## 怎么用

先看本文件。原始文件只在两种情况下需要打开：想核对某条结论的原始措辞；或要判断某条被本文件标为"不可采信"的内容到底是怎么来的。

**不要把原始文件里的任何一条直接当成事实。** 该文件里同时保存了被引用者和被反驳者的话，两者措辞同样自信；`wf_c589c5a5-742` 那 14 条 finding 的判定绝大多数后来被证实与仓库实际不符或凭空编造（详见 §4）。

## 1. 工作流一览

| workflow | 日期 | 名称 | agents | 目的与持久产出 |
| --- | --- | --- | --- | --- |
| `wf_10cb34d8-2e8` | 09-11 | historical-readonly-audit | 18/19 | 按项目真实历史问题记录做只读复查；产出**仍存在的 2 个历史问题**与 4 类潜在复现模式（§2.1） |
| `wf_913c56b1-ba4` | 09-12 | engineering-standards-adversarial-audit | 15/16 | 对抗审查工程规范：违规、矛盾、可执行性、过度设计（§2.2） |
| `wf_67e2afa6-e1e` | 09-12 | architecture-optimization-execution-plan | 7/9 | 从最终目标反推分层，找分层违规（§2.2） |
| `wf_99661f75-301` | 09-12 | atomic-problems-audit | 3/6 | 审计 27 个原子问题实现状态（结论已被后续验证取代） |
| `wf_2aa1603f-a4d` | 09-12 | implement-research-scheduler | 7/8 | 并行实验调度器设计+实现（已落地，见 §3） |
| `wf_6639d153-e97` | 09-13 | analyze-simulator-abstraction | 3/3 | P6.1 仿真器后端抽象：IsaacLab 耦合点、验收拆解（§3） |
| `wf_ff98459c-35a` | 09-13 | verify-completed-problems | 35/35 | 验证 17 个已完成原子问题；产出 **6 个"已完成但不达标"**（§2.3） |
| `wf_c9c1c833-dcd` | 09-13 | implement-p6-3-observation-normalization | 5/5 | P6.3 观测归一化设计+实现（已落地，见 §3） |
| `wf_326a6c6a-dd5` | 09-13 | migrate-console-ssh-calls | 22/22 | 控制台 88 处直接 SSH 调用的侦察与迁移模板；产出**抽象层缺口清单**（§2.4） |
| `wf_760cfeb9-11e` | 09-13 | p4.2-checkpoint-management-design | 3/3 | P4.2 需求分析/架构/实现计划（已落地，见 §3） |
| `wf_a1cae386-5eb` | 09-13 | analyze-next-task | 8/8 | 可推进非 GPU 阻塞任务的扫描与建议（已过时） |
| `wf_6fba570d-e2e` | 09-14 | project-status-review | 5/5 | 项目状态综合审查（P4.2/P4.3 当时状态，含尚未完成的远程测试，见 §2.5） |
| `wf_a99b5c33-31a` | 09-14 | p44-understand-codebase | 5/5 | P4.4 实现所需的代码约定、配置管线、调度基础设施勘察（§2.6，**仍在用**） |
| `wf_91e7f0db-43b` | 09-14 | p44-hyperparameter-search-analysis | 1/2 | P4.4 需求分析；确认 skrl 超参当前硬编码位置（§2.6） |
| `wf_c589c5a5-742` | 09-14 | p44-task1-review | 24/24 | 对抗审查搜索空间层；**产出质量低**（14 条 found 的判定绝大多数为编造，§4.1） |
| `wf_c7ea802e-091` | 09-14 | review-hyperparameter-sampler | 12/12 | 对抗审查采样器+空间层；3 条 sustained（浮点端点/2^53 塌缩/指纹排序键）、1 条 refuted（见 `P4.4_review_record.md`） |
| `wf_d6494430-4c0` | 09-14 | review-hyperparameter-search | 12/12 | 同上再一轮；最终聚合 `4 raised, 1 upheld, 3 refuted`（§4.2） |
| `wf_b7b6bc70-6fe` | 09-14 | review-hyperparameter-space | **0/3** | **未产出任何结果**，`agents=0/3`，journal 无 findings 也无 verdict |

未跑满的：`wf_99661f75-301`(3/6)、`wf_67e2afa6-e1e`(7/9)、`wf_2aa1603f-a4d`(7/8)、
`wf_10cb34d8-2e8`(18/19)、`wf_b7b6bc70-6fe`(0/3)、`wf_91e7f0db-43b`(1/2)。这些的覆盖不完整，不能当成"查过了没问题"。

## 2. 仍然有效、且未见于其它文档的调查结论

**本节的核实状态不一致，逐条标了：**

- §2.1：**已在 2026-09-14 于当前代码上逐条重跑命令复核**，行号与内容均已确认。
- §2.2 / §2.3 / §2.4 / §2.5：**引自工作流产出，未逐条重跑**。写法上保留了它们给的行号，
  但行号会随改动漂移；引用前自己开一次文件。
- §2.6：部分重跑（`taili_blind_config.yaml` 的超参行号已确认），其余为引述。

### 2.1 仍存在的历史问题（`wf_10cb34d8-2e8`，09-11，2026-09-14 复核）

该轮逐条复查历史问题记录，11 条判 `still_exists=false`（已修复），**2 条判 `still_exists=true`**：

1. **D2/F2 聚合映射错误**（`products/taili/blind_locomotion/physeval_suite.py:44`）
   已复核：`D_TERRAIN_RUNS = ("slope", "rough", "boxes", "stairs")`，第 125 行是
   `spec_items["D"] = _aggregate([_item(terr, f"D[{terr}]") for terr in D_TERRAIN_RUNS])`。
   但仓库里 `stairs_down` 是独立的 terrain family（grep 有 30+ 处引用），因此 `D2` 被映射成
   `D[stairs]` 而漏掉下楼梯。调查者判定 `fix_visible: false`。
   （行号与内容 2026-09-14 复核一致；**"漏掉 stairs_down 是不是错"这一层未复核**——
   要看 `D2` 的验收定义到底是"上楼梯"还是"楼梯两侧"，那是 taili_spec 的问题。）

2. **标称姿态不一致**（`products/taili/blind_locomotion/gen_taili_gaits.py:107,142`）
   已复核：两处都是 `fk_foot(0.6, -1.2)`（第 142 行注释"用新 L1/FOOT_OFF 重算名义站姿"），
   而策略默认姿态是 `Q_DEFAULT_THIGH = 0.7`、`Q_DEFAULT_CALF = -1.4`
   （`taili_amp_reference.py:32-33`，`taili_nominal.yaml:23`、`traditional_control/profile.py:115` 另有两处一致）。
   调查者注：commit `a54b1c4` 声称修掉了这一类问题，但 `gen_taili_gaits.py` 未被覆盖；无测试覆盖。
   （行号与数值 2026-09-14 复核一致；**"这两组数应该相等"这一层未复核**——名义站姿与策略
   初始化姿态是否必须一致，取决于 gen_taili_gaits 的产物被谁消费，那要看调用链。）

同一轮还列出 4 类"潜在复现模式"及高风险位置（**只给了位置，未确认是否已存在具体缺陷**，引用时须自己复核）：

| 模式 | 高风险位置 |
| --- | --- |
| Reward 联合局部最优 | `core/taili_reward.py:367-396`(HIGH)、`873-880`(HIGH)、`1742-1750`、`283,1025-1032`；`blind_locomotion/taili_amp_env.py:3249-3290`；`blind_tp_env.py:2593-2641`；`taili_reward.py:1633-1700` |
| 课程语义松弛-升级与能力不匹配 | `taili_amp_env_cfg.py:151-152`(HIGH)：`terrain_curriculum_stair_height_min=0.18m`(代码默认) vs level-0 实际台阶 0.06m(down)/0.102m(up)——YAML 覆盖成 0.035m，但代码默认值会挡住所有 level-0 升级 |
| 指标定义-测量不一致 | `acceptance_score.py:36-40`(HIGH)、`physeval_blind.py:455-458`(HIGH)；`taili_spec.md:94,138-139` 与 `physeval_blind_e.py:64-68` 已标注为历史 bug |
| 配置-实现静默覆盖/死代码 | `taili_blind_config.yaml:259`(HIGH)、`taili_amp_env_cfg.py:242`(HIGH)、`taili_amp_env.py:1505-1527`/`1719-1740`(HIGH)、**`blind_tp_env.py:1501`(CRITICAL)**、`blind_tp_env.py:261-264`(HIGH) |
| 训练-部署观测契约不一致 | `taili_amp_env.py:1804-1823`(HIGH)、`blind_tp_env.py:1344-1352`(HIGH)、`taili_amp_env.py:4409`、`core/taili_reward.py:1865-1880`(HIGH)、`blind_tp_env.py:1055-1069`(HIGH) |

该轮有 2 条 agent 返回空 verdict（`refuted=False` 但无内容），即那两条实际未被反驳也没被支持。

### 2.2 工程规范与分层违规（`wf_913c56b1-ba4`、`wf_67e2afa6-e1e`，09-12）

**已确认的分层违规（有具体行号，未见后续修复记录）：**

- `autotuner/execution/deployment.py:44`：`root: str = "/root/gpufree-data/rl-agent"` —— 执行层硬编码产品特定远程路径。
- `autotuner/execution/runtime.py:88,206`：把 `backend` 默认值硬编码为 `"isaaclab"`，执行层因此感知具体训练框架。
- `autotuner/locomotion_console/config_manager.py:238-244`、`datasource.py:528-532` 等：控制台直接 `from autotuner.infrastructure.remote import RemoteSSH` 并实例化，绕过任何适配层。
- `autotuner/training/__init__.py:7-9`：用显式 `from ... import ...` + `__all__` 暴露符号，而不是仓库其它兼容层统一的 `__getattr__` 转发。

**一条 severity=critical 的运行时缺陷：**

- `autotuner/product/task_pipeline.py:224`：`register_asset` 只传 `lineage=AssetLineage(source_product_ref=product_ref)`，缺 `source_task_contract_ref` 和 `source_run_ref`；而 `asset_catalog.py:300-306` 对 checkpoint / policy_checkpoint / training_baseline / optimizer_state / normalizer_state 这几类资产**强制要求两者非空**，违反时抛 `AssetCatalogError`。也就是说该类资产的登记路径会在运行时被自己的校验拒绝。有测试（`test_asset_catalog.py:28-37`、`test_task_artifacts.py:308-312`）明确锁定这条约束，所以不是校验写错了。

**同源逻辑多处独立实现、阈值互不一致（活动命令判定）：**

| 位置 | 判定式 |
| --- | --- |
| `blind_tp_env.py:1296` | `(spd_xy > 0.1) \| (commands[:, 2].abs() > 0.05)` |
| `tools/taili_mujoco_long_horizon.py:279-280` | `norm(command[:2]) > 0.1 or abs(command[2]) > 0.05` |
| `autotuner/control/gait.py:152,212,393` | `norm(command[:2]) > 0.02 or abs(wz) > 0.02` |

gait clock 的冻结/归零逻辑同样重复实现（`taili_amp_env.py:1821-1823`、`4409`、`taili_mujoco_long_horizon.py:269-284`）。
这条与 09-11 那轮记录的"部署侧 C++ 曾与训练侧判定不一致"是同一个根因的延续。

**规范层面被判"不可静态强制"的规则**（`wf_913c56b1-ba4` 的可执行性 lens）：
"新增注释使用中文并说明原因/边界/不变量"、"复杂数学保留单位/参考系/窗口/门控语义"——两条都需要语义理解，现有 `tools/check_*.py` 无法验证，属纯人工自律。
`check_repository_structure.py` 也**无法**检测：是否有人在 `output/`、`strategy_backups/`、`products/taili/payload/archive/` 的历史副本上改业务逻辑；`autotuner/` 下是否混入产品特定配置字段。

**规范自身的矛盾（未决，需要拍板）：**

1. "每个职责只有一个权威实现" vs "产品差异通过合同字段和插件角色表达"：产品插件提供自定义环境/奖励时，谁算权威实现？加第二个产品是否就违反唯一性？
2. "payload 只能由清单从权威源码生成，远程副本不是源码" vs 实际运维：训练 12 小时后远程 payload 触发一行可修的 bug，按规则必须本地改→重生成→重部署→重启（checkpoint 可能不兼容）。规范没有定义热修复通道。

**被判"过度设计"的机制：** Product Contract + Plugin Roles —— 460 行 YAML 配置、13 个抽象层文件、62 处插件调用点、8 个兼容转发模块，而 `config/products/` 下只有 `taili.yaml` 一个产品，`ProductRegistry.get()` 的多产品分支从未执行过。判 `premature`（非"错误"，是"先于需求"）。

**`output/` 清理：** 当时 75 个子目录、约 3.4 GB、1206 个数据文件、68 个 diagnostics 子目录（15 个来自 2026-07-31）；`.gitignore` 第 72 行忽略 `/output/`，但没有任何清理机制。

### 2.3 已完成但不达标的原子问题（`wf_ff98459c-35a`，09-13，35 agents）

该轮逐个复核 17 个已标"已完成"的问题，**6 个 `meetsAcceptanceCriteria: false`**：

| 问题 | 判否理由（severity=critical 的部分） | 当前状态 |
| --- | --- | --- |
| **P6.3** 观测归一化 | `taili_obs.py` 只组装原始观测，没有任何归一化/clip，代码库中不存在 `RunningMeanStd`；测试只查 shape 不查分布；验收要求"95% 观测值落在 [-3,3]"无人验证 | **已修复且已核实**：`products/taili/core/taili_normalization.py` 存在，`class RunningMeanStd`(:35)，提交 `162d958 feat(P6.3)` |
| **P3.5** 运行隔离与清理 | 失败/回滚后工作空间目录不删（`research_supervisor.py:324` 建、无人删）；`ResearchScheduler._cleanup_job()` 只释放 GPU（`:768-770`），不管目录 → 磁盘泄漏；4 个相关测试文件无一验证清理行为 | 未见修复记录 |
| **P3.2** 远程命令执行抽象 | 控制台层仍有 88 处直接 `remote.exec/exec_out/put`，9 个文件直连 `RemoteSSH`，验收"所有控制台调用走抽象接口"未达成；Protocol 只定义 `exec`+`put` | 部分推进：`wf_326a6c6a-dd5` 出过迁移模板（§2.4） |
| **P1.4** 诊断报告查看 | 缺"报告加载 < 5s"的性能测试与基准；`report_for_job` 用 `except Exception` 吞掉超时/网络/损坏数据之别 | 未见修复记录 |
| **P2.2** 研究假设管理 | `ResearchLedgerStore.records()` 只能按 `record_type` 过滤，不能按 `hypothesis_id` / `symptom_ref` 查；缺"问题→symptom→HypothesisSet→hypothesis→ExperimentPlan"整链追溯测试 | 未见修复记录 |
| **P5.1** 观测契约定义 | 观测项没有物理单位标注、没有 min/max 范围；`PolicyDeploymentContract.observation/action` 是裸 `dict[str, Any]` | 未见修复记录 |

判 `true` 但带 major 项的（引用时注意）：
- P2.3：`continue` 裁决分支无测试覆盖；`assess()` 规则硬编码在 60+ 行静态方法里，可配置性有限。
- P6.5 / P2.5：`validate_robot_urdf()` 没有专用单元测试，错误场景（坏 URDF、关节数不符、零质量、惯性缺失）全无覆盖；`except Exception` 过宽。
- P1.3：提案-动作绑定用 `dict ==` 精确相等（`app.py:2163`），`args` 多一个 `force: True` 就能绕过提案；提案无 TTL；`LLMSessionStore` 文件读写无原子性。
- P4.5：`telemetry.py` 1969 行，测试覆盖 43%；注释语言中英混用。
- P4.1：`test_terrain_curriculum.py` 参数名不匹配（`height_delta` vs `support_height_delta`，`dist` 不存在，缺 `expected_height_direction`/`support_height_valid`），**4 个测试全失败**。

上表"当前状态"列是 09-14 归档时的提交历史印象（**未逐条重跑**）："未见修复记录"指当时没有一条提交信息声称修过它，
不代表一定还坏着。P6.3 那一行是已核实的反例——它被判 critical 未实现，但当天晚些时候就实现并提交了。
拿这些去开任务之前，先花两分钟复现一次。

### 2.4 控制台 SSH 迁移的抽象层缺口（`wf_326a6c6a-dd5`，09-13，22 agents）

侦察结论：**88 处**直接 SSH 调用集中在 `datasource.py`、`diagnostics.py`、`deploy.py` 等文件；
`llm_workflows.py`、`agent_eval/*`、`knowledge_model/*`、`diagnostic_history.py`、`tuning_ledger.py`、`context_manager.py`、`__init__.py` 逐文件核实为 **0 处直接 SSH**（它们的 "gap" 是别的东西，不是 SSH）。`paramiko` 只剩 2 处 docstring 提及，`.sudo(` 0 处。

文件传输的两种模式与位置：
1. 直接 SFTP：`datasource.py:1349`、`diagnostics.py:1566/1578/1588`、`deploy.py:190`，实现是 `infrastructure/remote.py:167-188,190-207` 的 paramiko SFTP + 指数退避重连（`min(2**attempt, 30)`s）。
2. `RemoteTransport` Protocol。

**关键结论：`RemoteTransport` 协议本身不足以承载迁移。** 现有协议只有
`exec(cmd, timeout=30) -> (stdout, exit_code)` 和 `put(local, remote) -> None`；
缺失的能力（迁移会被迫继续直连 SSH 的原因）：
`get`（下载）、`sudo`、`exists`、`chmod`、`chown`、`mkdir`、`rm`、`symlink`、`read/cat`。
另外只读查询（machine status / GPU info / tmux sessions）未纳入协议。

`ssh_pool.get_pooled_ssh()` 与 `fabric.Connection` 的匹配数均为 0——历史迁移已把它们清干净了，
所以 P3.2 剩下的活是"把 88 处 `RemoteSSH` 直连改成走协议"，不是"清掉两种旧范式"。

### 2.5 P4.2 / P4.3 当时状态（`wf_6fba570d-e2e`，09-14 13:06）

- P4.2 已完成并提交：`checkpoint_curator.py`（CheckpointRegistry / CheckpointSelector /
  CheckpointCurator / CapabilityPromotionService）+ `checkpoint_integration.py`；5 个测试文件 50 个测试全过
  （`50 passed in 0.29s`）。评分权重：reward 50% / stability 30% / episode 10% / recency 10%。
- P4.3：10 个文件 SFTP 上传成功，远程 GPU 确认为 RTX 4090 24GB；但**远程服务器当时不可达**
  （端口 31376 连接超时），50 个测试未在远程执行；远程 Python 是 externally-managed，装依赖需 `--break-system-packages`。
  该轮列的下一步是"服务器恢复后执行 `python scripts/remote_install_and_test.py`"。
- 该轮记的未提交文件是 `docs/P4.2_checkpoint_management_summary.md`、`docs/P4.3_remote_testing_instructions.md`、
  `scripts/remote_install_and_test.py` —— 到 09-14 17:00 仍未被提交。

### 2.6 P4.4 实现所依赖的仓库事实（`wf_a99b5c33-31a`、`wf_91e7f0db-43b`，09-14）

这几条在写 P4.4 代码时一直在用，但不在任何权威文档里：

- **测试风格双轨制**：`tests/autotuner/research/` 是"极简纯函数式"——无 fixture、无 parametrize、无 class，
  全部模块级 `test_` 函数 + 绝对导入 `autotuner.*`，文件系统用 pytest 内置 `tmp_path`（标注 `: Path`）。
  其它目录（如 `product/`）才用 fixture/parametrize/class。`tests/` 下**只有仓库根一个 `conftest.py`**，
  pytest 配置在根 `pyproject.toml`（`testpaths=["tests"]`、`pythonpath=["."]`、`--strict-markers --basetemp=.pytest-tmp`）。
- **类型风格双轨制**：需要 schema 校验/版本化/JSON 往返的领域记录用 pydantic `BaseModel`；
  轻量不可变值对象与内部 DTO 用 `@dataclass(frozen=True)`；有状态服务用普通 class，接口用 `Protocol`。
  校验失败统一 `raise ValueError`（validator 内也是），只有编排/状态层定义自定义异常。
- **`mechanism_specs.py` 的权威实现在 `autotuner/mechanisms/`**，`autotuner/locomotion_console/mechanism_specs.py` 只是 9 行 star-import 兼容壳。
- **超参配置管线是"纯 dict + deepcopy"**，没有 OmegaConf、没有通用 override：
  `load_taili_blind_config()` → `effective_config_with_overrides()`（只定向改 timesteps/write_interval/checkpoint_interval）
  → `build_skrl_config()`（整段 deepcopy，只补 `models.policy.actor_hidden/dropout`）
  → 写 `run_dir/agent.skrl.yaml` → `train_taili._load_experiment_cfg()` 读回 → 当 plain dict 传给 skrl Runner。
  仓库里确有递归 `_deep_merge`（`task_materializer.py:25`），但它由训练合同的 `config_overlay` 驱动，
  且 allowlist 只有 `training_recipe/model/env/reward/domain_randomization`——**不含 `skrl`**，所以覆盖不了 `learning_rate`。
- **`early_stop.py` 是死代码**：权威实现在 `products/taili/ops/early_stop.py`，`autotuner/training/early_stop.py` 是 re-export 垫片；
  两者**无任何调用方、无任何测试**（唯一提及来自 `config/repository_structure.toml:109` 的结构清单）。
  其逻辑是"单次训练运行内的启发式中止"（NaN / 地形课程卡住 / 回合过短），基于 IsaacLab 快照 dict 的绝对阈值，
  没有跨 trial 注册表、没有中位数/分位数计算 —— **无法复用实现 MedianPruner，必须从零写**（可借鉴其"纯函数 + 返回 reason 字符串"的风格：
  签名 `f(snapshots: List[Tuple[int, dict]]) -> Optional[str]`，`should_stop()` 返回第一条命中的 reason）。
- **`research_scheduler.py` 可作并行后端，但不是搜索组件**：队列/优先级/GPU 分配/非阻塞启动+轮询/取消/批量提交都已具备；
  三个缺口：(1) 结果只存在内存 `job.execution_result`（`:746`），进程退出即丢，无 ledger/状态文件；
  (2) 一个候选必须是完整 `ExperimentPlan` + 目录形式的 candidate artifact，且该目录里必须有 `mechanisms.json`（`:413-415`，否则 `FileNotFoundError`）；
  (3) 并发不是线程池/asyncio，而是 1 个守护线程轮询（`:610,625`），上限靠 `max_parallel_experiments`（默认 2）。
  `experiment_tracker.py` 是独立、未被 scheduler 引用的追踪器，无锁、状态枚举少 2 个、无 metrics 字段。
- **P4.4 待参数化的 skrl 超参当前硬编码位置**（`taili_blind_config.yaml`，行号 09-14 复核）：
  `discount_factor`:872 = 0.99、`learning_rate`:874 = 0.0001、`entropy_loss_scale`:892 = 0.02。

## 3. 已被后续实现取代的调查内容

这些工作流的产出已经落到代码或已提交，只有考古价值：

- `wf_760cfeb9-11e`（P4.2 设计）→ 已由 `feat(P4.2)` 实现并提交。
- `wf_c9c1c833-dcd`（P6.3 归一化设计）→ 已由 `taili_normalization.py` 实现并提交（Welford 在线算法、
  `register_buffer` 保存 mean/var/count 以随 checkpoint 走、freeze/unfreeze）。注意该轮有一个产物写错了位置：
  该轮还把一份实现副本写进了 `tests/products/taili/core/taili_normalization.py`（文件名不带 `test_` 且是 `.py` 而非 `test_*.py`，pytest 不会收集它）——09-14 复核时这个多余副本仍在，与真正的 `tests/products/taili/core/test_taili_normalization.py` 并列。
- `wf_2aa1603f-a4d`（调度器实现）→ `GPUResourcePool` / `ExperimentQueue` / `ResearchScheduler` /
  `ResearchSupervisor.execute_async()` 的设计与实现已落地（注意其中若干实现是在 `.claude/worktrees/` 的隔离副本里做的）。
- `wf_99661f75-301`（27 个原子问题审计）→ 结论已被 `wf_ff98459c-35a` 的逐条验证取代，且只跑了 3/6 个 agent。
- `wf_a1cae386-5eb`（下一步任务建议）→ 建议的 P4.2/P4.3 都已完成，P4.4 判断已过时。
- `wf_6639d153-e97`（仿真器抽象）→ 只产出耦合点清单与验收拆解，P6.1 至今未开工。

## 4. 不可采信的内容（防止将来被引用）

### 4.1 `wf_c589c5a5-742`（p44-task1-review）的 finder 产出

该轮 24 个 agent 结果里，两个 finder lens 各提 7 条、共 14 条 finding，
**反驳环节的绝大多数判为"与仓库实际实现不符 / 凭空编造"**。已核实的编造方式：

- 引用**不存在的测试名**：`test_active_assignment_reports_only_the_applicable_parameters`、
  `test_a_fingerprint_is_stable_across_equivalent_constructions` —— 全仓库 grep 无命中，`git log -S` 亦无历史记录。
- 引用**过时的正则**：声称 `_IDENTIFIER = ^[A-Za-z][A-Za-z0-9_.-]{0,127}$`，实际是
  `^[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)*$`（因此"点号允许空段"的整条发现不成立）。
- 声称**代码里没有的能力**：`fingerprint()` 不含 `schema_version`（实际第 516-521 行同时哈希
  `schema_version` 与 `specs`，且有专门的测试）、`set_parameter` 默认静默创建中间路径
  （实际签名 `create_missing: bool = False`，默认 `raise ValueError`）、`_validate_continuous/_validate_discrete`
  不检查有限性（实际有专门的 `_validate_bounds()` 逐字段 `math.isfinite`）、continuous 域没有分辨率字段
  （实际有 `grid_points`）。
- 把 `default_assignment()`（嵌套）与 `default_values()`（扁平）**混为一谈**，据此断言形状不兼容。

**教训（写进流程）**：审查结果必须经反驳环节过滤才能采信；单轮 finder 的原始产出不能直接当结论使用。

### 4.2 `wf_d6494430-4c0`（review-hyperparameter-search）的最终判定

该轮的最终聚合是 `findings: 4 raised, 1 upheld, 3 refuted`：

| 发现 | 判定 |
| --- | --- |
| `hyperparameter_sampler.py:383` `_describe_dead_end` 把各约束命中数求和当成"违反约束的抽取次数"（低） | **upheld（唯一维持的一条）** |
| `hyperparameter_sampler.py:409` log 连续域随机抽取走 `math.log`/`math.exp`，跨机器可能差 1 ulp，破坏可重放（高） | **refuted**：反驳者实测本机 `math.log(1e-5)`、`math.log(1e-3)` 及全部 10 个实际抽取值的 `math.exp` **均与 80 位十进制参考逐位一致**（正确舍入），因此"跨机器差 1 ulp"是无法演示的可移植性假设 |
| `hyperparameter_space.py:538` log 连续网格用 `**`（pow）构造，同源（中） | **refuted**：实测 `100.0**(1/3)` = 4.641588833612778 是本次运算的正确舍入双精度值；同上 |
| `hyperparameter_space.py:638` 守卫取值在约束下不可达、条件参数永久失效（中） | **refuted**（1 反驳 / 1 支持，多数反驳）：`_validate_condition_reaches_its_guard` 的契约只检查 `guard.contains(value)`（域成员性），加载期可满足性分析不是本模块的契约 |

另一轮 `wf_c7ea802e-091` 有 3 个 sustained 项，均已修复：`hyperparameter_space.py` 的
`discrete_value` 内部点浮点误差、`discrete_value` 在 2^53 以上塌缩出重复点、
`fingerprint()` 约束排序键漏 `of`（导致指纹依赖声明顺序）——详见 `docs/P4.4_review_record.md` 的 S-2/S-3/S-4。

### 4.3 `wf_b7b6bc70-6fe`

`agents=0/3`，journal 里既无 findings 也无 verdict。**它什么也没查到，不是"查过没问题"。**

## 5. 未决事项清单（需要决定，不是需要实现）

1. `.claude/worktrees/` 下遗留的多个工作树（均停在提交 `8dc3ed3`，是 master 的祖先、无独有提交）是否删除。
2. 18 个原始工作流转录目录（约 32 MB，仓库外）是否随会话清理。
3. §2.2 里两条**规范自身的矛盾**（权威实现唯一性 vs 插件；payload 无热修复通道）需要拍板，否则规范反复被引用却无法执行。
4. §2.1 的两个"仍存在"历史问题、§2.3 的 5 个"已完成但不达标"、§2.2 的 `task_pipeline.py:224` critical 缺陷：
   目前都只存在于本文件，尚未进入 `docs/atomic_problems.md` 或任何待办表。
