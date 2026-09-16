# 文档入口

## 当前权威

| 问题 | 文档 |
|---|---|
| 全局目标、平地/楼梯/DR 验收语义 | [`taili_spec.md`](taili_spec.md) |
| 真实源码、配置、payload 和运行边界 | [`taili_runtime_source_map.md`](taili_runtime_source_map.md) |
| 仓库目录和依赖方向 | [`repository_architecture.md`](repository_architecture.md) |
| 修改、清理和验证标准 | [`maintenance_standard.md`](maintenance_standard.md) |
| 当前训练/部署操作 | [`taili_ops_runbook.md`](taili_ops_runbook.md)、[`locomotion_console_startup.md`](locomotion_console_startup.md) |
| Taili 传统控制工程地图与当前状态 | [`traditional_control/PROJECT_MAP.md`](traditional_control/PROJECT_MAP.md) |
| Taili 传统控制维护与提交规则 | [`traditional_control/maintenance.md`](traditional_control/maintenance.md) |
| 研究 Agent 当前运行闭环 | [`rl_agent_architecture.md`](rl_agent_architecture.md)、[`rl_agent_live_system_operations.md`](rl_agent_live_system_operations.md) |

## 专题资料

- P4.4 研究层审查判定与处置：[`P4.4_review_record.md`](P4.4_review_record.md)
- P4.4 任务 4 搜索试验跟踪设计稿（设计提案；**已按本文实现并验收**，见开头第 2 条）：[`P4.4_task4_design.md`](P4.4_task4_design.md)
- P4.4 任务 5 早停与剪枝设计稿（**已实现**，实现位置 `autotuner/research/search_pruning.py`，1690 行实现 + 2087 行用例、123 个用例；第 9 节区分实测与仅读码，并附 99 条行号引用的独立复核。**该稿 §4 曾把计划写成已落地，已就地标注并补落地映射**；审查与变异测试结果见 `P4.4_review_record.md` §六）：[`P4.4_task5_design.md`](P4.4_task5_design.md)
- P4.4 任务 6 结果分析与最优配置导出设计稿（**已实现**，实现位置 `autotuner/research/search_analysis.py`，1554 行实现 + 2036 行用例、75 个用例（任务 7 又动了这两个文件，此处按当前值更新，原为 1545/1967/73）；§12 是"计划 → 落地"对照，含 8 个实现缺陷的可复核复现方式、7 条变异记录与**明确未验证清单**。**该稿 §12 的 sha256 表当天抓到过一次真事故**：模块上留有一条没被还原的变异。§12.3 另记两条"删掉也全绿"的守卫、一条被受控变异**证伪**的守卫推断）：[`P4.4_task6_design.md`](P4.4_task6_design.md)
- P4.4 任务 7 贝叶斯采样器设计稿（**已实现**，实现位置 `autotuner/research/bayesian_sampler.py`，1341 行实现 + 2795 行用例、94 个用例；第 11.4 节是 12 条**明确未验证**事项，§12 是落地实测——33 条变异按预期落定（32 条真变异被杀 + 1 条控制组存活）而非"30 条全部杀死"、两次"删掉也全绿"的漏网、两条死守卫与一条兜底分支的删除、负 EI 地板的实测值，以及 §12.1 的 sha256 表（用途见 `P4.4_review_record.md` §八与 `atomic_problems.md` 更新日志）：[`P4.4_task7_design.md`](P4.4_task7_design.md)
- Taili 环境边界：[`taili_env_boundary.md`](taili_env_boundary.md)
- 奖励所有权：[`taili_reward_ownership.md`](taili_reward_ownership.md)
- 遥测契约：[`taili_telemetry_contract.md`](taili_telemetry_contract.md)
- MuJoCo 楼梯：[`taili_mujoco_stairs.md`](taili_mujoco_stairs.md)
- 适配器说明：[`../autotuner/adapter/README.md`](../autotuner/adapter/README.md)

## 历史与提案

设计提案、长篇交接、调参日志和旧基线只用于追溯，不作为当前配置或验收依据。
它们统一放在 [`archive/`](archive/README.md)；当前活状态仍以运行台账和明确的 live handoff 为准。

2026-09-11 ~ 09-14 的 18 个多智能体调查/审查工作流归档在
[`archive/workflows/`](archive/workflows/README.md)：原始返回文本在本机会话目录里会被清理，
该目录保存摘录与筛出的有效结论，**同时保存不可采信清单**——引用前先读它的 §4。
