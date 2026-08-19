# 当前策略说明入口

本文档只作为入口索引，不再直接承载完整策略说明。

原因：旧版 `CURRENT_FINAL_STRATEGY.md` 记录的是早期 final payload 和旧哈希。当前训练策略已经多次调整，如果继续把旧内容留在根目录，会让搜索和人工判断误以为旧 payload 仍是权威来源。

## 每次监控和调参前必须确认

- 上下文接近压缩时，必须先刷新 `docs/taili_live_handoff.md`，封存当时的 run、payload、证据、判断、未完成动作和干预条件；关键部署、诊断与决策后也要增量刷新，不能只依赖压缩前最后一刻。
- 压缩恢复后先读实时交接，再核对远程运行状态；不得把压缩摘要或历史记忆直接当成当前事实。
- 先阅读 `docs/taili_spec.md` 的“§0 — 当前优化目标”，它记录当前不允许漂移的理想状态和工作边界。
- 本阶段以平地质量、上/下楼梯能力和 DR 强鲁棒性为三条并行主线；DR 不再等楼梯达到极限后才开启，但也不能掩盖或交换掉已有运动质量。斜坡、粗糙和 boxes 仍不是当前主目标。
- 历史只提供全局要求和经验。当前问题与原因必须由正在运行的 payload、完整训练趋势和真实诊断共同确定。

当前权威说明请看：

- `docs/taili_spec.md` §0：当前平地与楼梯的理想目标、优化原则和验证方法。
- `docs/taili_current_training_baseline.md`：当前训练基线、参数快照、已知问题和验证命令。
- `docs/taili_runtime_source_map.md`：训练、诊断、奖励、payload、配置加载的真实源码边界。

当前单一可编辑策略源仍是：

`autotuner/blind_locomotion/taili_blind_config.yaml`

当前整理原则：

- 训练代码和策略参数不要和文档整理混在同一步改。
- 改配置前先确认字段是否真实进入运行对象。
- 改结构前先做旁路审计和等价验证。
- 新增或修改注释使用中文，文件保持 UTF-8 无 BOM。

每次影响训练链路后，至少运行：

```powershell
python -m autotuner.training_payloads.taili_blind_runtime.payload_manifest
python -m pytest tests/autotuner/blind_locomotion/test_curriculum_config.py tests/autotuner/taili_core/test_taili_reward.py
```
